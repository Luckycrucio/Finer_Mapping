"""RGB-first, intensity-fallback colour lookup for deskewed LiDAR points.

`cache_frames` walks the bag once and saves every /rgb/image_raw frame to
disk as a JPEG plus a timestamp index, so the fusion pass never has to hold
more than a handful of full-resolution frames in memory at once.

`FrameColorizer` then, per scan: finds the camera frame closest in time
(falls back to intensity if nothing is within `max_dt`), projects the scan's
world-frame points into it with the camera's rational-polynomial distortion
model, keeps only the nearest-camera-depth point per output pixel (a
per-scan z-buffer -- since the LiDAR and camera are rigidly mounted together,
a single scan's own points already share close to the matched image's
viewpoint, so this catches ordinary self-occlusion without needing a
whole-map renderer), and samples BGR at that pixel. Points that have no
close-enough frame, land outside the image, are behind the camera, or lose
the z-buffer test fall back to a percentile-normalised grayscale of LiDAR
intensity (the same convention as the existing coverage1_edited pipeline's
color_intensity.py).
"""
import functools
from pathlib import Path

import cv2
import numpy as np


def cache_frames(reader_factory, image_topic, cache_dir, log=print):
    """reader_factory() must return a fresh, already-open rosbag2_py
    SequentialReader plus its {topic: type} map, as (reader, type_map)."""
    import cv_bridge
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    cache_dir = Path(cache_dir)
    (cache_dir / "frames").mkdir(parents=True, exist_ok=True)
    stamp_path = cache_dir / "timestamps.npy"

    reader, type_map = reader_factory()
    if image_topic not in type_map:
        raise KeyError(f"{image_topic} not found in bag; available: {sorted(type_map)}")
    msg_type = get_message(type_map[image_topic])
    bridge = cv_bridge.CvBridge()

    timestamps = []
    idx = 0
    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic != image_topic:
            continue
        msg = deserialize_message(data, msg_type)
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(cache_dir / "frames" / f"{idx:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        timestamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        idx += 1
        if idx % 500 == 0:
            log(f"  cached {idx} camera frames...")

    np.save(stamp_path, np.asarray(timestamps, dtype=np.float64))
    log(f"cached {idx} camera frames from {image_topic} -> {cache_dir}")
    return idx


class FrameColorizer:
    def __init__(self, cache_dir, sensor_config, max_dt=0.09,
                 intensity_lo_pct=1.0, intensity_hi_pct=99.0, lru_frames=8):
        cache_dir = Path(cache_dir)
        self.timestamps = np.load(cache_dir / "timestamps.npy")
        self.frames_dir = cache_dir / "frames"
        self.K = sensor_config.K
        self.D = sensor_config.D
        self.width = sensor_config.image_width
        self.height = sensor_config.image_height
        self.max_dt = max_dt
        self._intensity_lo_pct = intensity_lo_pct
        self._intensity_hi_pct = intensity_hi_pct
        self._intensity_lo = None
        self._intensity_hi = None
        self._load_frame = functools.lru_cache(maxsize=lru_frames)(self._load_frame_uncached)

    def _load_frame_uncached(self, index):
        return cv2.imread(str(self.frames_dir / f"{index:05d}.jpg"))

    def set_intensity_range(self, lo, hi):
        self._intensity_lo, self._intensity_hi = float(lo), float(hi)

    def nearest_frame_index(self, times):
        """times: (N,) -> (N,) int frame index, or -1 if nothing within max_dt."""
        idx = np.searchsorted(self.timestamps, times)
        idx = np.clip(idx, 1, len(self.timestamps) - 1)
        left, right = idx - 1, idx
        use_right = np.abs(self.timestamps[right] - times) < np.abs(self.timestamps[left] - times)
        chosen = np.where(use_right, right, left)
        dt = np.abs(self.timestamps[chosen] - times)
        return np.where(dt <= self.max_dt, chosen, -1)

    def intensity_to_gray(self, intensity):
        lo, hi = self._intensity_lo, self._intensity_hi
        if lo is None:
            lo, hi = np.percentile(intensity, [self._intensity_lo_pct, self._intensity_hi_pct])
        gray = np.clip((intensity - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0
        return np.repeat(gray[:, None], 3, axis=1)  # BGR order, grayscale

    def colorize_frame_group(self, points_world, world_from_camera, frame_index, intensity):
        """All points here share one matched camera frame. Returns
        (colors_bgr float (N,3) in [0,255], used_rgb bool (N,))."""
        n = len(points_world)
        colors = self.intensity_to_gray(intensity)
        used_rgb = np.zeros(n, dtype=bool)
        if frame_index < 0:
            return colors, used_rgb

        image = self._load_frame(int(frame_index))
        if image is None:
            return colors, used_rgb

        camera_from_world = np.linalg.inv(world_from_camera)
        pts_h = np.hstack([points_world, np.ones((n, 1))])
        pts_cam = (camera_from_world @ pts_h.T).T[:, :3]

        in_front = pts_cam[:, 2] > 0.05
        if not np.any(in_front):
            return colors, used_rgb

        proj, _ = cv2.projectPoints(
            pts_cam[in_front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D
        )
        proj = proj.reshape(-1, 2)
        px = np.round(proj).astype(np.int64)

        in_image = (
            (px[:, 0] >= 0) & (px[:, 0] < self.width) & (px[:, 1] >= 0) & (px[:, 1] < self.height)
        )

        front_idx = np.nonzero(in_front)[0]
        candidate_idx = front_idx[in_image]
        candidate_px = px[in_image]
        candidate_depth = pts_cam[candidate_idx, 2]

        # per-pixel z-buffer: keep only the nearest-depth candidate per pixel
        pixel_key = candidate_px[:, 1].astype(np.int64) * self.width + candidate_px[:, 0]
        order = np.argsort(candidate_depth)
        pixel_key, candidate_idx, candidate_px = pixel_key[order], candidate_idx[order], candidate_px[order]
        _, first_of_pixel = np.unique(pixel_key, return_index=True)
        winners = candidate_idx[first_of_pixel]
        winner_px = candidate_px[first_of_pixel]

        image_h, image_w = image.shape[:2]
        winner_px[:, 0] = np.clip(winner_px[:, 0], 0, image_w - 1)
        winner_px[:, 1] = np.clip(winner_px[:, 1], 0, image_h - 1)
        colors[winners] = image[winner_px[:, 1], winner_px[:, 0]].astype(np.float32)
        used_rgb[winners] = True
        return colors, used_rgb
