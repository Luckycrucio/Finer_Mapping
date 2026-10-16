#!/usr/bin/env python3
"""Measure the camera's time offset against the LiDAR/trajectory clock.

Usage: measure_camera_time_offset.py <bag_name>  (e.g. coverage1)

build_finer_map.py looks up each camera frame's pose in GLIM's trajectory at
the frame's capture time, header stamp + --image-time-offset. If the camera
driver stamps frames late (or early), every frame is projected from where
the camera was a moment later (earlier), and colours slide off the geometry
whenever the robot turns. This script finds the offset that makes the
camera and the LiDAR agree:

  1. Pick `--frames` camera frames taken while the robot turns fastest (at
     least `--min-gap` seconds apart): turning is where a timing error moves
     the projection most.
  2. For each, take the LiDAR depth edges (the near side of within-ring range
     jumps over 0.5 m: object silhouettes) measured within +/-60 ms of the
     frame, deskewed into the world with the same trajectory, LiDAR time
     offset and preprocessing as the build.
  3. For every candidate offset d, project those edges into the frame from
     the camera pose at (stamp + d), and average the image gradient
     magnitude (Sobel, blurred, normalised per frame) at the projected
     pixels. Silhouettes land on image edges when d is right.
  4. The offset maximising the mean score over all frames (refined between
     grid steps with a parabola) is the --image-time-offset to use.

Needs the build's camera frame cache (cache/<name>/, pass 1 of
build_finer_map.py) and the same bag/GLIM map arguments as the build; the
LiDAR time offset defaults to the GLIM map's points_time_offset, as there.

On coverage1 (GLIM map coverage1_off009, LiDAR offset +0.09 s) the score
peaks at -0.06 s: frames are stamped ~60 ms after capture. The same test on
the earlier map without the LiDAR correction peaks at -0.15 s, i.e. the same
-0.06 s plus that map's 0.09 s LiDAR error.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_finer_map import (BAG_SUFFIX, DATASETS_ROOT, MAPS_ROOT, POINTS_TOPIC,  # noqa: E402
                             glim_points_time_offset, prepare_scan)
from src.bagio import iter_topic  # noqa: E402
from src.extrinsics import SensorConfig  # noqa: E402
from src.ouster_scan import OrganizedScan  # noqa: E402
from src.trajectory import Trajectory  # noqa: E402

EDGE_JUMP_M = 0.5      # range jump to the farther neighbour that makes a point a silhouette edge
EDGE_WINDOW_S = 0.06   # LiDAR edges used per frame: measured within this of the frame's stamp


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="bag name, as for build_finer_map.py (e.g. coverage1)")
    p.add_argument("--datasets-root", default=DATASETS_ROOT)
    p.add_argument("--maps-root", default=MAPS_ROOT)
    p.add_argument("--bag-suffix", default=BAG_SUFFIX)
    p.add_argument("--bag", default=None, help="default: <datasets-root>/<name><bag-suffix>")
    p.add_argument("--map-dir", default=None, help="GLIM map with traj_lidar.txt and config/ (default: <maps-root>/<name>)")
    p.add_argument("--cache-dir", default=None, help="camera frame cache from the build (default: cache/<name>)")
    p.add_argument("--points-topic", default=POINTS_TOPIC)
    p.add_argument("--points-time-offset", type=float, default=None,
                   help="LiDAR time offset (default: the GLIM map's points_time_offset, as in the build)")
    p.add_argument("--frames", type=int, default=16, help="number of fast-turning frames to use")
    p.add_argument("--min-gap", type=float, default=2.0, help="minimum seconds between chosen frames")
    p.add_argument("--search", type=float, nargs=2, default=(-0.25, 0.10), metavar=("MIN", "MAX"),
                   help="candidate offsets to try, seconds")
    p.add_argument("--step", type=float, default=0.01, help="candidate offset step, seconds")
    args = p.parse_args()
    args.bag = args.bag or str(Path(args.datasets_root) / f"{args.name}{args.bag_suffix}")
    args.map_dir = Path(args.map_dir or Path(args.maps_root) / args.name)
    args.cache_dir = Path(args.cache_dir or here / "cache" / args.name)
    if not (args.cache_dir / "timestamps.npy").is_file():
        p.error(f"no camera frame cache in {args.cache_dir}: run build_finer_map.py (pass 1) first")
    if args.points_time_offset is None:
        args.points_time_offset, _ = glim_points_time_offset(args.map_dir)
    return args


def yaw_rate_deg_s(trajectory, t, dt=0.05):
    R = trajectory.matrices(np.array([t - dt, t + dt]))[:, :3, :3]
    r = R[0].T @ R[1]
    return np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))) / (2 * dt)


def pick_frames(trajectory, stamps, n, min_gap):
    inside = np.flatnonzero((stamps > trajectory.t_min + 0.5) & (stamps < trajectory.t_max - 0.5))
    rates = np.array([yaw_rate_deg_s(trajectory, stamps[i]) for i in inside])
    chosen = []
    for i in inside[np.argsort(-rates)]:
        if all(abs(stamps[i] - stamps[c]) > min_gap for c in chosen):
            chosen.append(int(i))
        if len(chosen) == n:
            break
    return chosen


def collect_edges(args, trajectory, stamps, chosen):
    """World-frame LiDAR silhouette points near each chosen frame's stamp."""
    prep_args = SimpleNamespace(edge_jump=1e9, bottom_rings=4, bottom_ring_min_range=0.8,
                                min_range=0.5, max_range=15.0)  # the build's defaults, edges kept
    windows = {c: (stamps[c] - EDGE_WINDOW_S, stamps[c] + EDGE_WINDOW_S) for c in chosen}
    last = max(w[1] for w in windows.values())
    edges = {c: [] for c in chosen}
    for msg in iter_topic(args.bag, args.points_topic):
        scan = OrganizedScan(msg, args.points_time_offset)
        start = scan.header_stamp
        if start > last:
            break
        hits = [c for c, (lo, hi) in windows.items() if lo - 0.11 < start < hi]
        if not hits:
            continue
        prep = prepare_scan(scan, trajectory, prep_args)
        if prep is None:
            continue
        r = scan.range_m
        silhouette = np.zeros_like(r, dtype=bool)
        for shift in (1, -1):
            silhouette |= (np.roll(r, shift, axis=1) - r > EDGE_JUMP_M) & (r > 0)
        is_edge = silhouette[prep.rows, prep.cols]
        t = prep.col_times[prep.cols]
        for c in hits:
            lo, hi = windows[c]
            edges[c].append(prep.points_world[is_edge & (t > lo) & (t < hi)])
    return {c: np.concatenate(v) for c, v in edges.items() if v and sum(map(len, v)) >= 50}


def gradient_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    g = cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))
    g = cv2.GaussianBlur(g, (0, 0), 4)
    return g / (g.mean() + 1e-6)


def edge_score(points, world_from_camera, sensors, grad):
    pc = (np.c_[points, np.ones(len(points))] @ np.linalg.inv(world_from_camera).T)[:, :3]
    pc = pc[pc[:, 2] > 0.3]
    if len(pc) < 20:
        return None
    px, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), sensors.K, sensors.D)
    px = np.round(px.reshape(-1, 2)).astype(np.int64)
    inside = (px[:, 0] >= 0) & (px[:, 0] < grad.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < grad.shape[0])
    if inside.sum() < 20:
        return None
    return float(grad[px[inside, 1], px[inside, 0]].mean())


def main():
    args = parse_args()
    trajectory = Trajectory(args.map_dir / "traj_lidar.txt")
    sensors = SensorConfig(args.map_dir / "config" / "config_sensors.json")
    stamps = np.load(args.cache_dir / "timestamps.npy")  # raw header stamps
    print(f"LiDAR time offset {args.points_time_offset:+.3f} s, {len(stamps)} cached camera frames")

    chosen = pick_frames(trajectory, stamps, args.frames, args.min_gap)
    rates = [yaw_rate_deg_s(trajectory, stamps[c]) for c in chosen]
    print(f"using {len(chosen)} frames turning at {min(rates):.0f}-{max(rates):.0f} deg/s")
    edges = collect_edges(args, trajectory, stamps, chosen)
    if not edges:
        sys.exit("no LiDAR silhouettes found near the chosen frames")

    deltas = np.round(np.arange(args.search[0], args.search[1] + args.step / 2, args.step), 6)
    scores = np.full((len(edges), len(deltas)), np.nan)
    for row, (c, pts) in enumerate(edges.items()):
        grad = gradient_image(args.cache_dir / "frames" / f"{c:05d}.jpg")
        for col, d in enumerate(deltas):
            pose = trajectory.matrices(np.array([stamps[c] + d]))[0] @ sensors.T_lidar_camera
            s = edge_score(pts, pose, sensors, grad)
            if s is not None:
                scores[row, col] = s
    mean = np.nanmean(scores, axis=0)
    best = int(np.nanargmax(mean))
    refined = deltas[best]
    if 0 < best < len(deltas) - 1:  # parabola through the peak and its neighbours
        y0, y1, y2 = mean[best - 1:best + 2]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            refined += 0.5 * (y0 - y2) / denom * args.step

    lo, hi = np.nanmin(mean), np.nanmax(mean)
    for d, s in zip(deltas, mean):
        print(f"  camera pose at stamp {d:+.3f} s: alignment {s:.3f} {'#' * int(40 * (s - lo) / max(hi - lo, 1e-9))}")
    per_frame = deltas[np.nanargmax(np.where(np.isnan(scores), -np.inf, scores), axis=1)]
    print(f"best offset {refined:+.3f} s (grid {deltas[best]:+.3f} s, {len(edges)} frames; per-frame peaks "
          f"median {np.median(per_frame):+.3f} s, IQR {np.percentile(per_frame, 25):+.3f}..{np.percentile(per_frame, 75):+.3f} s)")
    if best in (0, len(deltas) - 1):
        print("warning: the peak is at the edge of the search range; widen --search")
    print(f"-> build_finer_map.py ... --image-time-offset {refined:.3f}")


if __name__ == "__main__":
    main()
