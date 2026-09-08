#!/usr/bin/env python3
"""Build a finer-detail, RGB-colourised mesh of coverage1 via TSDF fusion.

See README.md for the full pipeline explanation. Summary:

  1. Load the GLIM global-mapping trajectory (traj_lidar.txt) and interpolate
     it at any timestamp with `src.trajectory.Trajectory`.
  2. Cache every /rgb/image_raw frame from the source rosbag to disk once
     (`src.colorizer.cache_frames`).
  3. Stream every raw /ouster/points scan from the bag (NOT the downsampled
     points GLIM's own map already stored) and, per scan:
       - drop invalid returns and within-ring depth-discontinuity edge noise
         using the organized (ring, column) layout (`src.ouster_scan`),
       - deskew every point into the world frame using the trajectory pose at
         that point's own capture time (all 64 beams in an Ouster column fire
         simultaneously, so this is one interpolation per column, not per
         point),
       - colourise each point from the nearest-in-time camera frame via
         projection with the calibrated extrinsics/distortion model, z-buffered
         per scan, falling back to intensity grayscale where RGB is
         unavailable (`src.colorizer`),
       - weight each point by how face-on it was seen (an incidence-angle
         estimate from organized-neighbour normals) and fuse it into a dense
         TSDF volume (`src.tsdf_volume`).
  4. Extract the zero level set with marching cubes, restricted to
     actually-observed voxels, and export PLY/OBJ/STL plus a JSON report.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.bagio import iter_topic, open_bag  # noqa: E402
from src.colorizer import FrameColorizer, cache_frames  # noqa: E402
from src.extrinsics import SensorConfig  # noqa: E402
from src.ouster_scan import OrganizedScan  # noqa: E402
from src.tsdf_volume import TSDFVolume  # noqa: E402
from src.trajectory import Trajectory  # noqa: E402

POINTS_TOPIC = "/ouster/points"
IMAGE_TOPIC = "/rgb/image_raw"


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", default="/home/autosweep/autosweep/dataset22jul/coverage1_lipede",
                   help="source rosbag2 directory (must contain metadata.yaml)")
    p.add_argument("--map-dir", default="/home/autosweep/autosweep/glim_maps/coverage1",
                   help="GLIM map directory providing traj_lidar.txt and config/config_sensors.json")
    p.add_argument("--out-dir", default=str(here / "outputs"))
    p.add_argument("--cache-dir", default=str(here / "cache"))
    p.add_argument("--voxel-size", type=float, default=0.03, help="TSDF voxel edge length, metres")
    p.add_argument("--truncation", type=float, default=None,
                   help="TSDF truncation distance, metres (default: 4 * voxel-size)")
    p.add_argument("--min-range", type=float, default=0.5, help="drop returns closer than this, metres")
    p.add_argument("--max-range", type=float, default=15.0, help="drop returns farther than this, metres")
    p.add_argument("--bottom-rings", type=int, default=4,
                   help="number of steepest-downward-looking rings (highest ring index) that use "
                        "--bottom-ring-min-range instead of --min-range, to clear the robot's own "
                        "chassis/wheel returns those rings see at close range (fixed azimuth, just "
                        "past --min-range) that otherwise get fused into a trace of the trajectory")
    p.add_argument("--bottom-ring-min-range", type=float, default=0.8,
                   help="min-range override for the bottom rings, metres")
    p.add_argument("--edge-jump", type=float, default=0.3,
                   help="within-ring range discontinuity, metres, above which a point is dropped as edge noise")
    p.add_argument("--padding", type=float, default=6.0,
                   help="metres of TSDF volume padding around the trajectory bounding box")
    p.add_argument("--max-image-dt", type=float, default=0.09,
                   help="max seconds between a scan column and the camera frame used to colour it")
    p.add_argument("--min-weight", type=float, default=1.5,
                   help="minimum accumulated TSDF weight for a voxel to be considered observed")
    p.add_argument("--limit-scans", type=int, default=None, help="debug: only process the first N scans")
    p.add_argument("--skip-frame-cache", action="store_true", help="reuse an existing frame cache as-is")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def estimate_intensity_range(bag_path, stride=8, sample_cap=2_000_000):
    """Cheap single pass over every `stride`-th scan's intensity field only,
    to fix a global percentile-normalised grayscale range up front (matches
    the fixed global range used by the earlier color_intensity.py pipeline,
    rather than a per-scan range that would flicker frame to frame)."""
    samples = []
    total = 0
    for i, msg in enumerate(iter_topic(bag_path, POINTS_TOPIC)):
        if i % stride != 0:
            continue
        from sensor_msgs_py import point_cloud2
        arr = point_cloud2.read_points(msg, field_names=["intensity", "range"], skip_nans=False)
        valid = arr["range"] > 0
        vals = arr["intensity"][valid]
        if len(vals):
            samples.append(vals)
            total += len(vals)
        if total >= sample_cap:
            break
    all_vals = np.concatenate(samples)
    lo, hi = np.percentile(all_vals, [1.0, 99.0])
    log(f"intensity range estimate from {len(all_vals)} samples: [{lo:.1f}, {hi:.1f}]")
    return float(lo), float(hi)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truncation = args.truncation if args.truncation is not None else 4.0 * args.voxel_size

    map_dir = Path(args.map_dir)
    sensors = SensorConfig(map_dir / "config" / "config_sensors.json")
    trajectory = Trajectory(map_dir / "traj_lidar.txt")
    log(f"trajectory: {len(trajectory.times)} poses, t in [{trajectory.t_min:.2f}, {trajectory.t_max:.2f}]")

    bounds_min, bounds_max = trajectory.bounds(padding=args.padding)
    log(f"TSDF bounds: min={bounds_min}, max={bounds_max}, voxel={args.voxel_size} m, trunc={truncation} m")
    volume = TSDFVolume(bounds_min, bounds_max, args.voxel_size, truncation)
    log(f"TSDF grid dims: {volume.dims.tolist()} ({int(np.prod(volume.dims)):,} voxels)")

    stamps_path = cache_dir / "timestamps.npy"
    if args.skip_frame_cache and stamps_path.exists():
        log("reusing existing frame cache")
    else:
        log("pass 1/3: caching camera frames from the bag...")
        cache_frames(lambda: open_bag(args.bag), IMAGE_TOPIC, cache_dir, log=log)

    log("pass 2/3: estimating a global intensity normalisation range...")
    lo, hi = estimate_intensity_range(args.bag)

    colorizer = FrameColorizer(cache_dir, sensors, max_dt=args.max_image_dt)
    colorizer.set_intensity_range(lo, hi)

    log("pass 3/3: streaming scans, deskewing, colourising and fusing into the TSDF...")
    n_scans = 0
    n_points_total = 0
    n_points_rgb = 0
    t_start = time.time()

    for msg in iter_topic(args.bag, POINTS_TOPIC):
        if args.limit_scans is not None and n_scans >= args.limit_scans:
            break

        scan = OrganizedScan(msg)
        keep = scan.edge_filter(jump_m=args.edge_jump)
        min_range_per_ring = np.where(
            np.arange(scan.rows) >= scan.rows - args.bottom_rings,
            args.bottom_ring_min_range, args.min_range,
        )
        keep &= (scan.range_m >= min_range_per_ring[:, None]) & (scan.range_m <= args.max_range)

        col_times = scan.column_times_s()
        col_in_range = trajectory.in_range(col_times)
        keep &= col_in_range[None, :]
        if not np.any(keep):
            n_scans += 1
            continue

        # one pose per column (all beams in a column share a timestamp)
        col_T_world_lidar = trajectory.matrices(col_times)  # (cols, 4, 4)
        col_frame_idx = colorizer.nearest_frame_index(col_times)  # (cols,)

        rows, cols = np.nonzero(keep)
        xyz_local = scan.xyz[rows, cols]
        intensity = scan.intensity[rows, cols]

        R = col_T_world_lidar[cols, :3, :3]
        t = col_T_world_lidar[cols, :3, 3]
        points_world = np.einsum("nij,nj->ni", R, xyz_local) + t
        origins_world = t  # sensor position at each point's own capture time

        normals, normals_valid = scan.organized_normals()
        n_sel = normals[rows, cols]
        nv_sel = normals_valid[rows, cols]
        ray_dir = xyz_local / np.linalg.norm(xyz_local, axis=1, keepdims=True)
        incidence = np.abs(np.einsum("ij,ij->i", n_sel, -ray_dir))
        weights = np.where(nv_sel, np.clip(incidence, 0.2, 1.0), 0.5).astype(np.float32)

        colors = np.empty((len(points_world), 3), dtype=np.float32)
        point_frame_idx = col_frame_idx[cols]
        for fid in np.unique(point_frame_idx):
            sel = point_frame_idx == fid
            if fid < 0:
                colors[sel] = colorizer.intensity_to_gray(intensity[sel])
                continue
            world_from_camera = trajectory.matrices(colorizer.timestamps[fid:fid + 1])[0] @ sensors.T_lidar_camera
            frame_colors, used_rgb = colorizer.colorize_frame_group(
                points_world[sel], world_from_camera, fid, intensity[sel]
            )
            colors[sel] = frame_colors
            n_points_rgb += int(used_rgb.sum())

        volume.integrate(points_world, origins_world, colors, weights)

        n_scans += 1
        n_points_total += len(points_world)
        if n_scans % 200 == 0:
            elapsed = time.time() - t_start
            log(f"  {n_scans} scans, {n_points_total:,} points fused "
                f"({elapsed:.0f}s, {n_scans / elapsed:.1f} scans/s)")

    log(f"fusion done: {n_scans} scans, {n_points_total:,} points "
        f"({100.0 * n_points_rgb / max(n_points_total, 1):.1f}% coloured from RGB)")

    log("extracting mesh from the TSDF...")
    verts, faces, vertex_colors, stats = volume.extract_mesh(min_weight=args.min_weight)
    log(f"raw marching-cubes mesh: {len(verts):,} vertices, {len(faces):,} triangles")

    export_mesh(out_dir, verts, faces, vertex_colors)

    report = {
        "bag": str(args.bag),
        "map_dir": str(args.map_dir),
        "voxel_size_m": args.voxel_size,
        "truncation_distance_m": truncation,
        "min_range_m": args.min_range,
        "max_range_m": args.max_range,
        "bottom_rings": args.bottom_rings,
        "bottom_ring_min_range_m": args.bottom_ring_min_range,
        "edge_jump_m": args.edge_jump,
        "padding_m": args.padding,
        "max_image_dt_s": args.max_image_dt,
        "min_weight": args.min_weight,
        "scans_processed": n_scans,
        "points_fused": n_points_total,
        "points_coloured_from_rgb_pct": 100.0 * n_points_rgb / max(n_points_total, 1),
        "intensity_normalisation_range": [lo, hi],
        "vertices": int(len(verts)),
        "triangles": int(len(faces)),
        **stats,
    }
    with open(out_dir / "finer_map_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log(f"report -> {out_dir / 'finer_map_report.json'}")


def export_mesh(out_dir, verts, faces, vertex_colors):
    import vtk
    from vtk.util import numpy_support

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float64)))

    cells = vtk.vtkCellArray()
    n_faces = len(faces)
    cell_data = np.empty((n_faces, 4), dtype=np.int64)
    cell_data[:, 0] = 3
    cell_data[:, 1:] = faces
    cells.SetCells(n_faces, numpy_support.numpy_to_vtkIdTypeArray(cell_data.reshape(-1)))

    colors_u8 = np.clip(vertex_colors, 0, 255).astype(np.uint8)[:, ::-1]  # BGR -> RGB
    color_array = numpy_support.numpy_to_vtk(np.ascontiguousarray(colors_u8), array_type=vtk.VTK_UNSIGNED_CHAR)
    color_array.SetName("RGB")

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    poly.GetPointData().SetScalars(color_array)

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()
    cleaned = clean.GetOutput()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(cleaned)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    final = normals.GetOutput()

    for ext, writer_cls in [("ply", vtk.vtkPLYWriter), ("obj", vtk.vtkOBJWriter), ("stl", vtk.vtkSTLWriter)]:
        writer = writer_cls()
        writer.SetFileName(str(out_dir / f"coverage1_finer_mesh.{ext}"))
        writer.SetInputData(final)
        if ext == "ply":
            writer.SetArrayName("RGB")
            writer.SetColorModeToDefault()
            writer.SetFileTypeToBinary()
        if ext == "stl":
            writer.SetFileTypeToBinary()
        writer.Write()


if __name__ == "__main__":
    main()
