#!/usr/bin/env python3
"""Build a finer-detail, RGB-colourised mesh of a recorded bag via TSDF fusion.

Usage: build_finer_map.py <bag_name>  (e.g. coverage1)

<bag_name> selects both the raw rosbag2 directory
<datasets-root>/<bag_name><bag-suffix> (the `_lipede` recording by default, e.g.
dataset22jul/coverage1_lipede) and the GLIM map directory with its odometry
<maps-root>/<bag_name> (e.g. glim_maps/coverage1). Outputs go to outputs/<bag_name>/ and the camera frame
cache to cache/<bag_name>/.

See README.md for the full pipeline explanation. Summary:

  1. Load the GLIM global-mapping trajectory (traj_lidar.txt) and interpolate
     it at any timestamp with `src.trajectory.Trajectory`.
  2. Cache every /rgb/image_raw frame from the source rosbag to disk once
     (`src.colorizer.cache_frames`).
  3. Estimate the floor from every `--floor-sample-stride`-th scan: RANSAC
     then SVD plane fit on horizontal surfaces near the sensor, plus a coarse
     per-cell height correction (`src.floor`).
  4. Stream every raw /ouster/points scan from the bag (NOT the downsampled
     points GLIM's own map already stored) and, per scan:
       - drop invalid returns and within-ring depth-discontinuity edge noise
         using the organized (ring, column) layout (`src.ouster_scan`),
       - deskew every point into the world frame using the trajectory pose at
         that point's own capture time (all 64 beams in an Ouster column fire
         simultaneously, so this is one interpolation per column, not per
         point),
       - drop points more than `--floor-margin` below the local floor (LiDAR
         reflections off the floor) and, optionally, above a ceiling height,
       - classify floor points (within `--floor-band` of the floor, facing
         up): they are not fused, only used to colour a synthetic floor,
       - colourise each point from the nearest-in-time camera frame via
         projection with the calibrated extrinsics/distortion model, z-buffered
         per scan, falling back to intensity grayscale where RGB is
         unavailable (`src.colorizer`),
       - weight each point by how face-on it was seen (an incidence-angle
         estimate from organized-neighbour normals) and fuse it into a dense
         TSDF volume (`src.tsdf_volume`),
       - carve free space along a subsample of the scan's rays (rays that
         hit the floor right down to it), so things that moved (people) fade
         out of the map instead of staying as ghosts.
  5. Extract the zero level set with marching cubes, restricted to cubes
     whose corners were all observed, add a synthetic floor over the map's
     floor outline, lying exactly on the floor model and coloured from the
     floor points (`src.floor_slab`), shift the result in z so the floor is
     at z = 0 under the GLIM origin, and export them together as a single PLY, plus the
     floor model (floor_model.npz, reused by refine_mesh.py) and a JSON
     report.
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
from src.floor import FloorModel, estimate_floor, select_candidates  # noqa: E402
from src.floor_slab import FloorColorGrid, build_slab  # noqa: E402
from src.mesh_io import finish_and_write, polydata_from_arrays  # noqa: E402
from src.ouster_scan import OrganizedScan  # noqa: E402
from src.tsdf_volume import TSDFVolume  # noqa: E402
from src.trajectory import Trajectory  # noqa: E402

POINTS_TOPIC = "/ouster/points"
IMAGE_TOPIC = "/rgb/image_raw"
DATASETS_ROOT = "/home/autosweep/autosweep/dataset22jul"
MAPS_ROOT = "/home/autosweep/autosweep/glim_maps"
BAG_SUFFIX = "_lipede"


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="bag name: reads the rosbag2 <datasets-root>/<name><bag-suffix> and the "
                                "GLIM map/odometry <maps-root>/<name>, e.g. coverage1")
    p.add_argument("--datasets-root", default=DATASETS_ROOT, help="directory holding the rosbag2 directories")
    p.add_argument("--maps-root", default=MAPS_ROOT, help="directory holding the GLIM map directories")
    p.add_argument("--bag-suffix", default=BAG_SUFFIX,
                   help="appended to <name> to get the rosbag2 directory name (default: %(default)s)")
    p.add_argument("--bag", default=None,
                   help="override the source rosbag2 directory (default: <datasets-root>/<name><bag-suffix>)")
    p.add_argument("--map-dir", default=None,
                   help="override the GLIM map directory providing traj_lidar.txt and "
                        "config/config_sensors.json (default: <maps-root>/<name>)")
    p.add_argument("--points-topic", default=POINTS_TOPIC, help="Ouster PointCloud2 topic (e.g. /ousterDome/points)")
    p.add_argument("--image-topic", default=IMAGE_TOPIC, help="camera image topic")
    p.add_argument("--out-dir", default=None, help="default: outputs/<name>")
    p.add_argument("--cache-dir", default=None, help="default: cache/<name>")
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
    p.add_argument("--padding-below-floor", type=float, default=0.5,
                   help="metres of TSDF volume padding below the lowest point of the floor model "
                        "(replaces --padding on the bottom face; ignored without a floor model)")
    p.add_argument("--max-image-dt", type=float, default=0.09,
                   help="max seconds between a scan column and the camera frame used to colour it")
    p.add_argument("--intensity-color-weight", type=float, default=0.02,
                   help="colour-average weight of an intensity-grayscale sample relative to an RGB "
                        "sample, so RGB dominates any voxel the camera saw (1.0 = equal weight)")
    p.add_argument("--min-weight", type=float, default=1.5,
                   help="minimum accumulated TSDF weight for a voxel to be considered observed")
    p.add_argument("--no-floor-filter", action="store_true",
                   help="skip floor estimation and keep below-floor points (also disables the "
                        "floor clearance used by carving)")
    p.add_argument("--floor-model", default=None,
                   help="load a floor_model.npz from an earlier run instead of estimating it again")
    p.add_argument("--floor-margin", type=float, default=0.05,
                   help="drop points more than this many metres below the local floor")
    p.add_argument("--floor-band", type=float, default=0.05,
                   help="points at most this many metres above the local floor, with an upward normal "
                        "(or none), are floor: not fused, only used to colour the synthetic floor")
    p.add_argument("--floor-normal-angle", type=float, default=30.0,
                   help="max angle, degrees, between a floor point's normal and the floor normal")
    p.add_argument("--floor-slab-top", choices=["model", "highest", "plane"], default="model",
                   help="synthetic floor: 'model' = a single surface exactly on the floor model (global "
                        "plane + smooth per-cell correction; walls and objects meet it with no gap); "
                        "'highest' / 'plane' = a closed flat slab parallel to the global plane, its top at "
                        "the measured floor's highest point / on the global plane")
    p.add_argument("--floor-slab-thickness", type=float, default=None,
                   help="flat slab thickness, metres ('highest'/'plane' only; default: down to the lowest "
                        "point of the measured floor minus --floor-margin)")
    p.add_argument("--floor-fill-radius", type=float, default=0.15,
                   help="close gaps up to this many metres in the observed floor before filling holes")
    p.add_argument("--ceiling-height", type=float, default=None,
                   help="drop points more than this many metres above the floor (default: keep all)")
    p.add_argument("--floor-cell-size", type=float, default=1.0,
                   help="cell size, metres, of the floor model's per-cell height correction")
    p.add_argument("--floor-sample-stride", type=int, default=10,
                   help="use every N-th scan to estimate the floor")
    p.add_argument("--carve-ray-stride", type=int, default=8,
                   help="free-space carving uses one ray in N per scan (0 disables carving)")
    p.add_argument("--carve-weight", type=float, default=0.2,
                   help="weight of one free-space observation per voxel per scan")
    p.add_argument("--carve-floor-clearance", type=float, default=0.0,
                   help="never carve voxels within this many metres of the floor (default 0: the floor "
                        "is not fused, so carving cannot erode it)")
    p.add_argument("--limit-scans", type=int, default=None, help="debug: only process the first N scans")
    p.add_argument("--skip-frame-cache", action="store_true", help="reuse an existing frame cache as-is")
    args = p.parse_args()
    args.bag = args.bag or str(Path(args.datasets_root) / f"{args.name}{args.bag_suffix}")
    args.map_dir = args.map_dir or str(Path(args.maps_root) / args.name)
    args.out_dir = args.out_dir or str(here / "outputs" / args.name)
    args.cache_dir = args.cache_dir or str(here / "cache" / args.name)

    missing = [f for f in (Path(args.bag) / "metadata.yaml",
                           Path(args.map_dir) / "traj_lidar.txt",
                           Path(args.map_dir) / "config" / "config_sensors.json") if not f.is_file()]
    if missing:
        p.error("missing input file(s) for '%s':\n  %s" % (args.name, "\n  ".join(map(str, missing))))
    return args


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def estimate_intensity_range(bag_path, points_topic, stride=8, sample_cap=2_000_000):
    """Cheap single pass over every `stride`-th scan's intensity field only,
    to fix a global percentile-normalised grayscale range up front (matches
    the fixed global range used by the earlier color_intensity.py pipeline,
    rather than a per-scan range that would flicker frame to frame)."""
    samples = []
    total = 0
    for i, msg in enumerate(iter_topic(bag_path, points_topic)):
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

    stamps_path = cache_dir / "timestamps.npy"
    if args.skip_frame_cache and stamps_path.exists():
        log("reusing existing frame cache")
    else:
        log("pass 1/4: caching camera frames from the bag...")
        cache_frames(lambda: open_bag(args.bag), args.image_topic, cache_dir, log=log)

    log("pass 2/4: estimating a global intensity normalisation range...")
    lo, hi = estimate_intensity_range(args.bag, args.points_topic)

    colorizer = FrameColorizer(cache_dir, sensors, max_dt=args.max_image_dt)
    colorizer.set_intensity_range(lo, hi)

    floor = None
    if args.floor_model:
        floor = FloorModel.load(args.floor_model)
        log(f"loaded floor model from {args.floor_model}: {floor.summary()}")
    elif not args.no_floor_filter:
        log("pass 3/4: estimating the floor...")
        floor = estimate_floor_from_bag(args, trajectory)
    if floor is not None:
        floor.save(out_dir / "floor_model.npz")

    bounds_min, bounds_max = trajectory.bounds(padding=args.padding)
    if floor is not None:
        # nothing is ever observed below the floor, so only pad a little under it
        # instead of the full --padding (which roughly doubled the grid's height)
        bounds_min[2] = floor.lowest_z(bounds_min[:2], bounds_max[:2]) - args.padding_below_floor
    log(f"TSDF bounds: min={bounds_min}, max={bounds_max}, voxel={args.voxel_size} m, trunc={truncation} m")
    volume = TSDFVolume(bounds_min, bounds_max, args.voxel_size, truncation)
    log(f"TSDF grid dims: {volume.dims.tolist()} ({int(np.prod(volume.dims)):,} voxels)")
    floor_grid = FloorColorGrid(volume.origin[:2], args.voxel_size, volume.dims[:2]) if floor is not None else None
    cos_floor = np.cos(np.radians(args.floor_normal_angle))

    carve_keep = None
    if floor is not None and args.carve_floor_clearance > 0:
        carve_keep = lambda centres: floor.height_above_floor(centres) > args.carve_floor_clearance  # noqa: E731

    log("pass 4/4: streaming scans, deskewing, colourising and fusing into the TSDF...")
    rng = np.random.default_rng(0)
    n_scans = 0  # scans read from the bag (what --limit-scans counts)
    n_scans_fused = 0  # scans that contributed at least one point
    n_points_total = 0
    n_points_rgb = 0
    n_points_floor_dropped = 0
    n_points_floor = 0
    n_voxels_carved = 0
    t_start = time.time()

    for msg in iter_topic(args.bag, args.points_topic):
        if args.limit_scans is not None and n_scans >= args.limit_scans:
            break

        prep = prepare_scan(OrganizedScan(msg), trajectory, args)
        if prep is not None and floor is not None:
            height = floor.height_above_floor(prep.points_world)
            keep = height >= -args.floor_margin
            if args.ceiling_height is not None:
                keep &= height <= args.ceiling_height
            n_points_floor_dropped += int((~keep).sum())
            prep = prep.subset(keep)
            height = height[keep]
        if prep is None or len(prep.points_world) == 0:
            n_scans += 1
            continue

        points_world, origins_world, intensity = prep.points_world, prep.origins_world, prep.intensity
        cols = prep.cols
        col_frame_idx = colorizer.nearest_frame_index(prep.col_times)  # (cols,)

        incidence = np.abs(np.einsum("ij,ij->i", prep.normals_local, -prep.ray_dir_local))
        weights = np.where(prep.normals_valid, np.clip(incidence, 0.2, 1.0), 0.5).astype(np.float32)

        colors = np.empty((len(points_world), 3), dtype=np.float32)
        point_rgb = np.zeros(len(points_world), dtype=bool)
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
            point_rgb[sel] = used_rgb
            n_points_rgb += int(used_rgb.sum())

        color_weights = weights * np.where(point_rgb, 1.0, args.intensity_color_weight).astype(np.float32)

        # floor points only colour the synthetic slab; everything else is fused
        is_floor = np.zeros(len(points_world), dtype=bool)
        if floor_grid is not None:
            normals_world = np.einsum("nij,nj->ni", prep.R, prep.normals_local)
            facing_up = np.abs(normals_world @ floor.normal) >= cos_floor
            is_floor = (height <= args.floor_band) & (facing_up | ~prep.normals_valid)
            floor_grid.add(points_world[is_floor], colors[is_floor], color_weights[is_floor])
        fuse = ~is_floor
        volume.integrate(points_world[fuse], origins_world[fuse], colors[fuse], weights[fuse], color_weights[fuse])
        if args.carve_ray_stride > 0:
            rays = rng.random(len(points_world)) < 1.0 / args.carve_ray_stride
            # a ray that ended on the (unfused) floor carves right down to it
            margin = np.where(is_floor[rays], args.voxel_size, truncation + args.voxel_size)
            n_voxels_carved += volume.carve(points_world[rays], origins_world[rays], args.carve_weight,
                                            keep_voxel=carve_keep, stop_margin=margin)

        n_scans += 1
        n_scans_fused += 1
        n_points_total += int(fuse.sum())
        n_points_floor += int(is_floor.sum())
        if n_scans % 200 == 0:
            elapsed = time.time() - t_start
            log(f"  {n_scans} scans read, {n_points_total:,} points fused "
                f"({elapsed:.0f}s, {n_scans / elapsed:.1f} scans/s)")

    log(f"fusion done: {n_scans_fused} of {n_scans} scans fused "
        f"({n_scans - n_scans_fused} skipped: outside the trajectory or nothing left after filtering), "
        f"{n_points_total:,} points "
        f"+ {n_points_floor:,} floor points "
        f"({100.0 * n_points_rgb / max(n_points_total + n_points_floor, 1):.1f}% coloured from RGB), "
        f"{n_points_floor_dropped:,} dropped below the floor/above the ceiling, "
        f"{n_voxels_carved:,} free-space voxel updates")

    log("extracting mesh from the TSDF...")
    verts, faces, vertex_colors, stats = volume.extract_mesh(min_weight=args.min_weight)
    log(f"raw marching-cubes mesh: {len(verts):,} vertices, {len(faces):,} triangles")

    slab_report = None
    if floor_grid is not None:
        mask = floor_grid.outline(args.floor_fill_radius)
        if args.floor_slab_top == "model":
            top_z, bottom_z, thickness = floor.floor_z, None, 0.0
        else:
            top_h = float(floor.grid.max()) if args.floor_slab_top == "highest" else 0.0
            if args.floor_slab_thickness is not None:
                bottom_h = top_h - args.floor_slab_thickness
            else:
                bottom_h = min(float(floor.grid.min()), 0.0) - args.floor_margin
            top_z = lambda xy: floor.plane_z(xy, top_h)  # noqa: E731
            bottom_z = lambda xy: floor.plane_z(xy, bottom_h)  # noqa: E731
            thickness = top_h - bottom_h
        s_verts, s_faces, s_colors = build_slab(mask, floor_grid.colors(), floor_grid.origin, args.voxel_size,
                                                top_z, bottom_z)
        slab_report = {
            "top": args.floor_slab_top,
            "thickness_m": round(thickness, 4),
            "area_m2": round(float(mask.sum()) * args.voxel_size ** 2, 2),
            "observed_floor_pct_of_area": round(
                100.0 * float(((floor_grid.weight > 0).reshape(mask.shape) & mask).sum()) / max(mask.sum(), 1), 1),
            "vertices": int(len(s_verts)),
            "triangles": int(len(s_faces)),
        }
        log(f"floor slab: {slab_report}")
        s_colors_u8 = np.clip(s_colors, 0, 255).astype(np.uint8)[:, ::-1]

    colors_u8 = np.clip(vertex_colors, 0, 255).astype(np.uint8)[:, ::-1]  # BGR -> RGB
    z_offset = 0.0
    if slab_report is not None:
        faces = np.concatenate([faces, s_faces + len(verts)])
        verts = np.concatenate([verts, s_verts])
        colors_u8 = np.concatenate([colors_u8, s_colors_u8])
        # shift the output along z only, so the floor's top surface is at
        # z = 0 under the GLIM origin (the floor stays tilted like in GLIM's frame)
        z_offset = -float(top_z(np.zeros((1, 2)))[0])
        verts = verts + np.array([0.0, 0.0, z_offset])
        log(f"output frame: z shifted by {z_offset:+.4f} m (slab top at z = 0 under the GLIM origin)")
    # the only mesh output: fused objects + floor slab (refine_mesh.py separates
    # the slab again by itself, see src.floor_slab.find_slab)
    finish_and_write(polydata_from_arrays(verts, faces, colors_u8), out_dir, f"{args.name}_finer_mesh",
                     formats=("ply",))

    report = {
        "name": args.name,
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
        "padding_below_floor_m": args.padding_below_floor if floor is not None else None,
        "tsdf_bounds_min": [round(float(v), 4) for v in volume.origin],
        "max_image_dt_s": args.max_image_dt,
        "intensity_color_weight": args.intensity_color_weight,
        "min_weight": args.min_weight,
        "floor_filter": floor is not None,
        "floor_margin_m": args.floor_margin,
        "ceiling_height_m": args.ceiling_height,
        "floor_band_m": args.floor_band,
        "floor_normal_angle_deg": args.floor_normal_angle,
        "floor_fill_radius_m": args.floor_fill_radius,
        "floor_slab": slab_report,
        # added to every output z; everything else in this report (bounds,
        # floor model) is in GLIM's frame
        "output_z_offset_m": round(z_offset, 6),
        "floor_model": floor.summary() if floor is not None else None,
        "points_dropped_by_floor_ceiling": n_points_floor_dropped,
        "points_floor_not_fused": n_points_floor,
        "carve_ray_stride": args.carve_ray_stride,
        "carve_weight": args.carve_weight,
        "carve_floor_clearance_m": args.carve_floor_clearance,
        "free_space_voxel_updates": n_voxels_carved,
        "scans_read": n_scans,
        "scans_fused": n_scans_fused,
        "points_fused": n_points_total,
        "points_coloured_from_rgb_pct": 100.0 * n_points_rgb / max(n_points_total + n_points_floor, 1),
        "intensity_normalisation_range": [lo, hi],
        "vertices": int(len(verts)),
        "triangles": int(len(faces)),
        **stats,
    }
    with open(out_dir / "finer_map_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log(f"report -> {out_dir / 'finer_map_report.json'}")


class PreparedScan:
    """One scan's filtered points, deskewed into the world frame."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def subset(self, mask):
        per_point = ["rows", "cols", "xyz_local", "intensity", "points_world", "origins_world",
                     "R", "normals_local", "normals_valid", "ray_dir_local"]
        fields = {k: (v[mask] if k in per_point else v) for k, v in self.__dict__.items()}
        return PreparedScan(**fields)


def prepare_scan(scan, trajectory, args):
    """Range/edge/trajectory-coverage filtering and per-column deskewing
    shared by the floor-estimation and fusion passes. None if nothing is left."""
    keep = scan.edge_filter(jump_m=args.edge_jump)
    min_range_per_ring = np.where(
        np.arange(scan.rows) >= scan.rows - args.bottom_rings,
        args.bottom_ring_min_range, args.min_range,
    )
    keep &= (scan.range_m >= min_range_per_ring[:, None]) & (scan.range_m <= args.max_range)

    col_times = scan.column_times_s()
    keep &= trajectory.in_range(col_times)[None, :]
    if not np.any(keep):
        return None

    # one pose per column (all beams in a column share a timestamp)
    col_T_world_lidar = trajectory.matrices(col_times)  # (cols, 4, 4)

    rows, cols = np.nonzero(keep)
    xyz_local = scan.xyz[rows, cols]
    R = col_T_world_lidar[cols, :3, :3]
    t = col_T_world_lidar[cols, :3, 3]
    normals, normals_valid = scan.organized_normals()
    return PreparedScan(
        col_times=col_times, rows=rows, cols=cols, xyz_local=xyz_local,
        intensity=scan.intensity[rows, cols],
        points_world=np.einsum("nij,nj->ni", R, xyz_local) + t,
        origins_world=t,  # sensor position at each point's own capture time
        R=R, normals_local=normals[rows, cols], normals_valid=normals_valid[rows, cols],
        ray_dir_local=xyz_local / np.linalg.norm(xyz_local, axis=1, keepdims=True),
    )


def estimate_floor_from_bag(args, trajectory, per_scan_cap=4000):
    rng = np.random.default_rng(0)
    pts, dzs = [], []
    for i, msg in enumerate(iter_topic(args.bag, args.points_topic)):
        if args.limit_scans is not None and i >= args.limit_scans:
            break
        if i % args.floor_sample_stride != 0:
            continue
        prep = prepare_scan(OrganizedScan(msg), trajectory, args)
        if prep is None:
            continue
        sensor_up = prep.R[:, :, 2]
        normals_world = np.einsum("nij,nj->ni", prep.R, prep.normals_local)
        mask, dz = select_candidates(prep.points_world, prep.origins_world, normals_world, sensor_up,
                                     prep.normals_valid)
        sel = np.flatnonzero(mask)
        if len(sel) > per_scan_cap:
            sel = rng.choice(sel, per_scan_cap, replace=False)
        pts.append(prep.points_world[sel])
        dzs.append(dz[sel])
    up = trajectory.rotations.as_matrix()[:, :, 2].mean(axis=0)
    return estimate_floor(np.concatenate(pts), np.concatenate(dzs), up,
                          cell_size_m=args.floor_cell_size, log=log)


if __name__ == "__main__":
    main()
