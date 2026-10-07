#!/usr/bin/env python3
"""Post-process a finished finer_mapping mesh into a cleaner, lighter one.

Usage: refine_mesh.py <bag_name>  (e.g. coverage1)

Reads outputs/<name>/<name>_finer_mesh.ply and the floor model the build
saved next to it (outputs/<name>/floor_model.npz), and writes
outputs/<name>/<name>_refined_mesh.{ply,obj,stl} plus refine_report.json.
Every step runs in seconds to minutes, so they can be tuned without re-running
TSDF fusion; each can be switched off by setting its parameter to 0.

  1. Clip: drop triangles with a vertex more than --clip-margin below the local
     floor (or above --ceiling-height). The build already filters below-floor
     points before fusion, so this is mostly a safety net, and what makes this
     script useful on meshes built without the floor filter.
  2. Remove small disconnected pieces (fewer than --min-component triangles):
     floating fragments from noise or surfaces glimpsed through glass.
  3. Smooth with a windowed-sinc (Taubin-style) filter: removes the voxel-scale
     stair-stepping marching cubes leaves, without the shrinkage of plain
     Laplacian smoothing.
  4. Flatten the floor: vertices within --flatten-tolerance of the floor whose
     normal is close to the floor normal are projected onto it.
  5. Fill holes up to --fill-holes metres across (mostly small floor gaps).
  6. Quadric decimation by --decimate (fraction of triangles to remove),
     which mostly merges triangles on flat areas.

The build's synthetic floor is found in the input (the connected piece lying
exactly on the floor model, or on two planes parallel to it), set aside before step 1
and added back unchanged after step 6; step 4 is skipped then, since there is
no fused floor left to flatten. --refine-floor-slab processes it like the
rest instead.

If the floor model is missing (e.g. a mesh built before floor estimation
existed), it is estimated from the mesh itself, using upward-facing vertices
near the GLIM trajectory (--map-dir).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import vtk
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.floor import FloorModel, estimate_floor  # noqa: E402
from src.floor_slab import find_slab  # noqa: E402
from src.mesh_io import arrays_from_polydata, finish_and_write, polydata_from_arrays, read_ply, run_filter  # noqa: E402
from src.trajectory import Trajectory  # noqa: E402

MAPS_ROOT = "/home/autosweep/autosweep/glim_maps"


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="bag name used for the build, e.g. coverage1")
    p.add_argument("--input", default=None, help="mesh to refine (default: outputs/<name>/<name>_finer_mesh.ply)")
    p.add_argument("--floor-model", default=None,
                   help="floor_model.npz (default: next to the input mesh; estimated from the mesh if absent)")
    p.add_argument("--map-dir", default=None,
                   help="GLIM map dir, only needed to estimate a missing floor model (default: <maps-root>/<name>)")
    p.add_argument("--maps-root", default=MAPS_ROOT)
    p.add_argument("--out-dir", default=None, help="default: the input mesh's directory")
    p.add_argument("--clip-margin", type=float, default=0.05,
                   help="drop triangles with a vertex more than this many metres below the floor")
    p.add_argument("--ceiling-height", type=float, default=None,
                   help="drop triangles with a vertex more than this many metres above the floor")
    p.add_argument("--min-component", type=int, default=500,
                   help="remove connected pieces with fewer triangles than this (0 keeps all)")
    p.add_argument("--smooth-iterations", type=int, default=15, help="windowed-sinc iterations (0 disables)")
    p.add_argument("--smooth-passband", type=float, default=0.1,
                   help="windowed-sinc pass band, (0, 2]; lower smooths more")
    p.add_argument("--flatten-tolerance", type=float, default=0.03,
                   help="project floor vertices within this many metres of the floor onto it (0 disables)")
    p.add_argument("--flatten-max-angle", type=float, default=25.0,
                   help="only flatten vertices whose normal is within this many degrees of the floor normal")
    p.add_argument("--fill-holes", type=float, default=0.3,
                   help="fill holes up to about this size in metres (0 disables)")
    p.add_argument("--refine-floor-slab", action="store_true",
                   help="treat the build's synthetic floor slab like the rest of the mesh instead of "
                        "setting it aside and adding it back unchanged")
    p.add_argument("--decimate", type=float, default=0.5,
                   help="fraction of triangles to remove by quadric decimation (0 disables)")
    args = p.parse_args()

    finer = here / "outputs" / args.name / f"{args.name}_finer_mesh.ply"
    if args.input is None and finer.is_file():
        args.input = finer
    args.input = Path(args.input or here / "outputs" / args.name / f"{args.name}_mesh.ply")
    args.out_dir = Path(args.out_dir or args.input.parent)
    args.floor_model = Path(args.floor_model or args.input.parent / "floor_model.npz")
    args.map_dir = Path(args.map_dir or Path(args.maps_root) / args.name)
    if not args.input.is_file():
        p.error(f"input mesh not found: {args.input}")
    return args


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vertex_normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def keep_faces(verts, faces, colors, face_mask):
    """Drop faces not in `face_mask`, then any vertex no remaining face uses."""
    faces = faces[face_mask]
    used = np.zeros(len(verts), dtype=bool)
    used[faces.reshape(-1)] = True
    remap = np.cumsum(used) - 1
    return verts[used], remap[faces], colors[used]


def floor_from_mesh(verts, faces, map_dir, radius_m=4.0, max_normal_deg=15.0):
    """Floor candidates for `estimate_floor` from the mesh itself: upward-
    facing vertices within `radius_m` of the trajectory, with their height
    measured against the nearest trajectory pose."""
    traj = Trajectory(map_dir / "traj_lidar.txt")
    up = traj.rotations.as_matrix()[:, :, 2].mean(axis=0)
    up /= np.linalg.norm(up)
    normals = vertex_normals(verts, faces)
    facing = np.abs(normals @ up) >= np.cos(np.radians(max_normal_deg))
    dist, nearest = cKDTree(traj.positions[:, :2]).query(verts[:, :2])
    dz = np.einsum("ij,j->i", verts - traj.positions[nearest], up)
    cand = facing & (dist <= radius_m) & (dz < -0.1)
    return estimate_floor(verts[cand], dz[cand], up, log=log)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"input": str(args.input)}

    verts, faces, colors = arrays_from_polydata(read_ply(args.input))
    if colors is None:
        colors = np.full((len(verts), 3), 180, dtype=np.uint8)
    log(f"loaded {len(verts):,} vertices, {len(faces):,} triangles from {args.input}")
    report["input_triangles"] = int(len(faces))

    if args.floor_model.is_file():
        floor = FloorModel.load(args.floor_model)
        log(f"floor model from {args.floor_model}")
        # the build saves the floor model in GLIM's frame but may shift its mesh
        # along z (output_z_offset_m in its report): follow the mesh
        build_report = args.input.parent / "finer_map_report.json"
        if build_report.is_file():
            z_offset = json.loads(build_report.read_text()).get("output_z_offset_m", 0.0)
            if z_offset:
                floor = floor.shifted_z(z_offset)
                log(f"floor model shifted by {z_offset:+.4f} m in z to match the build's output frame")
    else:
        log(f"no {args.floor_model}; estimating the floor from the mesh and {args.map_dir / 'traj_lidar.txt'}")
        floor = floor_from_mesh(verts, faces, args.map_dir)
        floor.save(args.out_dir / "floor_model.npz")
    report["floor_model"] = floor.summary()

    # set the synthetic floor slab aside: none of the steps below may touch it
    slab = None
    if not args.refine_floor_slab:
        is_slab = find_slab(verts, faces, floor)
        if is_slab.any():
            slab = keep_faces(verts, faces, colors, is_slab)
            verts, faces, colors = keep_faces(verts, faces, colors, ~is_slab)
            log(f"floor slab: set aside {int(is_slab.sum()):,} triangles, added back unchanged at the end")
            report["floor_slab_triangles"] = int(is_slab.sum())

    # 1. clip below the floor / above the ceiling
    if args.clip_margin > 0 or args.ceiling_height is not None:
        h = floor.height_above_floor(verts)
        bad = h < -args.clip_margin if args.clip_margin > 0 else np.zeros(len(verts), dtype=bool)
        if args.ceiling_height is not None:
            bad |= h > args.ceiling_height
        n_before = len(faces)
        verts, faces, colors = keep_faces(verts, faces, colors, ~bad[faces].any(axis=1))
        log(f"clip: removed {n_before - len(faces):,} triangles below the floor/above the ceiling")
        report["clipped_triangles"] = int(n_before - len(faces))

    # 2. small connected components
    if args.min_component > 0:
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
        graph = sparse.coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
                                  shape=(len(verts), len(verts)))
        n_comp, labels = connected_components(graph, directed=False)
        face_label = labels[faces[:, 0]]
        sizes = np.bincount(face_label, minlength=n_comp)
        keep = sizes[face_label] >= args.min_component
        n_small = int((sizes[sizes > 0] < args.min_component).sum())
        n_before = len(faces)
        verts, faces, colors = keep_faces(verts, faces, colors, keep)
        log(f"components: {n_comp:,} pieces, removed {n_small:,} with < {args.min_component} triangles "
            f"({n_before - len(faces):,} triangles)")
        report["removed_components"] = n_small
        report["removed_component_triangles"] = int(n_before - len(faces))

    poly = polydata_from_arrays(verts, faces, colors)

    # 3. windowed-sinc (Taubin-style, non-shrinking) smoothing
    if args.smooth_iterations > 0:
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetNumberOfIterations(args.smooth_iterations)
        smooth.SetPassBand(args.smooth_passband)
        smooth.NormalizeCoordinatesOn()
        smooth.BoundarySmoothingOff()
        smooth.FeatureEdgeSmoothingOff()
        smooth.NonManifoldSmoothingOn()
        poly = run_filter(smooth, poly)
        log(f"smoothing: {args.smooth_iterations} windowed-sinc iterations, pass band {args.smooth_passband}")

    # 4. flatten the floor (not when a synthetic slab replaces it: the input
    # then has no floor, only the bases of objects standing on it)
    if slab is not None and args.flatten_tolerance > 0:
        log("floor flattening: skipped, the synthetic floor slab replaces the floor")
    elif args.flatten_tolerance > 0:
        verts, faces, colors = arrays_from_polydata(poly)
        h = floor.height_above_floor(verts)
        normals = vertex_normals(verts, faces)
        on_floor = (np.abs(h) < args.flatten_tolerance) & \
                   (np.abs(normals @ floor.normal) >= np.cos(np.radians(args.flatten_max_angle)))
        verts[on_floor] = floor.project_to_floor(verts[on_floor])
        poly = polydata_from_arrays(verts, faces, colors)
        log(f"floor flattening: projected {int(on_floor.sum()):,} vertices onto the floor "
            f"(residual before: median {np.median(np.abs(h[on_floor])) * 100:.1f} cm)")
        report["flattened_vertices"] = int(on_floor.sum())

    # 5. fill small holes
    if args.fill_holes > 0:
        n_before = poly.GetNumberOfCells()
        fill = vtk.vtkFillHolesFilter()
        fill.SetHoleSize(args.fill_holes)
        poly = run_filter(vtk.vtkTriangleFilter(), run_filter(fill, poly))  # hole caps may be n-gons
        log(f"hole filling (<= {args.fill_holes} m): added {poly.GetNumberOfCells() - n_before:,} triangles")
        report["hole_fill_triangles"] = int(poly.GetNumberOfCells() - n_before)

    # 6. quadric decimation on geometry only; colours are transferred afterwards
    if args.decimate > 0:
        n_before = poly.GetNumberOfCells()
        before_verts, _, before_colors = arrays_from_polydata(poly)
        dec = vtk.vtkQuadricDecimation()
        dec.SetTargetReduction(args.decimate)
        dec.VolumePreservationOn()
        # VTK's attribute error metric (to carry colours through the collapses)
        # aborts the whole decimation early when one attribute matrix cannot be
        # factored ("Unable to factor attribute matrix!"), so it stays off
        dec.AttributeErrorMetricOff()
        poly = run_filter(dec, poly)
        verts, faces, _ = arrays_from_polydata(poly)
        dist, nearest = cKDTree(before_verts).query(verts, distance_upper_bound=1.0)
        colors = before_colors[np.minimum(nearest, len(before_verts) - 1)]
        # a quadric collapse whose 3x3 system is singular ("Unable to factor
        # linear system") can place its vertex metres away (seen: x = 92 m in
        # a 20 m room); drop triangles using any vertex that left the surface
        stray = dist > 0.10
        verts, faces, colors = keep_faces(verts, faces, colors, ~stray[faces].any(axis=1))
        poly = polydata_from_arrays(verts, faces, colors)
        log(f"decimation: {n_before:,} -> {poly.GetNumberOfCells():,} triangles "
            f"({int(stray.sum())} stray vertices dropped)")
        if poly.GetNumberOfCells() > n_before * (1.0 - args.decimate) * 1.05:
            log("  warning: decimation stopped short of --decimate")
        report["decimation_stray_vertices"] = int(stray.sum())

    # 7. add the synthetic floor slab back, untouched by the steps above
    if slab is not None:
        verts, faces, colors = arrays_from_polydata(poly)
        s_verts, s_faces, s_colors = slab
        poly = polydata_from_arrays(np.concatenate([verts, s_verts]),
                                    np.concatenate([faces, s_faces + len(verts)]),
                                    np.concatenate([colors, s_colors]))
        log(f"floor slab: added back {len(s_faces):,} triangles")

    final = finish_and_write(poly, args.out_dir, f"{args.name}_refined_mesh")
    report["output_vertices"] = int(final.GetNumberOfPoints())
    report["output_triangles"] = int(final.GetNumberOfCells())
    report["params"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(args.out_dir / "refine_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {args.out_dir / f'{args.name}_refined_mesh.ply'} "
        f"({report['output_vertices']:,} vertices, {report['output_triangles']:,} triangles)")


if __name__ == "__main__":
    main()
