#!/usr/bin/env python3
"""Map refinement: post-process a finished finer_mapping build for simulation.

Usage: map_refinement.py <bag_name>  (e.g. coverage1)

Reads outputs/<name>/<name>_finer_mesh.ply, plus the floor model
(floor_model.npz) and report (finer_map_report.json) the build saved next to
it, runs the refinement steps in order and writes
outputs/<name>/<name>_refined_map.ply (the refined, vertex-coloured mesh),
map_refinement_report.json and, through the export steps, a Gazebo model
under outputs/<name>/gazebo/. Nothing here needs the bag or the TSDF; the
whole chain takes about a minute.

Before the first step, the build's synthetic floor is identified in the
input mesh (the connected piece lying exactly on the floor model) and its
triangles labelled as such, so steps can tell it apart from the fused
geometry. Steps (see src/refinement/; --steps picks a subset/order):

  cleanup    Remove broken triangles, all fused geometry close to the
             (detected) ceiling, then small disconnected pieces.
  enclosure  One-sided, inward-facing walls and roof: walls on a robust
             (Cauchy) quadrilateral fit of the floor footprint after a
             morphological opening (removes door spikes and corridors),
             roof at the mesh's highest vertex. Walls take the nearest mesh
             vertex's colour, the roof the median colour of the mesh.
  floor      Complete the floor's missing cells up to the enclosure's
             quadrilateral, keeping the existing floor.
  clip       Remove geometry outside the enclosure.
  texture    Export: decimated, UV-unwrapped, texture-baked visual mesh
             (OBJ + MTL + PNG).
  collision  Export: very coarse collision mesh (STL).
  gazebo     Export: model.sdf/model.config around the two meshes + a world.

Adding a step: subclass src.refinement.RefinementStep in a new module of
src/refinement/ and append it to STEPS in src/refinement/__init__.py; its
options and report entry are picked up here automatically.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.floor import FloorModel  # noqa: E402
from src.floor_slab import find_slab  # noqa: E402
from src.mesh_io import arrays_from_polydata, finish_and_write, polydata_from_arrays, read_ply  # noqa: E402
from src.refinement import STEPS, MapMesh, RefinementContext  # noqa: E402
from src.refinement.base import PART_FLOOR, PART_FUSED  # noqa: E402

STEP_BY_NAME = {s.name: s for s in STEPS}


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="bag name used for the build, e.g. coverage1")
    p.add_argument("--input", default=None, help="mesh to refine (default: outputs/<name>/<name>_finer_mesh.ply)")
    p.add_argument("--floor-model", default=None, help="default: floor_model.npz next to the input mesh")
    p.add_argument("--build-report", default=None, help="default: finer_map_report.json next to the input mesh")
    p.add_argument("--out-dir", default=None, help="default: the input mesh's directory")
    p.add_argument("--steps", default=",".join(STEP_BY_NAME),
                   help=f"comma-separated steps to run, in order (available: {', '.join(STEP_BY_NAME)}; "
                        "default: %(default)s); empty runs none")
    p.add_argument("--formats", default="ply",
                   help="comma-separated output formats among ply, obj, stl (only PLY keeps vertex colour)")
    for step in STEPS:
        step.add_args(p)
    args = p.parse_args()

    args.input = Path(args.input or here / "outputs" / args.name / f"{args.name}_finer_mesh.ply")
    args.floor_model = Path(args.floor_model or args.input.parent / "floor_model.npz")
    args.build_report = Path(args.build_report or args.input.parent / "finer_map_report.json")
    args.out_dir = Path(args.out_dir or args.input.parent)
    args.steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in args.steps if s not in STEP_BY_NAME]
    if unknown:
        p.error(f"unknown step(s) {unknown}; available: {list(STEP_BY_NAME)}")
    args.formats = tuple(f for f in args.formats.split(",") if f)
    return args


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    args = parse_args()
    missing = [str(f) for f in (args.input, args.floor_model, args.build_report) if not f.exists()]
    if missing:
        sys.exit("missing input(s):\n  " + "\n  ".join(missing))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(args.build_report) as f:
        build_report = json.load(f)
    # the build saves the floor model in GLIM's frame and shifts its mesh in z
    z_offset = float(build_report.get("output_z_offset_m") or 0.0)
    floor = FloorModel.load(args.floor_model).shifted_z(z_offset)

    log(f"reading {args.input}")
    verts, faces, colors = arrays_from_polydata(read_ply(args.input))
    if colors is None:
        colors = np.full((len(verts), 3), 180, dtype=np.uint8)
    parts = np.where(find_slab(verts, faces, floor), PART_FLOOR, PART_FUSED)
    mesh = MapMesh(verts, faces, colors, parts)
    log(f"input: {mesh.summary()} (floor model shifted by {z_offset:+.4f} m into the mesh frame)")

    ctx = RefinementContext(args.name, floor, build_report, args.out_dir, log)
    step_reports = {}
    for name in args.steps:
        t = time.time()
        log(f"--- step: {name}")
        step_reports[name] = STEP_BY_NAME[name](args).run(mesh, ctx)
        step_reports[name]["seconds"] = round(time.time() - t, 2)

    stem = f"{args.name}_refined_map"
    # keep every triangle's winding: the enclosure is one-sided and must face inwards
    final = finish_and_write(polydata_from_arrays(mesh.verts, mesh.faces, mesh.colors), args.out_dir, stem,
                             formats=args.formats, orient=False)
    log(f"wrote {', '.join(str(args.out_dir / f'{stem}.{e}') for e in args.formats)} "
        f"({final.GetNumberOfPoints():,} vertices, {final.GetNumberOfCells():,} triangles)")

    report = {
        "name": args.name,
        "input": str(args.input),
        "floor_model": str(args.floor_model),
        "build_report": str(args.build_report),
        "steps": args.steps,
        "input_mesh": MapMesh(verts, faces, colors, parts).summary(),
        "output_mesh": {**mesh.summary(), "vertices_after_merge": int(final.GetNumberOfPoints())},
        **step_reports,
        "seconds": round(time.time() - t0, 2),
    }
    with open(args.out_dir / "map_refinement_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log(f"report -> {args.out_dir / 'map_refinement_report.json'}")


if __name__ == "__main__":
    main()
