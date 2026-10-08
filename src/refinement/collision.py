"""Collision: a very coarse mesh for the physics engine (binary STL).

Physics cost grows with triangle count and gains nothing from visual
detail, so the collision mesh is rebuilt from scratch:
  - fused geometry without the pieces a robot cannot touch (entirely
    above `max_height` over the floor: ceiling fragments, lamps) or too
    small to matter (fewer than `min_piece` triangles), decimated to
    `triangles` triangles in total (quadric edge collapse; open borders may
    move, nothing here is seen). Dropping the small pieces first is what
    lets the budget be met: every separate piece costs a few triangles
    however hard it is decimated, and the ceiling alone has thousands,
  - the (clipped) floor decimated to `floor_triangles` triangles,
  - each wall as a single rectangle (2 triangles) from just below the
    lowest point of the floor along it up to the roof,
  - the roof as 2 triangles.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .base import PART_FLOOR, PART_FUSED, RefinementStep
from .geometry import compact, decimate, face_normals
from .texture import model_dir


def write_stl(path, verts, faces):
    n = face_normals(verts, faces)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, verts[faces]
    with open(path, "wb") as f:
        f.write(b"finer_mapping collision mesh".ljust(80, b" "))
        f.write(np.uint32(len(faces)).tobytes())
        f.write(rec.tobytes())


class CollisionStep(RefinementStep):
    name = "collision"
    help = "write a very coarse collision mesh (STL)"

    @classmethod
    def add_args(cls, p):
        g = p.add_argument_group("collision step")
        g.add_argument("--collision-triangles", type=int, default=20000,
                       help="triangle budget for the fused geometry")
        g.add_argument("--collision-max-height", type=float, default=2.5,
                       help="drop fused pieces entirely higher than this above the floor, metres")
        g.add_argument("--collision-min-piece", type=int, default=200,
                       help="drop fused pieces with fewer triangles than this")
        g.add_argument("--collision-floor-triangles", type=int, default=2000,
                       help="triangle budget for the floor")

    def run(self, mesh, ctx):
        a, floor = self.args, ctx.floor
        enc = ctx.require("enclosure", self.name)
        corners, roof_h, sink = enc["corners"], enc["roof_height"], enc["wall_sink"]
        pieces = []

        fv, ff, _, _ = mesh.submesh(mesh.parts == PART_FUSED)
        edges = np.concatenate([ff[:, [0, 1]], ff[:, [1, 2]]])
        n_comp, labels = connected_components(sparse.coo_matrix(
            (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(len(fv), len(fv))),
            directed=False)
        face_comp = labels[ff[:, 0]]
        lowest = np.full(n_comp, np.inf)
        np.minimum.at(lowest, labels, ctx.floor.height_above_floor(fv))
        keep_comp = (np.bincount(face_comp, minlength=n_comp) >= a.collision_min_piece) & \
            (lowest <= a.collision_max_height)
        kv, kf, _ = compact(fv, ff[keep_comp[face_comp]])
        ctx.log(f"collision: {int(keep_comp.sum()):,} of {len(np.unique(face_comp)):,} fused pieces kept "
                f"({len(kf):,} triangles) before decimation")
        pieces.append(decimate(kv, kf, target=a.collision_triangles, preserve_border=False))

        lv, lf, _, _ = mesh.submesh(mesh.parts == PART_FLOOR)
        pieces.append(decimate(lv, lf, target=a.collision_floor_triangles, preserve_border=False))

        for k in range(4):
            ab = np.stack([corners[k], corners[(k + 1) % 4]])
            along = ab[0] + np.linspace(0, 1, 50)[:, None] * (ab[1] - ab[0])
            zb = float(floor.floor_z(along).min()) - sink
            zt = floor.plane_z(ab, roof_h)
            v = np.array([[*ab[0], zb], [*ab[1], zb], [*ab[1], zt[1]], [*ab[0], zt[0]]])
            pieces.append((v, np.array([[0, 2, 1], [0, 3, 2]])))  # facing into the room
        pieces.append((np.column_stack([corners, floor.plane_z(corners, roof_h)]), np.array([[0, 2, 1], [0, 3, 2]])))

        verts, faces, base = [], [], 0
        for v, f in pieces:
            verts.append(v)
            faces.append(f + base)
            base += len(v)
        verts, faces = np.concatenate(verts), np.concatenate(faces)
        out = model_dir(ctx) / "meshes"
        out.mkdir(parents=True, exist_ok=True)
        write_stl(out / "collision.stl", verts, faces)
        ctx.state["collision_mesh"] = out / "collision.stl"
        ctx.log(f"collision: wrote {out / 'collision.stl'} ({len(faces):,} triangles, "
                f"{len(pieces[0][1]):,} of them fused geometry)")
        return {"triangles": int(len(faces)), "fused_triangles": int(len(pieces[0][1])),
                "file": str(out / "collision.stl")}
