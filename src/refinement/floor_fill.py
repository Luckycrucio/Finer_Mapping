"""Floor: complete the synthetic floor up to the enclosure's quadrilateral.

The build's floor stops wherever the floor was never seen (behind furniture
against a wall, unseen patches at the edge), so the background shows through
between it and the walls. This step keeps every existing floor triangle and
only adds the missing cells:

  - The build's floor lies on a regular lattice (cell corners every voxel
    size, see src.floor_slab.build_slab); the lattice is recovered from the
    floor's own vertices.
  - Cells already covered by a floor triangle are left alone. Every other
    cell overlapping the quadrilateral (centre inside it grown by half a
    cell diagonal, so the floor reaches under the walls) gets two
    triangles, on the floor model's surface like the rest of the floor.
  - New cells reuse the existing floor's vertices wherever they share a
    corner, so the completed floor is one seamless surface, not a second
    layer; new vertices take the colour of the nearest existing floor
    vertex.
"""
import numpy as np
from scipy.spatial import cKDTree

from .base import PART_FLOOR, RefinementStep
from .geometry import inside_quad


class FloorStep(RefinementStep):
    name = "floor"
    help = "complete the floor's missing cells up to the enclosure (existing floor kept)"

    def run(self, mesh, ctx):
        corners = ctx.require("enclosure", self.name)["corners"]
        cell = float(ctx.build_report.get("voxel_size_m", 0.03))
        is_floor = mesh.parts == PART_FLOOR
        if not np.any(is_floor):
            raise RuntimeError("floor: the input mesh has no synthetic floor to complete")
        lv, lf, lc, used = mesh.submesh(is_floor)

        # the floor's lattice: corner (i, j) at origin + (i, j) * cell
        origin = lv[0, :2]
        ij = np.round((lv[:, :2] - origin) / cell).astype(np.int64)
        off_lattice = np.abs(lv[:, :2] - (origin + ij * cell)).max()
        if off_lattice > 1e-4:
            raise RuntimeError(f"floor: floor vertices are not on a {cell} m lattice (off by {off_lattice:.2g} m)")

        # cells already floored: a lattice square's two triangles both have its (i, j) as their minimum corner
        lo = ij.min(axis=0) - 1
        span = ij.max(axis=0) - lo + 2
        occupied = np.zeros(span, dtype=bool)
        cells = ij[lf].min(axis=1) - lo
        occupied[cells[:, 0], cells[:, 1]] = True

        # cells overlapping the quadrilateral
        qlo = np.floor((corners.min(axis=0) - origin) / cell).astype(np.int64) - 1
        qhi = np.ceil((corners.max(axis=0) - origin) / cell).astype(np.int64) + 1
        gi, gj = np.meshgrid(np.arange(qlo[0], qhi[0]), np.arange(qlo[1], qhi[1]), indexing="ij")
        gi, gj = gi.ravel(), gj.ravel()
        centre = origin + (np.column_stack([gi, gj]) + 0.5) * cell
        want = inside_quad(centre, corners, margin=cell * np.sqrt(0.5))
        ri, rj = gi - lo[0], gj - lo[1]
        inside_span = (ri >= 0) & (rj >= 0) & (ri < span[0]) & (rj < span[1])
        covered = np.zeros(len(gi), dtype=bool)
        covered[inside_span] = occupied[ri[inside_span], rj[inside_span]]
        fill_i, fill_j = gi[want & ~covered], gj[want & ~covered]

        # corner vertices: the existing floor's where present, new ones otherwise
        corner_ij = np.concatenate([np.column_stack([fill_i + di, fill_j + dj])
                                    for di, dj in [(0, 0), (1, 0), (1, 1), (0, 1)]])
        # one int64 key per lattice corner (indices shifted to be non-negative first)
        base = np.minimum(ij.min(axis=0), corner_ij.min(axis=0))
        key = lambda a: (a[:, 0] - base[0]) * (1 << 32) + (a[:, 1] - base[1])  # noqa: E731
        uniq, inverse = np.unique(key(corner_ij), return_inverse=True)
        existing_keys = key(ij)
        order = np.argsort(existing_keys)
        pos = np.clip(np.searchsorted(existing_keys, uniq, sorter=order), 0, len(order) - 1)
        found = existing_keys[order[pos]] == uniq
        index = np.empty(len(uniq), dtype=np.int64)
        index[found] = used[order[pos[found]]]

        new_ij = np.column_stack([uniq[~found] >> 32, uniq[~found] & 0xFFFFFFFF]) + base
        new_xy = origin + new_ij * cell
        new_v = np.column_stack([new_xy, ctx.floor.floor_z(new_xy)])
        _, nearest = cKDTree(lv[:, :2]).query(new_xy, k=1, workers=-1)
        index[~found] = len(mesh.verts) + np.arange(len(new_v))
        mesh.append(new_v, np.zeros((0, 3), dtype=np.int64), lc[nearest], PART_FLOOR)

        a, b, c, d = index[inverse].reshape(4, -1)
        mesh.add_faces(np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]), PART_FLOOR)
        area = len(fill_i) * cell ** 2
        ctx.log(f"floor: added {len(fill_i):,} missing {cell} m cells ({area:.1f} m2, {2 * len(fill_i):,} triangles, "
                f"{len(new_v):,} new vertices) to complete the floor up to the walls")
        return {"cell_m": cell, "cells_added": int(len(fill_i)), "area_added_m2": round(area, 2),
                "triangles_added": int(2 * len(fill_i)), "vertices_added": int(len(new_v))}
