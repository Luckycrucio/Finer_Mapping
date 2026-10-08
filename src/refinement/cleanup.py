"""Cleanup: remove broken triangles, small disconnected pieces, and
everything close to the ceiling.

  - Broken triangles (a repeated vertex, zero area, or the same three
    vertices as another triangle) are removed everywhere; they upset some
    loaders and physics engines.
  - Ceiling: every fused triangle with a vertex less than `ceiling_margin`
    below the ceiling is removed (ceiling, beams, lamps: seen sparsely at
    grazing angles, they come out as ragged patches); the enclosure's roof
    then closes the map just above what is left. The ceiling's height is
    detected as the lowest strong layer of horizontal surfaces in the upper
    half of the map: an area-weighted histogram (5 cm bins) of the fused
    triangles facing up or down, whose first bin with at least
    `ceiling_peak` of the highest bin's area is the ceiling's underside.
    Walls contribute little horizontal area, so they do not trigger it.
    `ceiling_height` overrides the detection.
  - Small pieces, after the ceiling cut (so pieces it leaves dangling go
    too): connected components of the fused mesh with fewer than
    `min_triangles` triangles (floating noise, surfaces glimpsed through
    glass, carving leftovers) are removed. The synthetic floor is never
    touched.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .base import PART_FUSED, RefinementStep
from .geometry import drop_degenerate, face_normals


def detect_ceiling(verts, faces, floor, peak=0.5, bin_m=0.05):
    """Height above the floor of the ceiling's underside (see module docstring)."""
    n = face_normals(verts, faces)
    area = np.linalg.norm(n, axis=1) / 2
    horizontal = np.abs(n @ floor.normal) > 0.9 * 2 * np.maximum(area, 1e-30)
    h = floor.height_above_floor(verts[faces].mean(axis=1))
    top = float(h.max())
    edges = np.arange(top / 2, top + bin_m, bin_m)
    hist, _ = np.histogram(h[horizontal], edges, weights=area[horizontal])
    if hist.max() <= 0:
        return None
    return float(edges[np.argmax(hist >= peak * hist.max())])


class CleanupStep(RefinementStep):
    name = "cleanup"
    help = "drop broken triangles, small disconnected pieces and everything close to the ceiling"

    @classmethod
    def add_args(cls, p):
        g = p.add_argument_group("cleanup step")
        g.add_argument("--cleanup-min-triangles", type=int, default=200,
                       help="fused components with fewer triangles than this are removed")
        g.add_argument("--cleanup-ceiling-margin", type=float, default=0.1,
                       help="remove fused geometry from this far below the ceiling upwards, metres")
        g.add_argument("--cleanup-ceiling-peak", type=float, default=0.5,
                       help="ceiling detection: first horizontal-area bin reaching this fraction of the "
                            "largest one in the upper half of the map")
        g.add_argument("--cleanup-ceiling-height", type=float, default=None,
                       help="ceiling height above the floor, metres (default: detected); negative disables "
                            "the ceiling removal")

    def run(self, mesh, ctx):
        a, floor = self.args, ctx.floor
        ok = drop_degenerate(mesh.verts, mesh.faces)
        n_broken = int((~ok).sum())
        mesh.keep_faces(ok)

        # ceiling first, so small pieces the cut leaves dangling go with the rest
        fused = mesh.parts == PART_FUSED
        ceiling = a.cleanup_ceiling_height
        detected = ceiling is None
        if detected:
            ceiling = detect_ceiling(mesh.verts, mesh.faces[fused], floor, a.cleanup_ceiling_peak)
        n_ceiling_tri, cut = 0, None
        if ceiling is not None and ceiling >= 0:
            cut = ceiling - a.cleanup_ceiling_margin
            high = floor.height_above_floor(mesh.verts) > cut
            drop = fused & high[mesh.faces].any(axis=1)
            n_ceiling_tri = int(drop.sum())
            mesh.keep_faces(~drop)

        # small pieces
        fused = mesh.parts == PART_FUSED
        faces = mesh.faces[fused]
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
        n_v = len(mesh.verts)
        n_comp, labels = connected_components(sparse.coo_matrix(
            (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n_v, n_v)), directed=False)
        face_comp = labels[faces[:, 0]]
        n_tri = np.bincount(face_comp, minlength=n_comp)
        small = (n_tri > 0) & (n_tri < a.cleanup_min_triangles)
        drop = np.zeros(len(mesh.faces), dtype=bool)
        drop[np.flatnonzero(fused)] = small[face_comp]
        n_small_tri = int(drop.sum())
        mesh.keep_faces(~drop)

        ctx.log(f"cleanup: removed {n_broken:,} broken triangles, "
                + (f"{n_ceiling_tri:,} triangles above {cut:.2f} m (ceiling "
                   f"{'detected' if detected else 'given'} at {ceiling:.2f} m), " if cut is not None else "")
                + f"then {int(small.sum()):,} pieces with < {a.cleanup_min_triangles} triangles "
                  f"({n_small_tri:,} triangles)")
        return {
            "broken_triangles_removed": n_broken,
            "min_triangles": a.cleanup_min_triangles,
            "small_pieces_removed": int(small.sum()),
            "small_piece_triangles_removed": n_small_tri,
            "ceiling_height_m": None if ceiling is None else round(ceiling, 3),
            "ceiling_detected": detected,
            "ceiling_cut_height_m": None if cut is None else round(cut, 3),
            "ceiling_triangles_removed": n_ceiling_tri,
        }
