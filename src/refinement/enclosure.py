"""Enclosure: four walls and a roof around the map, shaped like its floor.

1. **Floor footprint.** The synthetic floor is rasterised on a `cell` grid
   and morphologically opened with a disk of radius `core_radius`, keeping
   the largest region: every part narrower than 2 * `core_radius` (door
   spikes, corridors leaving the room) disappears, while straight stretches
   of wall are untouched. The footprint's boundary cells are the margin.
2. **Quadrilateral fit.** Starting from the minimum-area rectangle around
   the footprint, iteratively reweighted least squares: each margin point is
   assigned to its nearest side, each side is refitted as a total-least-
   squares line with Cauchy weights 1 / (1 + (r/scale)^2), and adjacent
   lines are intersected into the new corners. The Cauchy weights keep what
   the opening leaves of notches and rounded corners from dragging the walls.
3. **Walls and roof.** Walls on the four sides, from `wall_sink` below the
   floor model (so no gap shows at the floor's edge) up to a roof on the
   plane parallel to the floor's global plane through the mesh's highest
   vertex. No bottom: the synthetic floor is the floor. Tessellated on a
   `resolution` lattice with shared seam vertices, one-sided, every normal
   pointing into the room (invisible from outside with back-face culling).
4. **Colour.** Wall vertices take the RGB of the nearest vertex of the
   mesh; the roof is a single colour, the per-channel median of the mesh.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

from .base import PART_ENCLOSURE, PART_FLOOR, RefinementStep


def floor_footprint(xy, cell, radius):
    """Opened floor footprint of floor vertices `xy`: (K,2) centres of its
    boundary cells and (M,2) centres of all its cells."""
    pad = int(np.ceil(radius / cell)) + 2
    origin = xy.min(axis=0) - pad * cell
    ij = np.floor((xy - origin) / cell).astype(np.int64)
    occ = np.zeros(ij.max(axis=0) + pad + 1, dtype=bool)
    occ[ij[:, 0], ij[:, 1]] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ))
    r = int(round(radius / cell))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    core = ndimage.binary_opening(occ, structure=xx ** 2 + yy ** 2 <= r ** 2)
    labels, n = ndimage.label(core)
    if n == 0:
        raise RuntimeError(f"enclosure: no part of the floor is wider than {2 * radius} m")
    core = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    edge = core & ~ndimage.binary_erosion(core)
    return origin + (np.argwhere(edge) + 0.5) * cell, origin + (np.argwhere(core) + 0.5) * cell


def min_area_rectangle(points):
    """Counter-clockwise corners (4,2) of the minimum-area rectangle around
    `points` (rotating calipers: one side is collinear with a hull edge)."""
    hull = points[ConvexHull(points).vertices]
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    best = None
    for a in np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2)):
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])  # world -> rotated
        p = hull @ R.T
        lo, hi = p.min(axis=0), p.max(axis=0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) @ R)
    return best[1]


def segment_distances(points, corners):
    """(N,4) distance from each point to each side (corner k -> k+1)."""
    a, b = corners, np.roll(corners, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("nkd,kd->nk", points[:, None] - a[None], ab) / np.einsum("kd,kd->k", ab, ab), 0, 1)
    return np.linalg.norm(points[:, None] - (a[None] + t[..., None] * ab[None]), axis=2)


def fit_quadrilateral(points, corners, scale, iterations=100):
    """Robust (Cauchy IRLS) quadrilateral fit of `points`, starting from `corners`."""
    for _ in range(iterations):
        side = segment_distances(points, corners).argmin(axis=1)
        normals, offsets = np.empty((4, 2)), np.empty(4)
        for k in range(4):
            d = corners[(k + 1) % 4] - corners[k]
            normals[k] = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            offsets[k] = normals[k] @ corners[k]
            p = points[side == k]
            if len(p) < 2:
                continue  # no point nearest to this side: keep it
            w = 1.0 / (1.0 + ((p @ normals[k] - offsets[k]) / scale) ** 2)
            c = (w[:, None] * p).sum(axis=0) / w.sum()
            q = p - c
            n = np.linalg.eigh((w[:, None, None] * q[:, :, None] * q[:, None, :]).sum(axis=0))[1][:, 0]
            normals[k] = n if n @ normals[k] > 0 else -n
            offsets[k] = normals[k] @ c
        # corner k is where side k-1 meets side k
        new = np.array([np.linalg.solve(np.stack([normals[k - 1], normals[k]]), [offsets[k - 1], offsets[k]])
                        for k in range(4)])
        moved = np.abs(new - corners).max()
        corners = new
        if moved < 1e-6:
            break
    return corners


def _grid_faces(ni, nj):
    """Two triangles per cell of an (ni+1, nj+1) row-major vertex grid, facing +i x +j."""
    i, j = (g.ravel() for g in np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij"))
    a, b, c, d = i * (nj + 1) + j, (i + 1) * (nj + 1) + j, (i + 1) * (nj + 1) + j + 1, i * (nj + 1) + j + 1
    return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def quad_lattice(corners, resolution):
    """(nu+1, nv+1, 2) bilinear lattice over a quadrilateral, u along corner
    0 -> 1 and v along 0 -> 3, cells at most ~`resolution` metres."""
    c0, c1, c2, c3 = corners
    nu = max(int(np.ceil(max(np.linalg.norm(c1 - c0), np.linalg.norm(c2 - c3)) / resolution)), 1)
    nv = max(int(np.ceil(max(np.linalg.norm(c3 - c0), np.linalg.norm(c2 - c1)) / resolution)), 1)
    u = np.linspace(0.0, 1.0, nu + 1)[:, None, None]
    v = np.linspace(0.0, 1.0, nv + 1)[None, :, None]
    return (1 - u) * (1 - v) * c0 + u * (1 - v) * c1 + u * v * c2 + (1 - u) * v * c3


def lattice_faces(lattice):
    """Faces of a quad lattice, facing up for a counter-clockwise quadrilateral."""
    return _grid_faces(lattice.shape[0] - 1, lattice.shape[1] - 1)


def build_walls(corners, bottom_z, top_z, resolution):
    """Walls and roof over a counter-clockwise quadrilateral, all normals
    pointing inwards. Returns (verts, faces, is_roof vertex mask)."""
    lattice = quad_lattice(corners, resolution)
    flat = lattice.reshape(-1, 2)
    roof = np.column_stack([flat, top_z(flat)])
    roof_faces = lattice_faces(lattice)[:, ::-1]  # (u, v) is counter-clockwise: reversed, faces point down

    # the lattice's perimeter, counter-clockwise and closed (first point repeated)
    loop = np.concatenate([lattice[:, 0], lattice[-1, 1:], lattice[-2::-1, -1], lattice[0, -2::-1]])
    zb, zt = bottom_z(loop), top_z(loop)
    m = max(int(np.ceil((zt - zb).max() / resolution)), 1)
    t = np.linspace(0.0, 1.0, m + 1)
    wall = np.empty((len(loop), m + 1, 3))
    wall[:, :, :2] = loop[:, None]
    wall[:, :, 2] = zb[:, None] + t[None] * (zt - zb)[:, None]
    # i runs counter-clockwise along the perimeter and j up, so +i x +j points out: reversed
    wall_faces = _grid_faces(len(loop) - 1, m)[:, ::-1] + len(roof)

    verts = np.concatenate([roof, wall.reshape(-1, 3)])
    is_roof = np.concatenate([np.ones(len(roof), dtype=bool), np.tile(t == 1.0, len(loop))])
    return verts, np.concatenate([roof_faces, wall_faces]), is_roof


class EnclosureStep(RefinementStep):
    name = "enclosure"
    help = "walls on a robust quadrilateral fit of the opened floor footprint, roof at the highest vertex"

    @classmethod
    def add_args(cls, p):
        g = p.add_argument_group("enclosure step")
        g.add_argument("--enclosure-core-radius", type=float, default=1.0,
                       help="opening radius, metres: floor parts narrower than twice this (door spikes, "
                            "corridors) are removed from the footprint before fitting")
        g.add_argument("--enclosure-cell", type=float, default=0.05,
                       help="raster cell, metres, of the floor footprint")
        g.add_argument("--enclosure-scale", type=float, default=0.10,
                       help="Cauchy loss scale, metres: margin much further than this from a wall barely "
                            "influences it")
        g.add_argument("--enclosure-resolution", type=float, default=0.05,
                       help="edge length, metres, of the walls' tessellation (sets their colour detail)")
        g.add_argument("--enclosure-wall-sink", type=float, default=0.02,
                       help="walls start this far below the floor model, metres, so no gap shows")
        g.add_argument("--enclosure-plot", action="store_true",
                       help="also save enclosure_fit.png (footprint, margin and fitted quadrilateral)")

    def run(self, mesh, ctx):
        a, floor = self.args, ctx.floor
        floor_faces = mesh.faces[mesh.parts == PART_FLOOR]
        if len(floor_faces) == 0:
            raise RuntimeError("enclosure: the input mesh has no synthetic floor")
        margin, core = floor_footprint(mesh.verts[np.unique(floor_faces), :2], a.enclosure_cell,
                                       a.enclosure_core_radius)
        init = min_area_rectangle(core)
        corners = fit_quadrilateral(margin, init, a.enclosure_scale)
        x, y = corners[:, 0], corners[:, 1]
        if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
            corners = corners[::-1].copy()

        d = segment_distances(margin, corners).min(axis=1)
        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        prev, nxt = np.roll(corners, 1, axis=0) - corners, np.roll(corners, -1, axis=0) - corners
        angles = np.degrees(np.arccos(np.einsum("kd,kd->k", prev, nxt)
                                      / (np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1))))
        ctx.log(f"enclosure: corners {corners.round(3).tolist()}, sides {sides.round(2).tolist()} m, "
                f"angles {angles.round(2).tolist()} deg, margin median distance {np.median(d) * 100:.1f} cm")

        roof_h = float((mesh.verts @ floor.normal + floor.offset).max())
        verts, faces, is_roof = build_walls(corners, lambda xy: floor.floor_z(xy) - a.enclosure_wall_sink,
                                            lambda xy: floor.plane_z(xy, roof_h), a.enclosure_resolution)
        roof_color = np.median(mesh.colors, axis=0).astype(np.uint8)
        colors = np.empty((len(verts), 3), dtype=np.uint8)
        colors[is_roof] = roof_color
        _, nearest = cKDTree(mesh.verts).query(verts[~is_roof], k=1, workers=-1)
        colors[~is_roof] = mesh.colors[nearest]
        mesh.append(verts, faces, colors, PART_ENCLOSURE)
        ctx.state["enclosure"] = {"corners": corners, "roof_height": roof_h, "roof_color": roof_color,
                                  "wall_sink": a.enclosure_wall_sink}
        ctx.log(f"enclosure: roof {roof_h:.3f} m above the floor plane, colour {roof_color.tolist()}; "
                f"{len(verts):,} vertices, {len(faces):,} triangles")

        if a.enclosure_plot:
            _plot(ctx.out_dir / "enclosure_fit.png", core, margin, init, corners)
        return {
            "corners_xy": corners.round(4).tolist(),
            "side_lengths_m": sides.round(4).tolist(),
            "interior_angles_deg": angles.round(3).tolist(),
            "initial_rectangle": init.round(4).tolist(),
            "margin_median_distance_m": round(float(np.median(d)), 4),
            "roof_height_above_floor_plane_m": round(roof_h, 4),
            "roof_color_rgb": roof_color.tolist(),
            "vertices": int(len(verts)),
            "triangles": int(len(faces)),
        }


def _plot(path, core, margin, init, corners):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(core[:, 0], core[:, 1], s=0.2, c="0.85", label="opened floor footprint")
    ax.scatter(margin[:, 0], margin[:, 1], s=1, c="0.3", label="footprint margin")
    for c, style, label in [(init, "--", "initial rectangle"), (corners, "-", "fitted quadrilateral")]:
        loop = np.vstack([c, c[:1]])
        ax.plot(loop[:, 0], loop[:, 1], style, lw=1.5, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
