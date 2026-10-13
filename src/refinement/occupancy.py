"""Occupancy: a 2-D occupancy grid for Nav2 (map_server .pgm + .yaml).

Every triangle except the floor's (the fused geometry and the enclosure's
walls and roof) is sampled densely enough that no cell is skipped: its
vertices plus a barycentric grid of points at most half a cell apart. A
`resolution` cell is
  - occupied if any of those points lies between `min_height` and
    `max_height` above the local floor (heights from the floor model, not
    from z: the map frame is tilted, so z alone would drift by decimetres
    across the room),
  - free if the floor covers it and nothing occupies it,
  - unknown otherwise (outside the walls, nothing seen).
The floor is never an obstacle; it only tells free space from unknown.

Written as outputs/<name>/nav2/<name>_map.pgm + <name>_map.yaml in the
map_server format (trinary: occupied 0, free 254, unknown 205; the image's
top row is the largest y, the yaml's origin is the bottom-left corner of the
bottom-left cell), in the same x-y frame as the meshes. Obstacle inflation
is left to Nav2's costmap.
"""
import numpy as np

from .base import PART_FLOOR, RefinementStep

OCCUPIED, FREE, UNKNOWN = 0, 254, 205


def sample_triangles(verts, faces, spacing):
    """Points covering every triangle with gaps of at most ~`spacing`: the
    vertices, plus barycentric grids on triangles longer than `spacing`."""
    pts = [verts[np.unique(faces)]]
    tri = verts[faces]
    longest = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    n = np.ceil(longest / spacing).astype(np.int64)
    for k in np.unique(n[n > 1]):
        t = tri[n == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        w = np.stack([i[keep], j[keep], k - i[keep] - j[keep]], axis=1) / k  # (S, 3)
        pts.append(np.einsum("sk,tkd->tsd", w, t).reshape(-1, 3))
    return np.concatenate(pts)


class OccupancyStep(RefinementStep):
    name = "occupancy"
    help = "write a Nav2 occupancy grid (.pgm + .yaml) of everything but the floor in a height band"

    @classmethod
    def add_args(cls, p):
        g = p.add_argument_group("occupancy step")
        g.add_argument("--occupancy-resolution", type=float, default=0.05, help="cell size, metres")
        g.add_argument("--occupancy-min-height", type=float, default=0.0,
                       help="obstacles start this high above the floor, metres")
        g.add_argument("--occupancy-max-height", type=float, default=1.5,
                       help="obstacles end this high above the floor, metres")

    def run(self, mesh, ctx):
        a, floor = self.args, ctx.floor
        res = a.occupancy_resolution
        is_floor = mesh.parts == PART_FLOOR
        obstacle_pts = sample_triangles(mesh.verts, mesh.faces[~is_floor], res / 2)
        h = floor.height_above_floor(obstacle_pts)
        obstacle_pts = obstacle_pts[(h >= a.occupancy_min_height) & (h <= a.occupancy_max_height)]
        floor_pts = sample_triangles(mesh.verts, mesh.faces[is_floor], res / 2)

        xy = np.concatenate([obstacle_pts[:, :2], floor_pts[:, :2]])
        origin = np.floor(xy.min(axis=0) / res) * res - res  # one cell of margin, snapped to the grid
        width, height = (np.ceil((xy.max(axis=0) - origin) / res).astype(int) + 1)
        grid = np.full((height, width), UNKNOWN, dtype=np.uint8)  # row 0 = smallest y for now

        def cells(p):
            ij = np.floor((p[:, :2] - origin) / res).astype(np.int64)
            return ij[:, 1], ij[:, 0]  # row (y), column (x)

        grid[cells(floor_pts)] = FREE
        grid[cells(obstacle_pts)] = OCCUPIED
        image = grid[::-1]  # map_server: the top row is the largest y

        out = ctx.out_dir / "nav2"
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{ctx.name}_map"
        with open(out / f"{stem}.pgm", "wb") as f:
            f.write(f"P5\n# finer_mapping occupancy grid, {res} m/cell\n{width} {height}\n255\n".encode())
            f.write(image.tobytes())
        (out / f"{stem}.yaml").write_text(
            f"image: {stem}.pgm\n"
            f"mode: trinary\n"
            f"resolution: {res}\n"
            f"origin: [{origin[0]:.4f}, {origin[1]:.4f}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.25\n")

        n_occ, n_free = int((grid == OCCUPIED).sum()), int((grid == FREE).sum())
        ctx.log(f"occupancy: {width}x{height} cells of {res} m, {n_occ:,} occupied, {n_free:,} free, "
                f"{grid.size - n_occ - n_free:,} unknown (obstacles {a.occupancy_min_height}-"
                f"{a.occupancy_max_height} m above the floor) -> {out / (stem + '.yaml')}")
        return {
            "resolution_m": res,
            "height_band_m": [a.occupancy_min_height, a.occupancy_max_height],
            "size_cells": [int(width), int(height)],
            "origin_xy_m": [round(float(origin[0]), 4), round(float(origin[1]), 4)],
            "occupied_cells": n_occ,
            "free_cells": n_free,
            "unknown_cells": int(grid.size - n_occ - n_free),
            "files": [str(out / f"{stem}.pgm"), str(out / f"{stem}.yaml")],
        }
