"""Floor detection: a robust global plane plus a coarse per-cell height correction.

GLIM's world frame is not guaranteed to be gravity-aligned (coverage1's is
tilted ~2.1 deg, measured from both a plane fit through the trajectory and the
LiDAR's own z-axis in world coordinates, which agree), and over a whole map
the floor also drifts a few cm away from any single plane. So "below the
floor" cannot just be a fixed world-z cutoff. The model built here is:

1. **Candidates.** Points whose organized-cloud normal is within
   `max_normal_deg` of the sensor's up axis (i.e. horizontal surfaces) and
   that lie within `radius_m` horizontally of the sensor they were measured
   from. The robot drives on the floor, so near the sensor the dominant
   horizontal surface is the floor, at a height below the sensor equal to the
   (unknown but constant) mounting height.
2. **Mounting height.** A histogram of candidate heights relative to the
   sensor (`dz`, along the sensor's up axis) has its dominant peak at minus
   the mounting height; only candidates within `band_m` of that peak are kept.
   Measuring relative to the sensor at capture time means trajectory drift in
   z does not blur this step.
3. **Global plane, RANSAC then SVD.** SVD alone is a least-squares fit, so the
   very below-floor outliers this is meant to remove would pull it down.
   RANSAC (normals constrained to within `max_tilt_deg` of the sensor up axis)
   finds the inlier set first, then SVD on the inliers alone gives the
   precise plane.
4. **Per-cell correction.** The median residual to that plane of candidates
   within `cell_band_m` of it (so low furniture tops, steps or the robot's own
   footprint further away do not count) is taken per `cell_size_m` cell and
   clamped to +/-`cell_band_m`; empty cells are filled from the
   nearest measured cell, and the grid is lightly median-filtered. Heights are
   bilinearly interpolated between cell centres, so there are no steps at
   cell edges.

`FloorModel.height_above_floor(points)` then gives a signed distance to the
local floor (positive above), used by the fusion pass to drop below-floor
returns (LiDAR multipath reflections off glossy floors, mainly) and by
`refine_mesh.py` to clip and flatten.
"""
import json

import numpy as np
from scipy import ndimage
from scipy.stats import binned_statistic_2d


class FloorModel:
    def __init__(self, normal, offset, grid_origin, cell_size, grid):
        self.normal = np.asarray(normal, dtype=np.float64)
        self.offset = float(offset)
        self.grid_origin = np.asarray(grid_origin, dtype=np.float64)
        self.cell_size = float(cell_size)
        self.grid = np.asarray(grid, dtype=np.float64)

    def _correction(self, xy):
        coords = (xy - self.grid_origin) / self.cell_size - 0.5  # cell-centre coordinates
        return ndimage.map_coordinates(self.grid, coords.T, order=1, mode="nearest")

    def height_above_floor(self, points):
        """Signed distance (metres, along the floor normal) from each (N,3)
        world point to the local floor; negative below it."""
        points = np.asarray(points, dtype=np.float64)
        return points @ self.normal + self.offset - self._correction(points[:, :2])

    def project_to_floor(self, points):
        points = np.asarray(points, dtype=np.float64)
        return points - self.height_above_floor(points)[:, None] * self.normal

    def tilt_deg(self):
        return float(np.degrees(np.arccos(np.clip(abs(self.normal[2]), -1.0, 1.0))))

    def summary(self):
        return {
            "normal": self.normal.round(6).tolist(),
            "offset_m": round(self.offset, 6),
            "tilt_from_world_z_deg": round(self.tilt_deg(), 3),
            "cell_size_m": self.cell_size,
            "grid_shape": list(self.grid.shape),
            "cell_correction_range_m": [round(float(self.grid.min()), 4), round(float(self.grid.max()), 4)],
        }

    def save(self, path):
        np.savez(path, normal=self.normal, offset=self.offset, grid_origin=self.grid_origin,
                 cell_size=self.cell_size, grid=self.grid, summary=json.dumps(self.summary()))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["normal"], float(d["offset"]), d["grid_origin"], float(d["cell_size"]), d["grid"])


def fit_plane_svd(points):
    """Least-squares plane through `points`: returns (unit normal, offset)
    with normal . x + offset = 0. The normal is the right-singular vector of
    the centred points with the smallest singular value."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    return normal, -float(normal @ centroid)


def ransac_plane(points, up, inlier_thresh, max_tilt_deg, iterations=400, rng=None):
    """Plane with the most inliers among `iterations` random 3-point
    hypotheses whose normal lies within `max_tilt_deg` of `up`. Returns
    (normal, offset, inlier mask)."""
    rng = np.random.default_rng(0) if rng is None else rng
    tri = points[rng.integers(0, len(points), size=(iterations, 3))]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    ok = norms > 1e-9
    normals[ok] /= norms[ok, None]
    normals *= np.sign(normals @ up)[:, None]
    ok &= normals @ up >= np.cos(np.radians(max_tilt_deg))
    if not np.any(ok):
        raise RuntimeError("floor RANSAC: no plane hypothesis close enough to the sensor up axis")
    normals, tri = normals[ok], tri[ok]
    offsets = -np.einsum("ij,ij->i", normals, tri[:, 0])

    best, best_count = 0, -1
    for chunk in range(0, len(normals), 50):  # (50, N) distance blocks keep memory bounded
        d = np.abs(normals[chunk:chunk + 50] @ points.T + offsets[chunk:chunk + 50, None])
        counts = (d < inlier_thresh).sum(axis=1)
        i = int(np.argmax(counts))
        if counts[i] > best_count:
            best, best_count = chunk + i, int(counts[i])
    inliers = np.abs(points @ normals[best] + offsets[best]) < inlier_thresh
    return normals[best], float(offsets[best]), inliers


def select_candidates(points_world, origins_world, normals_world, sensor_up, normals_valid,
                      max_normal_deg=15.0, radius_m=4.0):
    """Per-point floor-candidate selection for one scan. Returns (mask, dz)
    where dz is each point's height relative to its own sensor position
    along that sensor's up axis."""
    rel = points_world - origins_world
    dz = np.einsum("ij,ij->i", rel, sensor_up)
    horiz = np.linalg.norm(rel - dz[:, None] * sensor_up, axis=1)
    facing = np.abs(np.einsum("ij,ij->i", normals_world, sensor_up))
    mask = normals_valid & (facing >= np.cos(np.radians(max_normal_deg))) & (horiz <= radius_m) & (dz < -0.1)
    return mask, dz


def estimate_floor(points, dz, up, cell_size_m=1.0, band_m=0.15, cell_band_m=0.08, inlier_thresh_m=0.03,
                   max_tilt_deg=10.0, min_cell_points=30, ransac_sample=150_000, log=print):
    """Fit a `FloorModel` to floor-candidate points (see module docstring).

    points: (N,3) world candidates; dz: (N,) height relative to the sensor
    that measured each; up: mean sensor up axis in world coordinates."""
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    if len(points) < 1000:
        raise RuntimeError(f"only {len(points)} floor candidates -- not enough to estimate a floor")

    hist, edges = np.histogram(dz, bins=np.arange(-3.0, -0.1, 0.01))
    hist = np.convolve(hist, np.ones(5) / 5.0, mode="same")
    peak = float(edges[np.argmax(hist)] + 0.005)
    in_band = np.abs(dz - peak) < band_m
    pts = points[in_band]
    log(f"floor: {len(points):,} horizontal-surface candidates, sensor mounted ~{-peak:.3f} m above the floor, "
        f"{len(pts):,} within +/-{band_m} m of that")

    rng = np.random.default_rng(0)
    sample = pts if len(pts) <= ransac_sample else pts[rng.choice(len(pts), ransac_sample, replace=False)]
    normal, offset, _ = ransac_plane(sample, up, inlier_thresh_m, max_tilt_deg, rng=rng)
    # SVD refinement on the inliers only; iterate so the inlier set follows the refined plane
    for _ in range(3):
        inliers = np.abs(pts @ normal + offset) < 2.0 * inlier_thresh_m
        normal, offset = fit_plane_svd(pts[inliers])
        if normal @ up < 0:
            normal, offset = -normal, -offset
    residual = pts @ normal + offset
    log(f"floor plane: normal={normal.round(4).tolist()} ({np.degrees(np.arccos(abs(normal[2]))):.2f} deg from "
        f"world z), {inliers.mean() * 100:.0f}% of in-band points within {2 * inlier_thresh_m:.2f} m, "
        f"inlier residual std {residual[inliers].std() * 100:.1f} cm")

    near = np.abs(residual) < cell_band_m
    xy, r = pts[near, :2], residual[near]
    grid_origin = xy.min(axis=0) - cell_size_m
    shape = np.ceil((xy.max(axis=0) + cell_size_m - grid_origin) / cell_size_m).astype(int) + 1
    bins = [grid_origin[i] + cell_size_m * np.arange(shape[i] + 1) for i in range(2)]
    median, _, _, _ = binned_statistic_2d(xy[:, 0], xy[:, 1], r, statistic="median", bins=bins)
    count, _, _, _ = binned_statistic_2d(xy[:, 0], xy[:, 1], r, statistic="count", bins=bins)
    measured = count >= min_cell_points
    grid = np.where(measured, median, np.nan)
    _, nearest = ndimage.distance_transform_edt(~measured, return_indices=True)
    grid = grid[nearest[0], nearest[1]]
    grid = np.clip(ndimage.median_filter(grid, size=3, mode="nearest"), -cell_band_m, cell_band_m)
    log(f"floor grid: {shape[0]}x{shape[1]} cells of {cell_size_m} m, {int(measured.sum())} measured, "
        f"per-cell correction {grid.min() * 100:+.1f} .. {grid.max() * 100:+.1f} cm")

    model = FloorModel(normal, offset, grid_origin, cell_size_m, grid)
    final = model.height_above_floor(pts)
    log(f"floor: in-band candidates vs final local floor, |h| median {np.median(np.abs(final)) * 100:.1f} cm "
        f"(vs {np.median(np.abs(residual[near])) * 100:.1f} cm against the global plane alone)")
    return model
