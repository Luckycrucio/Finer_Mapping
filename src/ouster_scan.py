"""Parsing and preprocessing of raw Ouster PointCloud2 scans.

Ouster publishes one *organized* PointCloud2 per revolution: message height
equals the number of laser rings (64 here) and width the number of azimuth
firings per revolution (1024 here); row index equals the `ring` field exactly
(checked directly against the bag). Keeping that (ring, column) structure
lets a couple of preprocessing steps use cheap organized-image operations
(neighbour lookups by simple array indexing) instead of a k-nearest-neighbour
search over an unorganized cloud:

- `edge_filter` drops points sitting on a within-ring depth discontinuity
  (mixed pixels at silhouette edges, e.g. a door frame in front of a far
  wall) before they corrupt the TSDF with a surface that was never there.
- `organized_normals` gets a cheap per-point normal from neighbouring beams,
  used only to down-weight grazing-incidence returns (noisier) during fusion.

All Ouster invalid returns (`range == 0`) already come through as NaN x/y/z
from the ROS driver, confirmed by inspecting the raw bag.
"""
import numpy as np
from sensor_msgs_py import point_cloud2

FIELDS = ["x", "y", "z", "intensity", "t", "ring", "range"]


class OrganizedScan:
    def __init__(self, msg):
        structured = point_cloud2.read_points(msg, field_names=FIELDS, skip_nans=False)
        arr = structured.reshape(msg.height, msg.width)

        self.rows, self.cols = msg.height, msg.width
        self.xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float64)
        self.intensity = arr["intensity"].astype(np.float32)
        self.range_m = arr["range"].astype(np.float32) / 1000.0  # driver reports range in mm
        self.ring = arr["ring"].astype(np.int32)
        # All 64 beams fire simultaneously per column, so a per-column time is exact
        # (confirmed identical across rows for a fixed column in the raw bag).
        self._column_time_ns = arr["t"][0, :].astype(np.int64)
        self.header_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.valid = self.range_m > 0.0

    def column_times_s(self):
        return self.header_stamp + self._column_time_ns * 1e-9

    def edge_filter(self, jump_m=0.3):
        """Boolean (rows, cols) mask of points that are valid and not sitting
        on a depth discontinuity relative to their immediate azimuth
        neighbours in the same ring."""
        r = self.range_m
        left = np.roll(r, 1, axis=1)
        right = np.roll(r, -1, axis=1)
        jump = np.maximum(np.abs(r - left), np.abs(r - right))
        return self.valid & (jump < jump_m)

    def organized_normals(self):
        """Finite-difference unit normals from neighbouring rings/columns.

        Returns (normals (rows, cols, 3), valid (rows, cols)). This is a
        cheap approximation meant only for incidence-angle TSDF weighting,
        not a precise surface normal estimate."""
        xyz = self.xyz
        d_col = np.roll(xyz, -1, axis=1) - xyz
        d_row = np.empty_like(xyz)
        d_row[:-1] = xyz[1:] - xyz[:-1]
        d_row[-1] = d_row[-2]

        n = np.cross(d_col, d_row)
        norm = np.linalg.norm(n, axis=-1)

        next_col_valid = np.roll(self.valid, -1, axis=1)
        next_row_valid = np.vstack([self.valid[1:], self.valid[-1:]])
        valid = self.valid & next_col_valid & next_row_valid & (norm > 1e-9)

        normals = np.zeros_like(n)
        normals[valid] = n[valid] / norm[valid, None]
        return normals, valid
