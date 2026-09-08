"""Pose interpolation for a GLIM TUM-format trajectory file (traj_lidar.txt).

Each row is ``t x y z qx qy qz qw`` and gives T_world_lidar: the pose that
carries a point expressed in the LiDAR (os_sensor) frame into the shared
world/map frame GLIM solved for. coverage1_edited/traj_lidar.txt samples this
at roughly 100 ms, matching the ~10 Hz scan rate, so linear interpolation of
position plus spherical interpolation (slerp) of orientation between the two
bracketing samples is an accurate way to get a pose at any in-between time,
including at the sub-scan (per-column) granularity used for deskewing.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class Trajectory:
    def __init__(self, tum_path):
        data = np.loadtxt(tum_path)
        order = np.argsort(data[:, 0])
        data = data[order]
        # Slerp requires strictly increasing sample times.
        keep = np.concatenate([[True], np.diff(data[:, 0]) > 0])
        data = data[keep]

        self.times = data[:, 0]
        self.positions = data[:, 1:4]
        self.rotations = Rotation.from_quat(data[:, 4:8])
        self._slerp = Slerp(self.times, self.rotations)

    @property
    def t_min(self):
        return self.times[0]

    @property
    def t_max(self):
        return self.times[-1]

    def in_range(self, t):
        return (t >= self.t_min) & (t <= self.t_max)

    def matrices(self, t):
        """t: (N,) epoch-second timestamps. Returns (N,4,4) world<-lidar
        homogeneous transforms. Timestamps outside [t_min, t_max] are clamped
        to the nearest end; callers should filter with `in_range` first if
        they want those samples dropped instead of held constant."""
        t = np.clip(np.asarray(t, dtype=np.float64), self.t_min, self.t_max)
        idx = np.searchsorted(self.times, t)
        idx = np.clip(idx, 1, len(self.times) - 1)
        t0, t1 = self.times[idx - 1], self.times[idx]
        alpha = np.where(t1 > t0, (t - t0) / (t1 - t0), 0.0)
        p0, p1 = self.positions[idx - 1], self.positions[idx]
        positions = p0 + (p1 - p0) * alpha[:, None]
        rotmats = self._slerp(t).as_matrix()

        T = np.zeros((len(t), 4, 4))
        T[:, :3, :3] = rotmats
        T[:, :3, 3] = positions
        T[:, 3, 3] = 1.0
        return T

    def bounds(self, padding=0.0):
        lo = self.positions.min(axis=0) - padding
        hi = self.positions.max(axis=0) + padding
        return lo, hi
