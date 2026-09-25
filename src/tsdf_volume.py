"""Dense point/ray-based TSDF volume for LiDAR fusion.

Standard TSDF fusion (KinectFusion and Open3D's ScalableTSDFVolume) projects
a *pinhole depth image* into the volume every frame. A spinning LiDAR scan is
not a pinhole image (it is a spherical/equirectangular sweep), so this
implements the point-cloud generalisation used by tools like VDBFusion
instead: every measured point p is paired with the sensor origin o it was
shot from, giving a ray direction d = (p - o) / |p - o|. A short band of
voxels around p (+/- truncation_distance, sampled at voxel resolution) is
updated with

    sdf(voxel) = dot(p - voxel_center, d)

which is positive between the sensor and the surface (free space) and
negative beyond it (occluded/inside), exactly the KinectFusion projective-TSDF
convention, generalised to an arbitrary ray direction per point instead of a
shared camera ray grid. Voxels are combined across scans with a running
weighted average, the same rule projective TSDF fusion uses.

`carve` adds the other half of what a real sensor ray tells you: every voxel
it passed through *before* reaching the band around its endpoint was empty at
that moment. Without it, only the +/- truncation band is ever updated, so a
person walking through the room is fused as a permanent ghost surface --
nothing later ever contradicts it. Carving pushes already-observed voxels a
ray has since flown through towards "free" (+truncation) with a small weight
per scan, so a surface seen a handful of times and then repeatedly seen
*through* fades, while static geometry (observed thousands of times) barely
moves. Carving never creates observations in voxels that had none, and can be
kept away from the floor, where near-grazing rays would otherwise slowly erode
it.

This is a plain dense NumPy grid (no octree/hashing), which is only tractable
because the scene is a single room-scale indoor map -- see the README for the
memory-vs-voxel-size trade-off.
"""
import numpy as np


class TSDFVolume:
    def __init__(self, bounds_min, bounds_max, voxel_size, truncation_distance):
        self.voxel_size = float(voxel_size)
        self.truncation_distance = float(truncation_distance)
        self.origin = np.asarray(bounds_min, dtype=np.float64)

        dims = np.ceil((np.asarray(bounds_max, dtype=np.float64) - self.origin) / self.voxel_size).astype(np.int64)
        self.dims = np.maximum(dims, 1) + 1  # (nx, ny, nz)
        self._strides = np.array(
            [self.dims[1] * self.dims[2], self.dims[2], 1], dtype=np.int64
        )
        n = int(self.dims[0]) * int(self.dims[1]) * int(self.dims[2])

        self.weight = np.zeros(n, dtype=np.float32)
        self.wsdf = np.zeros(n, dtype=np.float32)
        self.wcolor = np.zeros((n, 3), dtype=np.float32)

        n_samples = max(3, int(round(2 * truncation_distance / voxel_size)) + 1)
        self._sample_offsets = np.linspace(-truncation_distance, truncation_distance, n_samples)

    def _voxel_index(self, world_xyz):
        return np.floor((world_xyz - self.origin) / self.voxel_size).astype(np.int64)

    def integrate(self, points, origins, colors, weights):
        """points, origins, colors: (N,3) world-frame arrays; weights: (N,).
        `origins` is the sensor position each point was measured from (per
        point, since a scan is deskewed against a moving trajectory)."""
        if len(points) == 0:
            return

        directions = points - origins
        ranges = np.linalg.norm(directions, axis=1)
        keep = ranges > 1e-6
        if not np.any(keep):
            return
        points, colors, weights = points[keep], colors[keep], weights[keep]
        directions, ranges = directions[keep], ranges[keep]
        directions = directions / ranges[:, None]

        # (N, S, 3) sample positions along each ray around its surface point
        samples = points[:, None, :] + directions[:, None, :] * self._sample_offsets[None, :, None]
        n_pts, n_samples, _ = samples.shape
        samples = samples.reshape(-1, 3)

        idx = self._voxel_index(samples)
        in_bounds = np.all((idx >= 0) & (idx < self.dims), axis=1)
        if not np.any(in_bounds):
            return
        idx = idx[in_bounds]
        linear = idx @ self._strides

        voxel_centers = self.origin + (idx.astype(np.float64) + 0.5) * self.voxel_size
        pts_rep = np.repeat(points, n_samples, axis=0)[in_bounds]
        dirs_rep = np.repeat(directions, n_samples, axis=0)[in_bounds]
        colors_rep = np.repeat(colors, n_samples, axis=0)[in_bounds]
        w_rep = np.repeat(weights, n_samples, axis=0)[in_bounds]

        sdf = np.einsum("ij,ij->i", pts_rep - voxel_centers, dirs_rep)
        sdf = np.clip(sdf, -self.truncation_distance, self.truncation_distance)
        # taper contribution weight near the truncation limit so a single new
        # scan's coarse guess at the band edge cannot overwrite a voxel that
        # many earlier scans have already converged on
        w_rep = w_rep * np.clip(1.0 - np.abs(sdf) / self.truncation_distance, 0.05, 1.0)

        uniq, inv = np.unique(linear, return_inverse=True)
        local_w = np.bincount(inv, weights=w_rep, minlength=len(uniq))
        local_wsdf = np.bincount(inv, weights=w_rep * sdf, minlength=len(uniq))

        self.weight[uniq] += local_w
        self.wsdf[uniq] += local_wsdf
        for c in range(3):
            self.wcolor[uniq, c] += np.bincount(inv, weights=w_rep * colors_rep[:, c], minlength=len(uniq))

    def carve(self, points, origins, carve_weight, keep_voxel=None):
        """Free-space update along the rays origins -> points, stopping one
        truncation distance plus one voxel short of each endpoint (so the
        surface band itself is left to `integrate`). Every already-observed
        voxel crossed by at least one ray gets one free-space observation
        (sdf = +truncation, weight `carve_weight`) per call; each voxel's
        colour average is preserved. `keep_voxel`, if given, maps (M,3) voxel
        centres to a boolean mask of voxels allowed to be carved."""
        if len(points) == 0:
            return 0
        directions = points - origins
        ranges = np.linalg.norm(directions, axis=1)
        stop = ranges - (self.truncation_distance + self.voxel_size)
        ok = stop > self.voxel_size
        if not np.any(ok):
            return 0
        origins, directions, ranges, stop = origins[ok], directions[ok], ranges[ok], stop[ok]
        directions = directions / ranges[:, None]

        steps = np.arange(1, int(np.ceil(stop.max() / self.voxel_size)) + 1) * self.voxel_size
        along = steps[None, :] < stop[:, None]
        ray_i, step_i = np.nonzero(along)
        samples = origins[ray_i] + directions[ray_i] * steps[step_i, None]

        idx = self._voxel_index(samples)
        in_bounds = np.all((idx >= 0) & (idx < self.dims), axis=1)
        linear = np.unique(idx[in_bounds] @ self._strides)
        linear = linear[self.weight[linear] > 0]
        if keep_voxel is not None and len(linear):
            vox = np.stack(np.unravel_index(linear, self.dims), axis=1)
            linear = linear[keep_voxel(self.origin + (vox + 0.5) * self.voxel_size)]
        if len(linear) == 0:
            return 0

        w_old = self.weight[linear]
        w_new = w_old + carve_weight
        self.wcolor[linear] *= (w_new / w_old)[:, None]
        self.weight[linear] = w_new
        self.wsdf[linear] += carve_weight * self.truncation_distance
        return len(linear)

    def extract_mesh(self, min_weight=1.0):
        from skimage.measure import marching_cubes

        weight = self.weight.reshape(self.dims)
        wsdf = self.wsdf.reshape(self.dims)
        wcolor = self.wcolor.reshape(*self.dims, 3)

        observed = weight > min_weight
        if not np.any(observed):
            raise RuntimeError("no voxels reached the requested minimum weight -- lower --min-weight")
        # Only evaluate cubes whose 8 corners are all observed. Unobserved
        # voxels hold a +truncation placeholder, so a cube straddling the edge
        # of the observed region would compare real (often negative, behind-
        # surface) values against it and produce a fake zero crossing: a
        # phantom surface at the back of the truncation band (e.g. a second
        # floor ~7 cm under the real one) and "curtains" along every
        # observation boundary. skimage tests `mask` at a cube's *upper*
        # corner (the cube spanning [i-1, i] on each axis is kept if mask[i]),
        # checked empirically against skimage 0.26.
        all_corners = observed[1:, 1:, 1:].copy()
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
            all_corners &= observed[1 - dx:observed.shape[0] - dx,
                                    1 - dy:observed.shape[1] - dy,
                                    1 - dz:observed.shape[2] - dz]
        mask = np.zeros_like(observed)
        mask[1:, 1:, 1:] = all_corners

        tsdf = np.where(observed, wsdf / np.maximum(weight, 1e-6), self.truncation_distance)
        color = np.zeros_like(wcolor)
        color[observed] = wcolor[observed] / np.maximum(weight[observed], 1e-6)[:, None]

        verts, faces, _normals, _values = marching_cubes(
            tsdf, level=0.0, spacing=(self.voxel_size,) * 3, mask=mask
        )
        # marching cubes places grid sample i at i * voxel_size, but sample i
        # holds voxel i's *centre* (origin + (i + 0.5) * voxel_size, the same
        # convention `integrate` uses), hence the half-voxel shift
        verts_world = verts + self.origin + 0.5 * self.voxel_size

        # colour from the nearest voxel centre (a vertex lies on the edge between two)
        vidx = np.rint(verts / self.voxel_size).astype(np.int64)
        vidx = np.clip(vidx, 0, self.dims - 1)
        vertex_colors = color[vidx[:, 0], vidx[:, 1], vidx[:, 2]]

        stats = {
            "grid_dims": [int(d) for d in self.dims],
            "observed_voxels": int(observed.sum()),
            "total_voxels": int(observed.size),
        }
        return verts_world, faces, vertex_colors, stats
