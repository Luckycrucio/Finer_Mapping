"""Synthetic floor replacing the fused floor.

Floor points are not fused into the TSDF (a measured floor is never quite
flat, and anything a person left on it stays joined to it). Instead their
colours are accumulated into a 2-D grid of `cell_size` cells in world xy
(`FloorColorGrid`), with the same RGB-over-intensity weighting the TSDF uses.
At the end, the cells that saw floor, closed over small gaps, with enclosed
holes (under furniture, unseen patches) filled and reduced to the largest
connected region, give the map's floor outline. `build_slab` turns that
outline into a mesh: by default a single surface lying exactly on the floor
model (global plane + smooth per-cell correction), or a closed flat slab
parallel to the global plane; per-vertex colour comes from the grid (cells
with no floor colour of their own take the nearest one's).
"""
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components


class FloorColorGrid:
    def __init__(self, origin_xy, cell_size, shape):
        self.origin = np.asarray(origin_xy, dtype=np.float64)
        self.cell_size = float(cell_size)
        self.shape = (int(shape[0]), int(shape[1]))
        n = self.shape[0] * self.shape[1]
        self.weight = np.zeros(n, dtype=np.float64)
        self.wcolor = np.zeros((n, 3), dtype=np.float64)

    def add(self, points, colors, color_weights):
        ij = np.floor((points[:, :2] - self.origin) / self.cell_size).astype(np.int64)
        ok = np.all((ij >= 0) & (ij < self.shape), axis=1)
        if not np.any(ok):
            return
        linear = ij[ok, 0] * self.shape[1] + ij[ok, 1]
        w = color_weights[ok]
        n = len(self.weight)
        self.weight += np.bincount(linear, weights=w, minlength=n)
        for c in range(3):
            self.wcolor[:, c] += np.bincount(linear, weights=w * colors[ok, c], minlength=n)

    def outline(self, fill_radius):
        """Boolean (nx, ny) floor mask: observed cells, closed over gaps up to
        `fill_radius` metres, enclosed holes filled, largest region only."""
        seen = (self.weight > 0).reshape(self.shape)
        if not np.any(seen):
            return seen
        r = int(round(fill_radius / self.cell_size))
        mask = seen
        if r > 0:
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            disk = xx ** 2 + yy ** 2 <= r ** 2
            # pad so the closing's erosion does not eat the array border
            mask = ndimage.binary_closing(np.pad(seen, r), structure=disk)[r:-r, r:-r]
        mask = ndimage.binary_fill_holes(mask)
        labels, n = ndimage.label(mask)
        if n > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            mask = labels == np.argmax(sizes)
        return mask

    def colors(self):
        """(nx, ny, 3) colour per cell; cells with no floor sample take the
        nearest observed cell's colour."""
        seen = (self.weight > 0).reshape(self.shape)
        color = np.zeros((*self.shape, 3))
        if not np.any(seen):
            return color
        color[seen] = (self.wcolor[self.weight > 0] / self.weight[self.weight > 0, None])
        _, (ii, jj) = ndimage.distance_transform_edt(~seen, return_indices=True)
        return color[ii, jj]


def build_slab(mask, cell_colors, origin_xy, cell_size, top_z, bottom_z=None):
    """Floor mesh over the `mask` cells. `top_z(xy)` gives the floor surface's
    z at (N,2) points; with `bottom_z` too, the result is a closed solid (a
    bottom face at `bottom_z` plus side walls along the outline), otherwise a
    single upward-facing surface. Vertices sit on the cell corners, so the
    surface follows `top_z` exactly at every corner.
    Returns (verts (N,3), faces (M,3), colours (N,3) float, BGR like the TSDF)."""
    nx, ny = mask.shape
    ci, cj = np.nonzero(mask)
    if len(ci) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3))

    # corner lattice: corner (i, j) is at origin + (i, j) * cell_size
    corner_used = np.zeros((nx + 1, ny + 1), dtype=bool)
    corner_csum = np.zeros((nx + 1, ny + 1, 3))
    corner_cnt = np.zeros((nx + 1, ny + 1))
    for di, dj in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        corner_used[ci + di, cj + dj] = True
        np.add.at(corner_csum, (ci + di, cj + dj), cell_colors[ci, cj])
        np.add.at(corner_cnt, (ci + di, cj + dj), 1.0)
    ki, kj = np.nonzero(corner_used)
    n_top = len(ki)
    index = np.full((nx + 1, ny + 1), -1, dtype=np.int64)
    index[ki, kj] = np.arange(n_top)

    xy = np.asarray(origin_xy, dtype=np.float64) + np.stack([ki, kj], axis=1) * cell_size
    corner_color = corner_csum[ki, kj] / corner_cnt[ki, kj, None]

    a, b = index[ci, cj], index[ci + 1, cj]
    c, d = index[ci + 1, cj + 1], index[ci, cj + 1]
    faces = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]                       # top, facing up
    verts = np.column_stack([xy, top_z(xy)])
    colors = corner_color
    if bottom_z is None:
        return verts, np.concatenate(faces).astype(np.int64), colors

    verts = np.concatenate([verts, np.column_stack([xy, bottom_z(xy)])])
    colors = np.concatenate([corner_color, corner_color])
    faces += [np.stack([a, c, b], 1) + n_top, np.stack([a, d, c], 1) + n_top]      # bottom, facing down

    # side walls on every cell edge whose neighbour is outside the mask; (u, v)
    # ordered so the outward direction is on the left of u -> v
    padded = np.pad(mask, 1)
    for (oi, oj), (u, v) in [((-1, 0), ((0, 0), (0, 1))), ((1, 0), ((1, 1), (1, 0))),
                             ((0, -1), ((1, 0), (0, 0))), ((0, 1), ((0, 1), (1, 1)))]:
        edge = ~padded[ci + 1 + oi, cj + 1 + oj]
        ei, ej = ci[edge], cj[edge]
        tu = index[ei + u[0], ej + u[1]]
        tv = index[ei + v[0], ej + v[1]]
        faces += [np.stack([tu, tv, tv + n_top], 1), np.stack([tu, tv + n_top, tu + n_top], 1)]

    return verts, np.concatenate(faces).astype(np.int64), colors


def find_slab(verts, faces, floor, tol=1e-3, min_faces=100):
    """Boolean face mask of the synthetic floor inside a merged mesh: the
    connected pieces whose vertices all lie (within `tol` metres) either on the
    floor model's surface itself, or on one of two planes parallel to its
    global plane (a flat slab). Fused geometry is never that exact, so nothing
    else matches. `floor` must be in the mesh's frame (see FloorModel.shifted_z)."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    graph = sparse.coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
                              shape=(len(verts), len(verts)))
    n_comp, labels = connected_components(graph, directed=False)
    used = np.zeros(len(verts), dtype=bool)
    used[faces.ravel()] = True
    vl = labels[used]
    n_faces = np.bincount(labels[faces[:, 0]], minlength=n_comp)

    # on the floor model's surface
    off_model = np.abs(floor.height_above_floor(verts[used])) >= tol
    on_model = np.bincount(vl, weights=off_model, minlength=n_comp) == 0

    # on two planes parallel to the global plane
    vh = verts[used] @ floor.normal + floor.offset
    lo = np.full(n_comp, np.inf)
    hi = np.full(n_comp, -np.inf)
    np.minimum.at(lo, vl, vh)
    np.maximum.at(hi, vl, vh)
    off_planes = (np.abs(vh - lo[vl]) >= tol) & (np.abs(vh - hi[vl]) >= tol)
    on_planes = (np.bincount(vl, weights=off_planes, minlength=n_comp) == 0) & (hi - lo > 2 * tol)

    slab_comp = (on_model | on_planes) & (n_faces >= min_faces)
    return slab_comp[labels[faces[:, 0]]]
