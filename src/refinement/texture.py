"""Texture: the detailed visual mesh, decimated and textured (OBJ + PNG).

Per-vertex colour ties colour detail to vertex density, so a mesh cannot be
simplified without blurring it, and simulators/game engines expect textured
meshes anyway. This step writes a textured copy of the current mesh; the
mesh itself (and the vertex-coloured PLY) is left unchanged.

1. **Geometry**, per part:
   - fused: decimated to `keep` of its triangles (quadric edge collapse,
     open borders preserved; at 0.3 the median deviation is <1 mm), then
     split into spatial tiles of at most `tile_triangles` triangles
     (recursive median cuts) and UV-unwrapped tile by tile with xatlas, in
     parallel. One xatlas run over the whole ~800k-triangle mesh did not
     finish in 30 CPU-minutes; a 13k-triangle tile takes 0.5 s;
   - floor: the (clipped) synthetic floor decimated to `floor_keep` of its
     triangles (it is a smooth surface, so this costs ~nothing), UV =
     planar projection;
   - walls: one rectangle per wall, columns every `wall_step` so the bottom
     follows the floor model, UV = (distance along the wall, height);
   - roof: two triangles, single colour.
2. **Packing.** Every piece's UV chart is a rectangle of texels at
   `texel` metres per texel; rectangles are shelf-packed into pages of at
   most `page` x `page` texels (one material per page).
3. **Baking.** Every texel covered by a triangle is mapped back to its 3D
   point on that triangle (barycentric, clamped for edge texels), and its
   colour interpolated (Gaussian weights, sigma = 0.6 voxel, 8 nearest
   vertices) from the detailed,
   vertex-coloured mesh: fused texels from the fused vertices, floor texels
   from the floor's, wall texels from fused + floor (the scanned walls just
   behind or in front of them); the roof gets the enclosure's median colour.
   Empty texels around the charts are filled from the nearest covered one,
   so bilinear filtering and mipmaps do not bleed background into seams.
4. **Output**: <model>/meshes/visual.obj + visual.mtl + visual_<page>.png,
   with per-vertex normals (one-sided parts keep their inward/up winding).
"""
import multiprocessing as mp
import time

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .base import PART_FLOOR, PART_FUSED, RefinementStep
from .geometry import compact, decimate, vertex_normals


def model_dir(ctx):
    return ctx.out_dir / "gazebo" / f"{ctx.name}_map"


class Piece:
    """A chunk of the visual mesh with its own rectangle of texture:
    verts (N,3), faces (M,3), uv (N,2) in texels from the rectangle's
    corner, normals (N,3), and how to colour it (`source`)."""

    def __init__(self, verts, faces, uv, normals, source):
        self.verts, self.faces, self.uv, self.normals, self.source = verts, faces, uv, normals, source
        self.size = np.ceil(uv.max(axis=0) + 1).astype(int) if len(uv) else np.array([1, 1])
        self.page, self.offset = None, None


# ---------------------------------------------------------------- unwrap

def split_tiles(verts, faces, max_triangles):
    """Face index arrays of spatial tiles: recursive median cuts along the
    longest extent until no tile has more than `max_triangles` faces."""
    centroid = verts[faces].mean(axis=1)
    out, stack = [], [np.arange(len(faces))]
    while stack:
        idx = stack.pop()
        if len(idx) <= max_triangles:
            out.append(idx)
            continue
        c = centroid[idx]
        axis = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
        order = np.argsort(c[:, axis], kind="stable")
        half = len(idx) // 2
        stack += [idx[order[:half]], idx[order[half:]]]
    return out


def _unwrap(args):
    import xatlas
    verts, faces, texels_per_unit = args
    atlas = xatlas.Atlas()
    atlas.add_mesh(verts.astype(np.float32), faces.astype(np.uint32))
    chart = xatlas.ChartOptions()
    chart.max_iterations = 1
    pack = xatlas.PackOptions()
    pack.texels_per_unit = texels_per_unit
    pack.padding = 2
    pack.bilinear = True
    pack.resolution = 0  # one atlas, as large as needed
    atlas.generate(chart, pack)
    vmap, idx, uv = atlas[0]
    return vmap.astype(np.int64), idx.astype(np.int64), uv * np.array([atlas.width, atlas.height])


def fused_pieces(verts, faces, texel, tile_triangles, workers, log):
    normals = vertex_normals(verts, faces)
    tiles = []
    for idx in split_tiles(verts, faces, tile_triangles):
        v, f, used = compact(verts, faces[idx])
        tiles.append((v, f, used))
    t = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_unwrap, [(v, f, 1.0 / texel) for v, f, _ in tiles], chunksize=1)
    log(f"texture: unwrapped {len(tiles)} fused tiles with xatlas in {time.time() - t:.1f} s")
    pieces = []
    for (v, f, used), (vmap, idx, uv) in zip(tiles, results):
        pieces.append(Piece(v[vmap], idx.reshape(-1, 3), uv, normals[used[vmap]], "fused"))
    return pieces


# ------------------------------------------------- floor / walls / roof

def _rotation_to_side0(corners):
    d = corners[1] - corners[0]
    a = np.arctan2(d[1], d[0])
    return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def floor_piece(verts, faces, corners, texel):
    p = verts[:, :2] @ _rotation_to_side0(corners).T
    uv = (p - p.min(axis=0)) / texel + 1.0
    return Piece(verts, faces, uv, vertex_normals(verts, faces), "floor")


def wall_pieces(corners, bottom_z, top_z, step, texel):
    pieces = []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        length = np.linalg.norm(b - a)
        s = np.linspace(0.0, 1.0, max(int(np.ceil(length / step)), 1) + 1)
        xy = a + s[:, None] * (b - a)
        zb, zt = bottom_z(xy), top_z(xy)
        verts = np.concatenate([np.column_stack([xy, zb]), np.column_stack([xy, zt])])
        n = len(s)
        i = np.arange(n - 1)
        # bottom i, i+1, top i+1, i: wound so the normal points into the room (corners counter-clockwise)
        faces = np.concatenate([np.stack([i, i + n + 1, i + 1], 1), np.stack([i, i + n, i + n + 1], 1)])
        # u along the wall, v down from its top (image rows grow downwards)
        uv = np.column_stack([np.r_[s, s] * length, np.r_[zt.max() - zb, zt.max() - zt]]) / texel + 1.0
        fn = np.cross(verts[faces[0, 1]] - verts[faces[0, 0]], verts[faces[0, 2]] - verts[faces[0, 0]])
        normals = np.repeat((fn / np.linalg.norm(fn))[None], len(verts), axis=0)
        pieces.append(Piece(verts, faces, uv, normals, "wall"))
    return pieces


def roof_piece(corners, top_z):
    verts = np.column_stack([corners, top_z(corners)])
    faces = np.array([[0, 2, 1], [0, 3, 2]])  # counter-clockwise corners: reversed, facing down
    uv = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
    return Piece(verts, faces, uv, np.repeat([[0.0, 0.0, -1.0]], 4, axis=0), "roof")


# ------------------------------------------------------- pack and bake

def shelf_pack(pieces, page):
    """Assign every piece a page and a texel offset; returns the pages'
    (width, height) actually used."""
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i].size[1])
    sizes, x, y, row_h, cur = [], 0, 0, 0, 0
    sizes.append([0, 0])
    for i in order:
        w, h = pieces[i].size
        if w > page or h > page:
            raise RuntimeError(f"texture: a chart of {w}x{h} texels does not fit a {page} page; raise "
                               "--texture-page or --texture-texel")
        if x + w > page:
            x, y, row_h = 0, y + row_h, 0
        if y + h > page:
            cur, x, y, row_h = cur + 1, 0, 0, 0
            sizes.append([0, 0])
        pieces[i].page, pieces[i].offset = cur, np.array([x, y])
        x += w
        row_h = max(row_h, h)
        sizes[cur] = [max(sizes[cur][0], x), max(sizes[cur][1], y + h)]
    return sizes


def bake_piece(piece, image, sources, roof_color, sigma):
    """Rasterise `piece` into its rectangle of `image` (H,W,3 uint8 RGB)."""
    w, h = piece.size
    ox, oy = piece.offset
    if piece.source == "roof":
        image[oy:oy + h, ox:ox + w] = roof_color
        return
    # triangle id per texel; OpenCV puts pixel centres on integer coordinates
    ids = np.zeros((h, w), dtype=np.float64)
    shift = 8
    pts = np.round((piece.uv[piece.faces] - 0.5) * (1 << shift)).astype(np.int32)
    for t, tri in enumerate(pts):
        cv2.fillConvexPoly(ids, tri, float(t + 1), lineType=cv2.LINE_8, shift=shift)
    rows, cols = np.nonzero(ids)
    tri = ids[rows, cols].astype(np.int64) - 1
    p = np.column_stack([cols + 0.5, rows + 0.5])
    a, b, c = (piece.uv[piece.faces[tri, k]] for k in range(3))
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    lam = np.clip(np.column_stack([1 - l1 - l2, l1, l2]), 0, None)
    lam /= np.maximum(lam.sum(axis=1, keepdims=True), 1e-12)
    xyz = sum(lam[:, k, None] * piece.verts[piece.faces[tri, k]] for k in range(3))

    tree, colors = sources[piece.source]
    dist, nn = tree.query(xyz, k=8, workers=-1)
    # Gaussian weights: blends neighbouring vertices like the vertex-coloured
    # mesh's own interpolation (inverse-distance weights act like nearest-
    # neighbour and show every vertex as a flat patch)
    wgt = np.exp(-0.5 * (dist / sigma) ** 2) + 1e-12
    rgb = (colors[nn] * wgt[..., None]).sum(axis=1) / wgt.sum(axis=1, keepdims=True)

    tile = np.zeros((h, w, 3), dtype=np.uint8)
    tile[rows, cols] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    empty = ids == 0
    if empty.any() and (~empty).any():
        _, (ii, jj) = ndimage.distance_transform_edt(empty, return_indices=True)
        tile = tile[ii, jj]
    image[oy:oy + h, ox:ox + w] = tile


def write_obj(path, pieces, page_sizes, image_names):
    """One OBJ with a material per texture page."""
    mtl = path.with_suffix(".mtl")
    with open(mtl, "w") as f:
        for i, name in enumerate(image_names):
            f.write(f"newmtl page{i}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd {name}\n\n")
    with open(path, "w") as f:
        f.write(f"mtllib {mtl.name}\n")
        base = 0
        blocks = {i: [] for i in range(len(page_sizes))}
        for p in pieces:
            pw, ph = page_sizes[p.page]
            uv = (p.uv + p.offset) / np.array([pw, ph])
            uv[:, 1] = 1.0 - uv[:, 1]
            np.savetxt(f, p.verts, fmt="v %.5f %.5f %.5f")
            np.savetxt(f, uv, fmt="vt %.6f %.6f")
            np.savetxt(f, p.normals, fmt="vn %.4f %.4f %.4f")
            blocks[p.page].append(p.faces + base + 1)
            base += len(p.verts)
        for i, faces in blocks.items():
            if not faces:
                continue
            f.write(f"usemtl page{i}\n")
            ff = np.concatenate(faces)
            np.savetxt(f, np.repeat(ff, 3, axis=1), fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")


class TextureStep(RefinementStep):
    name = "texture"
    help = "write the detailed, decimated, textured visual mesh (OBJ + PNG pages)"

    @classmethod
    def add_args(cls, p):
        g = p.add_argument_group("texture step")
        g.add_argument("--texture-texel", type=float, default=0.015,
                       help="texel size, metres (the fused colour itself is sampled every ~voxel size)")
        g.add_argument("--texture-keep", type=float, default=0.3,
                       help="fraction of the fused triangles kept by decimation (1 keeps all)")
        g.add_argument("--texture-floor-keep", type=float, default=0.02,
                       help="fraction of the (smooth) floor's triangles kept by decimation")
        g.add_argument("--texture-wall-step", type=float, default=0.25,
                       help="vertex spacing, metres, along the wall bottoms (which follow the floor)")
        g.add_argument("--texture-tile-triangles", type=int, default=30000,
                       help="max triangles per xatlas tile")
        g.add_argument("--texture-page", type=int, default=8192, help="max texture page size, texels")
        g.add_argument("--texture-format", choices=["png", "jpg"], default="png")
        g.add_argument("--texture-workers", type=int, default=max(min(mp.cpu_count() - 1, 8), 1),
                       help="parallel xatlas processes (each is a fork of this one: 15 ran out of 32 GB)")

    def run(self, mesh, ctx):
        a, floor = self.args, ctx.floor
        enc = ctx.require("enclosure", self.name)
        corners = enc["corners"]
        top_z = lambda xy: floor.plane_z(xy, enc["roof_height"])  # noqa: E731
        bottom_z = lambda xy: floor.floor_z(xy) - enc["wall_sink"]  # noqa: E731

        fv, ff, fc, _ = mesh.submesh(mesh.parts == PART_FUSED)
        lv, lf, lc, _ = mesh.submesh(mesh.parts == PART_FLOOR)
        dv, df = decimate(fv, ff, keep_fraction=a.texture_keep)
        ctx.log(f"texture: fused geometry decimated {len(ff):,} -> {len(df):,} triangles")
        pieces = fused_pieces(dv, df, a.texture_texel, a.texture_tile_triangles, a.texture_workers, ctx.log)
        pieces.append(floor_piece(*decimate(lv, lf, keep_fraction=a.texture_floor_keep), corners, a.texture_texel))
        pieces += wall_pieces(corners, bottom_z, top_z, a.texture_wall_step, a.texture_texel)
        pieces.append(roof_piece(corners, top_z))

        page_sizes = shelf_pack(pieces, a.texture_page)
        sources = {"fused": (cKDTree(fv), fc.astype(np.float64)),
                   "floor": (cKDTree(lv), lc.astype(np.float64)),
                   "wall": (cKDTree(np.concatenate([fv, lv])), np.concatenate([fc, lc]).astype(np.float64))}
        sigma = 0.6 * float(ctx.build_report.get("voxel_size_m", 0.03))
        out = model_dir(ctx) / "meshes"
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        names = []
        for i, (w, h) in enumerate(page_sizes):
            image = np.zeros((h, w, 3), dtype=np.uint8)
            for p in pieces:
                if p.page == i:
                    bake_piece(p, image, sources, enc["roof_color"], sigma)
            names.append(f"visual_{i}.{a.texture_format}")
            cv2.imwrite(str(out / names[-1]), image[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95] if a.texture_format == "jpg" else [])
        ctx.log(f"texture: baked {len(page_sizes)} page(s) {page_sizes} in {time.time() - t:.1f} s")

        write_obj(out / "visual.obj", pieces, page_sizes, names)
        n_tri = sum(len(p.faces) for p in pieces)
        n_v = sum(len(p.verts) for p in pieces)
        ctx.state["visual_mesh"] = out / "visual.obj"
        ctx.log(f"texture: wrote {out / 'visual.obj'} ({n_v:,} vertices, {n_tri:,} triangles)")
        return {
            "texel_m": a.texture_texel,
            "fused_keep_fraction": a.texture_keep,
            "fused_triangles": int(len(df)),
            "fused_tiles": int(sum(p.source == "fused" for p in pieces)),
            "triangles": int(n_tri),
            "vertices": int(n_v),
            "pages": [list(map(int, s)) for s in page_sizes],
            "files": [str(out / "visual.obj"), str(out / "visual.mtl")] + [str(out / n) for n in names],
        }
