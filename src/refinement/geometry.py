"""Mesh helpers shared by the refinement steps."""
import numpy as np


def decimate(verts, faces, keep_fraction=None, target=None, preserve_border=True, passes=4):
    """Quadric edge-collapse decimation (pyfqmr) to `target` triangles or
    `keep_fraction` of them. Returns (verts, faces). A single pyfqmr run can
    stall well above the target (e.g. 31k instead of 20k on coverage1's
    collision geometry); a fresh run on its output usually gets there, so up
    to `passes` runs are chained while more than 5% above the target."""
    import pyfqmr
    target = max(int(target if target is not None else len(faces) * keep_fraction), 4)
    v, f = np.ascontiguousarray(verts, dtype=np.float64), np.asarray(faces)
    for _ in range(passes):
        if len(f) <= target * 1.05:
            break
        s = pyfqmr.Simplify()
        s.setMesh(v, np.ascontiguousarray(f, dtype=np.int32))
        s.simplify_mesh(target_count=target, aggressiveness=5, preserve_border=preserve_border, verbose=False)
        v, f, _ = s.getMesh()
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def face_normals(verts, faces):
    n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    return n


def vertex_normals(verts, faces):
    """Area-weighted unit vertex normals (from the faces' winding)."""
    fn = face_normals(verts, faces)
    vn = np.zeros_like(verts, dtype=np.float64)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.where(norm > 0, vn / np.maximum(norm, 1e-30), np.array([0.0, 0.0, 1.0]))


def compact(verts, faces):
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces], used


def drop_degenerate(verts, faces, min_area=1e-12):
    """Boolean keep mask: False for faces with a repeated vertex, (near-)zero
    area, or the same vertex set as an earlier face."""
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    keep &= np.linalg.norm(face_normals(verts, faces), axis=1) > 2 * min_area
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    unique = np.zeros(len(faces), dtype=bool)
    unique[first] = True
    return keep & unique


def inside_quad(xy, corners, margin=0.0):
    """Points inside a counter-clockwise convex quadrilateral grown by
    `margin` metres (signed distance to every side's line >= -margin)."""
    e = np.roll(corners, -1, axis=0) - corners
    n = np.stack([-e[:, 1], e[:, 0]], axis=1) / np.linalg.norm(e, axis=1, keepdims=True)  # inward
    return np.all(np.einsum("nkd,kd->nk", xy[:, None, :] - corners[None], n) >= -margin, axis=1)
