"""VTK mesh conversion and export shared by build_finer_map.py and map_refinement.py."""
import numpy as np
import vtk
from vtk.util import numpy_support


def polydata_from_arrays(verts, faces, colors_rgb_u8):
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float64)))

    cells = vtk.vtkCellArray()
    n_faces = len(faces)
    cell_data = np.empty((n_faces, 4), dtype=np.int64)
    cell_data[:, 0] = 3
    cell_data[:, 1:] = faces
    cells.SetCells(n_faces, numpy_support.numpy_to_vtkIdTypeArray(cell_data.reshape(-1)))

    color_array = numpy_support.numpy_to_vtk(np.ascontiguousarray(colors_rgb_u8, dtype=np.uint8),
                                             array_type=vtk.VTK_UNSIGNED_CHAR)
    color_array.SetName("RGB")

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    poly.GetPointData().SetScalars(color_array)
    return poly


def arrays_from_polydata(poly):
    """(verts (N,3) float64, faces (M,3) int64, colours (N,3) uint8 or None).
    Assumes an all-triangle mesh."""
    verts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    conn = numpy_support.vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).astype(np.int64)
    faces = conn.reshape(-1, 3)
    scalars = poly.GetPointData().GetScalars()
    colors = numpy_support.vtk_to_numpy(scalars).astype(np.uint8) if scalars is not None else None
    return verts, faces, colors


def read_ply(path):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = vtk.vtkPolyData()
    poly.DeepCopy(reader.GetOutput())
    scalars = poly.GetPointData().GetScalars()
    if scalars is not None:
        scalars.SetName("RGB")
    return poly


def run_filter(f, poly):
    f.SetInputData(poly)
    f.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(f.GetOutput())
    return out


def finish_and_write(poly, out_dir, stem, formats=("ply", "obj", "stl"), orient=True):
    """Merge duplicate points, compute normals and write <stem>.ply (binary,
    per-vertex RGB), .obj and .stl (geometry only). With `orient`, triangle
    windings are first made consistent and outward-facing; without it every
    triangle keeps its winding (needed for one-sided, inward-facing parts)."""
    clean = vtk.vtkCleanPolyData()
    cleaned = run_filter(clean, poly)

    normals = vtk.vtkPolyDataNormals()
    normals.SplittingOff()
    if orient:
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
    else:
        normals.ConsistencyOff()
        normals.AutoOrientNormalsOff()
    final = run_filter(normals, cleaned)

    for ext, writer_cls in [("ply", vtk.vtkPLYWriter), ("obj", vtk.vtkOBJWriter), ("stl", vtk.vtkSTLWriter)]:
        if ext not in formats:
            continue
        writer = writer_cls()
        writer.SetFileName(str(out_dir / f"{stem}.{ext}"))
        writer.SetInputData(final)
        if ext == "ply":
            writer.SetArrayName("RGB")
            writer.SetColorModeToDefault()
            writer.SetFileTypeToBinary()
        if ext == "stl":
            writer.SetFileTypeToBinary()
        writer.Write()
    return final
