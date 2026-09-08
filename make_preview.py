#!/usr/bin/env python3
"""Render a static PNG preview of the finer mesh using its own RGB vertex
colours (a scatter of a random vertex subsample, not the full triangulated
surface -- fast enough for a quick sanity check on an 11M-triangle mesh, and
a more direct check of the RGB colourisation than a height-coloured render
would be). Mirrors the preview approach in ../coverage1_edited/finish_mesh.py.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from vtk.util import numpy_support

here = Path(__file__).resolve().parent
ply_path = here / "outputs" / "coverage1_finer_mesh.ply"
out_path = here / "outputs" / "preview.png"

reader = vtk.vtkPLYReader()
reader.SetFileName(str(ply_path))
reader.Update()
poly = reader.GetOutput()

points = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
colors = numpy_support.vtk_to_numpy(poly.GetPointData().GetScalars()) / 255.0
print(f"loaded {len(points):,} vertices from {ply_path}")

rng = np.random.default_rng(0)
n_sample = min(400_000, len(points))
idx = rng.choice(len(points), size=n_sample, replace=False)
pts, cols = points[idx], colors[idx]

fig = plt.figure(figsize=(12, 9), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.35, linewidths=0)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"coverage1 finer TSDF mesh -- {len(points):,} vertices ({n_sample:,} shown, RGB/intensity colour)")
ax.view_init(elev=55, azim=-65)

ranges = pts.max(axis=0) - pts.min(axis=0)
ax.set_box_aspect(ranges)

fig.tight_layout()
fig.savefig(out_path)
print(f"wrote {out_path}")
