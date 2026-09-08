# Finer-detail RGB mesh via TSDF fusion

![Mesh preview in a 3D viewer](outputs/mesh_screenshot.png)

This is a second, independent reconstruction of the same environment as
[`../coverage1_edited/`](../coverage1_edited/), built to get past that
pipeline's main limitation: GLIM's saved map only keeps a downsampled point
cloud (149,813 points, ~6.35 cm median spacing), so no meshing algorithm run
on it can recover detail finer than that spacing.

Instead of reading GLIM's saved map, this pipeline goes back to the original
ROS 2 bag and re-derives everything from the **raw sensor stream**: every
`/ouster/points` scan (full ~60k points/revolution instead of GLIM's
decimated keyframe map) plus the `/rgb/image_raw` camera feed, fused with
GLIM's own solved trajectory. It also switches reconstruction algorithm, from
screened Poisson (which fits an implicit function to oriented points and can
extrapolate across gaps) to **TSDF volumetric fusion** (which only ever
records what was actually observed along real sensor rays), and adds
photographic RGB colour where the camera saw a point, instead of grayscale
LiDAR intensity only.

**Recommended output:** [`outputs/coverage1_finer_mesh.ply`](outputs/coverage1_finer_mesh.ply)
(binary PLY with per-vertex RGB, 5,855,301 vertices / 10,943,452 triangles in
the current build — see `outputs/finer_map_report.json` for the exact figures
behind whatever is currently in `outputs/`). OBJ and STL are also exported;
STL has no portable vertex-colour support, use the PLY for the coloured
result. [`outputs/preview.png`](outputs/preview.png) is a static top-down
scatter render (`make_preview.py`) for a quick look without a mesh viewer.
See "Visualizing the output" below for how to view either.

## Why this needed raw data, not GLIM's saved map

| | `coverage1_edited` (Poisson) | `finer_mapping` (TSDF, this pipeline) |
|---|---|---|
| Point source | GLIM's saved, decimated submap points | Raw `/ouster/points` from the bag |
| Points used | 149,813 | tens of millions (every valid raw return) |
| Median point spacing | ~6.35 cm | limited by the sensor's own angular resolution, well under 6.35 cm at typical room range |
| Colour | LiDAR intensity (grayscale) only | Camera RGB where visible, intensity grayscale fallback |
| Reconstruction | Screened Poisson (PCL) | TSDF volumetric fusion (custom, this pipeline) |
| Extrapolation behaviour | Can bridge gaps with a plausible-looking but unmeasured surface (trimmed afterwards to 20 cm of source data) | Only ever produces surface within one voxel of somewhere a ray actually terminated |

GLIM's own map storage is intentionally decimated for real-time SLAM, so
`coverage1_edited`'s 6.35 cm spacing is a hard ceiling on Poisson's output no
matter how the mesh is post-processed. Rebuilding from the raw bag removes
that ceiling; the sensor's own angular resolution becomes the limit instead.

## Source data

- **Bag:** `/home/autosweep/autosweep/dataset22jul/coverage1/coverage1.db3`
  (identified as the source for `coverage1_edited` by matching epoch
  timestamps: the bag starts at 1784717473.91s and `traj_lidar.txt`'s samples
  fall inside the bag's [start, start+311s] window).
- **Trajectory:** `coverage1_edited/traj_lidar.txt` — GLIM's global-mapping
  (loop-closure-corrected) LiDAR trajectory, TUM format (`t x y z qx qy qz
  qw`), 2,466 poses at ~100 ms spacing covering a continuous 300.5 s span
  with no gap over 0.5 s.
- **Calibration:** `coverage1_edited/config/config_sensors.json` —
  `T_lidar_camera` extrinsic, camera intrinsics/distortion
  (`rational_polynomial`, 8 coefficients, OpenCV-compatible order). Verified
  against the bag's own `/tf_static` (`T_lidar_imu` matches the recorded
  `os_sensor -> os_imu` transform exactly) and against GLIM's own
  `image_frame: rgb_camera_link` convention.
- **Sensor:** Ouster LiDAR at `/ouster/points` (organized `PointCloud2`,
  64 rings x 1024 columns/revolution, ~10 Hz, fields include `x y z
  intensity t ring range`) and an RGB camera at `/rgb/image_raw`
  (2048x1536, `bgra8`, ~9.3 Hz).

Only points whose timestamp falls inside the trajectory's covered time range
are used — the raw bag starts about 10.6 s before GLIM's first solved pose
(before enough data had accumulated to initialize global mapping), and those
early scans are skipped rather than extrapolated.

## Pipeline (`build_finer_map.py`)

### 1. Trajectory interpolation (`src/trajectory.py`)

`traj_lidar.txt` gives `T_world_lidar` (world <- LiDAR/`os_sensor`) at ~100 ms
steps. Any in-between pose is obtained by linearly interpolating position and
spherically interpolating (slerp) orientation between the two bracketing
samples — accurate here because 100 ms is already fine compared to how fast
the platform moves.

### 2. Per-column deskewing, not per-scan (`src/ouster_scan.py`, `build_finer_map.py`)

A LiDAR "scan" is a full 360 deg revolution, so its first and last points
are up to ~100 ms apart — assigning every point in a scan the same pose
(what GLIM's own saved keyframe poses effectively give you) smears moving-
platform motion into the geometry. This pipeline instead reads the Ouster
`t` field (a per-point capture-time offset within the scan) to interpolate a
*separate* trajectory pose for every column. Since Ouster fires all 64 beams
in a column simultaneously (confirmed directly against the raw bag: the `t`
field is identical across all rows for a fixed column), this only costs one
trajectory interpolation per column (1024) rather than per point (~65,000).

### 3. Organized-cloud preprocessing (`src/ouster_scan.py`)

Ouster's `PointCloud2` is published *organized*: message height = 64 equals
the ring count, and row index equals the `ring` field exactly (also checked
directly). That structure is used for two things a plain unordered point
cloud can't do cheaply:

- **Edge/mixed-pixel filtering** — at a depth discontinuity (e.g. a doorframe
  in front of a far wall), some LiDAR returns fall roughly halfway between
  the near and far surface and don't correspond to anything real. Comparing
  each point's range to its immediate azimuth neighbours *within the same
  ring* (an O(1) array lookup thanks to the organized layout, no KNN search
  needed) flags and drops these before they corrupt the TSDF.
- **Incidence-angle weighting** — a cheap per-point normal from
  finite-differencing neighbouring beams (row and column direction) gives an
  estimate of how face-on each return was. Grazing-angle returns (surface
  nearly parallel to the ray) are noisier and get down-weighted in the TSDF
  fusion (clipped to [0.2, 1.0], never dropped outright).

Invalid returns (`range == 0`, arriving as NaN x/y/z from the ROS driver)
are always excluded.

**Per-ring min-range for the bottom rings.** Ring index correlates directly
with beam elevation: sampling `coverage1_lipede` every 15th scan, per-ring
median range falls monotonically from ~9.9 m at ring 0 (near-horizontal) to
~1.75 m at ring 63 (steepest downward) — expected, since the steepest beams
hit the floor closest to the sensor. Within rings 60-63 specifically, a
narrow band of ~120 `(ring, column)` cells sits noticeably closer than even
that ring's own floor baseline (e.g. ring 63's floor median is 1.75 m, but a
cluster there sits at 0.35-0.55 m) and recurs at nearly the same azimuths in
80-90%+ of scans - almost certainly the robot's own chassis/wheel/brush
rather than mapped environment, since real scene geometry wouldn't stay
that close at a fixed bearing for the entire trajectory. The global
`--min-range` (0.5 m) sits *inside* that cluster's range, so part of it
(0.5-0.66 m) survived filtering and got fused every scan - and because a
near-constant LiDAR-local offset deskews to world coordinates as
`R @ small_offset + t`, it stayed close to the sensor position `t` itself,
tracing something close to the trajectory shape in the output mesh.
`--bottom-rings`/`--bottom-ring-min-range` raises the cutoff to 0.8 m for
just the last 4 rings (clearing the observed cluster with margin) rather
than raising `--min-range` globally, which would also discard legitimate
close-up detail the upper rings pick up near walls and furniture.

### 4. Colourisation, RGB-first (`src/colorizer.py`)

`cache_frames()` walks the bag once and saves every `/rgb/image_raw` frame to
disk as JPEG plus a timestamp index (`cache/frames/`, `cache/timestamps.npy`)
so the fusion pass never holds more than a handful of full-resolution frames
in memory.

For each scan column, the camera frame closest in time (within
`--max-image-dt`, default 90 ms) is looked up. Deskewed world-frame points
are projected into that frame using the calibrated extrinsic
(`T_lidar_camera`) evaluated at *the image's own capture time* (not the
LiDAR column's time — the camera moved between those two instants too) and
the rational-polynomial distortion model, via `cv2.projectPoints` directly
against the raw (not undistorted) image. A per-scan z-buffer keeps only the
nearest-camera-depth point per output pixel — since the LiDAR and camera are
rigidly co-mounted, a single scan's own points already share close to the
matched image's viewpoint, so this catches ordinary self-occlusion (e.g. a
near wall vs. the corridor behind it) without needing a full-scene renderer.
It will *not* catch occlusion from geometry outside the current scan (e.g. a
thin object mapped in an earlier scan that no longer occludes now) — a known,
documented limitation, not a silent one.

Points with no camera frame close enough in time, that land outside the
image, are behind the camera, or lose the z-buffer test fall back to a
percentile-normalised (1st/99th) grayscale of LiDAR intensity, using the same
convention as `coverage1_edited/color_intensity.py`. In the run that produced
the current output, 16.9% of fused points were coloured from RGB (see
`points_coloured_from_rgb_pct` in `outputs/finer_map_report.json` for
whatever the most recent run actually measured) — the camera has a much
narrower field of view than the LiDAR's full 360 deg sweep, so most points
simply never appear in any frame; this is expected, not a bug.

### 5. TSDF volumetric fusion (`src/tsdf_volume.py`)

Classic TSDF fusion (KinectFusion, Open3D's `ScalableTSDFVolume`) integrates
a *pinhole depth image* every frame — that doesn't apply here since a
spinning LiDAR scan is a spherical sweep, not a pinhole image. This
implements the point-cloud generalisation instead (the same idea tools like
VDBFusion are built around, reimplemented here directly in NumPy since no
prebuilt OpenVDB-based Python package was installable in this environment):
for every point `p` measured from sensor origin `o`, the ray direction
`d = (p - o) / |p - o|` defines a 1-D signed-distance axis, and a short band
of voxels around `p` (+/- `truncation_distance`, default 4 voxels) is updated
with

```
sdf(voxel) = dot(p - voxel_center, d)
```

— positive between the sensor and the surface (free space), negative beyond
it (occluded) — combined across scans with a running incidence-weighted
average. This only touches voxels near an actual measured surface (a dense
NumPy grid stays tractable at room scale without an octree/hash structure),
and, unlike Poisson, never invents surface far from anywhere a ray actually
terminated.

The zero level set is extracted with marching cubes (`scikit-image`),
restricted to voxels with enough accumulated weight (`--min-weight`, plus a
1-voxel dilation so cubes straddling that boundary are still evaluated) —
unobserved voxels are never treated as valid surface data. Per-vertex colour
is sampled from the corresponding voxel's accumulated colour average, and the
mesh is cleaned and given consistent normals with VTK before export
(`vtkCleanPolyData` + `vtkPolyDataNormals`, matching the finishing step
`coverage1_edited/finish_mesh.py` uses).

## Parameters (`build_finer_map.py` CLI)

| Flag | Default | Meaning |
|---|---|---|
| `--voxel-size` | 0.03 m | TSDF voxel edge length — the main detail/memory/runtime dial |
| `--truncation` | 4 x voxel size | Half-width of the band around each surface point that gets updated |
| `--min-range` / `--max-range` | 0.5 / 15.0 m | Discard returns outside this range |
| `--bottom-rings` / `--bottom-ring-min-range` | 4 / 0.8 m | Raise the min-range for the steepest-downward-looking rings only (see below) |
| `--edge-jump` | 0.3 m | Within-ring range-discontinuity threshold for edge/mixed-pixel filtering |
| `--padding` | 6.0 m | Metres of TSDF volume padding around the trajectory bounding box |
| `--max-image-dt` | 0.09 s | Max time gap between a scan column and the camera frame used to colour it |
| `--min-weight` | 1.5 | Minimum accumulated TSDF weight for a voxel to count as observed |
| `--limit-scans` | none | Debug: process only the first N scans |
| `--skip-frame-cache` | off | Reuse an existing `cache/` frame dump instead of re-extracting it |

**Voxel size is a memory trade-off**, since the TSDF grid is dense (no
octree): this run's bounding box needed ~380M voxels at 3 cm, ~5 float32
arrays each -> a few GB of RAM. Halving the voxel size multiplies the voxel
count (and memory) by roughly 8x — see `finer_map_report.json`'s
`grid_dims`/`total_voxels` for the actual numbers from this run before
lowering `--voxel-size` further.

## Running it

```bash
cd /home/autosweep/finer_mapping
source .venv/bin/activate      # created with: python3 -m venv .venv --system-site-packages
python3 build_finer_map.py
```

`.venv` was created with `--system-site-packages` so it inherits this
machine's ROS 2 Jazzy install (`rclpy`, `rosbag2_py`, `sensor_msgs_py`,
`cv_bridge`) and system OpenCV/VTK, adding only `scikit-image` on top (see
`requirements.txt`). It is *not* portable to another machine's Python/ROS
install; recreate it there instead of copying it:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
```

`cache/` (camera frame JPEGs + timestamps, ~900 MB) is produced by the first
run's pass 1 and reused on subsequent runs via `--skip-frame-cache`; delete
it to force re-extraction (e.g. after pointing `--bag` at different data).

## Visualizing the output

**Full interactive mesh (recommended):** MeshLab is installed on this
machine and reads the PLY's per-vertex RGB directly.

```bash
meshlab outputs/coverage1_finer_mesh.ply
```

`meshlabserver` is also available for headless/scripted use (e.g. batch
screenshots or format conversion via a MeshLab `.mlx` script) if a GUI isn't
available.

**Quick static preview, no mesh viewer needed:** `make_preview.py` renders a
random 400k-vertex subsample as a coloured 3D scatter plot with matplotlib
and writes `outputs/preview.png`. It hardcodes both paths
(`outputs/coverage1_finer_mesh.ply` in, `outputs/preview.png` out, relative
to the script), so it takes no arguments and must be re-run after any new
`build_finer_map.py` run to refresh the preview:

```bash
cd /home/autosweep/finer_mapping
source .venv/bin/activate
python3 make_preview.py
```

This is a scatter of raw vertices, not the triangulated surface, so it's
faster than loading the full mesh but won't show shading/normals - use
MeshLab for an actual look at surface quality.

## Output inventory

| File | Purpose |
|---|---|
| `outputs/coverage1_finer_mesh.ply` | Recommended: full mesh with per-vertex RGB |
| `outputs/coverage1_finer_mesh.obj` | Geometry-only OBJ alternative |
| `outputs/coverage1_finer_mesh.stl` | Geometry-only STL; assume metres, no colour |
| `outputs/finer_map_report.json` | Parameters used, scan/point counts, RGB coverage %, mesh/grid statistics |
| `outputs/preview.png` | Static top-down scatter preview, coloured by actual vertex RGB/intensity |
| `outputs/build.log` | Console log from the run that produced the current `outputs/` contents |
| `cache/frames/*.jpg`, `cache/timestamps.npy` | Cached camera frames (pass 1 output, reused across runs) |
| `make_preview.py` | Regenerates `outputs/preview.png` from the current mesh |
| `src/trajectory.py` | GLIM trajectory loading and pose interpolation |
| `src/extrinsics.py` | Sensor calibration loading (`config_sensors.json` -> matrices) |
| `src/ouster_scan.py` | Raw Ouster scan parsing, edge filtering, organized-cloud normals |
| `src/colorizer.py` | Camera frame caching and RGB/intensity point colourisation |
| `src/tsdf_volume.py` | The TSDF grid: integration and marching-cubes mesh extraction |
| `src/bagio.py` | Small rosbag2 reading helpers |
| `build_finer_map.py` | Driver script tying the above together |

## Known limitations

- **Not watertight**, same as `coverage1_edited`: this reconstructs a scanned
  scene surface, not a solid. Boundaries exist wherever the sensor never saw
  the far side of something.
- **Occlusion handling only within a single scan's own points** (see step 4
  above) — colour can very occasionally leak between a foreground and
  background surface that a *different* scan, not the one being coloured,
  observed at that pixel.
- **Trajectory-bounded**: points outside `traj_lidar.txt`'s covered time
  range are dropped rather than pose-extrapolated, so the very start/end of
  the bag (before GLIM had enough data to solve a global pose) is excluded.
- **Dense (non-hashed) voxel grid**: workable at this room's scale at 3 cm
  voxels; pushing well past that resolution or covering a much larger space
  would need an octree/hashed TSDF (e.g. a proper OpenVDB-backed tool) to
  stay memory-tractable — see `--voxel-size` above.
- **Camera FOV is narrower than the LiDAR's**, so a large fraction of the
  mesh is coloured from LiDAR intensity, not RGB — check
  `finer_map_report.json`'s `points_coloured_from_rgb_pct` for the actual
  figure from a given run.
