"""Shared state and the step interface of the map refinement pipeline.

Every step receives the same `MapMesh` and `RefinementContext`, edits the
mesh in place and returns a JSON-serialisable dict for the report. Faces carry
a `part` label so a step can leave alone what an earlier one produced (e.g. a
future decimation step skipping the synthetic floor or the enclosure).
"""
import numpy as np

PART_FUSED = 0       # marching-cubes geometry from the TSDF
PART_FLOOR = 1       # the build's synthetic floor
PART_ENCLOSURE = 2   # walls / roof added by the enclosure step
PART_NAMES = {PART_FUSED: "fused", PART_FLOOR: "floor", PART_ENCLOSURE: "enclosure"}


class MapMesh:
    """Triangle mesh: verts (N,3) float64, faces (M,3) int64, colors (N,3)
    uint8 RGB and a per-face part label (M,) int8."""

    def __init__(self, verts, faces, colors, parts):
        self.verts = np.asarray(verts, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.colors = np.asarray(colors, dtype=np.uint8)
        self.parts = np.asarray(parts, dtype=np.int8)

    def part_faces(self, part):
        return self.faces[self.parts == part]

    def submesh(self, mask):
        """(verts, faces, colors, vertex index into self) of the faces in
        boolean mask `mask`, compacted to the vertices they use."""
        faces = self.faces[mask]
        used = np.unique(faces)
        remap = np.full(len(self.verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return self.verts[used], remap[faces], self.colors[used], used

    def append(self, verts, faces, colors, part):
        """Add a separate piece; `faces` index into its own `verts`."""
        self.faces = np.concatenate([self.faces, np.asarray(faces, dtype=np.int64) + len(self.verts)])
        self.verts = np.concatenate([self.verts, verts])
        self.colors = np.concatenate([self.colors, np.asarray(colors, dtype=np.uint8)])
        self.parts = np.concatenate([self.parts, np.full(len(faces), part, dtype=np.int8)])

    def add_faces(self, faces, part):
        """Add faces between existing vertices."""
        self.faces = np.concatenate([self.faces, np.asarray(faces, dtype=np.int64)])
        self.parts = np.concatenate([self.parts, np.full(len(faces), part, dtype=np.int8)])

    def keep_faces(self, keep):
        """Keep only the faces in boolean mask `keep`, dropping vertices no
        longer used."""
        self.faces, self.parts = self.faces[keep], self.parts[keep]
        used = np.zeros(len(self.verts), dtype=bool)
        used[self.faces.ravel()] = True
        remap = np.cumsum(used) - 1
        self.faces = remap[self.faces]
        self.verts, self.colors = self.verts[used], self.colors[used]

    def summary(self):
        return {"vertices": int(len(self.verts)), "triangles": int(len(self.faces)),
                **{f"{name}_triangles": int((self.parts == p).sum()) for p, name in PART_NAMES.items()}}


class RefinementContext:
    """What steps share besides the mesh: `floor` is the build's floor model
    already moved into the mesh's (output) frame, `build_report` the build's
    finer_map_report.json, `out_dir` where steps write their files, `name`
    the map's name, and `state` a dict through which a step hands results to
    later ones (e.g. the enclosure's corners, read by clip/floor/texture)."""

    def __init__(self, name, floor, build_report, out_dir, log):
        self.name = name
        self.floor = floor
        self.build_report = build_report
        self.out_dir = out_dir
        self.log = log
        self.state = {}

    def require(self, key, step):
        if key not in self.state:
            raise RuntimeError(f"{step}: needs '{key}' from an earlier step (check --steps order)")
        return self.state[key]


class RefinementStep:
    """One pipeline step. Subclasses set `name` (used on the command line and
    as the CLI option prefix) and `help`, declare their options in
    `add_args` (prefix them with --<name>- to keep steps independent) and do
    their work in `run`."""

    name = None
    help = ""

    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, args):
        self.args = args

    def run(self, mesh, ctx):
        """Edit `mesh` in place; return a dict for the report."""
        raise NotImplementedError
