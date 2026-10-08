"""Clip: drop everything outside the enclosure.

Faces (other than the enclosure's own) whose centroid lies more than
`margin` outside the enclosure's quadrilateral are removed: the corridor and
rooms seen through doors, door spikes of the floor. Invisible from inside
the walls anyway, they would still cost triangles and exist for physics.
The margin keeps objects that touch a wall from being cut flush with it.
"""
from .base import PART_ENCLOSURE, RefinementStep
from .geometry import inside_quad


class ClipStep(RefinementStep):
    name = "clip"
    help = "remove geometry outside the enclosure"

    @classmethod
    def add_args(cls, p):
        g = p.add_argument_group("clip step")
        g.add_argument("--clip-margin", type=float, default=0.1,
                       help="keep faces up to this many metres outside the walls")

    def run(self, mesh, ctx):
        corners = ctx.require("enclosure", self.name)["corners"]
        centroid = mesh.verts[mesh.faces].mean(axis=1)
        keep = inside_quad(centroid[:, :2], corners, self.args.clip_margin) | (mesh.parts == PART_ENCLOSURE)
        removed = int((~keep).sum())
        mesh.keep_faces(keep)
        ctx.log(f"clip: removed {removed:,} triangles more than {self.args.clip_margin} m outside the walls")
        return {"margin_m": self.args.clip_margin, "triangles_removed": removed}
