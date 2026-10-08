"""Map refinement steps (see map_refinement.py). STEPS is the default
pipeline, in order; add a new step by subclassing `RefinementStep` and
listing it here."""
from .base import MapMesh, RefinementContext, RefinementStep  # noqa: F401
from .cleanup import CleanupStep
from .clip import ClipStep
from .collision import CollisionStep
from .enclosure import EnclosureStep
from .floor_fill import FloorStep
from .gazebo import GazeboStep
from .texture import TextureStep

STEPS = [CleanupStep, EnclosureStep, FloorStep, ClipStep, TextureStep, CollisionStep, GazeboStep]
