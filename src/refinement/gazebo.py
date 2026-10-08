"""Gazebo: package the visual and collision meshes as a Gazebo model.

Writes, under outputs/<name>/gazebo/:
  <name>_map/model.config, <name>_map/model.sdf  a static model whose single
      link uses meshes/visual.obj as visual and meshes/collision.stl as
      collision (mesh URIs relative to the SDF file),
  <name>_world.sdf  a world containing that model, ambient-lit: the
      enclosure's roof would put the whole map in the sun's shadow, so the
      sun casts no shadows and the scene's ambient light is high.
Load it with `gz sim outputs/<name>/gazebo/<name>_world.sdf`, or add the
gazebo/ directory to GZ_SIM_RESOURCE_PATH and include model://<name>_map.
"""
from .base import RefinementStep
from .texture import model_dir

MODEL_SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}_map">
    <static>true</static>
    <link name="map">
{visual}{collision}    </link>
  </model>
</sdf>
"""
VISUAL = """      <visual name="visual">
        <geometry><mesh><uri>{uri}</uri></mesh></geometry>
      </visual>
"""
COLLISION = """      <collision name="collision">
        <geometry><mesh><uri>{uri}</uri></mesh></geometry>
      </collision>
"""
MODEL_CONFIG = """<?xml version="1.0"?>
<model>
  <name>{name}_map</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Static map of {name} from finer_mapping (build_finer_map.py + map_refinement.py):
textured visual mesh and coarse collision mesh.</description>
</model>
"""
WORLD_SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{name}">
    <physics name="default" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <scene>
      <ambient>0.85 0.85 0.85 1</ambient>
      <background>0.6 0.6 0.6 1</background>
      <shadows>false</shadows>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.4 0.4 0.4 1</diffuse>
      <specular>0.05 0.05 0.05 1</specular>
      <direction>-0.3 0.2 -0.9</direction>
    </light>
    <include>
      <uri>{model_dir}</uri>
    </include>
  </world>
</sdf>
"""


class GazeboStep(RefinementStep):
    name = "gazebo"
    help = "write a Gazebo model (model.sdf/model.config) and a world around the meshes"

    def run(self, mesh, ctx):
        mdir = model_dir(ctx)
        mdir.mkdir(parents=True, exist_ok=True)
        visual, collision = ctx.state.get("visual_mesh"), ctx.state.get("collision_mesh")
        if visual is None and collision is None:
            raise RuntimeError("gazebo: run the texture and/or collision steps first")
        rel = lambda p: p.relative_to(mdir).as_posix()  # noqa: E731
        (mdir / "model.sdf").write_text(MODEL_SDF.format(
            name=ctx.name, visual=VISUAL.format(uri=rel(visual)) if visual else "",
            collision=COLLISION.format(uri=rel(collision)) if collision else ""))
        (mdir / "model.config").write_text(MODEL_CONFIG.format(name=ctx.name))
        world = mdir.parent / f"{ctx.name}_world.sdf"
        world.write_text(WORLD_SDF.format(name=ctx.name, model_dir=mdir.resolve().as_uri()))
        ctx.log(f"gazebo: model {mdir}, world {world}")
        return {"model_dir": str(mdir), "world": str(world)}
