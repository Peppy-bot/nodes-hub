"""The physics a stage simulates on, against real USD: the CPU pipeline, CPU
dynamics with the MBP broadphase, authored on the stage's physics scene, which
is the pipeline Isaac Sim pairs with the CPU device the engine's articulation
views read and write on, and the floor its robots stand on while no scene
brings its own."""

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from _world import world_module


def test_an_empty_stage_is_given_a_physics_scene_on_the_cpu_pipeline():
    world = world_module()
    stage = Usd.Stage.CreateInMemory()

    scene = world.World._physics_scene(stage)

    assert scene.GetPath() == f"{world.WORLD_PRIM}/physicsScene"
    assert scene.IsA(UsdPhysics.Scene)
    # usd-core does not register PhysX's schemas, so the applied schema is read
    # as authored: the list Isaac Sim composes its schemas from.
    assert world.PHYSX_SCENE_API in scene.GetMetadata("apiSchemas").ApplyOperations([])
    assert scene.GetAttribute(world.GPU_DYNAMICS).Get() is False
    assert scene.GetAttribute(world.BROADPHASE).Get() == "MBP"


def test_a_stage_carrying_a_physics_scene_runs_that_scene_on_the_cpu_pipeline():
    world = world_module()
    stage = Usd.Stage.CreateInMemory()
    carried = UsdPhysics.Scene.Define(stage, "/PhysicsScene").GetPrim()

    scene = world.World._physics_scene(stage)

    assert scene.GetPath() == carried.GetPath()
    assert [prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)] == [carried.GetPath()]
    assert scene.GetAttribute(world.GPU_DYNAMICS).Get() is False
    assert scene.GetAttribute(world.BROADPHASE).Get() == "MBP"


def test_the_stage_stands_its_robots_on_a_floor_at_z_0():
    world = world_module()
    stage = Usd.Stage.CreateInMemory()

    floor = world.World.lay_floor(stage)

    assert floor.GetPath() == world.GROUND_PRIM
    plane = UsdGeom.Plane(floor)
    assert plane.GetAxisAttr().Get() == "Z"
    assert UsdGeom.Xformable(floor).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) == Gf.Matrix4d(1.0)
    assert floor.HasAPI(UsdPhysics.CollisionAPI)
    assert not floor.HasAPI(UsdPhysics.RigidBodyAPI)
    assert plane.GetPurposeAttr().Get() == "guide"


def test_a_stage_needs_its_floor_while_no_scene_stands_on_it():
    world = world_module()
    stage = Usd.Stage.CreateInMemory()

    assert world.World.needs_floor(stage.GetPrimAtPath("/World/RuntimeScene"))


def test_a_scene_carrying_no_collider_brings_no_floor():
    world = world_module()
    stage = Usd.Stage.CreateInMemory()
    scene = stage.DefinePrim("/World/RuntimeScene", "Xform")
    UsdGeom.Mesh.Define(stage, "/World/RuntimeScene/backdrop")

    assert world.World.needs_floor(scene)


def test_a_scene_carrying_a_collider_brings_its_own_floor():
    world = world_module()
    stage = Usd.Stage.CreateInMemory()
    scene = stage.DefinePrim("/World/RuntimeScene", "Xform")
    floor = UsdGeom.Mesh.Define(stage, "/World/RuntimeScene/Environment/floor")
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    assert not world.World.needs_floor(scene)


def test_a_collider_inside_an_instance_brings_its_own_floor():
    """A referenced environment keeps its geometry in instances, whose
    prims a plain traversal does not visit."""
    world = world_module()
    stage = Usd.Stage.CreateInMemory()
    prototype = stage.DefinePrim("/Prototypes/tile", "Xform")
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, "/Prototypes/tile/collider").GetPrim())
    scene = stage.DefinePrim("/World/RuntimeScene", "Xform")
    tile = stage.DefinePrim("/World/RuntimeScene/tile", "Xform")
    tile.GetReferences().AddInternalReference(prototype.GetPath())
    tile.SetInstanceable(True)

    assert not world.World.needs_floor(scene)
