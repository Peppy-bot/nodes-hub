"""The physics a stage simulates on, against real USD: the CPU pipeline, CPU
dynamics with the MBP broadphase, authored on the stage's physics scene, which
is the pipeline Isaac Sim pairs with the CPU device the engine's articulation
views read and write on."""

from pxr import Usd, UsdPhysics

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
