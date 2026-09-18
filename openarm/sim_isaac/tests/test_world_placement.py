"""Where a robot stands on the stage, against real USD: the order an xform's
ops are listed in decides where a turned robot ends up, and a referenced
model brings its own ops."""

import math
import sys
from types import ModuleType

import pytest
from pxr import Gf, Usd, UsdGeom

from _world import world_module as _world_module

QUARTER_TURN = math.pi / 2


@pytest.fixture(name="place")
def _place():
    return _world_module().World._place


def _prim(carries_orient: bool):
    """A prim as a reference leaves it: some models carry an orient op of
    their own and no translate. The stage comes back with it, because a prim
    outlives nothing."""
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/openarm/alpha", "Xform")
    if carries_orient:
        UsdGeom.Xformable(prim).AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
    return stage, prim


def _base_of(prim):
    """Where the robot's own origin ends up in the world."""
    local = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
    return local.Transform(Gf.Vec3d(0, 0, 0))


@pytest.mark.parametrize("carries_orient", [False, True], ids=["bare", "model-has-orient"])
def test_a_turned_robot_stands_where_it_was_placed(place, carries_orient):
    _stage, prim = _prim(carries_orient)

    place(prim, (1.0, 0.0, 0.0), QUARTER_TURN)

    base = _base_of(prim)
    assert base[0] == pytest.approx(1.0, abs=1e-6), base
    assert base[1] == pytest.approx(0.0, abs=1e-6), base


@pytest.mark.parametrize("carries_orient", [False, True], ids=["bare", "model-has-orient"])
def test_the_robot_turns_about_its_own_base(place, carries_orient):
    _stage, prim = _prim(carries_orient)

    place(prim, (1.0, 0.0, 0.0), QUARTER_TURN)

    # A point a metre ahead of the base swings to a metre to its left.
    local = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
    ahead = local.Transform(Gf.Vec3d(1, 0, 0))
    assert ahead[0] == pytest.approx(1.0, abs=1e-6), ahead
    assert ahead[1] == pytest.approx(1.0, abs=1e-6), ahead


def test_the_placement_ops_lead_whatever_the_model_carried(place):
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/openarm/alpha", "Xform")
    xform = UsdGeom.Xformable(prim)
    xform.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
    xform.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 2.0))

    place(prim, (0.0, 0.0, 0.0), 0.0)

    assert [op.GetOpName() for op in xform.GetOrderedXformOps()] == [
        "xformOp:translate",
        "xformOp:orient",
        "xformOp:scale",
    ]


class _FakeStage:
    """Enough of a stage for World.move: it hands back the prim it was
    given, as the live stage does."""

    def __init__(self, prims):
        self._prims = prims

    def GetPrimAtPath(self, path):  # noqa: N802  (USD's own spelling)
        return self._prims[path]


def test_a_move_carries_the_record_with_the_prim(monkeypatch, place):
    world_module = _world_module()
    stage = Usd.Stage.CreateInMemory()
    alpha = stage.DefinePrim("/World/alpha", "Xform")
    bravo = stage.DefinePrim("/World/bravo", "Xform")

    world = world_module.World(catalogue=object())
    at = world_module.Placement.of((0.0, 0.0, 0.0), 0.0)
    world._robots = {
        "alpha": world_module.Robot(instance="alpha", model="openarm_v2", placement=at),
        "bravo": world_module.Robot(instance="bravo", model="openarm_v1", placement=at),
    }
    fake_usd = ModuleType("omni.usd")
    fake_usd.get_context = lambda: type(
        "Ctx", (), {"get_stage": staticmethod(lambda: _FakeStage({"/World/alpha": alpha, "/World/bravo": bravo}))}
    )()
    monkeypatch.setitem(sys.modules, "omni.usd", fake_usd)
    # `import omni.usd` then reads it off the parent package.
    monkeypatch.setattr(sys.modules["omni"], "usd", fake_usd, raising=False)

    moved = world.move("alpha", (2.0, -1.0, 0.0))

    # The listing follows the prim, so the scene does not report a stale spot.
    assert moved.placement.position == (2.0, -1.0, 0.0)
    assert world._robots["alpha"].placement.position == (2.0, -1.0, 0.0)
    assert _base_of(alpha)[0] == pytest.approx(2.0, abs=1e-6)
    # The robot that was not asked to move stands where it was.
    assert world._robots["bravo"].placement.position == (0.0, 0.0, 0.0)


def test_moving_a_robot_that_does_not_stand_says_so(monkeypatch):
    world_module = _world_module()
    world = world_module.World(catalogue=object())
    with pytest.raises(KeyError, match="charlie"):
        world.move("charlie", (0.0, 0.0, 0.0))


def test_a_spot_promised_to_an_admitted_robot_is_not_offered_twice():
    """Standing a robot happens later, on the thread that steps the scene, so
    between admitting two robots the first is not standing anywhere yet."""
    world_module = _world_module()
    world = world_module.World(catalogue=object())

    first = world.free_spot()
    second = world.free_spot(promised=(first,))

    assert second.position != first.position
    assert world.occupied(first, promised=(first,))
    assert not world.occupied(second, promised=(first,))


def test_the_listing_waits_while_a_robot_is_being_stood():
    """Robots are stood on the thread that steps the scene and listed on the
    one serving the contracts, and the listing waits for a stand to finish
    before it reads the stage."""
    import threading

    world_module = _world_module()
    world = world_module.World(catalogue=object())
    at = world_module.Placement.of((0.0, 0.0, 0.0), 0.0)

    inside = threading.Event()
    finish = threading.Event()

    def slow_add():
        with world._lock:
            inside.set()
            finish.wait(5)
            world._robots["alpha"] = world_module.Robot(
                instance="alpha", model="openarm_v2", placement=at
            )

    stepping = threading.Thread(target=slow_add)
    stepping.start()
    assert inside.wait(5), "the stepping thread never took the world"

    listed = []
    reader = threading.Thread(target=lambda: listed.append(world.robots()))
    reader.start()
    reader.join(0.2)
    assert reader.is_alive(), "the listing read the world mid-change"

    finish.set()
    stepping.join(5)
    reader.join(5)
    assert [robot.instance for robot in listed[0]] == ["alpha"]


@pytest.mark.parametrize(
    "precision",
    [
        UsdGeom.XformOp.PrecisionDouble,
        UsdGeom.XformOp.PrecisionFloat,
        UsdGeom.XformOp.PrecisionHalf,
    ],
    ids=["double", "float", "half"],
)
def test_a_model_turns_whatever_width_its_orient_op_was_authored_at(place, precision):
    """USD refuses a quaternion of the wrong width, and a referenced model
    brings whichever its author chose."""
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/openarm/alpha", "Xform")
    UsdGeom.Xformable(prim).AddOrientOp(precision)

    place(prim, (1.0, 0.0, 0.0), QUARTER_TURN)

    local = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
    ahead = local.Transform(Gf.Vec3d(1, 0, 0))
    assert ahead[0] == pytest.approx(1.0, abs=1e-2), ahead
    assert ahead[1] == pytest.approx(1.0, abs=1e-2), ahead
