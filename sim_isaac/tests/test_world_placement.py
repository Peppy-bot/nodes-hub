"""Where a robot stands on the stage, against real USD: the order an xform's
ops are listed in decides where a turned robot ends up, and a referenced
model brings its own ops."""

import math
import sys
from types import ModuleType

import pytest
from pxr import Gf, Usd, UsdGeom

from _world import known
from _world import world_module as _world_module

QUARTER_TURN = math.pi / 2


@pytest.fixture(name="place")
def _place():
    return _world_module().World.place


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

    world = world_module.World(head_camera_pack=None)
    at = world_module.Placement.of((0.0, 0.0, 0.0), 0.0)
    world._robots = {
        "alpha": world_module.Robot(instance="alpha", known=known("openarm_v2"), placement=at),
        "bravo": world_module.Robot(instance="bravo", known=known("so101"), placement=at),
    }
    fake_usd = ModuleType("omni.usd")
    fake_usd.get_context = lambda: type(
        "Ctx", (), {"get_stage": staticmethod(lambda: _FakeStage({"/World/alpha": alpha, "/World/bravo": bravo}))}
    )()
    monkeypatch.setitem(sys.modules, "omni.usd", fake_usd)
    # `import omni.usd` then reads it off the parent package.
    monkeypatch.setattr(sys.modules["omni"], "usd", fake_usd, raising=False)

    moved = world.move("alpha", (2.0, -1.0, 0.0), QUARTER_TURN)

    # The listing follows the prim, so the scene does not report a stale spot
    # or a stale heading.
    assert moved.placement == world_module.Placement.of((2.0, -1.0, 0.0), QUARTER_TURN)
    assert world._robots["alpha"].placement == moved.placement
    # It is the robot it was, of the model it joined as.
    assert (moved.instance, moved.model) == ("alpha", "openarm_v2")
    assert _base_of(alpha)[0] == pytest.approx(2.0, abs=1e-6)
    assert _base_of(alpha)[1] == pytest.approx(-1.0, abs=1e-6)
    # The prim faces the yaw it was moved to: a point a metre ahead of the
    # base swings to a metre to its left.
    local = UsdGeom.Xformable(alpha).GetLocalTransformation(Usd.TimeCode.Default())
    ahead = local.Transform(Gf.Vec3d(1, 0, 0))
    assert ahead[0] == pytest.approx(2.0, abs=1e-6), ahead
    assert ahead[1] == pytest.approx(0.0, abs=1e-6), ahead
    # The robot that was not asked to move stands where it was, as it was.
    assert world._robots["bravo"].placement == at


def test_moving_a_robot_that_does_not_stand_says_so(monkeypatch):
    world_module = _world_module()
    world = world_module.World(head_camera_pack=None)
    with pytest.raises(KeyError, match="charlie"):
        world.move("charlie", (0.0, 0.0, 0.0), 0.0)


def test_a_spot_promised_to_an_admitted_robot_is_not_offered_twice():
    """Standing a robot happens later, on the thread that steps the scene, so
    between admitting two robots the first is not standing anywhere yet."""
    world_module = _world_module()
    world = world_module.World(head_camera_pack=None)

    first = world.free_spot()
    second = world.free_spot(promised=(first,))

    assert second.position != first.position
    assert world.occupied(first, promised=(first,))
    assert not world.occupied(second, promised=(first,))


@pytest.mark.parametrize("position", [(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)])
def test_a_placement_that_is_not_finite_is_refused(position):
    """A prim placed at NaN reports no error and simulates nothing."""
    world_module = _world_module()

    with pytest.raises(ValueError, match="finite in every coordinate"):
        world_module.Placement.of(position, 0.0)


def test_a_placement_of_the_wrong_width_is_refused():
    world_module = _world_module()

    with pytest.raises(ValueError, match="3 coordinates"):
        world_module.Placement.of((0.0, 0.0), 0.0)


def test_a_yaw_that_is_not_finite_is_refused():
    world_module = _world_module()

    with pytest.raises(ValueError, match="finite in every coordinate"):
        world_module.Placement.of((0.0, 0.0, 0.0), float("nan"))


def test_a_robot_with_no_name_is_refused():
    world_module = _world_module()

    with pytest.raises(ValueError, match="stands under the name of the copy it runs as"):
        world_module.name_in_the_stage("")


def test_a_name_standing_twice_is_refused():
    """Admission has already found the caller a name, so a second robot
    under it is the stage saying what the registry did not."""
    world_module = _world_module()
    world = world_module.World(head_camera_pack=None)
    spot = world_module.Placement.of((0.0, 0.0, 0.0), 0.0)
    world._robots["alpha"] = world_module.Robot(  # pylint: disable=W0212
        instance="alpha", known=known("openarm_v2"), placement=spot
    )

    with pytest.raises(ValueError, match=r"'alpha' is still on the stage"):
        world.add("alpha", known("openarm_v2"), spot)


def test_taking_out_a_robot_that_stands_nowhere_changes_nothing():
    """A take-out that raced a stage the robot never reached leaves the
    robots standing alone."""
    world_module = _world_module()
    world = world_module.World(head_camera_pack=None)
    world._robots["alpha"] = world_module.Robot(  # pylint: disable=W0212
        instance="alpha",
        known=known("openarm_v2"),
        placement=world_module.Placement.of((0.0, 0.0, 0.0), 0.0),
    )

    world.remove("ghost")

    assert [robot.instance for robot in world.robots()] == ["alpha"]


def test_the_robots_of_a_stage_are_listed_in_the_order_they_joined():
    """Every name a robot answers to carries its own, so the order is what
    a reader of `stack list` and of this engine's log sees."""
    world_module = _world_module()
    world = world_module.World(head_camera_pack=None)
    for name, spot in (("alpha", 0.0), ("bravo", 1.5), ("charlo", 3.0)):
        world._robots[name] = world_module.Robot(  # pylint: disable=W0212
            instance=name,
            known=known("openarm_v2"),
            placement=world_module.Placement.of((spot, 0.0, 0.0), 0.0),
        )

    assert [robot.instance for robot in world.robots()] == ["alpha", "bravo", "charlo"]


def test_a_robot_nearer_than_the_lattice_leaves_them_counts_as_on_the_spot():
    """It is the distance between two placements that counts, not which
    lattice square each falls in: two robots nearer together than the lattice
    leaves them resolve their overlap by throwing each other."""
    world_module = _world_module()
    world = world_module.World(head_camera_pack=None)
    pitch = world_module.SPOT_PITCH_M
    promised = world_module.Placement.of((pitch * 0.49, 0.0, 0.0), 0.0)

    # Hand-picked spots either side of a lattice line, 2 cm apart.
    beside = world_module.Placement.of((pitch * 0.51, 0.0, 0.0), 0.0)

    assert world.occupied(beside, promised=(promised,))
    # A spot a whole pitch from the one promised is free, wherever the
    # lattice's own lines fall between them.
    clear = world_module.Placement.of((pitch * 0.49 + pitch, 0.0, 0.0), 0.0)
    assert not world.occupied(clear, promised=(promised,))


def test_the_robot_standing_in_the_way_is_the_one_named():
    """A spot is held by a robot standing on it as much as by one promised
    it, and which of the two decides what the caller is told."""
    world_module = _world_module()
    world = world_module.World(head_camera_pack=None)
    spot = world_module.Placement.of((0.0, 0.0, 0.0), 0.0)
    world._robots["alpha"] = world_module.Robot(  # pylint: disable=W0212
        instance="alpha", known=known("openarm_v2"), placement=spot
    )

    assert world.standing_within(spot).instance == "alpha"
    assert world.occupied(spot)
    assert world.standing_within(
        world_module.Placement.of((world_module.SPOT_PITCH_M, 0.0, 0.0), 0.0)
    ) is None


def test_a_name_carrying_the_prim_separator_is_refused():
    """Every robot is a prim of the stage under its own name, so a name
    carrying the separator nests one robot inside another's path."""
    world_module = _world_module()

    with pytest.raises(ValueError, match=r"peppy stack join LAUNCHER -i left_arm"):
        world_module.name_in_the_stage("left/arm")

    assert world_module.name_in_the_stage("alpha") == "alpha"


def test_a_full_stage_is_refused_the_way_a_taken_spot_is(monkeypatch):
    """Admission answers the goal with the reason its robot was refused, and
    reads every one of those reasons off a ValueError. Two rings put a robot
    on the origin and on the eight spots around it."""
    world_module = _world_module()
    monkeypatch.setattr(world_module, "_MAX_RINGS", 2)
    world = world_module.World(head_camera_pack=None)
    pitch = world_module.SPOT_PITCH_M
    taken = tuple(
        world_module.Placement.of((row * pitch, column * pitch, 0.0), 0.0)
        for row in (-1, 0, 1)
        for column in (-1, 0, 1)
    )

    with pytest.raises(ValueError, match=r"every spot this stage lays out is taken"):
        world.free_spot(taken)


def test_the_listing_waits_while_a_robot_is_being_stood():
    """Robots are stood on the thread that steps the scene and listed on the
    one serving the contracts, and the listing waits for a stand to finish
    before it reads the stage."""
    import threading

    world_module = _world_module()
    world = world_module.World(head_camera_pack=None)
    at = world_module.Placement.of((0.0, 0.0, 0.0), 0.0)

    inside = threading.Event()
    finish = threading.Event()

    def slow_add():
        with world._lock:
            inside.set()
            finish.wait(5)
            world._robots["alpha"] = world_module.Robot(
                instance="alpha", known=known("openarm_v2"), placement=at
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
