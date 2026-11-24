"""The model aligner against fake senses, fake moves and a scripted step
model: what a step shows the model, each guard, the moves it sends, how
an alignment ends, and a property over any answers the model can give."""

import asyncio
import math
from collections import deque
from types import SimpleNamespace
from typing import Optional

import pytest
from conftest import limb_states_message, rgb_frame
from hypothesis import given, settings
from hypothesis import strategies as st
from test_loop import FakeStepModel

from brain_core.limb_state import NO_SNAPSHOT, LimbStateStore
from brain_core.manipulators.scripted.aligning import Aligned, AlignmentFailed, AlignRequest, GaveUp, OutOfSteps, Stopped
from brain_core.manipulators.scripted.grasp import GraspDirection, approach, jaw_line_deg, lifted, orientation
from brain_core.ports import Cancelled, CancelToken, FailureKind, Gripper, Item, Refusal, TargetItem
from brain_core.robot import MoveEnd, MoveResult
from brain_core.stages import CarryReports, GrabReports, Report
from control_core_py.geometry import Placement, UnitQuaternion

from vlm_router_brain.manipulators.vlm.aligner import MIN_FRAME_WAIT_S, NOT_STARTED, ModelAligner

LEFT = Gripper("left_gripper", "left_arm")
SEEN = (0.50, 0.10, 0.72)
POINT = (0.50, 0.10, 0.695)
PRE = (0.50, 0.10, 0.770)
LIFT_M = 0.075
DOWN_0 = orientation(GraspDirection.DOWN, 0.0)
# A wrist camera looking straight down, its picture's right along the
# robot's +y: a half turn about (1, 1, 0).
LOOKING_DOWN = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
# The frames the pictures name, and where the robot's tree places each.
WRIST_FRAME = "wrist_left_color_optical_frame"
CHEST_FRAME = "chest_color_optical_frame"
POSES = {
    WRIST_FRAME: Placement((0.2, 0.2, 0.5), UnitQuaternion(LOOKING_DOWN)),
    CHEST_FRAME: Placement((0.08, 0.03, 0.79), UnitQuaternion.IDENTITY),
}


def cube(position=SEEN) -> Item:
    return Item("cube_1-t0", "cube", position, None, 0.9, 1)


async def not_followed(report: Report) -> None:
    """The reporter of a test that does not read the feedback."""


# The reports of an alignment that no test reads.
QUIET = GrabReports(not_followed, "cube")


# When the chest picture and the first wrist picture were taken.
CHEST_TAKEN_S = 1.0
WRIST_TAKEN_S = 20.0


def _taken_at(frame, timestamp: float):
    frame.header.timestamp = timestamp
    return frame


class FakeWrists:
    """The wrist frames a test queues: each `newer_than` takes the next one,
    or gives none when the queue is empty."""

    def __init__(self, camera: Optional[str] = "wrist_left", frames: int = 10, period_s: Optional[float] = None) -> None:
        self.camera = camera
        self.frames: deque = deque(_taken_at(rgb_frame(capture_index=i, frame=WRIST_FRAME), WRIST_TAKEN_S + i) for i in range(1, frames + 1))
        self.period = period_s
        self.waited: list[float] = []

    def camera_of(self, arm):
        return self.camera if arm == "left_arm" else None

    def why_unavailable(self, arm):
        return "" if arm == "left_arm" and self.camera else f"no wrist camera for {arm}: the wrist_cameras parameter names none"

    def latest_stamp(self, arm):
        return 1.0

    def period_s(self, arm):
        return self.period

    async def newer_than(self, arm, timestamp, timeout_s):
        self.waited.append(timeout_s)
        return self.frames.popleft() if self.frames else None


class FakeChest:
    """The chest perceiver: its latest frame, taken at CHEST_TAKEN_S."""

    def __init__(self, *, frame=True, unavailable: str = "") -> None:
        self.frames = SimpleNamespace(latest=self._latest)
        self._frame = frame
        self.unavailable = unavailable

    def why_unavailable(self):
        return self.unavailable

    def _latest(self):
        if not self._frame:
            raise Refusal(FailureKind.RESOURCE_UNAVAILABLE, "no camera frame received")
        return SimpleNamespace(color=_taken_at(rgb_frame(), CHEST_TAKEN_S))


class FakeMoves:
    """Records every check and move. A check plans unless the test queued
    a refusal for that point's name; a move lands on its target unless the
    test queued a result."""

    def __init__(self) -> None:
        self.checks: list[tuple] = []
        self.moves: list[tuple] = []
        self.refusals: dict[str, str] = {}
        self.results: deque = deque()

    async def refusal_for(self, arm, points, orientation, cancel):
        assert arm == "left_arm" and len(points) == 1
        self.checks.append((points[0], orientation))
        return self.refusals.get(_which(points[0]), "")

    async def move_to(self, arm, point, orientation):
        assert arm == "left_arm"
        self.moves.append((point, orientation))
        return self.results.popleft() if self.results else MoveResult(MoveEnd.COMPLETED, final_position=point, final_orientation=orientation)


def _which(point) -> str:
    """Which of the three points a check is about, by its height."""
    if abs(point[2] - PRE[2]) < 1e-9:
        return "pre"
    if abs(point[2] - POINT[2]) < 1e-9:
        return "point"
    return "lift"


# Each (frame, time) of a picture the fake robot was asked to place.
PLACED: list[tuple[str, float]] = []


async def camera_pose(header):
    """Where the frame `header` names stands, as POSES gives it, at every
    time."""
    PLACED.append((header.frame, header.timestamp))
    return POSES[header.frame]


async def undescribed(header):
    raise Refusal(FailureKind.RESOURCE_UNAVAILABLE, "the robot's sensors have not been described yet")


def senses(*, wrists=None, chest=None, store=None, pose=camera_pose):
    if store is None:
        store = LimbStateStore()
        store.add(limb_states_message(1.0, left_arm=(PRE, DOWN_0), left_gripper=1.0))
    return SimpleNamespace(robot=None, limb_states=store, chest=chest or FakeChest(), wrists=wrists or FakeWrists(), camera_pose=pose)


def request(item=None) -> AlignRequest:
    return AlignRequest(item or cube(), LEFT, PRE, POINT, DOWN_0, LIFT_M)


def aligner(*answers, the_senses=None, **kwargs):
    model = FakeStepModel(*answers)
    a = ModelAligner(model, **kwargs)
    a.start(the_senses or senses())
    return a, model


async def test_ready_at_once_keeps_the_points_and_shows_the_model_the_state_and_two_pictures():
    a, model = aligner('{"action": "ready"}')
    moves = FakeMoves()
    end = await a.align(request(), moves, CancelToken(), QUIET)
    assert end == Aligned(PRE, POINT, lifted(POINT, LIFT_M), DOWN_0, steps=1)
    assert moves.checks == [] and moves.moves == []
    (turn,) = model.turns
    assert "the cube" in turn.goal and "the left wrist" in turn.goal
    assert [picture.name for picture in turn.pictures] == ["wrist", "chest"]
    assert all(picture.jpeg.startswith(b"\xff\xd8") for picture in turn.pictures)
    assert "+0.0 cm in x and +0.0 cm in y from the cube's seen centre" in turn.state
    assert "7.5 cm above the height it closes at" in turn.state
    assert "90 degrees counter-clockwise from +x" in turn.state
    assert "Jaws open at 1.00" in turn.state
    # The wrist camera looks down with its right along +y: +x is at 6, +y at 3.
    assert "+x points toward 6 o'clock and +y toward 3 o'clock" in turn.state
    # The chest camera of the tests is the robot frame itself: +x at 3, +y at 6.
    assert "+x toward 3 o'clock and +y toward 6 o'clock" in turn.state
    assert [tool.name for tool in turn.tools] == ["move", "ready", "give_up"]


async def test_a_move_inside_the_limits_checks_the_three_points_turned_and_moves_at_the_same_height():
    a, model = aligner('{"action": "move", "dx": 1.0, "dy": -2.0, "roll_deg": 45}', '{"action": "ready"}')
    moves = FakeMoves()
    end = await a.align(request(), moves, CancelToken(), QUIET)
    pre = (0.51, 0.08, PRE[2])
    point = (0.51, 0.08, POINT[2])
    assert [p for p, _ in moves.checks] == [pre, point, lifted(point, LIFT_M)]
    (moved_to, q) = moves.moves[0]
    assert moved_to == pre
    assert jaw_line_deg(q) == pytest.approx(135.0) and approach(q) == pytest.approx((0.0, 0.0, -1.0))
    assert all(jaw_line_deg(check_q) == pytest.approx(135.0) for _, check_q in moves.checks)
    assert isinstance(end, Aligned) and end.steps == 2
    assert (end.pre, end.point, end.lift) == (pre, point, lifted(point, LIFT_M))
    assert jaw_line_deg(end.orientation) == pytest.approx(135.0)
    (line,) = model.turns[1].history
    assert line.endswith("-> moved: the grasp point is now +0.0 cm in x and +0.0 cm in y from the cube's seen centre; "
                         "the jaws close along a line 135 degrees counter-clockwise from +x")


@pytest.mark.parametrize("answer, said", [
    ('{"action": "move", "dx": 3.0, "dy": 2.8, "roll_deg": 0}', "moved nothing: a step moves at most 3 cm, this one is 4.1 cm"),
    ('{"action": "move", "dx": 0, "dy": 0, "roll_deg": 95}', "moved nothing: one step turns at most 90 degrees"),
    ('{"action": "move", "dx": 0, "dy": 0, "roll_deg": -91}', "moved nothing: one step turns at most 90 degrees"),
    ('{"action": "move", "dx": 1, "dy": "far", "roll_deg": 0}', "nothing was done: the answer could not be read: dy must be a number"),
    ('{"action": "move", "dx": 1, "dy": 2}', "nothing was done: the answer could not be read: roll_deg must be a number"),
    ('{"action": "move", "dx": true, "dy": 2, "roll_deg": 0}', "nothing was done: the answer could not be read: dx must be a number"),
])
async def test_a_step_over_a_limit_or_unreadable_moves_nothing_counts_and_is_fed_back(answer, said):
    a, model = aligner(answer, '{"action": "ready"}')
    moves = FakeMoves()
    end = await a.align(request(), moves, CancelToken(), QUIET)
    assert moves.checks == [] and moves.moves == []
    assert end == Aligned(PRE, POINT, lifted(POINT, LIFT_M), DOWN_0, steps=2)
    (line,) = model.turns[1].history
    assert line.endswith(said), line


async def test_a_step_that_ends_over_10_cm_from_the_item_moves_nothing():
    # The gripper starts 9 cm left of the item's centre; 2 cm further is 11 cm.
    a, model = aligner('{"action": "move", "dx": 0, "dy": 2.0, "roll_deg": 0}', '{"action": "ready"}')
    far = (SEEN[0], SEEN[1] + 0.09, PRE[2])
    far_request = AlignRequest(cube(), LEFT, far, (far[0], far[1], POINT[2]), DOWN_0, LIFT_M)
    moves = FakeMoves()
    await a.align(far_request, moves, CancelToken(), QUIET)
    assert moves.moves == []
    (line,) = model.turns[1].history
    assert line.endswith("moved nothing: the gripper stays within 10 cm of the item, this step ends 11.0 cm from it")


async def test_a_point_the_backbone_has_no_plan_for_moves_nothing_and_is_named():
    a, model = aligner('{"action": "move", "dx": 1, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    moves.refusals["point"] = "no joint solution"
    end = await a.align(request(), moves, CancelToken(), QUIET)
    assert len(moves.checks) == 2 and moves.moves == []
    (line,) = model.turns[1].history
    assert line.endswith("moved nothing: the backbone has no plan for the grasp point: no joint solution")
    assert end.pre == PRE


async def test_a_move_that_fails_keeps_the_points_and_tells_the_model_where_the_gripper_is():
    a, model = aligner('{"action": "move", "dx": 1, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    moves.results.append(MoveResult(MoveEnd.FAILED, "the arm did not settle", "timeout"))
    end = await a.align(request(), moves, CancelToken(), QUIET)
    (line,) = model.turns[1].history
    assert "-> the move failed: the arm did not settle; the grasp point is now" in line
    assert end == Aligned(PRE, POINT, lifted(POINT, LIFT_M), DOWN_0, steps=2)


async def test_a_move_a_stop_ended_ends_the_alignment_stopped():
    a, model = aligner('{"action": "move", "dx": 1, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    moves.results.append(MoveResult(MoveEnd.CANCELLED, "stopped: operator"))
    end = await a.align(request(), moves, CancelToken(), QUIET)
    assert end == Stopped("stopped: operator", steps=1)
    assert len(model.turns) == 1


async def test_give_up_and_running_out_of_steps():
    a, _ = aligner('{"action": "give_up"}')
    assert await a.align(request(), FakeMoves(), CancelToken(), QUIET) == GaveUp(steps=1)
    a, model = aligner(*['{"action": "move", "dx": 0.5, "dy": 0, "roll_deg": 10}'] * 5, max_steps=5)
    assert await a.align(request(), FakeMoves(), CancelToken(), QUIET) == OutOfSteps(5)
    assert len(model.turns) == 5


async def test_the_aligner_says_why_an_arm_cannot_be_aligned():
    a = ModelAligner(FakeStepModel())
    assert a.why_unavailable("left_arm") == NOT_STARTED
    with pytest.raises(AlignmentFailed, match=NOT_STARTED):
        await a.align(request(), FakeMoves(), CancelToken(), QUIET)
    a.start(senses(wrists=FakeWrists(camera=None)))
    assert a.why_unavailable("left_arm") == "no wrist camera for left_arm: the wrist_cameras parameter names none"
    a.start(senses(chest=FakeChest(unavailable="no perception source: no camera frame received")))
    assert a.why_unavailable("left_arm") == "no perception source: no camera frame received"
    a.start(senses())
    assert a.why_unavailable("left_arm") == ""


async def test_a_step_with_no_wrist_frame_after_the_move_fails_the_alignment():
    a, _ = aligner('{"action": "ready"}', the_senses=senses(wrists=FakeWrists(frames=0)))
    with pytest.raises(AlignmentFailed, match="no frame from wrist camera 'wrist_left' after the move"):
        await a.align(request(), FakeMoves(), CancelToken(), QUIET)


async def test_each_camera_is_placed_where_the_frame_its_picture_names_stood_when_the_picture_was_taken():
    PLACED.clear()
    a, _ = aligner('{"action": "ready"}', the_senses=senses())
    await a.align(request(), FakeMoves(), CancelToken(), QUIET)
    assert PLACED == [(WRIST_FRAME, WRIST_TAKEN_S + 1), (CHEST_FRAME, CHEST_TAKEN_S)]


async def test_a_step_with_no_chest_frame_no_chest_pose_or_no_limbs_fails_the_alignment():
    a, _ = aligner('{"action": "ready"}', the_senses=senses(chest=FakeChest(frame=False)))
    with pytest.raises(AlignmentFailed, match="no camera frame received"):
        await a.align(request(), FakeMoves(), CancelToken(), QUIET)
    a, _ = aligner('{"action": "ready"}', the_senses=senses(pose=undescribed))
    with pytest.raises(AlignmentFailed, match="the robot's sensors have not been described yet"):
        await a.align(request(), FakeMoves(), CancelToken(), QUIET)
    a, _ = aligner('{"action": "ready"}', the_senses=senses(store=LimbStateStore()))
    with pytest.raises(AlignmentFailed, match=NO_SNAPSHOT):
        await a.align(request(), FakeMoves(), CancelToken(), QUIET)


async def test_a_picture_whose_frame_the_robot_cannot_place_fails_the_alignment_with_its_words():
    async def no_pose(header):
        raise Refusal(
            FailureKind.RESOURCE_UNAVAILABLE,
            f"picture {header.capture_index} dropped: unknown_frame: the frame '{header.frame}' at 21.000 s: no transform names it",
        )

    a, _ = aligner('{"action": "ready"}', the_senses=senses(pose=no_pose))
    with pytest.raises(AlignmentFailed, match=f"^picture 1 dropped: unknown_frame: the frame '{WRIST_FRAME}'"):
        await a.align(request(), FakeMoves(), CancelToken(), QUIET)


@pytest.mark.parametrize(
    "period_s, waited",
    [
        # A camera the brain measured at 0.4 Hz, as a loaded host renders an
        # idle simulated one: two and a half of its periods.
        (2.5, 6.25),
        # A fast camera: the least wait.
        (1 / 30, 0.5),
        # A camera that sent one frame alone: the least wait.
        (None, 0.5),
    ],
)
async def test_a_step_waits_for_the_frame_periods_of_the_wrist_camera_and_the_least_wait(period_s, waited):
    wrists = FakeWrists(period_s=period_s)
    a, _ = aligner('{"action": "ready"}', the_senses=senses(wrists=wrists), frame_wait_periods=2.5, min_frame_wait_s=0.5)
    await a.align(request(), FakeMoves(), CancelToken(), QUIET)
    assert wrists.waited == [pytest.approx(waited)]


async def test_a_step_waits_at_least_the_least_wait_by_default():
    wrists = FakeWrists()
    a, _ = aligner('{"action": "ready"}', the_senses=senses(wrists=wrists))
    await a.align(request(), FakeMoves(), CancelToken(), QUIET)
    assert wrists.waited == [MIN_FRAME_WAIT_S]


async def test_a_cancel_during_the_models_call_raises_cancelled():
    a, model = aligner()
    cancel = CancelToken()
    running = asyncio.ensure_future(a.align(request(), FakeMoves(), cancel, QUIET))
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(model.turns) == 1
    cancel.cancel("aborted: test", signalled=False)
    with pytest.raises(Cancelled):
        await running


# The property: whatever the model answers and the backbone does, every move
# the aligner sends keeps the limits, and the alignment ends one of the
# four ways within five steps.

moves_st = st.builds(
    lambda dx, dy, roll: f'{{"action": "move", "dx": {dx}, "dy": {dy}, "roll_deg": {roll}}}',
    st.floats(-5, 5, allow_nan=False), st.floats(-5, 5, allow_nan=False), st.floats(-180, 180, allow_nan=False),
)
answers_st = st.lists(
    st.one_of(moves_st, st.just('{"action": "ready"}'), st.just('{"action": "give_up"}'), st.just("junk"), st.just('{"action": "move"}')),
    min_size=0, max_size=8,
)
refusals_st = st.dictionaries(st.sampled_from(["pre", "point", "lift"]), st.just("no plan"), max_size=3)
results_st = st.lists(
    st.one_of(
        st.just(None),
        st.just(MoveResult(MoveEnd.FAILED, "did not settle", "timeout")),
        st.just(MoveResult(MoveEnd.CANCELLED, "stopped")),
    ),
    max_size=8,
)


@settings(max_examples=200, deadline=None)
@given(answers=answers_st, refusals=refusals_st, results=results_st)
def test_every_move_keeps_the_limits_and_the_alignment_ends_within_five_steps(answers, refusals, results):
    async def run():
        # The scripted answers end with a ready, so the loop always ends.
        a, model = aligner(*answers, '{"action": "ready"}')
        moves = FakeMoves()
        moves.refusals = refusals
        moves.results.extend(r for r in results if r is not None)
        end = await a.align(request(), moves, CancelToken(), QUIET)
        assert len(model.turns) <= 5
        previous, previous_q = PRE, DOWN_0
        for point, q in moves.moves:
            assert point[2] == pytest.approx(PRE[2])
            assert math.hypot(point[0] - previous[0], point[1] - previous[1]) <= 0.03 + 1e-9
            assert math.hypot(point[0] - SEEN[0], point[1] - SEEN[1]) <= 0.10 + 1e-9
            turn = abs(((jaw_line_deg(q) - jaw_line_deg(previous_q) + 90) % 180) - 90)
            assert turn <= 90 + 1e-6
            assert approach(q) == pytest.approx((0.0, 0.0, -1.0))
            previous, previous_q = point, q
        assert isinstance(end, (Aligned, GaveUp, OutOfSteps, Stopped))
        if isinstance(end, Aligned):
            assert end.point[2] == pytest.approx(POINT[2]) and end.lift == lifted(end.point, LIFT_M)
            assert (end.pre[0], end.pre[1]) == (end.point[0], end.point[1])

    asyncio.run(run())


# A place: the same tools above the spot the held item goes to, the limit
# measured from the target point.

TARGET = (0.50, 0.10, 0.70)


def place_request(pre=PRE, target=TARGET) -> AlignRequest:
    return AlignRequest(None, LEFT, pre, (pre[0], pre[1], target[2]), DOWN_0, LIFT_M, target=target)


async def test_a_place_request_speaks_of_the_held_item_and_measures_from_the_target_point():
    a, model = aligner('{"action": "ready"}')
    end = await a.align(place_request(), FakeMoves(), CancelToken(), QUIET)
    assert end == Aligned(PRE, (PRE[0], PRE[1], TARGET[2]), lifted((PRE[0], PRE[1], TARGET[2]), LIFT_M), DOWN_0, steps=1)
    (turn,) = model.turns
    assert turn.goal.startswith("You guide a robot gripper that holds an item")
    assert "the held item straight above the spot the target point is in" in turn.goal
    assert "from the target point;" in turn.state and "7.0 cm above the height it releases at" in turn.state
    assert "The jaws hold the item at 1.00" in turn.state
    move, ready, give_up = turn.tools
    assert move.description.endswith("the gripper stays within 10 cm of the target point.")
    assert ready.description.startswith("the held item is centred above its spot")
    assert give_up.description.startswith("no spot within reach of the moves can take the item")


async def test_a_place_step_that_ends_over_10_cm_from_the_target_point_moves_nothing():
    a, model = aligner('{"action": "move", "dx": 0, "dy": 2.0, "roll_deg": 0}', '{"action": "ready"}')
    far = (TARGET[0], TARGET[1] + 0.09, PRE[2])
    moves = FakeMoves()
    await a.align(place_request(pre=far), moves, CancelToken(), QUIET)
    assert moves.moves == []
    (line,) = model.turns[1].history
    assert line.endswith("moved nothing: the gripper stays within 10 cm of the item, this step ends 11.0 cm from it")


async def test_a_place_move_keeps_the_release_height_and_the_history_measures_from_the_target():
    a, model = aligner('{"action": "move", "dx": 1.0, "dy": 0.0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    end = await a.align(place_request(), moves, CancelToken(), QUIET)
    (moved_to, _q) = moves.moves[0]
    assert moved_to == (0.51, 0.10, PRE[2])
    assert end.point == (0.51, 0.10, TARGET[2])
    (line,) = model.turns[1].history
    assert "from the target point" in line


# Above a target item: the texts name it and its width, the limit is the
# smaller of 10 cm and half its width less 2 cm, and a drop, with no
# descent after it, checks the pre point alone.

BOWL = TargetItem("wooden_bowl_1-t0", "wooden bowl", (0.50, 0.10), 0.72, 0.65, 0.18)
BOWL_BOTTOM = (0.50, 0.10, 0.65)


def into_request(target=BOWL, descends=True) -> AlignRequest:
    point = BOWL_BOTTOM if descends else PRE
    return AlignRequest(None, LEFT, PRE, point, DOWN_0, LIFT_M, target=(*target.centre, target.top_z), into=target, descends=descends)


async def test_a_place_into_a_bowl_names_the_bowl_and_its_width_and_measures_from_its_centre():
    a, model = aligner('{"action": "ready"}')
    end = await a.align(into_request(), FakeMoves(), CancelToken(), QUIET)
    assert isinstance(end, Aligned) and end.point == BOWL_BOTTOM
    (turn,) = model.turns
    assert turn.goal.startswith("You guide a robot gripper that holds an item to a free spot of the wooden bowl, about 18 cm wide.")
    assert "The gripper then goes straight down until the item rests, and opens." in turn.goal
    assert "from the centre of the wooden bowl;" in turn.state and "12.0 cm above the bottom of the wooden bowl." in turn.state
    move, ready, give_up = turn.tools
    # Half of 18 cm less 2 cm: 7 cm.
    assert move.description.endswith("the gripper stays within 7 cm of the centre of the wooden bowl.")
    assert ready.description.startswith("the held item is centred above a free spot of the wooden bowl: the gripper goes straight down")
    assert give_up.description.startswith("no free spot of the wooden bowl is within reach of the moves")


async def test_a_step_that_ends_beyond_half_the_bowls_width_less_the_margin_moves_nothing():
    a, model = aligner('{"action": "move", "dx": 2.5, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    off_centre = (0.55, 0.10, PRE[2])
    moves = FakeMoves()
    request = AlignRequest(None, LEFT, off_centre, (0.55, 0.10, 0.65), DOWN_0, LIFT_M, target=(0.50, 0.10, 0.72), into=BOWL)
    await a.align(request, moves, CancelToken(), QUIET)
    assert moves.moves == []
    (line,) = model.turns[1].history
    assert line.endswith("moved nothing: the gripper stays within 7 cm of the item, this step ends 7.5 cm from it")


@pytest.mark.parametrize("width_m, radius_words", [(None, "10 cm"), (0.30, "10 cm"), (0.03, "0 cm")])
async def test_the_limit_above_a_target_is_10_cm_at_most_and_never_below_0(width_m, radius_words):
    a, model = aligner('{"action": "ready"}')
    target = TargetItem("tray_1-t0", "tray", (0.50, 0.10), 0.72, 0.65, width_m)
    await a.align(into_request(target), FakeMoves(), CancelToken(), QUIET)
    move = model.turns[0].tools[0]
    assert move.description.endswith(f"the gripper stays within {radius_words} of the centre of the tray.")


async def test_a_drop_above_a_bowl_checks_the_pre_point_alone_and_says_the_jaws_open_at_this_height():
    a, model = aligner('{"action": "move", "dx": 1.0, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    end = await a.align(into_request(descends=False), moves, CancelToken(), QUIET)
    assert moves.checks == [((0.51, 0.10, PRE[2]), DOWN_0)]
    assert end.pre == (0.51, 0.10, PRE[2])
    turn = model.turns[0]
    assert "The jaws then open at this height, with no descent: the item falls into the wooden bowl." in turn.goal
    assert "; the jaws open at this height." in turn.state
    assert turn.tools[1].description.startswith("the held item is centred above a free spot of the wooden bowl: the jaws open at this height")


# The alignment of a place: two answers, the whole move in the first, no
# limit of a step, and Aligned when the answers run out after a move.

async def test_a_place_has_two_answers_and_its_goal_asks_for_the_whole_move_in_the_first():
    a, model = aligner('{"action": "ready"}')
    await a.align(place_request(), FakeMoves(), CancelToken(), QUIET)
    (turn,) = model.turns
    assert turn.max_steps == 2
    assert "and opens. You have 2 answers: make the whole move in the first, then answer ready." in turn.goal
    assert turn.tools[0].description.startswith(
        "moves the gripper sideways by dx and dy and turns it by roll_deg, at the same height. One move goes anywhere at once"
    )


async def test_one_move_of_a_place_goes_anywhere_inside_the_limit_beyond_3_cm():
    a, model = aligner('{"action": "move", "dx": 8.0, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    end = await a.align(place_request(), moves, CancelToken(), QUIET)
    assert moves.moves[0][0] == (0.58, 0.10, PRE[2]) and isinstance(end, Aligned) and end.steps == 2


async def test_a_place_whose_answers_run_out_after_a_move_that_completed_ends_aligned_there():
    a, model = aligner(
        '{"action": "move", "dx": 1.0, "dy": 0, "roll_deg": 0}', '{"action": "move", "dx": 1.0, "dy": 0, "roll_deg": 0}',
        '{"action": "ready"}',
    )
    moves = FakeMoves()
    end = await a.align(place_request(), moves, CancelToken(), QUIET)
    assert len(model.turns) == 2
    assert end == Aligned((0.52, 0.10, PRE[2]), (0.52, 0.10, TARGET[2]), lifted((0.52, 0.10, TARGET[2]), LIFT_M), DOWN_0, steps=2)


async def test_a_place_whose_last_answer_moved_nothing_runs_out_of_steps():
    a, model = aligner('{"action": "move", "dx": 1.0, "dy": 0, "roll_deg": 0}', '{"action": "move", "dx": 0, "dy": 20.0, "roll_deg": 0}')
    end = await a.align(place_request(), FakeMoves(), CancelToken(), QUIET)
    assert end == OutOfSteps(2)


async def test_a_grab_keeps_five_steps_of_3_cm_at_most():
    a, model = aligner('{"action": "move", "dx": 4.0, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    await a.align(request(), moves, CancelToken(), QUIET)
    assert moves.moves == [] and model.turns[0].max_steps == 5


# The reports of each step: the question at the start of the step, before
# the wait for the wrist frame and the model's answer, and the correction
# before its move, each with the number of the step and the steps of the
# alignment.


class Published:
    """A reporter that keeps each report and, for each, its stage, number
    and maximum, with how many wrist frames the alignment had waited for,
    questions the model had got and moves the gripper had made when it was
    published."""

    def __init__(self, wrists: FakeWrists, model, moves: FakeMoves) -> None:
        self.wrists = wrists
        self.model = model
        self.moves = moves
        self.reports: list[Report] = []
        self.seen: list[tuple] = []

    async def __call__(self, report: Report) -> None:
        self.reports.append(report)
        self.seen.append(
            (report.stage.value, report.number, report.maximum, len(self.wrists.waited), len(self.model.turns), len(self.moves.moves))
        )


def published_alignment(*answers):
    wrists = FakeWrists()
    a, model = aligner(*answers, the_senses=senses(wrists=wrists))
    moves = FakeMoves()
    return a, moves, Published(wrists, model, moves)


async def test_each_step_reports_its_question_before_it_looks_and_each_correction_before_its_move():
    a, moves, published = published_alignment(
        '{"action": "move", "dx": 1.0, "dy": 0, "roll_deg": 0}',
        # Over the limit of a step: it moves nothing, so it reports no correction.
        '{"action": "move", "dx": 5.0, "dy": 0, "roll_deg": 0}',
        '{"action": "ready"}',
    )
    end = await a.align(request(), moves, CancelToken(), GrabReports(published, "cube"))
    assert isinstance(end, Aligned) and end.steps == 3
    # (stage, number, maximum, wrist frames waited for, questions, moves)
    assert published.seen == [
        ("align_look", 1, 5, 0, 0, 0),
        ("align_move", 1, 5, 1, 1, 0),
        ("align_look", 2, 5, 1, 1, 1),
        ("align_look", 3, 5, 2, 2, 1),
    ]
    assert [report.activity for report in published.reports] == [
        "The robot looks at the cube and decides how to correct its hand (1 of 5)",
        "The robot corrects the position of its hand above the cube (1 of 5)",
        "The robot looks at the cube and decides how to correct its hand (2 of 5)",
        "The robot looks at the cube and decides how to correct its hand (3 of 5)",
    ]


async def test_a_correction_the_backbone_has_no_plan_for_reports_no_move():
    a, moves, published = published_alignment('{"action": "move", "dx": 1, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves.refusals["point"] = "no joint solution"
    await a.align(request(), moves, CancelToken(), GrabReports(published, "cube"))
    assert [stage for stage, *_ in published.seen] == ["align_look", "align_look"]


async def test_a_place_reports_its_steps_out_of_the_answers_of_a_place():
    a, moves, published = published_alignment('{"action": "move", "dx": 1.0, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    await a.align(place_request(), moves, CancelToken(), CarryReports(published, "cube"))
    assert [(stage, number, maximum) for stage, number, maximum, *_ in published.seen] == [
        ("align_look", 1, 2), ("align_move", 1, 2), ("align_look", 2, 2),
    ]
    assert published.reports[0].activity == "The robot looks down and decides where to put the cube (1 of 2)"


async def test_a_drop_above_a_bowl_reports_its_steps_naming_the_bowl():
    a, moves, published = published_alignment('{"action": "ready"}')
    await a.align(into_request(descends=False), moves, CancelToken(), CarryReports(published, "cube", BOWL))
    assert [report.activity for report in published.reports] == [
        "The robot looks at the wooden bowl and decides where to put the cube (1 of 2)",
    ]


async def test_a_step_whose_wrist_frame_does_not_come_reported_its_question_and_fails():
    wrists = FakeWrists(frames=0)
    a, model = aligner('{"action": "ready"}', the_senses=senses(wrists=wrists))
    moves = FakeMoves()
    published = Published(wrists, model, moves)
    with pytest.raises(AlignmentFailed, match="no frame from wrist camera"):
        await a.align(request(), moves, CancelToken(), GrabReports(published, "cube"))
    assert published.seen == [("align_look", 1, 5, 0, 0, 0)]
