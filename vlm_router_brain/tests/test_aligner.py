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
from brain_core.perception.camera import CameraPose
from brain_core.ports import Cancelled, CancelToken, FailureKind, Gripper, Item, Refusal
from brain_core.robot import MoveEnd, MoveResult

from vlm_router_brain.manipulators.vlm.aligner import NOT_STARTED, ModelAligner

LEFT = Gripper("left_gripper", "left_arm")
SEEN = (0.50, 0.10, 0.72)
POINT = (0.50, 0.10, 0.695)
PRE = (0.50, 0.10, 0.770)
LIFT_M = 0.075
DOWN_0 = orientation(GraspDirection.DOWN, 0.0)
# A wrist camera looking straight down, its picture's right along the
# robot's +y: a half turn about (1, 1, 0).
LOOKING_DOWN = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
CHEST_POSE = CameraPose((0.08, 0.03, 0.79), (0.0, 0.0, 0.0, 1.0))


def cube(position=SEEN) -> Item:
    return Item("cube_1-t0", "cube", position, None, 0.9, 1)


class FakeWrists:
    """The wrist frames a test queues: each `newer_than` takes the next one,
    or gives none when the queue is empty."""

    def __init__(self, camera: Optional[str] = "wrist_left", frames: int = 10) -> None:
        self.camera = camera
        self.frames: deque = deque(rgb_frame(frame_id=i) for i in range(1, frames + 1))
        self.waited: list[float] = []

    def camera_of(self, arm):
        return self.camera if arm == "left_arm" else None

    def why_unavailable(self, arm):
        return "" if arm == "left_arm" and self.camera else f"no wrist camera for {arm}: the wrist_cameras parameter names none"

    def latest_stamp(self, arm):
        return 1.0

    async def newer_than(self, arm, timestamp, timeout_s):
        self.waited.append(timeout_s)
        return self.frames.popleft() if self.frames else None


class FakeChest:
    def __init__(self, *, frame=True, pose: Optional[CameraPose] = CHEST_POSE, unavailable: str = "") -> None:
        self.frames = SimpleNamespace(latest=self._latest)
        self.camera = SimpleNamespace(pose=pose)
        self._frame = frame
        self.unavailable = unavailable

    def why_unavailable(self):
        return self.unavailable

    def _latest(self):
        if not self._frame:
            raise Refusal(FailureKind.RESOURCE_UNAVAILABLE, "no camera frame received")
        return SimpleNamespace(color=rgb_frame())


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


async def camera_pose(name):
    assert name == "wrist_left"
    return (0.2, 0.2, 0.5), LOOKING_DOWN


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
    end = await a.align(request(), moves, CancelToken())
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
    end = await a.align(request(), moves, CancelToken())
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
    end = await a.align(request(), moves, CancelToken())
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
    await a.align(far_request, moves, CancelToken())
    assert moves.moves == []
    (line,) = model.turns[1].history
    assert line.endswith("moved nothing: the gripper stays within 10 cm of the item, this step ends 11.0 cm from it")


async def test_a_point_the_backbone_has_no_plan_for_moves_nothing_and_is_named():
    a, model = aligner('{"action": "move", "dx": 1, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    moves.refusals["point"] = "no joint solution"
    end = await a.align(request(), moves, CancelToken())
    assert len(moves.checks) == 2 and moves.moves == []
    (line,) = model.turns[1].history
    assert line.endswith("moved nothing: the backbone has no plan for the grasp point: no joint solution")
    assert end.pre == PRE


async def test_a_move_that_fails_keeps_the_points_and_tells_the_model_where_the_gripper_is():
    a, model = aligner('{"action": "move", "dx": 1, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    moves.results.append(MoveResult(MoveEnd.FAILED, "the arm did not settle", "timeout"))
    end = await a.align(request(), moves, CancelToken())
    (line,) = model.turns[1].history
    assert "-> the move failed: the arm did not settle; the grasp point is now" in line
    assert end == Aligned(PRE, POINT, lifted(POINT, LIFT_M), DOWN_0, steps=2)


async def test_a_move_a_stop_ended_ends_the_alignment_stopped():
    a, model = aligner('{"action": "move", "dx": 1, "dy": 0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    moves.results.append(MoveResult(MoveEnd.CANCELLED, "stopped: operator"))
    end = await a.align(request(), moves, CancelToken())
    assert end == Stopped("stopped: operator", steps=1)
    assert len(model.turns) == 1


async def test_give_up_and_running_out_of_steps():
    a, _ = aligner('{"action": "give_up"}')
    assert await a.align(request(), FakeMoves(), CancelToken()) == GaveUp(steps=1)
    a, model = aligner(*['{"action": "move", "dx": 0.5, "dy": 0, "roll_deg": 10}'] * 5, max_steps=5)
    assert await a.align(request(), FakeMoves(), CancelToken()) == OutOfSteps(5)
    assert len(model.turns) == 5


async def test_the_aligner_says_why_an_arm_cannot_be_aligned():
    a = ModelAligner(FakeStepModel())
    assert a.why_unavailable("left_arm") == NOT_STARTED
    with pytest.raises(AlignmentFailed, match=NOT_STARTED):
        await a.align(request(), FakeMoves(), CancelToken())
    a.start(senses(wrists=FakeWrists(camera=None)))
    assert a.why_unavailable("left_arm") == "no wrist camera for left_arm: the wrist_cameras parameter names none"
    a.start(senses(chest=FakeChest(unavailable="no perception source: no camera frame received")))
    assert a.why_unavailable("left_arm") == "no perception source: no camera frame received"
    a.start(senses())
    assert a.why_unavailable("left_arm") == ""


async def test_a_step_with_no_wrist_frame_after_the_move_fails_the_alignment():
    a, _ = aligner('{"action": "ready"}', the_senses=senses(wrists=FakeWrists(frames=0)))
    with pytest.raises(AlignmentFailed, match="no frame from wrist camera 'wrist_left' after the move"):
        await a.align(request(), FakeMoves(), CancelToken())


async def test_a_step_with_no_chest_frame_no_chest_pose_or_no_limbs_fails_the_alignment():
    a, _ = aligner('{"action": "ready"}', the_senses=senses(chest=FakeChest(frame=False)))
    with pytest.raises(AlignmentFailed, match="no camera frame received"):
        await a.align(request(), FakeMoves(), CancelToken())
    a, _ = aligner('{"action": "ready"}', the_senses=senses(chest=FakeChest(pose=None)))
    with pytest.raises(AlignmentFailed, match="the chest camera's pose is not known yet"):
        await a.align(request(), FakeMoves(), CancelToken())
    a, _ = aligner('{"action": "ready"}', the_senses=senses(store=LimbStateStore()))
    with pytest.raises(AlignmentFailed, match=NO_SNAPSHOT):
        await a.align(request(), FakeMoves(), CancelToken())


async def test_a_camera_mounts_refusal_fails_the_alignment_with_its_words():
    async def no_pose(name):
        raise Refusal(FailureKind.RESOURCE_UNAVAILABLE, "the robot carries no camera named 'wrist_left'")

    a, _ = aligner('{"action": "ready"}', the_senses=senses(pose=no_pose))
    with pytest.raises(AlignmentFailed, match="carries no camera named 'wrist_left'"):
        await a.align(request(), FakeMoves(), CancelToken())


async def test_the_step_waits_for_the_frame_as_long_as_the_aligner_says():
    wrists = FakeWrists()
    a, _ = aligner('{"action": "ready"}', the_senses=senses(wrists=wrists), frame_wait_s=0.25)
    await a.align(request(), FakeMoves(), CancelToken())
    assert wrists.waited == [0.25]


async def test_a_cancel_during_the_models_call_raises_cancelled():
    a, model = aligner()
    cancel = CancelToken()
    running = asyncio.ensure_future(a.align(request(), FakeMoves(), cancel))
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
        end = await a.align(request(), moves, CancelToken())
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
    end = await a.align(place_request(), FakeMoves(), CancelToken())
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
    await a.align(place_request(pre=far), moves, CancelToken())
    assert moves.moves == []
    (line,) = model.turns[1].history
    assert line.endswith("moved nothing: the gripper stays within 10 cm of the item, this step ends 11.0 cm from it")


async def test_a_place_move_keeps_the_release_height_and_the_history_measures_from_the_target():
    a, model = aligner('{"action": "move", "dx": 1.0, "dy": 0.0, "roll_deg": 0}', '{"action": "ready"}')
    moves = FakeMoves()
    end = await a.align(place_request(), moves, CancelToken())
    (moved_to, _q) = moves.moves[0]
    assert moved_to == (0.51, 0.10, PRE[2])
    assert end.point == (0.51, 0.10, TARGET[2])
    (line,) = model.turns[1].history
    assert "from the target point" in line
