"""The alignment step of a grab and of a place: a model looks and corrects
the gripper's place and turn above the item, or above the spot the held
item goes to, before the descent.

The scripted grab brings the open gripper to the pre-grasp point, the
scripted place brings the held item to the pre-place point, and each then
hands the aligner the gripper, the points of the descent and the
orientation it stands in (`AlignRequest`): with the item to grasp for a
grab, with the target point alone for a place at a pose, and with the
target item for a place or a drop into an item (`AlignRequest.into`). The aligner runs a
`ToolLoop` with three tools: `move` (sideways by dx and dy, in
centimetres, and a turn about the vertical by roll_deg, counter-clockwise
seen from above, at the same height), `ready` (centred and turned: go
down) and `give_up`. Each step the model sees the picture of the wrist
camera of the gripper's arm, captured after the arm settled, which the step
waits for `frame_wait_periods` periods of that camera as the brain measures
them, at least `min_frame_wait_s`, the chest
picture, and the state as text: where the grasp point stands against the
item's seen centre or the target point, how high above the close or the
release, the line the jaws close along, the jaws' opening, and where the
robot's +x and +y point in each picture, as clock hours.

The code holds the limits, not the model. A step moves at most
`step_max_m`, turns at most `roll_max_deg`, keeps the gripper within
`radius_m` of the item's seen centre or the target point, and never
changes the height. Above a target item the limit is the smaller of
`radius_m` and half the item's width less `target_margin_m`, from the
centre of the item, so the held item stays clear of its rim. A step over a
limit, or one the backbone has no plan for, moves nothing: it counts as a
step and its reason goes into the next step's history, as does an answer
the loop cannot read. The three points of the descent (pre, point, lifted)
are checked before each move, so the descent that follows a `ready` has a
plan; above a target item with no descent after it, a drop, the pre point
alone.

A place, at a pose or into a target item, has `place_max_steps` answers,
and its goal text says so and asks for the whole move in the first: one
move may go anywhere inside the limit, with no limit of a step. A grab has
`max_steps`, each step within `step_max_m`.

Each step publishes its question through the reports of the goal
(`brain_core.stages.AlignReports`): `align_look` at the start of the step,
before it waits for the wrist frame and asks the model, and `align_move`
before the move of a correction that passed its checks, each with the
number of the step and the steps of the alignment (`max_steps` for a grab,
`place_max_steps` for a place or a drop). A step that moves nothing
publishes no `align_move`. The words of each report are brain_core's.

The alignment ends `Aligned` with the new points and orientation, `GaveUp`,
`OutOfSteps` after the last step, or `Stopped` when a stop ended a move; a
place whose last step was a move that completed ends `Aligned` there, with
no `ready` after it. A cancel raises `Cancelled` from the loop. `AlignmentFailed` names what the
step could not get: a wrist frame after the move, the chest frame, a
camera's pose, the limbs.

The aligner keeps nothing between grabs: each alignment runs in a `_Run`
of its own. The request, the endings and the moves are the `Aligner` port
of brain_core's scripted manipulator (`manipulators.scripted.aligning`):
this module is the model behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional

from brain_core.manipulators.scripted.aligning import (
    Aligned,
    Alignment,
    AlignmentFailed,
    AlignRequest,
    ArmMoves,
    GaveUp,
    OutOfSteps,
    Stopped,
)
from brain_core.manipulators.scripted.grasp import jaw_line_deg, lifted, turned_about_z
from brain_core.perception.wrist import jpeg_of
from brain_core.ports import CancelToken, Refusal, Vec3
from brain_core.stages import AlignReports

from .directions import axes_in_picture
from .loop import Observation, Picture, StepModel, ToolLoop, ToolResult

MAX_STEPS = 5
# The answers of the alignment of a place: the whole move, then ready.
PLACE_MAX_STEPS = 2
STEP_MAX_M = 0.03
RADIUS_M = 0.10
# How far inside the rim of a target item the held item's centre stays: the
# limit above a target is half its width less this.
TARGET_MARGIN_M = 0.02
ROLL_MAX_DEG = 90.0
# How long a step waits for a wrist frame captured after the arm settled, in
# periods of the wrist camera as the brain measured it (`WristFrames.period_s`):
# the first frame to come can be one captured just before the arm settled, and
# a loaded host sends the next one late.
FRAME_WAIT_PERIODS = 2.5
# The least time a step waits for that frame, and its wait for a camera whose
# period the brain has not measured yet: it has sent one frame alone.
MIN_FRAME_WAIT_S = 2.5
NOT_STARTED = "the aligner has not been started"
WRIST = "wrist"
CHEST = "chest"

FRAME_WORDS = (
    'Picture 1, "wrist": the camera on the {arm} wrist, looking down past the jaws. '
    'Picture 2, "chest": the robot\'s chest camera, looking forward and down at the table. '
    "Where the robot's +x and +y point in each picture is given with the state.\n\n"
    "Robot frame: x forward, away from the robot; y to the robot's left; z up. Distances in centimetres. "
    "roll_deg turns the gripper about the vertical, positive counter-clockwise seen from above."
)
GRAB_GOAL = (
    "You align a robot gripper above an item before it grasps it. Goal: the gripper straight above the {label}, "
    "centred on it, turned so that its two jaws close across the narrowest width of the {label}, on two opposite flat "
    "faces when it has them.\n\n" + FRAME_WORDS
)
PLACE_GOAL = (
    "You guide a robot gripper that holds an item to the spot where it sets the item down. Goal: the held item straight "
    "above the spot the target point is in: the inside of a container (a bowl, a crate, a box) or a free patch of a "
    "surface, centred on it and clear of the container's rim. A container that already holds items is still the "
    "spot: the item may rest on them, so aim at a gap between them or at the lowest of them. The gripper then goes "
    "straight down until the item rests, and opens.\n\n" + FRAME_WORDS
)
PLACE_ANSWERS = " You have {answers} answers: make the whole move in the first, then answer ready."
INTO_GOAL = (
    "You guide a robot gripper that holds an item to a free spot of the {target}{width}. Goal: the held item straight "
    "above a free spot inside the {target}, clear of its rim. Items already in the {target} do not make it full: the "
    "item may rest on them, so aim at a gap between them or at the lowest of them. {after}\n\n" + FRAME_WORDS
)
AFTER_INTO = {
    True: "The gripper then goes straight down until the item rests, and opens.",
    False: "The jaws then open at this height, with no descent: the item falls into the {target}.",
}


@dataclass(frozen=True)
class Move:
    dx_m: float
    dy_m: float
    roll_deg: float


class ModelAligner:
    def __init__(
        self,
        model: StepModel,
        *,
        max_steps: int = MAX_STEPS,
        place_max_steps: int = PLACE_MAX_STEPS,
        step_max_m: float = STEP_MAX_M,
        radius_m: float = RADIUS_M,
        roll_max_deg: float = ROLL_MAX_DEG,
        frame_wait_periods: float = FRAME_WAIT_PERIODS,
        min_frame_wait_s: float = MIN_FRAME_WAIT_S,
        target_margin_m: float = TARGET_MARGIN_M,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.place_max_steps = place_max_steps
        self.step_max_m = step_max_m
        self.radius_m = radius_m
        self.roll_max_deg = roll_max_deg
        self.frame_wait_periods = frame_wait_periods
        self.min_frame_wait_s = min_frame_wait_s
        self.target_margin_m = target_margin_m
        self._senses = None

    def start(self, senses) -> None:
        self._senses = senses

    def frame_wait_s(self, period_s: Optional[float]) -> float:
        """How long a step waits for a wrist frame captured after the arm
        settled, from the period of the wrist camera: `frame_wait_periods`
        of it, and at least `min_frame_wait_s`, which is also the wait when
        the period is not known."""
        if period_s is None:
            return self.min_frame_wait_s
        return max(self.min_frame_wait_s, self.frame_wait_periods * period_s)

    def why_unavailable(self, arm: str) -> str:
        """Why `arm` cannot be aligned now, or the empty string: no wrist
        camera or no frame of it yet, or no chest picture."""
        if self._senses is None:
            return NOT_STARTED
        return self._senses.wrists.why_unavailable(arm) or self._senses.chest.why_unavailable()

    async def align(self, request: AlignRequest, moves: ArmMoves, cancel: CancelToken, reports: AlignReports) -> Alignment:
        if self._senses is None:
            raise AlignmentFailed(NOT_STARTED)
        steps = self.place_max_steps if request.is_place else self.max_steps
        run = _Run(self, self._senses, request, moves, reports, steps)
        goal = _goal(request)
        if request.is_place:
            goal = goal.replace("\n\n", PLACE_ANSWERS.format(answers=steps) + "\n\n", 1)
        loop = ToolLoop(self.model, run.tools, goal, steps, run.observe)
        end = await loop.run(cancel)
        if end.out_of_steps and request.is_place and run.moved_in_step:
            return Aligned(run.pre, run.point, run.lift, run.orientation, steps=end.steps)
        if end.out_of_steps:
            return OutOfSteps(end.steps)
        return replace(end.value, steps=end.steps)


def _goal(request: AlignRequest) -> str:
    """The goal text of an alignment: a grab, a place at a pose, or a place
    or a drop into a target item, told its label and width."""
    arm = request.gripper.arm.split("_")[0]
    if request.into is not None:
        target = request.into.label or "item"
        width = f", about {100 * request.into.width_m:.0f} cm wide" if request.into.width_m is not None else ""
        after = AFTER_INTO[request.descends].format(target=target)
        return INTO_GOAL.format(target=target, width=width, after=after, arm=arm)
    template = PLACE_GOAL if request.is_place else GRAB_GOAL
    return template.format(label=request.label, arm=arm)


def radius_of(aligner: ModelAligner, request: AlignRequest) -> float:
    """How far the gripper may stand from the centre of `request`: the
    aligner's radius, and above a target item of known width no more than
    half that width less the margin."""
    into = request.into
    if into is None or into.width_m is None:
        return aligner.radius_m
    return max(0.0, min(aligner.radius_m, into.width_m / 2 - aligner.target_margin_m))


class _Run:
    """One alignment of `steps` steps at most: the points as the steps move
    them, the tools over them, the observation of each step, and the step
    that runs, which each report names."""

    def __init__(
        self, aligner: ModelAligner, senses, request: AlignRequest, moves: ArmMoves, reports: AlignReports, steps: int
    ) -> None:
        self.aligner = aligner
        self.senses = senses
        self.request = request
        self.moves = moves
        self.reports = reports
        self.steps = steps
        # The number of the step that runs, from 1.
        self.step = 0
        self.pre = request.pre
        self.point = request.point
        self.lift = lifted(request.point, request.lift_m)
        self.orientation = request.orientation
        self.centre = request.centre
        self.radius_m = radius_of(aligner, request)
        # Whether the step that runs now moved the gripper to a new place.
        self.moved_in_step = False
        self.tools = (_MoveTool(self), _ReadyTool(self), _GiveUpTool(request))

    @property
    def arm(self) -> str:
        return self.request.gripper.arm

    async def observe(self, step: int) -> Observation:
        """The observation of step `step`, whose question starts here."""
        self.step = step
        self.moved_in_step = False
        await self.reports.align_look(step, self.steps)
        try:
            return await self._observe()
        except Refusal as refusal:
            raise AlignmentFailed(refusal.message) from None

    async def _observe(self) -> Observation:
        senses = self.senses
        wrists = senses.wrists
        camera = wrists.camera_of(self.arm)
        wait_s = self.aligner.frame_wait_s(wrists.period_s(self.arm))
        frame = await wrists.newer_than(self.arm, wrists.latest_stamp(self.arm), wait_s)
        if frame is None:
            raise AlignmentFailed(f"no frame from wrist camera '{camera}' after the move")
        chest_frame = senses.chest.frames.latest().color
        # Each camera where the frame its picture names stood when the
        # picture was taken.
        wrist_pose = await senses.camera_pose(frame.header)
        chest_pose = await senses.camera_pose(chest_frame.header)
        measured = self._measured()
        wrist_x, wrist_y = axes_in_picture(wrist_pose.orientation.xyzw)
        chest_x, chest_y = axes_in_picture(chest_pose.orientation.xyzw)
        where = self.where(measured.position)
        above_cm = 100 * (measured.position[2] - self.point[2])
        into = self.request.into
        if into is not None and not self.request.descends:
            height = "; the jaws open at this height"
        elif into is not None:
            height = f", {above_cm:.1f} cm above the bottom of the {into.label or 'item'}"
        elif self.request.is_place:
            height = f", {above_cm:.1f} cm above the height it releases at"
        else:
            height = f", {above_cm:.1f} cm above the height it closes at"
        if self.request.is_place:
            jaws = f"The jaws hold the item at {measured.opening:.2f} (1.00 is about 15 cm)."
        else:
            jaws = f"Jaws open at {measured.opening:.2f} (1.00 is about 15 cm)."
        state = (
            f"{where[0].upper()}{where[1:]}{height}. {jaws} In the wrist picture the robot's +x points toward {wrist_x} o'clock "
            f"and +y toward {wrist_y} o'clock (12 is up); in the chest picture +x toward {chest_x} o'clock and +y toward "
            f"{chest_y} o'clock."
        )
        return Observation(state, (Picture(WRIST, jpeg_of(frame)), Picture(CHEST, jpeg_of(chest_frame))))

    def where(self, position: Optional[Vec3] = None, *, now: bool = False) -> str:
        """Where the grasp point stands against the item's seen centre and
        how the jaws lie, from the measured `position` or the latest one;
        with `now`, as the history tells it after a step."""
        if position is None:
            position = self._measured().position
        dx = 100 * (position[0] - self.centre[0])
        dy = 100 * (position[1] - self.centre[1])
        return (
            f"the grasp point is{' now' if now else ''} {dx:+.1f} cm in x and {dy:+.1f} cm in y from {self.request.centre_words}; "
            f"the jaws close along a line {jaw_line_deg(self.orientation):.0f} degrees counter-clockwise from +x"
        )

    def _measured(self) -> "_Measured":
        snapshot = self.senses.limb_states.latest
        if snapshot is None:
            raise AlignmentFailed(self.senses.limb_states.why_unavailable())
        arm = snapshot.arms.get(self.arm)
        opening = snapshot.opening(self.request.gripper.name)
        if arm is None or opening is None:
            raise AlignmentFailed(f"limb_states names no arm '{self.arm}' or no gripper '{self.request.gripper.name}'")
        return _Measured(arm.position, opening)


@dataclass(frozen=True)
class _Measured:
    position: Vec3
    opening: float


class _MoveTool:
    name = "move"
    fields = {"dx": "number", "dy": "number", "roll_deg": "number"}

    def __init__(self, run: _Run) -> None:
        self.alignment = run
        a = run.aligner
        if run.request.is_place:
            reach = f"One move goes anywhere at once and turns at most {a.roll_max_deg:.0f} degrees"
        else:
            reach = f"One step moves at most {100 * a.step_max_m:.0f} cm and turns at most {a.roll_max_deg:.0f} degrees"
        self.description = (
            f"moves the gripper sideways by dx and dy and turns it by roll_deg, at the same height. {reach}; the gripper "
            f"stays within {100 * run.radius_m:.0f} cm of {run.request.centre_words}."
        )

    def parse(self, fields: Mapping[str, object]) -> Move:
        values = []
        for name in ("dx", "dy", "roll_deg"):
            value = fields.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a number")
            values.append(float(value))
        return Move(values[0] / 100.0, values[1] / 100.0, values[2])

    async def run(self, move: Move, cancel: CancelToken) -> ToolResult:
        run, a = self.alignment, self.alignment.aligner
        step_cm = 100 * math.hypot(move.dx_m, move.dy_m)
        if not run.request.is_place and step_cm > 100 * a.step_max_m + 1e-9:
            return ToolResult(f"moved nothing: a step moves at most {100 * a.step_max_m:.0f} cm, this one is {step_cm:.1f} cm")
        if abs(move.roll_deg) > a.roll_max_deg + 1e-9:
            return ToolResult(f"moved nothing: one step turns at most {a.roll_max_deg:.0f} degrees")
        pre = (run.pre[0] + move.dx_m, run.pre[1] + move.dy_m, run.pre[2])
        off_cm = 100 * math.hypot(pre[0] - run.centre[0], pre[1] - run.centre[1])
        if off_cm > 100 * run.radius_m + 1e-9:
            return ToolResult(
                f"moved nothing: the gripper stays within {100 * run.radius_m:.0f} cm of the item, this step ends {off_cm:.1f} cm from it"
            )
        orientation = turned_about_z(run.orientation, math.radians(move.roll_deg))
        point = (pre[0], pre[1], run.point[2])
        lift = lifted(point, run.request.lift_m)
        checked = (("the pre-grasp point", pre), ("the grasp point", point), ("the lifted point", lift))
        if not run.request.descends:
            checked = (("the pre-place point", pre),)
        for name, target in checked:
            refusal = await run.moves.refusal_for(run.arm, (target,), orientation, cancel)
            if refusal:
                return ToolResult(f"moved nothing: the backbone has no plan for {name}: {refusal}")
        await run.reports.align_move(run.step, run.steps)
        moved = await run.moves.move_to(run.arm, pre, orientation)
        cancel.check()
        if moved.cancelled:
            message = moved.message or "the move was stopped"
            return ToolResult(f"the move was stopped: {message}", ends=True, value=Stopped(message))
        if not moved.completed:
            return ToolResult(f"the move failed: {moved.message}; {run.where(now=True)}")
        run.pre, run.point, run.lift, run.orientation = pre, point, lift, orientation
        run.moved_in_step = True
        return ToolResult(f"moved: {run.where(now=True)}")


class _ReadyTool:
    name = "ready"
    fields: dict[str, str] = {}

    def __init__(self, run: _Run) -> None:
        self.alignment = run
        into = run.request.into
        if into is not None and not run.request.descends:
            self.description = (
                f"the held item is centred above a free spot of the {into.label or 'item'}: the jaws open at this height "
                "and the item falls into it."
            )
        elif into is not None:
            self.description = (
                f"the held item is centred above a free spot of the {into.label or 'item'}: the gripper goes straight down "
                "until the item rests, opens and climbs back."
            )
        elif run.request.is_place:
            self.description = (
                "the held item is centred above its spot and turned right: the gripper goes straight down until the "
                "item rests or the target point is reached, opens and climbs back."
            )
        else:
            self.description = (
                f"the gripper is centred above the {run.request.label} and turned right: it goes straight down, closes and lifts."
            )

    def parse(self, fields: Mapping[str, object]) -> None:
        return None

    async def run(self, arguments: None, cancel: CancelToken) -> ToolResult:
        run = self.alignment
        return ToolResult("ready: the grab goes on", ends=True, value=Aligned(run.pre, run.point, run.lift, run.orientation))


class _GiveUpTool:
    name = "give_up"
    fields: dict[str, str] = {}

    def __init__(self, request: AlignRequest) -> None:
        if request.into is not None:
            self.description = (
                f"no free spot of the {request.into.label or 'item'} is within reach of the moves: items already in it are "
                "not such a case, the item may rest on them."
            )
        elif request.is_place:
            self.description = (
                "no spot within reach of the moves can take the item: a container with items in it is not such a case, "
                "the item may rest on them."
            )
        else:
            self.description = f"a grasp from above cannot take the {request.label}."

    def parse(self, fields: Mapping[str, object]) -> None:
        return None

    async def run(self, arguments: None, cancel: CancelToken) -> ToolResult:
        return ToolResult("gave up", ends=True, value=GaveUp())


# The actions of an alignment, the names a step model's probe may answer.
ACTIONS = (_MoveTool.name, _ReadyTool.name, _GiveUpTool.name)
