"""The alignment step of a grab and of a place: a model looks and corrects
the gripper's place and turn above the item, or above the spot the held
item goes to, before the descent.

The scripted grab brings the open gripper to the pre-grasp point, the
scripted place brings the held item to the pre-place point, and each then
hands the aligner the gripper, the points of the descent and the
orientation it stands in (`AlignRequest`): with the item to grasp for a
grab, with the target point alone for a place. The aligner runs a
`ToolLoop` with three tools: `move` (sideways by dx and dy, in
centimetres, and a turn about the vertical by roll_deg, counter-clockwise
seen from above, at the same height), `ready` (centred and turned: go
down) and `give_up`. Each step the model sees the picture of the wrist
camera of the gripper's arm, captured after the arm settled, the chest
picture, and the state as text: where the grasp point stands against the
item's seen centre or the target point, how high above the close or the
release, the line the jaws close along, the jaws' opening, and where the
robot's +x and +y point in each picture, as clock hours.

The code holds the limits, not the model. A step moves at most
`step_max_m`, turns at most `roll_max_deg`, keeps the gripper within
`radius_m` of the item's seen centre or the target point, and never
changes the height. A step over a limit, or one the backbone has no plan
for, moves nothing: it counts as a step and its reason goes into the next
step's history, as does an answer the loop cannot read. The three points
of the descent (pre, point, lifted) are checked before each move, so the
descent that follows a `ready` has a plan.

The alignment ends `Aligned` with the new points and orientation, `GaveUp`,
`OutOfSteps` after `max_steps`, or `Stopped` when a stop ended a move; a
cancel raises `Cancelled` from the loop. `AlignmentFailed` names what the
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

from .directions import axes_in_picture
from .loop import Observation, Picture, StepModel, ToolLoop, ToolResult

MAX_STEPS = 5
STEP_MAX_M = 0.03
RADIUS_M = 0.10
ROLL_MAX_DEG = 90.0
# How long a step waits for a wrist frame captured after the arm settled.
FRAME_WAIT_S = 1.0
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
        step_max_m: float = STEP_MAX_M,
        radius_m: float = RADIUS_M,
        roll_max_deg: float = ROLL_MAX_DEG,
        frame_wait_s: float = FRAME_WAIT_S,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.step_max_m = step_max_m
        self.radius_m = radius_m
        self.roll_max_deg = roll_max_deg
        self.frame_wait_s = frame_wait_s
        self._senses = None

    def start(self, senses) -> None:
        self._senses = senses

    def why_unavailable(self, arm: str) -> str:
        """Why `arm` cannot be aligned now, or the empty string: no wrist
        camera or no frame of it yet, or no chest picture."""
        if self._senses is None:
            return NOT_STARTED
        return self._senses.wrists.why_unavailable(arm) or self._senses.chest.why_unavailable()

    async def align(self, request: AlignRequest, moves: ArmMoves, cancel: CancelToken) -> Alignment:
        if self._senses is None:
            raise AlignmentFailed(NOT_STARTED)
        run = _Run(self, self._senses, request, moves)
        template = PLACE_GOAL if request.is_place else GRAB_GOAL
        goal = template.format(label=request.label, arm=request.gripper.arm.split("_")[0])
        loop = ToolLoop(self.model, run.tools, goal, self.max_steps, run.observe)
        end = await loop.run(cancel)
        if end.out_of_steps:
            return OutOfSteps(end.steps)
        return replace(end.value, steps=end.steps)


class _Run:
    """One alignment: the points as the steps move them, the tools over
    them, and the observation of each step."""

    def __init__(self, aligner: ModelAligner, senses, request: AlignRequest, moves: ArmMoves) -> None:
        self.aligner = aligner
        self.senses = senses
        self.request = request
        self.moves = moves
        self.pre = request.pre
        self.point = request.point
        self.lift = lifted(request.point, request.lift_m)
        self.orientation = request.orientation
        self.centre = request.centre
        self.tools = (_MoveTool(self), _ReadyTool(self), _GiveUpTool(request))

    @property
    def arm(self) -> str:
        return self.request.gripper.arm

    async def observe(self) -> Observation:
        try:
            return await self._observe()
        except Refusal as refusal:
            raise AlignmentFailed(refusal.message) from None

    async def _observe(self) -> Observation:
        senses = self.senses
        wrists = senses.wrists
        camera = wrists.camera_of(self.arm)
        frame = await wrists.newer_than(self.arm, wrists.latest_stamp(self.arm), self.aligner.frame_wait_s)
        if frame is None:
            raise AlignmentFailed(f"no frame from wrist camera '{camera}' after the move")
        chest_frame = senses.chest.frames.latest().color
        _position, wrist_orientation = await senses.camera_pose(camera)
        chest_pose = senses.chest.camera.pose
        if chest_pose is None:
            raise AlignmentFailed("the chest camera's pose is not known yet")
        measured = self._measured()
        wrist_x, wrist_y = axes_in_picture(wrist_orientation)
        chest_x, chest_y = axes_in_picture(chest_pose.orientation)
        where = self.where(measured.position)
        if self.request.is_place:
            height = "releases at"
            jaws = f"The jaws hold the item at {measured.opening:.2f} (1.00 is about 15 cm)."
        else:
            height = "closes at"
            jaws = f"Jaws open at {measured.opening:.2f} (1.00 is about 15 cm)."
        state = (
            f"{where[0].upper()}{where[1:]}, {100 * (measured.position[2] - self.point[2]):.1f} cm above the height it "
            f"{height}. {jaws} In the wrist picture the robot's +x points toward {wrist_x} o'clock and +y toward "
            f"{wrist_y} o'clock (12 is up); in the chest picture +x toward {chest_x} o'clock and +y toward {chest_y} o'clock."
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
        self.description = (
            f"moves the gripper sideways by dx and dy and turns it by roll_deg, at the same height. One step moves at "
            f"most {100 * a.step_max_m:.0f} cm and turns at most {a.roll_max_deg:.0f} degrees; the gripper stays within "
            f"{100 * a.radius_m:.0f} cm of {run.request.centre_words}."
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
        if step_cm > 100 * a.step_max_m + 1e-9:
            return ToolResult(f"moved nothing: a step moves at most {100 * a.step_max_m:.0f} cm, this one is {step_cm:.1f} cm")
        if abs(move.roll_deg) > a.roll_max_deg + 1e-9:
            return ToolResult(f"moved nothing: one step turns at most {a.roll_max_deg:.0f} degrees")
        pre = (run.pre[0] + move.dx_m, run.pre[1] + move.dy_m, run.pre[2])
        off_cm = 100 * math.hypot(pre[0] - run.centre[0], pre[1] - run.centre[1])
        if off_cm > 100 * a.radius_m + 1e-9:
            return ToolResult(
                f"moved nothing: the gripper stays within {100 * a.radius_m:.0f} cm of the item, this step ends {off_cm:.1f} cm from it"
            )
        orientation = turned_about_z(run.orientation, math.radians(move.roll_deg))
        point = (pre[0], pre[1], run.point[2])
        lift = lifted(point, run.request.lift_m)
        for name, target in (("the pre-grasp point", pre), ("the grasp point", point), ("the lifted point", lift)):
            refusal = await run.moves.refusal_for(run.arm, (target,), orientation, cancel)
            if refusal:
                return ToolResult(f"moved nothing: the backbone has no plan for {name}: {refusal}")
        moved = await run.moves.move_to(run.arm, pre, orientation)
        cancel.check()
        if moved.cancelled:
            message = moved.message or "the move was stopped"
            return ToolResult(f"the move was stopped: {message}", ends=True, value=Stopped(message))
        if not moved.completed:
            return ToolResult(f"the move failed: {moved.message}; {run.where(now=True)}")
        run.pre, run.point, run.lift, run.orientation = pre, point, lift, orientation
        return ToolResult(f"moved: {run.where(now=True)}")


class _ReadyTool:
    name = "ready"
    fields: dict[str, str] = {}

    def __init__(self, run: _Run) -> None:
        self.alignment = run
        if run.request.is_place:
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
        if request.is_place:
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
