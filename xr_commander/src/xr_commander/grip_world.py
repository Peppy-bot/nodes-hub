"""The headset's grips, the robots it holds, its posture move and the fleet
facts, as one world that each event steps.

A squeeze on a robot that the headset does not drive yet takes the robot.
The headset takes its lease, or, while another client opened its lease for a
take-over, asks the daemon whether it may move it. It then stops the planned
moves in flight on the robot, and only then streams. A press of a posture
button takes the robot the same way for the time of its move.

The headset holds a robot while a hand drives its arm or its base, through
the release burst of the last of them, and while a posture move of its own
runs; then it releases the lease. It lets go at once, as at a grip release,
when its lease ends, at a halt, when the open lease it drives under closes
or ends, and when a goal of the fleet ends. A lease that the newest listing
shows the headset holding on a robot it neither holds nor works on is
released, so a take the daemon served late does not stay with the headset.
One squeeze is one claim: a squeeze held through a let-go of the chosen
robot, a refusal, a change of robot, the loss of its pair or a halt drives
nothing until it is released.

`step` is a pure function of the world and the event: the shell (`drive`)
turns each effect into a call, a timer, a note or a refresh, and each
outcome into an event.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Generic, Literal, TypeVar

from xr_commander.bus import Answered, Failed, Unanswered
from xr_commander.devices import Hand
from xr_commander.fleet import (
    DAEMON_SILENT,
    FleetView,
    Following,
    GoalsStatus,
    Holder,
    Starting,
    halted_text,
    status_text,
)
from xr_commander.robots import JOIN_COMMAND, SlotPairs

# What a grip drives: a hand's arm and gripper, or the base, which either
# hand's grip drives.
GripName = Hand | Literal["base"]
BASE: GripName = "base"
GRIP_NAMES: tuple[GripName, ...] = (Hand.LEFT, Hand.RIGHT, BASE)
# Why the headset holds a robot: a grip that drives it, or a posture move.
Reason = GripName | Literal["posture"]
POSTURE: Reason = "posture"
# Where a note shows: a grip's row, the robot row, or the posture row.
NoteTarget = GripName | Literal["robot", "posture"]
ROBOT_NOTE: NoteTarget = "robot"
POSTURE_NOTE: NoteTarget = "posture"
NOTE_TARGETS: tuple[NoteTarget, ...] = (*GRIP_NAMES, ROBOT_NOTE, POSTURE_NOTE)

# Why a press does nothing while a grip holds the chosen robot.
GRIP_HELD = "release the grip first"
GRIP_JUST_RELEASED = "the grip was just released"
# Why an op ends when the goals of the fleet it started under ended.
GOALS_ENDED = "a goal of the fleet ended"
# The cause of a lease end whose feedback gives none.
LEASE_ENDED = "the lease ended"
# Why a posture start whose goal send got no answer refuses: the move may run.
GOAL_UNSENT = "the goal could not be sent (see log)"
# Why the headset lets go of every robot as the node stops.
NODE_STOPS = "the node stops"


class Reading(StrEnum):
    """A grip as one tick reads it: squeezed, released, or unknown while no
    fresh frame tracks it."""

    SQUEEZE = "squeeze"
    RELEASED = "released"
    UNKNOWN = "unknown"


def reading_of(squeezing: bool | None) -> Reading:
    """A hand's reading from a fresh sample's grip, None with no fresh sample
    or no tracked hand."""
    match squeezing:
        case None:
            return Reading.UNKNOWN
        case True:
            return Reading.SQUEEZE
        case False:
            return Reading.RELEASED


class Slot(StrEnum):
    """A pairing slot a grip drives: an arm grip's arm and gripper slots, the
    base grip's base slot."""

    ARM = "arm"
    GRIPPER = "gripper"
    BASE = "base"


@dataclass(frozen=True)
class Pin:
    """A pair on one slot; `peer` is the pair's `PeerInfo`, compared by
    value."""

    slot: Slot
    peer: Hashable


Pins = frozenset[Pin]


def slot_pins(reads: Mapping[Slot, SlotPairs], chosen: str | None) -> tuple[Pins, Pins]:
    """One tick's reads of a grip's slots: every pair on them, and the chosen
    robot's first pair in delivered order on each slot that has one."""
    peers = frozenset(
        Pin(slot, peer) for slot, read in reads.items() for peer in read.peers
    )
    pins = frozenset(
        Pin(slot, peer)
        for slot, read in reads.items()
        if (peer := read.pin_of(chosen)) is not None
    )
    return peers, pins


def pin_on(pins: Pins, slot: Slot) -> Hashable | None:
    """The pair of `pins` on `slot`, None while it pins none there."""
    return next((pin.peer for pin in pins if pin.slot is slot), None)


@dataclass(frozen=True)
class Ready:
    """The grip may request its robot."""


@dataclass(frozen=True)
class NotReady:
    """The grip may not request; `why` is its note's cause."""

    why: str


Readiness = Ready | NotReady


@dataclass(frozen=True)
class TickInputs:
    """Everything one tick of a grip reads, read once before its step."""

    # The reading of each hand of the grip: an arm grip's own, both for the
    # base.
    readings: Mapping[Hand, Reading]
    chosen: str | None
    # Every pair of the grip's slots, and the chosen robot's pin on each.
    peers: Pins
    pins: Pins
    # The grip's readiness on `pins`, which a request reads.
    ready: Readiness
    # The grip's readiness on the pins of its phase at the tick, which a
    # rejoin reads: an `Engaged` or `Lingering` phase's pins, none for the
    # others.
    phase_ready: Readiness

    def __post_init__(self) -> None:
        # Frozen in depth, as `devices.XrFrame` holds its hands.
        object.__setattr__(self, "readings", MappingProxyType(dict(self.readings)))
        assert self.pins <= self.peers, "a pin is one of the slots' pairs"
        assert len({pin.slot for pin in self.pins}) == len(self.pins), (
            "one pin per slot"
        )
        assert isinstance(self.ready, NotReady) or self.pins, "ready needs a pin"
        assert self.chosen is not None or not self.pins, "no robot pins nothing"


@dataclass(frozen=True)
class PostureTarget:
    """The postures producer of a robot as a press reads it once: `producer`
    for the goal, `instance` for the daemon's check."""

    producer: object
    instance: str


@dataclass(frozen=True)
class PressInputs:
    """Everything a posture press reads, read once at its rising edge."""

    # A fresh sample of either hand squeezes.
    squeezing: bool
    chosen: str | None
    target: PostureTarget | None


# A value kept per hand.
HandValue = TypeVar("HandValue")


@dataclass(frozen=True)
class ByHand(Generic[HandValue]):
    """A value for each hand."""

    left: HandValue
    right: HandValue

    def of(self, hand: Hand) -> HandValue:
        return getattr(self, str(hand))

    def with_hand(self, hand: Hand, value: HandValue) -> ByHand[HandValue]:
        return replace(self, **{str(hand): value})


@dataclass(frozen=True)
class HandState:
    """One hand as the world keeps it: its last reading that was not
    `UNKNOWN`, None before the first, and whether its squeeze counts for no
    grip until a fresh `RELEASED`."""

    last_fresh: Reading | None = None
    must_release: bool = False


def hands_of(grip: GripName) -> tuple[Hand, ...]:
    """A grip's hands: an arm grip's own, both for the base."""
    return (Hand.LEFT, Hand.RIGHT) if grip == BASE else (Hand(grip),)


@dataclass(frozen=True)
class Idle:
    """The grip drives nothing and asks for nothing."""


@dataclass(frozen=True)
class Requesting:
    """Request `request` of the grip is in `robot`'s acquisition; `pins` are
    the pins its latest tick read."""

    robot: str
    request: int
    pins: Pins


@dataclass(frozen=True)
class Engaged:
    """The grip drives `pins` under the hold of `robot` at `epoch`."""

    robot: str
    pins: Pins
    epoch: int


@dataclass(frozen=True)
class Lingering:
    """The stream's release burst to `pins` runs; the grip's reason stays in
    the hold at `epoch`."""

    robot: str
    pins: Pins
    epoch: int


Phase = Idle | Requesting | Engaged | Lingering


def robot_of_phase(phase: Phase) -> str | None:
    """The robot a phase is on, None for `Idle`."""
    match phase:
        case Idle():
            return None
        case Requesting(robot=robot) | Engaged(robot=robot) | Lingering(robot=robot):
            return robot


def driven_pins(phase: Phase) -> Pins:
    """The pins a stream sends to: an `Engaged` or `Lingering` phase's, else
    none."""
    match phase:
        case Engaged(pins=pins) | Lingering(pins=pins):
            return pins
        case _:
            return frozenset()


@dataclass(frozen=True)
class Grip:
    """One grip as the world keeps it: its phase, and the last reading other
    than `UNKNOWN` that its own ticks took of each hand, which gives it its
    own rising edge."""

    phase: Phase = Idle()
    seen: ByHand[Reading | None] = ByHand(None, None)


@dataclass(frozen=True)
class Grips:
    """The three grips, by name."""

    left: Grip = Grip()
    right: Grip = Grip()
    base: Grip = Grip()

    def of(self, name: GripName) -> Grip:
        return getattr(self, str(name))

    def with_grip(self, name: GripName, grip: Grip) -> Grips:
        return replace(self, **{str(name): grip})

    def items(self) -> tuple[tuple[GripName, Grip], ...]:
        return tuple((name, self.of(name)) for name in GRIP_NAMES)


@dataclass(frozen=True)
class Taken:
    """The headset holds the lease of the robot."""


@dataclass(frozen=True)
class UnderOpen:
    """The headset drives under the lease that `holder` opened."""

    holder: str


Mode = Taken | UnderOpen


@dataclass(frozen=True)
class PendingEnd:
    """A lease end that the daemon answered with a yes at `revision`: the
    first listing that recent tells whether it ended the headset's lease, for
    the cause `why`."""

    revision: int
    why: str


@dataclass(frozen=True)
class Brought:
    """`reason` is in a hold by request or posture start `by`."""

    reason: Reason
    by: int


@dataclass(frozen=True)
class AcceptedGoal:
    """The accepted goal of posture start `start`, to `label`."""

    start: int
    label: str
    goal: object


@dataclass(frozen=True)
class Hold:
    """The headset's hold on a robot: its mode, why it holds it, and the
    epoch it began at."""

    mode: Mode
    reasons: frozenset[Brought]
    epoch: int
    pending_end: PendingEnd | None = None
    goal: AcceptedGoal | None = None

    def __post_init__(self) -> None:
        assert self.reasons, "a hold has a reason"
        assert len({b.reason for b in self.reasons}) == len(self.reasons), (
            "a reason once"
        )
        assert self.goal is None or self.brought_by(POSTURE) == self.goal.start, (
            "a goal belongs to the posture start in the hold"
        )

    def has(self, reason: Reason) -> bool:
        """Whether `reason` is in the hold."""
        return any(brought.reason == reason for brought in self.reasons)

    def brought_by(self, reason: Reason) -> int | None:
        """The request or start that brought `reason`, None while it is
        absent."""
        return next((b.by for b in self.reasons if b.reason == reason), None)

    def with_reason(self, reason: Reason, by: int) -> Hold:
        """The hold with `reason` brought by `by`; asserts `reason` is
        absent."""
        assert not self.has(reason), "a reason joins a hold once"
        return replace(self, reasons=self.reasons | {Brought(reason, by)})

    def without(self, reason: Reason) -> Hold | None:
        """The hold without `reason`, None when it was the last; `POSTURE`
        takes `goal` with it."""
        reasons = frozenset(b for b in self.reasons if b.reason != reason)
        if not reasons:
            return None
        goal = None if reason == POSTURE else self.goal
        return replace(self, reasons=reasons, goal=goal)


@dataclass(frozen=True)
class Requester:
    """Request `request` of grip `grip`."""

    grip: GripName
    request: int


class TakeStage(StrEnum):
    """Where a `Take` is: its admit call, then its stop."""

    ADMIT = "admit"
    STOP = "stop"


@dataclass(frozen=True)
class Take:
    """An acquisition with no hold: `take(r)`, or `check_move(r)` under
    another holder's open lease, then `stop(r)`."""

    id: int
    epoch: int
    generation: int
    mode: Mode
    requesters: tuple[Requester, ...]
    stage: TakeStage
    call: int
    took: bool = False


class JoinStage(StrEnum):
    """Where a `Join` is: its move check, then its stop."""

    CHECK = "check"
    STOP = "stop"


@dataclass(frozen=True)
class Join:
    """An acquisition beside a hold that needs calls: `check_move(r)` unless
    a posture move was in the hold at its start, then `stop(r)`."""

    id: int
    epoch: int
    generation: int
    mode: Mode
    requesters: tuple[Requester, ...]
    stage: JoinStage
    call: int


class PostureStage(StrEnum):
    """Where a `PostureStart` is: the steps of a `Take`, then the instance
    check and the goal."""

    ADMIT = "admit"
    STOP = "stop"
    CHECK_INSTANCE = "check_instance"
    FIRE = "fire"


@dataclass(frozen=True)
class PostureStart:
    """A press's start: the steps of a `Take` for the reason `POSTURE`, then
    the instance check and the goal."""

    # The press's start id, as `PostureMoving.start` names it.
    id: int
    epoch: int
    generation: int
    label: str
    target: PostureTarget
    mode: Mode
    stage: PostureStage
    call: int
    took: bool = False
    # The why of the first let-go of the robot while the start runs.
    let_go: str | None = None


@dataclass(frozen=True)
class Release:
    """`release(r)` of a lease the headset may hold."""

    id: int
    call: int


@dataclass(frozen=True)
class Confirm:
    """`check_move(r)` after a lease end, for the cause `why`."""

    id: int
    epoch: int
    generation: int
    why: str
    call: int


Op = Take | Join | PostureStart | Release | Confirm


@dataclass(frozen=True)
class QueuedPosture:
    """A press waiting for its turn on the robot."""

    start: int
    label: str
    target: PostureTarget


@dataclass(frozen=True)
class QueuedConfirm:
    """A lease end waiting for its turn on the robot."""

    why: str


Queued = Requester | QueuedPosture | QueuedConfirm


@dataclass(frozen=True)
class Robot:
    """One robot as the world keeps it."""

    name: str
    hold: Hold | None = None
    op: Op | None = None
    queue: tuple[Queued, ...] = ()
    release_owed: bool = False
    # The timer of a release that waits for its `RetryDue`.
    retry: int | None = None
    # The call of a stop of a posture move that may run, outside the op slot.
    lost_stop: int | None = None
    epoch: int = 0
    # The world's `listing_seq` when the robot's last `Release` started.
    release_seq: int = 0


@dataclass(frozen=True)
class FleetFacts:
    """The fleet as the headset knows it: the newest listing while one
    stands, the starts of the goals, and their status."""

    me: Holder
    view: FleetView | None = None
    generation: int = 0
    status: GoalsStatus = Starting()

    @property
    def goals_run(self) -> bool:
        return isinstance(self.status, Following)

    @property
    def halted(self) -> bool:
        return self.view is not None and self.view.halted_by is not None

    def down_cause(self) -> str | None:
        """Why no squeeze counts and no press starts now: the goals' status
        text while they do not run, the halt while the newest listing halts
        the stack, None otherwise."""
        if not self.goals_run:
            text = status_text(self.status)
            assert text is not None, "a status other than following has a text"
            return text
        if self.halted:
            return halted_text(self.view.halted_by)
        return None

    def listed(self) -> FleetView:
        """The newest listing, which stands at every event that reads it."""
        assert self.view is not None, "a listing stands while the goals run"
        return self.view


@dataclass(frozen=True)
class PostureMoving:
    """The headset's posture move, from its press to its end: its label, its
    start id, its robot, and the epoch of its hold once it holds."""

    label: str
    start: int
    robot: str
    epoch: int | None = None


@dataclass(frozen=True)
class World:
    """The headset's grips, robots, posture move and fleet facts."""

    fleet: FleetFacts
    hands: ByHand[HandState] = ByHand(HandState(), HandState())
    grips: Grips = Grips()
    # The chosen robot at the last tick of any grip, and whether one ever
    # was.
    seen_robot: str | None = None
    had_chosen: bool = False
    # Every robot the world holds state for, in name order.
    robots: tuple[Robot, ...] = ()
    posture_moving: PostureMoving | None = None
    # The number of listings stepped so far.
    listing_seq: int = 0
    next_id: int = 0

    @classmethod
    def start(cls, me: Holder) -> World:
        """The world at the node's start: no listing, the goals starting."""
        return cls(FleetFacts(me))

    def robot(self, name: str) -> Robot:
        """`name`'s state, a fresh `Robot` for one the world holds none of."""
        return next((robot for robot in self.robots if robot.name == name), Robot(name))

    def with_robot(self, robot: Robot) -> World:
        """The world with `robot` in place, in name order."""
        others = tuple(kept for kept in self.robots if kept.name != robot.name)
        ordered = tuple(sorted((*others, robot), key=lambda kept: kept.name))
        return replace(self, robots=ordered)


@dataclass(frozen=True)
class TakeLease:
    """`take(r)`: the lease of the robot under the headset's lease name."""


@dataclass(frozen=True)
class CheckMove:
    """`check_move(r)`: whether the headset may move the robot."""


@dataclass(frozen=True)
class StopMoves:
    """`stop(r)`: the planned moves in flight on the robot stop."""


@dataclass(frozen=True)
class ReleaseLease:
    """`release(r)`: the lease of the robot goes back."""


@dataclass(frozen=True)
class CheckMoveInstance:
    """`check_move` of the posture producer `instance` the goal goes to."""

    instance: str


@dataclass(frozen=True)
class FireGoal:
    """The posture goal to `label` at `target`."""

    label: str
    target: PostureTarget


CallRequest = (
    TakeLease | CheckMove | StopMoves | ReleaseLease | CheckMoveInstance | FireGoal
)


@dataclass(frozen=True)
class Raised:
    """A call that raised: it may have reached its peer. `cause` is the
    exception's `repr`."""

    cause: str


# `bus.Answered`, `bus.Failed` (a refused call) and `bus.Unanswered` are the
# other outcomes.
CallOutcome = Answered | Failed | Unanswered | Raised


@dataclass(frozen=True)
class Finished:
    """A posture result came; `note` is the posture row's text."""

    note: str


@dataclass(frozen=True)
class ResultLost:
    """A posture result did not come."""


PostureOutcome = Finished | ResultLost


@dataclass(frozen=True)
class Tick:
    """A tick of the grip's stream, with what it read before its step."""

    grip: GripName
    inputs: TickInputs


@dataclass(frozen=True)
class Streamed:
    """The same tick, after its burst stepped: whether the burst runs."""

    grip: GripName
    burst_running: bool


@dataclass(frozen=True)
class CallDone:
    """Call `call` of `robot` ended with `outcome`."""

    robot: str
    call: int
    outcome: CallOutcome


@dataclass(frozen=True)
class RetryDue:
    """The timer `timer` of an unanswered release of `robot` ran out."""

    robot: str
    timer: int


@dataclass(frozen=True)
class Listing:
    """A fleet listing of the changes goal."""

    view: FleetView


@dataclass(frozen=True)
class LeaseEnded:
    """The lease goal's feedback ended a lease of the headset on `robot`."""

    robot: str
    why: str


@dataclass(frozen=True)
class GoalsStatusChanged:
    """A write of the goals' status changed it."""

    status: GoalsStatus


@dataclass(frozen=True)
class GoalsEnded:
    """A goal of the fleet ended, for the cause `why`."""

    why: str


@dataclass(frozen=True)
class GoalsStarted:
    """The goals of the fleet started for the `generation`th time."""

    generation: int


@dataclass(frozen=True)
class Press:
    """A rising edge of the posture button of `label`."""

    label: str
    inputs: PressInputs


@dataclass(frozen=True)
class PostureResult:
    """The watcher of posture start `start`'s goal: its result, or none."""

    robot: str
    start: int
    outcome: PostureOutcome


@dataclass(frozen=True)
class Shutdown:
    """The node stops: the world's last event."""


Event = (
    Tick
    | Streamed
    | CallDone
    | RetryDue
    | Listing
    | LeaseEnded
    | GoalsStatusChanged
    | GoalsEnded
    | GoalsStarted
    | Press
    | PostureResult
    | Shutdown
)


@dataclass(frozen=True)
class Call:
    """Run call `call` of `robot`; its outcome comes back as `CallDone`."""

    robot: str
    call: int
    request: CallRequest


@dataclass(frozen=True)
class Watch:
    """Watch the accepted goal of posture start `start` for its result."""

    robot: str
    start: int
    label: str
    goal: object


@dataclass(frozen=True)
class CancelGoal:
    """Cancel the posture goal to `label` without waiting on it."""

    robot: str
    label: str
    goal: object


@dataclass(frozen=True)
class Note:
    """Flash `text` on the row of `target` for a few seconds, and log it."""

    target: NoteTarget
    text: str


@dataclass(frozen=True)
class StartRetry:
    """Step `RetryDue(robot, timer)` once the retry's time passed."""

    robot: str
    timer: int


@dataclass(frozen=True)
class Changed:
    """A hold or the fleet facts changed: the page and the panel refresh."""


Effect = Call | Watch | CancelGoal | Note | StartRetry | Changed
Effects = tuple[Effect, ...]


def step(world: World, event: Event) -> tuple[World, Effects]:
    """The world after `event`, and the effects to run in order. Every step
    but `Shutdown`'s ends with the lease reconcile and the next ops."""
    after, applied = _apply(world, event)
    after, followed = _follow_hands(after, note=None)
    started: Effects = ()
    if not isinstance(event, Shutdown):
        after, started = _next_ops(_reconcile(after))
    assert after.fleet.view is None or after.fleet.goals_run, (
        "a listing stands only while the goals run"
    )
    changed = (Changed(),) if _changed(world, after) else ()
    return after, applied + followed + started + changed


def _apply(world: World, event: Event) -> tuple[World, Effects]:
    match event:
        case Tick(grip=grip, inputs=inputs):
            return _tick(world, grip, inputs)
        case Streamed(grip=grip, burst_running=burst_running):
            return _streamed(world, grip, burst_running)
        case CallDone(robot=robot, call=call, outcome=outcome):
            return _call_done(world, robot, call, outcome)
        case RetryDue(robot=robot, timer=timer):
            return _retry_due(world, robot, timer), ()
        case Listing(view=view):
            return _listing(world, view)
        case LeaseEnded(robot=robot, why=why):
            return _lease_ended(world, robot, why)
        case GoalsStatusChanged(status=status):
            return _goals_status_changed(world, status), ()
        case GoalsEnded(why=why):
            return _goals_ended(world, why)
        case GoalsStarted(generation=generation):
            return _goals_started(world, generation), ()
        case Press(label=label, inputs=inputs):
            return _press(world, label, inputs)
        case PostureResult(robot=robot, start=start, outcome=outcome):
            return _posture_result(world, robot, start, outcome)
        case Shutdown():
            return _shutdown(world)


def _holds(world: World) -> dict[str, Hold]:
    return {robot.name: robot.hold for robot in world.robots if robot.hold is not None}


def _changed(before: World, after: World) -> bool:
    """Whether the step changed a hold or the fleet facts and ends with the
    fleet facts whole: a listing standing, or the goals not running."""
    whole = after.fleet.view is not None or not after.fleet.goals_run
    return whole and (_holds(before) != _holds(after) or before.fleet != after.fleet)


def _new_id(world: World) -> tuple[World, int]:
    """A new request, op, call, timer or posture start id."""
    return replace(world, next_id=world.next_id + 1), world.next_id


def _new_call_and_op(world: World) -> tuple[World, int, int]:
    """A new op's first call id, then its op id."""
    world, call = _new_id(world)
    world, op_id = _new_id(world)
    return world, call, op_id


def _update_robot(world: World, r: str, **changes) -> World:
    return world.with_robot(replace(world.robot(r), **changes))


def _set_phase(world: World, g: GripName, phase: Phase) -> World:
    grip = replace(world.grips.of(g), phase=phase)
    return replace(world, grips=world.grips.with_grip(g, grip))


def _requesting(world: World, requester: Requester, r: str) -> Requesting:
    """The phase of `requester`'s grip, which requests `r` by that request."""
    phase = world.grips.of(requester.grip).phase
    assert (
        isinstance(phase, Requesting)
        and phase.robot == r
        and phase.request == requester.request
    ), "a requester's grip requests its robot"
    return phase


def _squeeze_refused(why: str) -> str:
    """The note of a squeeze that requests nothing, for the cause `why`."""
    return f"squeeze refused: {why}"


def _move_refused(label: str, why: str) -> str:
    """The note of a posture move to `label` that the headset gives up, for
    the cause `why`."""
    return f"move to {label} refused: {why}"


def _move_ignored(label: str, why: str) -> str:
    """The note of a press of `label` that a grip, or the posture move that
    runs, turns away, for the cause `why`."""
    return f"move to {label} ignored: {why}"


def _owe_release(world: World, r: str) -> World:
    """`r` owes a release of a lease the headset may hold, while the goals
    run: the end of the goals ended every lease of their lease goal."""
    if not world.fleet.goals_run:
        return world
    return _update_robot(world, r, release_owed=True)


def _end_posture_move(world: World, start: int) -> World:
    """The posture move of `start` ends."""
    moving = world.posture_moving
    assert moving is not None and moving.start == start, "the move is its start's"
    return replace(world, posture_moving=None)


def _interrupt(world: World, hands: Iterable[Hand]) -> World:
    """Each hand of `hands` whose last fresh reading is a squeeze must be
    released before its squeeze counts again."""
    states = world.hands
    for hand in frozenset(hands):
        state = states.of(hand)
        if state.last_fresh is Reading.SQUEEZE:
            states = states.with_hand(hand, replace(state, must_release=True))
    return replace(world, hands=states)


def _is_held(world: World, g: GripName, inputs: TickInputs) -> bool:
    """Whether a hand of `g` reads a squeeze at this tick with no
    `must_release`."""
    return any(
        inputs.readings[h] is Reading.SQUEEZE and not world.hands.of(h).must_release
        for h in hands_of(g)
    )


def _still_held(world: World, g: GripName) -> bool:
    """Whether a hand of `g` last read a squeeze and has no `must_release`."""
    return any(
        world.hands.of(h).last_fresh is Reading.SQUEEZE
        and not world.hands.of(h).must_release
        for h in hands_of(g)
    )


def _follow_hands(world: World, note: str | None) -> tuple[World, Effects]:
    """Every `Requesting` grip that no hand holds any more turns `Idle` and
    its request is withdrawn; with `note`, each such grip notes it."""
    effects: Effects = ()
    for g, grip in world.grips.items():
        match grip.phase:
            case Requesting(robot=r, request=q) if not _still_held(world, g):
                world = _withdraw(_set_phase(world, g, Idle()), r, q)
                effects += () if note is None else (Note(g, note),)
    return world, effects


def _tick(world: World, g: GripName, inputs: TickInputs) -> tuple[World, Effects]:
    assert set(inputs.readings) == set(hands_of(g)), "a tick reads its grip's hands"
    seen = world.grips.of(g).seen
    previous = {h: seen.of(h) for h in hands_of(g)}
    world, noted = _read_hands(world, g, inputs, previous)
    world = _see_choice(world, inputs.chosen)
    world = _check_pins(world, g, inputs)
    world, effects = _tick_phase(world, g, inputs, previous)
    return world, noted + effects


def _read_hands(
    world: World,
    g: GripName,
    inputs: TickInputs,
    previous: Mapping[Hand, Reading | None],
) -> tuple[World, Effects]:
    """Each fresh reading of the grip's hands; while down, a squeeze read is
    an interruption, noted at the grip's own rising edge."""
    hands, seen = world.hands, world.grips.of(g).seen
    for h in hands_of(g):
        reading = inputs.readings[h]
        if reading is Reading.UNKNOWN:
            continue
        must_release = hands.of(h).must_release and reading is not Reading.RELEASED
        hands = hands.with_hand(h, HandState(reading, must_release))
        seen = seen.with_hand(h, reading)
    grip = replace(world.grips.of(g), seen=seen)
    world = replace(world, hands=hands, grips=world.grips.with_grip(g, grip))
    down = world.fleet.down_cause()
    if down is None:
        return world, ()
    squeezing = tuple(h for h in hands_of(g) if inputs.readings[h] is Reading.SQUEEZE)
    world = _interrupt(world, squeezing)
    if not any(previous[h] is not Reading.SQUEEZE for h in squeezing):
        return world, ()
    return world, (Note(g, _squeeze_refused(down)),)


def _see_choice(world: World, chosen: str | None) -> World:
    """A change of the chosen robot, after one was chosen, interrupts both
    hands."""
    if chosen == world.seen_robot:
        return world
    if world.had_chosen:
        world = _interrupt(world, Hand)
    return replace(
        world,
        seen_robot=chosen,
        had_chosen=world.had_chosen or chosen is not None,
    )


def _check_pins(world: World, g: GripName, inputs: TickInputs) -> World:
    """A pin of the grip's phase that left its slot, or a request whose tick
    reads no pin, interrupts the grip's hands."""
    match world.grips.of(g).phase:
        case Engaged(pins=pins) | Lingering(pins=pins) if not pins <= inputs.peers:
            return _interrupt(world, hands_of(g))
        case Requesting() if not inputs.pins:
            return _interrupt(world, hands_of(g))
    return world


def _tick_phase(
    world: World,
    g: GripName,
    inputs: TickInputs,
    previous: Mapping[Hand, Reading | None],
) -> tuple[World, Effects]:
    """The phase row of the grip's tick, first match."""
    held = _is_held(world, g, inputs)
    chosen = inputs.chosen
    match world.grips.of(g).phase:
        case Engaged(robot=r, pins=pins, epoch=e):
            if held and chosen == r and pins <= inputs.peers:
                return world, ()
            return _set_phase(world, g, Lingering(r, pins, e)), ()
        case Requesting(robot=r, request=q):
            if held and chosen == r and inputs.pins:
                return _set_phase(world, g, Requesting(r, q, inputs.pins)), ()
            return _withdraw(_set_phase(world, g, Idle()), r, q), ()
        case Lingering(robot=r, pins=pins, epoch=e):
            rejoins = (
                held
                and chosen == r
                and pins <= inputs.peers
                and isinstance(inputs.phase_ready, Ready)
                and _allows_rejoin(world, r, g, e)
            )
            if rejoins:
                return _set_phase(world, g, Engaged(r, pins, e)), ()
            return world, ()
        case Idle():
            return _idle_tick(world, g, inputs, held, previous)


def _idle_tick(
    world: World,
    g: GripName,
    inputs: TickInputs,
    held: bool,
    previous: Mapping[Hand, Reading | None],
) -> tuple[World, Effects]:
    chosen = inputs.chosen
    if held and chosen is not None and isinstance(inputs.ready, Ready):
        world, q = _new_id(world)
        world = _set_phase(world, g, Requesting(chosen, q, inputs.pins))
        return _request(world, chosen, g, q)
    rising = any(
        inputs.readings[h] is Reading.SQUEEZE
        and previous[h] is not Reading.SQUEEZE
        and not world.hands.of(h).must_release
        for h in hands_of(g)
    )
    match inputs.ready:
        case NotReady(why=why) if rising:
            return world, (Note(g, _squeeze_refused(why)),)
    return world, ()


def _allows_rejoin(world: World, r: str, g: GripName, e: int) -> bool:
    """Whether a `Lingering` grip may engage again: it would join at once,
    and its reason is in the hold of its epoch."""
    hold = world.robot(r).hold
    return _joins_at_once(world, r, g) and hold.epoch == e and hold.has(g)


def _streamed(
    world: World, g: GripName, burst_running: bool
) -> tuple[World, Effects]:
    """A `Lingering` grip whose burst ended drops its reason."""
    match world.grips.of(g).phase:
        case Lingering(robot=r, epoch=e) if not burst_running:
            return _drop(_set_phase(world, g, Idle()), r, g, e), ()
    return world, ()


def _engage(
    world: World, r: str, requesters: Iterable[Requester], epoch: int
) -> World:
    """Each requester's grip drives its latest pins under the hold of
    `epoch`."""
    for requester in requesters:
        pins = _requesting(world, requester, r).pins
        world = _set_phase(world, requester.grip, Engaged(r, pins, epoch))
    return world


def _refuse(
    world: World, r: str, requesters: tuple[Requester, ...], why: str
) -> tuple[World, Effects]:
    """Each requester turns `Idle` with the note `why`; the refusal
    interrupts their hands, and every other request those hands alone held
    is withdrawn with the same note."""
    note = _squeeze_refused(why)
    effects: Effects = ()
    for requester in requesters:
        _requesting(world, requester, r)
        world = _set_phase(world, requester.grip, Idle())
        effects += (Note(requester.grip, note),)
        world = _withdraw(world, r, requester.request)
    world = _interrupt(
        world, (h for requester in requesters for h in hands_of(requester.grip))
    )
    world, followed = _follow_hands(world, note=note)
    return world, effects + followed


def _let_go_grips(world: World, r: str, why: str) -> tuple[World, Effects]:
    """Every grip on `r`: a request turns `Idle` with the note `why`, an
    engaged grip lingers. A let-go of the chosen robot interrupts both
    hands."""
    effects: Effects = ()
    for g, grip in world.grips.items():
        if robot_of_phase(grip.phase) != r:
            continue
        match grip.phase:
            case Requesting(request=q):
                world = _set_phase(world, g, Idle())
                effects += (Note(g, _squeeze_refused(why)),)
                world = _withdraw(world, r, q)
            case Engaged(pins=pins, epoch=e):
                world = _set_phase(world, g, Lingering(r, pins, e))
    if r == world.seen_robot:
        world = _interrupt(world, Hand)
    return world, effects


def _request_of(op: Op) -> CallRequest:
    """The request of the call `op` waits on."""
    match op:
        case (
            Take(stage=TakeStage.ADMIT, mode=mode)
            | PostureStart(stage=PostureStage.ADMIT, mode=mode)
        ):
            return CheckMove() if isinstance(mode, UnderOpen) else TakeLease()
        case (
            Take(stage=TakeStage.STOP)
            | Join(stage=JoinStage.STOP)
            | PostureStart(stage=PostureStage.STOP)
        ):
            return StopMoves()
        case Join(stage=JoinStage.CHECK) | Confirm():
            return CheckMove()
        case PostureStart(stage=PostureStage.CHECK_INSTANCE, target=target):
            return CheckMoveInstance(target.instance)
        case PostureStart(stage=PostureStage.FIRE, label=label, target=target):
            return FireGoal(label, target)
        case Release():
            return ReleaseLease()


def _why(request: CallRequest, outcome: CallOutcome) -> str:
    """The cause of a call that was not answered."""
    match outcome:
        case Failed(cause=cause) | Raised(cause=cause):
            return f"stop failed: {cause}" if isinstance(request, StopMoves) else cause
        case Unanswered():
            assert not isinstance(request, StopMoves), (
                "a stop names its own cause when it gets no answer"
            )
            return DAEMON_SILENT
        case Answered():
            raise AssertionError("an answered call has no cause")


def _op_mode(op: Take | Join | PostureStart | Confirm) -> Mode:
    """The mode an op runs under: a `Confirm` runs under `Taken` only."""
    return Taken() if isinstance(op, Confirm) else op.mode


def _lost(world: World, r: str, op: Take | Join | PostureStart | Confirm) -> str | None:
    """Why `op` may not go on: the goals it started under ended, the stack is
    halted, or the open lease it drives under closed; None while it may."""
    fleet = world.fleet
    if op.generation != fleet.generation or not fleet.goals_run:
        return GOALS_ENDED
    view = fleet.listed()
    if view.halted_by is not None:
        return halted_text(view.halted_by)
    match _op_mode(op):
        case UnderOpen(holder=holder) if not _open_by(view, r, holder):
            return _open_lease_closed(holder)
    return None


def _call_done(
    world: World, r: str, call: int, outcome: CallOutcome
) -> tuple[World, Effects]:
    robot = world.robot(r)
    if robot.lost_stop == call:
        return _lost_stop_done(world, r, outcome)
    assert robot.op is not None and robot.op.call == call, "a call ends its own op"
    world = _take_facts(world, r, robot.op, outcome)
    op = world.robot(r).op
    if isinstance(op, Take | Join) and not op.requesters:
        return _end_requesterless(world, r, op), ()
    if isinstance(op, Release):
        return _release_done(world, r, outcome)
    lost = _lost(world, r, op)
    moved = op.epoch != world.robot(r).epoch
    match op:
        case Take():
            return _take_done(world, r, op, outcome, lost, moved)
        case Join():
            return _join_done(world, r, op, outcome, lost, moved)
        case PostureStart(stage=PostureStage.ADMIT | PostureStage.STOP):
            return _posture_start_done(world, r, op, outcome, lost, moved)
        case PostureStart():
            return _posture_goal_done(world, r, op, outcome, lost, moved)
        case Confirm():
            return _confirm_done(world, r, op, outcome, lost, moved)


def _take_facts(world: World, r: str, op: Op, outcome: CallOutcome) -> World:
    """The facts of a `take(r)` call, whatever its op's state: an answer
    took the lease and owes no release, and a take that may have reached
    the daemon owes one."""
    if not isinstance(_request_of(op), TakeLease):
        return world
    match outcome:
        case Answered():
            world = _update_robot(world, r, op=replace(op, took=True))
            return _update_robot(world, r, release_owed=False, retry=None)
        case Unanswered() | Raised():
            return _owe_release(world, r)
        case Failed():
            return world


def _end_requesterless(world: World, r: str, op: Take | Join) -> World:
    """An acquisition with no requester left ends at its call's outcome; a
    take that took the lease owes its release."""
    world = _update_robot(world, r, op=None)
    if isinstance(op, Take) and op.took:
        return _owe_release(world, r)
    return world


def _call_next(world: World, r: str, op: Op) -> tuple[World, Effects]:
    """`op` at its next stage, with a new call of that stage's request."""
    world, call = _new_id(world)
    op = replace(op, call=call)
    return _update_robot(world, r, op=op), (Call(r, call, _request_of(op)),)


def _stop_after_admit(world: World, r: str) -> tuple[World, Effects]:
    """A `Take` or `PostureStart` whose admit was answered stops the
    robot."""
    op = world.robot(r).op
    stage = TakeStage.STOP if isinstance(op, Take) else PostureStage.STOP
    return _call_next(world, r, replace(op, stage=stage))


def _take_done(
    world: World,
    r: str,
    op: Take,
    outcome: CallOutcome,
    lost: str | None,
    moved: bool,
) -> tuple[World, Effects]:
    assert not moved, "a take runs with no hold, and a let-go withdraws its requesters"
    if lost is not None:
        return _refuse_take(world, r, op, lost)
    if not isinstance(outcome, Answered):
        return _refuse_take(world, r, op, _why(_request_of(op), outcome))
    if op.stage is TakeStage.ADMIT:
        return _stop_after_admit(world, r)
    epoch = world.robot(r).epoch + 1
    reasons = frozenset(Brought(q.grip, q.request) for q in op.requesters)
    hold = Hold(op.mode, reasons, epoch)
    world = _update_robot(world, r, hold=hold, epoch=epoch, op=None)
    return _engage(world, r, op.requesters, epoch), ()


def _refuse_take(world: World, r: str, op: Take, why: str) -> tuple[World, Effects]:
    world, effects = _refuse(world, r, op.requesters, why)
    world = _update_robot(world, r, op=None)
    return (_owe_release(world, r) if op.took else world), effects


def _join_done(
    world: World,
    r: str,
    op: Join,
    outcome: CallOutcome,
    lost: str | None,
    moved: bool,
) -> tuple[World, Effects]:
    assert lost is None and not moved, (
        "a let-go withdraws the requesters of a join, and every loss lets go"
    )
    if not isinstance(outcome, Answered):
        world, effects = _refuse(
            world, r, op.requesters, _why(_request_of(op), outcome)
        )
        return _update_robot(world, r, op=None), effects
    if op.stage is JoinStage.CHECK:
        return _call_next(world, r, replace(op, stage=JoinStage.STOP))
    epoch = world.robot(r).hold.epoch
    world = _update_robot(world, r, op=None)
    return _engage(world, r, op.requesters, epoch), ()


def _end_posture_start(
    world: World, r: str, op: PostureStart, cause: str
) -> tuple[World, Effects]:
    """A posture start that ends before its hold: refused with `cause`; a
    take that took the lease owes its release."""
    world = _update_robot(world, r, op=None)
    if op.took:
        world = _owe_release(world, r)
    world = _end_posture_move(world, op.id)
    return world, (Note(POSTURE_NOTE, _move_refused(op.label, cause)),)


def _posture_start_done(
    world: World,
    r: str,
    op: PostureStart,
    outcome: CallOutcome,
    lost: str | None,
    moved: bool,
) -> tuple[World, Effects]:
    """A `PostureStart`'s admit or stop ended."""
    if lost is not None:
        return _end_posture_start(world, r, op, lost)
    if moved:
        assert op.let_go is not None, "a let-go moved the posture start"
        return _end_posture_start(world, r, op, op.let_go)
    if not isinstance(outcome, Answered):
        return _end_posture_start(world, r, op, _why(_request_of(op), outcome))
    if op.stage is PostureStage.ADMIT:
        return _stop_after_admit(world, r)
    epoch = world.robot(r).epoch + 1
    hold = Hold(op.mode, frozenset({Brought(POSTURE, op.id)}), epoch)
    moving = world.posture_moving
    assert moving is not None and moving.start == op.id, "the move is its start's"
    world = replace(world, posture_moving=replace(moving, epoch=epoch))
    world = _update_robot(world, r, hold=hold, epoch=epoch)
    return _call_next(
        world, r, replace(op, epoch=epoch, stage=PostureStage.CHECK_INSTANCE)
    )


def _posture_goal_done(
    world: World,
    r: str,
    op: PostureStart,
    outcome: CallOutcome,
    lost: str | None,
    moved: bool,
) -> tuple[World, Effects]:
    """A `PostureStart`'s instance check or goal ended, after its hold."""
    match op.stage, outcome:
        case PostureStage.FIRE, Unanswered():
            return _goal_may_run(world, r, _move_refused(op.label, GOAL_UNSENT))
        case PostureStage.FIRE, Raised(cause=cause):
            return _goal_may_run(world, r, _move_refused(op.label, cause))
    if lost is not None or moved:
        cause = lost if lost is not None else f"the headset let go of {r}"
        cancelled: Effects = ()
        if op.stage is PostureStage.FIRE and isinstance(outcome, Answered):
            cancelled = (CancelGoal(r, op.label, outcome.response),)
        world = _end_posture_move(_update_robot(world, r, op=None), op.id)
        return world, cancelled + (Note(POSTURE_NOTE, _move_refused(op.label, cause)),)
    if not isinstance(outcome, Answered):
        why = _why(_request_of(op), outcome)
        world = _drop(world, r, POSTURE, op.epoch, by=op.id)
        world = _end_posture_move(_update_robot(world, r, op=None), op.id)
        return world, (Note(POSTURE_NOTE, _move_refused(op.label, why)),)
    if op.stage is PostureStage.CHECK_INSTANCE:
        return _call_next(world, r, replace(op, stage=PostureStage.FIRE))
    goal = AcceptedGoal(op.id, op.label, outcome.response)
    hold = replace(world.robot(r).hold, goal=goal)
    world = _update_robot(world, r, hold=hold, op=None)
    return world, (Watch(r, op.id, op.label, outcome.response),)


def _goal_may_run(world: World, r: str, note: str) -> tuple[World, Effects]:
    """A goal send that may have reached the backbone: the start ends and
    the move is stopped."""
    world, stopped = _lost_stop(_update_robot(world, r, op=None), r)
    return world, (Note(POSTURE_NOTE, note), *stopped)


def _release_done(
    world: World, r: str, outcome: CallOutcome
) -> tuple[World, Effects]:
    world = _update_robot(world, r, op=None)
    match outcome:
        case Answered() | Failed():
            return _update_robot(world, r, release_owed=False), ()
        case Unanswered() | Raised():
            if not world.robot(r).release_owed:
                return world, ()
            world, timer = _new_id(world)
            return _update_robot(world, r, retry=timer), (StartRetry(r, timer),)


def _confirm_done(
    world: World,
    r: str,
    op: Confirm,
    outcome: CallOutcome,
    lost: str | None,
    moved: bool,
) -> tuple[World, Effects]:
    world = _update_robot(world, r, op=None)
    if lost is not None or moved:
        return world, ()
    match outcome:
        case Answered(response=revision):
            hold = world.robot(r).hold
            pending = hold.pending_end
            if pending is None or revision < pending.revision:
                hold = replace(hold, pending_end=PendingEnd(revision, op.why))
                world = _update_robot(world, r, hold=hold)
            return _settle(world, r, world.fleet.listed())
        case Failed():
            return _let_go(world, r, op.why, release=False, cancel_goal=True)
        case Unanswered():
            return _let_go(world, r, DAEMON_SILENT, release=True, cancel_goal=True)
        case Raised(cause=cause):
            return _let_go(world, r, cause, release=True, cancel_goal=True)


def _running_acquisition(world: World, r: str) -> bool:
    """Whether a `Take` or `Join` of `r` runs that has neither moved nor is
    lost."""
    robot = world.robot(r)
    op = robot.op
    return (
        isinstance(op, Take | Join)
        and op.epoch == robot.epoch
        and _lost(world, r, op) is None
    )


def _joins_at_once(world: World, r: str, g: GripName) -> bool:
    """Whether `g` joins the hold of `r` with no call: no `Confirm` runs, and
    another grip is engaged on `r`, or the hold is taken with no posture
    move."""
    robot = world.robot(r)
    hold = robot.hold
    if hold is None or isinstance(robot.op, Confirm):
        return False
    beside = any(
        other != g and isinstance(grip.phase, Engaged) and grip.phase.robot == r
        for other, grip in world.grips.items()
    )
    return beside or (isinstance(hold.mode, Taken) and not hold.has(POSTURE))


def _request(world: World, r: str, g: GripName, q: int) -> tuple[World, Effects]:
    """Grip `g`'s request `q` for a hold of `r`, first matching rule."""
    assert world.fleet.down_cause() is None, "no grip is held while down"
    robot = world.robot(r)
    requester = Requester(g, q)
    if _running_acquisition(world, r):
        op = robot.op
        joined = replace(op, requesters=(*op.requesters, requester))
        world = _update_robot(world, r, op=joined)
        if isinstance(op, Join):
            world = _update_robot(world, r, hold=robot.hold.with_reason(g, q))
        return world, ()
    if _joins_at_once(world, r, g):
        world = _update_robot(world, r, hold=robot.hold.with_reason(g, q))
        return _engage(world, r, (requester,), robot.hold.epoch), ()
    if robot.op is None:
        return _start_acquisition(world, r, (requester,))
    return _update_robot(world, r, queue=(*robot.queue, requester)), ()


def _start_acquisition(
    world: World, r: str, requesters: tuple[Requester, ...]
) -> tuple[World, Effects]:
    if world.robot(r).hold is None:
        return _start_take(world, r, requesters)
    return _start_join(world, r, requesters)


def _not_listed(r: str) -> str:
    return (
        f"{r} is not in the fleet listing, which holds the copies of"
        f" the stack: join a robot as a copy with {JOIN_COMMAND}"
    )


def _mode_of(world: World, r: str) -> Mode:
    """How the headset would drive `r`: under the open lease of another
    holder, else under its own lease."""
    lease = world.fleet.listed().leases.get(r)
    if lease is not None and lease.open and lease.holder != world.fleet.me:
        return UnderOpen(lease.holder.instance)
    return Taken()


def _start_take(
    world: World, r: str, requesters: tuple[Requester, ...]
) -> tuple[World, Effects]:
    assert world.fleet.down_cause() is None, "no take starts while down"
    if r not in world.fleet.listed().robots:
        return _refuse(world, r, requesters, _not_listed(r))
    mode = _mode_of(world, r)
    world, call, take_id = _new_call_and_op(world)
    op = Take(
        take_id,
        world.robot(r).epoch,
        world.fleet.generation,
        mode,
        requesters,
        TakeStage.ADMIT,
        call,
    )
    return _update_robot(world, r, op=op), (Call(r, call, _request_of(op)),)


def _start_join(
    world: World, r: str, requesters: tuple[Requester, ...]
) -> tuple[World, Effects]:
    robot = world.robot(r)
    hold = robot.hold
    if hold.has(POSTURE):
        stage = JoinStage.STOP
    else:
        assert isinstance(hold.mode, UnderOpen), (
            "a taken hold with no posture move joins at once"
        )
        stage = JoinStage.CHECK
    for requester in requesters:
        hold = hold.with_reason(requester.grip, requester.request)
    world, call, join_id = _new_call_and_op(world)
    op = Join(
        join_id,
        robot.epoch,
        world.fleet.generation,
        hold.mode,
        requesters,
        stage,
        call,
    )
    world = _update_robot(world, r, hold=hold, op=op)
    return world, (Call(r, call, _request_of(op)),)


def _withdraw(world: World, r: str, q: int) -> World:
    """Request `q` leaves `r`'s acquisition, its reason with it for a
    `Join`, or the queue. The call in flight runs to its outcome."""
    op = world.robot(r).op
    match op:
        case Take(requesters=requesters) | Join(requesters=requesters):
            leaving = next((x for x in requesters if x.request == q), None)
            if leaving is not None:
                kept = tuple(x for x in requesters if x.request != q)
                world = _update_robot(world, r, op=replace(op, requesters=kept))
                if isinstance(op, Join):
                    world = _drop(world, r, leaving.grip, op.epoch, by=q)
    queue = world.robot(r).queue
    kept_queue = tuple(
        entry
        for entry in queue
        if not (isinstance(entry, Requester) and entry.request == q)
    )
    if kept_queue == queue:
        return world
    return _update_robot(world, r, queue=kept_queue)


def _grip_refusal(world: World, r: str | None) -> str | None:
    """Why a posture move of `r` does nothing for a grip: a grip requests or
    drives `r`, or its burst holds it; None while no grip is on `r`."""
    if r is None:
        return None
    phases = tuple(
        grip.phase for _, grip in world.grips.items() if robot_of_phase(grip.phase) == r
    )
    if any(isinstance(phase, Requesting | Engaged) for phase in phases):
        return GRIP_HELD
    if any(isinstance(phase, Lingering) for phase in phases):
        return GRIP_JUST_RELEASED
    return None


def _start_posture(
    world: World, r: str, start: int, label: str, target: PostureTarget
) -> tuple[World, Effects]:
    refusal = _grip_refusal(world, r)
    if refusal is not None:
        world = _end_posture_move(world, start)
        return world, (Note(POSTURE_NOTE, _move_ignored(label, refusal)),)
    assert world.robot(r).hold is None, "a posture start finds no hold"
    assert world.fleet.down_cause() is None, "no posture start runs while down"
    if r not in world.fleet.listed().robots:
        world = _end_posture_move(world, start)
        note = _move_refused(label, _not_listed(r))
        return world, (Note(POSTURE_NOTE, note),)
    mode = _mode_of(world, r)
    world, call = _new_id(world)
    op = PostureStart(
        start,
        world.robot(r).epoch,
        world.fleet.generation,
        label,
        target,
        mode,
        PostureStage.ADMIT,
        call,
    )
    return _update_robot(world, r, op=op), (Call(r, call, _request_of(op)),)


def _press_refusal(world: World, label: str, inputs: PressInputs) -> str | None:
    """The note of a press that starts no move, first match; None for a
    press that starts one."""
    r = inputs.chosen
    if inputs.squeezing:
        return _move_ignored(label, GRIP_HELD)
    refusal = _grip_refusal(world, r)
    if refusal is not None:
        return _move_ignored(label, refusal)
    if world.posture_moving is not None:
        return _move_ignored(label, f"still moving to {world.posture_moving.label}")
    if inputs.target is None:
        if r is None:
            return f"move to {label}: no robot paired"
        return f"move to {label}: {r} has no postures"
    assert r is not None and world.robot(r).hold is None, (
        "a hold of the chosen robot has a grip on it or a posture move"
    )
    down = world.fleet.down_cause()
    if down is not None:
        return _move_refused(label, down)
    return None


def _press(world: World, label: str, inputs: PressInputs) -> tuple[World, Effects]:
    """A posture press: refused with its note, else the move starts or
    queues."""
    refusal = _press_refusal(world, label, inputs)
    if refusal is not None:
        return world, (Note(POSTURE_NOTE, refusal),)
    r = inputs.chosen
    world, start = _new_id(world)
    world = replace(world, posture_moving=PostureMoving(label, start, r))
    robot = world.robot(r)
    if robot.op is None:
        return _start_posture(world, r, start, label, inputs.target)
    queued = QueuedPosture(start, label, inputs.target)
    return _update_robot(world, r, queue=(*robot.queue, queued)), ()


def _may_stop_lost_move(world: World, r: str) -> bool:
    """Whether the headset stops a posture move that may run on `r`: while
    its hold has the move, or while no other client holds `r`'s lease."""
    hold = world.robot(r).hold
    if hold is not None and hold.has(POSTURE):
        return True
    view = world.fleet.view
    lease = view.leases.get(r) if view is not None else None
    return lease is None or lease.holder == world.fleet.me


def _lost_stop(world: World, r: str) -> tuple[World, Effects]:
    """A posture move that may run on `r` is stopped outside the op slot, or
    left to the client that holds the lease."""
    moving = world.posture_moving
    assert (
        world.robot(r).lost_stop is None
        and moving is not None
        and moving.robot == r
        and moving.epoch is not None
    ), "a lost stop stops the move of a hold"
    if _may_stop_lost_move(world, r):
        world, call = _new_id(world)
        world = _update_robot(world, r, lost_stop=call)
        return world, (Call(r, call, StopMoves()),)
    world = _drop(world, r, POSTURE, moving.epoch, by=moving.start)
    return replace(world, posture_moving=None), ()


def _lost_stop_done(
    world: World, r: str, outcome: CallOutcome
) -> tuple[World, Effects]:
    """The stop of a posture move that may run ended: the move ends with it,
    and a failed stop is noted."""
    moving = world.posture_moving
    assert moving is not None and moving.robot == r and moving.epoch is not None, (
        "a lost stop ends the move of a hold"
    )
    world = _update_robot(world, r, lost_stop=None)
    world = _drop(world, r, POSTURE, moving.epoch, by=moving.start)
    world = replace(world, posture_moving=None)
    match outcome:
        case Answered():
            return world, ()
        case Failed(cause=cause) | Raised(cause=cause):
            note = f"move to {moving.label}: its stop failed: {cause}"
            return world, (Note(POSTURE_NOTE, note),)
        case Unanswered():
            raise AssertionError("a stop names its own cause when it gets no answer")


def _posture_result(
    world: World, r: str, start: int, outcome: PostureOutcome
) -> tuple[World, Effects]:
    moving = world.posture_moving
    if moving is None or moving.start != start:
        return world, ()
    assert moving.robot == r and moving.epoch is not None, (
        "a watched goal is the move's, under its hold"
    )
    match outcome:
        case Finished(note=note):
            world = _drop(world, r, POSTURE, moving.epoch, by=start)
            return replace(world, posture_moving=None), (Note(POSTURE_NOTE, note),)
        case ResultLost():
            unknown = Note(POSTURE_NOTE, f"move to {moving.label}: result unknown")
            world, stopped = _lost_stop(world, r)
            return world, (unknown, *stopped)


def _next_ops(world: World) -> tuple[World, Effects]:
    """Each robot with no op starts its next one, in name order."""
    effects: Effects = ()
    for name in tuple(robot.name for robot in world.robots):
        world, started = _next_op(world, name)
        effects += started
    return world, effects


def _release_due(world: World, robot: Robot) -> bool:
    return (
        robot.release_owed
        and robot.hold is None
        and robot.retry is None
        and world.fleet.goals_run
    )


def _next_op(world: World, r: str) -> tuple[World, Effects]:
    """`r`'s queue's head starts while no op runs; with an empty queue, an
    owed release."""
    robot = world.robot(r)
    if robot.op is not None:
        return world, ()
    if robot.queue:
        world = _update_robot(world, r, queue=robot.queue[1:])
        world, started = _start_queued(world, r, robot.queue[0])
        world, after = _next_op(world, r)
        return world, started + after
    if not _release_due(world, robot):
        return world, ()
    world, call, release_id = _new_call_and_op(world)
    release = Release(release_id, call)
    world = _update_robot(world, r, op=release, release_seq=world.listing_seq)
    return world, (Call(r, call, ReleaseLease()),)


def _reconcile(world: World) -> World:
    """Each robot with no hold, op, queue or retry owes a release while the
    newest listing, newer than the robot's last `Release` and not halted,
    names the headset as its lease holder: the daemon's book holds that
    lease for the headset. A listing stands only while the goals run."""
    fleet = world.fleet
    view = fleet.view
    if view is None or view.halted_by is not None:
        return world
    for r in sorted(view.leases):
        robot = world.robot(r)
        idle = (
            robot.hold is None
            and robot.op is None
            and not robot.queue
            and robot.retry is None
        )
        listed_since = world.listing_seq > robot.release_seq
        if idle and listed_since and _leased_by(view, r, fleet.me):
            world = _owe_release(world, r)
    return world


def _start_queued(world: World, r: str, head: Queued) -> tuple[World, Effects]:
    match head:
        case Requester(grip=g, request=q):
            _requesting(world, head, r)
            if _joins_at_once(world, r, g):
                hold = world.robot(r).hold
                world = _update_robot(world, r, hold=hold.with_reason(g, q))
                return _engage(world, r, (head,), hold.epoch), ()
            queue = world.robot(r).queue
            others = tuple(entry for entry in queue if isinstance(entry, Requester))
            rest = tuple(entry for entry in queue if not isinstance(entry, Requester))
            world = _update_robot(world, r, queue=rest)
            return _start_acquisition(world, r, (head, *others))
        case QueuedPosture(start=start, label=label, target=target):
            return _start_posture(world, r, start, label, target)
        case QueuedConfirm(why=why):
            return _start_confirm(world, r, why)


def _start_confirm(world: World, r: str, why: str) -> tuple[World, Effects]:
    """A lease end of a taken hold asks the daemon whether the headset may
    still move `r`; with no taken hold it ends at once."""
    robot = world.robot(r)
    if robot.hold is None or not isinstance(robot.hold.mode, Taken):
        return world, ()
    world, call, confirm_id = _new_call_and_op(world)
    op = Confirm(confirm_id, robot.epoch, world.fleet.generation, why, call)
    return _update_robot(world, r, op=op), (Call(r, call, CheckMove()),)


def _retry_due(world: World, r: str, timer: int) -> World:
    if world.robot(r).retry != timer:
        return world
    return _update_robot(world, r, retry=None)


def _let_go(
    world: World, r: str, why: str, *, release: bool, cancel_goal: bool
) -> tuple[World, Effects]:
    """Let go of `r` at once, as at a grip release: the hold ends, its goal
    is cancelled when `cancel_goal`, its lease released when `release`, the
    epoch moves, the grips on `r` let go, and the queue empties."""
    effects: Effects = ()
    hold = world.robot(r).hold
    if hold is not None:
        world = _update_robot(world, r, hold=None)
        if hold.goal is not None and cancel_goal:
            effects += (CancelGoal(r, hold.goal.label, hold.goal.goal),)
        effects += (Note(ROBOT_NOTE, f"let go of {r}: {why}"),)
        if release and isinstance(hold.mode, Taken):
            world = _owe_release(world, r)
    robot = world.robot(r)
    op = robot.op
    if isinstance(op, PostureStart) and op.let_go is None:
        op = replace(op, let_go=why)
    world = _update_robot(world, r, epoch=robot.epoch + 1, op=op)
    world, released = _let_go_grips(world, r, why)
    effects += released
    queue = world.robot(r).queue
    assert all(isinstance(entry, QueuedPosture | QueuedConfirm) for entry in queue), (
        "a let-go withdraws every queued request"
    )
    for entry in queue:
        match entry:
            case QueuedPosture(start=start, label=label):
                effects += (Note(POSTURE_NOTE, _move_refused(label, why)),)
                world = _end_posture_move(world, start)
    return _update_robot(world, r, queue=()), effects


def _drop(
    world: World, r: str, reason: Reason, epoch: int, by: int | None = None
) -> World:
    """`reason` leaves the hold of `epoch`, brought by `by` when given; the
    last reason ends the hold, and a taken lease owes its release. Any other
    epoch changes nothing."""
    robot = world.robot(r)
    hold = robot.hold
    if hold is None or hold.epoch != epoch or not hold.has(reason):
        return world
    if by is not None and hold.brought_by(reason) != by:
        return world
    rest = hold.without(reason)
    if rest is not None:
        return _update_robot(world, r, hold=rest)
    world = _update_robot(world, r, hold=None, epoch=robot.epoch + 1)
    return _owe_release(world, r) if isinstance(hold.mode, Taken) else world


def _settle(world: World, r: str, view: FleetView) -> tuple[World, Effects]:
    """Once `view` is as recent as the yes to the pending lease end, let go
    of `r` unless `view` names the headset as its holder."""
    hold = world.robot(r).hold
    end = hold.pending_end if hold is not None else None
    if end is None or view.revision < end.revision:
        return world, ()
    world = _update_robot(world, r, hold=replace(hold, pending_end=None))
    if _leased_by(view, r, world.fleet.me):
        return world, ()
    return _let_go(world, r, end.why, release=False, cancel_goal=True)


def _every_robot(world: World) -> tuple[str, ...]:
    """Each robot the world holds state for, each robot the listing names,
    and the chosen robot, in name order."""
    names = {robot.name for robot in world.robots}
    if world.fleet.view is not None:
        names |= world.fleet.view.robots
    if world.seen_robot is not None:
        names.add(world.seen_robot)
    return tuple(sorted(names))


def _let_go_of_every_robot(
    world: World, why: str, *, cancel_goal: bool
) -> tuple[World, Effects]:
    effects: Effects = ()
    for r in _every_robot(world):
        world, let_go = _let_go(world, r, why, release=False, cancel_goal=cancel_goal)
        effects += let_go
    return world, effects


def _clear_owed(world: World) -> World:
    """No robot owes a release or waits on a retry: the end of the lease goal
    ended every lease the headset held under it."""
    robots = tuple(
        replace(robot, release_owed=False, retry=None) for robot in world.robots
    )
    return replace(world, robots=robots)


def _listing(world: World, view: FleetView) -> tuple[World, Effects]:
    """A listing: a halt's start lets go of every robot; otherwise each hold
    under an open lease that the listing does not show open by its holder
    lets go, and each other hold settles its pending lease end."""
    fleet = world.fleet
    assert fleet.goals_run, "a listing comes after the goals' status says they run"
    starts = view.halted_by is not None and not fleet.halted
    world = replace(
        world, fleet=replace(fleet, view=view), listing_seq=world.listing_seq + 1
    )
    if starts:
        world, effects = _let_go_of_every_robot(
            world, halted_text(view.halted_by), cancel_goal=True
        )
        return _interrupt(world, Hand), effects
    effects: Effects = ()
    for r in tuple(robot.name for robot in world.robots if robot.hold is not None):
        match world.robot(r).hold.mode:
            case UnderOpen(holder=holder) if not _open_by(view, r, holder):
                world, let_go = _let_go(
                    world,
                    r,
                    _open_lease_closed(holder),
                    release=False,
                    cancel_goal=True,
                )
            case _:
                world, let_go = _settle(world, r, view)
        effects += let_go
    return world, effects


def _goals_status_changed(world: World, status: GoalsStatus) -> World:
    fleet = world.fleet
    assert status != fleet.status, "the shell steps a change of the status only"
    assert isinstance(status, Following) or fleet.view is None, (
        "no listing stands while the goals do not run"
    )
    return replace(world, fleet=replace(fleet, status=status))


def _goals_started(world: World, generation: int) -> World:
    fleet = world.fleet
    assert generation == fleet.generation + 1, "the goals start once at a time"
    return replace(world, fleet=replace(fleet, generation=generation))


def _goals_ended(world: World, why: str) -> tuple[World, Effects]:
    """A goal of the fleet ended: let go of every robot, owe no release, and
    forget the listing."""
    world, effects = _let_go_of_every_robot(world, why, cancel_goal=True)
    world = _clear_owed(world)
    world = replace(world, fleet=replace(world.fleet, view=None))
    return _interrupt(world, Hand), effects


def _lease_ended(world: World, r: str, why: str) -> tuple[World, Effects]:
    why = why or LEASE_ENDED
    robot = world.robot(r)
    if robot.op is None:
        return _start_confirm(world, r, why)
    return _update_robot(world, r, queue=(*robot.queue, QueuedConfirm(why))), ()


def _shutdown(world: World) -> tuple[World, Effects]:
    """The node stops, and its goals with it: let go of every robot with no
    cancel and no release."""
    return _let_go_of_every_robot(world, NODE_STOPS, cancel_goal=False)


def _leased_by(view: FleetView, robot: str, holder: Holder) -> bool:
    """Whether `holder` holds the lease of `robot` in `view`."""
    lease = view.leases.get(robot)
    return lease is not None and lease.holder == holder


def _open_by(view: FleetView, robot: str, holder: str) -> bool:
    """Whether `holder`'s lease of `robot` is open in `view`."""
    lease = view.leases.get(robot)
    return lease is not None and lease.open and lease.holder.instance == holder


def _open_lease_closed(holder: str) -> str:
    return f"the open lease of {holder} closed"
