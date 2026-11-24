"""The shell of the grip world (`grip_world`): the driver holds the world,
steps it at each event, and runs each step's effects before any await. A
call runs as a task whose outcome is the next event, a retry waits on the
clock, a note flashes on its row for a few seconds, and a change of a hold
or of the fleet refreshes the page."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal, Protocol

from xr_commander import grip_world
from xr_commander.bus import (
    REAL_CLOCK,
    Answered,
    Clock,
    Failed,
    FlashNote,
    Latch,
    Outcome,
    Unanswered,
    cause,
    log,
    outcome_of,
    refusal_of,
    spawn,
)
from xr_commander.devices import Hand
from xr_commander.fleet import (
    LEASE_NAME,
    DrivenHere,
    FleetView,
    Free,
    GoalsStatus,
    Holder,
    Timing,
    halted_text,
    robot_state,
    state_text,
    status_text,
)
from xr_commander.grip_world import (
    NOTE_TARGETS,
    POSTURE_NOTE,
    ROBOT_NOTE,
    Call,
    CallDone,
    CallOutcome,
    CallRequest,
    CancelGoal,
    Changed,
    CheckMove,
    CheckMoveInstance,
    Effect,
    Event,
    Finished,
    FireGoal,
    GoalsEnded,
    GoalsStarted,
    GoalsStatusChanged,
    GripName,
    LeaseEnded,
    Listing,
    Note,
    NoteTarget,
    Phase,
    PostureResult,
    PostureTarget,
    Raised,
    ReleaseLease,
    ResultLost,
    RetryDue,
    Shutdown,
    StartRetry,
    StopMoves,
    TakeLease,
    UnderOpen,
    Watch,
    World,
    hands_of,
)
from xr_commander.robots import PairedRobots

Tone = Literal["driving", "waiting", "resting"]


class LeaseCalls(Protocol):
    """The calls of a lease holder to its daemon (`fleet.FleetCalls`)."""

    async def take(self, robot: str) -> Outcome: ...
    async def release(self, robot: str) -> Outcome: ...
    async def check_move(
        self, *, robot: str | None = None, instance: str | None = None
    ) -> Outcome: ...


class Stops(Protocol):
    """Stops the planned moves in flight on a robot: None once stopped, else
    the cause."""

    async def stop(self, robot: str) -> str | None: ...


class RobotStops:
    """`stop` of robot_stop on the robot's own backbone, with the reason
    that names the headset."""

    def __init__(
        self,
        node_runner,
        robots: PairedRobots,
        stop_module: ModuleType,
        timing: Timing,
    ) -> None:
        self._node_runner = node_runner
        self._robots = robots
        self._stop = stop_module
        self._timeout_s = timing.call_timeout_s

    async def stop(self, robot: str) -> str | None:
        producer = self._robots.producer_of(self._stop, robot)
        if producer is None:
            return f"{robot} has no robot_stop: the launcher links none to the headset"
        request = self._stop.Request(reason=LEASE_NAME)
        call = self._stop.poll(self._node_runner, producer, request, self._timeout_s)
        stopped = await outcome_of(call, self._stop.Failure, f"stop of {robot}")
        return refusal_of(stopped, f"{robot} did not answer")


@dataclass(frozen=True)
class RobotLine:
    """The chosen robot as the status panel's robot row gives it."""

    text: str
    tone: Tone


class Goals(Protocol):
    """The posture goals of the headset (`postures.PostureGoals`)."""

    async def fire(self, label: str, target: PostureTarget) -> object | None:
        """The goal handle, accepted or refused, None when the send failed."""

    async def result(self, label: str, goal: object) -> str:
        """The posture row's note of how the move ended; raises when the
        result does not come."""

    def cancel(self, label: str, goal: object) -> None:
        """Cancel the goal without waiting on it."""


@dataclass(frozen=True)
class GripStatus:
    """A grip as the panel reads it: its phase, the hands of the grip whose
    squeeze must be released first, and its note now ("" with none)."""

    phase: Phase
    must_release: frozenset[Hand]
    note: str


class Driver:
    """The headset's grip world and the shell that runs it. Every method
    runs on the node's loop; one step runs to its end, its effects included,
    before the next."""

    def __init__(
        self,
        me: Holder,
        calls: LeaseCalls,
        stops: Stops,
        goals: Goals,
        *,
        timing: Timing,
        on_change: Callable[[], None],
        clock: Clock = REAL_CLOCK,
    ) -> None:
        self._world = World.start(me)
        self._calls = calls
        self._stops = stops
        self._goals = goals
        self._timing = timing
        self._on_change = on_change
        self._clock = clock
        self._notes = {target: FlashNote(clock) for target in NOTE_TARGETS}
        self._stepping = False
        # `Shutdown` was stepped: no event steps after it.
        self._shut = False
        self._refreshing = Latch()

    @property
    def world(self) -> World:
        """The world after the last step."""
        return self._world

    def step(self, event: Event) -> World:
        """Step the world at `event` and run the step's effects in order,
        with no await: the world after the step. After `Shutdown` no event
        steps, so a call, a watcher, a timer or a tick that ends later
        changes nothing."""
        if self._shut:
            return self._world
        assert not self._stepping, "a step runs its effects before the next step"
        self._stepping = True
        try:
            self._world, effects = grip_world.step(self._world, event)
            self._shut = isinstance(event, Shutdown)
            for effect in effects:
                self._run(effect)
        finally:
            self._stepping = False
        return self._world

    def _run(self, effect: Effect) -> None:
        match effect:
            case Call(robot=robot, call=call, request=request):
                spawn(self._call(robot, call, request))
            case Watch(robot=robot, start=start, label=label, goal=goal):
                spawn(self._watch(robot, start, label, goal))
            case CancelGoal(label=label, goal=goal):
                self._goals.cancel(label, goal)
            case Note(target=target, text=text):
                self._notes[target].set(text)
                log(text)
            case StartRetry(robot=robot, timer=timer):
                spawn(self._retry(robot, timer))
            case Changed():
                self._refresh()

    async def _call(self, robot: str, call: int, request: CallRequest) -> None:
        """Run the call, and step its outcome; an exception is its outcome
        too."""
        try:
            outcome = await self._outcome_of(robot, request)
        except Exception as e:
            log(f"{type(request).__name__} of {robot} raised: {e!r}")
            outcome = Raised(repr(e))
        self.step(CallDone(robot, call, outcome))

    async def _outcome_of(self, robot: str, request: CallRequest) -> CallOutcome:
        match request:
            case TakeLease():
                return await self._calls.take(robot)
            case CheckMove():
                return await self._calls.check_move(robot=robot)
            case ReleaseLease():
                return await self._calls.release(robot)
            case CheckMoveInstance(instance=instance):
                return await self._calls.check_move(instance=instance)
            case StopMoves():
                refused = await self._stops.stop(robot)
                return Answered(None) if refused is None else Failed(refused)
            case FireGoal(label=label, target=target):
                fired = await self._goals.fire(label, target)
                if fired is None:
                    return Unanswered()
                if fired.accepted:
                    return Answered(fired)
                return Failed(cause(fired.refusal.message))

    async def _watch(self, robot: str, start: int, label: str, goal: object) -> None:
        """Step the result of the goal of posture start `start`, or its loss.
        A cancelled watcher steps nothing."""
        try:
            note = await self._goals.result(label, goal)
        except Exception as e:
            log(f"move to {label}: result unknown: {e!r}")
            self.step(PostureResult(robot, start, ResultLost()))
            return
        self.step(PostureResult(robot, start, Finished(note)))

    async def _retry(self, robot: str, timer: int) -> None:
        await self._clock.sleep(self._timing.retry_s)
        self.step(RetryDue(robot, timer))

    def _refresh(self) -> None:
        try:
            self._on_change()
            self._refreshing.clear()
        except Exception as e:
            self._refreshing.trip(f"the page's refresh failed: {e!r}")

    def note(self, target: NoteTarget) -> str:
        """The note of `target` now, "" with none."""
        return self._notes[target].text()

    def grip_status(self, grip: GripName) -> GripStatus:
        """`grip` as the panel reads it now."""
        world = self._world
        must_release = frozenset(
            hand for hand in hands_of(grip) if world.hands.of(hand).must_release
        )
        return GripStatus(world.grips.of(grip).phase, must_release, self.note(grip))

    def posture_line(self) -> str | None:
        """What the posture row says: its note while it stands, else the move
        in flight, else None."""
        note = self.note(POSTURE_NOTE)
        if note:
            return note
        moving = self._world.posture_moving
        if moving is not None:
            return f"moving to {moving.label}..."
        return None

    def state_of(self, robot: str) -> str:
        """`robot` as the page and the status panel say it."""
        unavailable = status_text(self._world.fleet.status)
        return unavailable if unavailable is not None else self._state(robot)[0]

    def robot_line(self, chosen: str | None) -> RobotLine:
        """The chosen robot as the panel's robot row gives it: why the
        headset let go, for a few seconds, else why it cannot ask the
        fleet, else the robot's state."""
        note = self.note(ROBOT_NOTE)
        if note:
            return RobotLine(note, "waiting")
        unavailable = status_text(self._world.fleet.status)
        if unavailable is not None:
            return RobotLine(unavailable, "waiting")
        if chosen is None:
            return RobotLine("no robot paired", "resting")
        text, tone = self._state(chosen)
        return RobotLine(f"{chosen}: {text}", tone)

    def _state(self, robot: str) -> tuple[str, Tone]:
        """`robot`'s state in words, and its tone, while the listing is
        current."""
        fleet = self._world.fleet
        view = fleet.listed()
        if view.halted_by is not None:
            return halted_text(view.halted_by), "waiting"
        hold = self._world.robot(robot).hold
        driven_under_open = hold is not None and isinstance(hold.mode, UnderOpen)
        state = robot_state(view, robot, fleet.me, driven_under_open=driven_under_open)
        match state:
            case DrivenHere():
                tone = "driving"
            case Free():
                tone = "resting"
            case _:
                tone = "waiting"
        return state_text(state), tone

    def goals_status(self, status: GoalsStatus) -> None:
        """Each write of the goals' status; a change steps the world."""
        if status != self._world.fleet.status:
            self.step(GoalsStatusChanged(status))

    def goals_started(self, generation: int) -> None:
        """A start of the goals steps `GoalsStarted`."""
        self.step(GoalsStarted(generation))

    def fleet_changed(self, view: FleetView) -> None:
        """A listing steps `Listing`."""
        self.step(Listing(view))

    def own_lease_ended(self, robot: str, why: str) -> None:
        """The end of a lease of the headset steps `LeaseEnded`."""
        self.step(LeaseEnded(robot, why))

    def goals_ended(self, why: str) -> None:
        """The end of a goal of the fleet steps `GoalsEnded`."""
        self.step(GoalsEnded(why))
