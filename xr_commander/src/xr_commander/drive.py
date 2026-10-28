"""The robots the headset drives: the lease follows the grip.

A squeeze on a robot that the headset does not drive yet takes the robot.
The headset takes its lease, or, while another client opened its lease for a
take-over, asks the daemon whether it may move it. It then stops the planned
moves in flight on the robot, and only then streams. A press of a posture
button takes the robot the same way for the time of its move.

The headset holds a robot while a hand drives it, through the release burst
of the last hand, and while a posture move of its own runs; then it releases
the lease. It lets go at once, as at a grip release, when its lease ends, at
a halt, when the open lease it drives under closes or ends, and when a goal
of the fleet ends. The lease feedback names the robot of each lease that
ends: the headset asks the daemon whether it may still move that robot, and
lets go unless a fleet listing as recent as a yes names it as the holder.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal, Protocol, TypeVar

from xr_commander.bus import (
    REAL_CLOCK,
    Answered,
    Clock,
    Failed,
    FlashNote,
    Outcome,
    Unanswered,
    log,
    outcome_of,
    refusal_of,
    spawn,
)
from xr_commander.devices import Handedness, HandSample
from xr_commander.fleet import (
    DAEMON_SILENT,
    LEASE_NAME,
    DrivenHere,
    Fleet,
    FleetView,
    Free,
    Holder,
    Timing,
    halted_text,
    robot_state,
    state_text,
)
from xr_commander.robots import JOIN_COMMAND, PairedRobots

# Why the headset holds a robot: a hand that drives it, or a posture move.
Reason = Handedness | Literal["posture"]
POSTURE: Reason = "posture"
HANDS: tuple[Handedness, ...] = ("left", "right")

# Why a press does nothing while a hand holds the chosen robot.
GRIP_HELD = "release the grip first"
GRIP_JUST_RELEASED = "the grip was just released"

# Why a take ends while the goals of the fleet restart.
GOALS_ENDED = "a goal of the fleet ended"

Tone = Literal["driving", "waiting", "resting"]

Started = TypeVar("Started")


@dataclass
class HandState:
    """One hand's hold on a robot, shared by its two streams and the panel.
    The hand's pose stream writes it at each tick, and a let-go of its robot
    disengages it."""

    note: FlashNote
    # The hand's streams command `robot`.
    engaged: bool = False
    # The robot the hand holds, from its squeeze to the end of its release
    # burst.
    robot: str | None = None
    # A grip held through a refusal, a let-go or a change of robot drives
    # nothing until it is released.
    wait_release: bool = False
    # The robot chosen at the hand's last tick, and whether one ever was.
    chosen: str | None = None
    had_chosen: bool = False

    def observe(self, chosen: str | None, sample: HandSample | None) -> None:
        """This tick's chosen robot and grip. A squeeze held across a change
        of robot waits for a release, once a robot was chosen at all: the
        first robot chosen engages a squeeze already held. Only a fresh
        frame showing the grip released clears the wait; a stale tick says
        nothing about the grip, though it disengages the hand."""
        squeezing = sample is not None and sample.squeezing
        if chosen != self.chosen:
            self.wait_release = self.wait_release or (squeezing and self.had_chosen)
            self.engaged = False
            self.chosen = chosen
            self.had_chosen = self.had_chosen or chosen is not None
        if sample is not None and not sample.squeezing:
            self.wait_release = False
        if not squeezing:
            self.engaged = False

    def may_engage(self, sample: HandSample | None) -> bool:
        """Whether this tick's squeeze asks to engage the hand."""
        squeezing = sample is not None and sample.squeezing
        return squeezing and not self.engaged and not self.wait_release


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
    first listing that recent tells whether it ended the headset's lease,
    for the cause `why`."""

    revision: int
    why: str


@dataclass
class Hold:
    """The headset's hold on one robot, and why it holds it."""

    mode: Mode
    reasons: set[Reason]
    # Cancels the posture move of the headset on the robot, while one runs.
    cancel_posture: Callable[[], None] | None = None
    # The earliest lease end that a listing has not settled yet.
    pending_end: PendingEnd | None = None


@dataclass(frozen=True)
class Refusal:
    """Why the headset does not take a robot."""

    cause: str


@dataclass(frozen=True)
class Lost:
    """A posture start whose send failed: the backbone may run the move."""

    cause: str


@dataclass(frozen=True)
class Withdrawn:
    """The squeeze ended, or the person chose another robot, before the
    headset held the robot."""


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


class LimbStops:
    """`stop` of limb_motion on the robot's own backbone, with the reason
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
            return f"{robot} has no limb_motion: the launcher links none to the headset"
        request = self._stop.Request(reason=LEASE_NAME)
        call = self._stop.poll(self._node_runner, producer, request, self._timeout_s)
        stopped = await outcome_of(call, self._stop.Failure, f"stop of {robot}")
        return refusal_of(stopped, f"{robot} did not answer")


@dataclass(frozen=True)
class RobotLine:
    """The chosen robot as the status panel's robot row gives it."""

    text: str
    tone: Tone


def _always() -> bool:
    return True


class Driver:
    """The holds of the headset on its robots, by robot.

    Every method runs on the node's loop. One acquire, release or lease
    check of a robot runs at a time. `on_change` runs after each change of a
    hold or of the listing, so the page follows each of them."""

    def __init__(
        self,
        fleet: Fleet,
        calls: LeaseCalls,
        stops: Stops,
        *,
        timing: Timing,
        on_change: Callable[[], None],
        clock: Clock = REAL_CLOCK,
    ) -> None:
        self._fleet = fleet
        self._calls = calls
        self._stops = stops
        self._timing = timing
        self._on_change = on_change
        self._clock = clock
        self.hands = {hand: HandState(FlashNote(clock)) for hand in HANDS}
        # Why the headset let go last, for the panel.
        self.note = FlashNote(clock)
        self._holds: dict[str, Hold] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def holds(self, robot: str | None) -> bool:
        """Whether the headset holds `robot`."""
        return robot in self._holds

    def holds_for(self, robot: str | None, reason: Reason) -> bool:
        """Whether the headset holds `robot` for `reason`."""
        hold = self._holds.get(robot)
        return hold is not None and reason in hold.reasons

    def hand_holds(self, robot: str | None) -> bool:
        """Whether a hand holds `robot`, driving it or in its release
        burst."""
        return any(self.holds_for(robot, hand) for hand in HANDS)

    def driven_under_open(self, robot: str) -> bool:
        hold = self._holds.get(robot)
        return hold is not None and isinstance(hold.mode, UnderOpen)

    def press_refusal(self, robot: str | None) -> str | None:
        """Why a press on `robot` does nothing for a hand: a hand drives it,
        or a hand's release burst holds it. None while no hand holds it."""
        if self._driven_by_a_hand(robot):
            return GRIP_HELD
        if self.hand_holds(robot):
            return GRIP_JUST_RELEASED
        return None

    def _driven_by_a_hand(self, robot: str | None) -> bool:
        return any(hand.engaged and hand.robot == robot for hand in self.hands.values())

    async def acquire(
        self, robot: str, hand: Handedness, wanted: Callable[[], bool]
    ) -> Refusal | Withdrawn | None:
        """Take `robot` for a squeeze of `hand`: None once the headset holds
        it for the hand, else why not. `wanted` says whether the squeeze
        still asks for the robot, before the headset stops the moves in
        flight on it and once they stopped."""
        async with self._locks[robot]:
            held = await self._hold(robot, hand, wanted)
        return None if isinstance(held, Hold) else held

    async def start_posture_move(
        self, robot: str, start: Callable[[], Awaitable[Started | Lost | Refusal]]
    ) -> Started | Refusal:
        """Take `robot` for a posture move, as a squeeze does, then run
        `start`, the start of the move, while no other acquire of the robot
        can: a squeeze then stops a move that runs. A refused start drops
        the hold of the move, and a lost one ends as `end_lost_posture_move`
        says. A start that a let-go overtakes is refused, and
        `on_posture_cancel` cancels its move."""
        async with self._locks[robot]:
            held = await self._hold(robot, POSTURE, _always)
            if isinstance(held, Refusal):
                return held
            assert isinstance(held, Hold), "a press always wants its robot"
            started = await start()
            if isinstance(started, Lost):
                await self.end_lost_posture_move(robot)
                return Refusal(started.cause)
            if self._holds.get(robot) is not held:
                let_go = Refusal(f"the headset let go of {robot}")
                return started if isinstance(started, Refusal) else let_go
            if isinstance(started, Refusal):
                self.drop(robot, POSTURE)
            return started

    async def end_lost_posture_move(self, robot: str) -> None:
        """The send or the result of the posture move of the headset on
        `robot` is lost, so the backbone may run the move, and no goal
        handle cancels it. The headset stops the moves in flight on the
        robot while the move holds it, and after a let-go while no other
        client holds the robot's lease, as at a halt. Then the move drops its
        hold of the robot."""
        if self._may_stop_lost_move(robot):
            stopped = await self._stops.stop(robot)
            if stopped is not None:
                log(f"the stop of the lost posture move on {robot} failed: {stopped}")
        self.drop(robot, POSTURE)

    def _may_stop_lost_move(self, robot: str) -> bool:
        if self.holds_for(robot, POSTURE):
            return True
        view = self._fleet.view
        lease = view.leases.get(robot) if view is not None else None
        return lease is None or lease.holder == self._fleet.me

    async def _hold(
        self, robot: str, reason: Reason, wanted: Callable[[], bool]
    ) -> Hold | Refusal | Withdrawn:
        unavailable = self._fleet.unavailable()
        if unavailable is not None:
            return Refusal(unavailable)
        hold = self._holds.get(robot)
        if hold is None:
            return await self._take_robot(robot, reason, wanted)
        return await self._join(robot, hold, reason, wanted)

    async def _join(
        self, robot: str, hold: Hold, reason: Reason, wanted: Callable[[], bool]
    ) -> Hold | Refusal | Withdrawn:
        """`reason` joins a hold the headset has. A press joins none. A
        squeeze beside the other hand, or in a release burst under the
        headset's own lease, engages at once. A squeeze during a posture
        move of the headset stops that move, and one in a release burst
        under an open lease asks the daemon again and stops the moves in
        flight. The squeeze joins before the stop, so the end of the move it
        stops releases nothing, and a squeeze in its own burst leaves the
        burst's hold as it is."""
        if reason == POSTURE:
            return Refusal(
                self.press_refusal(robot) or f"a posture move runs on {robot}"
            )
        posture_runs = POSTURE in hold.reasons
        engages_at_once = isinstance(hold.mode, Taken) or self._driven_by_a_hand(robot)
        joined = reason not in hold.reasons
        hold.reasons.add(reason)
        if engages_at_once and not posture_runs:
            return hold
        refused = None
        if not posture_runs:
            checked = await self._calls.check_move(robot=robot)
            refused = refusal_of(checked, DAEMON_SILENT)
        if refused is None and self._holds.get(robot) is hold and wanted():
            stopped = await self._stops.stop(robot)
            refused = None if stopped is None else f"stop failed: {stopped}"
        if self._holds.get(robot) is not hold:
            return Refusal(f"the headset let go of {robot}")
        if refused is None and wanted():
            return hold
        if joined:
            self.drop(robot, reason)
        return Refusal(refused) if refused is not None else Withdrawn()

    async def _take_robot(
        self, robot: str, reason: Reason, wanted: Callable[[], bool]
    ) -> Hold | Refusal | Withdrawn:
        """Take the lease of `robot`, or the right to drive under its open
        lease, then stop the moves in flight on it: the hold, else why not.
        A squeeze that ended before the stop stops nothing."""
        generation = self._fleet.generation
        mode = await self._admit(robot)
        if isinstance(mode, Refusal):
            return mode
        ended = self._lost(robot, mode, generation)
        if ended is None and wanted():
            stopped = await self._stops.stop(robot)
            ended = (
                f"stop failed: {stopped}"
                if stopped is not None
                else self._lost(robot, mode, generation)
            )
        if ended is None and wanted():
            hold = Hold(mode, {reason})
            self._holds[robot] = hold
            self._on_change()
            return hold
        if isinstance(mode, Taken):
            self._release_later(robot)
        return Refusal(ended) if ended is not None else Withdrawn()

    async def _admit(self, robot: str) -> Mode | Refusal:
        view = self._fleet.view
        if view.halted_by is not None:
            return Refusal(halted_text(view.halted_by))
        if robot not in view.robots:
            return Refusal(
                f"{robot} is not in the fleet listing, which holds the copies of"
                f" the stack: join a robot as a copy with {JOIN_COMMAND}"
            )
        lease = view.leases.get(robot)
        if lease is not None and lease.open and lease.holder != self._fleet.me:
            checked = await self._calls.check_move(robot=robot)
            refused = refusal_of(checked, DAEMON_SILENT)
            return Refusal(refused) if refused else UnderOpen(lease.holder.instance)
        taken = await self._calls.take(robot)
        if isinstance(taken, Unanswered):
            # The daemon may have recorded the take: the release undoes it.
            self._release_later(robot)
        refused = refusal_of(taken, DAEMON_SILENT)
        return Refusal(refused) if refused else Taken()

    def _lost(self, robot: str, mode: Mode, generation: int) -> str | None:
        """What ended the right to drive `robot` while the daemon or the
        backbone answered, None while the headset still may."""
        view = self._fleet.view
        if self._fleet.generation != generation or view is None:
            return GOALS_ENDED
        if view.halted_by is not None:
            return halted_text(view.halted_by)
        if isinstance(mode, UnderOpen) and not _open_by(view, robot, mode.holder):
            return _open_lease_closed(mode.holder)
        return None

    def drop(self, robot: str | None, reason: Reason) -> None:
        """`reason` no longer holds `robot`: a hand's release burst ended, or
        a posture move ended. The last reason releases the lease."""
        hold = self._holds.get(robot)
        if hold is None or reason not in hold.reasons:
            return
        hold.reasons.discard(reason)
        if reason == POSTURE:
            hold.cancel_posture = None
        if hold.reasons:
            return
        del self._holds[robot]
        if isinstance(hold.mode, Taken):
            self._release_later(robot)
        self._on_change()

    def on_posture_cancel(self, robot: str, cancel: Callable[[], None]) -> None:
        """How a let-go of `robot` cancels the posture move of the headset.
        A move that starts after the let-go is cancelled at once."""
        hold = self._holds.get(robot)
        if hold is None or POSTURE not in hold.reasons:
            cancel()
            return
        hold.cancel_posture = cancel

    def let_go(self, robot: str, why: str, *, release: bool) -> None:
        """Let go of `robot` at once, as at a grip release: each hand on it
        disengages and waits for its grip to be released, the posture move
        of the headset is cancelled, and, when `release`, the lease is
        released."""
        hold = self._holds.pop(robot, None)
        for hand in self.hands.values():
            if hand.robot == robot and hand.engaged:
                hand.engaged = False
                hand.wait_release = True
        if hold is None:
            return
        if hold.cancel_posture is not None:
            hold.cancel_posture()
        self.note.set(f"let go of {robot}: {why}")
        log(f"let go of {robot}: {why}")
        if release and isinstance(hold.mode, Taken):
            self._release_later(robot)
        self._on_change()

    def fleet_changed(self, view: FleetView) -> None:
        """Let go of every robot at a halt, of each robot whose open lease
        the headset drives under once that lease is no longer open, and of
        each robot whose pending lease end `view` settles against the
        headset."""
        for robot, hold in list(self._holds.items()):
            why = _why_let_go(view, robot, hold)
            if why is not None:
                self.let_go(robot, why, release=False)
            else:
                self._settle_end(robot, hold, view)
        self._on_change()

    def own_lease_ended(self, robot: str, why: str) -> None:
        """The lease feedback gave the end of a lease of the headset on
        `robot`: the lease it holds, or one it released. The headset asks the
        daemon which, once no acquire or release of the robot runs."""
        generation = self._fleet.generation
        spawn(self._confirm_lease(robot, why or "the lease ended", generation))

    async def _confirm_lease(self, robot: str, why: str, generation: int) -> None:
        """Ask the daemon whether the headset may still move `robot`. A
        refusal lets go. A yes also holds for the open lease of another
        client, so the first listing as recent as the yes settles the end
        (`_settle_end`). A daemon that does not answer may still hold the
        lease, so the headset lets go and releases it."""
        async with self._locks[robot]:
            hold = self._holds.get(robot)
            if hold is None or not isinstance(hold.mode, Taken):
                return
            if self._fleet.generation != generation:
                return
            confirmed = await self._calls.check_move(robot=robot)
            if self._holds.get(robot) is not hold:
                return
            match confirmed:
                case Answered(response=revision):
                    pending = hold.pending_end
                    if pending is None or revision < pending.revision:
                        hold.pending_end = PendingEnd(revision, why)
                    if self._fleet.view is not None:
                        self._settle_end(robot, hold, self._fleet.view)
                case Failed():
                    self.let_go(robot, why, release=False)
                case Unanswered():
                    self.let_go(robot, DAEMON_SILENT, release=True)

    def _settle_end(self, robot: str, hold: Hold, view: FleetView) -> None:
        """Once `view` is as recent as the yes to the pending lease end of
        `hold`, let go of `robot` unless `view` names the headset as its
        holder."""
        end = hold.pending_end
        if end is None or view.revision < end.revision:
            return
        hold.pending_end = None
        if not _leased_by(view, robot, self._fleet.me):
            self.let_go(robot, end.why, release=False)

    def let_go_of_every_robot(self, why: str) -> None:
        """A goal of the fleet ended: the headset ends the other one, and
        each lease ends with the lease goal, so nothing is released."""
        for robot in list(self._holds):
            self.let_go(robot, why, release=False)

    def _release_later(self, robot: str) -> None:
        spawn(self._release(robot, self._fleet.generation))

    async def _release(self, robot: str, generation: int) -> None:
        """Release the lease of `robot`, asking again while the daemon does
        not answer, as long as the goals it was first asked under run and
        the headset does not hold the robot again."""
        while True:
            async with self._locks[robot]:
                if robot in self._holds or self._fleet.generation != generation:
                    return
                if not isinstance(await self._calls.release(robot), Unanswered):
                    return
            await self._clock.sleep(self._timing.retry_s)

    async def check_move(self, instance: str) -> str | None:
        """Whether the headset may call a member that moves the robot of
        `instance`, as before each posture move: None when it may, else the
        cause."""
        checked = await self._calls.check_move(instance=instance)
        return refusal_of(checked, DAEMON_SILENT)

    def state_of(self, robot: str) -> str:
        """`robot` as the page and the status panel say it."""
        unavailable = self._fleet.unavailable()
        return unavailable if unavailable is not None else self._state(robot)[0]

    def robot_line(self, chosen: str | None) -> RobotLine:
        """The chosen robot as the panel's robot row gives it: why the
        headset let go, for a few seconds, else why it cannot ask the
        fleet, else the robot's state."""
        note = self.note.text()
        if note:
            return RobotLine(note, "waiting")
        unavailable = self._fleet.unavailable()
        if unavailable is not None:
            return RobotLine(unavailable, "waiting")
        if chosen is None:
            return RobotLine("no robot paired", "resting")
        text, tone = self._state(chosen)
        return RobotLine(f"{chosen}: {text}", tone)

    def _state(self, robot: str) -> tuple[str, Tone]:
        """`robot`'s state in words, and its tone, while the listing is
        current."""
        view = self._fleet.view
        if view.halted_by is not None:
            return halted_text(view.halted_by), "waiting"
        state = robot_state(
            view, robot, self._fleet.me, driven_under_open=self.driven_under_open(robot)
        )
        match state:
            case DrivenHere():
                tone = "driving"
            case Free():
                tone = "resting"
            case _:
                tone = "waiting"
        return state_text(state), tone


def _why_let_go(view: FleetView, robot: str, hold: Hold) -> str | None:
    """Why `view` ends the headset's right to drive `robot`, None while it
    does not."""
    if view.halted_by is not None:
        return halted_text(view.halted_by)
    match hold.mode:
        case UnderOpen(holder=holder) if not _open_by(view, robot, holder):
            return _open_lease_closed(holder)
    return None


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
