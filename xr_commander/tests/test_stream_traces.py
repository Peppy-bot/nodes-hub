"""The base and pose streams' traces: hypothesis draws orders of the
headset's events, the fleet's and each follower's, runs the streams' real
ticks and the real driver on them with no wire, and checks at each tick that
the sends keep the leader rule.

Every call the driver makes waits until a rule answers it. Behind each base
pair runs drive_link's follower (`follower_model`), and each message on its
way arrives after a drawn delay of at most 0.5 s, in order, or not at all.
The daemon keeps its book by `daemon_book`, each change listed while the goals
run, and each stream's sends keep the rule of `stream_model`. The host's clock
is a `ManualClock` and the instance's clock a count of ns that the rules move.
Each machine that drives the driver runs every rule on an event loop of its
own, so each task the driver spawns runs at the rule's end, as on the node's
loop. Every run draws the same traces.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType, SimpleNamespace
from typing import TypeVar

from control_core_py.geometry import Pose
from hypothesis import event, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    precondition,
    rule,
    run_state_machine_as_test,
)

from tests.daemon_book import Daemon
from tests.follower_model import (
    MAX_DELIVERY_DELAY_NS,
    REPORT_PERIOD_NS,
    FollowerModel,
    InOrderDelivery,
    Published,
)
from tests.hand_covers import (
    Choice,
    let_go_covers,
    noted_let_go,
    pins_lost,
    tick_covers,
)
from tests.helpers import (
    CHANGES_ENDED,
    DIFFERENTIAL,
    ELSEWHERE,
    HALTER,
    HEADSET,
    HOLONOMIC,
    HOME_REACHED,
    MOST_PAIRS,
    POSE,
    REFUSED_GOAL,
    SETTINGS,
    START_NS,
    TICK_S,
    ManualClock,
    NoWireSlot,
    OnCue,
    Peer,
    ScriptedHeadset,
    StopsOnCue,
    controller,
    listed,
    pose_at,
    postures_of,
    reported,
    setpoint_sent,
    settle,
    states,
    stick_axes,
)
from tests.stream_model import BURST_S, StreamModel
from xr_commander.base import (
    STILL,
    BaseLimits,
    BaseReports,
    NotedStops,
    Twist,
    base_tick,
    twist_of,
)
from xr_commander.bus import Answered, Failed, Outcome, Unanswered
from xr_commander.clutch import HandClutch, Setpoint
from xr_commander.devices import (
    RESTING,
    Hand,
    HandSample,
    Thumbstick,
    XrFrame,
    fresh_frame,
    parse_hands,
)
from xr_commander.drive import Driver, Stops
from xr_commander.fleet import FleetView, Following, Starting, Timing
from xr_commander.grip_world import (
    BASE,
    ByHand,
    Effect,
    Engaged,
    Idle,
    Lingering,
    Phase,
    Pin,
    Pins,
    PostureTarget,
    Press,
    PressInputs,
    Reading,
    Requesting,
    Slot,
    World,
    driven_pins,
    pin_on,
    robot_of_phase,
)
from xr_commander.publish import GripStream, MeasuredPoses, pose_tick
from xr_commander.robots import PairedRobots

# The draws of every trace: the same at each run, none kept between runs.
TRACES = settings(
    max_examples=200,
    stateful_step_count=60,
    deadline=None,
    derandomize=True,
    database=None,
)

# The robots of the traces, each with a pair on the slot at the start, and
# the limits of each one's base: alpha's differential, bravo's sideways too.
ROBOTS = ("alpha", "bravo")
LIMITS = MappingProxyType({"alpha": DIFFERENTIAL, "bravo": HOLONOMIC})
NS_PER_S = 1_000_000_000
# drive_link's half second: a report counts this long after its arrival, and
# a following starts the stream only this long after the newest stop ended.
HALF_SECOND_NS = 500_000_000
# How far `advance` moves both clocks, below drive_link's half second, and
# how far a `leap` moves them (ns); a step back of the instance's clock goes
# back at most as far as a leap goes on.
LONGEST_ADVANCE_NS = 450_000_000
SHORTEST_LEAP_NS = 500_000_000
LONGEST_LEAP_NS = 2_000_000_000
# A message's delay on its way (ns), None for one dropped: no delay, the
# shortest, a report period, half the bound, and each side of the bound.
DELAYS_NS = st.sampled_from(
    (
        None,
        0,
        1,
        REPORT_PERIOD_NS,
        MAX_DELIVERY_DELAY_NS // 2,
        MAX_DELIVERY_DELAY_NS - 1,
        MAX_DELIVERY_DELAY_NS,
    )
)
# How often a follower whose wheels report sends a drive_states (ns): half
# base_state's bound, and the bound.
REPORT_PERIODS_NS = (REPORT_PERIOD_NS // 2, REPORT_PERIOD_NS)
# Each axis of a stick: at rest, within joy's rest band, half a deflection
# past it, full, and a value no gamepad gives.
STICKS = st.sampled_from((0.0, 0.03, -0.03, 0.525, -0.525, 1.0, -1.0, math.nan))
# How far past the stale bound a stale frame is (s).
STALE_BY_S = 0.1
# drive_link's follow_state of a base that follows.
FOLLOWING = "following"

# The posture the traces press, and a goal the backbone accepts.
POSTURE_LABEL = "home"
ACCEPTED = SimpleNamespace(accepted=True)

# Any value kept per hand.
Value = TypeVar("Value")


def by_hand(value_of: Callable[[Hand], Value]) -> ByHand[Value]:
    """`value_of` each hand."""
    return ByHand(value_of(Hand.LEFT), value_of(Hand.RIGHT))


class CallKind(StrEnum):
    """A call of the driver, as the rule that answers it names it."""

    TAKE = "take"
    RELEASE = "release"
    CHECK_MOVE = "check_move"
    CHECK_INSTANCE = "check_move_instance"
    STOP = "stop"
    FIRE = "fire_goal"
    RESULT = "goal_result"


class Answer(StrEnum):
    """How a rule answers a waiting call."""

    ANSWERED = "answered"
    FAILED = "failed"
    UNANSWERED = "unanswered"
    RAISED = "raised"
    FINISHED = "finished"
    LOST = "lost"


# The answers each call may get: a stop names its own cause when it gets no
# answer, and a goal's result comes or does not.
CALL_ANSWERS = (Answer.ANSWERED, Answer.FAILED, Answer.UNANSWERED, Answer.RAISED)
ANSWERS: Mapping[CallKind, tuple[Answer, ...]] = MappingProxyType(
    {
        CallKind.TAKE: CALL_ANSWERS,
        CallKind.RELEASE: CALL_ANSWERS,
        CallKind.CHECK_MOVE: CALL_ANSWERS,
        CallKind.CHECK_INSTANCE: CALL_ANSWERS,
        CallKind.STOP: (Answer.ANSWERED, Answer.FAILED, Answer.RAISED),
        CallKind.FIRE: CALL_ANSWERS,
        CallKind.RESULT: (Answer.FINISHED, Answer.LOST),
    }
)


@dataclass(frozen=True)
class CallOf:
    """What a waiting lease or goal call names: its kind, and its robot,
    instance or posture."""

    kind: CallKind
    names: str


class CallsOnCue(OnCue):
    """The daemon's side of the lease calls (`drive.LeaseCalls`): each call
    waits until a rule answers it."""

    async def take(self, robot: str) -> Outcome:
        return await self.wait_for_answer(CallOf(CallKind.TAKE, robot))

    async def release(self, robot: str) -> Outcome:
        return await self.wait_for_answer(CallOf(CallKind.RELEASE, robot))

    async def check_move(
        self, *, robot: str | None = None, instance: str | None = None
    ) -> Outcome:
        if robot is not None:
            return await self.wait_for_answer(CallOf(CallKind.CHECK_MOVE, robot))
        return await self.wait_for_answer(CallOf(CallKind.CHECK_INSTANCE, instance))


class GoalsOnCue(OnCue):
    """The posture goals (`drive.Goals`): each send and each result waits
    until a rule answers it."""

    async def fire(self, label: str, target: PostureTarget) -> object | None:
        return await self.wait_for_answer(CallOf(CallKind.FIRE, label))

    async def result(self, label: str, goal: object) -> str:
        return await self.wait_for_answer(CallOf(CallKind.RESULT, label))

    def cancel(self, label: str, goal: object) -> None:
        """The backbone takes the cancel: the goal's result still comes when
        a rule answers it."""


@dataclass(frozen=True)
class WaitingCall:
    """A call that waits on a rule's answer: the fake it waits on, its index
    there, its kind, and what it names."""

    fake: OnCue
    index: int
    kind: CallKind
    names: str


class HandGrip(StrEnum):
    """A controller's grip as a rule sets it."""

    SQUEEZE = "squeeze"
    RELEASE = "release"
    UNTRACKED = "untracked"


def reading_of_grip(grip: HandGrip, stale: bool) -> Reading:
    """A hand's reading by its grip, UNKNOWN in a stale frame."""
    if stale or grip is HandGrip.UNTRACKED:
        return Reading.UNKNOWN
    return Reading.SQUEEZE if grip is HandGrip.SQUEEZE else Reading.RELEASED


@dataclass(frozen=True)
class HandClaim:
    """One hand as the deadman reads it: its last reading other than
    UNKNOWN, and whether an interruption covered its squeeze until a fresh
    RELEASED."""

    last_fresh: Reading | None = None
    covered: bool = False

    def read(self, reading: Reading) -> HandClaim:
        """The hand once a tick read `reading`."""
        if reading is Reading.UNKNOWN:
            return self
        return HandClaim(reading, self.covered and reading is not Reading.RELEASED)

    def interrupted(self) -> HandClaim:
        """The hand once an interruption reached it: a squeeze it holds
        counts no more."""
        covered = self.covered or self.last_fresh is Reading.SQUEEZE
        return replace(self, covered=covered)

    def holds(self, reading: Reading) -> bool:
        """Whether `reading` of the hand is a squeeze that counts."""
        return reading is Reading.SQUEEZE and not self.covered

    @property
    def claims(self) -> bool:
        """Whether the hand last read a squeeze that counts."""
        return self.last_fresh is not None and self.holds(self.last_fresh)


@dataclass(frozen=True)
class BaseEngagement:
    """The base grip's engagement as the deadman holds it: its robot and its
    base pin."""

    robot: str
    pin: Hashable


@dataclass(frozen=True)
class DeadmanModel:
    """The base's deadman as a trace's own events and the world's let-go
    notes give it. It holds from the base's engagement on a squeeze of the
    chosen robot, at the answer of its acquisition's stop or a rejoin at its
    own tick, until a tick with no held hand, a change of the chosen robot,
    the pin leaving, a let-go of the robot, a halt's start or the goals' end.
    A let-go is the robot row's note `let go of <robot>: <why>`, which the
    world sets in the step that lets go of a robot it held: the deadman reads
    that output of the world, and the world's row tests hold the lease rules
    that decide when it comes. A hand whose squeeze an
    interruption covered counts again only after a fresh RELEASED: the
    interruptions of a tick and of a let-go (`hand_covers`), a refusal of the
    base's request, a halt's start and the goals' end."""

    hands: ByHand[HandClaim] = ByHand(HandClaim(), HandClaim())
    choice: Choice = Choice()
    engaged: BaseEngagement | None = None

    def _covered(self, hands: Iterable[Hand]) -> DeadmanModel:
        claims = self.hands
        for hand in hands:
            claims = claims.with_hand(hand, claims.of(hand).interrupted())
        return replace(self, hands=claims)

    def _held(self, readings: ByHand[Reading]) -> bool:
        return any(self.hands.of(hand).holds(readings.of(hand)) for hand in Hand)

    def ticked(
        self,
        readings: ByHand[Reading],
        chosen: str | None,
        phase: Phase,
        peers: Pins,
        pins: Pins,
        *,
        down: bool,
    ) -> DeadmanModel:
        """The deadman after a base tick read `readings`, with `chosen` the
        chosen robot, `phase` the base's phase before the tick, `peers` on
        the base slot and `pins` the chosen robot's, `down` while the stack
        is halted or the goals do not run."""
        hands = by_hand(lambda hand: self.hands.of(hand).read(readings.of(hand)))
        covered, choice = tick_covers(
            Hand,
            {hand: readings.of(hand) for hand in Hand},
            chosen,
            self.choice,
            down=down,
            lost=pins_lost(phase, peers, pins),
        )
        model = replace(self, hands=hands, choice=choice)._covered(covered)
        engaged = model.engaged
        if engaged is None:
            return model
        on_its_pin = Pin(Slot.BASE, engaged.pin) in peers
        if model._held(readings) and chosen == engaged.robot and on_its_pin:
            return model
        return replace(model, engaged=None)

    def _engaging(self, phase: Engaged, holds: bool) -> DeadmanModel:
        """The deadman once the base engaged on `phase`, holding when
        `holds`."""
        engagement = BaseEngagement(phase.robot, pin_on(phase.pins, Slot.BASE))
        return replace(self, engaged=engagement if holds else None)

    def engages_at_tick(
        self, phase: Engaged, readings: ByHand[Reading]
    ) -> DeadmanModel:
        """The deadman once the base engaged at its own tick: it holds while
        that tick read a squeeze that counts, on the chosen robot."""
        holds = self._held(readings) and self.choice.seen == phase.robot
        return self._engaging(phase, holds)

    def engages_by_stop(self, phase: Engaged) -> DeadmanModel:
        """The deadman once the base engaged at the answer of its
        acquisition's stop: it holds from its next tick while a hand still
        claims a squeeze, on the chosen robot."""
        claimed = any(self.hands.of(hand).claims for hand in Hand)
        holds = claimed and self.choice.seen == phase.robot
        return self._engaging(phase, holds)

    def refuses_engagement(self) -> DeadmanModel:
        """The deadman once the base engaged at a step that engages no
        base."""
        return replace(self, engaged=None)

    def refused(self) -> DeadmanModel:
        """The deadman once the base's request was refused, or withdrawn by a
        let-go: the base's hands are interrupted."""
        return self._covered(Hand)

    def listed(self, view: FleetView, previous: FleetView | None) -> DeadmanModel:
        """The deadman once the listing `view` came after `previous`: a halt's
        start lets go."""
        halted = previous is not None and previous.halted_by is not None
        if view.halted_by is not None and not halted:
            return self.lets_go()
        return self

    def let_go_of(self, robot: str) -> DeadmanModel:
        """The deadman once the world noted its let-go of `robot`: an
        engagement on it ends, and the let-go covers the hands of
        `let_go_covers`."""
        engaged = self.engaged
        ended = engaged is not None and engaged.robot == robot
        model = replace(self, engaged=None) if ended else self
        return model._covered(let_go_covers(robot, self.choice))

    def lets_go(self) -> DeadmanModel:
        """The deadman at a halt's start or the goals' end: it lets go, and
        both hands are interrupted."""
        return replace(self._covered(Hand), engaged=None)


@dataclass(frozen=True)
class BasePair:
    """A pair of the base slot: its robot, its peer, drive_link's follower
    behind it, and the order of its drive_states and of its setpoints on
    their way."""

    robot: str
    peer: Peer
    follower: FollowerModel[SimpleNamespace]
    states: InOrderDelivery
    setpoints: InOrderDelivery


@dataclass(frozen=True)
class StatesSent:
    """A drive_states a follower published, and its place among its pair's
    publications."""

    published: Published[SimpleNamespace]
    index: int


@dataclass(frozen=True)
class OnTheWay:
    """A message on its way from or to `peer`: a drive_states to the
    headset, or a setpoint to the follower, arriving at `arrival_ns` on the
    instance's clock."""

    arrival_ns: int
    peer: Peer
    message: StatesSent | Twist


@dataclass(frozen=True)
class Arrival:
    """A drive_states as the headset's drain took it: when, on the elapsed
    instance clock, its place among its pair's publications, its
    follow_state and its limits."""

    elapsed_ns: int
    index: int
    follow_state: str
    limits: SimpleNamespace


@dataclass(frozen=True)
class StopNoted:
    """A stop the driver sent: its robot, when it went out on the elapsed
    instance clock, how many drive_states each pair had published by then,
    and when it was answered or refused, None while in flight."""

    robot: str
    sent_ns: int
    published: Mapping[Peer, int]
    ended_ns: int | None = None


@dataclass(frozen=True)
class TickRecord:
    """One base tick: the host's clock, what the frame's sticks command under
    the report of the pin it drives, the base phase after it, and its
    send."""

    host_s: float
    frame_twist: Twist | None
    phase_after: Phase
    send: tuple[Peer, Twist] | None


def stick_of(frame: XrFrame | None, hand: Hand) -> Thumbstick:
    """`hand`'s stick in `frame`, at rest while it is not tracked."""
    sample = frame.hand(hand) if frame is not None else None
    return RESTING if sample is None else sample.thumbstick


def commanded(frame: XrFrame | None, report: Arrival | None) -> Twist | None:
    """What `frame`'s sticks command under `report`'s limits, None with no
    report: the left stick's forward and leftward deflection, and the right
    stick's leftward one as the turn."""
    if report is None:
        return None
    limits = BaseLimits.parse(report.limits)
    return twist_of(stick_of(frame, Hand.LEFT), stick_of(frame, Hand.RIGHT), limits)


class NotingDriver(Driver):
    """The real driver, which keeps the robot of each let-go its steps note
    (`noted_let_go`), in the order it runs their effects."""

    let_gos: tuple[str, ...] = ()

    def _run(self, effect: Effect) -> None:
        robot = noted_let_go(effect)
        if robot is not None:
            self.let_gos = (*self.let_gos, robot)
        super()._run(effect)


class DriverTraces(RuleBasedStateMachine):
    """What the base and the pose traces share: an event loop of the
    machine's own, the host's clock, the real driver with each call waiting
    until a rule answers it, keeping each let-go it notes (`NotingDriver`),
    the daemon's book with each change listed while the goals run, whether
    the goals run, one slot whose pairs the rules change, and the choice of
    robot. A subclass builds `robots`, `driver` and `stream_model` on these,
    and notes what each rule did (`observe`) and each pair that joins
    (`paired`) or leaves (`unpaired`)."""

    robots: PairedRobots
    driver: NotingDriver
    stream_model: StreamModel

    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        # Each task of the loop that has not ended, so that a rule fails on
        # the exception of any.
        self.tasks: list[asyncio.Task] = []
        self.loop.set_task_factory(self.kept_task)
        self.clock = ManualClock()
        self.calls = CallsOnCue()
        self.stops = StopsOnCue()
        self.goals = GoalsOnCue()
        self.daemon = Daemon.start()
        # Whether the goals run, as the headset was told, and the listing it
        # got last, None while the goals do not run.
        self.goals_run = True
        self.listed_view: FleetView | None = self.daemon.view(ROBOTS)
        self.session = ScriptedHeadset(self.clock)
        self.slot = NoWireSlot()
        self.pairs: tuple[tuple[str, Peer], ...] = ()
        self.peers_made = 0

    def kept_task(self, loop, coro, **kwargs) -> asyncio.Task:
        """The loop's task factory: each task is kept until it ends."""
        task = asyncio.Task(coro, loop=loop, **kwargs)
        self.tasks.append(task)
        return task

    def teardown(self) -> None:
        """Every task still waiting is cancelled, then the loop closes."""

        async def cancel_waiting() -> None:
            me = asyncio.current_task()
            waiting = [task for task in asyncio.all_tasks() if task is not me]
            for task in waiting:
                task.cancel()
            await asyncio.gather(*waiting, return_exceptions=True)

        self.loop.run_until_complete(cancel_waiting())
        self.loop.close()

    def listed_driver(self, stops: Stops) -> NotingDriver:
        """The real driver on the waiting calls and `stops`, keeping each
        let-go it notes, its goals running and both robots listed free."""
        driver = NotingDriver(
            HEADSET,
            self.calls,
            stops,
            self.goals,
            timing=Timing(),
            on_change=lambda: None,
            clock=self.clock,
        )
        listed(driver, *ROBOTS)
        return driver

    def run(self, body: Callable[[], Awaitable[object]]) -> None:
        """`body` on the machine's loop, then each task it woke up to its
        next wait, as the node's loop runs them; a task that raised fails
        the rule. Then the machine notes what the rule did."""

        async def rule_body() -> None:
            await body()
            await settle()

        self.loop.run_until_complete(rule_body())
        ended = [task for task in self.tasks if task.done()]
        self.tasks = [task for task in self.tasks if not task.done()]
        for task in ended:
            if not task.cancelled() and (error := task.exception()) is not None:
                raise error
        self.observe()

    def act(self, action: Callable[[], object]) -> None:
        """`action`, a step of the driver or a change it reads, run as `run`
        runs a body."""

        async def body() -> None:
            action()

        self.run(body)

    def observe(self) -> None:
        """What the machine notes after each rule."""
        raise NotImplementedError

    def paired(self, robot: str, peer: Peer, data: st.DataObject | None) -> None:
        """What the machine keeps for a pair that joined the slot; `data`
        draws what the pair is, None for a pair of the trace's start."""
        raise NotImplementedError

    def unpaired(self, peer: Peer) -> None:
        """What the machine drops of a pair that left the slot."""
        raise NotImplementedError

    def pairs_of(self, robot: str) -> tuple[Peer, ...]:
        """`robot`'s pairs on the slot, in delivered order."""
        return tuple(peer for name, peer in self.pairs if name == robot)

    def slot_peers(self) -> frozenset[Peer]:
        """Every pair on the slot."""
        return frozenset(peer for _robot, peer in self.pairs)

    def paired_robots(self) -> list[str]:
        """Each robot with a pair on the slot, in name order."""
        return sorted({robot for robot, _peer in self.pairs})

    def joinable(self) -> list[str]:
        """Each robot that may take one more pair."""
        return [robot for robot in ROBOTS if len(self.pairs_of(robot)) < MOST_PAIRS]

    def new_peer(self, robot: str) -> Peer:
        """A pair of `robot` that the slot never held."""
        self.peers_made += 1
        return Peer(f"{robot} {self.peers_made}")

    def set_pairs(self, pairs: tuple[tuple[str, Peer], ...]) -> None:
        """The slot holds `pairs`, in delivered order, from its next read
        on."""
        self.pairs = pairs
        self.slot.set_pairs(*pairs)

    def join(self, robot: str, *, ahead: bool, data: st.DataObject | None) -> None:
        """A new pair of `robot` joins the slot, first in delivered order
        when `ahead`; `data` draws what the pair is, None at the trace's
        start."""
        peer = self.new_peer(robot)
        joined = (robot, peer)
        self.set_pairs((joined, *self.pairs) if ahead else (*self.pairs, joined))
        self.paired(robot, peer, data)
        if ahead:
            self.stream_model = self.stream_model.with_join_ahead(peer)

    def waiting_calls(self) -> list[WaitingCall]:
        """Every call that waits on a rule's answer."""
        stops = [
            WaitingCall(self.stops, index, CallKind.STOP, self.stops.calls[index][0])
            for index in self.stops.unanswered()
        ]
        calls = [
            WaitingCall(
                fake, index, fake.calls[index][0].kind, fake.calls[index][0].names
            )
            for fake in (self.calls, self.goals)
            for index in fake.unanswered()
        ]
        return stops + calls

    def give(self, call: WaitingCall, answer: Answer) -> None:
        """`call` gets `answer`; a take or a release the daemon serves moves
        its book, and one its book refuses is refused."""
        match call.kind, answer:
            case _, Answer.RAISED:
                call.fake.throw(call.index, RuntimeError(f"the {call.kind} broke off"))
            case CallKind.RESULT, Answer.LOST:
                call.fake.throw(call.index, TimeoutError("no result came"))
            case CallKind.RESULT, Answer.FINISHED:
                call.fake.answer(call.index, HOME_REACHED)
            case CallKind.STOP, Answer.ANSWERED:
                call.fake.answer(call.index, None)
            case CallKind.STOP, Answer.FAILED:
                call.fake.answer(call.index, f"{call.names} refused the stop")
            case CallKind.FIRE, Answer.ANSWERED:
                call.fake.answer(call.index, ACCEPTED)
            case CallKind.FIRE, Answer.FAILED:
                call.fake.answer(call.index, REFUSED_GOAL)
            case CallKind.FIRE, Answer.UNANSWERED:
                call.fake.answer(call.index, None)
            case CallKind.TAKE, Answer.ANSWERED:
                book = self.daemon.take(call.names, goals_down=not self.goals_run)
                self.served(call, book)
            case CallKind.RELEASE, Answer.ANSWERED:
                self.served(call, self.daemon.release(call.names))
            case CallKind.CHECK_MOVE | CallKind.CHECK_INSTANCE, Answer.ANSWERED:
                call.fake.answer(call.index, Answered(self.daemon.revision))
            case _, Answer.FAILED:
                call.fake.answer(call.index, Failed(f"the {call.kind} was refused"))
            case _, Answer.UNANSWERED:
                call.fake.answer(call.index, Unanswered())
            case _:
                raise AssertionError(f"a {call.kind} takes no answer {answer}")

    def served(self, call: WaitingCall, book: Daemon | None) -> None:
        """A take or a release the daemon served: its answer and the book
        after it, or its refusal when the book is None."""
        if book is None:
            call.fake.answer(call.index, Failed(f"the daemon refused the {call.kind}"))
            return
        self.set_book(book)
        call.fake.answer(call.index, Answered(None))

    def set_book(self, book: Daemon) -> None:
        """The daemon's book becomes `book`, its change listed while the
        goals run, inside the run of a rule."""
        changed = book != self.daemon
        self.daemon = book
        if changed and self.goals_run:
            self.listing()

    def listing(self) -> None:
        """The changes goal's feedback of the daemon's book: the status,
        then the listing."""
        view = self.daemon.view(ROBOTS)
        self.driver.goals_status(Following())
        self.driver.fleet_changed(view)
        self.goals_run = True
        self.listed_view = view

    @precondition(lambda self: self.joinable())
    @rule(data=st.data(), ahead=st.booleans())
    def pair_joins(self, data: st.DataObject, ahead: bool) -> None:
        """A new pair of a robot joins the slot, first in delivered order
        when `ahead`."""
        robot = data.draw(st.sampled_from(self.joinable()), label="robot")
        self.join(robot, ahead=ahead, data=data)

    @precondition(lambda self: self.pairs)
    @rule(data=st.data())
    def pair_leaves(self, data: st.DataObject) -> None:
        """A pair leaves the slot."""
        _robot, peer = data.draw(st.sampled_from(self.pairs), label="pair")
        self.set_pairs(tuple(pair for pair in self.pairs if pair[1] != peer))
        self.unpaired(peer)

    @precondition(lambda self: self.pairs)
    @rule(data=st.data())
    def pair_restarts(self, data: st.DataObject) -> None:
        """A robot's first pair restarts as a new pair in its place."""
        robot = data.draw(st.sampled_from(self.paired_robots()), label="robot")
        old, new = self.pairs_of(robot)[0], self.new_peer(robot)
        self.set_pairs(
            tuple((name, new if peer == old else peer) for name, peer in self.pairs)
        )
        self.unpaired(old)
        self.paired(robot, new, data)

    @rule(robot=st.sampled_from(ROBOTS))
    def choose(self, robot: str) -> None:
        """The page chooses `robot`, refused while it has no pair."""
        self.robots.choose(robot)

    @precondition(lambda self: self.waiting_calls())
    @rule(data=st.data())
    def answer(self, data: st.DataObject) -> None:
        """A waiting call gets one of the answers of its kind."""
        call = data.draw(st.sampled_from(self.waiting_calls()), label="call")
        answer = data.draw(st.sampled_from(ANSWERS[call.kind]), label="answer")
        self.act(lambda: self.give(call, answer))


class BaseTraces(DriverTraces):
    """`base_tick` and the real driver on the base slot, against drive_link's
    follower behind each pair. Every tick keeps the deadman, follows now,
    the headset's own stops, the liveness, the burst, the lease, the value
    and the pins; the step rate rests on a tick that holds no await. The
    burst's zeros alone while the base does not follow come from follows
    now, and the burst's length from the stream's rule."""

    def __init__(self) -> None:
        super().__init__()
        self.instance_ns = START_NS
        self.reports = BaseReports(lambda: self.instance_ns)
        # The elapsed instance clock, on which a step back counts as no time.
        self.elapsed_ns = self.reports.mark()
        self.noted = NotedStops(self.stops, self.reports)
        self.driver = self.listed_driver(self.noted)
        self.robots = PairedRobots(None, arms={}, grippers={}, base=self.slot)
        self.base_tick = base_tick(
            topic_module=self.slot,
            driver=self.driver,
            stops=self.noted,
            reports=self.reports,
            session=self.session,
            robots=self.robots,
            settings=SETTINGS,
            clock=self.clock,
        )
        assert not inspect.iscoroutinefunction(self.base_tick), (
            "the step rate: the base tick awaits"
        )
        self.stream_model = StreamModel[Twist](Slot.BASE, BURST_S)
        self.base_pairs: dict[Peer, BasePair] = {}
        # The pairs whose follower's clock gave no reading yet.
        self.unread_clocks: frozenset[Peer] = frozenset()
        # How many drive_states each pair published, and its newest one that
        # arrived.
        self.published: dict[Peer, int] = {}
        self.newest_arrival: dict[Peer, Arrival] = {}
        self.on_the_way: tuple[OnTheWay, ...] = ()
        self.stop_notes: list[StopNoted] = []
        # The index in `stop_notes` of each stop answered or refused, in the
        # order they ended, and of each stop a follower took.
        self.stops_ended: list[int] = []
        self.taken: frozenset[int] = frozenset()
        self.hand_grips = {hand: HandGrip.UNTRACKED for hand in Hand}
        self.deflections = (0.0, 0.0, 0.0)
        self.deadman = DeadmanModel()
        # How many of the let-gos the driver noted the deadman read.
        self.let_gos_read = 0
        self.base_phase: Phase = Idle()
        self.generation = 1
        for robot in ROBOTS:
            self.join(robot, ahead=False, data=None)

    # The pairs and their followers.

    def paired(self, robot: str, peer: Peer, data: st.DataObject | None) -> None:
        """A follower behind the new pair, its wheels reporting. A pair of
        the trace's start reports every 0.1 s and its clock reads now; a pair
        that joins later draws its report period and whether its clock reads
        now or at the `clock_reads` rule."""
        period = REPORT_PERIOD_NS
        reads_now = True
        if data is not None:
            period = data.draw(st.sampled_from(REPORT_PERIODS_NS), label="period")
            reads_now = data.draw(st.booleans(), label="its clock reads now")
        follower = FollowerModel(LIMITS[robot], period)
        if reads_now:
            follower.advance_to(self.instance_ns)
        else:
            self.unread_clocks = self.unread_clocks | {peer}
        follower.silence_ends()
        self.base_pairs[peer] = BasePair(
            robot, peer, follower, InOrderDelivery(), InOrderDelivery()
        )
        self.published[peer] = 0

    def unpaired(self, peer: Peer) -> None:
        """The pair is gone, and every message on its way from or to it."""
        del self.base_pairs[peer]
        del self.published[peer]
        self.unread_clocks = self.unread_clocks - {peer}
        self.newest_arrival.pop(peer, None)
        self.on_the_way = tuple(m for m in self.on_the_way if m.peer != peer)

    def followers_of(self, robot: str) -> tuple[FollowerModel[SimpleNamespace], ...]:
        """The followers of `robot`'s pairs, in delivered order."""
        return tuple(self.base_pairs[peer].follower for peer in self.pairs_of(robot))

    def untaken_stops(self) -> list[int]:
        """Each stop sent that no follower took, of a robot with a pair."""
        return [
            index
            for index, note in enumerate(self.stop_notes)
            if index not in self.taken and self.pairs_of(note.robot)
        ]

    # The messages on their way.

    def publish(
        self,
        data: st.DataObject,
        pair: BasePair,
        published: list[Published[SimpleNamespace]],
    ) -> None:
        """Each of `published`, numbered among its pair's publications, on
        its way after a drawn delay, or dropped."""
        for publication in published:
            index = self.published[pair.peer]
            self.published[pair.peer] = index + 1
            delay_ns = data.draw(DELAYS_NS, label="drive_states delay (ns)")
            if delay_ns is None:
                continue
            arrival_ns = pair.states.arrival_ns(publication.published_ns, delay_ns)
            sent = StatesSent(publication, index)
            self.on_the_way += (OnTheWay(arrival_ns, pair.peer, sent),)

    def arriving(self, kind: type, until_ns: int) -> list[OnTheWay]:
        """The messages of `kind` due by `until_ns`, in order of arrival,
        taken off their way."""

        def due(message: OnTheWay) -> bool:
            return isinstance(message.message, kind) and message.arrival_ns <= until_ns

        arriving = [message for message in self.on_the_way if due(message)]
        self.on_the_way = tuple(m for m in self.on_the_way if not due(m))
        return sorted(arriving, key=lambda message: message.arrival_ns)

    async def move_clocks(self, data: st.DataObject, dt_ns: int) -> None:
        """Both clocks move on by `dt_ns`: each follower whose clock reads
        takes the setpoints due at their arrivals and publishes its
        drive_states, one whose clock gave no reading drops them, each
        drive_states due reaches the headset at its arrival, and the host's
        sleepers due wake."""
        start_ns, end_ns = self.instance_ns, self.instance_ns + dt_ns
        for message in self.arriving(Twist, end_ns):
            pair = self.base_pairs[message.peer]
            if pair.peer not in self.unread_clocks:
                self.publish(data, pair, pair.follower.advance_to(message.arrival_ns))
            pair.follower.setpoint()
        for pair in self.base_pairs.values():
            if pair.peer not in self.unread_clocks:
                self.publish(data, pair, pair.follower.advance_to(end_ns))
        for message in self.arriving(StatesSent, end_ns):
            published = message.message.published
            self.instance_ns = message.arrival_ns
            self.reports.record(
                message.peer, states(published.follow_state, published.limits)
            )
            self.newest_arrival[message.peer] = Arrival(
                self.elapsed_ns + message.arrival_ns - start_ns,
                message.message.index,
                published.follow_state,
                published.limits,
            )
        self.instance_ns = end_ns
        self.elapsed_ns += dt_ns
        assert self.reports.mark() == self.elapsed_ns, "the elapsed clock drifted"
        await self.clock.advance(dt_ns / NS_PER_S)

    def send_setpoint(self, data: st.DataObject, peer: Peer, twist: Twist) -> None:
        """A setpoint the base tick sent, on its way to the pair's follower
        after a drawn delay, or dropped."""
        delay_ns = data.draw(DELAYS_NS, label="setpoint delay (ns)")
        if delay_ns is None:
            return
        arrival_ns = self.base_pairs[peer].setpoints.arrival_ns(
            self.instance_ns, delay_ns
        )
        self.on_the_way += (OnTheWay(arrival_ns, peer, twist),)

    # What the checks read.

    @property
    def down(self) -> bool:
        """Whether no squeeze counts: the goals not running, or the stack
        halted by the listing the headset got last."""
        view = self.listed_view
        halted = view is not None and view.halted_by is not None
        return not self.goals_run or halted

    def readings(self) -> ByHand[Reading]:
        """Each hand as a tick reads the frame the rules set."""
        stale = self.session.age_s > SETTINGS.stale_timeout_s
        return by_hand(lambda hand: reading_of_grip(self.hand_grips[hand], stale))

    def report_of(self, pin: Hashable | None) -> Arrival | None:
        """`pin`'s newest drive_states by arrival, while it counts: within
        drive_link's half second of its arrival."""
        arrival = self.newest_arrival.get(pin)
        if arrival is None or self.elapsed_ns - arrival.elapsed_ns > HALF_SECOND_NS:
            return None
        return arrival

    def newest_stop(self, robot: str) -> StopNoted | None:
        """`robot`'s stop that was answered or refused last."""
        return next(
            (
                self.stop_notes[index]
                for index in reversed(self.stops_ended)
                if self.stop_notes[index].robot == robot
            ),
            None,
        )

    def follows(self, robot: str, report: Arrival | None) -> bool:
        """Whether the base of `report` follows now: the report gives
        following, no stop of `robot` is in flight, and the report arrived
        more than drive_link's half second after the newest stop of `robot`
        ended."""
        if report is None or report.follow_state != FOLLOWING:
            return False
        notes = (note for note in self.stop_notes if note.robot == robot)
        if any(note.ended_ns is None for note in notes):
            return False
        newest = self.newest_stop(robot)
        return (
            newest is not None and report.elapsed_ns - newest.ended_ns > HALF_SECOND_NS
        )

    def slot_pins(self, chosen: str | None) -> tuple[Pins, Pins]:
        """Every pair of the base slot, and the chosen robot's first pair on
        it, as pins."""
        peers = frozenset(Pin(Slot.BASE, peer) for peer in self.slot_peers())
        first = self.pairs_of(chosen)[:1] if chosen is not None else ()
        return peers, frozenset(Pin(Slot.BASE, peer) for peer in first)

    # The tick and its checks.

    def tick_base(self, data: st.DataObject) -> None:
        """One base tick: the deadman and the stream's model stepped, its
        send checked, and its setpoint on its way."""
        before = self.driver.world.grips.base.phase
        frame = fresh_frame(self.session, SETTINGS.stale_timeout_s, self.clock)
        sent = setpoint_sent(self.base_tick())
        _names, chosen = self.robots.standing()
        world = self.driver.world
        after = world.grips.base.phase
        readings = self.readings()
        peers, pins = self.slot_pins(chosen)
        self.deadman = self.deadman.ticked(
            readings, chosen, before, peers, pins, down=self.down
        )
        if isinstance(after, Engaged) and not isinstance(before, Engaged):
            self.deadman = self.deadman.engages_at_tick(after, readings)
            self.stream_model = self.stream_model.engaged(after.pins)
        engaged = self.deadman.engaged
        pin = (
            engaged.pin
            if engaged is not None
            else pin_on(driven_pins(after), Slot.BASE)
        )
        record = TickRecord(
            host_s=self.clock.monotonic(),
            frame_twist=commanded(frame, self.report_of(pin)),
            phase_after=after,
            send=None if sent is None else (sent[0], Twist(*sent[1])),
        )
        self.check_tick(record, world)
        self.base_phase = after
        if record.send is not None:
            self.send_setpoint(data, *record.send)

    def check_tick(self, record: TickRecord, world: World) -> None:
        """The leader rule at the tick of `record`, after its steps of
        `world`."""
        send = record.send
        if send is not None and not send[1].is_zero:
            self.check_stream_setpoint(record, world)
        engaged = self.deadman.engaged
        twist = record.frame_twist
        owed = (
            engaged is not None
            and twist is not None
            and not twist.is_zero
            and self.follows(engaged.robot, self.report_of(engaged.pin))
        )
        if owed:
            event("the base streams the sticks")
        assert not owed or send == (engaged.pin, twist), (
            f"the liveness: {send!r}, not {twist!r} to {engaged.pin!r}"
        )
        target = None if send is None or send[1].is_zero else send[1]
        ticked = self.stream_model.stepped(
            record.phase_after,
            target=target,
            held=STILL,
            peers=self.slot_peers(),
            now_s=record.host_s,
        )
        assert not ticked.broken, "; ".join(ticked.broken)
        assert send == ticked.send, f"the burst: {send!r}, by the rule {ticked.send!r}"
        self.stream_model = ticked.model
        if ticked.burst_sent:
            event("the base sends a burst value")
        self.check_lease(record.phase_after, world, ticked.burst_sent)

    def check_lease(self, phase: Phase, world: World, burst_sent: bool) -> None:
        """The lease at a tick that left the base in `phase`: the base's
        reason stays in a hold only while the base has a phase on its robot,
        and in the hold of a lingering base's epoch while that hold stands;
        and the base lingers only through its burst."""
        for robot in world.robots:
            if robot.hold is not None and robot.hold.has(BASE):
                assert robot_of_phase(phase) == robot.name, (
                    f"the lease: the base's reason outlives its grip on {robot.name}"
                )
        match phase:
            case Lingering(robot=r, epoch=e) if world.robot(r).epoch == e:
                hold = world.robot(r).hold
                assert hold is not None and hold.has(BASE), (
                    f"the lease: the base lingers on {r} at epoch {e} with {hold}"
                )
        assert burst_sent or not isinstance(phase, Lingering), (
            "the lease: the base lingers past its burst"
        )

    def check_stream_setpoint(self, record: TickRecord, world: World) -> None:
        """The deadman, follows now, the own stops, the value and the lease
        at a tick that sent a stream setpoint."""
        pin, twist = record.send
        engaged = self.deadman.engaged
        assert engaged is not None and engaged.pin == pin, (
            f"the deadman: {twist!r} to {pin!r} outside a squeeze of its robot"
        )
        report = self.report_of(pin)
        assert self.follows(engaged.robot, report), (
            f"follows now: {twist!r} to {pin!r}, whose base does not follow now"
        )
        newest = self.newest_stop(engaged.robot)
        assert report.index >= newest.published.get(pin, 0), (
            f"the own stops: {twist!r} on a report from before the stop {newest!r}"
        )
        assert twist == record.frame_twist, (
            f"the value: {twist!r}, not the sticks' {record.frame_twist!r}"
        )
        hold = world.robot(engaged.robot).hold
        assert hold is not None and hold.has(BASE), (
            f"the lease: {twist!r} with no hold of {engaged.robot} for the base"
        )

    def observe(self) -> None:
        """The stops the rule sent and ended, then the deadman after the
        rule's steps: each let-go the driver noted, a base request refused or
        withdrawn by a let-go, by its phase turning idle outside its own tick,
        and a base engaged at the answer of its stop."""
        answered = None
        for robot, _stopped in self.stops.calls[len(self.stop_notes) :]:
            published = MappingProxyType(dict(self.published))
            self.stop_notes.append(StopNoted(robot, self.elapsed_ns, published))
        for index, note in enumerate(self.stop_notes):
            stopped = self.stops.calls[index][1]
            if note.ended_ns is not None or not stopped.done():
                continue
            self.stop_notes[index] = replace(note, ended_ns=self.elapsed_ns)
            self.stops_ended.append(index)
            if stopped.cancelled() or stopped.exception() is not None:
                continue
            if stopped.result() is None:
                answered = note.robot
        for robot in self.driver.let_gos[self.let_gos_read :]:
            self.deadman = self.deadman.let_go_of(robot)
        self.let_gos_read = len(self.driver.let_gos)
        phase = self.driver.world.grips.base.phase
        if isinstance(self.base_phase, Requesting) and isinstance(phase, Idle):
            self.deadman = self.deadman.refused()
        if isinstance(phase, Engaged) and not isinstance(self.base_phase, Engaged):
            self.deadman = (
                self.deadman.engages_by_stop(phase)
                if answered == phase.robot
                else self.deadman.refuses_engagement()
            )
            self.stream_model = self.stream_model.engaged(phase.pins)
        self.base_phase = phase

    # The frame.

    def show(self) -> None:
        """The session's frame: each tracked controller's grip and stick."""
        forward, left, turn = self.deflections
        axes = {Hand.LEFT: stick_axes(forward, left), Hand.RIGHT: stick_axes(left=turn)}
        self.session.hands = parse_hands(
            [
                controller(hand, squeezing=grip is HandGrip.SQUEEZE, axes=axes[hand])
                for hand, grip in self.hand_grips.items()
                if grip is not HandGrip.UNTRACKED
            ]
        )

    @rule(left=st.sampled_from(HandGrip), right=st.sampled_from(HandGrip))
    def grips(self, left: HandGrip, right: HandGrip) -> None:
        """Each controller's grip."""
        self.hand_grips = {Hand.LEFT: left, Hand.RIGHT: right}
        self.show()

    @rule(forward=STICKS, left=STICKS, turn=STICKS)
    def sticks(self, forward: float, left: float, turn: float) -> None:
        """The left stick forward and leftward, and the right stick
        leftward, the turn."""
        self.deflections = (forward, left, turn)
        self.show()

    @rule(age_s=st.sampled_from((0.0, SETTINGS.stale_timeout_s + STALE_BY_S)))
    def stale(self, age_s: float) -> None:
        """The frames come fresh, or past the stale bound."""
        self.session.age_s = age_s

    # The clocks and the ticks.

    @rule(data=st.data(), dt_ns=st.integers(1, LONGEST_ADVANCE_NS))
    def advance(self, data: st.DataObject, dt_ns: int) -> None:
        """Both clocks move on by `dt_ns`, then the base ticks."""

        async def body() -> None:
            await self.move_clocks(data, dt_ns)
            self.tick_base(data)

        self.run(body)

    @rule(data=st.data(), dt_ns=st.integers(SHORTEST_LEAP_NS, LONGEST_LEAP_NS))
    def leap(self, data: st.DataObject, dt_ns: int) -> None:
        """Both clocks leap on by `dt_ns`, with no tick."""
        self.run(lambda: self.move_clocks(data, dt_ns))

    @rule(back_ns=st.integers(1, LONGEST_LEAP_NS))
    def clock_back(self, back_ns: int) -> None:
        """The instance's clock alone steps back by `back_ns`, the clock of
        each follower that reads it with it, and every message on its way is
        dropped."""
        self.instance_ns -= back_ns
        for pair in self.base_pairs.values():
            if pair.peer not in self.unread_clocks:
                pair.follower.clock_went_back(self.instance_ns)
            pair.states.clock_went_back()
            pair.setpoints.clock_went_back()
        self.on_the_way = ()
        assert self.reports.mark() == self.elapsed_ns, "a step back took time"

    @rule(data=st.data())
    def tick(self, data: st.DataObject) -> None:
        """One base tick, with no time passing."""
        self.act(lambda: self.tick_base(data))

    # The followers.

    @precondition(lambda self: self.unread_clocks)
    @rule(data=st.data())
    def clock_reads(self, data: st.DataObject) -> None:
        """The clock of a follower that gave no reading yet reads now."""
        unread = sorted(self.unread_clocks, key=lambda peer: peer.name)
        peer = data.draw(st.sampled_from(unread), label="pair")
        self.unread_clocks = self.unread_clocks - {peer}
        pair = self.base_pairs[peer]
        self.publish(data, pair, pair.follower.advance_to(self.instance_ns))

    @precondition(lambda self: self.untaken_stops())
    @rule(data=st.data())
    def follower_takes_stop(self, data: st.DataObject) -> None:
        """The follower of a robot's first pair takes a stop the driver
        sent, waiting or answered."""
        index = data.draw(st.sampled_from(self.untaken_stops()), label="stop")
        self.taken = self.taken | {index}
        self.followers_of(self.stop_notes[index].robot)[0].stop()

    @precondition(lambda self: self.pairs)
    @rule(data=st.data(), on=st.booleans())
    def silence(self, data: st.DataObject, on: bool) -> None:
        """A robot's wheels fall silent, or report again."""
        robot = data.draw(st.sampled_from(self.paired_robots()), label="robot")
        for follower in self.followers_of(robot):
            if on:
                follower.silence_starts()
            else:
                follower.silence_ends()

    @precondition(lambda self: self.pairs)
    @rule(data=st.data(), on=st.booleans())
    def goal(self, data: st.DataObject, on: bool) -> None:
        """A goal of a robot's first pair's node asks to start, or ends."""
        robot = data.draw(st.sampled_from(self.paired_robots()), label="robot")
        follower = self.followers_of(robot)[0]
        if on:
            follower.start_goal()
        else:
            follower.end_goal()

    # The fleet, as `follow_fleet` tells the driver of it.

    def listing(self) -> None:
        """The changes goal's feedback of the daemon's book, once the deadman
        took its listing after the one before."""
        self.deadman = self.deadman.listed(self.daemon.view(ROBOTS), self.listed_view)
        super().listing()

    def change_book(self, *books: Daemon) -> None:
        """The daemon's book becomes each of `books` in turn, each listed
        while the goals run."""
        for book in books:
            self.daemon = book
            if self.goals_run:
                self.act(self.listing)

    def openable(self) -> list[str]:
        """Each robot whose lease no client holds, while the stack is not
        halted."""
        if self.daemon.halted_by is not None:
            return []
        return [robot for robot in ROBOTS if robot not in self.daemon.leases]

    def closable(self) -> list[str]:
        """Each robot whose lease another client holds open."""
        return [
            robot
            for robot in ROBOTS
            if (lease := self.daemon.leases.get(robot)) is not None and lease.open
        ]

    @precondition(lambda self: self.daemon.halted_by is None)
    @rule()
    def halt(self) -> None:
        """The stack halts: it starts with every lease, then ends each."""
        self.change_book(*self.daemon.halt(HALTER))

    @precondition(lambda self: self.daemon.halted_by is not None)
    @rule()
    def clear(self) -> None:
        """The halt clears."""
        self.change_book(self.daemon.cleared())

    @precondition(lambda self: self.openable())
    @rule(data=st.data())
    def open_lease(self, data: st.DataObject) -> None:
        """The director takes the lease of a free robot, then opens it for a
        take-over."""
        robot = data.draw(st.sampled_from(self.openable()), label="robot")
        self.change_book(*self.daemon.director_opens(robot))

    @precondition(lambda self: self.closable())
    @rule(data=st.data())
    def close_lease(self, data: st.DataObject) -> None:
        """Another client closes its open lease of a robot."""
        robot = data.draw(st.sampled_from(self.closable()), label="robot")
        closed = replace(self.daemon.leases[robot], open=False)
        self.change_book(self.daemon.with_lease(robot, closed))

    @precondition(lambda self: self.goals_run)
    @rule(robot=st.sampled_from(ROBOTS))
    def lease_end(self, robot: str) -> None:
        """The lease goal's feedback ends a lease of the headset on `robot`,
        then the listing of the book that dropped it comes; a book that held
        no such lease stays."""
        book = self.daemon.without_lease_of(robot, HEADSET)

        def end() -> None:
            self.driver.own_lease_ended(robot, "")
            self.set_book(book)

        self.act(end)

    @precondition(lambda self: self.goals_run)
    @rule()
    def goals_end(self) -> None:
        """A goal of the fleet ends, and every lease of the headset with
        it."""
        self.daemon = self.daemon.without_leases_of(HEADSET)

        def end() -> None:
            self.driver.goals_ended(CHANGES_ENDED)
            self.driver.goals_status(Starting())

        self.deadman = self.deadman.lets_go()
        self.act(end)
        self.goals_run = False
        self.listed_view = None

    @precondition(lambda self: not self.goals_run)
    @rule()
    def goals_start(self) -> None:
        """The goals start again, and the changes goal's first feedback
        comes."""
        self.generation += 1

        def start() -> None:
            self.driver.goals_started(self.generation)
            self.driver.goals_status(Starting())
            self.listing()

        self.act(start)

    @rule()
    def press(self) -> None:
        """A rising edge of the home posture button, read as the button's
        task reads it."""
        frame = fresh_frame(self.session, SETTINGS.stale_timeout_s, self.clock)
        squeezing = frame is not None and any(
            sample.squeezing for sample in frame.hands.values()
        )
        _names, chosen = self.robots.standing()
        target = None if chosen is None else postures_of(chosen)
        inputs = PressInputs(squeezing, chosen, target)
        self.act(lambda: self.driver.step(Press(POSTURE_LABEL, inputs)))


# The hand whose pose stream the pose traces drive, where a rule sets it
# (m), and how far `advance` moves the host's clock (s): a tick, and past
# the stale bound of a measured pose.
HAND = Hand.RIGHT
HAND_POSITIONS = ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.1, 0.02, 0.0))
POSE_STEPS_S = (TICK_S, 0.1, 0.3)
# An arm's setpoint as the traces compare it: its frame, its position and
# its orientation.
ArmValue = tuple[str, tuple[float, ...], tuple[float, ...]]


class ArmReport(StrEnum):
    """How a robot's arm reports its measured pose, as a rule sets it."""

    FRESH = "fresh"
    STALE = "stale"
    MOVED = "moved"


@dataclass(frozen=True)
class ArmPose:
    """An arm pair's measured pose as a trace reports it: whether it
    reports, and whether it stands moved away from `POSE`."""

    reporting: bool = True
    moved: bool = False

    @property
    def pose(self) -> Pose:
        return ELSEWHERE if self.moved else POSE

    def after(self, report: ArmReport) -> ArmPose:
        """The arm once a rule set `report`."""
        match report:
            case ArmReport.FRESH:
                return replace(self, reporting=True)
            case ArmReport.STALE:
                return replace(self, reporting=False)
            case ArmReport.MOVED:
                return ArmPose(reporting=True, moved=not self.moved)


def arm_value(setpoint: Setpoint) -> ArmValue:
    """`setpoint` as the traces compare it."""
    pose = setpoint.pose
    return (
        setpoint.frame,
        tuple(pose.position.position),
        tuple(pose.orientation.xyzw),
    )


def arm_sent(send: tuple[Hashable, tuple], stamp_s: float) -> tuple[Hashable, ArmValue]:
    """A pose tick's send as its pin and its setpoint; each setpoint carries
    the stamp of its tick."""
    pin, (frame, timestamp, position, orientation) = send
    assert timestamp == stamp_s, f"a setpoint stamped {timestamp}, not {stamp_s}"
    return pin, (frame, tuple(position), tuple(orientation))


@dataclass(frozen=True)
class Anchor:
    """The clutch's relative mapping as the pose traces hold it: the hand's
    position and the measured pose at the setpoint that anchored it, and the
    measured pose read at its last setpoint."""

    hand: tuple[float, ...]
    pose: ArmValue
    last: ArmValue


def mapped(
    anchor: Anchor | None, engaged: bool, hand: Pose | None, held: ArmValue | None
) -> tuple[Anchor | None, ArmValue | None]:
    """The anchor after a tick of the pose stream and the target the tick
    maps, None for none, with `engaged` whether the grip is engaged after
    the tick's step, `hand` the hand's pose and `held` the measured pose of
    the engaged arm. No engaged grip, hand or measured pose drops the
    anchor; a tick with no anchor anchors at the measured pose and targets
    it; a measured pose that changed since the last setpoint drops the
    anchor, since the traces' arms stand in one frame at `POSE` or 0.4 m away
    at `ELSEWHERE`, past the clutch's jump; any other tick targets the anchor
    moved by `motion_scale` times the hand's travel, and the traces' hands
    never turn."""
    if not engaged or hand is None or held is None:
        return None, None
    position = tuple(hand.position.position)
    if anchor is None:
        return Anchor(position, held, held), held
    if held != anchor.last:
        return None, None
    frame, origin, orientation = anchor.pose
    scale = SETTINGS.motion_scale
    moved = tuple(
        start + scale * (now - then)
        for start, now, then in zip(origin, position, anchor.hand, strict=True)
    )
    return replace(anchor, last=held), (frame, moved, orientation)


class PoseTraces(DriverTraces):
    """`pose_tick` of the right hand and the real driver on the arm slot:
    every send keeps the stream's rule on the arm slot with the targets of
    the clutch's relative mapping (`mapped`), the first setpoint of each
    anchor is the measured pose read at its tick, and every burst value is
    the measured pose of the burst's first tick."""

    def __init__(self) -> None:
        super().__init__()
        self.gripper = NoWireSlot()
        self.robots = PairedRobots(
            None,
            arms={HAND: self.slot},
            grippers={HAND: self.gripper},
            base=NoWireSlot(),
        )
        self.measured = MeasuredPoses(self.clock)
        self.clutch = HandClutch(SETTINGS.motion_scale)
        self.driver = self.listed_driver(self.stops)
        self.pose_tick = pose_tick(
            topic_module=self.slot,
            gripper_module=self.gripper,
            handedness=HAND,
            driver=self.driver,
            clutch=self.clutch,
            session=self.session,
            measured=self.measured,
            robots=self.robots,
            settings=SETTINGS,
            clock=self.clock,
            timestamp_s=self.clock.monotonic,
        )
        self.stream_model = StreamModel[ArmValue](Slot.ARM, BURST_S)
        self.arm_poses: dict[Peer, ArmPose] = {}
        self.arm_phase: Phase = Idle()
        # The right hand's pose while it is tracked, and the relative
        # mapping's anchor.
        self.hand_pose: Pose | None = None
        self.anchor: Anchor | None = None
        for robot in ROBOTS:
            self.join(robot, ahead=False, data=None)

    def paired(self, robot: str, peer: Peer, data: st.DataObject | None) -> None:
        """The new pair's arm reports its pose."""
        self.arm_poses[peer] = ArmPose()
        self.report_pose(peer)

    def unpaired(self, peer: Peer) -> None:
        """The pair's arm reports no more."""
        del self.arm_poses[peer]

    def report_pose(self, peer: Peer) -> None:
        """`peer`'s pose_states arrives now, while its arm reports."""
        arm = self.arm_poses[peer]
        if arm.reporting:
            self.measured.set(peer, reported(arm.pose))

    def observe(self) -> None:
        """An engagement the rule's steps made."""
        phase = self.driver.world.grips.of(HAND).phase
        if isinstance(phase, Engaged) and not isinstance(self.arm_phase, Engaged):
            self.stream_model = self.stream_model.engaged(phase.pins)
        self.arm_phase = phase

    def tick_arm(self) -> None:
        """One tick of the pose stream: its send checked against the
        stream's rule with the target of the relative mapping and the
        burst's value."""
        before = self.driver.world.grips.of(HAND).phase
        now_s = self.clock.monotonic()
        send = self.pose_tick()
        after = self.driver.world.grips.of(HAND).phase
        peers = self.slot_peers()
        pin = pin_on(driven_pins(after), Slot.ARM)
        stale_s = SETTINGS.stale_timeout_s
        measured = self.measured.fresh(pin, stale_s, now_s) if pin in peers else None
        held = (
            None
            if measured is None
            else arm_value(Setpoint(measured.frame, measured.value))
        )
        sent = None if send is None else arm_sent(send, now_s)
        if isinstance(after, Engaged) and not isinstance(before, Engaged):
            self.stream_model = self.stream_model.engaged(after.pins)
        engaged = isinstance(after, Engaged)
        self.anchor, target = mapped(self.anchor, engaged, self.hand_pose, held)
        ticked = self.stream_model.stepped(
            after, target=target, held=held, peers=peers, now_s=now_s
        )
        assert not ticked.broken, "; ".join(ticked.broken)
        assert sent == ticked.send, (
            f"the setpoint: {sent!r}, by the rule and the mapping {ticked.send!r}"
        )
        self.stream_model = ticked.model
        self.arm_phase = after

    @rule(grip=st.sampled_from(HandGrip), at=st.sampled_from(HAND_POSITIONS))
    def hand(self, grip: HandGrip, at: tuple[float, float, float]) -> None:
        """The right controller's grip, at `at`."""
        squeezing = grip is HandGrip.SQUEEZE
        sample = HandSample(pose=pose_at(at), squeezing=squeezing, trigger=0.0)
        tracked = grip is not HandGrip.UNTRACKED
        self.session.hands = {HAND: sample} if tracked else {}
        self.hand_pose = sample.pose if tracked else None

    @rule(robot=st.sampled_from(ROBOTS), report=st.sampled_from(ArmReport))
    def arm_reports(self, robot: str, report: ArmReport) -> None:
        """`robot`'s arms report their measured pose where they stand, report
        none, or report it moved."""
        for peer in self.pairs_of(robot):
            self.arm_poses[peer] = self.arm_poses[peer].after(report)
            self.report_pose(peer)

    @rule(dt_s=st.sampled_from(POSE_STEPS_S))
    def advance(self, dt_s: float) -> None:
        """The host's clock moves on by `dt_s`, each arm that reports reports
        its pose, then the stream ticks."""

        async def body() -> None:
            await self.clock.advance(dt_s)
            for peer in self.arm_poses:
                self.report_pose(peer)
            self.tick_arm()

        self.run(body)

    @rule()
    def tick(self) -> None:
        """One tick of the stream, with no time passing."""
        self.act(self.tick_arm)


# The pairs the slot of a stream alone holds, its grip's robot, the values it
# drives and holds, and how far its host's clock moves at a step (s).
STREAM_PEERS = (Peer("alpha 0"), Peer("alpha 1"), Peer("bravo 0"))
STREAM_ROBOT = "alpha"
TARGETS = ("forward", "turn")
HELD_VALUE = "still"
STREAM_STEPS_S = (0.0, 0.01, 0.1, 0.3)


class GripStreamRules(RuleBasedStateMachine):
    """`GripStream[str]` alone, its phase changed only as the world changes
    it: every send keeps the stream's rule, and `burst_running` is true at
    each step whose send is a burst value, the burst's last included."""

    def __init__(self) -> None:
        super().__init__()
        self.stream = GripStream[str](Slot.BASE)
        self.model = StreamModel[str](Slot.BASE, BURST_S)
        self.phase: Phase = Idle()
        self.peers: frozenset[Peer] = frozenset(STREAM_PEERS)
        self.now_s = 0.0
        self.epochs = 0

    def pinned(self) -> bool:
        """Whether the phase's pair is on the slot."""
        return pin_on(driven_pins(self.phase), Slot.BASE) in self.peers

    @precondition(lambda self: isinstance(self.phase, Idle) and self.peers)
    @rule(data=st.data())
    def engage(self, data: st.DataObject) -> None:
        """The idle grip engages on a pair of the slot."""
        peers = sorted(self.peers, key=lambda peer: peer.name)
        peer = data.draw(st.sampled_from(peers), label="pair")
        self.epochs += 1
        pins = frozenset({Pin(Slot.BASE, peer)})
        self.phase = Engaged(STREAM_ROBOT, pins, self.epochs)

    @precondition(lambda self: isinstance(self.phase, Engaged))
    @rule()
    def linger(self) -> None:
        """The engaged grip lets go, on the same pins."""
        phase = self.phase
        self.phase = Lingering(phase.robot, phase.pins, phase.epoch)

    @precondition(lambda self: isinstance(self.phase, Lingering) and self.pinned())
    @rule()
    def rejoin(self) -> None:
        """The lingering grip engages again, on the same pins."""
        phase = self.phase
        self.phase = Engaged(phase.robot, phase.pins, phase.epoch)

    @rule(peers=st.frozensets(st.sampled_from(STREAM_PEERS)))
    def slot_changes(self, peers: frozenset[Peer]) -> None:
        """The slot holds `peers`."""
        self.peers = peers

    @rule(
        drawn_target=st.one_of(st.none(), st.sampled_from(TARGETS)),
        held=st.one_of(st.none(), st.just(HELD_VALUE)),
        dt_s=st.sampled_from(STREAM_STEPS_S),
    )
    def tick(self, drawn_target: str | None, held: str | None, dt_s: float) -> None:
        """One step of the stream, `dt_s` after the last: an engaged grip
        whose pair left lingers first, a target goes only to an engaged
        grip, and a lingering grip whose step sends no burst value turns
        idle after it."""
        phase = self.phase
        if isinstance(phase, Engaged) and not self.pinned():
            self.phase = Lingering(phase.robot, phase.pins, phase.epoch)
        driving = drawn_target if isinstance(self.phase, Engaged) else None
        self.now_s += dt_s
        step = self.stream.step(
            self.phase, target=driving, held=held, peers=self.peers, now_s=self.now_s
        )
        ticked = self.model.stepped(
            self.phase, target=driving, held=held, peers=self.peers, now_s=self.now_s
        )
        assert not ticked.broken, "; ".join(ticked.broken)
        assert step.send == ticked.send, (
            f"the pins: {step.send!r}, by the rule {ticked.send!r}"
        )
        assert step.burst_running is ticked.burst_sent, (
            f"the burst: burst_running is {step.burst_running} at {step.send!r}"
        )
        self.model = ticked.model
        if isinstance(self.phase, Lingering) and not step.burst_running:
            self.phase = Idle()


def test_every_drawn_trace_of_the_base_stream_keeps_the_leader_rule():
    run_state_machine_as_test(BaseTraces, settings=TRACES)


def test_every_drawn_trace_of_a_hands_pose_stream_keeps_its_pins_and_burst():
    run_state_machine_as_test(PoseTraces, settings=TRACES)


def test_every_drawn_step_of_a_grip_stream_alone_keeps_its_pins_and_burst():
    run_state_machine_as_test(GripStreamRules, settings=TRACES)
