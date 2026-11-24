"""The grip world, one test per rule: each test steps the pure world through
events from a world the events reached, and reads the world and the effects
of the steps."""

import pytest

from tests.helpers import (
    CHANGES_ENDED,
    DIRECTOR,
    DIRECTOR_LEASE,
    FIRST_REVISION,
    HALTED,
    HALTER,
    HEADSET,
    HOME_REACHED,
    OTHER,
    Peer,
    fleet_view,
    postures_of,
)
from xr_commander.base import NO_BASE_PAIR, NOT_REPORTING
from xr_commander.bus import Answered, Failed, Unanswered
from xr_commander.devices import Hand
from xr_commander.fleet import (
    LEASE_NAME,
    DaemonLost,
    Following,
    Lease,
    Starting,
)
from xr_commander.grip_world import (
    BASE,
    POSTURE,
    POSTURE_NOTE,
    ROBOT_NOTE,
    AcceptedGoal,
    Brought,
    Call,
    CallDone,
    CancelGoal,
    Changed,
    CheckMove,
    CheckMoveInstance,
    Confirm,
    Engaged,
    Finished,
    FireGoal,
    GoalsEnded,
    GoalsStarted,
    GoalsStatusChanged,
    Idle,
    Join,
    LeaseEnded,
    Lingering,
    Listing,
    NotReady,
    Note,
    PendingEnd,
    Pin,
    PostureMoving,
    PostureResult,
    PostureStage,
    PostureStart,
    Press,
    PressInputs,
    QueuedConfirm,
    QueuedPosture,
    Raised,
    Ready,
    Reading,
    Release,
    ReleaseLease,
    Requester,
    Requesting,
    ResultLost,
    RetryDue,
    Robot,
    Shutdown,
    Slot,
    StartRetry,
    StopMoves,
    Streamed,
    Take,
    TakeLease,
    Tick,
    TickInputs,
    UnderOpen,
    Watch,
    World,
    driven_pins,
    pin_on,
    reading_of,
    slot_pins,
    step,
)
from xr_commander.publish import NO_MEASURED_POSE, NO_PAIR
from xr_commander.robots import JOIN_COMMAND, SlotPairs


LEFT, RIGHT = Hand.LEFT, Hand.RIGHT
SQUEEZE, RELEASED, UNKNOWN = Reading.SQUEEZE, Reading.RELEASED, Reading.UNKNOWN
ALPHA, BRAVO, CHARLIE, DELTA, ECHO = "alpha", "bravo", "charlie", "delta", "echo"


def own_pin(robot: str, grip) -> Pin:
    """The pin `robot` gives `grip` in a test: its base, or the hand's arm."""
    if grip == BASE:
        return Pin(Slot.BASE, Peer(f"{robot} base"))
    return Pin(Slot.ARM, Peer(f"{robot} {grip} arm"))


ALPHA_ARM = own_pin(ALPHA, LEFT)
ALPHA_GRIPPER = Pin(Slot.GRIPPER, Peer("alpha left gripper"))
ALPHA_BASE = own_pin(ALPHA, BASE)
HOME = postures_of(ALPHA)
BRAVO_HOME = postures_of(BRAVO)
# The handle of an accepted posture goal.
GOAL = "the goal of home"
READY = Ready()
# The leases of a test's listings: another client's lease opened for a
# take-over, and the headset's own.
OPEN = {ALPHA: Lease(DIRECTOR, DIRECTOR_LEASE, open=True)}
MINE = {ALPHA: Lease(HEADSET, LEASE_NAME, open=False)}
OPEN_CLOSED = "the open lease of director_inst closed"
RAISED = "RuntimeError('x')"
# A goal of the fleet ends, then its status write.
GOALS_END = (GoalsEnded(CHANGES_ENDED), GoalsStatusChanged(Starting()))


def listing_of(*robots, leases=None, halted_by=None, revision=2) -> Listing:
    return Listing(
        fleet_view(robots, leases=leases, halted_by=halted_by, revision=revision)
    )


def halt(*robots, revision=2) -> Listing:
    """A listing that shows the stack halted by `HALTER`."""
    return listing_of(*robots, halted_by=HALTER, revision=revision)


def listed_world(*robots, leases=None, halted_by=None) -> World:
    """The world once the goals run and a listing of `robots` came:
    `GoalsStarted(1)`, `GoalsStatusChanged(Following())`, `Listing(...)`."""
    world, _ = run(
        World.start(HEADSET),
        GoalsStarted(1),
        GoalsStatusChanged(Following()),
        listing_of(*robots, leases=leases, halted_by=halted_by, revision=1),
    )
    return world


def inputs(
    grip,
    *,
    left=None,
    right=None,
    chosen=ALPHA,
    pins=None,
    others=frozenset(),
    ready=None,
    phase_ready=READY,
) -> TickInputs:
    """A tick of `grip`: an arm grip's own hand reads `left` or `right`,
    `SQUEEZE` when not given; the base's hands read what is given, `UNKNOWN`
    otherwise. `pins` defaults to the chosen robot's pin for the grip, and
    `ready` to ready while the tick has a pin; every pair is
    `pins | others`."""
    if grip == BASE:
        readings = {
            LEFT: left if left is not None else UNKNOWN,
            RIGHT: right if right is not None else UNKNOWN,
        }
    else:
        own = left if grip == LEFT else right
        readings = {grip: own if own is not None else SQUEEZE}
    if pins is None:
        pins = frozenset() if chosen is None else frozenset({own_pin(chosen, grip)})
    if ready is None:
        ready = READY if pins else NotReady(NO_BASE_PAIR if grip == BASE else NO_PAIR)
    return TickInputs(readings, chosen, pins | others, pins, ready, phase_ready)


def run(world: World, *events):
    """Each event in turn; the last world and every effect in order."""
    effects = ()
    for event in events:
        world, stepped = step(world, event)
        effects += stepped
    return world, effects


def tick(world: World, grip, **given):
    """One tick of `grip` with `inputs(grip, **given)`."""
    return step(world, Tick(grip, inputs(grip, **given)))


def calls(effects) -> list[tuple[str, type]]:
    """(robot, request class) of each `Call`."""
    return [(e.robot, type(e.request)) for e in effects if isinstance(e, Call)]


def requests(effects) -> list:
    """The request of each `Call`."""
    return [e.request for e in effects if isinstance(e, Call)]


def answer(world: World, robot: str, outcome=None):
    """`CallDone` of `robot`'s op's call: `Answered(None)` unless given, and
    `Answered(FIRST_REVISION)` for a `Confirm`."""
    op = world.robot(robot).op
    if outcome is None:
        outcome = Answered(FIRST_REVISION if isinstance(op, Confirm) else None)
    return step(world, CallDone(robot, op.call, outcome))


def answer_lost_stop(world: World, robot: str, outcome=Answered(None)):
    """`CallDone` of `robot`'s `lost_stop` call."""
    return step(world, CallDone(robot, world.robot(robot).lost_stop, outcome))


def notes(effects) -> list[tuple]:
    return [(e.target, e.text) for e in effects if isinstance(e, Note)]


def phase(world: World, grip=LEFT):
    return world.grips.of(grip).phase


def must_release(world: World, hand=LEFT) -> bool:
    return world.hands.of(hand).must_release


def taken(world: World, grip=LEFT, robot=ALPHA, **given) -> World:
    """`world` once `grip` squeezes on `robot` and its take, or move check,
    and its stop are answered."""
    world, _ = tick(world, grip, chosen=robot, **given)
    world, _ = answer(world, robot)
    world, _ = answer(world, robot)
    return world


def released(world: World) -> World:
    """`world` once the left hand reads released and its burst ended."""
    world, _ = run(
        world, Tick(LEFT, inputs(LEFT, left=RELEASED)), Streamed(LEFT, False)
    )
    return world


def pressed(world: World, *, chosen=ALPHA, target=HOME, squeezing=False):
    """A press of home's button."""
    return step(world, Press("home", PressInputs(squeezing, chosen, target)))


def posture_held(world: World, robot=ALPHA, target=HOME) -> World:
    """`world` once a press of home on `robot` took it, checked the instance
    and its goal was accepted."""
    world, _ = pressed(world, chosen=robot, target=target)
    for _ in range(3):
        world, _ = answer(world, robot)
    world, _ = answer(world, robot, Answered(GOAL))
    return world


def release_running() -> World:
    """Alpha's taken hold ended at the left burst's end: its release runs."""
    return released(taken(listed_world(ALPHA)))


def lease_ended(world: World):
    return step(world, LeaseEnded(ALPHA, ""))


def timer_of(effects) -> int:
    return next(e.timer for e in effects if isinstance(e, StartRetry))


def test_a_reading_comes_from_a_fresh_samples_grip():
    assert reading_of(None) is UNKNOWN
    assert reading_of(True) is SQUEEZE
    assert reading_of(False) is RELEASED


def test_slot_pins_give_every_pair_and_the_chosen_robots_first_pair_per_slot():
    first, second, other = Peer("alpha 1"), Peer("alpha 2"), Peer("bravo")
    gripper = Peer("bravo gripper")
    reads = {
        Slot.ARM: SlotPairs(((ALPHA, first), (BRAVO, other), (ALPHA, second))),
        Slot.GRIPPER: SlotPairs(((BRAVO, gripper),)),
    }
    every = frozenset(
        {
            Pin(Slot.ARM, first),
            Pin(Slot.ARM, other),
            Pin(Slot.ARM, second),
            Pin(Slot.GRIPPER, gripper),
        }
    )
    assert slot_pins(reads, ALPHA) == (every, frozenset({Pin(Slot.ARM, first)}))
    bravo = frozenset({Pin(Slot.ARM, other), Pin(Slot.GRIPPER, gripper)})
    assert slot_pins(reads, BRAVO) == (every, bravo)
    assert slot_pins(reads, None) == (every, frozenset())


def test_a_pin_on_a_slot_is_its_pair_and_none_where_it_pins_nothing():
    pins = frozenset({ALPHA_ARM, ALPHA_GRIPPER})
    assert pin_on(pins, Slot.ARM) == ALPHA_ARM.peer
    assert pin_on(pins, Slot.GRIPPER) == ALPHA_GRIPPER.peer
    assert pin_on(pins, Slot.BASE) is None


def test_a_stream_drives_the_pins_of_an_engaged_or_lingering_phase_only():
    pins = frozenset({ALPHA_ARM})
    assert driven_pins(Engaged(ALPHA, pins, 1)) == pins
    assert driven_pins(Lingering(ALPHA, pins, 1)) == pins
    assert driven_pins(Requesting(ALPHA, 0, pins)) == frozenset()
    assert driven_pins(Idle()) == frozenset()


def test_a_fresh_reading_is_kept_and_an_unknown_one_is_not():
    world = listed_world(ALPHA)
    world, _ = tick(world, LEFT, left=SQUEEZE, chosen=None)
    assert world.hands.left.last_fresh is SQUEEZE
    world, _ = tick(world, LEFT, left=UNKNOWN, chosen=None)
    assert world.hands.left.last_fresh is SQUEEZE
    world, _ = tick(world, LEFT, left=RELEASED, chosen=None)
    assert world.hands.left.last_fresh is RELEASED


def test_a_released_reading_clears_the_hands_must_release():
    world, _ = tick(listed_world(ALPHA, BRAVO), LEFT)
    world, _ = tick(world, LEFT, chosen=BRAVO)
    assert must_release(world)
    world, _ = tick(world, LEFT, left=RELEASED, chosen=BRAVO)
    assert not must_release(world)


def test_each_grip_keeps_its_own_reading_of_a_hand():
    world = listed_world(ALPHA)
    world, by_base = tick(
        world, BASE, left=SQUEEZE, pins=frozenset(), ready=NotReady(NO_BASE_PAIR)
    )
    world, by_arm = tick(world, LEFT, ready=NotReady(NO_MEASURED_POSE))
    assert notes(by_base) == [(BASE, "squeeze refused: no base pair")]
    assert notes(by_arm) == [(LEFT, "squeeze refused: no measured pose")]


def _left_must_release_both_squeeze(world: World) -> World:
    world, _ = tick(world, LEFT, chosen=ALPHA)
    world, _ = tick(world, LEFT, chosen=BRAVO)
    assert must_release(world, LEFT) and not must_release(world, RIGHT)
    return world


def _both_must_release(world: World) -> World:
    world, _ = tick(world, BASE, left=SQUEEZE, right=SQUEEZE, chosen=ALPHA)
    world, _ = tick(world, BASE, left=SQUEEZE, right=SQUEEZE, chosen=BRAVO)
    assert must_release(world, LEFT) and must_release(world, RIGHT)
    return world


@pytest.mark.parametrize(
    ("setup", "left", "right", "requests_bravo"),
    [
        (lambda world: world, SQUEEZE, RELEASED, True),
        (_left_must_release_both_squeeze, SQUEEZE, SQUEEZE, True),
        (_both_must_release, SQUEEZE, SQUEEZE, False),
    ],
    ids=["left squeezes", "right free of must_release", "both must release"],
)
def test_the_base_is_held_by_either_hand_free_of_must_release(
    setup, left, right, requests_bravo
):
    world = setup(listed_world(ALPHA, BRAVO))
    world, effects = tick(world, BASE, left=left, right=right, chosen=BRAVO)
    assert isinstance(phase(world, BASE), Requesting) is requests_bravo
    assert calls(effects) == ([(BRAVO, TakeLease)] if requests_bravo else [])


def test_the_first_robot_chosen_engages_a_squeeze_already_held():
    world, _ = tick(listed_world(ALPHA), LEFT, chosen=None)
    world, effects = tick(world, LEFT, chosen=ALPHA)
    assert not must_release(world)
    assert calls(effects) == [(ALPHA, TakeLease)]
    world, _ = answer(world, ALPHA)
    world, _ = answer(world, ALPHA)
    assert phase(world) == Engaged(ALPHA, frozenset({ALPHA_ARM}), 1)


def test_a_change_of_robot_after_one_was_chosen_makes_both_squeezing_hands_release_first():
    world, _ = run(
        listed_world(ALPHA, BRAVO),
        Tick(LEFT, inputs(LEFT)),
        Tick(RIGHT, inputs(RIGHT)),
        Tick(LEFT, inputs(LEFT, chosen=BRAVO)),
    )
    assert must_release(world, LEFT) and must_release(world, RIGHT)
    world, effects = tick(world, BASE, left=SQUEEZE, chosen=BRAVO)
    assert phase(world, BASE) == Idle()
    assert calls(effects) == []


def test_a_change_of_robot_seen_on_a_stale_tick_after_a_squeeze_makes_the_hand_release_first():
    world, _ = tick(listed_world(ALPHA, BRAVO), LEFT)
    world, _ = tick(world, LEFT, left=UNKNOWN, chosen=BRAVO)
    assert must_release(world)
    world, effects = tick(world, LEFT, chosen=BRAVO)
    assert phase(world) == Idle()
    assert calls(effects) == []


def test_a_change_of_robot_after_a_release_leaves_the_hand_free():
    world, _ = run(
        listed_world(ALPHA, BRAVO),
        Tick(LEFT, inputs(LEFT)),
        Tick(LEFT, inputs(LEFT, left=RELEASED)),
        Tick(LEFT, inputs(LEFT, left=RELEASED, chosen=BRAVO)),
    )
    assert not must_release(world)
    world, effects = tick(world, LEFT, chosen=BRAVO)
    assert calls(effects) == [(BRAVO, TakeLease)]


def test_a_change_to_no_robot_and_back_is_a_change_each_time():
    world, _ = tick(listed_world(ALPHA), LEFT)
    world, _ = tick(world, LEFT, chosen=None)
    assert must_release(world)
    world, _ = run(
        world,
        Tick(LEFT, inputs(LEFT, left=RELEASED, chosen=None)),
        Tick(LEFT, inputs(LEFT, chosen=None)),
    )
    assert not must_release(world)
    world, effects = tick(world, LEFT, chosen=ALPHA)
    assert must_release(world)
    assert calls(effects) == []


def test_a_let_go_of_the_chosen_robot_makes_both_squeezing_hands_release_first():
    world = taken(listed_world(ALPHA, leases=OPEN))
    world, _ = tick(world, RIGHT)
    assert isinstance(phase(world, RIGHT), Engaged)
    world, effects = step(world, listing_of(ALPHA))
    assert notes(effects) == [(ROBOT_NOTE, f"let go of alpha: {OPEN_CLOSED}")]
    assert isinstance(phase(world, LEFT), Lingering)
    assert isinstance(phase(world, RIGHT), Lingering)
    assert must_release(world, LEFT) and must_release(world, RIGHT)


def test_a_let_go_of_a_robot_not_chosen_covers_no_hand():
    world = taken(listed_world(ALPHA, BRAVO, leases=OPEN))
    world, _ = run(
        world,
        Tick(LEFT, inputs(LEFT, left=RELEASED)),
        Tick(LEFT, inputs(LEFT, left=RELEASED, chosen=BRAVO)),
        Tick(RIGHT, inputs(RIGHT, chosen=BRAVO)),
    )
    world, effects = step(world, listing_of(ALPHA, BRAVO))
    assert notes(effects) == [(ROBOT_NOTE, f"let go of alpha: {OPEN_CLOSED}")]
    assert not must_release(world, RIGHT)
    assert isinstance(phase(world, RIGHT), Requesting)


def _refused_left_under_a_closed_open_lease(*queued_ticks: Tick):
    """Left's take under the director's open lease, lost to a listing that
    closes the lease, `queued_ticks` queued behind it, then its answer."""
    world, _ = run(
        listed_world(ALPHA, leases=OPEN),
        Tick(LEFT, inputs(LEFT)),
        listing_of(ALPHA),
        *queued_ticks,
    )
    return answer(world, ALPHA)


def test_a_refusal_makes_the_refused_grips_hands_release_first():
    world, effects = _refused_left_under_a_closed_open_lease(
        Tick(RIGHT, inputs(RIGHT))
    )
    assert notes(effects) == [(LEFT, f"squeeze refused: {OPEN_CLOSED}")]
    assert must_release(world, LEFT) and not must_release(world, RIGHT)
    assert calls(effects) == [(ALPHA, TakeLease)]


def test_a_refusal_withdraws_every_request_its_hands_alone_held():
    world, effects = _refused_left_under_a_closed_open_lease(
        Tick(BASE, inputs(BASE, left=SQUEEZE))
    )
    refused = f"squeeze refused: {OPEN_CLOSED}"
    assert notes(effects) == [(LEFT, refused), (BASE, refused)]
    assert phase(world, BASE) == Idle()
    assert world.robot(ALPHA).queue == ()
    assert calls(effects) == []


def _both_squeezing_on_alpha() -> World:
    world, _ = tick(taken(listed_world(ALPHA)), RIGHT)
    assert isinstance(phase(world, RIGHT), Engaged)
    return world


def test_a_halt_start_makes_both_squeezing_hands_release_first():
    world, _ = step(_both_squeezing_on_alpha(), halt(ALPHA))
    assert must_release(world, LEFT) and must_release(world, RIGHT)


def test_the_end_of_the_goals_makes_both_squeezing_hands_release_first():
    world, _ = run(_both_squeezing_on_alpha(), *GOALS_END)
    assert must_release(world, LEFT) and must_release(world, RIGHT)


def test_a_listing_that_keeps_the_halt_lets_go_of_nothing():
    world = listed_world(ALPHA, halted_by=HALTER)
    epoch = world.robot(ALPHA).epoch
    world, effects = step(world, halt(ALPHA, revision=2))
    assert notes(effects) == []
    assert world.robot(ALPHA).epoch == epoch


def _down_while_halted():
    world, effects = tick(listed_world(ALPHA, halted_by=HALTER), LEFT)
    return world, effects, HALTED, (listing_of(ALPHA),)


def _down_while_halted_and_not_ready():
    world = listed_world(ALPHA, halted_by=HALTER)
    world, effects = tick(world, LEFT, ready=NotReady(NO_MEASURED_POSE))
    return world, effects, HALTED, (listing_of(ALPHA),)


def _down_in_the_burst_a_halt_left():
    world, _ = run(
        taken(listed_world(ALPHA)),
        halt(ALPHA),
        Tick(LEFT, inputs(LEFT, left=RELEASED)),
    )
    assert isinstance(phase(world), Lingering)
    world, effects = tick(world, LEFT)
    world, _ = step(world, Streamed(LEFT, False))
    return world, effects, HALTED, (listing_of(ALPHA, revision=3),)


def _down_at_boot():
    world, effects = tick(World.start(HEADSET), LEFT)
    goals_run = (
        GoalsStarted(1),
        GoalsStatusChanged(Following()),
        listing_of(ALPHA),
    )
    return world, effects, "waiting for the fleet listing", goals_run


def _down_with_the_daemon_lost():
    world, _ = run(
        listed_world(ALPHA),
        GoalsEnded("the daemon is lost"),
        GoalsStatusChanged(DaemonLost()),
    )
    world, effects = tick(world, LEFT)
    goals_run = (
        GoalsStarted(2),
        GoalsStatusChanged(Starting()),
        GoalsStatusChanged(Following()),
        listing_of(ALPHA),
    )
    return world, effects, "the daemon is lost", goals_run


@pytest.mark.parametrize(
    "down",
    [
        _down_while_halted,
        _down_while_halted_and_not_ready,
        _down_in_the_burst_a_halt_left,
        _down_at_boot,
        _down_with_the_daemon_lost,
    ],
    ids=["halted", "halted, not ready", "halted, lingering", "at boot", "daemon lost"],
)
def test_a_squeeze_read_while_halted_or_with_the_goals_down_must_be_released_after(
    down,
):
    world, effects, cause, recovery = down()
    assert notes(effects) == [(LEFT, f"squeeze refused: {cause}")]
    assert must_release(world)
    world, _ = run(world, *recovery)
    world, held = tick(world, LEFT)
    assert phase(world) == Idle()
    assert calls(held) == []
    world, _ = tick(world, LEFT, left=RELEASED)
    world, squeezed = tick(world, LEFT)
    assert calls(squeezed) == [(ALPHA, TakeLease)]


def test_a_squeeze_not_read_during_the_halt_counts_after_the_clear():
    world = listed_world(ALPHA, halted_by=HALTER)
    world, _ = tick(world, LEFT, left=UNKNOWN)
    world, _ = step(world, listing_of(ALPHA))
    world, effects = tick(world, LEFT)
    assert not must_release(world)
    assert calls(effects) == [(ALPHA, TakeLease)]


def test_the_halt_note_comes_at_the_squeeze_only():
    world, first = tick(listed_world(ALPHA, halted_by=HALTER), LEFT)
    world, held = tick(world, LEFT)
    assert notes(first) == [(LEFT, f"squeeze refused: {HALTED}")]
    assert notes(held) == []


def test_the_halt_note_comes_on_each_grip_whose_own_reading_rises():
    world = listed_world(ALPHA, halted_by=HALTER)
    world, by_base = tick(world, BASE, left=SQUEEZE)
    world, by_arm = tick(world, LEFT)
    assert notes(by_base) == [(BASE, f"squeeze refused: {HALTED}")]
    assert notes(by_arm) == [(LEFT, f"squeeze refused: {HALTED}")]


def test_a_pin_that_leaves_an_engaged_grip_interrupts_its_hands():
    world, _ = tick(taken(listed_world(ALPHA)), LEFT, pins=frozenset())
    assert must_release(world)
    assert phase(world) == Lingering(ALPHA, frozenset({ALPHA_ARM}), 1)


def test_a_pin_that_leaves_a_lingering_grip_interrupts_its_hands():
    world, _ = tick(taken(listed_world(ALPHA)), LEFT, left=UNKNOWN)
    assert not must_release(world)
    world, _ = tick(world, LEFT, left=UNKNOWN, pins=frozenset())
    assert must_release(world)
    assert phase(world) == Lingering(ALPHA, frozenset({ALPHA_ARM}), 1)


def test_a_requesting_grip_whose_tick_reads_no_pin_interrupts_its_hands():
    world, _ = tick(listed_world(ALPHA), LEFT)
    world, _ = tick(world, LEFT, pins=frozenset())
    assert must_release(world)
    assert phase(world) == Idle()
    assert world.robot(ALPHA).op.requesters == ()


def _arm_and_base_of_one_squeeze() -> World:
    world, _ = run(
        listed_world(ALPHA),
        Tick(LEFT, inputs(LEFT)),
        Tick(BASE, inputs(BASE, left=SQUEEZE)),
    )
    world, _ = answer(world, ALPHA)
    world, _ = answer(world, ALPHA)
    assert isinstance(phase(world, LEFT), Engaged)
    assert isinstance(phase(world, BASE), Engaged)
    return world


def test_a_base_pair_that_leaves_makes_the_arm_of_the_same_squeeze_linger():
    world, _ = tick(
        _arm_and_base_of_one_squeeze(), BASE, left=SQUEEZE, pins=frozenset()
    )
    assert must_release(world)
    world, _ = tick(world, LEFT)
    assert isinstance(phase(world, LEFT), Lingering)


def test_an_arm_pair_that_leaves_makes_the_base_of_the_same_squeeze_linger():
    world, _ = tick(_arm_and_base_of_one_squeeze(), LEFT, pins=frozenset())
    assert must_release(world)
    world, _ = tick(world, BASE, left=SQUEEZE)
    assert isinstance(phase(world, BASE), Lingering)


def test_a_stale_tick_and_a_released_reading_are_no_interruption():
    world = taken(listed_world(ALPHA))
    engaged = phase(world)
    for reading in (UNKNOWN, RELEASED):
        world, _ = tick(world, LEFT, left=reading)
        assert isinstance(phase(world), Lingering)
        assert not must_release(world)
        world, effects = tick(world, LEFT)
        assert phase(world) == engaged
        assert calls(effects) == []


def test_a_released_reading_at_another_grips_tick_withdraws_the_request_that_hand_alone_held():
    world, _ = tick(listed_world(ALPHA), BASE, left=SQUEEZE)
    world, _ = tick(world, LEFT, left=RELEASED)
    assert phase(world, BASE) == Idle()
    assert world.robot(ALPHA).op.requesters == ()
    world, effects = answer(world, ALPHA)
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_a_change_of_robot_seen_by_another_grip_withdraws_the_request():
    world, _ = tick(listed_world(ALPHA, BRAVO), LEFT)
    world, effects = tick(world, BASE, chosen=BRAVO)
    assert phase(world, LEFT) == Idle()
    assert notes(effects) == []


def test_a_request_held_by_the_other_hand_stays():
    world, _ = tick(listed_world(ALPHA), BASE, left=SQUEEZE, right=SQUEEZE)
    world, _ = tick(world, LEFT, left=RELEASED)
    assert isinstance(phase(world, BASE), Requesting)


def test_an_engaged_grip_held_on_its_robot_with_its_pins_stays_engaged():
    world = taken(listed_world(ALPHA))
    engaged = phase(world)
    world, _ = tick(world, LEFT)
    assert phase(world) == engaged


def test_an_engaged_grip_keeps_its_pins_when_another_pair_comes_first():
    world = taken(listed_world(ALPHA))
    ahead = Pin(Slot.ARM, Peer("alpha left arm, second"))
    world, _ = tick(world, LEFT, pins=frozenset({ahead}), others=frozenset({ALPHA_ARM}))
    assert phase(world) == Engaged(ALPHA, frozenset({ALPHA_ARM}), 1)


def _released(world: World) -> World:
    return tick(world, LEFT, left=RELEASED)[0]


def _unknown(world: World) -> World:
    return tick(world, LEFT, left=UNKNOWN)[0]


def _interrupted_by_the_base(world: World) -> World:
    world, _ = run(
        world,
        Tick(BASE, inputs(BASE, left=SQUEEZE)),
        Tick(BASE, inputs(BASE, left=SQUEEZE, pins=frozenset())),
    )
    assert must_release(world)
    return tick(world, LEFT)[0]


def _another_robot(world: World) -> World:
    return tick(world, LEFT, chosen=BRAVO)[0]


def _pin_gone(world: World) -> World:
    return tick(world, LEFT, pins=frozenset())[0]


@pytest.mark.parametrize(
    "after",
    [_released, _unknown, _interrupted_by_the_base, _another_robot, _pin_gone],
    ids=["released", "unknown", "must_release", "another robot", "a pin gone"],
)
def test_an_engaged_grip_lingers_otherwise(after):
    world = after(taken(listed_world(ALPHA, BRAVO)))
    assert phase(world) == Lingering(ALPHA, frozenset({ALPHA_ARM}), 1)


def test_a_requesting_grip_held_on_its_robot_takes_the_new_pins():
    world, _ = tick(listed_world(ALPHA), LEFT)
    both = frozenset({ALPHA_ARM, ALPHA_GRIPPER})
    world, _ = tick(world, LEFT, pins=both)
    assert phase(world) == Requesting(ALPHA, phase(world).request, both)


@pytest.mark.parametrize(
    "given",
    [
        {"left": RELEASED},
        {"left": UNKNOWN},
        {"chosen": BRAVO},
        {"pins": frozenset()},
    ],
    ids=["released", "unknown", "another robot", "no pin"],
)
def test_a_requesting_grip_withdraws_otherwise(given):
    world, _ = tick(listed_world(ALPHA, BRAVO), LEFT)
    world, _ = tick(world, LEFT, **given)
    assert phase(world) == Idle()
    assert world.robot(ALPHA).op.requesters == ()


@pytest.mark.parametrize(
    "given",
    [
        {},
        {
            "pins": frozenset({Pin(Slot.ARM, Peer("alpha left arm, second"))}),
            "others": frozenset({ALPHA_ARM}),
            "ready": NotReady(NO_MEASURED_POSE),
        },
    ],
    ids=["its pins", "a new arm pair ahead with no measured pose"],
)
def test_a_lingering_grip_rejoins_under_the_headsets_own_lease(given):
    world = taken(listed_world(ALPHA))
    engaged = phase(world)
    world, _ = tick(world, LEFT, left=UNKNOWN)
    world, effects = tick(world, LEFT, **given)
    assert phase(world) == engaged
    assert calls(effects) == []


def test_a_lingering_grip_rejoins_beside_another_engaged_grip_under_an_open_lease():
    world, _ = tick(taken(listed_world(ALPHA, leases=OPEN)), RIGHT)
    engaged = phase(world, RIGHT)
    world, _ = tick(world, RIGHT, right=UNKNOWN)
    world, effects = tick(world, RIGHT)
    assert phase(world, RIGHT) == engaged
    assert calls(effects) == []


def _lingering(world: World) -> World:
    return tick(world, LEFT, left=UNKNOWN)[0]


def _under_an_open_lease_alone():
    return _lingering(taken(listed_world(ALPHA, leases=OPEN))), {}


def _lingering_with_must_release():
    world = _lingering(_arm_and_base_of_one_squeeze())
    world, _ = tick(world, BASE, left=SQUEEZE, pins=frozenset())
    assert must_release(world)
    return world, {}


def _beside_a_posture_move_alone():
    world, _ = tick(posture_held(listed_world(ALPHA)), LEFT)
    world, _ = answer(world, ALPHA)
    return _lingering(world), {}


def _while_a_confirm_runs():
    world, _ = lease_ended(_lingering(taken(listed_world(ALPHA))))
    return world, {}


def _with_another_robot_chosen():
    world = _lingering(taken(listed_world(ALPHA, BRAVO)))
    world, _ = tick(world, LEFT, left=RELEASED, chosen=BRAVO)
    return world, {"chosen": BRAVO, "others": frozenset({ALPHA_ARM})}


def _not_ready():
    world = _lingering(taken(listed_world(ALPHA)))
    not_ready = NotReady(NO_MEASURED_POSE)
    return world, {"ready": not_ready, "phase_ready": not_ready}


def _not_ready_on_its_own_pins():
    world = _lingering(taken(listed_world(ALPHA)))
    return world, {"phase_ready": NotReady(NO_MEASURED_POSE)}


def _beside_the_hold_of_another_epoch():
    world, _ = lease_ended(_lingering(taken(listed_world(ALPHA))))
    world, _ = answer(world, ALPHA, Failed("not the headset's"))
    assert world.robot(ALPHA).hold is None
    world = taken(world, BASE, right=SQUEEZE)
    assert world.robot(ALPHA).hold.has(BASE)
    world, _ = tick(world, LEFT, left=RELEASED)
    return world, {}


@pytest.mark.parametrize(
    "lingering",
    [
        _under_an_open_lease_alone,
        _lingering_with_must_release,
        _beside_a_posture_move_alone,
        _while_a_confirm_runs,
        _with_another_robot_chosen,
        _not_ready,
        _beside_the_hold_of_another_epoch,
        _not_ready_on_its_own_pins,
    ],
    ids=[
        "under an open lease alone",
        "must_release",
        "a posture in the hold and no grip engaged",
        "a Confirm running",
        "another robot chosen",
        "not ready",
        "the hold of another epoch",
        "not ready on the phase's pins while the tick's pins are ready",
    ],
)
def test_a_lingering_grip_stays_lingering_otherwise(lingering):
    world, given = lingering()
    before = phase(world)
    assert isinstance(before, Lingering)
    world, effects = tick(world, LEFT, **given)
    assert phase(world) == before
    assert calls(effects) == []


def test_an_idle_grip_held_and_ready_requests_the_chosen_robot():
    world, effects = tick(listed_world(ALPHA), LEFT)
    requesting = phase(world)
    assert requesting == Requesting(ALPHA, requesting.request, frozenset({ALPHA_ARM}))
    assert world.robot(ALPHA).op.requesters == (Requester(LEFT, requesting.request),)
    assert calls(effects) == [(ALPHA, TakeLease)]


def test_an_idle_grip_with_must_release_requests_nothing():
    world, _ = tick(listed_world(ALPHA, BRAVO), LEFT)
    world, _ = tick(world, LEFT, chosen=BRAVO)
    world, effects = tick(world, LEFT, chosen=BRAVO)
    assert phase(world) == Idle()
    assert calls(effects) == []


@pytest.mark.parametrize(
    ("grip", "pins", "why"),
    [
        (LEFT, frozenset(), NO_PAIR),
        (LEFT, frozenset({ALPHA_ARM}), NO_MEASURED_POSE),
        (BASE, frozenset(), NO_BASE_PAIR),
        (BASE, frozenset({ALPHA_BASE}), NOT_REPORTING),
    ],
    ids=[NO_PAIR, NO_MEASURED_POSE, NO_BASE_PAIR, NOT_REPORTING],
)
def test_an_idle_grip_not_ready_notes_why_at_its_own_rising_edge_only(grip, pins, why):
    given = {"left": SQUEEZE, "pins": pins, "ready": NotReady(why)}
    world, rising = tick(listed_world(ALPHA), grip, **given)
    world, held = tick(world, grip, **given)
    assert notes(rising) == [(grip, f"squeeze refused: {why}")]
    assert notes(held) == []
    assert calls(rising + held) == []
    assert phase(world, grip) == Idle()


def test_an_idle_grip_released_stays_idle():
    world, effects = tick(listed_world(ALPHA), LEFT, left=RELEASED)
    assert phase(world) == Idle()
    assert effects == ()


def test_a_lingering_grip_drops_once_its_burst_has_ended():
    world, _ = tick(taken(listed_world(ALPHA)), LEFT, left=RELEASED)
    world, running = step(world, Streamed(LEFT, True))
    assert isinstance(phase(world), Lingering)
    world, ended = step(world, Streamed(LEFT, False))
    assert phase(world) == Idle()
    assert world.robot(ALPHA).hold is None
    assert calls(running) == []
    assert calls(ended) == [(ALPHA, ReleaseLease)]


def test_a_lingering_grip_drops_its_own_robot_after_a_change_of_robot():
    world, _ = tick(taken(listed_world(ALPHA, BRAVO)), LEFT, chosen=BRAVO)
    assert isinstance(phase(world), Lingering)
    world, effects = step(world, Streamed(LEFT, False))
    assert world.robot(ALPHA).hold is None
    assert calls(effects) == [(ALPHA, ReleaseLease)]
    assert world.robot(BRAVO) == Robot(BRAVO)


def test_a_lingering_grip_whose_burst_runs_stays():
    world, _ = tick(taken(listed_world(ALPHA)), LEFT, left=RELEASED)
    lingering = phase(world)
    world, effects = step(world, Streamed(LEFT, True))
    assert phase(world) == lingering
    assert effects == ()


@pytest.mark.parametrize(
    "reach",
    [
        lambda: listed_world(ALPHA),
        lambda: tick(listed_world(ALPHA), LEFT)[0],
        lambda: taken(listed_world(ALPHA)),
    ],
    ids=["idle", "requesting", "engaged"],
)
def test_streamed_changes_nothing_outside_lingering(reach):
    world = reach()
    after, effects = step(world, Streamed(LEFT, False))
    assert after == world
    assert effects == ()


def test_a_grip_held_at_once_by_its_own_request_engages_in_its_tick():
    world, effects = tick(taken(listed_world(ALPHA)), RIGHT)
    assert phase(world, RIGHT) == Engaged(ALPHA, frozenset({own_pin(ALPHA, RIGHT)}), 1)
    assert calls(effects) == []


def test_a_grip_held_by_a_call_engages_in_that_step_on_its_latest_pins():
    world, _ = tick(listed_world(ALPHA), LEFT)
    world, _ = answer(world, ALPHA)
    both = frozenset({ALPHA_ARM, ALPHA_GRIPPER})
    world, _ = tick(world, LEFT, pins=both)
    world, _ = answer(world, ALPHA)
    assert phase(world) == Engaged(ALPHA, both, 1)


def test_a_refusal_turns_each_requester_idle_with_its_note():
    world, asked = run(
        listed_world(ALPHA),
        Tick(LEFT, inputs(LEFT)),
        Tick(BASE, inputs(BASE, left=SQUEEZE)),
    )
    leased = f"leased by director_inst ({DIRECTOR_LEASE})"
    world, effects = answer(world, ALPHA, Failed(leased))
    refused = f"squeeze refused: {leased}"
    assert calls(asked) == [(ALPHA, TakeLease)]
    assert notes(effects) == [(LEFT, refused), (BASE, refused)]
    assert phase(world, LEFT) == Idle() and phase(world, BASE) == Idle()


def test_a_let_go_lingers_engaged_grips_and_turns_requesting_ones_idle_with_a_note():
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world, _ = tick(world, BASE, right=SQUEEZE)
    assert isinstance(phase(world, BASE), Requesting)
    world, effects = answer(world, ALPHA, Failed("not the headset's"))
    assert notes(effects) == [
        (ROBOT_NOTE, "let go of alpha: the lease ended"),
        (BASE, "squeeze refused: the lease ended"),
    ]
    assert isinstance(phase(world, LEFT), Lingering)
    assert phase(world, BASE) == Idle()
    assert world.robot(ALPHA).queue == ()


def test_a_take_that_raised_owes_a_release():
    world, _ = tick(listed_world(ALPHA), LEFT)
    world, effects = answer(world, ALPHA, Raised(RAISED))
    assert notes(effects) == [(LEFT, f"squeeze refused: {RAISED}")]
    assert world.robot(ALPHA).release_owed
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_an_answered_take_clears_the_release_owed_and_its_retry():
    world, _ = tick(listed_world(ALPHA), LEFT)
    world, _ = answer(world, ALPHA, Unanswered())
    world, unanswered = answer(world, ALPHA, Unanswered())
    timer = timer_of(unanswered)
    world, _ = run(
        world,
        Tick(LEFT, inputs(LEFT, left=RELEASED)),
        Tick(LEFT, inputs(LEFT)),
    )
    world, _ = answer(world, ALPHA)
    assert not world.robot(ALPHA).release_owed
    assert world.robot(ALPHA).retry is None
    world, effects = step(world, RetryDue(ALPHA, timer))
    assert calls(effects) == []


def test_an_unanswered_take_owes_a_release():
    world, _ = tick(listed_world(ALPHA), LEFT)
    world, effects = answer(world, ALPHA, Unanswered())
    assert notes(effects) == [(LEFT, "squeeze refused: the daemon did not answer")]
    assert world.robot(ALPHA).release_owed
    assert calls(effects) == [(ALPHA, ReleaseLease)]


@pytest.mark.parametrize(
    ("since", "released_calls"),
    [
        ((), []),
        (
            (
                GoalsStarted(2),
                GoalsStatusChanged(Following()),
                listing_of(ALPHA),
            ),
            [(ALPHA, ReleaseLease)],
        ),
    ],
    ids=["goals down", "goals running again"],
)
def test_a_take_answered_after_its_goals_ended_owes_a_release_once_the_goals_run_again(
    since, released_calls
):
    world, _ = tick(listed_world(ALPHA), LEFT)
    world, _ = run(world, *GOALS_END, *since)
    world, effects = answer(world, ALPHA)
    assert isinstance(world.robot(ALPHA).op, Release) is bool(released_calls)
    assert calls(effects) == released_calls


def test_nothing_owes_a_release_while_the_goals_do_not_run():
    world, _ = tick(listed_world(ALPHA), LEFT)
    world, _ = run(world, *GOALS_END)
    world, answered = answer(world, ALPHA, Unanswered())
    assert not world.robot(ALPHA).release_owed
    world, again = run(
        world,
        GoalsStarted(2),
        GoalsStatusChanged(Following()),
        listing_of(ALPHA),
    )
    assert calls(answered + again) == []


@pytest.mark.parametrize("answered_first", [0, 1], ids=["ADMIT", "STOP"])
def test_a_take_with_no_requester_left_ends_at_its_answer(answered_first):
    world, _ = tick(listed_world(ALPHA), LEFT)
    for _ in range(answered_first):
        world, _ = answer(world, ALPHA)
    world, _ = tick(world, LEFT, left=RELEASED)
    world, effects = answer(world, ALPHA)
    assert world.robot(ALPHA).hold is None
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_a_join_with_no_requester_left_ends_with_no_further_call():
    world = _released(taken(listed_world(ALPHA, leases=OPEN)))
    world, joining = tick(world, RIGHT)
    assert calls(joining) == [(ALPHA, CheckMove)]
    world, _ = tick(world, RIGHT, right=RELEASED)
    world, effects = answer(world, ALPHA)
    assert world.robot(ALPHA).op is None
    assert calls(effects) == []


def _posture_admit_lost_to_a_halt():
    world, _ = run(pressed(listed_world(ALPHA))[0], halt(ALPHA))
    return answer(world, ALPHA)


def _squeeze_on_a_robot_not_listed():
    return tick(listed_world(ALPHA), LEFT, chosen=BRAVO)


def _squeeze_at_boot():
    return tick(World.start(HEADSET), LEFT)


def _posture_admit_across_the_goals_end():
    world, _ = run(pressed(listed_world(ALPHA))[0], *GOALS_END)
    return answer(world, ALPHA)


def _take_under_an_open_lease_that_closes():
    return _refused_left_under_a_closed_open_lease()


def _posture_admit_across_a_halt_and_its_clear():
    world, _ = run(
        pressed(listed_world(ALPHA))[0], halt(ALPHA), listing_of(ALPHA, revision=3)
    )
    return answer(world, ALPHA)


def _take_answered(outcome):
    def reach():
        world, _ = tick(listed_world(ALPHA), LEFT)
        return answer(world, ALPHA, outcome)

    return reach


def _stop_answered(outcome):
    def reach():
        world, _ = tick(listed_world(ALPHA), LEFT)
        world, _ = answer(world, ALPHA)
        return answer(world, ALPHA, outcome)

    return reach


def _at_the_goal(world: World) -> World:
    """`world` once a press of home took alpha and checked the instance:
    its goal is in flight."""
    world, _ = pressed(world)
    for _ in range(3):
        world, _ = answer(world, ALPHA)
    assert world.robot(ALPHA).op.stage is PostureStage.FIRE
    return world


def _goal_accepted_after_a_halt_and_its_clear():
    world, _ = run(
        _at_the_goal(listed_world(ALPHA)), halt(ALPHA), listing_of(ALPHA, revision=3)
    )
    return answer(world, ALPHA, Answered(GOAL))


def _goal_unanswered():
    return answer(_at_the_goal(listed_world(ALPHA)), ALPHA, Unanswered())


def _press_while_halted():
    return pressed(listed_world(ALPHA, halted_by=HALTER))


NOT_LISTED = (
    "bravo is not in the fleet listing, which holds the copies of the stack:"
    f" join a robot as a copy with {JOIN_COMMAND}"
)


@pytest.mark.parametrize(
    ("reach", "note"),
    [
        (
            _posture_admit_lost_to_a_halt,
            (POSTURE_NOTE, f"move to home refused: {HALTED}"),
        ),
        (_squeeze_on_a_robot_not_listed, (LEFT, f"squeeze refused: {NOT_LISTED}")),
        (_squeeze_at_boot, (LEFT, "squeeze refused: waiting for the fleet listing")),
        (
            _posture_admit_across_the_goals_end,
            (POSTURE_NOTE, "move to home refused: a goal of the fleet ended"),
        ),
        (
            _take_under_an_open_lease_that_closes,
            (LEFT, f"squeeze refused: {OPEN_CLOSED}"),
        ),
        (
            _posture_admit_across_a_halt_and_its_clear,
            (POSTURE_NOTE, f"move to home refused: {HALTED}"),
        ),
        (_take_answered(Failed("leased")), (LEFT, "squeeze refused: leased")),
        (
            _take_answered(Unanswered()),
            (LEFT, "squeeze refused: the daemon did not answer"),
        ),
        (_take_answered(Raised(RAISED)), (LEFT, f"squeeze refused: {RAISED}")),
        (
            _stop_answered(Failed("alpha did not answer")),
            (LEFT, "squeeze refused: stop failed: alpha did not answer"),
        ),
        (
            _stop_answered(Raised(RAISED)),
            (LEFT, f"squeeze refused: stop failed: {RAISED}"),
        ),
        (
            _goal_accepted_after_a_halt_and_its_clear,
            (POSTURE_NOTE, "move to home refused: the headset let go of alpha"),
        ),
        (
            _goal_unanswered,
            (
                POSTURE_NOTE,
                "move to home refused: the goal could not be sent (see log)",
            ),
        ),
        (_press_while_halted, (POSTURE_NOTE, f"move to home refused: {HALTED}")),
    ],
    ids=[
        "a posture start lost to a halt",
        "a robot not listed",
        "at boot",
        "a posture start across the goals' end",
        "a take under an open lease that closes",
        "a posture start across a halt and its clear",
        "a refused take",
        "an unanswered take",
        "a take that raised",
        "a refused stop",
        "a stop that raised",
        "a goal accepted after a let-go",
        "an unanswered goal send",
        "a press while halted",
    ],
)
def test_each_cause_names_its_case(reach, note):
    _world, effects = reach()
    assert note in notes(effects)


def test_an_op_both_moved_and_lost_takes_its_lost_cause():
    world, _ = step(_at_the_goal(listed_world(ALPHA)), halt(ALPHA))
    world, effects = answer(world, ALPHA, Answered(GOAL))
    assert notes(effects) == [(POSTURE_NOTE, f"move to home refused: {HALTED}")]
    assert CancelGoal(ALPHA, "home", GOAL) in effects


def test_a_take_asks_the_lease_then_the_stop():
    world, asked = tick(listed_world(ALPHA), LEFT)
    world, stopped = answer(world, ALPHA)
    assert calls(asked) == [(ALPHA, TakeLease)]
    assert calls(stopped) == [(ALPHA, StopMoves)]


def test_a_take_under_another_clients_open_lease_checks_the_move_then_stops():
    world, asked = tick(listed_world(ALPHA, leases=OPEN), LEFT)
    world, stopped = answer(world, ALPHA)
    world, _ = answer(world, ALPHA)
    assert calls(asked) == [(ALPHA, CheckMove)]
    assert calls(stopped) == [(ALPHA, StopMoves)]
    assert world.robot(ALPHA).hold.mode == UnderOpen("director_inst")


def test_a_take_whose_stop_answers_holds_every_requester_at_a_new_epoch():
    world, _ = run(
        listed_world(ALPHA),
        Tick(LEFT, inputs(LEFT)),
        Tick(BASE, inputs(BASE, left=SQUEEZE)),
    )
    left, base = phase(world, LEFT).request, phase(world, BASE).request
    world, _ = answer(world, ALPHA)
    world, _ = answer(world, ALPHA)
    hold = world.robot(ALPHA).hold
    assert hold.reasons == frozenset({Brought(LEFT, left), Brought(BASE, base)})
    assert hold.epoch == world.robot(ALPHA).epoch == 1
    assert phase(world, LEFT) == Engaged(ALPHA, frozenset({ALPHA_ARM}), 1)
    assert phase(world, BASE) == Engaged(ALPHA, frozenset({ALPHA_BASE}), 1)


def test_a_failed_stop_after_a_take_owes_a_release():
    world, effects = _stop_answered(Failed("alpha did not answer"))()
    assert world.robot(ALPHA).hold is None
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_a_lost_take_refuses_its_requesters():
    world, effects = _refused_left_under_a_closed_open_lease()
    assert phase(world) == Idle()
    assert world.robot(ALPHA).op is None
    assert notes(effects) == [(LEFT, f"squeeze refused: {OPEN_CLOSED}")]
    assert calls(effects) == []


def test_a_take_of_a_robot_not_listed_is_refused_at_once():
    world, effects = tick(listed_world(ALPHA), LEFT, chosen=BRAVO)
    assert phase(world) == Idle()
    assert must_release(world)
    assert notes(effects) == [(LEFT, f"squeeze refused: {NOT_LISTED}")]
    assert calls(effects) == []


def test_a_join_under_an_open_lease_with_no_grip_engaged_checks_then_stops():
    world = _released(taken(listed_world(ALPHA, leases=OPEN)))
    world, checked = tick(world, RIGHT)
    assert isinstance(world.robot(ALPHA).op, Join)
    world, stopped = answer(world, ALPHA)
    world, engaged = answer(world, ALPHA)
    assert calls(checked) == [(ALPHA, CheckMove)]
    assert calls(stopped) == [(ALPHA, StopMoves)]
    assert phase(world, RIGHT) == Engaged(
        ALPHA, frozenset({own_pin(ALPHA, RIGHT)}), world.robot(ALPHA).hold.epoch
    )
    assert calls(engaged) == []


def test_a_join_during_a_posture_move_stops_with_no_check():
    world, stopped = tick(posture_held(listed_world(ALPHA)), LEFT)
    assert calls(stopped) == [(ALPHA, StopMoves)]
    world, _ = answer(world, ALPHA)
    assert isinstance(phase(world), Engaged)


def test_a_join_whose_posture_ends_before_its_stop_answers_engages_with_no_check():
    world = posture_held(listed_world(ALPHA, leases=OPEN))
    start = world.posture_moving.start
    world, effects = run(
        world,
        Tick(LEFT, inputs(LEFT)),
        PostureResult(ALPHA, start, Finished(HOME_REACHED)),
    )
    assert not world.robot(ALPHA).hold.has(POSTURE)
    world, engaged = answer(world, ALPHA)
    assert isinstance(phase(world), Engaged)
    assert calls(effects + engaged) == [(ALPHA, StopMoves)]


def test_a_join_whose_stop_answers_holds_every_requester_at_the_holds_epoch():
    world, _ = run(
        posture_held(listed_world(ALPHA)),
        Tick(LEFT, inputs(LEFT)),
        Tick(BASE, inputs(BASE, left=SQUEEZE)),
    )
    epoch = world.robot(ALPHA).hold.epoch
    world, _ = answer(world, ALPHA)
    assert phase(world, LEFT) == Engaged(ALPHA, frozenset({ALPHA_ARM}), epoch)
    assert phase(world, BASE) == Engaged(ALPHA, frozenset({ALPHA_BASE}), epoch)


def test_the_end_of_a_posture_move_a_join_stops_releases_nothing():
    world = posture_held(listed_world(ALPHA))
    start = world.posture_moving.start
    world, effects = run(
        world,
        Tick(LEFT, inputs(LEFT)),
        PostureResult(ALPHA, start, Finished(HOME_REACHED)),
    )
    hold = world.robot(ALPHA).hold
    assert hold.has(LEFT) and not hold.has(POSTURE)
    assert not world.robot(ALPHA).release_owed
    world, engaged = answer(world, ALPHA)
    assert isinstance(phase(world), Engaged)
    assert ReleaseLease not in [kind for _, kind in calls(effects + engaged)]


def test_a_refused_join_refuses_its_requesters_and_drops_their_reasons():
    world, _ = tick(posture_held(listed_world(ALPHA)), LEFT)
    world, effects = answer(world, ALPHA, Failed("alpha did not answer"))
    assert notes(effects) == [
        (LEFT, "squeeze refused: stop failed: alpha did not answer")
    ]
    assert phase(world) == Idle()
    hold = world.robot(ALPHA).hold
    assert hold.has(POSTURE) and not hold.has(LEFT)


def _press_queued_beside_an_engaged_grip():
    """Left's take loses its requester, a press queues behind it, and left
    squeezes again: it joins the take, which engages it."""
    world, _ = run(
        listed_world(ALPHA),
        Tick(LEFT, inputs(LEFT)),
        Tick(LEFT, inputs(LEFT, left=RELEASED)),
    )
    world, _ = pressed(world)
    assert isinstance(world.robot(ALPHA).queue[0], QueuedPosture)
    world, _ = tick(world, LEFT)
    world, _ = answer(world, ALPHA)
    world, effects = answer(world, ALPHA)
    assert isinstance(phase(world), Engaged)
    return world, effects, "release the grip first"


def _press_queued_beside_a_lingering_grip():
    """A lease end, then a press, queued behind left's take while it has no
    requester; left squeezes again and joins the take, the lease end's
    `Confirm` runs while left lingers, and its answer settles with the
    headset named."""
    world, _ = run(
        listed_world(ALPHA),
        Tick(LEFT, inputs(LEFT)),
        Tick(LEFT, inputs(LEFT, left=RELEASED)),
        LeaseEnded(ALPHA, ""),
    )
    world, _ = pressed(world)
    queued = [type(entry) for entry in world.robot(ALPHA).queue]
    assert queued == [QueuedConfirm, QueuedPosture]
    world, _ = tick(world, LEFT)
    world, _ = answer(world, ALPHA)
    world, _ = step(world, listing_of(ALPHA, leases=MINE))
    world, _ = answer(world, ALPHA)
    assert isinstance(world.robot(ALPHA).op, Confirm)
    world, effects = answer(_lingering(world), ALPHA)
    assert isinstance(phase(world), Lingering)
    return world, effects, "the grip was just released"


@pytest.mark.parametrize(
    "reach",
    [_press_queued_beside_an_engaged_grip, _press_queued_beside_a_lingering_grip],
    ids=["Engaged", "Lingering"],
)
def test_a_posture_start_beside_a_grip_phase_is_refused_at_its_start(reach):
    world, effects, why = reach()
    assert (POSTURE_NOTE, f"move to home ignored: {why}") in notes(effects)
    assert world.posture_moving is None
    assert FireGoal not in [kind for _, kind in calls(effects)]
    assert not isinstance(world.robot(ALPHA).op, PostureStart)


def test_a_posture_start_takes_stops_holds_checks_the_instance_then_fires():
    world, taking = pressed(listed_world(ALPHA))
    world, stopping = answer(world, ALPHA)
    world, checking = answer(world, ALPHA)
    hold = world.robot(ALPHA).hold
    assert hold.reasons == frozenset({Brought(POSTURE, world.posture_moving.start)})
    assert hold.epoch == world.posture_moving.epoch == 1
    world, firing = answer(world, ALPHA)
    assert requests(taking) == [TakeLease()]
    assert requests(stopping) == [StopMoves()]
    assert requests(checking) == [CheckMoveInstance("alpha_inst")]
    assert requests(firing) == [FireGoal("home", HOME)]


def test_an_accepted_posture_goal_keeps_the_hold_and_its_goal_and_is_watched():
    world = _at_the_goal(listed_world(ALPHA))
    start = world.posture_moving.start
    world, effects = answer(world, ALPHA, Answered(GOAL))
    hold = world.robot(ALPHA).hold
    assert hold.has(POSTURE)
    assert hold.goal == AcceptedGoal(start, "home", GOAL)
    assert Watch(ALPHA, start, "home", GOAL) in effects
    assert world.robot(ALPHA).op is None


@pytest.mark.parametrize(
    ("checked", "goal", "why"),
    [
        (Failed("no move"), None, "no move"),
        (Unanswered(), None, "the daemon did not answer"),
        (Raised(RAISED), None, RAISED),
        (Answered(FIRST_REVISION), Failed("busy"), "busy"),
    ],
    ids=["check refused", "check unanswered", "check raised", "goal refused"],
)
def test_a_refused_instance_check_or_goal_drops_the_posture_and_notes_why(
    checked, goal, why
):
    world, _ = pressed(listed_world(ALPHA))
    world, _ = answer(world, ALPHA)
    world, _ = answer(world, ALPHA)
    world, effects = answer(world, ALPHA, checked)
    if goal is not None:
        world, effects = answer(world, ALPHA, goal)
    assert notes(effects) == [(POSTURE_NOTE, f"move to home refused: {why}")]
    assert world.robot(ALPHA).hold is None
    assert world.posture_moving is None
    assert calls(effects) == [(ALPHA, ReleaseLease)]


@pytest.mark.parametrize(
    ("outcome", "why"),
    [
        (Unanswered(), "the goal could not be sent (see log)"),
        (Raised(RAISED), RAISED),
    ],
    ids=["unanswered", "raised"],
)
def test_a_posture_goal_whose_send_got_no_answer_is_stopped(outcome, why):
    world, effects = answer(_at_the_goal(listed_world(ALPHA)), ALPHA, outcome)
    assert notes(effects) == [(POSTURE_NOTE, f"move to home refused: {why}")]
    assert calls(effects) == [(ALPHA, StopMoves)]
    assert world.robot(ALPHA).lost_stop is not None
    assert world.robot(ALPHA).hold.has(POSTURE)


def test_a_posture_goal_accepted_after_a_let_go_is_cancelled_and_not_watched():
    _world, effects = _goal_accepted_after_a_halt_and_its_clear()
    assert CancelGoal(ALPHA, "home", GOAL) in effects
    assert not [effect for effect in effects if isinstance(effect, Watch)]


def test_a_posture_start_let_go_before_its_hold_is_refused_with_the_let_gos_cause():
    world, _ = run(pressed(listed_world(ALPHA))[0], Shutdown())
    world, effects = answer(world, ALPHA)
    assert notes(effects) == [(POSTURE_NOTE, "move to home refused: the node stops")]
    assert world.posture_moving is None


def test_posture_moving_clears_when_its_start_ends_with_no_goal():
    world, _ = pressed(listed_world(ALPHA))
    assert world.posture_moving is not None
    world, effects = answer(world, ALPHA, Failed("leased"))
    assert notes(effects) == [(POSTURE_NOTE, "move to home refused: leased")]
    assert world.posture_moving is None


def _result_lost(world: World):
    return step(world, PostureResult(ALPHA, world.posture_moving.start, ResultLost()))


def test_a_lost_stop_is_sent_while_the_hold_has_the_posture():
    world, effects = _result_lost(posture_held(listed_world(ALPHA, leases=OPEN)))
    assert notes(effects) == [(POSTURE_NOTE, "move to home: result unknown")]
    assert calls(effects) == [(ALPHA, StopMoves)]
    assert world.robot(ALPHA).lost_stop is not None


def test_a_lost_stop_is_sent_after_a_let_go_while_no_other_client_holds_the_lease():
    world, _ = lease_ended(posture_held(listed_world(ALPHA)))
    world, _ = answer(world, ALPHA, Failed("not the headset's"))
    assert world.robot(ALPHA).hold is None
    world, effects = _result_lost(world)
    assert calls(effects) == [(ALPHA, StopMoves)]


def test_a_lost_stop_is_sent_after_the_goals_end_cleared_the_listing():
    world, _ = run(posture_held(listed_world(ALPHA, leases=OPEN)), *GOALS_END)
    assert world.fleet.view is None
    world, effects = _result_lost(world)
    assert calls(effects) == [(ALPHA, StopMoves)]


def test_a_withheld_lost_stop_drops_the_posture_and_clears_the_move():
    closed = {ALPHA: Lease(DIRECTOR, DIRECTOR_LEASE, open=False)}
    world, _ = step(
        posture_held(listed_world(ALPHA, leases=OPEN)),
        listing_of(ALPHA, leases=closed),
    )
    world, effects = _result_lost(world)
    assert calls(effects) == []
    assert world.posture_moving is None
    assert world.robot(ALPHA).lost_stop is None


@pytest.mark.parametrize(
    ("outcome", "noted"),
    [
        (Answered(None), []),
        (
            Failed("alpha did not answer"),
            [(POSTURE_NOTE, "move to home: its stop failed: alpha did not answer")],
        ),
        (
            Raised(RAISED),
            [(POSTURE_NOTE, f"move to home: its stop failed: {RAISED}")],
        ),
    ],
    ids=["answered", "refused", "raised"],
)
def test_a_lost_stops_answer_drops_the_posture_and_clears_the_move(outcome, noted):
    world, _ = _result_lost(posture_held(listed_world(ALPHA)))
    world, effects = answer_lost_stop(world, ALPHA, outcome)
    assert notes(effects) == noted
    assert world.robot(ALPHA).hold is None
    assert world.robot(ALPHA).lost_stop is None
    assert world.posture_moving is None
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_a_release_starts_once_a_taken_hold_ends():
    world, _ = tick(taken(listed_world(ALPHA)), LEFT, left=RELEASED)
    world, effects = step(world, Streamed(LEFT, False))
    assert calls(effects) == [(ALPHA, ReleaseLease)]
    assert isinstance(world.robot(ALPHA).op, Release)


@pytest.mark.parametrize(
    "outcome", [Answered(None), Failed("not held")], ids=["answered", "refused"]
)
def test_an_answered_or_refused_release_clears_the_release_owed(outcome):
    world, effects = answer(release_running(), ALPHA, outcome)
    assert not world.robot(ALPHA).release_owed
    assert world.robot(ALPHA).op is None
    assert calls(effects) == []


@pytest.mark.parametrize(
    "outcome", [Unanswered(), Raised(RAISED)], ids=["unanswered", "raised"]
)
def test_an_unanswered_or_raised_release_starts_a_retry_timer(outcome):
    world, effects = answer(release_running(), ALPHA, outcome)
    timer = timer_of(effects)
    assert world.robot(ALPHA).retry == timer
    assert world.robot(ALPHA).release_owed
    assert calls(effects) == []


def test_an_unanswered_release_after_the_owed_release_cleared_starts_no_timer():
    world, _ = run(release_running(), *GOALS_END)
    world, effects = answer(world, ALPHA, Unanswered())
    assert not [effect for effect in effects if isinstance(effect, StartRetry)]
    assert world.robot(ALPHA).retry is None


@pytest.mark.parametrize(
    "reach",
    [
        lambda: listed_world(ALPHA),
        lambda: taken(listed_world(ALPHA, leases=OPEN)),
    ],
    ids=["no hold", "UnderOpen"],
)
def test_a_confirm_ends_at_once_unless_the_hold_is_taken(reach):
    world, effects = lease_ended(reach())
    assert world.robot(ALPHA).op is None
    assert calls(effects) == []


def test_a_confirm_answered_sets_the_pending_end_and_settles():
    world, asked = lease_ended(taken(listed_world(ALPHA, leases=MINE)))
    assert calls(asked) == [(ALPHA, CheckMove)]
    world, _ = answer(world, ALPHA, Answered(2))
    assert world.robot(ALPHA).hold.pending_end == PendingEnd(2, "the lease ended")
    world, effects = step(world, listing_of(ALPHA, leases=MINE, revision=2))
    assert world.robot(ALPHA).hold.pending_end is None
    assert notes(effects) == []


def test_the_earliest_answered_confirm_settles_two_lease_ends():
    world = taken(listed_world(ALPHA, leases=MINE))
    world, _ = step(world, LeaseEnded(ALPHA, "first"))
    world, _ = answer(world, ALPHA, Answered(3))
    world, _ = step(world, LeaseEnded(ALPHA, "second"))
    world, _ = answer(world, ALPHA, Answered(3))
    assert world.robot(ALPHA).hold.pending_end == PendingEnd(3, "first")
    world, effects = step(world, listing_of(ALPHA, revision=3))
    assert notes(effects) == [(ROBOT_NOTE, "let go of alpha: first")]


def test_a_refused_confirm_lets_go_with_no_release():
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world, effects = answer(world, ALPHA, Failed("not the headset's"))
    assert notes(effects) == [(ROBOT_NOTE, "let go of alpha: the lease ended")]
    assert world.robot(ALPHA).hold is None
    assert not world.robot(ALPHA).release_owed
    assert calls(effects) == []


@pytest.mark.parametrize(
    ("outcome", "why"),
    [(Unanswered(), "the daemon did not answer"), (Raised(RAISED), RAISED)],
    ids=["unanswered", "raised"],
)
def test_an_unanswered_or_raised_confirm_lets_go_and_releases(outcome, why):
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world, effects = answer(world, ALPHA, outcome)
    assert notes(effects) == [(ROBOT_NOTE, f"let go of alpha: {why}")]
    assert world.robot(ALPHA).release_owed
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_a_confirm_whose_hold_ended_ends_with_no_effect():
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world = released(world)
    assert world.robot(ALPHA).hold is None
    assert isinstance(world.robot(ALPHA).op, Confirm)
    world, effects = answer(world, ALPHA, Failed("not the headset's"))
    assert notes(effects) == []
    assert calls(effects) == [(ALPHA, ReleaseLease)]


@pytest.mark.parametrize("ends", [1, 2], ids=["no op", "an op running"])
def test_a_lease_end_starts_a_confirm_or_queues_it(ends):
    world = taken(listed_world(ALPHA))
    for _ in range(ends):
        world, effects = lease_ended(world)
    robot = world.robot(ALPHA)
    assert isinstance(robot.op, Confirm)
    assert robot.op.why == "the lease ended"
    if ends == 1:
        assert calls(effects) == [(ALPHA, CheckMove)]
        assert robot.queue == ()
    else:
        assert calls(effects) == []
        assert robot.queue == (QueuedConfirm("the lease ended"),)


def test_a_settled_end_that_does_not_name_the_headset_lets_go():
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world, _ = answer(world, ALPHA, Answered(2))
    world, effects = step(world, listing_of(ALPHA, revision=2))
    assert notes(effects) == [(ROBOT_NOTE, "let go of alpha: the lease ended")]
    assert world.robot(ALPHA).hold is None


def test_a_settled_end_that_names_the_headset_clears_the_pending_end():
    world, _ = lease_ended(taken(listed_world(ALPHA, leases=MINE)))
    world, _ = answer(world, ALPHA, Answered(2))
    world, effects = step(world, listing_of(ALPHA, leases=MINE, revision=2))
    hold = world.robot(ALPHA).hold
    assert hold is not None and hold.pending_end is None
    assert notes(effects) == []


def test_the_queued_op_starts_when_the_op_ends():
    world, _ = tick(release_running(), LEFT)
    assert isinstance(world.robot(ALPHA).queue[0], Requester)
    world, effects = answer(world, ALPHA)
    assert calls(effects) == [(ALPHA, TakeLease)]
    assert isinstance(world.robot(ALPHA).op, Take)


def _owed_beside_an_open_hold() -> tuple[World, int]:
    """Alpha owes a release whose retry waits on its timer, and the left grip
    holds alpha under the director's open lease, which owes no release of its
    own: with alpha's retry timer."""
    world, effects = answer(release_running(), ALPHA, Unanswered())
    world, _ = step(world, listing_of(ALPHA, leases=OPEN))
    world = taken(world)
    assert world.robot(ALPHA).hold.mode == UnderOpen("director_inst")
    return world, timer_of(effects)


def test_an_owed_release_starts_only_with_no_hold_no_retry_and_an_empty_queue():
    world, effects = answer(release_running(), ALPHA, Unanswered())
    timer = timer_of(effects)
    assert world.robot(ALPHA).release_owed
    assert world.robot(ALPHA).op is None
    assert calls(effects) == []
    world, effects = step(world, RetryDue(ALPHA, timer))
    assert calls(effects) == [(ALPHA, ReleaseLease)]
    world, timer = _owed_beside_an_open_hold()
    world, beside = step(world, RetryDue(ALPHA, timer))
    assert world.robot(ALPHA).release_owed
    assert calls(beside) == []
    world, after = run(
        world, Tick(LEFT, inputs(LEFT, left=RELEASED)), Streamed(LEFT, False)
    )
    assert calls(after) == [(ALPHA, ReleaseLease)]


def test_a_queued_request_takes_before_an_owed_release():
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world = released(world)
    assert world.robot(ALPHA).release_owed
    world, _ = tick(world, RIGHT)
    world, effects = answer(world, ALPHA, Failed("not the headset's"))
    assert calls(effects) == [(ALPHA, TakeLease)]


def test_a_release_never_queues():
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world = released(world)
    assert world.robot(ALPHA).release_owed
    assert world.robot(ALPHA).queue == ()
    world, effects = answer(world, ALPHA, Failed("not the headset's"))
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_retry_due_of_its_timer_starts_the_release_and_of_another_changes_nothing():
    world, effects = answer(release_running(), ALPHA, Unanswered())
    timer = timer_of(effects)
    other, effects = step(world, RetryDue(ALPHA, timer + 100))
    assert other == world
    assert effects == ()
    world, effects = step(world, RetryDue(ALPHA, timer))
    assert world.robot(ALPHA).retry is None
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_a_request_joins_the_running_take():
    world, asked = run(
        listed_world(ALPHA),
        Tick(LEFT, inputs(LEFT)),
        Tick(BASE, inputs(BASE, left=SQUEEZE)),
    )
    assert [r.grip for r in world.robot(ALPHA).op.requesters] == [LEFT, BASE]
    world, stopped = answer(world, ALPHA)
    world, _ = answer(world, ALPHA)
    assert calls(asked + stopped) == [(ALPHA, TakeLease), (ALPHA, StopMoves)]
    assert isinstance(phase(world, LEFT), Engaged)
    assert isinstance(phase(world, BASE), Engaged)


def test_a_request_joins_the_running_join_and_its_reason():
    world, _ = run(
        posture_held(listed_world(ALPHA)),
        Tick(LEFT, inputs(LEFT)),
        Tick(RIGHT, inputs(RIGHT)),
    )
    op = world.robot(ALPHA).op
    assert isinstance(op, Join)
    assert [r.grip for r in op.requesters] == [LEFT, RIGHT]
    assert world.robot(ALPHA).hold.has(RIGHT)


def test_a_request_beside_a_running_join_waits_for_its_stop():
    world = posture_held(listed_world(ALPHA))
    start = world.posture_moving.start
    world, _ = run(
        world,
        Tick(LEFT, inputs(LEFT)),
        PostureResult(ALPHA, start, Finished(HOME_REACHED)),
        Tick(RIGHT, inputs(RIGHT)),
    )
    hold = world.robot(ALPHA).hold
    assert hold.has(LEFT) and hold.has(RIGHT) and not hold.has(POSTURE)
    assert isinstance(phase(world, RIGHT), Requesting)
    assert [r.grip for r in world.robot(ALPHA).op.requesters] == [LEFT, RIGHT]
    world, _ = answer(world, ALPHA)
    assert isinstance(phase(world, RIGHT), Engaged)


def test_a_request_joins_a_taken_hold_at_once():
    world, effects = tick(taken(listed_world(ALPHA)), RIGHT)
    assert isinstance(phase(world, RIGHT), Engaged)
    assert world.robot(ALPHA).hold.has(RIGHT)
    assert calls(effects) == []


def test_a_request_joins_at_once_beside_another_engaged_grip_under_an_open_lease():
    world, effects = tick(taken(listed_world(ALPHA, leases=OPEN)), RIGHT)
    assert isinstance(phase(world, RIGHT), Engaged)
    assert calls(effects) == []


def test_a_request_joins_at_once_beside_another_engaged_grip_under_a_posture_move():
    world, joining = tick(posture_held(listed_world(ALPHA)), LEFT)
    world, stopped = answer(world, ALPHA)
    world, beside = tick(world, RIGHT)
    assert isinstance(phase(world, RIGHT), Engaged)
    assert calls(joining + stopped + beside) == [(ALPHA, StopMoves)]


def test_a_request_waits_while_a_confirm_runs():
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world, effects = tick(world, RIGHT)
    assert isinstance(phase(world, RIGHT), Requesting)
    assert world.robot(ALPHA).queue == (Requester(RIGHT, phase(world, RIGHT).request),)
    assert calls(effects) == []


def test_a_request_does_not_join_at_once_while_a_posture_runs_alone():
    world, effects = tick(posture_held(listed_world(ALPHA)), LEFT)
    assert isinstance(phase(world), Requesting)
    assert isinstance(world.robot(ALPHA).op, Join)
    assert calls(effects) == [(ALPHA, StopMoves)]


def test_a_request_waits_behind_an_acquisition_that_moved_or_is_lost():
    world, _ = run(
        listed_world(ALPHA, leases=OPEN),
        Tick(LEFT, inputs(LEFT)),
        listing_of(ALPHA),
        Tick(RIGHT, inputs(RIGHT)),
    )
    robot = world.robot(ALPHA)
    assert [r.grip for r in robot.op.requesters] == [LEFT]
    assert robot.queue == (Requester(RIGHT, phase(world, RIGHT).request),)


@pytest.mark.parametrize(
    ("reach", "kind"),
    [
        (lambda: listed_world(ALPHA), Take),
        (lambda: posture_held(listed_world(ALPHA)), Join),
    ],
    ids=["no hold", "a posture hold"],
)
def test_a_request_with_no_op_starts_a_take_or_a_join(reach, kind):
    world, _ = tick(reach(), LEFT)
    assert isinstance(world.robot(ALPHA).op, kind)


def _behind_a_release():
    return release_running(), LEFT


def _behind_a_confirm():
    return lease_ended(taken(listed_world(ALPHA)))[0], RIGHT


def _behind_a_posture_start():
    return pressed(listed_world(ALPHA))[0], LEFT


@pytest.mark.parametrize(
    "reach",
    [_behind_a_release, _behind_a_confirm, _behind_a_posture_start],
    ids=["Release", "Confirm", "PostureStart"],
)
def test_a_request_waits_behind_a_release_a_confirm_or_a_posture_start(reach):
    world, grip = reach()
    world, effects = tick(world, grip)
    assert world.robot(ALPHA).queue == (Requester(grip, phase(world, grip).request),)
    assert calls(effects) == []


def test_a_queued_acquisition_at_its_turn_joins_at_once_beside_an_engaged_grip():
    world, _ = lease_ended(taken(listed_world(ALPHA, leases=MINE)))
    world, _ = tick(world, BASE, right=SQUEEZE)
    assert isinstance(phase(world, BASE), Requesting)
    world, effects = answer(world, ALPHA)
    assert phase(world, BASE) == Engaged(
        ALPHA, frozenset({ALPHA_BASE}), world.robot(ALPHA).hold.epoch
    )
    assert calls(effects) == []


def test_a_queued_acquisition_at_its_turn_takes_every_other_queued_one():
    world, _ = run(
        release_running(),
        Tick(LEFT, inputs(LEFT)),
        Tick(BASE, inputs(BASE, right=SQUEEZE)),
    )
    world, effects = answer(world, ALPHA)
    op = world.robot(ALPHA).op
    assert isinstance(op, Take)
    assert [r.grip for r in op.requesters] == [LEFT, BASE]
    assert world.robot(ALPHA).queue == ()
    assert calls(effects) == [(ALPHA, TakeLease)]


def test_a_withdrawn_request_leaves_the_take_the_join_or_the_queue():
    taking, _ = tick(listed_world(ALPHA), LEFT)
    taking, _ = tick(taking, LEFT, left=RELEASED)
    assert taking.robot(ALPHA).op.requesters == ()
    joining, _ = tick(posture_held(listed_world(ALPHA)), LEFT)
    assert joining.robot(ALPHA).hold.has(LEFT)
    joining, _ = tick(joining, LEFT, left=RELEASED)
    assert joining.robot(ALPHA).op.requesters == ()
    assert not joining.robot(ALPHA).hold.has(LEFT)
    queued, _ = tick(release_running(), LEFT)
    assert queued.robot(ALPHA).queue != ()
    queued, _ = tick(queued, LEFT, left=RELEASED)
    assert queued.robot(ALPHA).queue == ()


def test_a_let_go_of_a_hold_cancels_its_goal_notes_the_robot_and_moves_the_epoch():
    world, _ = lease_ended(posture_held(listed_world(ALPHA)))
    epoch = world.robot(ALPHA).epoch
    world, effects = answer(world, ALPHA, Failed("not the headset's"))
    assert effects[:2] == (
        CancelGoal(ALPHA, "home", GOAL),
        Note(ROBOT_NOTE, "let go of alpha: the lease ended"),
    )
    assert world.robot(ALPHA).hold is None
    assert world.robot(ALPHA).epoch == epoch + 1


def test_a_let_go_with_no_hold_moves_the_epoch_and_notes_nothing():
    world = listed_world(ALPHA)
    world, effects = step(world, halt(ALPHA))
    assert world.robot(ALPHA).epoch == 1
    assert notes(effects) == []


@pytest.mark.parametrize(
    ("outcome", "owed"),
    [(Unanswered(), True), (Failed("not the headset's"), False)],
    ids=["asked", "not asked"],
)
def test_a_let_go_releases_a_taken_hold_only_when_asked(outcome, owed):
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world, _ = answer(world, ALPHA, outcome)
    assert world.robot(ALPHA).hold is None
    assert world.robot(ALPHA).release_owed is owed


def test_a_let_go_drops_the_queued_requests_posture_starts_and_confirms():
    world, _ = pressed(release_running())
    world, _ = run(world, Tick(LEFT, inputs(LEFT)), LeaseEnded(ALPHA, ""))
    queue = world.robot(ALPHA).queue
    assert [type(entry) for entry in queue] == [
        QueuedPosture,
        Requester,
        QueuedConfirm,
    ]
    world, effects = step(world, halt(ALPHA))
    assert (LEFT, f"squeeze refused: {HALTED}") in notes(effects)
    assert (POSTURE_NOTE, f"move to home refused: {HALTED}") in notes(effects)
    assert phase(world) == Idle()
    assert world.robot(ALPHA).queue == ()
    assert world.posture_moving is None


def _five_robots() -> tuple[World, int]:
    """Alpha held by the left arm; bravo owing a release whose retry waits on
    its timer; charlie listed with no state; echo's take answered, its stop
    in flight; delta chosen, not listed, with no state. With bravo's
    timer."""
    world = listed_world(ALPHA, BRAVO, CHARLIE, ECHO)
    world, _ = tick(world, BASE, right=SQUEEZE, chosen=BRAVO)
    world, _ = answer(world, BRAVO, Unanswered())
    world, unanswered = answer(world, BRAVO, Unanswered())
    timer = timer_of(unanswered)
    world, _ = tick(world, LEFT, left=RELEASED, chosen=ALPHA)
    world = taken(world)
    world, _ = run(
        world,
        Tick(RIGHT, inputs(RIGHT, right=RELEASED, chosen=ECHO)),
        Tick(RIGHT, inputs(RIGHT, chosen=ECHO)),
    )
    world, _ = answer(world, ECHO)
    world, _ = tick(world, BASE, chosen=DELTA, pins=frozenset())
    assert world.robot(ALPHA).hold is not None
    assert world.robot(BRAVO).retry == timer
    assert isinstance(world.robot(ECHO).op, Take)
    assert world.seen_robot == DELTA
    return world, timer


EVERY_ROBOT = (ALPHA, BRAVO, CHARLIE, DELTA, ECHO)


def test_a_halt_start_lets_go_of_every_robot():
    before, timer = _five_robots()
    world, effects = step(before, halt(ALPHA, BRAVO, CHARLIE, ECHO))
    assert all(
        world.robot(r).epoch == before.robot(r).epoch + 1 for r in EVERY_ROBOT
    )
    assert notes(effects) == [(ROBOT_NOTE, f"let go of alpha: {HALTED}")]
    assert world.robot(BRAVO).release_owed
    assert world.robot(BRAVO).retry == timer
    world, retried = step(world, RetryDue(BRAVO, timer))
    assert calls(retried) == [(BRAVO, ReleaseLease)]
    world, stopped = answer(world, ECHO)
    assert calls(stopped) == [(ECHO, ReleaseLease)]


@pytest.mark.parametrize(
    "lease",
    [
        Lease(DIRECTOR, DIRECTOR_LEASE, open=False),
        None,
        Lease(OTHER, DIRECTOR_LEASE, open=True),
    ],
    ids=["closed", "ended", "another holder"],
)
def test_an_open_lease_that_closes_ends_or_passes_lets_go(lease):
    world = taken(listed_world(ALPHA, leases=OPEN))
    leases = {} if lease is None else {ALPHA: lease}
    world, effects = step(world, listing_of(ALPHA, leases=leases))
    assert notes(effects) == [(ROBOT_NOTE, f"let go of alpha: {OPEN_CLOSED}")]
    assert world.robot(ALPHA).hold is None
    assert calls(effects) == []


def test_the_end_of_the_goals_lets_go_and_clears_every_release_owed_retry_and_the_listing():
    before, timer = _five_robots()
    world, effects = run(before, *GOALS_END)
    assert all(
        world.robot(r).epoch == before.robot(r).epoch + 1 for r in EVERY_ROBOT
    )
    assert notes(effects) == [(ROBOT_NOTE, f"let go of alpha: {CHANGES_ENDED}")]
    assert not world.robot(BRAVO).release_owed
    assert world.robot(BRAVO).retry is None
    assert world.fleet.view is None
    world, retried = step(world, RetryDue(BRAVO, timer))
    assert retried == ()
    world, stopped = answer(world, ECHO)
    assert not world.robot(ECHO).release_owed
    assert calls(stopped) == []


def test_shutdown_lets_go_of_every_robot():
    before = posture_held(listed_world(ALPHA, BRAVO))
    world, effects = step(before, Shutdown())
    assert notes(effects) == [(ROBOT_NOTE, "let go of alpha: the node stops")]
    assert not [effect for effect in effects if isinstance(effect, CancelGoal)]
    assert world.robot(ALPHA).hold is None
    assert world.robot(ALPHA).epoch == before.robot(ALPHA).epoch + 1
    assert world.robot(BRAVO).epoch == 1


def test_shutdown_starts_no_release():
    world, timer = _owed_beside_an_open_hold()
    world, _ = step(world, RetryDue(ALPHA, timer))
    world, effects = step(world, Shutdown())
    assert world.robot(ALPHA).hold is None
    assert world.robot(ALPHA).release_owed
    assert calls(effects) == []
    assert world.robot(ALPHA).op is None


def test_a_drop_of_the_last_reason_ends_the_hold_and_owes_a_release_of_a_taken_lease():
    before, _ = tick(taken(listed_world(ALPHA)), LEFT, left=RELEASED)
    world, effects = step(before, Streamed(LEFT, False))
    robot = world.robot(ALPHA)
    assert robot.hold is None
    assert robot.epoch == before.robot(ALPHA).epoch + 1
    assert robot.release_owed
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_a_drop_under_an_open_lease_owes_nothing():
    world, _ = tick(taken(listed_world(ALPHA, leases=OPEN)), LEFT, left=RELEASED)
    world, effects = step(world, Streamed(LEFT, False))
    assert world.robot(ALPHA).hold is None
    assert not world.robot(ALPHA).release_owed
    assert calls(effects) == []


def test_a_drop_under_another_epoch_changes_nothing():
    world, _ = _beside_the_hold_of_another_epoch()
    hold = world.robot(ALPHA).hold
    world, effects = step(world, Streamed(LEFT, False))
    assert phase(world) == Idle()
    assert world.robot(ALPHA).hold == hold
    assert calls(effects) == []


def _moving_on_bravo() -> World:
    return posture_held(listed_world(ALPHA, BRAVO), BRAVO, BRAVO_HOME)


def _lingering_moving_halted() -> World:
    world = taken(_moving_on_bravo())
    world, _ = step(world, halt(ALPHA, BRAVO))
    assert isinstance(phase(world), Lingering)
    assert world.posture_moving is not None
    return world


def _requesting_moving() -> World:
    world, _ = tick(_moving_on_bravo(), LEFT)
    assert isinstance(phase(world), Requesting)
    return world


def _moving_halted() -> World:
    world, _ = step(_moving_on_bravo(), halt(ALPHA, BRAVO))
    return world


def _halted() -> World:
    return listed_world(ALPHA, halted_by=HALTER)


@pytest.mark.parametrize(
    ("reach", "press", "note"),
    [
        (
            _lingering_moving_halted,
            PressInputs(True, ALPHA, None),
            "move to home ignored: release the grip first",
        ),
        (
            _requesting_moving,
            PressInputs(False, ALPHA, None),
            "move to home ignored: release the grip first",
        ),
        (
            _lingering_moving_halted,
            PressInputs(False, ALPHA, None),
            "move to home ignored: the grip was just released",
        ),
        (
            _moving_halted,
            PressInputs(False, ALPHA, None),
            "move to home ignored: still moving to home",
        ),
        (_halted, PressInputs(False, None, None), "move to home: no robot paired"),
        (
            _halted,
            PressInputs(False, ALPHA, None),
            "move to home: alpha has no postures",
        ),
        (_halted, PressInputs(False, ALPHA, HOME), f"move to home refused: {HALTED}"),
    ],
    ids=[
        "a squeeze held",
        "a grip requesting",
        "a grip lingering",
        "a move running",
        "no robot",
        "no postures",
        "halted",
    ],
)
def test_each_press_refusal_notes_why_first_match(reach, press, note):
    before = reach()
    world, effects = step(before, Press("home", press))
    assert notes(effects) == [(POSTURE_NOTE, note)]
    assert calls(effects) == []
    assert world.posture_moving == before.posture_moving


def test_a_press_sets_the_posture_move_and_starts_or_queues_its_start():
    world, effects = pressed(listed_world(ALPHA))
    moving = world.posture_moving
    assert moving == PostureMoving("home", world.robot(ALPHA).op.id, ALPHA)
    assert calls(effects) == [(ALPHA, TakeLease)]
    world, again = pressed(world)
    still = "move to home ignored: still moving to home"
    assert notes(again) == [(POSTURE_NOTE, still)]
    assert world.posture_moving == moving
    world, effects = pressed(release_running())
    (queued,) = world.robot(ALPHA).queue
    assert world.posture_moving == PostureMoving("home", queued.start, ALPHA)
    assert queued == QueuedPosture(queued.start, "home", HOME)
    assert calls(effects) == []


def test_a_press_reads_no_hand_into_the_world():
    world, _ = tick(listed_world(ALPHA), LEFT, left=RELEASED)
    after, _ = pressed(world, squeezing=True)
    assert after.hands == world.hands


def test_a_posture_result_drops_the_posture_and_clears_the_move():
    world = posture_held(listed_world(ALPHA))
    start = world.posture_moving.start
    world, effects = step(
        world, PostureResult(ALPHA, start, Finished(HOME_REACHED))
    )
    assert notes(effects) == [(POSTURE_NOTE, HOME_REACHED)]
    assert world.robot(ALPHA).hold is None
    assert world.posture_moving is None
    assert calls(effects) == [(ALPHA, ReleaseLease)]


def test_a_lost_result_stops_the_move():
    world, effects = _result_lost(posture_held(listed_world(ALPHA)))
    assert notes(effects) == [(POSTURE_NOTE, "move to home: result unknown")]
    assert calls(effects) == [(ALPHA, StopMoves)]


def test_a_result_of_another_start_changes_nothing():
    world = posture_held(listed_world(ALPHA))
    start = world.posture_moving.start
    after, effects = step(
        world, PostureResult(ALPHA, start + 100, Finished(HOME_REACHED))
    )
    assert after == world
    assert effects == ()


def test_the_goals_run_from_their_following_status_and_stop_at_a_goal_end():
    world = World.start(HEADSET)
    assert not world.fleet.goals_run
    world, _ = step(world, GoalsStarted(1))
    assert world.fleet.generation == 1 and not world.fleet.goals_run
    world, _ = step(world, GoalsStatusChanged(Following()))
    assert world.fleet.goals_run and world.fleet.view is None
    world, _ = step(world, listing_of(ALPHA, revision=1))
    assert world.fleet.listed() == fleet_view([ALPHA])
    world, _ = step(world, GoalsEnded(CHANGES_ENDED))
    assert world.fleet.view is None
    world, _ = step(world, GoalsStatusChanged(Starting()))
    assert not world.fleet.goals_run
    assert world.fleet.down_cause() == "waiting for the fleet listing"


def test_a_step_that_changes_a_hold_or_the_fleet_facts_ends_with_changed():
    world, started = step(World.start(HEADSET), GoalsStarted(1))
    world, following = step(world, GoalsStatusChanged(Following()))
    world, listed = step(world, listing_of(ALPHA, revision=1))
    assert started[-1] == Changed()
    assert Changed() not in following
    assert listed[-1] == Changed()
    world, requested = tick(world, LEFT)
    world, _ = answer(world, ALPHA)
    world, held = answer(world, ALPHA)
    assert Changed() not in requested
    assert held[-1] == Changed()
    world, ended = step(world, GoalsEnded(CHANGES_ENDED))
    world, stopped = step(world, GoalsStatusChanged(Starting()))
    assert Changed() not in ended
    assert stopped[-1] == Changed()


def test_a_squeeze_during_a_release_takes_the_lease_anew_after_it():
    world, _ = tick(release_running(), LEFT)
    world, released_effects = answer(world, ALPHA)
    world, stopped = answer(world, ALPHA)
    world, _ = answer(world, ALPHA)
    assert calls(released_effects) == [(ALPHA, TakeLease)]
    assert calls(stopped) == [(ALPHA, StopMoves)]
    assert isinstance(phase(world), Engaged)


def test_a_squeeze_whose_burst_ends_during_a_confirm_takes_again():
    world, _ = lease_ended(taken(listed_world(ALPHA)))
    world = released(world)
    world, waiting = tick(world, LEFT)
    assert calls(waiting) == []
    world, effects = answer(world, ALPHA, Failed("not the headset's"))
    assert calls(effects) == [(ALPHA, TakeLease)]


def test_a_pinned_pair_restarted_with_the_same_peer_info_keeps_the_grip():
    world = taken(listed_world(ALPHA))
    engaged = phase(world)
    restarted = Pin(Slot.ARM, Peer("alpha left arm"))
    assert restarted is not ALPHA_ARM
    world, _ = tick(world, LEFT, pins=frozenset({restarted}))
    assert phase(world) == engaged


def test_a_pair_that_joins_while_engaged_is_not_pinned():
    world = taken(listed_world(ALPHA))
    world, _ = tick(world, LEFT, pins=frozenset({ALPHA_ARM, ALPHA_GRIPPER}))
    assert phase(world) == Engaged(ALPHA, frozenset({ALPHA_ARM}), 1)


@pytest.mark.parametrize(
    ("read", "stops"),
    [
        (None, True),
        ({"left": RELEASED}, False),
        ({"chosen": BRAVO}, False),
        ({"left": UNKNOWN}, False),
    ],
    ids=["no tick read it", "a release read", "a choice read", "a stale frame read"],
)
def test_a_release_or_a_choice_no_tick_read_before_a_take_is_answered_still_gets_its_stop(
    read, stops
):
    world, _ = tick(listed_world(ALPHA, BRAVO), LEFT)
    if read is not None:
        world, _ = tick(world, LEFT, **read)
    world, effects = answer(world, ALPHA)
    assert ((ALPHA, StopMoves) in calls(effects)) is stops


def test_a_press_queued_ahead_of_a_squeeze_is_refused_at_its_start_and_the_squeeze_takes_the_robot():
    world, _ = pressed(release_running())
    world, _ = tick(world, LEFT)
    world, effects = answer(world, ALPHA)
    assert notes(effects) == [
        (POSTURE_NOTE, "move to home ignored: release the grip first")
    ]
    assert calls(effects) == [(ALPHA, TakeLease)]
    assert world.posture_moving is None
    assert isinstance(world.robot(ALPHA).op, Take)


def test_a_press_in_a_burst_that_outlives_a_let_go_is_refused_as_just_released():
    world = taken(listed_world(ALPHA, leases=OPEN))
    world, _ = step(world, listing_of(ALPHA))
    assert isinstance(phase(world), Lingering)
    world, refused = pressed(world)
    assert notes(refused) == [
        (POSTURE_NOTE, "move to home ignored: the grip was just released")
    ]
    assert calls(refused) == []
    world, _ = step(world, Streamed(LEFT, False))
    world, started = pressed(world)
    assert calls(started) == [(ALPHA, TakeLease)]


def test_a_press_while_a_grip_requests_is_refused_whatever_the_hand_read_since():
    world, _ = tick(listed_world(ALPHA), LEFT)
    world, effects = pressed(world, squeezing=False)
    assert notes(effects) == [
        (POSTURE_NOTE, "move to home ignored: release the grip first")
    ]


def _burst_after_a_let_go():
    world = taken(listed_world(ALPHA, leases=OPEN))
    world, _ = step(world, listing_of(ALPHA))
    world, _ = tick(world, LEFT, left=RELEASED)
    return world, [[(ALPHA, TakeLease)], [(ALPHA, StopMoves)]]


def _burst_under_an_open_lease():
    world, _ = tick(taken(listed_world(ALPHA, leases=OPEN)), LEFT, left=RELEASED)
    return world, [[(ALPHA, CheckMove)], [(ALPHA, StopMoves)]]


def _burst_beside_a_posture():
    world, _ = tick(posture_held(listed_world(ALPHA)), LEFT)
    world, _ = answer(world, ALPHA)
    world, _ = tick(world, LEFT, left=RELEASED)
    return world, [[(ALPHA, StopMoves)]]


@pytest.mark.parametrize(
    "reach",
    [_burst_after_a_let_go, _burst_under_an_open_lease, _burst_beside_a_posture],
    ids=["no hold left", "under an open lease", "beside a posture reason"],
)
def test_a_resqueeze_in_its_own_burst_waits_for_the_bursts_end(reach):
    world, then = reach()
    lingering = phase(world)
    assert isinstance(lingering, Lingering)
    assert not must_release(world)
    world, waiting = tick(world, LEFT)
    assert phase(world) == lingering
    assert calls(waiting) == []
    world, _ = step(world, Streamed(LEFT, False))
    world, effects = tick(world, LEFT)
    asked = [calls(effects)]
    for _ in then[1:]:
        world, effects = answer(world, ALPHA)
        asked.append(calls(effects))
    assert asked == then


def test_each_hands_squeeze_is_its_own_claim_on_the_base():
    world, _ = tick(listed_world(ALPHA), BASE, left=SQUEEZE)
    world, _ = answer(world, ALPHA, Failed("leased"))
    assert must_release(world, LEFT)
    world = taken(world, BASE, left=SQUEEZE, right=SQUEEZE)
    assert isinstance(phase(world, BASE), Engaged)
    world, _ = tick(world, BASE, left=SQUEEZE, right=SQUEEZE)
    assert isinstance(phase(world, BASE), Engaged)
    assert must_release(world, LEFT)
