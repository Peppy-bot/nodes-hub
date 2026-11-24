"""The hands' command streams: the pure ticks and the release burst, and the
streams on the wire against the harness's mocks, each stream ticking on a
manual clock so one step is one tick of each stream.

Where a test proves that nothing was streamed, it streams a distinct message
next: the wire delivers in order, so the first message read being that one
proves that nothing went out before it.
"""

import asyncio
import time
from dataclasses import dataclass
from types import SimpleNamespace

import peppygen.clock
import pytest
from control_core_py.geometry import Stamped
from peppygen.paired_topics.left_arm import pose_setpoints as left_arm_pose_setpoints
from peppygen.paired_topics.left_gripper import (
    gripper_setpoints as left_gripper_setpoints,
)
from peppygen.paired_topics.right_arm import pose_setpoints as right_arm_pose_setpoints
from peppygen.paired_topics.right_arm import pose_states as right_arm_pose_states
from peppygen.paired_topics.right_gripper import (
    gripper_setpoints as right_gripper_setpoints,
)

from tests.helpers import (
    AGREED,
    CALL_S,
    ELSEWHERE,
    HALTER,
    HELD_TICKS,
    POSE,
    ROBOT_FRAME,
    SETTINGS,
    TICK_S,
    WIRE_TICKS,
    FakeCalls,
    FakeToken,
    ManualClock,
    NoWireSlot,
    Peer,
    ScriptedHeadset,
    StubTopic,
    boot,
    boot_clocked,
    eventually,
    fleet_view,
    parked,
    peer_of,
    pose_at,
    reported,
    robots_of,
    running,
    running_drain,
    scripted_driver,
    settle,
    ticks_until,
)
from xr_commander import publish
from xr_commander.bus import REAL_CLOCK
from xr_commander.clutch import HandClutch, Setpoint
from xr_commander.devices import Hand, HandSample
from xr_commander.drive import Driver
from xr_commander.grip_world import Engaged, Idle, Lingering, Pin, Requesting, Slot
from xr_commander.robots import PairedRobots


def test_a_measured_pose_is_kept_per_peer_and_goes_stale_rather_than_lying():
    holder = publish.MeasuredPoses()
    measured = reported(POSE)
    assert holder.fresh("alpha", 10.0) is None
    assert holder.fresh(None, 10.0) is None
    holder.set("alpha", measured)
    assert holder.fresh("alpha", 10.0) is measured
    assert holder.fresh("bravo", 10.0) is None
    assert holder.fresh("alpha", 10.0, now_monotonic=time.monotonic() + 60.0) is None


# One paired robot outside any copy, holding the right arm and gripper pairs
# under one instance id, as a backbone does.
ONE_ROBOT = {
    "right_arm_members": {"backbone_inst": None},
    "right_gripper_members": {"backbone_inst": None},
}


def test_a_failed_publisher_declaration_ends_the_stream_loudly(capsys):
    # A broken transport cannot be scripted onto the real wire; the stub
    # stands in for the runner refusing the declaration.
    class NoPublisherTopic:
        async def declare_publisher(self, _runner):
            raise RuntimeError("no transport")

    async def run():
        await asyncio.wait_for(
            publish.run_stream(
                None,
                NoPublisherTopic(),
                0.001,
                "left arm",
                FakeToken(),
                lambda: None,
                REAL_CLOCK,
            ),
            1.0,
        )

    asyncio.run(run())
    assert "failed to declare" in capsys.readouterr().out


# The gripper's tick, pure: what it sends from its own reads.


ALPHA_GRIPPER = Peer("alpha gripper")
GRIPPING = Engaged("alpha", frozenset({Pin(Slot.GRIPPER, ALPHA_GRIPPER)}), 1)


def trigger_at(trigger: float, *, squeezing: bool = True) -> HandSample:
    """A fresh sample of the hand with its trigger at `trigger`."""
    return HandSample(pose=POSE, squeezing=squeezing, trigger=trigger)


def test_the_gripper_tick_sends_nothing_without_a_squeeze():
    # No fresh sample of the hand: the gripper's own tick reads no squeeze.
    assert publish.gripper_tick(None, "alpha", False, GRIPPING, 1.0) is None


def test_the_gripper_tick_sends_nothing_while_the_hand_must_release():
    assert publish.gripper_tick(trigger_at(0.5), "alpha", True, GRIPPING, 1.0) is None


def test_the_gripper_tick_sends_nothing_off_the_engaged_robot():
    pins = GRIPPING.pins
    for phase in (Idle(), Requesting("alpha", 3, pins)):
        assert publish.gripper_tick(trigger_at(0.5), "alpha", False, phase, 1.0) is None
    # Engaged on alpha, with bravo chosen at the gripper's own read.
    assert publish.gripper_tick(trigger_at(0.5), "bravo", False, GRIPPING, 1.0) is None


def test_the_gripper_is_silent_while_its_hand_is_in_its_release_burst():
    lingering = Lingering("alpha", GRIPPING.pins, 1)
    assert (
        publish.gripper_tick(trigger_at(1.0), "alpha", False, lingering, 1.0) is None
    )


def test_the_gripper_tick_sends_to_the_phases_gripper_pin():
    pin, opening = publish.gripper_tick(trigger_at(0.25), "alpha", False, GRIPPING, 0.8)
    assert pin == ALPHA_GRIPPER
    assert opening == pytest.approx(0.6)
    # A phase that pins no gripper sends the trigger nowhere.
    arm_only = Engaged("alpha", frozenset({Pin(Slot.ARM, Peer("alpha arm"))}), 1)
    assert publish.gripper_tick(trigger_at(0.25), "alpha", False, arm_only, 1.0) is None


def test_the_gripper_tick_stops_at_the_release_reading():
    released = trigger_at(0.5, squeezing=False)
    # The phase is still engaged: the pose stream has not read the release.
    assert publish.gripper_tick(released, "alpha", False, GRIPPING, 1.0) is None


# The streams on the wire


TWO_ROBOTS = {
    "right_arm_members": {
        "alpha_backbone_inst": "alpha",
        "bravo_backbone_inst": "bravo",
    },
    "right_gripper_members": {
        "alpha_backbone_inst": "alpha",
        "bravo_backbone_inst": "bravo",
    },
}


def peers_by_robot(h, module):
    """Each robot's pair on `module`'s slot, by the copy it runs in."""
    return {member.copy: member.info for member in module.peers(h.node_runner)}


def pose_stream(h, clock, driver, session, measured, robots, handedness="right"):
    """The pose stream of one hand on the wire, ticking on `clock` every
    `TICK_S`."""
    arm, gripper = {
        "left": (left_arm_pose_setpoints, left_gripper_setpoints),
        "right": (right_arm_pose_setpoints, right_gripper_setpoints),
    }[handedness]
    return lambda token: publish.stream_pose(
        h.node_runner,
        topic_module=arm,
        gripper_module=gripper,
        handedness=handedness,
        driver=driver,
        clutch=HandClutch(1.0),
        session=session,
        measured=measured,
        robots=robots,
        settings=SETTINGS,
        token=token,
        clock=clock,
    )


def gripper_stream(h, clock, driver, session, robots):
    """The right hand's gripper stream on the wire, ticking on `clock` every
    `TICK_S` and stamping each setpoint with `clock`'s time."""
    return lambda token: publish.stream_gripper(
        h.node_runner,
        topic_module=right_gripper_setpoints,
        handedness="right",
        driver=driver,
        session=session,
        robots=robots,
        settings=SETTINGS,
        token=token,
        clock=clock,
        timestamp_s=clock.monotonic,
    )


def reporting(step, measured, poses):
    """`step` with every measured pose of `poses` reported again first, so
    each stays fresh however many ticks a test runs."""

    async def reported_step(count=1):
        for _ in range(count):
            for peer, pose in poses.items():
                measured.set(peer, reported(pose))
            await step()

    return reported_step


def ends_the_burst(times: list[float]) -> bool:
    """Whether a release burst whose ticks came at `times` ends after the
    send of the last of them: it has sent `RELEASE_HOLD_MESSAGES` values and
    `RELEASE_HOLD_S` passed since its first."""
    return (
        len(times) >= publish.RELEASE_HOLD_MESSAGES
        and times[-1] - times[0] >= publish.RELEASE_HOLD_S
    )


def burst_length(times: list[float]) -> int:
    """How many values a release burst whose ticks come at `times` sends;
    AssertionError when the burst does not end within them."""
    ends = (
        count for count in range(1, len(times) + 1) if ends_the_burst(times[:count])
    )
    length = next(ends, None)
    assert length is not None, f"a burst at {times} does not end"
    return length


async def sends_until_idle(tick, clock, driver, wire, hand="right") -> list:
    """Tick until `hand`'s grip is idle, reading the message of each tick
    before: each message with the time of its tick."""
    sent = []
    for _ in range(WIRE_TICKS):
        await tick()
        if isinstance(driver.world.grips.of(hand).phase, Idle):
            return sent
        sent.append((clock.monotonic(), await asyncio.wait_for(wire.next(), CALL_S)))
    raise AssertionError("the grip never went idle")


class MovableSession(ScriptedHeadset):
    """A session whose hand pose can be moved, so a driven target is
    distinguishable on the wire from the measured pose."""

    def __init__(self, clock: ManualClock) -> None:
        super().__init__(clock)
        self.hand_pose = POSE

    def press(self, **hands):
        super().press(pose=self.hand_pose, **hands)


async def test_the_gripper_stream_publishes_openings_while_the_hand_is_engaged():
    async with boot_clocked(**ONE_ROBOT) as h:
        clock = ManualClock()
        driver = scripted_driver("backbone_inst", clock=clock)
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        measured = publish.MeasuredPoses(clock)
        measured.set(peer_of(h, right_arm_pose_setpoints), reported(POSE))
        robots = robots_of(h)
        gripper = h.mocks.pairings.right_gripper[0].gripper_setpoints
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots),
            gripper_stream(h, clock, driver, session, robots),
            period=TICK_S,
        ) as step:
            # The squeeze takes the robot, and the next tick drives it.
            await step(2)
            message = await asyncio.wait_for(gripper.next(), CALL_S)
        assert message.opening == 1.0  # released trigger rests at open_fraction
        assert message.max_effort == 0.0
        assert message.timestamp == clock.monotonic()


async def test_the_pose_stream_refuses_to_engage_without_a_fresh_measured_pose():
    async with boot_clocked(**ONE_ROBOT) as h:
        clock = ManualClock()
        log = []
        driver = scripted_driver("backbone_inst", clock=clock, log=log)
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        measured = publish.MeasuredPoses(clock)
        wire = h.mocks.pairings.right_arm[0].pose_setpoints
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots_of(h)),
            period=TICK_S,
        ) as step:
            await step(5)
            assert log == []
            assert driver.note(Hand.RIGHT) == "squeeze refused: no measured pose"
            measured.set(peer_of(h, right_arm_pose_setpoints), reported(ELSEWHERE))
            await step(2)
            first = await asyncio.wait_for(wire.next(), CALL_S)
        assert first.position[0] == pytest.approx(0.4)


async def test_the_pose_stream_engages_once_the_follower_reports():
    async with boot_clocked(**ONE_ROBOT) as h:
        clock = ManualClock()
        driver = scripted_driver("backbone_inst", clock=clock)
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        measured = publish.MeasuredPoses(clock)
        measured.set(peer_of(h, right_arm_pose_setpoints), reported(POSE))
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots_of(h)),
            period=TICK_S,
        ) as step:
            before_s = peppygen.clock.now_ns() / 1e9
            await step(2)
            after_s = peppygen.clock.now_ns() / 1e9
            message = await asyncio.wait_for(
                h.mocks.pairings.right_arm[0].pose_setpoints.next(), CALL_S
            )
        # The engage tick commands the measured pose: a no-op that starts the
        # stream. It names the frame the follower measures in, at the time of
        # its tick on the instance's clock.
        assert message.frame == ROBOT_FRAME
        assert before_s <= message.timestamp <= after_s
        assert message.position == [0.0, 0.0, 0.0]
        assert message.orientation == [0.0, 0.0, 0.0, 1.0]


async def test_a_robot_change_mid_squeeze_releases_the_robot_left_behind_alone():
    async with boot_clocked(**TWO_ROBOTS) as h:
        clock = ManualClock()
        log = []
        driver = scripted_driver("alpha", "bravo", clock=clock, log=log)
        robots = robots_of(h)
        peers = peers_by_robot(h, right_arm_pose_setpoints)
        alpha_wire, bravo_wire = (
            mock.pose_setpoints for mock in h.mocks.pairings.right_arm
        )
        measured = publish.MeasuredPoses(clock)
        poses = {peers["alpha"]: POSE, peers["bravo"]: pose_at((0.5, 0.0, 0.0))}
        session = MovableSession(clock)
        session.press(right=(False, False, True))
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots),
            period=TICK_S,
        ) as step:
            tick = reporting(step, measured, poses)
            # Alpha, paired first, is driven: hand moved 0.3 away.
            await tick(2)
            await asyncio.wait_for(alpha_wire.next(), CALL_S)
            session.hand_pose = pose_at((0.3, 0.0, 0.0))
            session.press(right=(False, False, True))
            await tick()
            driven = await asyncio.wait_for(alpha_wire.next(), CALL_S)
            assert driven.position[0] == pytest.approx(0.3)
            assert robots.choose("bravo")
            # Alpha gets the whole release burst at its own measured pose.
            tail = await sends_until_idle(tick, clock, driver, alpha_wire)
            assert all(message.position[0] == 0.0 for _, message in tail)
            assert len(tail) == burst_length([at for at, _ in tail])
            assert ("release", "alpha") in log
            # Let go and squeeze again: bravo engages at its own pose.
            session.press(right=(False, False, False))
            await tick()
            session.press(right=(False, False, True))
            await tick(2)
            first = await asyncio.wait_for(bravo_wire.next(), CALL_S)
            assert first.position[0] == pytest.approx(0.5)


async def test_a_grip_held_across_a_robot_change_leaves_the_new_gripper_alone():
    async with boot_clocked(**TWO_ROBOTS) as h:
        clock = ManualClock()
        driver = scripted_driver("alpha", "bravo", clock=clock)
        robots = robots_of(h)
        alpha_wire, bravo_wire = (
            mock.gripper_setpoints for mock in h.mocks.pairings.right_gripper
        )
        measured = publish.MeasuredPoses(clock)
        poses = {
            peer: POSE
            for peer in peers_by_robot(h, right_arm_pose_setpoints).values()
        }
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots),
            gripper_stream(h, clock, driver, session, robots),
            period=TICK_S,
        ) as step:
            tick = reporting(step, measured, poses)
            await tick(2)
            assert (await asyncio.wait_for(alpha_wire.next(), CALL_S)).opening == 1.0
            assert robots.choose("bravo")
            # The squeeze held over the change, with the trigger open, sends
            # nothing to bravo's gripper.
            await tick(HELD_TICKS)
            # The trigger pulled at the release and the new squeeze: bravo's
            # first opening is the closed one, so nothing reached it while
            # the squeeze was held over.
            session.trigger = 1.0
            session.press(right=(False, False, False))
            await tick()
            session.press(right=(False, False, True))

            def engaged_on_bravo():
                phase = driver.world.grips.right.phase
                return isinstance(phase, Engaged) and phase.robot == "bravo"

            await ticks_until(tick, engaged_on_bravo)
            await tick()
            assert (await asyncio.wait_for(bravo_wire.next(), CALL_S)).opening == 0.0


async def test_the_pose_stream_goes_silent_when_frames_stop_mid_squeeze():
    """The composed deadman: an engaged hand whose headset frames age past
    the staleness window stops commanding. Losing the frames disengages the
    hand, so the release burst fires first and the stream then falls
    silent."""
    async with boot_clocked(**ONE_ROBOT) as h:
        clock = ManualClock()
        log = []
        driver = scripted_driver("backbone_inst", clock=clock, log=log)
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        measured = publish.MeasuredPoses(clock)
        peer = peer_of(h, right_arm_pose_setpoints)
        poses = {peer: POSE}
        wire = h.mocks.pairings.right_arm[0].pose_setpoints
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots_of(h)),
            period=TICK_S,
        ) as step:
            tick = reporting(step, measured, poses)
            await tick(2)
            await asyncio.wait_for(wire.next(), CALL_S)
            # The frames age past the staleness window; the measured pose
            # stays fresh: only the headset link dies.
            session.age_s = SETTINGS.stale_timeout_s + 1
            burst = await sends_until_idle(tick, clock, driver, wire)
            assert len(burst) == burst_length([at for at, _ in burst])
            assert all(
                message.position[0] == pytest.approx(0.0) for _, message in burst
            )
            assert log[-1] == ("release", "backbone_inst")
            # Fresh frames of the squeeze held through: it takes the robot
            # again, and the first message is the measured pose elsewhere.
            poses[peer] = ELSEWHERE
            session.age_s = 0.0
            await tick(2)
            first = await asyncio.wait_for(wire.next(), CALL_S)
            assert first.position[0] == pytest.approx(0.4)


async def test_releasing_the_clutch_streams_the_measured_pose_not_the_last_hand_pose():
    """The release freeze, end to end on the wire: a hand that drives the arm
    away and then lets go streams the pose the robot reports, so the
    follower stops where it is, for the burst's whole length."""
    async with boot_clocked(**ONE_ROBOT) as h:
        clock = ManualClock()
        driver = scripted_driver("backbone_inst", clock=clock)
        session = MovableSession(clock)
        session.press(right=(False, False, True))
        measured = publish.MeasuredPoses(clock)
        peer = peer_of(h, right_arm_pose_setpoints)
        poses = {peer: POSE}
        wire = h.mocks.pairings.right_arm[0].pose_setpoints
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots_of(h)),
            period=TICK_S,
        ) as step:
            tick = reporting(step, measured, poses)
            await tick(2)
            await asyncio.wait_for(wire.next(), CALL_S)
            # Drive the hand well away from the measured pose.
            session.hand_pose = pose_at((0.3, 0.0, 0.0))
            session.press(right=(False, False, True))
            await tick()
            driven = await asyncio.wait_for(wire.next(), CALL_S)
            assert driven.position[0] == pytest.approx(0.3)
            # Let go: the burst carries the measured pose.
            session.press(right=(False, False, False))
            burst = await sends_until_idle(tick, clock, driver, wire)
            assert len(burst) == burst_length([at for at, _ in burst])
            assert all(
                message.position[0] == pytest.approx(0.0) for _, message in burst
            )
            # A new squeeze at a measured pose elsewhere gives the next
            # message: the stream was silent after the burst.
            poses[peer] = ELSEWHERE
            session.press(right=(False, False, True))
            await tick(2)
            following = await asyncio.wait_for(wire.next(), CALL_S)
            assert following.position[0] == pytest.approx(0.4)


def test_a_failed_pose_states_subscribe_is_loud_and_fatal_to_the_drain(capsys):
    # Same seam as the failed declaration above: a refusing runner cannot be
    # scripted over the real wire.
    class ExplodingTopic:
        async def subscribe(self, _runner):
            raise RuntimeError("no transport")

    async def run():
        await asyncio.wait_for(
            publish.drain_pose_states(
                None, ExplodingTopic(), publish.MeasuredPoses(), "right", FakeToken()
            ),
            1.0,
        )

    asyncio.run(run())
    assert "pose_states subscribe failed" in capsys.readouterr().out


async def test_drain_pose_states_keeps_the_holder_at_the_newest_measured_pose():
    async with boot(right_arm_instances=1) as h:
        holder = publish.MeasuredPoses()
        peer = peer_of(h, right_arm_pose_setpoints)
        async with running_drain(
            lambda token: publish.drain_pose_states(
                h.node_runner, right_arm_pose_states, holder, "right", token
            )
        ):
            await h.mocks.pairings.right_arm[0].pose_states.publish(
                right_arm_pose_states.Message(
                    timestamp=1.0,
                    frame=ROBOT_FRAME,
                    position=[0.1, 0.2, 0.3],
                    orientation=[0.0, 0.0, 0.0, 1.0],
                )
            )
            await eventually(
                lambda: holder.fresh(peer, 10.0) is not None, message="the first pose"
            )
            await h.mocks.pairings.right_arm[0].pose_states.publish(
                right_arm_pose_states.Message(
                    timestamp=2.0,
                    frame=ROBOT_FRAME,
                    position=[0.4, 0.5, 0.6],
                    orientation=[0.0, 0.0, 0.0, 1.0],
                )
            )
            await eventually(
                lambda: (
                    (pose := holder.fresh(peer, 10.0)) is not None
                    and pose.value.position.position[0] == 0.4
                ),
                message="the newest pose",
            )
            # The measured pose keeps the frame and the time of its report.
            assert holder.fresh(peer, 10.0) == Stamped(
                ROBOT_FRAME, 2.0, pose_at((0.4, 0.5, 0.6))
            )


def test_drain_pose_states_survives_garbage():
    # Kept on a stub: every decodable wire message builds a Pose (the schema
    # carries float lists), so the unusable-pose latch is reachable only
    # through a hand-built message the generated modules cannot produce.
    good = SimpleNamespace(
        timestamp=1.0,
        frame=ROBOT_FRAME,
        position=[0.1, 0.2, 0.3],
        orientation=[0, 0, 0, 1],
    )
    bad = SimpleNamespace(
        timestamp=1.0, frame=ROBOT_FRAME, position=None, orientation=None
    )
    # A pose that names no frame places nowhere.
    frameless = SimpleNamespace(
        timestamp=1.0, frame="", position=[0.9, 0.0, 0.0], orientation=[0, 0, 0, 1]
    )
    # The peer is the key the holder files the pose under.
    producer = "backbone_inst"
    holder = publish.MeasuredPoses()
    token = FakeToken()

    async def run():
        drain = asyncio.create_task(
            publish.drain_pose_states(
                None,
                StubTopic([(producer, bad), (producer, good), (producer, frameless)]),
                holder,
                "right",
                token,
            )
        )
        await asyncio.sleep(0.05)
        token.cancel()
        await asyncio.wait_for(drain, 1.0)

    asyncio.run(run())
    pose = holder.fresh(producer, 10.0)
    assert pose is not None
    assert pose.frame == ROBOT_FRAME
    assert pose.value.position.position[0] == 0.1


# The release burst and the grip stream, pure


class TestReleaseHold:
    # Where the clutch drives the arm, and the measured pose the burst holds.
    TARGET = Setpoint(ROBOT_FRAME, pose_at((0.1, 0.2, 0.3)))
    HELD = Setpoint(ROBOT_FRAME, pose_at((0.4, 0.5, 0.6)))

    def drive(self, hold, now_s):
        return hold.step(target=self.TARGET, held=None, now_s=now_s, pair_present=True)

    def release(self, hold, held, now_s, *, pair_present=True):
        return hold.step(target=None, held=held, now_s=now_s, pair_present=pair_present)

    def burst(self, hold, times):
        """The values the hold streams at each of `times` with nothing
        driving it."""
        return [self.release(hold, self.HELD, now) for now in times]

    def by_the_rule(self, times):
        """The values a burst whose ticks come at `times` sends: the held
        value for its length, then silence."""
        sent = burst_length(times)
        return [self.HELD] * sent + [None] * (len(times) - sent)

    def test_a_driving_tick_streams_the_target_itself(self):
        hold = publish.ReleaseHold()
        assert self.drive(hold, 10.0) is self.TARGET
        assert not hold.running

    def test_the_burst_sends_until_three_sends_and_a_quarter_second(self):
        times = [k * TICK_S for k in range(40)]
        hold = publish.ReleaseHold()
        self.drive(hold, -TICK_S)
        assert self.burst(hold, times) == self.by_the_rule(times)
        assert not hold.running

    def test_the_burst_at_a_slow_rate_ends_after_three_sends(self):
        period_s = 1 / 3
        times = [k * period_s for k in range(6)]
        hold = publish.ReleaseHold()
        self.drive(hold, -period_s)
        assert burst_length(times) == publish.RELEASE_HOLD_MESSAGES
        assert self.burst(hold, times) == self.by_the_rule(times)

    def test_the_send_at_a_quarter_second_is_the_bursts_last(self):
        times = [0.0, 0.1, 0.2, publish.RELEASE_HOLD_S, 0.3]
        hold = publish.ReleaseHold()
        self.drive(hold, -0.1)
        assert self.burst(hold, times) == self.by_the_rule(times)
        assert self.by_the_rule(times)[3] == self.HELD

    def test_release_without_prior_driving_stays_silent(self):
        hold = publish.ReleaseHold()
        assert self.release(hold, self.HELD, 10.0) is None
        assert not hold.running

    def test_with_nothing_to_hold_no_burst_starts(self):
        # Nothing truthful to freeze on: the follower's pose is unknown.
        hold = publish.ReleaseHold()
        self.drive(hold, 9.9)
        assert self.release(hold, None, 10.0) is None
        assert not hold.running
        # And the edge is consumed: a pose arriving later does not revive it.
        assert self.release(hold, self.HELD, 10.1) is None

    def test_redriving_voids_the_pending_burst(self):
        hold = publish.ReleaseHold()
        self.drive(hold, 0.0)
        assert self.release(hold, self.HELD, TICK_S) == self.HELD
        assert self.drive(hold, 2 * TICK_S) is self.TARGET
        assert not hold.running
        # A fresh burst, counted from the release after the driving.
        times = [(3 + k) * TICK_S for k in range(40)]
        assert self.burst(hold, times) == self.by_the_rule(times)

    def test_a_voided_burst_cannot_resume_from_the_stale_pose(self):
        # Driving again voids the pending burst. With no fresh measured pose
        # at the next release there is nothing to freeze on, so the stream
        # is silent and the earlier burst's pose does not come back.
        hold = publish.ReleaseHold()
        self.drive(hold, 9.9)
        assert self.release(hold, self.HELD, 10.0) == self.HELD
        self.drive(hold, 10.05)
        assert self.release(hold, None, 10.1) is None, (
            "the voided burst resumed streaming a pose the robot may have left"
        )
        assert self.release(hold, self.HELD, 10.15) is None

    def test_the_burst_ends_at_the_tick_its_pair_left(self):
        hold = publish.ReleaseHold()
        self.drive(hold, 9.9)
        assert self.release(hold, self.HELD, 10.0) == self.HELD
        assert self.release(hold, self.HELD, 10.01, pair_present=False) is None
        assert not hold.running
        assert self.release(hold, self.HELD, 10.02) is None


ALPHA_ARM = Peer("alpha arm")
BRAVO_ARM = Peer("bravo arm")
ALPHA_ARM_PINS = frozenset({Pin(Slot.ARM, ALPHA_ARM)})
BOTH_ARMS = frozenset({ALPHA_ARM, BRAVO_ARM})


class TestGripStream:
    ENGAGED = Engaged("alpha", ALPHA_ARM_PINS, 1)
    LINGERING = Lingering("alpha", ALPHA_ARM_PINS, 1)

    def step(self, stream, phase, now_s, *, target=None, peers=BOTH_ARMS):
        return stream.step(phase, target=target, held="held", peers=peers, now_s=now_s)

    def test_a_target_goes_to_the_engaged_phases_pin(self):
        stream = publish.GripStream[str](Slot.ARM)
        step = self.step(stream, self.ENGAGED, 0.0, target="target")
        assert step == publish.StreamStep((ALPHA_ARM, "target"), False)

    def test_a_target_for_a_phase_not_engaged_fails_loudly(self):
        for phase in (self.LINGERING, Idle()):
            stream = publish.GripStream[str](Slot.ARM)
            with pytest.raises(AssertionError, match="a target drives an engaged"):
                self.step(stream, phase, 0.0, target="target")

    def test_the_burst_goes_to_the_pin_last_driven_and_reports_running(self):
        stream = publish.GripStream[str](Slot.ARM)
        self.step(stream, self.ENGAGED, 0.0, target="target")
        step = self.step(stream, self.LINGERING, TICK_S)
        assert step == publish.StreamStep((ALPHA_ARM, "held"), True)
        # The pin left the slot: the burst ends at this tick.
        step = self.step(stream, self.LINGERING, 2 * TICK_S, peers={BRAVO_ARM})
        assert step == publish.StreamStep(None, False)

    def test_the_burst_reports_running_through_its_last_value(self):
        stream = publish.GripStream[str](Slot.ARM)
        self.step(stream, self.ENGAGED, -TICK_S, target="target")
        times = [k * TICK_S for k in range(40)]
        steps = [self.step(stream, self.LINGERING, now) for now in times]
        sent = burst_length(times)
        assert steps[:sent] == [publish.StreamStep((ALPHA_ARM, "held"), True)] * sent
        assert steps[sent] == publish.StreamStep(None, False)

    def test_a_target_to_another_pin_while_the_burst_runs_fails_loudly(self):
        stream = publish.GripStream[str](Slot.ARM)
        self.step(stream, self.ENGAGED, 0.0, target="target")
        self.step(stream, self.LINGERING, TICK_S)
        on_bravo = Engaged("bravo", frozenset({Pin(Slot.ARM, BRAVO_ARM)}), 2)
        with pytest.raises(AssertionError, match="the burst goes to the pin last"):
            self.step(stream, on_bravo, 2 * TICK_S, target="target")

    def test_a_phase_with_no_pin_on_the_slot_sends_nothing(self):
        gripper_only = Engaged("alpha", frozenset({Pin(Slot.GRIPPER, Peer("g"))}), 1)
        for phase in (gripper_only, Idle()):
            stream = publish.GripStream[str](Slot.ARM)
            assert self.step(stream, phase, 0.0) == publish.StreamStep(None, False)


# Takes, interruptions and bursts, on the wire


async def test_a_squeeze_with_no_pair_or_no_fresh_pose_takes_nothing():
    async with boot_clocked(**ONE_ROBOT) as h:
        clock = ManualClock()
        log = []
        driver = scripted_driver("backbone_inst", clock=clock, log=log)
        session = ScriptedHeadset(clock)
        # The left hand has no pair on the robot; the right hand has one,
        # with no measured pose of its arm.
        session.press(left=(False, False, True), right=(False, False, True))
        measured = publish.MeasuredPoses(clock)
        robots = robots_of(h)
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots, "left"),
            pose_stream(h, clock, driver, session, measured, robots, "right"),
            period=TICK_S,
        ) as step:
            await step(5)
            assert log == []
            # The arm reports: the right hand's squeeze takes the robot.
            measured.set(peer_of(h, right_arm_pose_setpoints), reported(POSE))
            await step(2)
            assert log == [("take", "backbone_inst"), ("stop", "backbone_inst")]


async def test_a_squeeze_on_another_robot_in_the_burst_takes_it_once_the_burst_ends():
    async with boot_clocked(**TWO_ROBOTS) as h:
        clock = ManualClock()
        log = []
        driver = scripted_driver("alpha", "bravo", clock=clock, log=log)
        robots = robots_of(h)
        measured = publish.MeasuredPoses(clock)
        poses = {
            peer: POSE
            for peer in peers_by_robot(h, right_arm_pose_setpoints).values()
        }
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots),
            period=TICK_S,
        ) as step:
            tick = reporting(step, measured, poses)
            await tick(2)
            assert isinstance(driver.world.grips.right.phase, Engaged)
            # Release, choose bravo and squeeze again inside alpha's burst.
            session.press(right=(False, False, False))
            await tick()
            assert robots.choose("bravo")
            await tick()
            session.press(right=(False, False, True))
            # Alpha keeps its hold until its burst ends.
            for _ in range(WIRE_TICKS):
                if not isinstance(driver.world.grips.right.phase, Lingering):
                    break
                assert driver.world.robot("alpha").hold is not None
                assert ("release", "alpha") not in log
                await tick()

            def engaged_on_bravo():
                phase = driver.world.grips.right.phase
                return isinstance(phase, Engaged) and phase.robot == "bravo"

            await ticks_until(tick, engaged_on_bravo)
        assert driver.world.robot("alpha").hold is None
        assert log.index(("release", "alpha")) < log.index(("take", "bravo"))


async def a_take_interrupted_by(change) -> tuple[list, Driver]:
    """A squeeze on alpha whose take answers only once `change(robots,
    session)` ran and a tick read it: the calls the driver made, and the
    driver."""
    async with boot_clocked(**TWO_ROBOTS) as h:
        clock = ManualClock()
        log = []
        calls = FakeCalls(log)
        driver = scripted_driver("alpha", "bravo", clock=clock, log=log, calls=calls)
        takes = calls.takes_on_cue()
        robots = robots_of(h)
        assert robots.choose("alpha")
        measured = publish.MeasuredPoses(clock)
        for peer in peers_by_robot(h, right_arm_pose_setpoints).values():
            measured.set(peer, reported(POSE))
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        async with parked(
            clock,
            pose_stream(h, clock, driver, session, measured, robots),
            period=TICK_S,
        ) as step:
            await step()
            assert takes.unanswered() == [0]
            change(robots, session)
            # A tick reads the change before the take answers.
            await step()
            takes.answer(0, AGREED)
            await settle()
        return log, driver


async def test_a_robot_chosen_during_the_take_stops_nothing_and_releases_the_lease():
    log, driver = await a_take_interrupted_by(
        lambda robots, _session: robots.choose("bravo")
    )
    assert log == [("take", "alpha"), ("release", "alpha")]
    assert driver.world.robot("alpha").hold is None


async def test_a_frame_lost_during_the_take_stops_nothing_and_releases_the_lease():
    log, driver = await a_take_interrupted_by(lambda _robots, session: session.press())
    assert log == [("take", "alpha"), ("release", "alpha")]
    assert driver.world.robot("alpha").hold is None


# The order of a stream's step and its send, with no wire


@dataclass
class Unwired:
    """The right hand's pose stream on a slot with no wire, ticked by the
    test: the driver, its calls in one log, the slot and the session."""

    driver: Driver
    log: list
    arm: NoWireSlot
    session: ScriptedHeadset
    clock: ManualClock
    measured: publish.MeasuredPoses

    def stream(self):
        """The stream, as a function of a cancellation token."""
        gripper = NoWireSlot()
        robots = PairedRobots(
            None,
            arms={Hand.RIGHT: self.arm},
            grippers={Hand.RIGHT: gripper},
            base=NoWireSlot(),
        )
        tick = publish.pose_tick(
            topic_module=self.arm,
            gripper_module=gripper,
            handedness=Hand.RIGHT,
            driver=self.driver,
            clutch=HandClutch(1.0),
            session=self.session,
            measured=self.measured,
            robots=robots,
            settings=SETTINGS,
            clock=self.clock,
            timestamp_s=lambda: 0.0,
        )
        return lambda token: publish.run_stream(
            None, self.arm, TICK_S, "right arm", token, tick, self.clock
        )

    async def tick(self):
        """One tick of the stream, with the arm's pose reported first."""
        self.measured.set(ALPHA_ARM, reported(POSE))
        await self.clock.advance(TICK_S)

    def positions(self) -> list:
        """The position of each setpoint sent, in order."""
        return [fields[2] for _pin, fields in self.arm.publisher.sent]


def unwired() -> Unwired:
    clock = ManualClock()
    log = []
    session = ScriptedHeadset(clock)
    session.press(right=(False, False, True))
    return Unwired(
        scripted_driver("alpha", clock=clock, log=log),
        log,
        NoWireSlot(("alpha", ALPHA_ARM)),
        session,
        clock,
        publish.MeasuredPoses(clock),
    )


async def test_the_release_call_starts_after_the_last_burst_values_send_returned():
    r = unwired()
    async with running(r.stream()):
        # The squeeze takes the robot, and the next tick drives it.
        await r.tick()
        await r.tick()
        assert len(r.arm.publisher.sent) == 1
        session = r.session
        session.press(right=(False, False, False))
        # Tick the burst up to its last value.
        burst_times = []
        for _ in range(WIRE_TICKS):
            burst_times.append(r.clock.monotonic() + TICK_S)
            if ends_the_burst(burst_times):
                break
            await r.tick()
        # The burst's last value goes out, and its send waits.
        r.arm.publisher.gate.clear()
        await r.tick()
        assert len(r.arm.publisher.sent) == 1 + len(burst_times)
        assert ("release", "alpha") not in r.log
        r.arm.publisher.gate.set()
        await settle()
        assert ("release", "alpha") not in r.log
        # The next tick drops the grip, and its release goes out.
        await r.tick()
        assert r.log[-1] == ("release", "alpha")


async def test_a_let_go_inside_a_send_still_gets_its_burst():
    r = unwired()
    async with running(r.stream()):
        await r.tick()
        await r.tick()
        # The hand moves away, and that tick's send waits.
        r.session.press(pose=pose_at((0.3, 0.0, 0.0)), right=(False, False, True))
        r.arm.publisher.gate.clear()
        await r.tick()
        r.driver.fleet_changed(fleet_view(["alpha"], halted_by=HALTER))
        r.arm.publisher.gate.set()
        await settle()
        await r.tick()
    # The tick after the let-go sends the measured pose to the pin.
    assert r.positions()[-2:] == [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert r.arm.publisher.sent[-1][0] == ALPHA_ARM
