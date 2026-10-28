import asyncio
import contextlib
import time
from types import SimpleNamespace

import numpy as np
import pytest
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
    POSE,
    FakeCalls,
    FakeStops,
    FakeSubscription,
    FakeToken,
    ManualClock,
    ScriptedHeadset,
    boot,
    boot_clocked,
    default_settings,
    eventually,
    following_fleet,
    peer_of,
    robots_of,
    running,
    running_drain,
    scripted_driver,
    settle,
)
from xr_commander import publish
from xr_commander.bus import REAL_CLOCK, FlashNote
from xr_commander.clutch import HandClutch
from xr_commander.devices import (
    HandSample,
    XrFrame,
)
from xr_commander.drive import Driver, HandState
from xr_commander.fleet import Timing
from xr_commander.frames import Pose


def test_a_measured_pose_is_kept_per_peer_and_goes_stale_rather_than_lying():
    holder = publish.MeasuredPoses()
    assert holder.fresh("alpha", 10.0) is None
    assert holder.fresh(None, 10.0) is None
    holder.set("alpha", POSE)
    assert holder.fresh("alpha", 10.0) is POSE
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
            publish._run_stream(
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


async def test_the_gripper_stream_is_silent_without_the_squeeze_deadman():
    async with boot_clocked(**ONE_ROBOT) as h:
        hand = HandState(FlashNote())
        session = ScriptedHeadset()
        session.press(right=(False, False, False))
        async with running(
            lambda token: publish.stream_gripper(
                h.node_runner,
                topic_module=right_gripper_setpoints,
                handedness="right",
                hand=hand,
                session=session,
                robots=robots_of(h),
                settings=default_settings(),
                token=token,
            )
        ):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    h.mocks.pairings.right_gripper[0].gripper_setpoints.next(), 0.4
                )


async def test_the_gripper_stream_publishes_openings_while_the_hand_is_engaged():
    async with boot_clocked(**ONE_ROBOT) as h:
        hand = HandState(FlashNote(), engaged=True, robot="backbone_inst")
        session = ScriptedHeadset()
        session.press(right=(False, False, True))
        async with running(
            lambda token: publish.stream_gripper(
                h.node_runner,
                topic_module=right_gripper_setpoints,
                handedness="right",
                hand=hand,
                session=session,
                robots=robots_of(h),
                settings=default_settings(),
                token=token,
            )
        ):
            message = await asyncio.wait_for(
                h.mocks.pairings.right_gripper[0].gripper_setpoints.next(), 10.0
            )
        assert message.opening == 1.0  # released trigger rests at open_fraction
        assert message.max_effort == 0.0
        assert message.timestamp > 0.0  # stamped from the daemon-resolved clock


async def test_the_pose_stream_refuses_to_engage_without_a_fresh_measured_pose():
    async with boot_clocked(**ONE_ROBOT) as h:
        driver = scripted_driver("backbone_inst")
        session = ScriptedHeadset()
        session.press(right=(False, False, True))
        async with running(
            lambda token: publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                gripper_module=right_gripper_setpoints,
                handedness="right",
                hand=driver.hands["right"],
                driver=driver,
                clutch=HandClutch(1.0),
                session=session,
                measured=publish.MeasuredPoses(),
                robots=robots_of(h),
                settings=default_settings(),
                token=token,
            )
        ):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    h.mocks.pairings.right_arm[0].pose_setpoints.next(), 0.4
                )


async def test_the_pose_stream_engages_once_the_follower_reports():
    async with boot_clocked(**ONE_ROBOT) as h:
        driver = scripted_driver("backbone_inst")
        session = ScriptedHeadset()
        session.press(right=(False, False, True))
        measured = publish.MeasuredPoses()
        measured.set(peer_of(h, right_arm_pose_setpoints), POSE)
        async with running(
            lambda token: publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                gripper_module=right_gripper_setpoints,
                handedness="right",
                hand=driver.hands["right"],
                driver=driver,
                clutch=HandClutch(1.0),
                session=session,
                measured=measured,
                robots=robots_of(h),
                settings=default_settings(),
                token=token,
            )
        ):
            message = await asyncio.wait_for(
                h.mocks.pairings.right_arm[0].pose_setpoints.next(), 10.0
            )
        # The engage tick commands the measured pose: a no-op that starts the
        # stream.
        assert message.position == [0.0, 0.0, 0.0]
        assert message.orientation == [0.0, 0.0, 0.0, 1.0]


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


async def received_within(subscription, seconds: float) -> list:
    """Every message `subscription` receives within `seconds`."""
    received = []
    deadline = time.monotonic() + seconds
    while (left := deadline - time.monotonic()) > 0:
        try:
            received.append(await asyncio.wait_for(subscription.next(), left))
        except TimeoutError:
            break
    return received


async def test_a_robot_change_mid_squeeze_releases_the_robot_left_behind_alone():
    async with boot_clocked(**TWO_ROBOTS) as h:
        driver = scripted_driver("alpha", "bravo")
        robots = robots_of(h)
        peers = peers_by_robot(h, right_arm_pose_setpoints)
        alpha_wire, bravo_wire = (
            mock.pose_setpoints for mock in h.mocks.pairings.right_arm
        )
        measured = publish.MeasuredPoses()
        measured.set(peers["alpha"], POSE)
        measured.set(peers["bravo"], Pose(np.array([0.5, 0.0, 0.0]), POSE.orientation))
        session = MovableSession()
        settings = default_settings(command_rate_hz=200)
        session.press(right=(False, False, True))
        async with running(
            lambda token: publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                gripper_module=right_gripper_setpoints,
                handedness="right",
                hand=driver.hands["right"],
                driver=driver,
                clutch=HandClutch(1.0),
                session=session,
                measured=measured,
                robots=robots,
                settings=settings,
                token=token,
            )
        ):
            # Alpha, paired first, is driven: hand moved 0.3 away.
            await asyncio.wait_for(alpha_wire.next(), 10.0)
            session.hand_pose = Pose(np.array([0.3, 0.0, 0.0]), POSE.orientation)
            session.press(right=(False, False, True))
            while (await asyncio.wait_for(alpha_wire.next(), 10.0)).position[0] < 0.29:
                pass
            assert robots.choose("bravo")
            # Alpha gets the whole release burst at its own measured pose.
            tail = await received_within(alpha_wire, publish.RELEASE_HOLD_S + 1.0)
            frozen = [m for m in tail if m.position[0] == pytest.approx(0.0)]
            assert len(frozen) >= publish.RELEASE_HOLD_MESSAGES
            assert {round(m.position[0], 3) for m in tail} <= {0.0, 0.3}
            # Bravo hears nothing while the squeeze is held over.
            assert await received_within(bravo_wire, 0.4) == []
            # Let go and squeeze again: bravo engages at its own pose.
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(False, False, True))
            first = await asyncio.wait_for(bravo_wire.next(), 10.0)
            assert first.position[0] == pytest.approx(0.5)


async def test_a_grip_held_across_a_robot_change_leaves_the_new_gripper_alone():
    async with boot_clocked(**TWO_ROBOTS) as h:
        driver = scripted_driver("alpha", "bravo")
        robots = robots_of(h)
        alpha_wire, bravo_wire = (
            mock.gripper_setpoints for mock in h.mocks.pairings.right_gripper
        )
        measured = publish.MeasuredPoses()
        for peer in peers_by_robot(h, right_arm_pose_setpoints).values():
            measured.set(peer, POSE)
        session = ScriptedHeadset()
        session.press(right=(False, False, True))
        settings = default_settings()
        async with (
            running(
                lambda token: publish.stream_pose(
                    h.node_runner,
                    topic_module=right_arm_pose_setpoints,
                    gripper_module=right_gripper_setpoints,
                    handedness="right",
                    hand=driver.hands["right"],
                    driver=driver,
                    clutch=HandClutch(1.0),
                    session=session,
                    measured=measured,
                    robots=robots,
                    settings=settings,
                    token=token,
                )
            ),
            running(
                lambda token: publish.stream_gripper(
                    h.node_runner,
                    topic_module=right_gripper_setpoints,
                    handedness="right",
                    hand=driver.hands["right"],
                    session=session,
                    robots=robots,
                    settings=settings,
                    token=token,
                )
            ),
        ):
            await asyncio.wait_for(alpha_wire.next(), 10.0)
            assert robots.choose("bravo")
            assert await received_within(bravo_wire, 0.4) == []
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(False, False, True))
            await asyncio.wait_for(bravo_wire.next(), 10.0)


class FrozenSession(ScriptedHeadset):
    """A session whose frames stop aging forward: presses stamp once."""

    def __init__(self):
        super().__init__()
        self._stamp = time.monotonic()

    def press(self, **hands):
        super().press(**hands)
        self._stamp = time.monotonic()

    def latest(self):
        return XrFrame(received_monotonic_s=self._stamp, hands=self.hands)


async def test_the_pose_stream_goes_silent_when_frames_stop_mid_squeeze():
    """The composed deadman: an engaged hand whose headset frames age past
    the staleness window stops commanding. Losing the frames disengages the
    hand, so the release burst fires first and the stream then falls
    permanently silent."""
    async with boot_clocked(**ONE_ROBOT) as h:
        driver = scripted_driver("backbone_inst")
        session = FrozenSession()
        session.press(right=(False, False, True))
        measured = publish.MeasuredPoses()
        settings = default_settings(stale_timeout_s=0.05)
        token = FakeToken()

        # Measured stays fresh throughout: only the headset link dies.
        async def keep_measured_fresh():
            while not token.is_cancelled():
                measured.set(peer_of(h, right_arm_pose_setpoints), POSE)
                await asyncio.sleep(0.01)

        refresher = asyncio.create_task(keep_measured_fresh())
        task = asyncio.create_task(
            publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                gripper_module=right_gripper_setpoints,
                handedness="right",
                hand=driver.hands["right"],
                driver=driver,
                clutch=HandClutch(1.0),
                session=session,
                measured=measured,
                robots=robots_of(h),
                settings=settings,
                token=token,
            )
        )
        subscription = h.mocks.pairings.right_arm[0].pose_setpoints
        try:
            # Streamed while the frame was fresh.
            await asyncio.wait_for(subscription.next(), 10.0)
            # The frozen frame ages past 0.05 s. Drain the in-flight samples
            # and the release burst until a window longer than the burst
            # passes with nothing published: silence is the deadman.
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    await asyncio.wait_for(
                        subscription.next(), publish.RELEASE_HOLD_S + 0.1
                    )
                except TimeoutError:
                    break
                assert time.monotonic() < deadline, (
                    "the stream kept publishing after the headset frames went stale"
                )
        finally:
            token.cancel()
            task.cancel()
            refresher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            with contextlib.suppress(asyncio.CancelledError):
                await refresher


class MovableSession(ScriptedHeadset):
    """A session whose hand pose can be moved, so a driven target is
    distinguishable on the wire from the measured pose."""

    def __init__(self):
        super().__init__()
        self.hand_pose = POSE

    def press(self, **hands):
        super().press(**hands)
        self.hands = {
            side: HandSample(
                pose=self.hand_pose,
                squeezing=sample.squeezing,
                trigger=sample.trigger,
                primary_button=sample.primary_button,
                secondary_button=sample.secondary_button,
            )
            for side, sample in self.hands.items()
        }


async def test_releasing_the_clutch_streams_the_measured_pose_not_the_last_hand_pose():
    """The release freeze, end to end on the wire: a hand that drives the arm
    away and then lets go must stream the pose the robot actually reports, so
    the follower stops where it is instead of walking to the last hand
    target."""
    async with boot_clocked(**ONE_ROBOT) as h:
        driver = scripted_driver("backbone_inst")
        session = MovableSession()
        measured = publish.MeasuredPoses()
        measured.set(peer_of(h, right_arm_pose_setpoints), POSE)
        settings = default_settings(command_rate_hz=200)
        token = FakeToken()

        session.press(right=(False, False, True))
        task = asyncio.create_task(
            publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                gripper_module=right_gripper_setpoints,
                handedness="right",
                hand=driver.hands["right"],
                driver=driver,
                clutch=HandClutch(1.0),
                session=session,
                measured=measured,
                robots=robots_of(h),
                settings=settings,
                token=token,
            )
        )
        subscription = h.mocks.pairings.right_arm[0].pose_setpoints
        try:
            await asyncio.wait_for(subscription.next(), 10.0)
            # Drive the hand well away from the measured pose, and wait until
            # that displaced target is the one on the wire.
            driven = Pose(np.array([0.3, 0.0, 0.0]), POSE.orientation)
            session.hand_pose = driven
            session.press(right=(False, False, True))
            deadline = time.monotonic() + 10.0
            while True:
                message = await asyncio.wait_for(subscription.next(), 10.0)
                if message.position[0] == pytest.approx(0.3):
                    break
                assert time.monotonic() < deadline, (
                    "the driven target never reached the wire"
                )

            # Let go. The freeze burst must carry the measured pose, once the
            # samples the stream had already published when the clutch
            # released have passed: the wire holds them in order, so how many
            # arrive first is a matter of the host's speed, not of behavior.
            # A stream that walks on to the last hand target instead reaches
            # the end of its burst without ever sending the measured pose, and
            # the wait for one times out.
            session.press(right=(False, False, False))
            try:
                while True:
                    frozen = await asyncio.wait_for(subscription.next(), 10.0)
                    if frozen.position[0] == pytest.approx(0.0):
                        break
                    assert frozen.position[0] == pytest.approx(0.3), (
                        "only the driven target may precede the freeze on the wire"
                    )
            except TimeoutError:
                raise AssertionError(
                    "release streamed the last hand target instead of the measured pose"
                ) from None

            # And the burst is finite: the stream falls silent after it.
            with pytest.raises(asyncio.TimeoutError):
                deadline = time.monotonic() + publish.RELEASE_HOLD_S + 5.0
                while time.monotonic() < deadline:
                    await asyncio.wait_for(subscription.next(), 1.0)
        finally:
            token.cancel()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


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
                    position=[0.4, 0.5, 0.6],
                    orientation=[0.0, 0.0, 0.0, 1.0],
                )
            )
            await eventually(
                lambda: (
                    (pose := holder.fresh(peer, 10.0)) is not None
                    and pose.position[0] == 0.4
                ),
                message="the newest pose",
            )


def test_drain_pose_states_survives_garbage():
    # Kept on a stub: every decodable wire message builds a Pose (the schema
    # carries float lists), so the unusable-pose latch is reachable only
    # through a hand-built message the generated modules cannot produce.

    class StubTopic(FakeSubscription):
        async def subscribe(self, _runner):
            return self

    good = SimpleNamespace(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1])
    bad = SimpleNamespace(position=None, orientation=None)
    # The peer is the key the holder files the pose under.
    producer = "backbone_inst"
    holder = publish.MeasuredPoses()
    token = FakeToken()

    async def run():
        drain = asyncio.create_task(
            publish.drain_pose_states(
                None,
                StubTopic([(producer, bad), (producer, good)]),
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
    assert pose.position[0] == 0.1


# Fast enough that the burst's message floor never sets the hold length.
FAST_TICK_S = 0.001


class TestReleaseHold:
    def make_pose(self):
        return Pose(
            position=np.array([0.1, 0.2, 0.3]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )

    def drive(self, hold, target, now_s):
        return hold.step(target=target, measured_ee=None, now_s=now_s)

    def release(self, hold, measured, now_s):
        return hold.step(target=None, measured_ee=measured, now_s=now_s)

    def test_a_driving_tick_streams_the_clutch_target_itself(self):
        hold = publish.ReleaseHold(FAST_TICK_S, hold_s=0.25)
        target = self.make_pose()
        assert self.drive(hold, target, 10.0) is target

    def test_release_after_driving_streams_the_measured_pose_for_the_hold(self):
        hold = publish.ReleaseHold(FAST_TICK_S, hold_s=0.25)
        measured = self.make_pose()
        self.drive(hold, self.make_pose(), 9.9)
        assert self.release(hold, measured, 10.0) is measured
        assert self.release(hold, measured, 10.2) is measured
        assert self.release(hold, measured, 10.26) is None

    def test_release_without_prior_driving_stays_silent(self):
        hold = publish.ReleaseHold(FAST_TICK_S, hold_s=0.25)
        assert self.release(hold, self.make_pose(), 10.0) is None

    def test_release_with_stale_measured_stays_silent(self):
        # Nothing truthful to freeze on: the follower's pose is unknown.
        hold = publish.ReleaseHold(FAST_TICK_S, hold_s=0.25)
        self.drive(hold, self.make_pose(), 9.9)
        assert self.release(hold, None, 10.0) is None
        # And the edge is consumed: a pose arriving later does not revive it.
        assert self.release(hold, self.make_pose(), 10.1) is None

    def test_redriving_voids_the_pending_burst(self):
        hold = publish.ReleaseHold(FAST_TICK_S, hold_s=0.25)
        measured = self.make_pose()
        target = self.make_pose()
        self.drive(hold, target, 9.9)
        assert self.release(hold, measured, 10.0) is measured
        self.drive(hold, target, 10.05)
        assert self.release(hold, measured, 10.1) is measured, "a fresh edge re-arms"
        # Mid-burst re-engage then release again: the burst restarts from now.
        self.drive(hold, target, 10.15)
        assert self.release(hold, measured, 10.2) is measured
        assert self.release(hold, measured, 10.44) is measured
        assert self.release(hold, measured, 10.46) is None

    def test_a_voided_burst_cannot_resume_from_the_stale_pose(self):
        # Re-engaging must void the pending burst, not just re-arm it. If the
        # hand then disengages with no fresh measured pose there is nothing
        # to freeze on, so the stream must be silent rather than resuming the
        # earlier burst's now-stale pose.
        hold = publish.ReleaseHold(FAST_TICK_S, hold_s=0.25)
        measured = self.make_pose()
        self.drive(hold, self.make_pose(), 9.9)
        assert self.release(hold, measured, 10.0) is measured
        self.drive(hold, self.make_pose(), 10.05)
        assert self.release(hold, None, 10.1) is None, (
            "the voided burst resumed streaming a pose the robot may have left"
        )

    def test_the_burst_ends_exactly_at_expiry(self):
        # The window is half-open: the expiry instant itself is silence, so a
        # tick landing exactly on it cannot double the burst's stated length.
        hold = publish.ReleaseHold(FAST_TICK_S, hold_s=0.25)
        measured = self.make_pose()
        self.drive(hold, self.make_pose(), 9.9)
        assert self.release(hold, measured, 10.0) is measured
        assert self.release(hold, measured, 10.25) is None

    def test_a_slow_command_rate_still_gets_a_burst_not_one_message(self):
        # At 4 Hz the 0.25 s window would carry a single sample, which is the
        # one thing a burst exists to avoid depending on.
        slow_tick_s = 0.25
        hold = publish.ReleaseHold(slow_tick_s, hold_s=0.25)
        measured = self.make_pose()
        self.drive(hold, self.make_pose(), 9.9)
        held = [
            self.release(hold, measured, 10.0 + tick * slow_tick_s) is measured
            for tick in range(publish.RELEASE_HOLD_MESSAGES)
        ]
        assert all(held), "the burst shrank to fewer messages than its floor"


def frame_of(squeezing: bool) -> HandSample:
    return HandSample(
        pose=POSE,
        squeezing=squeezing,
        trigger=0.0,
        primary_button=False,
        secondary_button=False,
    )


def gate_engages(hand: HandState, chosen, sample) -> bool:
    """One tick of the hand's gate: whether its squeeze asks to engage."""
    hand.observe(chosen, sample)
    return hand.may_engage(sample)


def test_the_first_robot_engages_a_squeeze_already_held():
    assert gate_engages(HandState(FlashNote()), "alpha", frame_of(True))


def test_a_squeeze_held_across_any_change_of_robot_waits_for_a_fresh_release():
    hand = HandState(FlashNote())
    assert gate_engages(hand, "charlie", frame_of(True))
    assert gate_engages(hand, "charlie", frame_of(True))
    # Charlie pairs no left arm: the gate follows the robot, not the pair.
    assert not gate_engages(hand, "alpha", frame_of(True))
    # A stale tick says nothing about the grip.
    assert not gate_engages(hand, "alpha", None)
    assert not gate_engages(hand, "alpha", frame_of(True))
    assert hand.wait_release
    assert not gate_engages(hand, "alpha", frame_of(False))
    assert gate_engages(hand, "alpha", frame_of(True))
    # Gone and back through no robot at all: still a change.
    assert not gate_engages(hand, None, frame_of(True))
    assert not gate_engages(hand, "bravo", frame_of(True))


@contextlib.asynccontextmanager
async def parked(clock, *streams):
    """`streams` running on `clock`: yields a step that runs `count`
    iterations of every stream, each to its end."""
    period = default_settings().tick_period_s
    async with running(*streams) as tasks:

        async def step(count=1):
            for _ in range(count):
                await clock.advance(period, parked=tasks)

        yield step


def pose_stream(h, clock, driver, session, measured, robots, handedness="right"):
    """The pose stream of one hand, on `clock`."""
    arm, gripper = {
        "left": (left_arm_pose_setpoints, left_gripper_setpoints),
        "right": (right_arm_pose_setpoints, right_gripper_setpoints),
    }[handedness]
    return lambda token: publish.stream_pose(
        h.node_runner,
        topic_module=arm,
        gripper_module=gripper,
        handedness=handedness,
        hand=driver.hands[handedness],
        driver=driver,
        clutch=HandClutch(1.0),
        session=session,
        measured=measured,
        robots=robots,
        settings=default_settings(),
        token=token,
        clock=clock,
    )


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
        ) as step:
            await step(5)
            assert log == []
            # The arm reports: the right hand's squeeze takes the robot.
            measured.set(peer_of(h, right_arm_pose_setpoints), POSE)
            await step(2)
            assert log == [("take", "backbone_inst"), ("stop", "backbone_inst")]


async def test_the_gripper_is_silent_while_its_hand_is_in_its_release_burst():
    async with boot_clocked(**ONE_ROBOT) as h:
        clock = ManualClock()
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        # The hand holds the robot through its burst, no longer engaged.
        hand = HandState(FlashNote(clock), robot="backbone_inst")
        gripper = h.mocks.pairings.right_gripper[0].gripper_setpoints

        def stream(token):
            return publish.stream_gripper(
                h.node_runner,
                topic_module=right_gripper_setpoints,
                handedness="right",
                hand=hand,
                session=session,
                robots=robots_of(h),
                settings=default_settings(),
                token=token,
                clock=clock,
            )

        async with parked(clock, stream) as step:
            await step(5)
            # Engaged again with the trigger pulled: the first opening on the
            # wire is the closed one, so nothing went out in the burst.
            session.trigger = 1.0
            session.press(right=(False, False, True))
            hand.engaged = True
            await step()
            message = await asyncio.wait_for(gripper.next(), 10.0)
            assert message.opening == 0.0


async def test_a_change_of_robot_in_the_release_burst_releases_the_robot_left_behind():
    async with boot_clocked(**TWO_ROBOTS) as h:
        clock = ManualClock()
        log = []
        driver = scripted_driver("alpha", "bravo", clock=clock, log=log)
        robots = robots_of(h)
        measured = publish.MeasuredPoses(clock)
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        async with parked(
            clock, pose_stream(h, clock, driver, session, measured, robots)
        ) as step:

            async def report():
                for peer in peers_by_robot(h, right_arm_pose_setpoints).values():
                    measured.set(peer, POSE)
                await step()

            await report()
            assert driver.hand_holds("alpha")
            # Release, choose bravo and squeeze again inside alpha's burst.
            session.press(right=(False, False, False))
            await report()
            assert robots.choose("bravo")
            await report()
            session.press(right=(False, False, True))
            await report()
            assert driver.hand_holds("bravo")
            assert not driver.holds("alpha")
            assert ("release", "alpha") in log


async def a_take_interrupted_by(change) -> tuple[list, Driver]:
    """A squeeze on alpha whose take answers only once `change(robots,
    session)` ran: the calls the driver made, and the driver."""
    async with boot_clocked(**TWO_ROBOTS) as h:
        clock = ManualClock()
        log = []
        calls = FakeCalls(log)
        driver = Driver(
            following_fleet("alpha", "bravo"),
            calls,
            FakeStops(log),
            timing=Timing(),
            on_change=lambda: None,
            clock=clock,
        )
        take = calls.take
        taking = asyncio.Event()
        go_on = asyncio.Event()

        async def slow_take(robot):
            taking.set()
            await go_on.wait()
            return await take(robot)

        calls.take = slow_take
        robots = robots_of(h)
        assert robots.choose("alpha")
        measured = publish.MeasuredPoses(clock)
        for peer in peers_by_robot(h, right_arm_pose_setpoints).values():
            measured.set(peer, POSE)
        session = ScriptedHeadset(clock)
        session.press(right=(False, False, True))
        async with parked(
            clock, pose_stream(h, clock, driver, session, measured, robots)
        ) as step:
            await step()
            await taking.wait()
            change(robots, session)
            go_on.set()
            await settle()
        return log, driver


async def test_a_robot_chosen_during_the_take_stops_nothing_and_releases_the_lease():
    log, driver = await a_take_interrupted_by(
        lambda robots, _session: robots.choose("bravo")
    )
    assert log == [("take", "alpha"), ("release", "alpha")]
    assert not driver.holds("alpha")


async def test_a_frame_lost_during_the_take_stops_nothing_and_releases_the_lease():
    log, driver = await a_take_interrupted_by(lambda _robots, session: session.press())
    assert log == [("take", "alpha"), ("release", "alpha")]
    assert not driver.holds("alpha")
