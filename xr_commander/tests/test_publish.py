import asyncio
import contextlib
import time

import numpy as np
import pytest

from peppygen.consumed_actions.postures import move_to_home, move_to_ready
from peppygen.consumed_topics.color_cameras import video_stream as color_video_stream
from peppygen.paired_topics.right_arm import pose_setpoints as right_arm_pose_setpoints
from peppygen.paired_topics.right_arm import pose_states as right_arm_pose_states
from peppygen.paired_topics.right_gripper import (
    gripper_setpoints as right_gripper_setpoints,
)

from tests.helpers import (
    POSE,
    FakeSubscription,
    FakeToken,
    RecordingSink,
    boot,
    boot_clocked,
    default_parameters,
    eventually,
    press_until_goal,
    running_drain,
)
from xr_commander import config, publish
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample, XrFrame
from xr_commander.frames import Pose
from xr_commander.video import drain_frames


def test_latest_pose_goes_stale_rather_than_lying():
    holder = publish.LatestPose()
    assert holder.fresh(10.0) is None
    holder.set(POSE)
    assert holder.fresh(10.0) is POSE
    assert holder.fresh(10.0, now_monotonic=time.monotonic() + 60.0) is None


def posture_settings():
    return config.from_parameters(default_parameters())


def done(module):
    """The action's success verdict, as each completed-goal assertion uses."""
    return module.ResultResponseData(success=True, message="done")


def frame_message(width=1, height=1, encoding="bgr8", data=None):
    if data is None:
        data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
    return color_video_stream.Message(
        header=color_video_stream.MessageHeader(timestamp=0.0, frame_id=0),
        encoding=encoding,
        width=width,
        height=height,
        frame=data,
    )


async def test_drain_frames_routes_by_producer_and_skips_unknown_producers():
    async with boot(color_cameras_instances=2) as h:
        known_mock, unknown_mock = h.mocks.deps.color_cameras
        known_id = color_video_stream.bound_producers(h.node_runner)[0].instance_id
        sink = RecordingSink()
        async with running_drain(
            lambda token: drain_frames(
                h.node_runner,
                color_video_stream,
                {known_id: sink},
                token,
                "test",
            )
        ) as drain:
            # A bound producer without a sink is skipped, not fatal: the
            # known producer's frames keep landing afterwards.
            await unknown_mock.video_stream.publish(frame_message())
            await known_mock.video_stream.publish(frame_message())
            await eventually(
                lambda: len(sink.frames) == 1, message="the known producer's frame"
            )
            assert not drain.done()
            await unknown_mock.video_stream.publish(frame_message())
            await known_mock.video_stream.publish(frame_message())
            await eventually(
                lambda: len(sink.frames) == 2,
                message="the known producer's second frame",
            )


class FakeSession:
    """A scripted headset: the FrameSource seam is a device surface, not a
    generated module, so tests feed it hand states directly."""

    def __init__(self):
        self.hands = {}

    def press(self, **hands):
        self.hands = {
            side: HandSample(
                pose=POSE,
                squeezing=squeeze,
                trigger=0.0,
                primary_button=primary,
                secondary_button=secondary,
            )
            for side, (primary, secondary, squeeze) in hands.items()
        }

    def latest(self):
        return XrFrame(received_monotonic_s=time.monotonic(), hands=self.hands)


def _primary(sample):
    return sample.primary_button


def _secondary(sample):
    return sample.secondary_button


@contextlib.asynccontextmanager
async def posture_button(h, *, action_module, pressed):
    """`run_posture_button` running against the real runner; yields the
    scripted headset session."""
    session = FakeSession()
    token = FakeToken()
    task = asyncio.create_task(
        publish.run_posture_button(
            h.node_runner,
            action_module=action_module,
            pressed=pressed,
            session=session,
            settings=posture_settings(),
            token=token,
        )
    )
    try:
        yield session
    finally:
        token.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def test_the_result_wait_scales_with_long_posture_moves():
    assert publish._result_timeout_s(2.0) == 120.0
    assert publish._result_timeout_s(100.0) == 300.0


async def test_the_button_fires_once_per_rising_edge():
    async with boot(postures_instances=1) as h:
        mock = h.mocks.deps.postures[0].move_to_home
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            # The goal reached the real move_to_home producer, carrying the
            # configured posture duration.
            assert pending.request.duration_s == 2.0
            active = await pending.accept()
            # Still held: one edge, one goal.
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            await active.complete(
                done(move_to_home)
            )


async def test_squeezing_cancels_the_move_in_flight():
    async with boot(postures_instances=1) as h:
        mock = h.mocks.deps.postures[0].move_to_home
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            active = await pending.accept()
            # Taking manual control back is the cancel gesture.
            session.press(right=(False, False, True))
            await asyncio.wait_for(active.cancel_signal(), 10.0)
            await active.complete_cancelled(
                move_to_home.ResultResponseData(success=False, message="cancelled")
            )
            # The squeeze that cancelled must not have fired a fresh goal.
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)


async def test_a_press_landing_with_a_squeeze_does_not_fire():
    async with boot(postures_instances=1) as h:
        mock = h.mocks.deps.postures[0].move_to_home
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, True))
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)


async def test_a_button_already_held_at_first_tracking_does_not_fire():
    async with boot(postures_instances=1) as h:
        mock = h.mocks.deps.postures[0].move_to_home
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary
        ) as session:
            # A thumb already down on the first tracked frame is not a press.
            session.press(right=(True, False, False))
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            # An observed release and then a press is the first real edge.
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            await (await pending.accept()).complete(
                done(move_to_home)
            )


async def test_a_stale_gap_does_not_refire_a_held_button():
    """A dropout while the thumb stays down must not read as a fresh press:
    the gap makes the input unknown, not released."""
    async with boot(postures_instances=1) as h:
        mock = h.mocks.deps.postures[0].move_to_home
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            await (await pending.accept()).complete(
                done(move_to_home)
            )
            # The headset drops out while the thumb stays down...
            session.hands = {}
            await asyncio.sleep(0.05)
            # ...and comes back with the button still held: not a press.
            session.press(right=(True, False, False))
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            # A real release and press after the gap re-arms the button.
            pending = await press_until_goal(
                session, mock, {"right": (False, False, False)}, {"right": (True, False, False)}
            )
            await (await pending.accept()).complete(
                done(move_to_home)
            )


async def test_the_selector_decides_which_button_fires_the_action():
    # Under a secondary-button selector (the real move_to_ready wiring),
    # primary presses stay quiet and a secondary press fires.
    async with boot(postures_instances=1) as h:
        ready = h.mocks.deps.postures[0].move_to_ready
        async with posture_button(
            h, action_module=move_to_ready, pressed=_secondary
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            with pytest.raises(TimeoutError):
                await ready.next_goal(0.4)
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(False, True, False))
            pending = await ready.next_goal(10.0)
            await (await pending.accept()).complete(
                done(move_to_ready)
            )


@contextlib.asynccontextmanager
async def _running(make_stream):
    """One stream task under a hand-fired token, torn down on exit."""
    token = FakeToken()
    task = asyncio.create_task(make_stream(token))
    try:
        yield
    finally:
        token.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def test_a_failed_publisher_declaration_ends_the_stream_loudly(capsys):
    # A broken transport cannot be scripted onto the real wire; the stub
    # stands in for the runner refusing the declaration.
    class NoPublisherTopic:
        async def declare_publisher(self, _runner):
            raise RuntimeError("no transport")

    async def run():
        await asyncio.wait_for(
            publish._run_stream(
                None, NoPublisherTopic(), 0.001, "left arm", FakeToken(), lambda: None
            ),
            1.0,
        )

    asyncio.run(run())
    assert "failed to declare" in capsys.readouterr().out


async def test_the_gripper_stream_is_silent_without_the_squeeze_deadman():
    async with boot_clocked() as h:
        session = FakeSession()
        session.press(right=(False, False, False))
        async with _running(
            lambda token: publish.stream_gripper(
                h.node_runner,
                topic_module=right_gripper_setpoints,
                handedness="right",
                session=session,
                settings=posture_settings(),
                token=token,
            )
        ):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    h.mocks.pairings.right_gripper.gripper_setpoints.next(), 0.4
                )


async def test_the_gripper_stream_publishes_openings_while_squeezing():
    async with boot_clocked() as h:
        session = FakeSession()
        session.press(right=(False, False, True))
        async with _running(
            lambda token: publish.stream_gripper(
                h.node_runner,
                topic_module=right_gripper_setpoints,
                handedness="right",
                session=session,
                settings=posture_settings(),
                token=token,
            )
        ):
            message = await asyncio.wait_for(
                h.mocks.pairings.right_gripper.gripper_setpoints.next(), 10.0
            )
        assert message.opening == 1.0  # released trigger rests at open_fraction
        assert message.max_effort == 0.0
        assert message.timestamp > 0.0  # stamped from the daemon-resolved clock


async def test_the_pose_stream_refuses_to_engage_without_a_fresh_measured_pose():
    async with boot_clocked() as h:
        session = FakeSession()
        session.press(right=(False, False, True))
        async with _running(
            lambda token: publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                handedness="right",
                clutch=HandClutch(1.0),
                session=session,
                measured=publish.LatestPose(),
                settings=posture_settings(),
                token=token,
            )
        ):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    h.mocks.pairings.right_arm.pose_setpoints.next(), 0.4
                )


async def test_the_pose_stream_engages_once_the_follower_reports():
    async with boot_clocked() as h:
        session = FakeSession()
        session.press(right=(False, False, True))
        measured = publish.LatestPose()
        measured.set(POSE)
        async with _running(
            lambda token: publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                handedness="right",
                clutch=HandClutch(1.0),
                session=session,
                measured=measured,
                settings=posture_settings(),
                token=token,
            )
        ):
            message = await asyncio.wait_for(
                h.mocks.pairings.right_arm.pose_setpoints.next(), 10.0
            )
        # The engage tick commands the measured pose: a no-op that starts the
        # stream.
        assert message.position == [0.0, 0.0, 0.0]
        assert message.orientation == [0.0, 0.0, 0.0, 1.0]


class FrozenSession(FakeSession):
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
    async with boot_clocked() as h:
        session = FrozenSession()
        session.press(right=(False, False, True))
        measured = publish.LatestPose()
        settings = config.from_parameters(default_parameters(stale_timeout_s=0.05))
        token = FakeToken()

        # Measured stays fresh throughout: only the headset link dies.
        async def keep_measured_fresh():
            while not token.is_cancelled():
                measured.set(POSE)
                await asyncio.sleep(0.01)

        refresher = asyncio.create_task(keep_measured_fresh())
        task = asyncio.create_task(
            publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                handedness="right",
                clutch=HandClutch(1.0),
                session=session,
                measured=measured,
                settings=settings,
                token=token,
            )
        )
        subscription = h.mocks.pairings.right_arm.pose_setpoints
        try:
            # Streamed while the frame was fresh.
            await asyncio.wait_for(subscription.next(), 10.0)
            # The frozen frame ages past 0.05 s. Drain the in-flight samples
            # and the release burst until a window longer than the burst
            # passes with nothing published: silence is the deadman.
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    await asyncio.wait_for(subscription.next(), publish.RELEASE_HOLD_S + 0.1)
                except asyncio.TimeoutError:
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


class MovableSession(FakeSession):
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
    async with boot_clocked() as h:
        session = MovableSession()
        measured = publish.LatestPose()
        measured.set(POSE)
        settings = config.from_parameters(default_parameters(command_rate_hz=200))
        token = FakeToken()

        session.press(right=(False, False, True))
        task = asyncio.create_task(
            publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                handedness="right",
                clutch=HandClutch(1.0),
                session=session,
                measured=measured,
                settings=settings,
                token=token,
            )
        )
        subscription = h.mocks.pairings.right_arm.pose_setpoints
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
                assert time.monotonic() < deadline, "the driven target never reached the wire"

            # Let go. The freeze burst must carry the measured pose.
            session.press(right=(False, False, False))
            frozen = await asyncio.wait_for(subscription.next(), 10.0)
            assert frozen.position[0] == pytest.approx(0.0), (
                "release streamed the last hand target instead of the measured pose"
            )

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
                None, ExplodingTopic(), publish.LatestPose(), "right", FakeToken()
            ),
            1.0,
        )

    asyncio.run(run())
    assert "pose_states subscribe failed" in capsys.readouterr().out


async def test_drain_pose_states_keeps_the_holder_at_the_newest_measured_pose():
    async with boot() as h:
        holder = publish.LatestPose()
        async with running_drain(
            lambda token: publish.drain_pose_states(
                h.node_runner, right_arm_pose_states, holder, "right", token
            )
        ):
            await h.mocks.pairings.right_arm.pose_states.publish(
                right_arm_pose_states.Message(
                    timestamp=1.0,
                    position=[0.1, 0.2, 0.3],
                    orientation=[0.0, 0.0, 0.0, 1.0],
                )
            )
            await eventually(
                lambda: holder.fresh(10.0) is not None, message="the first pose"
            )
            await h.mocks.pairings.right_arm.pose_states.publish(
                right_arm_pose_states.Message(
                    timestamp=2.0,
                    position=[0.4, 0.5, 0.6],
                    orientation=[0.0, 0.0, 0.0, 1.0],
                )
            )
            await eventually(
                lambda: (pose := holder.fresh(10.0)) is not None
                and pose.position[0] == 0.4,
                message="the newest pose",
            )


def test_drain_pose_states_survives_garbage():
    # Kept on a stub: every decodable wire message builds a Pose (the schema
    # carries float lists), so the unusable-pose latch is reachable only
    # through a hand-built message the generated modules cannot produce.
    from types import SimpleNamespace

    class StubTopic(FakeSubscription):
        async def subscribe(self, _runner):
            return self

    good = SimpleNamespace(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1])
    bad = SimpleNamespace(position=None, orientation=None)
    producer = SimpleNamespace(instance_id="backbone_inst")
    holder = publish.LatestPose()
    token = FakeToken()

    async def run():
        drain = asyncio.create_task(
            publish.drain_pose_states(
                None, StubTopic([(producer, bad), (producer, good)]), holder, "right", token
            )
        )
        await asyncio.sleep(0.05)
        token.cancel()
        await asyncio.wait_for(drain, 1.0)

    asyncio.run(run())
    pose = holder.fresh(10.0)
    assert pose is not None
    assert pose.position[0] == 0.1


def test_the_face_buttons_wire_home_low_and_ready_high():
    # The wiring table is the one place a swap could hide: pin module to
    # button, so exchanging the two entries fails here and nowhere else.
    from xr_commander import __main__ as main

    (home_module, home_pressed), (ready_module, ready_pressed) = main._POSTURE_BUTTONS
    assert home_module.TARGET_ACTION_NAME == "move_to_home"
    assert ready_module.TARGET_ACTION_NAME == "move_to_ready"
    a_only = HandSample(pose=POSE, squeezing=False, trigger=0.0, primary_button=True)
    b_only = HandSample(pose=POSE, squeezing=False, trigger=0.0, secondary_button=True)
    assert home_pressed(a_only) and not home_pressed(b_only)
    assert ready_pressed(b_only) and not ready_pressed(a_only)


# Fast enough that the burst's message floor never sets the hold length.
FAST_TICK_S = 0.001


class TestReleaseHold:
    def make_pose(self):
        return Pose(position=np.array([0.1, 0.2, 0.3]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))

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
