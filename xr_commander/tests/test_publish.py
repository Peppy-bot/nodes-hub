import asyncio
import contextlib
import time
from types import SimpleNamespace

import numpy as np
import pytest
from peppygen.consumed_actions.postures import move_to_home, move_to_ready
from peppygen.paired_topics.right_arm import pose_setpoints as right_arm_pose_setpoints
from peppygen.paired_topics.right_arm import pose_states as right_arm_pose_states
from peppygen.paired_topics.right_gripper import (
    gripper_setpoints as right_gripper_setpoints,
)

from tests.helpers import (
    POSE,
    FakeSubscription,
    FakeToken,
    boot,
    boot_clocked,
    default_parameters,
    eventually,
    peer_of,
    press_until_goal,
    robots_of,
    running_drain,
)
from xr_commander import config, publish
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample, XrFrame
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


def posture_settings():
    return config.from_parameters(default_parameters())


def posture_result(module, success, message):
    """A posture result as a robot sends it. The commander reads only the
    status and the message, thus the arm pose in it is a fixed value."""
    return module.ResultResponseData(
        success=success,
        message=message,
        arm_names=["right_arm"],
        positions=[0.3, -0.2, 0.3],
        orientations=[0.0, 0.0, 0.0, 1.0],
    )


def done(module):
    """The action's success verdict, as each completed-goal assertion uses."""
    return posture_result(module, success=True, message="done")


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


def boot_postures():
    """A stack whose one paired robot, outside any copy, offers the
    postures: the same instance id names the pair and the producer."""
    return boot(
        postures_instance_ids=["backbone_inst"],
        right_arm_members={"backbone_inst": None},
    )


@contextlib.asynccontextmanager
async def posture_button(h, *, action_module, pressed, status=None):
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
            robots=robots_of(h),
            status=status or publish.PostureStatus(),
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


async def test_a_posture_move_shows_in_flight_then_its_outcome():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        status = publish.PostureStatus()
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary, status=status
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            active = await pending.accept()
            await eventually(lambda: status.state() == "moving to home...")
            await active.complete(done(move_to_home))
            await eventually(lambda: status.state() == "move to home completed")
            assert status.moving is None


async def test_a_failed_move_flashes_the_robots_reason():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        status = publish.PostureStatus()
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary, status=status
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            active = await pending.accept()
            await active.complete(
                posture_result(
                    move_to_home,
                    success=False,
                    message="the robot has not measured every limb yet",
                )
            )
            await eventually(
                lambda: (
                    status.state()
                    == "move to home failed: the robot has not measured every limb yet"
                )
            )


async def test_a_press_while_another_posture_runs_is_ignored_and_says_so():
    async with boot_postures() as h:
        home = h.mocks.deps.postures[0].move_to_home
        ready = h.mocks.deps.postures[0].move_to_ready
        status = publish.PostureStatus()
        async with (
            posture_button(
                h, action_module=move_to_home, pressed=_primary, status=status
            ) as session_a,
            posture_button(
                h, action_module=move_to_ready, pressed=_secondary, status=status
            ) as session_b,
        ):
            session_a.press(right=(False, False, False))
            session_b.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session_a.press(right=(True, False, False))
            pending = await home.next_goal(10.0)
            active = await pending.accept()
            await eventually(lambda: status.moving == "home")
            session_b.press(right=(False, True, False))
            await eventually(
                lambda: status.state() == "move to ready ignored: still moving to home"
            )
            with pytest.raises(TimeoutError):
                await ready.next_goal(0.4)
            await active.complete(done(move_to_home))
            await eventually(lambda: status.state() == "move to home completed")


async def test_a_refused_posture_flashes_its_cause():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        status = publish.PostureStatus()
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary, status=status
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            await pending.reject(
                "a leader is streaming arm commands; release the headset's grip, then retry"
            )
            await eventually(
                lambda: (
                    status.state()
                    == "move to home refused: a leader is streaming arm commands"
                )
            )
            assert status.moving is None


def test_the_result_wait_scales_with_long_posture_moves():
    assert publish._result_timeout_s(2.0) == 120.0
    assert publish._result_timeout_s(100.0) == 300.0


async def test_the_button_fires_once_per_rising_edge():
    async with boot_postures() as h:
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
            await active.complete(done(move_to_home))


async def test_squeezing_cancels_the_move_in_flight():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        status = publish.PostureStatus()
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary, status=status
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            active = await pending.accept()
            await eventually(lambda: status.moving == "home")
            # Taking manual control back is the cancel gesture.
            session.press(right=(False, False, True))
            await asyncio.wait_for(active.cancel_signal(), 10.0)
            # The row stops claiming a move as soon as the cancel is delivered.
            await eventually(lambda: status.moving is None)
            await active.complete_cancelled(
                posture_result(move_to_home, success=False, message="cancelled")
            )
            await eventually(lambda: status.state() == "move to home cancelled")
            # The squeeze that cancelled must not have fired a fresh goal.
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)


async def test_a_press_landing_with_a_squeeze_does_not_fire_and_says_to_let_go():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        status = publish.PostureStatus()
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary, status=status
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, True))
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            assert status.state() == "move to home ignored: release the grip first"


async def test_a_button_already_held_at_first_tracking_does_not_fire():
    async with boot_postures() as h:
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
            await (await pending.accept()).complete(done(move_to_home))


async def test_a_stale_gap_does_not_refire_a_held_button():
    """A dropout while the thumb stays down must not read as a fresh press:
    the gap makes the input unknown, not released."""
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        async with posture_button(
            h, action_module=move_to_home, pressed=_primary
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            await (await pending.accept()).complete(done(move_to_home))
            # The headset drops out while the thumb stays down...
            session.hands = {}
            await asyncio.sleep(0.05)
            # ...and comes back with the button still held: not a press.
            session.press(right=(True, False, False))
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            # A real release and press after the gap re-arms the button.
            pending = await press_until_goal(
                session,
                mock,
                {"right": (False, False, False)},
                {"right": (True, False, False)},
            )
            await (await pending.accept()).complete(done(move_to_home))


async def test_the_selector_decides_which_button_fires_the_action():
    # Under a secondary-button selector (the real move_to_ready wiring),
    # primary presses stay quiet and a secondary press fires.
    async with boot_postures() as h:
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
            await (await pending.accept()).complete(done(move_to_ready))


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
    async with boot_clocked(**ONE_ROBOT) as h:
        session = FakeSession()
        session.press(right=(False, False, False))
        async with _running(
            lambda token: publish.stream_gripper(
                h.node_runner,
                topic_module=right_gripper_setpoints,
                handedness="right",
                squeeze=publish.HeldOverSqueeze(),
                session=session,
                robots=robots_of(h),
                settings=posture_settings(),
                token=token,
            )
        ):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    h.mocks.pairings.right_gripper[0].gripper_setpoints.next(), 0.4
                )


async def test_the_gripper_stream_publishes_openings_while_squeezing():
    async with boot_clocked(**ONE_ROBOT) as h:
        session = FakeSession()
        session.press(right=(False, False, True))
        async with _running(
            lambda token: publish.stream_gripper(
                h.node_runner,
                topic_module=right_gripper_setpoints,
                handedness="right",
                squeeze=publish.HeldOverSqueeze(),
                session=session,
                robots=robots_of(h),
                settings=posture_settings(),
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
        session = FakeSession()
        session.press(right=(False, False, True))
        async with _running(
            lambda token: publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                handedness="right",
                clutch=HandClutch(1.0),
                squeeze=publish.HeldOverSqueeze(),
                session=session,
                measured=publish.MeasuredPoses(),
                robots=robots_of(h),
                settings=posture_settings(),
                token=token,
            )
        ):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    h.mocks.pairings.right_arm[0].pose_setpoints.next(), 0.4
                )


async def test_the_pose_stream_engages_once_the_follower_reports():
    async with boot_clocked(**ONE_ROBOT) as h:
        session = FakeSession()
        session.press(right=(False, False, True))
        measured = publish.MeasuredPoses()
        measured.set(peer_of(h, right_arm_pose_setpoints), POSE)
        async with _running(
            lambda token: publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                handedness="right",
                clutch=HandClutch(1.0),
                squeeze=publish.HeldOverSqueeze(),
                session=session,
                measured=measured,
                robots=robots_of(h),
                settings=posture_settings(),
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
        robots = robots_of(h)
        peers = peers_by_robot(h, right_arm_pose_setpoints)
        alpha_wire, bravo_wire = (
            mock.pose_setpoints for mock in h.mocks.pairings.right_arm
        )
        measured = publish.MeasuredPoses()
        measured.set(peers["alpha"], POSE)
        measured.set(peers["bravo"], Pose(np.array([0.5, 0.0, 0.0]), POSE.orientation))
        session = MovableSession()
        settings = config.from_parameters(default_parameters(command_rate_hz=200))
        session.press(right=(False, False, True))
        async with _running(
            lambda token: publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                handedness="right",
                clutch=HandClutch(1.0),
                squeeze=publish.HeldOverSqueeze(),
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
        robots = robots_of(h)
        alpha_wire, bravo_wire = (
            mock.gripper_setpoints for mock in h.mocks.pairings.right_gripper
        )
        session = FakeSession()
        session.press(right=(False, False, True))
        async with _running(
            lambda token: publish.stream_gripper(
                h.node_runner,
                topic_module=right_gripper_setpoints,
                handedness="right",
                squeeze=publish.HeldOverSqueeze(),
                session=session,
                robots=robots,
                settings=posture_settings(),
                token=token,
            )
        ):
            await asyncio.wait_for(alpha_wire.next(), 10.0)
            assert robots.choose("bravo")
            assert await received_within(bravo_wire, 0.4) == []
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(False, False, True))
            await asyncio.wait_for(bravo_wire.next(), 10.0)


async def test_a_posture_press_is_inert_while_the_chosen_robot_offers_none(capsys):
    async with boot(
        postures_instance_ids=["backbone_inst"],
        right_arm_members={"backbone_inst": None, "bravo_backbone_inst": "bravo"},
    ) as h:
        mock = h.mocks.deps.postures[0].move_to_home
        robots = robots_of(h)
        assert robots.choose("bravo")
        session = FakeSession()
        token = FakeToken()
        status = publish.PostureStatus()
        task = asyncio.create_task(
            publish.run_posture_button(
                h.node_runner,
                action_module=move_to_home,
                pressed=_primary,
                session=session,
                robots=robots,
                status=status,
                settings=posture_settings(),
                token=token,
            )
        )
        try:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            assert status.state() == "move to home: bravo has no postures"
        finally:
            token.cancel()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert capsys.readouterr().out.count("bravo has no postures") == 1


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
    async with boot_clocked(**ONE_ROBOT) as h:
        session = FrozenSession()
        session.press(right=(False, False, True))
        measured = publish.MeasuredPoses()
        settings = config.from_parameters(default_parameters(stale_timeout_s=0.05))
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
                handedness="right",
                clutch=HandClutch(1.0),
                squeeze=publish.HeldOverSqueeze(),
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
    async with boot_clocked(**ONE_ROBOT) as h:
        session = MovableSession()
        measured = publish.MeasuredPoses()
        measured.set(peer_of(h, right_arm_pose_setpoints), POSE)
        settings = config.from_parameters(default_parameters(command_rate_hz=200))
        token = FakeToken()

        session.press(right=(False, False, True))
        task = asyncio.create_task(
            publish.stream_pose(
                h.node_runner,
                topic_module=right_arm_pose_setpoints,
                handedness="right",
                clutch=HandClutch(1.0),
                squeeze=publish.HeldOverSqueeze(),
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


def test_the_first_robot_engages_a_squeeze_already_held():
    gate = publish.HeldOverSqueeze()
    gate.note_chosen("alpha")
    assert gate.engages(frame_of(True))


def test_a_squeeze_held_across_any_change_of_robot_waits_for_a_fresh_release():
    gate = publish.HeldOverSqueeze()
    # The arm and gripper streams both note the robot; a second note of the
    # same robot changes nothing.
    gate.note_chosen("charlie")
    gate.note_chosen("charlie")
    assert gate.engages(frame_of(True))
    # Charlie pairs no left arm: the gate follows the robot, not the pair.
    gate.note_chosen("alpha")
    assert not gate.engages(frame_of(True))
    # A stale tick says nothing about the grip.
    assert not gate.engages(None)
    assert not gate.engages(frame_of(True))
    assert gate.held_over()
    assert not gate.engages(frame_of(False))
    assert gate.engages(frame_of(True))
    # Gone and back through no robot at all: still a change.
    gate.note_chosen(None)
    gate.note_chosen("bravo")
    assert not gate.engages(frame_of(True))
