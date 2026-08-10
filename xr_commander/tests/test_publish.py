import asyncio
import contextlib
import time
from types import SimpleNamespace

import numpy as np

from tests.helpers import (
    POSE,
    FakeSubscription,
    FakeToken,
    FakeTopic,
    RecordingSink,
    default_parameters,
)
from xr_commander import config, publish
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample, XrFrame
from xr_commander.video import drain_frames


def test_latest_pose_goes_stale_rather_than_lying():
    holder = publish.LatestPose()
    assert holder.fresh(10.0) is None
    holder.set(POSE)
    assert holder.fresh(10.0) is POSE
    assert holder.fresh(10.0, now_monotonic=time.monotonic() + 60.0) is None


def frame_message(width=1, height=1):
    data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
    return SimpleNamespace(encoding="bgr8", width=width, height=height, frame=data)


def test_drain_frames_routes_by_producer_and_skips_unknown_producers():
    sink = RecordingSink()
    known = SimpleNamespace(instance_id="cam_a")
    unknown = SimpleNamespace(instance_id="ghost")
    subscription = FakeSubscription(
        [(known, frame_message()), (unknown, frame_message())]
    )
    token = FakeToken()

    async def run():
        drain = asyncio.create_task(
            drain_frames(None, FakeTopic(subscription), {"cam_a": sink}, token, "test")
        )
        await asyncio.sleep(0.05)
        token.cancel()
        await asyncio.wait_for(drain, 1.0)

    asyncio.run(run())
    assert len(sink.frames) == 1


class FakeSession:
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


class FakeHandle:
    accepted = True
    reason = ""

    def __init__(self, log):
        self._log = log

    async def cancel_goal(self, _timeout):
        self._log.append("cancel")

    async def get_result(self, _timeout):
        await asyncio.Event().wait()


class FakeActionModule:
    """The consumed-action module surface run_posture_button drives."""

    TARGET_ACTION_NAME = "move_to_home"

    def __init__(self, log, handle_cls=FakeHandle):
        self.log = log
        self.GoalRequest = lambda duration_s: duration_s
        outer = self

        class Handle:
            @staticmethod
            async def fire_goal(_runner, _target, _request, _timeout, _qos):
                outer.log.append("fire")
                return handle_cls(outer.log)

        self.ActionHandle = Handle

    def bound_producers(self, _runner):
        return [SimpleNamespace(instance_id="backbone_inst")]


def posture_settings():
    return config.from_parameters(default_parameters())


def drive_posture_button(
    script, pressed=lambda sample: sample.primary_button, handle_cls=FakeHandle
):
    """Run the button loop over (primary, secondary, squeeze) steps; a None
    step is a headset gap (no tracked hands). Returns the action-call log."""
    log = []
    session = FakeSession()

    async def run():
        token = FakeToken()
        task = asyncio.create_task(
            publish.run_posture_button(
                None,
                action_module=FakeActionModule(log, handle_cls),
                pressed=pressed,
                session=session,
                settings=posture_settings(),
                token=token,
            )
        )
        for step in script:
            if step is None:
                session.hands = {}
            else:
                session.press(right=step)
            await asyncio.sleep(0.03)
        token.cancel()
        # The loop only re-checks the token each tick; cancel the task too.
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())
    return log


def test_the_result_wait_scales_with_long_posture_moves():
    assert publish._result_timeout_s(2.0) == 120.0
    assert publish._result_timeout_s(100.0) == 300.0


def test_the_button_fires_once_per_rising_edge():
    log = drive_posture_button(
        [(False, False, False), (True, False, False), (True, False, False)]
    )
    assert log == ["fire"]


def test_squeezing_cancels_the_move_in_flight():
    log = drive_posture_button(
        [(False, False, False), (True, False, False), (False, False, True)]
    )
    assert log == ["fire", "cancel"]


def test_a_press_landing_with_a_squeeze_does_not_fire():
    log = drive_posture_button([(False, False, False), (True, False, True)])
    assert log == []


class QuickResultHandle(FakeHandle):
    async def get_result(self, _timeout):
        return SimpleNamespace(
            status=SimpleNamespace(name="SUCCEEDED"),
            data=SimpleNamespace(message="done"),
        )


def test_a_button_already_held_at_first_tracking_does_not_fire():
    held = (True, False, False)
    assert drive_posture_button([held, held, held]) == []
    # An observed release and then a press is the first real edge.
    assert (
        drive_posture_button([held, (False, False, False), held]) == ["fire"]
    )


def test_a_stale_gap_does_not_refire_a_held_button():
    """A dropout while the thumb stays down must not read as a fresh press:
    the gap makes the input unknown, not released."""
    held = (True, False, False)
    log = drive_posture_button(
        [(False, False, False), held, held, None, None, held, held],
        handle_cls=QuickResultHandle,
    )
    assert log == ["fire"]
    # A real release and press after the gap re-arms the button.
    log = drive_posture_button(
        [(False, False, False), held, None, held, (False, False, False), held],
        handle_cls=QuickResultHandle,
    )
    assert log == ["fire", "fire"]


def test_the_selector_decides_which_button_fires_the_action():
    # Under a secondary-button selector, primary presses stay quiet and a
    # secondary press fires.
    def secondary(sample):
        return sample.secondary_button

    assert (
        drive_posture_button(
            [(False, False, False), (True, False, False)], pressed=secondary
        )
        == []
    )
    assert drive_posture_button(
        [(False, False, False), (False, True, False)], pressed=secondary
    ) == ["fire"]


class FakePublisher:
    def __init__(self):
        self.published = []

    async def publish(self, payload):
        self.published.append(payload)


class FakeStreamTopic:
    def __init__(self, publisher):
        self._publisher = publisher
        self.built = []

    async def declare_publisher(self, _runner):
        return self._publisher

    def build_message(self, stamp, *fields):
        self.built.append(fields)
        return fields


def run_stream_briefly(coro_factory):
    async def run():
        token = FakeToken()
        task = asyncio.create_task(coro_factory(token))
        await asyncio.sleep(0.05)
        token.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_a_failed_publisher_declaration_ends_the_stream_loudly(capsys):
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


def test_the_gripper_stream_is_silent_without_the_squeeze_deadman():
    publisher = FakePublisher()
    topic = FakeStreamTopic(publisher)
    session = FakeSession()
    session.press(right=(False, False, False))
    run_stream_briefly(
        lambda token: publish.stream_gripper(
            None,
            topic_module=topic,
            handedness="right",
            session=session,
            settings=posture_settings(),
            token=token,
        )
    )
    assert publisher.published == []


def test_the_gripper_stream_publishes_openings_while_squeezing(monkeypatch):
    monkeypatch.setattr(publish, "_stamp_seconds", lambda: 0.0)
    publisher = FakePublisher()
    topic = FakeStreamTopic(publisher)
    session = FakeSession()
    session.press(right=(False, False, True))
    run_stream_briefly(
        lambda token: publish.stream_gripper(
            None,
            topic_module=topic,
            handedness="right",
            session=session,
            settings=posture_settings(),
            token=token,
        )
    )
    assert publisher.published, "a squeezed hand must stream openings"
    opening, max_effort = publisher.published[0]
    assert opening == 1.0  # released trigger rests at gripper_open_fraction
    assert max_effort == 0.0


def test_the_pose_stream_refuses_to_engage_without_a_fresh_measured_pose():
    publisher = FakePublisher()
    topic = FakeStreamTopic(publisher)
    session = FakeSession()
    session.press(right=(False, False, True))
    clutch = HandClutch(1.0)
    run_stream_briefly(
        lambda token: publish.stream_pose(
            None,
            topic_module=topic,
            handedness="right",
            clutch=clutch,
            session=session,
            measured=publish.LatestPose(),
            settings=posture_settings(),
            token=token,
        )
    )
    assert publisher.published == []


def test_the_pose_stream_engages_once_the_follower_reports(monkeypatch):
    monkeypatch.setattr(publish, "_stamp_seconds", lambda: 0.0)
    publisher = FakePublisher()
    topic = FakeStreamTopic(publisher)
    session = FakeSession()
    session.press(right=(False, False, True))
    measured = publish.LatestPose()
    measured.set(POSE)
    clutch = HandClutch(1.0)
    run_stream_briefly(
        lambda token: publish.stream_pose(
            None,
            topic_module=topic,
            handedness="right",
            clutch=clutch,
            session=session,
            measured=measured,
            settings=posture_settings(),
            token=token,
        )
    )
    assert publisher.published, "an engaged hand must stream setpoints"


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


def test_the_pose_stream_goes_silent_when_frames_stop_mid_squeeze(monkeypatch):
    """The composed deadman: an engaged hand whose headset frames age past
    the staleness window stops publishing entirely."""
    monkeypatch.setattr(publish, "_stamp_seconds", lambda: 0.0)
    publisher = FakePublisher()
    topic = FakeStreamTopic(publisher)
    session = FrozenSession()
    session.press(right=(False, False, True))
    measured = publish.LatestPose()
    clutch = HandClutch(1.0)
    settings = config.from_parameters(default_parameters(stale_timeout_s=0.05))

    async def run():
        # Measured stays fresh throughout: only the headset link dies.
        async def keep_measured_fresh(token):
            while not token.is_cancelled():
                measured.set(POSE)
                await asyncio.sleep(0.01)

        token = FakeToken()
        refresher = asyncio.create_task(keep_measured_fresh(token))
        task = asyncio.create_task(
            publish.stream_pose(
                None,
                topic_module=topic,
                handedness="right",
                clutch=clutch,
                session=session,
                measured=measured,
                settings=settings,
                token=token,
            )
        )
        await asyncio.sleep(0.03)
        streamed_while_fresh = len(publisher.published)
        await asyncio.sleep(0.1)  # the frozen frame ages past 0.05 s
        settled = len(publisher.published)
        await asyncio.sleep(0.05)
        assert streamed_while_fresh > 0
        assert len(publisher.published) == settled
        token.cancel()
        task.cancel()
        refresher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with contextlib.suppress(asyncio.CancelledError):
            await refresher

    asyncio.run(run())


def test_a_failed_pose_states_subscribe_is_loud_and_fatal_to_the_drain(capsys):
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


def test_drain_pose_states_keeps_the_holder_fresh_and_survives_garbage():
    good = SimpleNamespace(position=[0.1, 0.2, 0.3], orientation=[0, 0, 0, 1])
    bad = SimpleNamespace(position=None, orientation=None)
    producer = SimpleNamespace(instance_id="backbone_inst")
    subscription = FakeSubscription([(producer, bad), (producer, good)])
    holder = publish.LatestPose()
    token = FakeToken()

    async def run():
        drain = asyncio.create_task(
            publish.drain_pose_states(
                None, FakeTopic(subscription), holder, "right", token
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
