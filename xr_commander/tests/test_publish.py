import asyncio
import contextlib
import time
from types import SimpleNamespace

import numpy as np

from tests.helpers import IDENTITY, FakeSubscription, FakeToken, RecordingSink
from xr_commander import config, publish
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample, XrFrame
from xr_commander.frames import Pose
from xr_commander.video import drain_frames

POSE = Pose(np.zeros(3), IDENTITY)


def test_latest_pose_goes_stale_rather_than_lying():
    holder = publish.LatestPose()
    assert holder.fresh(10.0) is None
    holder.set(POSE)
    assert holder.fresh(10.0) is POSE
    holder._received_monotonic_s -= 60.0
    assert holder.fresh(10.0) is None


class FakeTopic:
    """A consumed-topic module surface: hands out one prebuilt subscription."""

    def __init__(self, subscription):
        self._subscription = subscription

    async def subscribe(self, _runner):
        return self._subscription


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
                pose=POSE, squeezing=squeeze, trigger=0.0, primary_button=primary
            )
            for side, (primary, squeeze) in hands.items()
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
    """The consumed-action module surface run_ready_button drives."""

    def __init__(self, log):
        self.log = log
        self.GoalRequest = lambda duration_s: duration_s
        outer = self

        class Handle:
            @staticmethod
            async def fire_goal(_runner, _target, _request, _timeout, _qos):
                outer.log.append("fire")
                return FakeHandle(outer.log)

        self.ActionHandle = Handle

    def bound_producers(self, _runner):
        return [SimpleNamespace(instance_id="backbone_inst")]


def ready_settings():
    return config.from_parameters(
        SimpleNamespace(
            command_rate_hz=1000,
            https_port=4443,
            https_host="0.0.0.0",
            motion_scale=1.0,
            gripper_open_fraction=1.0,
            ready_move_duration_s=2.0,
            stale_timeout_s=10.0,
            camera_names="",
        )
    )


def drive_ready_button(script):
    """Run the button loop over a list of (primary, squeeze) steps; return the
    action-call log."""
    log = []
    session = FakeSession()

    async def run():
        token = FakeToken()
        task = asyncio.create_task(
            publish.run_ready_button(
                None,
                action_module=FakeActionModule(log),
                session=session,
                settings=ready_settings(),
                token=token,
            )
        )
        for primary, squeeze in script:
            session.press(right=(primary, squeeze))
            await asyncio.sleep(0.03)
        token.cancel()
        # The loop only re-checks the token each tick; cancel the task too.
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())
    return log


def test_the_result_wait_scales_with_long_ready_moves():
    assert publish._result_timeout_s(2.0) == 120.0
    assert publish._result_timeout_s(100.0) == 300.0


def test_the_a_button_fires_once_per_rising_edge():
    log = drive_ready_button([(False, False), (True, False), (True, False)])
    assert log == ["fire"]


def test_squeezing_cancels_the_move_in_flight():
    log = drive_ready_button([(False, False), (True, False), (False, True)])
    assert log == ["fire", "cancel"]


def test_a_press_landing_with_a_squeeze_does_not_fire():
    log = drive_ready_button([(False, False), (True, True)])
    assert log == []


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


def test_the_gripper_stream_is_silent_without_the_squeeze_deadman():
    publisher = FakePublisher()
    topic = FakeStreamTopic(publisher)
    session = FakeSession()
    session.press(right=(False, False))
    run_stream_briefly(
        lambda token: publish.stream_gripper(
            None,
            topic_module=topic,
            handedness="right",
            session=session,
            settings=ready_settings(),
            token=token,
        )
    )
    assert publisher.published == []


def test_the_gripper_stream_publishes_openings_while_squeezing(monkeypatch):
    monkeypatch.setattr(publish, "_stamp_seconds", lambda: 0.0)
    publisher = FakePublisher()
    topic = FakeStreamTopic(publisher)
    session = FakeSession()
    session.press(right=(False, True))
    run_stream_briefly(
        lambda token: publish.stream_gripper(
            None,
            topic_module=topic,
            handedness="right",
            session=session,
            settings=ready_settings(),
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
    session.press(right=(False, True))
    clutch = HandClutch(1.0)
    run_stream_briefly(
        lambda token: publish.stream_pose(
            None,
            topic_module=topic,
            handedness="right",
            clutch=clutch,
            session=session,
            measured=publish.LatestPose(),
            settings=ready_settings(),
            token=token,
        )
    )
    assert publisher.published == []


def test_the_pose_stream_engages_once_the_follower_reports(monkeypatch):
    monkeypatch.setattr(publish, "_stamp_seconds", lambda: 0.0)
    publisher = FakePublisher()
    topic = FakeStreamTopic(publisher)
    session = FakeSession()
    session.press(right=(False, True))
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
            settings=ready_settings(),
            token=token,
        )
    )
    assert publisher.published, "an engaged hand must stream setpoints"


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
