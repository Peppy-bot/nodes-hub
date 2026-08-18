"""Wire-level boundary coverage through the generated fixtures and mocks:
the node's real `setup` boots in-process against an ephemeral router, mock
sources play the observer and camera slots over the real wire, and the
node's exposed action and services are driven from a fixture session, the
same path a real consumer takes.

Shape under test: one observed_joints source positionally paired with one
commanded_joints source (one limb), one color camera, and the remaining
zero_or_more slots deliberately empty. Synchronization is wire-observable
(goal admission, feedback, results, service responses) under bounded
timeouts; the only polling is retrying goal admission while the mocks'
first samples are still in flight, itself bounded."""

import asyncio
import json
import time

import peppylib
import pytest

from lerobot_recorder.__main__ import setup
from peppygen.consumed_topics.color_cameras import video_stream
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions import record_episode
from peppygen.fixtures.exposed_services import finish_session
from peppygen.paired_topics.commanded_joints import joint_setpoints
from peppygen.paired_topics.observed_joints import joint_states
from peppygen.parameters import Parameters

FPS = 10
# The geometry the sink integration suite proves out: smaller frames stall
# lerobot's SVT-AV1 encode (it pads toward 64x64 and hangs on 8x6 input).
W, H = 32, 16
FRAME = bytes([64, 96, 128]) * (W * H)  # bgr8: raw on the wire, no encoder needed
# The one limb the harness binds, under the mock identities the generated
# harness seeds (instance ids are suffixed per instance, link is the mock's
# source link).
OBSERVED = "mock-core/mock-observed_joints-0/mock_source"
COMMANDED = "mock-core/mock-commanded_joints-0/mock_source"
CAMERA = "mock-core/mock-color_cameras-0"
LIMB = "mock_observed_joints_0"


def params(tmp_path) -> Parameters:
    return Parameters(
        robot_type="bot",
        fps=FPS,
        storage_root=str(tmp_path),
        s3_uri="",
        image_writer_threads=1,
        # Staged-PNG encoding: its writer queue is unbounded, so a saved
        # episode's video deterministically covers every row (no drop-driven
        # video_error to race the result assertions against).
        streaming_encoding=False,
        # Generous: a loaded host importing torch mid-test must not read as a
        # dead source.
        max_staleness_s=5.0,
        min_remaining_disk_bytes=1,
    )


def boot(tmp_path):
    return harness.start(
        setup,
        parameters=params(tmp_path),
        observed_joints_instances=1,
        commanded_joints_instances=1,
        color_cameras_instances=1,
    )


async def stream(h, stop: asyncio.Event) -> None:
    """Play the bound sources at ~20 Hz: measured joints, commanded
    setpoints, and camera frames, all stamped from the wall clock the
    standalone node measures staleness against."""
    joints = h.mocks.observed.observed_joints[0].joint_states
    setpoints = h.mocks.observed.commanded_joints[0].joint_setpoints
    camera = h.mocks.deps.color_cameras[0].video_stream
    frame_id = 0
    while not stop.is_set():
        now = time.time()
        await joints.publish(
            joint_states.Message(
                timestamp=now, positions=[0.1, 0.2], velocities=[0.0, 0.0], efforts=[]
            )
        )
        await setpoints.publish(
            joint_setpoints.Message(
                timestamp=now, positions=[0.3, 0.4], velocities=[], efforts=[]
            )
        )
        await camera.publish(
            video_stream.Message(
                header=video_stream.MessageHeader(timestamp=now, frame_id=frame_id),
                encoding="bgr8",
                width=W,
                height=H,
                frame=FRAME,
            )
        )
        frame_id += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except TimeoutError:
            pass


async def goal_accepted(h, deadline_s: float = 240.0):
    """Send record_episode goals until the node admits one. Rejections while
    the mocks' first samples are in flight are the node's own admission
    answers ('... has not produced yet' / 'no fresh frame'), so the retry is
    driving the real wire protocol, bounded by deadline_s."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    reason = None
    while loop.time() < deadline:
        goal = await record_episode.send_goal(
            h,
            record_episode.GoalRequestData(task="harness"),
            peppylib.QoSProfile.Standard,
            30.0,
        )
        if goal.accepted:
            return goal
        reason = goal.reason
        assert reason, "a rejection must carry the node's refusal reason"
        await asyncio.sleep(0.2)
    pytest.fail(f"goal never accepted within {deadline_s}s; last reason: {reason!r}")


async def finish_rolled(h, deadline_s: float = 60.0):
    """Poll finish_session until the roll lands. The one refusal retried is
    'an episode is recording': the recorder clears that flag a beat after
    the goal's result is delivered, so a poll can race it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while True:
        response = await finish_session.poll(h, 30.0)
        if response.error is None or "recording" not in response.error:
            return response
        if loop.time() > deadline:
            pytest.fail(f"finish_session kept refusing: {response.error!r}")
        await asyncio.sleep(0.2)


def session_dirs(tmp_path):
    return sorted(p for p in tmp_path.iterdir() if p.is_dir())


async def test_boot_pairs_one_limb_and_refuses_goals_while_sources_are_silent(tmp_path):
    """Boot with 1 observed + 1 commanded joints source, 1 color camera and
    every other zero_or_more slot empty: discovery pairs one limb positionally
    and records the wire identities, and a goal sent before any mock has
    published is rejected over the wire with the node's own reason."""
    async with boot(tmp_path) as h:
        # Empty zero_or_more slots start no mocks.
        assert h.mocks.observed.observed_grippers == []
        assert h.mocks.observed.commanded_grippers == []
        assert h.mocks.deps.rgbd_cameras == []

        goal = await record_episode.send_goal(
            h,
            record_episode.GoalRequestData(task="too-early"),
            peppylib.QoSProfile.Standard,
            30.0,
        )
        assert not goal.accepted
        assert "has not produced yet" in goal.reason

        # The session opened during setup, from the seeded membership: the
        # provenance names the limb after the follower instance and maps both
        # of the pairing's sources onto it.
        (session_dir,) = session_dirs(tmp_path)
        provenance = json.loads((session_dir / "session.json").read_text())
        assert provenance["state_links"] == {OBSERVED: LIMB}
        assert provenance["action_links"] == {COMMANDED: LIMB}
        assert provenance["color_cameras"] == [CAMERA]
        assert provenance["rgbd_cameras"] == []


async def test_record_episode_feedback_flows_and_cancel_saves(tmp_path):
    """The full record_episode lifecycle over the real action engine: the
    goal is admitted once every source has produced, per-frame feedback flows
    while the mocks stream, a goal sent mid-episode is refused busy, and
    cancel (the operator stop) ends the episode with a save: a CANCELLED
    result carrying episode_index 0, the frame count, discarded=False and no
    error."""
    async with boot(tmp_path) as h:
        stop = asyncio.Event()
        streamer = asyncio.create_task(stream(h, stop))
        try:
            goal = await goal_accepted(h)

            busy = await record_episode.send_goal(
                h,
                record_episode.GoalRequestData(task="second"),
                peppylib.QoSProfile.Standard,
                30.0,
            )
            assert not busy.accepted
            assert "already recording" in busy.reason

            # One feedback message per recorded frame; disk_free rides it.
            first = await asyncio.wait_for(goal.on_next_feedback(), 240.0)
            second = await asyncio.wait_for(goal.on_next_feedback(), 60.0)
            assert first.frames_recorded >= 1
            assert second.frames_recorded > first.frames_recorded
            assert first.disk_free_bytes > 0

            cancel = await goal.cancel_goal(30.0)
            assert cancel.state == record_episode.CancelState.SIGNALLED

            result = await goal.get_result(180.0)
            assert result.status == record_episode.ResultStatus.CANCELLED
            assert result.data is not None
            assert result.data.discarded is False
            assert result.data.episode_index == 0
            assert result.data.frames_recorded >= second.frames_recorded
            # An operator stop carries no error; staged encoding cannot come
            # up short on video frames.
            assert result.data.error is None

            # d) finish_session rolls the saved session over to a fresh one.
            (recorded_dir,) = session_dirs(tmp_path)
            finished = await finish_rolled(h)
            assert finished.error is None
            assert finished.session == recorded_dir.name
            assert (recorded_dir / "dataset" / "meta" / "info.json").is_file()
            # The roll opened a successor session directory in place.
            assert len(session_dirs(tmp_path)) == 2

            # The rolled-to session holds no episodes yet, and the node says so.
            empty = await finish_session.poll(h, 30.0)
            assert empty.session == ""
            assert empty.error == "nothing recorded this session"
        finally:
            stop.set()
            await streamer
