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
first samples or a ticked instant are still in flight, itself bounded."""

import asyncio
import json
import time

import pytest

from lerobot_recorder.__main__ import setup
from lerobot_recorder.recording import MIN_EPISODE_FRAMES
from peppygen.consumed_topics.color_cameras import video_stream
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions.recording import record_episode
from peppygen.fixtures.exposed_services.recording import finish_session
from peppygen.paired_topics.commanded_joints import joint_setpoints
from peppygen.paired_topics.observed_joints import joint_states
from peppygen.parameters import Parameters
from tests.support import make_params

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
LIMB = "mock_source"
# What the node's refusal says while its clock has read no instant.
CLOCK_NOT_READY = "clock not ready"


def params(tmp_path) -> Parameters:
    # Staged-PNG encoding: its writer queue is unbounded, so a saved
    # episode's video deterministically covers every row (no dropped video
    # frames to race the result assertions against). Staleness is
    # generous: a loaded host importing torch mid-test must not read as a
    # dead source.
    return make_params(
        fps=FPS,
        storage_root=str(tmp_path),
        streaming_encoding=False,
        max_staleness_s=5.0,
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


async def first_goal_answer(h, settled, failure: str, deadline_s: float):
    """Send record_episode goals until the node's answer is `settled`, and
    return that answer. The refusals before it are the node's own admission
    answers while what it waits for is still in flight to it, so the retry is
    driving the real wire protocol, bounded by deadline_s. `failure` opens
    the message of a retry that runs out."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    reason = None
    while loop.time() < deadline:
        goal = await record_episode.send_goal(
            h,
            record_episode.GoalRequestData(task="harness"),
            30.0,
        )
        if settled(goal):
            return goal
        assert goal.refusal.kind is record_episode.FailureKind.PRECONDITION_FAILED
        reason = goal.refusal.message
        assert reason, "a refusal must carry the node's reason"
        await asyncio.sleep(0.2)
    pytest.fail(f"{failure} within {deadline_s}s; last reason: {reason!r}")


async def goal_accepted(h, deadline_s: float = 240.0):
    """Send record_episode goals until the node admits one, retrying the
    refusals while the mocks' first samples are in flight ('... has not
    produced yet' / 'no fresh frame')."""
    return await first_goal_answer(
        h, lambda goal: goal.accepted, "goal never accepted", deadline_s
    )


async def goal_past_the_clock(h, deadline_s: float = 60.0):
    """Send record_episode goals until the node answers with something other
    than its clock refusal, retrying that refusal while the instant the test
    ticked is in flight to the node's clock."""
    return await first_goal_answer(
        h,
        lambda goal: goal.accepted or CLOCK_NOT_READY not in goal.refusal.message,
        "the node never read the ticked instant",
        deadline_s,
    )


async def finish_rolled(h, deadline_s: float = 60.0):
    """Poll finish_session until the roll lands. The one error reply retried
    is 'an episode is recording': the recorder clears that flag a beat after
    the goal's result is delivered, so a poll can race it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while True:
        try:
            return await finish_session.poll(h, 30.0)
        except finish_session.Failure as failure:
            if failure.message != "an episode is recording":
                raise
            if loop.time() > deadline:
                pytest.fail(f"finish_session kept refusing: {failure}")
        await asyncio.sleep(0.2)


def session_dirs(tmp_path):
    return sorted(p for p in tmp_path.iterdir() if p.is_dir())


async def test_boot_pairs_one_limb_and_refuses_goals_while_sources_are_silent(tmp_path):
    """Boot with 1 observed + 1 commanded joints source, 1 color camera and
    every other zero_or_more slot empty: discovery pairs one limb positionally
    and records the wire identities, and a goal sent before any mock has
    published is refused over the wire with `precondition_failed` and the
    node's own reason."""
    async with boot(tmp_path) as h:
        # Empty zero_or_more slots start no mocks.
        assert h.mocks.observed.observed_grippers == []
        assert h.mocks.observed.commanded_grippers == []
        assert h.mocks.deps.rgbd_cameras == []

        goal = await record_episode.send_goal(
            h,
            record_episode.GoalRequestData(task="too-early"),
            30.0,
        )
        assert goal.refusal.kind is record_episode.FailureKind.PRECONDITION_FAILED
        assert "has not produced yet" in goal.refusal.message

        # The session opened during setup, from the seeded membership: the
        # provenance names the limb after the link its command travels on and
        # maps both of the pairing's sources onto it.
        (session_dir,) = session_dirs(tmp_path)
        provenance = json.loads((session_dir / "session.json").read_text())
        assert provenance["state_links"] == {OBSERVED: LIMB}
        assert provenance["action_links"] == {COMMANDED: LIMB}
        assert provenance["color_cameras"] == [CAMERA]
        assert provenance["rgbd_cameras"] == []


async def test_a_driven_clock_gates_admission_until_the_test_ticks_it(tmp_path):
    """Bound to the harness's clock domain the recorder reads it through
    `peppygen.clock`, so the clock-not-ready admission gate is exercised over
    the real wire: before the test publishes any instant a goal is refused
    with the node's own clock reason, and the first driven tick moves the
    refusal on to the next unmet precondition (the still-silent sources)."""
    async with harness.start(
        setup,
        parameters=params(tmp_path),
        clock="consumer",
        observed_joints_instances=1,
        commanded_joints_instances=1,
        color_cameras_instances=1,
    ) as h:
        goal = await record_episode.send_goal(
            h,
            record_episode.GoalRequestData(task="no-time-yet"),
            30.0,
        )
        assert goal.refusal.kind is record_episode.FailureKind.PRECONDITION_FAILED
        assert CLOCK_NOT_READY in goal.refusal.message

        # The test is the simulator: tick() waits for the node's clock
        # subscription (opened by the setup's clock init) and publishes the
        # instant. The node's subscription takes the instant in on a task of
        # its own, so a goal sent right after tick() can still meet the clock
        # refusal.
        await h.clock.tick(1_000_000_000)
        goal = await goal_past_the_clock(h)
        assert not goal.accepted
        assert "has not produced yet" in goal.refusal.message


async def test_record_episode_feedback_flows_and_cancel_saves(tmp_path):
    """The full record_episode lifecycle over the real action engine: the
    goal is admitted once every source has produced, per-frame feedback flows
    while the mocks stream, a goal sent mid-episode is refused busy, and
    cancel (the operator stop) ends the episode with a save: a CANCELLED
    result for the reason of the caller's cancel, carrying episode_index 0,
    the frame count and no missing video frame."""
    async with boot(tmp_path) as h:
        stop = asyncio.Event()
        streamer = asyncio.create_task(stream(h, stop))
        try:
            goal = await goal_accepted(h)

            busy = await record_episode.send_goal(
                h,
                record_episode.GoalRequestData(task="second"),
                30.0,
            )
            assert busy.refusal == record_episode.Failure(
                record_episode.FailureKind.PRECONDITION_FAILED, "an episode is already recording"
            )

            # One feedback message per recorded frame; disk_free rides it.
            first = await asyncio.wait_for(goal.on_next_feedback(), 240.0)
            second = await asyncio.wait_for(goal.on_next_feedback(), 60.0)
            assert first.frames_recorded >= 1
            assert second.frames_recorded > first.frames_recorded
            assert first.disk_free_bytes > 0
            # Cancelled once the episode holds enough frames to be saved.
            latest = second
            while latest.frames_recorded < MIN_EPISODE_FRAMES:
                latest = await asyncio.wait_for(goal.on_next_feedback(), 60.0)

            cancel = await goal.cancel_goal(30.0)
            assert cancel.state == record_episode.CancelState.SIGNALLED

            result = await goal.get_result(180.0)
            assert result.status == record_episode.ResultStatus.CANCELLED
            assert result.cancel_reason == "cancelled by its caller"
            assert result.data is not None
            assert result.data.episode_index == 0
            assert result.data.frames_recorded >= latest.frames_recorded
            assert result.data.video_frames_missing == 0

            # d) finish_session rolls the saved session over to a fresh one.
            (recorded_dir,) = session_dirs(tmp_path)
            finished = await finish_rolled(h)
            assert finished.session == recorded_dir.name
            assert (recorded_dir / "dataset" / "meta" / "info.json").is_file()
            # The roll opened a successor session directory in place.
            assert len(session_dirs(tmp_path)) == 2

            # The rolled-to session holds no episodes yet: the node gives an
            # error reply that says so.
            with pytest.raises(finish_session.Failure) as raised:
                await finish_session.poll(h, 30.0)
            assert raised.value == finish_session.Failure(
                finish_session.FailureKind.PRECONDITION_FAILED, "nothing recorded this session"
            )
        finally:
            stop.set()
            await streamer
