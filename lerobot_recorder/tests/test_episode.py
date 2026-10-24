"""Recorder state-machine tests against a fake sink and goal context, over
the real generated peppygen modules (no messaging: the goal context and
cancellation token are local fakes; wire-level coverage lives in
test_harness.py). The recorder's environment reads (clock, disk, depth
probe, pacing) are injected through its constructor: the wall clock stands
in for the daemon-resolved clock these tests run without, and no plan here
binds rgbd cameras unless a test says so."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from peppygen.exposed_actions.recording import record_episode
from peppygen.exposed_services.recording import finish_session, resume_session

from lerobot_recorder import episode, recording
from lerobot_recorder.episode import Recorder, Timing
from lerobot_recorder.recording import (
    Cache,
    CameraFrame,
    JointSample,
    LinkLayout,
    SourceSchema,
)
from lerobot_recorder.sink import ResumeManifest, VideoShortfall
from tests.test_recording import ARM0, STALENESS_S, joint_entry, make_plan


async def no_depth_units(_node_runner, _plan):
    return []


def free_disk(*readings: int):
    """A disk_usage whose only read field reports each of `readings` in turn,
    repeating the last one once the earlier ones are spent."""
    remaining = list(readings)

    def usage(_path):
        free = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return SimpleNamespace(total=0, used=0, free=free)

    return usage


def fresh_joints(n=2) -> JointSample:
    return JointSample(
        positions=(0.0,) * n, velocities=(0.0,) * n, efforts=(), timestamp_ns=time.time_ns()
    )


def arm_plan(color=0):
    return make_plan(state=(joint_entry(),), color=color)


def fresh_rgb(w=4, h=2) -> CameraFrame:
    return CameraFrame(
        encoding="rgb8", width=w, height=h, data=bytes(w * h * 3), timestamp_ns=time.time_ns()
    )


def goal_request(task="t"):
    return SimpleNamespace(data=SimpleNamespace(task=task))


class FakeSink:
    def __init__(
        self,
        fps=100,
        fail_save=0,
        fail_add=False,
        add_delay_s=0.0,
        created=True,
        video_missing=None,
    ):
        self.fps = fps
        self.created = created
        self.frames = 0
        self.saves = 0
        self.discards = 0
        self.creates = 0
        self.finalizes = 0
        self.finalized = False
        # Resume surface, mirroring the real Sink: the manifest the preflight
        # reports, and the failure each half may raise instead.
        self.preflight_episodes = 0
        self.preflight_error: Exception | None = None
        self.resume_error: Exception | None = None
        self._fail_save = fail_save
        self._fail_add = fail_add
        self._add_delay_s = add_delay_s
        # Per camera whose video comes up short, the frames it misses.
        self._video_missing = video_missing or {}

    @property
    def episodes_saved(self) -> int:
        # Mirrors the real Sink: the count lives in dataset metadata, so
        # asking before creation is a bug in the caller, not a 0.
        assert self.created, "created at the first accepted goal"
        return self.saves

    async def create(self, schema, plan) -> None:
        assert not self.created, "one dataset per session"
        self.creates += 1
        self.created = True

    def add_frame(self, row, task) -> None:
        assert self.created and not self.finalized, "frames need an open dataset"
        if self._fail_add:
            raise ValueError("boom")
        if self._add_delay_s:
            time.sleep(self._add_delay_s)
        self.frames += 1

    async def save_episode(self) -> VideoShortfall:
        assert self.created and not self.finalized, "saving needs an open dataset"
        if self._fail_save > 0:
            self._fail_save -= 1
            raise OSError("disk full")
        self.saves += 1
        return VideoShortfall(dict(self._video_missing))

    def discard_open_frames(self) -> None:
        assert self.created, "discarding needs an open dataset"
        self.discards += 1

    async def finalize(self) -> None:
        # Mirrors the real Sink: idempotent, a no-op with nothing open, and
        # add_frame/save_episode refuse after it.
        if not self.created or self.finalized:
            return
        self.finalizes += 1
        self.finalized = True

    async def preflight_resume(self) -> ResumeManifest:
        if self.preflight_error is not None:
            raise self.preflight_error
        return ResumeManifest(
            robot_type="bot", fps=self.fps, features={}, total_episodes=self.preflight_episodes
        )

    async def resume(self, manifest, schema, plan) -> None:
        assert not self.created, "one dataset per session"
        if self.resume_error is not None:
            raise self.resume_error
        self.created = True
        self.saves = manifest.total_episodes


class FailingCreateSink(FakeSink):
    async def create(self, schema, plan) -> None:
        raise OSError("mkdir failed")


# The kinds of the record_episode goals the recorder fails.
ATTEMPT_FAILED = record_episode.FailureKind.ATTEMPT_FAILED
PRECONDITION_FAILED = record_episode.FailureKind.PRECONDITION_FAILED


# The reason the generated goal context gives for each cause of a cancel
# signal, in peppy's words.
CANCEL_REASONS = {"caller": "cancelled by its caller", "sender_gone": "its sender is gone"}


class FakeCtx:
    """Records how the goal ended: ("cancelled", reason, episode_index,
    frames_recorded, video_frames_missing) or ("failed", kind, message). The goal of a
    record_episode never completes, so the fake has no complete."""

    def __init__(self, task="t"):
        self._request = goal_request(task)
        self._cancel = asyncio.Event()
        self._cause = None
        self.endings = []
        self.feedbacks = []

    def request(self):
        return self._request

    def request_cancel(self, cause="caller") -> None:
        self._cause = cause
        self._cancel.set()

    async def cancel_signal(self) -> None:
        await self._cancel.wait()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel_cause(self):
        return self._cause

    def cancel_reason(self):
        return CANCEL_REASONS.get(self._cause)

    async def publish_feedback(self, frames_recorded, disk_free_bytes) -> None:
        self.feedbacks.append((frames_recorded, disk_free_bytes))

    async def fail(self, kind, message) -> None:
        self.endings.append(("failed", kind, message))

    async def complete_cancelled(
        self, reason, episode_index, frames_recorded, video_frames_missing
    ) -> None:
        self.endings.append(
            ("cancelled", reason, episode_index, frames_recorded, video_frames_missing)
        )

    def ending(self):
        """The one way the goal ended."""
        (ending,) = self.endings
        return ending


class FakeToken:
    def __init__(self):
        self._event = asyncio.Event()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def cancelled(self) -> None:
        await self._event.wait()


def recorder_with(plan, sink, tmp_path, **overrides) -> tuple[Recorder, Cache]:
    cache = Cache.for_plan(plan)
    overrides.setdefault("now_ns", time.time_ns)
    overrides.setdefault("depth_probe", no_depth_units)
    recorder = Recorder(plan, cache, sink, tmp_path, STALENESS_S, 1 << 30, **overrides)
    return recorder, cache


async def _until(predicate, *, tick=None, poll_s=0.005):
    """Yield to the loop until a background task makes `predicate` true.
    `tick` runs each poll: a live source re-stamps, or nothing stays fresh
    long enough for the warm-up retry to see it."""
    while not predicate():
        if tick is not None:
            tick()
        await asyncio.sleep(poll_s)


def capture_now(cache, plan):
    return recording.capture_schema(cache, plan, None, time.time_ns(), STALENESS_S)


def test_save_failure_recovers_for_next_goal(tmp_path):
    plan = arm_plan()
    sink = FakeSink(fps=100, fail_save=1)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        # Stamped once: staleness ends each episode after STALENESS_S.
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        token = FakeToken()
        first = FakeCtx()
        await recorder._record(first, token)
        # The episode could not be saved: the goal fails, with no result.
        ending, kind, message = first.ending()
        assert (ending, kind) == ("failed", ATTEMPT_FAILED)
        assert "episode save failed" in message
        assert sink.discards == 1

        cache.links[ARM0] = fresh_joints()
        second = FakeCtx()
        await recorder._record(second, token)
        # The recorder ended the episode itself on a stale source, with a
        # save: the goal ends cancelled, for a reason that names the cause.
        ending, reason, index, frames, missing = second.ending()
        assert (ending, index, missing) == ("cancelled", 0, 0)
        assert frames > 0
        assert "stale" in reason
        assert sink.saves == 1

    asyncio.run(run())


def test_a_too_short_failure_reports_the_shortfall_and_what_ended_it():
    # Both halves are needed: an episode cut short by a low disk is a disk
    # problem, not a recording-technique problem.
    both = episode._too_short(1, "only 12 MB free under the session directory")
    assert "need 2" in both and "12 MB free" in both
    # Nothing captured at all still says why it ended.
    assert "cancelled by its caller" in episode._too_short(0, "cancelled by its caller")


def test_a_one_frame_episode_fails_rather_than_being_saved(tmp_path):
    # The library's episode stats change dtype below two samples, so saving
    # this would write a schema every later episode is rejected against.
    plan = arm_plan()
    sink = FakeSink(fps=1)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        ending, kind, message = ctx.ending()
        assert (ending, kind) == ("failed", ATTEMPT_FAILED)
        assert "too short to save: 1 frame(s), need 2" in message
        # What ended it survives alongside the shortfall.
        assert "stale" in message
        assert sink.saves == 0
        assert sink.discards == 1

    asyncio.run(run())


def test_a_too_short_episode_leaves_the_session_recordable(tmp_path):
    # The wedge this guards against is session-wide, so the next episode has
    # to still be savable.
    plan = arm_plan()
    sink = FakeSink(fps=1)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        token = FakeToken()
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        first = FakeCtx()
        await recorder._record(first, token)
        assert first.ending()[0] == "failed"  # too short

        sink.fps = 100
        cache.links[ARM0] = fresh_joints()
        second = FakeCtx()
        await recorder._record(second, token)
        ending, _reason, index, frames, missing = second.ending()
        # Index 0: a failed episode saves nothing, so it consumes no episode
        # index.
        assert (ending, index, missing) == ("cancelled", 0, 0)
        assert frames >= recording.MIN_EPISODE_FRAMES
        assert sink.saves == 1

    asyncio.run(run())


def test_add_frame_failure_fails_the_goal_and_discards(tmp_path):
    plan = arm_plan()
    sink = FakeSink(fail_add=True)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        ending, kind, message = ctx.ending()
        assert (ending, kind) == ("failed", ATTEMPT_FAILED)
        assert message == "frame write failed: boom"
        assert sink.discards == 1
        assert sink.saves == 0

    asyncio.run(run())


def test_schedule_lag_ends_episode_with_save(tmp_path):
    plan = arm_plan()
    # 20 ms per frame against a 10 ms period: lag grows every tick.
    sink = FakeSink(fps=100, add_delay_s=0.02)
    recorder, cache = recorder_with(
        plan, sink, tmp_path, timing=Timing(max_schedule_lag_s=0.15)
    )

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)

        async def refresher():
            while True:
                cache.links[ARM0] = fresh_joints()
                await asyncio.sleep(0.02)

        keep_fresh = asyncio.create_task(refresher())
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        keep_fresh.cancel()
        ending, reason, _index, frames, missing = ctx.ending()
        assert (ending, missing) == ("cancelled", 0)
        assert reason.startswith("cannot sustain 100 fps")
        assert frames > 0
        assert sink.saves == 1

    asyncio.run(run())


def test_disk_floor_trips_on_its_own_cadence(tmp_path):
    # Roomy on the first check and empty after, so the floor can only trip on
    # a later one: a tick-by-tick check would end the episode at frame one.
    plan = arm_plan()
    sink = FakeSink(fps=100)
    recorder, cache = recorder_with(
        plan,
        sink,
        tmp_path,
        disk_usage=free_disk(1 << 40, 0),
        timing=Timing(disk_check_period_s=0.05),
    )

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        ending, reason, _index, frames, missing = ctx.ending()
        assert (ending, missing) == ("cancelled", 0)
        assert reason == "only 0 MB free under the session directory"
        assert frames >= recording.MIN_EPISODE_FRAMES
        assert sink.saves == 1

    asyncio.run(run())


def test_feedback_per_frame_with_slow_disk_field(tmp_path):
    # Disk stays above the floor; the episode ends on state staleness.
    plan = arm_plan()
    recorder, cache = recorder_with(
        plan,
        FakeSink(fps=100),
        tmp_path,
        disk_usage=free_disk(1 << 40),
        timing=Timing(disk_check_period_s=60.0),
    )

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        assert len(ctx.feedbacks) > 1
        frames_seen = [frames for frames, _free in ctx.feedbacks]
        assert frames_seen == list(range(1, len(frames_seen) + 1))
        # The slow field repeats its last measurement on every frame.
        assert all(free == 1 << 40 for _frames, free in ctx.feedbacks)

    asyncio.run(run())


def test_decide_refuses_while_recording(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)
    recorder._recording = True
    decision = recorder._decide(goal_request())
    assert decision.refusal == record_episode.Failure(
        PRECONDITION_FAILED, "an episode is already recording"
    )


def test_decide_claims_the_slot_at_admission(tmp_path):
    """finish_session must not slip between accepting a goal and the run
    loop starting its episode."""
    recorder, cache = recorder_with(arm_plan(), FakeSink(), tmp_path)
    cache.links[ARM0] = fresh_joints()
    recorder._schema = capture_now(cache, arm_plan())

    assert recorder._decide(goal_request()).refusal is None
    assert recorder._recording
    second = recorder._decide(goal_request())
    assert second.refusal.kind is PRECONDITION_FAILED
    assert "already recording" in second.refusal.message


def test_decide_refuses_while_finishing(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)
    recorder._finishing = True
    decision = recorder._decide(goal_request())
    assert decision.refusal == record_episode.Failure(
        PRECONDITION_FAILED, "finishing the session"
    )
    assert not recorder._recording


def test_decide_refuses_instead_of_raising(tmp_path):
    """A decider that raises would kill the goal server for the node's life,
    leaving every later goal undecided."""

    def exploding_disk_usage(path):
        raise OSError("session directory is gone")

    recorder, _ = recorder_with(
        arm_plan(), FakeSink(), tmp_path, disk_usage=exploding_disk_usage
    )
    decision = recorder._decide(goal_request())
    assert decision.refusal.kind is ATTEMPT_FAILED
    assert "cannot start an episode" in decision.refusal.message
    assert not recorder._recording


def test_goal_refused_below_the_disk_floor(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path, disk_usage=free_disk(1))
    decision = recorder._decide(goal_request())
    assert decision.refusal == record_episode.Failure(
        PRECONDITION_FAILED, "only 0 MB free under the session directory"
    )
    assert not recorder._recording


def test_goal_refused_before_the_clock_is_ready(tmp_path):
    def unstarted_clock():
        raise RuntimeError("peppy clock not initialized")

    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path, now_ns=unstarted_clock)
    decision = recorder._decide(goal_request())
    assert decision.refusal.kind is PRECONDITION_FAILED
    assert "clock not ready" in decision.refusal.message


def test_goal_refused_when_camera_dead_after_first_episode(tmp_path):
    plan = arm_plan(color=1)
    recorder, cache = recorder_with(plan, FakeSink(), tmp_path)
    cache.links[ARM0] = fresh_joints()
    cache.color = [fresh_rgb()]
    recorder._schema = capture_now(cache, plan)

    stale = CameraFrame(
        encoding="rgb8", width=4, height=2, data=bytes(24),
        timestamp_ns=time.time_ns() - int(1.5e9),
    )
    cache.color = [stale]
    reason = recorder._refuse_reason()
    assert reason is not None
    assert "no fresh frame" in reason


def test_goal_before_creation_probes_no_depth_units(tmp_path):
    """Before the dataset exists there are no units to probe; fabricating
    them refused every pre-creation rgbd goal with a camera fault the camera
    never reported."""
    plan = make_plan(state=(joint_entry(),), rgbd=1)
    recorder, cache = recorder_with(plan, FakeSink(), tmp_path)
    cache.links[ARM0] = fresh_joints()
    cache.rgbd_video = [fresh_rgb()]
    cache.rgbd_depth = [fresh_rgb()]
    assert recorder._refuse_reason() is None


def test_goal_refused_on_schema_drift(tmp_path):
    plan = arm_plan()
    recorder, cache = recorder_with(plan, FakeSink(), tmp_path)
    cache.links[ARM0] = fresh_joints(n=2)
    recorder._schema = capture_now(cache, plan)

    cache.links[ARM0] = fresh_joints(n=3)
    reason = recorder._refuse_reason()
    assert reason is not None
    assert "changed shape" in reason


def test_create_failure_fails_the_goal_and_leaves_schema_unset(tmp_path):
    plan = arm_plan()
    sink = FailingCreateSink(created=False)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        ctx = FakeCtx()
        await recorder._run_episode(None, ctx, FakeToken())
        assert ctx.ending() == ("failed", ATTEMPT_FAILED, "dataset create failed: mkdir failed")
        assert recorder._schema is None
        # The next goal re-probes instead of assuming a dataset exists.
        assert recorder._refuse_reason() is None

    asyncio.run(run())


def test_contain_episode_failure_fails_the_goal(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)

    async def run():
        ctx = FakeCtx()
        await recorder._contain_episode_failure(ctx, RuntimeError("unexpected"))
        assert ctx.ending() == ("failed", ATTEMPT_FAILED, "internal error: unexpected")

    asyncio.run(run())


def test_tick_report_prints_only_overruns(capsys):
    report = episode._TickReport(0.01)
    report.record(0.0, sample_s=0.001, append_s=0.002)
    report.flush()
    assert capsys.readouterr().out == ""
    report.record(0.0, sample_s=0.020, append_s=0.005)
    report.record(0.1, sample_s=0.001, append_s=0.030)
    report.flush()
    out = capsys.readouterr().out
    assert "2 tick overrun(s)" in out
    assert "append 30 ms" in out
    # Flushed and re-armed: a healthy stretch after the burst stays silent.
    report.flush()
    assert capsys.readouterr().out == ""


def test_recorder_rejects_nonpositive_staleness(tmp_path):
    plan = arm_plan()
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="max_staleness_s"):
            Recorder(plan, Cache.for_plan(plan), FakeSink(), tmp_path, bad, 1 << 30)


def test_max_episode_length_ends_with_save(tmp_path):
    # 5 frames at fps=100: the bound trips within the first fresh window.
    plan = arm_plan()
    sink = FakeSink(fps=100)
    recorder, cache = recorder_with(plan, sink, tmp_path, timing=Timing(max_episode_s=0.05))

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        assert ctx.ending() == ("cancelled", "max episode length reached", 0, 5, 0)
        assert sink.saves == 1

    asyncio.run(run())


class StopAtFrame(FakeCtx):
    """A goal context that runs `stop` as the feedback of frame `frame` goes
    out: the next tick of the episode finds it, whatever the host's speed."""

    def __init__(self, frame, stop):
        super().__init__()
        self._frame = frame
        self._stop = stop

    async def publish_feedback(self, frames_recorded, disk_free_bytes) -> None:
        await super().publish_feedback(frames_recorded, disk_free_bytes)
        if frames_recorded == self._frame:
            self._stop(self)


def recorder_on_a_still_clock(tmp_path, sink):
    """A recorder whose clock stands at the stamp of its one fresh sample, so
    no source goes stale whatever the host's speed: only the test ends the
    episode, before its 100-frame bound."""
    plan = arm_plan()
    stamp = time.time_ns()
    recorder, cache = recorder_with(
        plan, sink, tmp_path, now_ns=lambda: stamp, timing=Timing(max_episode_s=1.0)
    )
    cache.links[ARM0] = JointSample(
        positions=(0.0, 0.0), velocities=(0.0, 0.0), efforts=(), timestamp_ns=stamp
    )
    recorder._schema = recording.capture_schema(cache, plan, None, stamp, STALENESS_S)
    return recorder


@pytest.mark.parametrize(
    ("cause", "reason"),
    [("caller", "cancelled by its caller"), ("sender_gone", "its sender is gone")],
)
def test_a_cancel_saves_the_episode_and_ends_the_goal_cancelled(tmp_path, cause, reason):
    sink = FakeSink(fps=100)
    recorder = recorder_on_a_still_clock(tmp_path, sink)
    ctx = StopAtFrame(2, lambda ctx: ctx.request_cancel(cause))

    asyncio.run(recorder._record(ctx, FakeToken()))
    # The episode saved: the goal ends cancelled with its result, for the
    # reason of the cause of the cancel.
    assert ctx.ending() == ("cancelled", reason, 0, 2, 0)
    assert sink.saves == 1


def test_a_shutdown_saves_the_episode_and_ends_the_goal_cancelled(tmp_path):
    sink = FakeSink(fps=100)
    recorder = recorder_on_a_still_clock(tmp_path, sink)
    token = FakeToken()
    ctx = StopAtFrame(2, lambda _ctx: token._event.set())

    async def run():
        await recorder._record(ctx, token)

    asyncio.run(run())
    assert ctx.ending() == ("cancelled", "recorder shutting down", 0, 2, 0)
    assert sink.saves == 1


def test_a_cancel_before_two_frames_fails_the_goal_with_nothing_saved(tmp_path):
    sink = FakeSink(fps=100)
    recorder = recorder_on_a_still_clock(tmp_path, sink)

    async def run():
        ctx = FakeCtx()
        ctx.request_cancel()
        await recorder._record(ctx, FakeToken())
        assert ctx.ending() == (
            "failed",
            ATTEMPT_FAILED,
            "too short to save: 0 frame(s), need 2 (cancelled by its caller)",
        )
        assert sink.saves == 0

    asyncio.run(run())


def test_create_failure_fails_a_cancelled_goal(tmp_path):
    plan = arm_plan()
    recorder, cache = recorder_with(plan, FailingCreateSink(created=False), tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        ctx = FakeCtx()
        ctx.request_cancel()
        await recorder._run_episode(None, ctx, FakeToken())
        # Nothing was saved, so the goal fails even though it was cancelled.
        assert ctx.ending() == ("failed", ATTEMPT_FAILED, "dataset create failed: mkdir failed")

    asyncio.run(run())


def test_depth_unit_drift_ends_goal_before_recording(tmp_path):
    async def drifted_probe(_node_runner, _plan):
        return [0.002]

    plan = make_plan(state=(joint_entry(),), rgbd=1)
    sink = FakeSink()
    recorder, _cache = recorder_with(plan, sink, tmp_path, depth_probe=drifted_probe)
    recorder._schema = SourceSchema(
        layouts=(LinkLayout(dims=2, has_velocities=True, has_efforts=False),),
        action_layouts=(),
        color_geometry=(),
        rgbd_geometry=((4, 2),),
        depth_geometry=((4, 2),),
        depth_units=(0.001,),
    )

    async def run():
        ctx = FakeCtx()
        await recorder._run_episode(None, ctx, FakeToken())
        assert ctx.ending() == (
            "failed",
            PRECONDITION_FAILED,
            "rgbd depth unit changed since the dataset was created",
        )
        assert sink.frames == 0

    asyncio.run(run())


def resume_recorder(tmp_path, current=None, existing=None, **overrides):
    """A recorder with an empty current session and a resumable target."""
    plan = arm_plan()
    current = current or FakeSink(created=False)
    recorder, cache = recorder_with(plan, current, tmp_path, **overrides)
    if existing is not None:
        target_dir = tmp_path / "2026-07-27_10-00-00"

        def open_existing(name):
            if name != "2026-07-27_10-00-00":
                raise resume_session.Failure(
                    resume_session.FailureKind.REJECTED,
                    f"no session named {name} under this storage target",
                )
            return target_dir, existing, None

        recorder._open_existing = open_existing
    return recorder, cache


def test_resume_swaps_onto_the_existing_session(tmp_path):
    existing = FakeSink(created=False)
    existing.preflight_episodes = 3
    recorder, cache = resume_recorder(tmp_path, existing=existing)

    async def run():
        cache.links[ARM0] = fresh_joints()
        session, episodes = await recorder.resume_session("2026-07-27_10-00-00", None)
        assert (session, episodes) == ("2026-07-27_10-00-00", 3)
        assert recorder._sink is existing
        assert recorder._session_dir.name == "2026-07-27_10-00-00"
        assert recorder._schema is not None
        assert not recorder._resuming
        # Recording continues where the session left off.
        assert recorder._sink.episodes_saved == 3

    asyncio.run(run())


def test_resume_finalizes_the_abandoned_empty_session(tmp_path):
    """Warm-up may have created the current (still empty) session's dataset;
    its image-writer threads only stop at finalize."""
    current = FakeSink(created=True)
    existing = FakeSink(created=False)
    recorder, cache = resume_recorder(tmp_path, current=current, existing=existing)

    async def run():
        cache.links[ARM0] = fresh_joints()
        await recorder.resume_session("2026-07-27_10-00-00", None)
        assert current.finalized

    asyncio.run(run())


RESUME_REJECTED = resume_session.FailureKind.REJECTED
RESUME_PRECONDITION_FAILED = resume_session.FailureKind.PRECONDITION_FAILED
RESUME_ATTEMPT_FAILED = resume_session.FailureKind.ATTEMPT_FAILED


def test_resume_error_replies_leave_state_untouched(tmp_path):
    existing = FakeSink(created=False)
    recorder, cache = resume_recorder(tmp_path, existing=existing)
    original_sink = recorder._sink
    original_dir = recorder._session_dir

    async def refused(name="2026-07-27_10-00-00") -> tuple:
        with pytest.raises(resume_session.Failure) as raised:
            await recorder.resume_session(name, None)
        assert recorder._sink is original_sink
        assert recorder._session_dir is original_dir
        assert recorder._schema is None
        assert not recorder._resuming
        return raised.value.kind, raised.value.message

    async def run():
        recorder._recording = True
        assert await refused() == (RESUME_PRECONDITION_FAILED, "an episode is recording")
        recorder._recording = False

        recorder._finishing = True
        assert await refused() == (RESUME_PRECONDITION_FAILED, "finishing the session")
        recorder._finishing = False

        recorder._roll_pending = True
        kind, message = await refused()
        assert kind is RESUME_PRECONDITION_FAILED and "retry finish_session" in message
        recorder._roll_pending = False

        # Unknown name from the factory.
        kind, message = await refused("2026-01-01_00-00-00")
        assert kind is RESUME_REJECTED and "no session named" in message

        # A stored session with nothing to append to.
        existing.preflight_error = ValueError("session has no saved episodes")
        assert await refused() == (RESUME_REJECTED, "session has no saved episodes")
        existing.preflight_error = None

        # Sources not live: schema capture fails.
        kind, message = await refused()
        assert kind is RESUME_PRECONDITION_FAILED and "not ready to resume" in message

        # The live sources no longer match the session.
        cache.links[ARM0] = fresh_joints()
        existing.resume_error = ValueError("dataset fps is 30, this recorder runs 100")
        kind, message = await refused()
        assert kind is RESUME_PRECONDITION_FAILED and "cannot resume" in message
        existing.resume_error = None

        # The resume itself fails.
        existing.resume_error = OSError("disk unplugged")
        assert await refused() == (RESUME_ATTEMPT_FAILED, "resume failed: disk unplugged")
        existing.resume_error = None

        # A current session with saved episodes must be finished first.
        recorder._sink.created = True
        recorder._sink.saves = 2
        kind, message = await refused()
        assert kind is RESUME_PRECONDITION_FAILED and "finish it first" in message

    asyncio.run(run())


def test_resume_refuses_the_current_session_name(tmp_path):
    recorder, _ = resume_recorder(tmp_path)
    recorder._open_existing = lambda name: (_ for _ in ()).throw(AssertionError)
    recorder._session_dir = tmp_path / "2026-07-27_10-00-00"

    async def run():
        with pytest.raises(resume_session.Failure) as raised:
            await recorder.resume_session("2026-07-27_10-00-00", None)
        assert raised.value == resume_session.Failure(
            RESUME_PRECONDITION_FAILED, "that session is currently recording"
        )

    asyncio.run(run())


def test_resume_unavailable_without_a_factory(tmp_path):
    recorder, _ = resume_recorder(tmp_path)

    async def run():
        with pytest.raises(resume_session.Failure) as raised:
            await recorder.resume_session("2026-07-27_10-00-00", None)
        assert raised.value == resume_session.Failure(
            RESUME_PRECONDITION_FAILED, "session resume unavailable"
        )

    asyncio.run(run())


def test_goals_and_finish_refused_while_resuming(tmp_path):
    recorder, _cache = resume_recorder(tmp_path)
    recorder._resuming = True
    decision = recorder._decide(goal_request())
    assert decision.refusal == record_episode.Failure(PRECONDITION_FAILED, "resuming a session")

    async def run():
        with pytest.raises(finish_session.Failure) as raised:
            await recorder.finish_session()
        assert raised.value == finish_session.Failure(
            finish_session.FailureKind.PRECONDITION_FAILED, "resuming a session"
        )
        with pytest.raises(resume_session.Failure) as raised:
            await recorder.resume_session("2026-07-27_10-00-00", None)
        assert raised.value == resume_session.Failure(
            RESUME_PRECONDITION_FAILED, "already resuming"
        )

    asyncio.run(run())


class DelayedCreateSink(FakeSink):
    """create() that parks mid-flight so a resume can race it."""

    def __init__(self):
        super().__init__(created=False)
        self.proceed = asyncio.Event()
        self.create_started = asyncio.Event()

    async def create(self, schema, plan) -> None:
        self.create_started.set()
        await self.proceed.wait()
        await super().create(schema, plan)


def test_resume_serializes_against_a_warmup_create_in_flight(tmp_path):
    current = DelayedCreateSink()
    existing = FakeSink(created=False)
    existing.preflight_episodes = 1
    recorder, cache = resume_recorder(
        tmp_path, current=current, existing=existing, timing=Timing(warmup_retry_s=0.005)
    )
    token = FakeToken()
    runner = SimpleNamespace(cancellation_token=lambda: token)

    def keep_live():
        cache.links[ARM0] = fresh_joints()

    async def run():
        warmup = asyncio.create_task(recorder.warm_up(runner))
        await asyncio.wait_for(
            _until(lambda: current.create_started.is_set(), tick=keep_live), timeout=5
        )
        # Warm-up holds the create lock mid-create; fire the resume and let
        # the parked create finish underneath it.
        resume = asyncio.create_task(
            recorder.resume_session("2026-07-27_10-00-00", runner)
        )
        await asyncio.sleep(0.02)
        assert not resume.done()
        current.proceed.set()
        keep_live()
        session, _episodes = await asyncio.wait_for(resume, timeout=5)
        assert session
        # The in-flight create landed on the OLD sink; the resumed sink never
        # saw a create, and the schema is resume's capture.
        assert current.created and current.finalized
        assert existing.creates == 0
        assert recorder._sink is existing
        assert recorder._schema is not None
        token._event.set()
        await asyncio.wait_for(warmup, timeout=5)

    asyncio.run(run())


def test_warmup_creates_dataset_before_any_goal(tmp_path):
    plan = arm_plan()
    sink = FakeSink(created=False)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        # No rgbd cameras in the plan, so the runner is never touched.
        await recorder._ensure_created(None)
        assert sink.created
        assert recorder._schema is not None
        # A second call is a no-op, not a second create.
        await recorder._ensure_created(None)
        assert sink.creates == 1

    asyncio.run(run())


def test_warmup_waits_for_a_live_source_then_creates(tmp_path):
    plan = arm_plan()
    sink = FakeSink(created=False)
    recorder, cache = recorder_with(plan, sink, tmp_path, timing=Timing(warmup_retry_s=0.005))
    token = FakeToken()
    runner = SimpleNamespace(cancellation_token=lambda: token)

    def keep_live():
        cache.links[ARM0] = fresh_joints()

    async def run():
        warmup = asyncio.create_task(recorder.warm_up(runner))
        # Nothing has produced yet, so schema capture keeps failing.
        await asyncio.sleep(0.02)
        assert not sink.created
        await asyncio.wait_for(_until(lambda: sink.created, tick=keep_live), timeout=5)
        token._event.set()
        await asyncio.wait_for(warmup, timeout=5)

    asyncio.run(run())


def test_warmup_rearms_the_session_finish_session_rolls_to(tmp_path):
    plan = arm_plan()
    first_sink = FakeSink(created=False)
    recorder, cache = recorder_with(plan, first_sink, tmp_path, timing=Timing(warmup_retry_s=0.005))
    second_sink = FakeSink(created=False)
    next_dir = tmp_path / "next"
    token = FakeToken()
    runner = SimpleNamespace(cancellation_token=lambda: token)

    def open_session():
        next_dir.mkdir()
        return next_dir, second_sink, None

    def keep_live():
        cache.links[ARM0] = fresh_joints()

    recorder._open_session = open_session

    async def run():
        warmup = asyncio.create_task(recorder.warm_up(runner))
        await asyncio.wait_for(
            _until(lambda: recorder._schema is not None, tick=keep_live), timeout=5
        )
        first_sink.saves = 2
        await recorder.finish_session()
        # The rolled-to session warms up on its own; without a persistent
        # warm-up its first episode would pay dataset creation at Record.
        await asyncio.wait_for(
            _until(lambda: second_sink.created, tick=keep_live), timeout=5
        )
        token._event.set()
        await asyncio.wait_for(warmup, timeout=5)

    asyncio.run(run())


def test_finish_session_rolls_to_a_fresh_session(tmp_path):
    plan = arm_plan()
    first_sink = FakeSink(created=False)
    recorder, cache = recorder_with(plan, first_sink, tmp_path)
    second_sink = FakeSink(created=False)
    next_dir = tmp_path / "next"

    def open_session():
        next_dir.mkdir()
        return next_dir, second_sink, None

    recorder._open_session = open_session

    async def run():
        cache.links[ARM0] = fresh_joints()
        await recorder._ensure_created(None)
        first_sink.saves = 2
        name = await recorder.finish_session()
        assert name == tmp_path.name
        assert first_sink.finalized
        assert recorder._sink is second_sink
        assert recorder._session_dir == next_dir
        assert recorder._schema is None

    asyncio.run(run())


FINISH_PRECONDITION_FAILED = finish_session.FailureKind.PRECONDITION_FAILED
FINISH_ATTEMPT_FAILED = finish_session.FailureKind.ATTEMPT_FAILED


async def finish_error_reply(recorder) -> finish_session.Failure:
    """The error reply finish_session gives now."""
    with pytest.raises(finish_session.Failure) as raised:
        await recorder.finish_session()
    return raised.value


def test_finish_session_refuses_when_empty_or_recording(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)
    recorder._open_session = lambda: (tmp_path, FakeSink(created=False), None)

    async def run():
        assert await finish_error_reply(recorder) == finish_session.Failure(
            FINISH_PRECONDITION_FAILED, "nothing recorded this session"
        )
        recorder._sink.saves = 1
        recorder._recording = True
        assert await finish_error_reply(recorder) == finish_session.Failure(
            FINISH_PRECONDITION_FAILED, "an episode is recording"
        )
        recorder._recording = False
        recorder._finishing = True
        assert await finish_error_reply(recorder) == finish_session.Failure(
            FINISH_PRECONDITION_FAILED, "already finishing"
        )

    asyncio.run(run())


def test_finish_session_refuses_before_the_dataset_exists(tmp_path):
    """An uncreated dataset cannot answer episodes_saved, so the emptiness
    guard must lead with it rather than letting the assertion escape."""
    recorder, _ = recorder_with(arm_plan(), FakeSink(created=False), tmp_path)
    recorder._open_session = lambda: (tmp_path, FakeSink(created=False), None)

    async def run():
        assert await finish_error_reply(recorder) == finish_session.Failure(
            FINISH_PRECONDITION_FAILED, "nothing recorded this session"
        )

    asyncio.run(run())


def test_finish_session_refuses_without_a_session_factory(tmp_path):
    sink = FakeSink()
    sink.saves = 1
    recorder, _ = recorder_with(arm_plan(), sink, tmp_path)

    async def run():
        assert await finish_error_reply(recorder) == finish_session.Failure(
            FINISH_PRECONDITION_FAILED, "session rolling unavailable"
        )
        assert not sink.finalized

    asyncio.run(run())


def test_finish_session_roll_failure_refuses_goals_until_retried(tmp_path):
    """After a successful finalize, a failed roll must not leave the
    finalized sink accepting goals; a finish_session retry completes the
    roll without re-finalizing."""
    sink = FakeSink()
    sink.saves = 1
    recorder, _ = recorder_with(arm_plan(), sink, tmp_path)
    next_sink = FakeSink(created=False)
    attempts = []

    def failing_then_working():
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("mkdir failed")
        return tmp_path / "next", next_sink, None

    recorder._open_session = failing_then_working

    async def run():
        assert await finish_error_reply(recorder) == finish_session.Failure(
            FINISH_ATTEMPT_FAILED, "finish failed: mkdir failed"
        )
        assert sink.finalized
        # The finalized sink must not take another episode.
        decision = recorder._decide(goal_request())
        assert decision.refusal.kind is PRECONDITION_FAILED
        assert "retry finish_session" in decision.refusal.message
        assert not recorder._recording
        name = await recorder.finish_session()
        assert name == tmp_path.name
        # The retry rolled without closing the already-finalized sink again.
        assert sink.finalizes == 1
        assert recorder._sink is next_sink
        assert not recorder._roll_pending

    asyncio.run(run())


class CountingMirror:
    def __init__(self):
        self.syncs = 0

    def sync(self) -> tuple[int, int]:
        self.syncs += 1
        return 0, 0


def test_shutdown_skips_final_mirror_when_finalize_fails(tmp_path):
    """Footer-less parquet must never be stamped as the final remote copy."""
    sink = FakeSink()

    async def failing_finalize():
        raise OSError("disk full")

    sink.finalize = failing_finalize
    recorder, _ = recorder_with(arm_plan(), sink, tmp_path)
    mirror = CountingMirror()
    recorder._mirror = mirror

    asyncio.run(recorder.close_for_shutdown())
    assert mirror.syncs == 0


def test_shutdown_runs_final_mirror_after_successful_finalize(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)
    mirror = CountingMirror()
    recorder._mirror = mirror

    asyncio.run(recorder.close_for_shutdown())
    assert recorder._sink.finalized
    assert mirror.syncs == 1


def test_finish_session_keeps_the_session_when_finalize_fails(tmp_path):
    """A failed finalize must not roll: the operator retries in place."""
    sink = FakeSink()
    sink.saves = 1

    async def failing_finalize():
        raise OSError("parquet footer failed")

    sink.finalize = failing_finalize
    recorder, _ = recorder_with(arm_plan(), sink, tmp_path)
    recorder._open_session = lambda: pytest.fail("rolled despite a failed finalize")

    async def run():
        assert await finish_error_reply(recorder) == finish_session.Failure(
            FINISH_ATTEMPT_FAILED, "finish failed: parquet footer failed"
        )
        assert recorder._sink is sink
        # Not left latched: a retry is allowed once the fault clears.
        assert not recorder._finishing

    asyncio.run(run())


def test_a_short_video_saves_the_episode_and_gives_its_missing_frames(tmp_path, capsys):
    """The episode saved, but its videos do not cover its rows. The goal ends
    as the episode's ending says, its result gives the frames the shortest
    video misses, and the run log names each short video, for the operator
    to know before training on the episode."""
    plan = arm_plan()
    sink = FakeSink(video_missing={"cam0": 3, "cam1": 7})
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        ending, reason, index, _frames, missing = ctx.ending()
        assert (ending, index, missing) == ("cancelled", 0, 7)
        # The episode ended on a stale source, and the reason names it.
        assert "stale" in reason
        assert sink.saves == 1, "a short video is still a saved episode"

    asyncio.run(run())
    log = capsys.readouterr().out
    assert "[recorder] episode 0: video is shorter than the state rows" in log
    assert "cam0 3, cam1 7 frame(s) missing" in log
