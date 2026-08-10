"""Recorder state-machine tests against a fake sink and goal context; the
generated peppy layer is the conftest stand-in."""

import asyncio
import shutil
import time
from types import SimpleNamespace

import pytest

from lerobot_recorder import episode, recording
from lerobot_recorder.episode import Recorder
from lerobot_recorder.recording import (
    Cache,
    CameraFrame,
    JointSample,
    LinkLayout,
    SourceSchema,
)
from lerobot_recorder.sink import ResumeManifest
from tests.test_recording import ARM0, STALENESS_S, joint_entry, make_plan


def fresh_joints(n=2) -> JointSample:
    return JointSample(
        positions=(0.0,) * n, velocities=(0.0,) * n, efforts=(), stamp_ns=time.time_ns()
    )


def arm_plan(color=0):
    return make_plan(state=(joint_entry(),), color=color)


def fresh_rgb(w=4, h=2) -> CameraFrame:
    return CameraFrame(
        encoding="rgb8", width=w, height=h, data=bytes(w * h * 3), stamp_ns=time.time_ns()
    )


def goal_request(task="t"):
    return SimpleNamespace(data=SimpleNamespace(task=task))


class FakeSink:
    def __init__(self, fps=100, fail_save=0, fail_add=False, add_delay_s=0.0, created=True):
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

    async def save_episode(self) -> None:
        assert self.created and not self.finalized, "saving needs an open dataset"
        if self._fail_save > 0:
            self._fail_save -= 1
            raise OSError("disk full")
        self.saves += 1

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


class FakeCtx:
    def __init__(self, task="t"):
        self._request = goal_request(task)
        self._cancel = asyncio.Event()
        self.completions = []
        self.feedbacks = []

    def request(self):
        return self._request

    async def cancel_signal(self) -> None:
        await self._cancel.wait()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    async def publish_feedback(self, frames_recorded, disk_free_bytes) -> None:
        self.feedbacks.append((frames_recorded, disk_free_bytes))

    async def complete(self, episode_index, frames_recorded, discarded, error) -> None:
        self.completions.append(("done", episode_index, frames_recorded, discarded, error))

    async def complete_cancelled(self, episode_index, frames_recorded, discarded, error) -> None:
        self.completions.append(
            ("cancelled", episode_index, frames_recorded, discarded, error)
        )


class FakeToken:
    def __init__(self):
        self._event = asyncio.Event()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def cancelled(self) -> None:
        await self._event.wait()


def recorder_with(plan, sink, tmp_path) -> tuple[Recorder, Cache]:
    cache = Cache.for_plan(plan)
    return Recorder(plan, cache, sink, tmp_path, STALENESS_S, 1 << 30), cache


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
        kind, index, frames, discarded, error = first.completions[0]
        assert (kind, index, discarded) == ("done", -1, True)
        assert frames > 0
        assert "episode save failed" in error
        assert sink.discards == 1

        cache.links[ARM0] = fresh_joints()
        second = FakeCtx()
        await recorder._record(second, token)
        kind, index, frames, discarded, error = second.completions[0]
        assert (kind, index, discarded) == ("done", 0, False)
        assert "stale" in error
        assert sink.saves == 1

    asyncio.run(run())


def test_add_frame_failure_ends_goal_with_discard(tmp_path):
    plan = arm_plan()
    sink = FakeSink(fail_add=True)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        kind, index, frames, discarded, error = ctx.completions[0]
        assert (kind, index, frames, discarded) == ("done", -1, 0, True)
        assert "frame write failed" in error
        assert sink.discards == 1
        assert sink.saves == 0

    asyncio.run(run())


def test_schedule_lag_ends_episode_with_save(tmp_path, monkeypatch):
    monkeypatch.setattr(recording, "MAX_SCHEDULE_LAG_S", 0.15)
    plan = arm_plan()
    # 20 ms per frame against a 10 ms period: lag grows every tick.
    sink = FakeSink(fps=100, add_delay_s=0.02)
    recorder, cache = recorder_with(plan, sink, tmp_path)

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
        kind, _index, frames, discarded, error = ctx.completions[0]
        assert (kind, discarded) == ("done", False)
        assert "cannot sustain" in error
        assert frames > 0
        assert sink.saves == 1

    asyncio.run(run())


def test_disk_floor_trips_on_its_own_cadence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: SimpleNamespace(total=0, used=0, free=0)
    )
    plan = arm_plan()
    sink = FakeSink(fps=100)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        kind, _index, frames, discarded, error = ctx.completions[0]
        assert (kind, discarded) == ("done", False)
        assert "MB free" in error
        assert frames > 0
        assert sink.saves == 1

    asyncio.run(run())


def test_feedback_per_frame_with_slow_disk_field(tmp_path, monkeypatch):
    # Disk stays above the floor; the episode ends on state staleness.
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=0, used=0, free=1 << 40),
    )
    monkeypatch.setattr(recording, "DISK_CHECK_PERIOD_S", 60.0)
    plan = arm_plan()
    recorder, cache = recorder_with(plan, FakeSink(fps=100), tmp_path)

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


def test_decide_rejects_while_recording(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)
    recorder._recording = True
    decision = recorder._decide(goal_request())
    assert not decision.accepted
    assert "already recording" in decision.reason


def test_decide_claims_the_slot_at_admission(tmp_path):
    """finish_session must not slip between accepting a goal and the run
    loop starting its episode."""
    recorder, cache = recorder_with(arm_plan(), FakeSink(), tmp_path)
    cache.links[ARM0] = fresh_joints()
    recorder._schema = capture_now(cache, arm_plan())

    assert recorder._decide(goal_request()).accepted
    assert recorder._recording
    second = recorder._decide(goal_request())
    assert not second.accepted
    assert "already recording" in second.reason


def test_decide_rejects_while_finishing(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)
    recorder._finishing = True
    decision = recorder._decide(goal_request())
    assert not decision.accepted
    assert "finishing the session" in decision.reason
    assert not recorder._recording


def test_decide_rejects_instead_of_raising(tmp_path, monkeypatch):
    """A decider that raises would kill the goal server for the node's life,
    leaving every later goal undecided."""
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)

    def exploding_disk_usage(path):
        raise OSError("session directory is gone")

    monkeypatch.setattr(shutil, "disk_usage", exploding_disk_usage)
    decision = recorder._decide(goal_request())
    assert not decision.accepted
    assert "cannot start an episode" in decision.reason
    assert not recorder._recording


def test_goal_refused_below_the_disk_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: SimpleNamespace(total=0, used=0, free=1)
    )
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)
    decision = recorder._decide(goal_request())
    assert not decision.accepted
    assert "MB free" in decision.reason
    assert not recorder._recording


def test_goal_refused_before_the_clock_is_ready(tmp_path, monkeypatch):
    def unstarted_clock():
        raise RuntimeError("peppy clock not initialized")

    monkeypatch.setattr(episode.peppygen.clock, "now_ns", unstarted_clock)
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)
    decision = recorder._decide(goal_request())
    assert not decision.accepted
    assert "clock not ready" in decision.reason


def test_goal_refused_when_camera_dead_after_first_episode(tmp_path):
    plan = arm_plan(color=1)
    recorder, cache = recorder_with(plan, FakeSink(), tmp_path)
    cache.links[ARM0] = fresh_joints()
    cache.color = [fresh_rgb()]
    recorder._schema = capture_now(cache, plan)

    stale = CameraFrame(
        encoding="rgb8", width=4, height=2, data=bytes(24),
        stamp_ns=time.time_ns() - int(1.5e9),
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


def test_create_failure_leaves_schema_unset(tmp_path):
    plan = arm_plan()
    sink = FailingCreateSink(created=False)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        ctx = FakeCtx()
        await recorder._run_episode(None, ctx, FakeToken())
        kind, index, frames, discarded, error = ctx.completions[0]
        assert (kind, index, frames, discarded) == ("done", -1, 0, True)
        assert "dataset create failed" in error
        assert recorder._schema is None
        # The next goal re-probes instead of assuming a dataset exists.
        assert recorder._refuse_reason() is None

    asyncio.run(run())


def test_contain_episode_failure_completes_goal(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)

    async def run():
        ctx = FakeCtx()
        await recorder._contain_episode_failure(ctx, RuntimeError("unexpected"))
        kind, index, frames, discarded, error = ctx.completions[0]
        assert (kind, index, frames, discarded) == ("done", -1, 0, True)
        assert "internal error: unexpected" in error

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


def test_max_episode_length_ends_with_save(tmp_path, monkeypatch):
    # 5 frames at fps=100: the bound trips within the first fresh window.
    monkeypatch.setattr(recording, "MAX_EPISODE_S", 0.05)
    plan = arm_plan()
    sink = FakeSink(fps=100)
    recorder, cache = recorder_with(plan, sink, tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        recorder._schema = capture_now(cache, plan)
        ctx = FakeCtx()
        await recorder._record(ctx, FakeToken())
        kind, _index, frames, discarded, error = ctx.completions[0]
        assert (kind, discarded) == ("done", False)
        assert error == "max episode length reached"
        assert frames == 5
        assert sink.saves == 1

    asyncio.run(run())


def test_create_failure_reports_cancellation(tmp_path):
    plan = arm_plan()
    recorder, cache = recorder_with(plan, FailingCreateSink(created=False), tmp_path)

    async def run():
        cache.links[ARM0] = fresh_joints()
        ctx = FakeCtx()
        ctx._cancel.set()
        await recorder._run_episode(None, ctx, FakeToken())
        kind, index, frames, discarded, error = ctx.completions[0]
        assert (kind, index, frames, discarded) == ("cancelled", -1, 0, True)
        assert "dataset create failed" in error

    asyncio.run(run())


def test_depth_unit_drift_ends_goal_before_recording(tmp_path, monkeypatch):
    from peppygen.consumed_services.rgbd_cameras import (
        depth_stream_info as rgbd_cameras_depth_stream_info,
    )
    from peppygen.consumed_topics.rgbd_cameras import (
        video_stream as rgbd_cameras_video_stream,
    )

    plan = make_plan(state=(joint_entry(),), rgbd=1)
    sink = FakeSink()
    recorder, _cache = recorder_with(plan, sink, tmp_path)
    recorder._schema = SourceSchema(
        layouts=(LinkLayout(dims=2, has_velocities=True, has_efforts=False),),
        action_layouts=(),
        color_geometry=(),
        rgbd_geometry=((4, 2),),
        depth_geometry=((4, 2),),
        depth_units=(0.001,),
    )

    async def drifted_poll(_runner, _producer, _timeout):
        return SimpleNamespace(data=SimpleNamespace(depth_unit=0.002))

    monkeypatch.setattr(rgbd_cameras_depth_stream_info, "poll", drifted_poll)
    monkeypatch.setattr(
        rgbd_cameras_video_stream, "bound_producers", lambda _runner: [object()]
    )

    async def run():
        ctx = FakeCtx()
        await recorder._run_episode(None, ctx, FakeToken())
        kind, index, frames, discarded, error = ctx.completions[0]
        assert (kind, index, frames, discarded) == ("done", -1, 0, True)
        assert "depth unit changed" in error
        assert sink.frames == 0

    asyncio.run(run())


def resume_recorder(tmp_path, current=None, existing=None):
    """A recorder with an empty current session and a resumable target."""
    plan = arm_plan()
    current = current or FakeSink(created=False)
    recorder, cache = recorder_with(plan, current, tmp_path)
    if existing is not None:
        target_dir = tmp_path / "2026-07-27_10-00-00"

        def open_existing(name):
            if name != "2026-07-27_10-00-00":
                raise ValueError(f"no session named {name} under this storage target")
            return target_dir, existing, None

        recorder._open_existing = open_existing
    return recorder, cache


def test_resume_swaps_onto_the_existing_session(tmp_path):
    existing = FakeSink(created=False)
    existing.preflight_episodes = 3
    recorder, cache = resume_recorder(tmp_path, existing=existing)

    async def run():
        cache.links[ARM0] = fresh_joints()
        session, episodes, error = await recorder.resume_session(
            "2026-07-27_10-00-00", None
        )
        assert (session, episodes, error) == ("2026-07-27_10-00-00", 3, None)
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
        _, _, error = await recorder.resume_session("2026-07-27_10-00-00", None)
        assert error is None
        assert current.finalized

    asyncio.run(run())


def test_resume_refusals_leave_state_untouched(tmp_path):
    existing = FakeSink(created=False)
    recorder, cache = resume_recorder(tmp_path, existing=existing)
    original_sink = recorder._sink
    original_dir = recorder._session_dir

    async def refused(name="2026-07-27_10-00-00"):
        session, episodes, error = await recorder.resume_session(name, None)
        assert session == "" and episodes == 0 and error
        assert recorder._sink is original_sink
        assert recorder._session_dir is original_dir
        assert recorder._schema is None
        assert not recorder._resuming
        return error

    async def run():
        recorder._recording = True
        assert "episode is recording" in await refused()
        recorder._recording = False

        recorder._finishing = True
        assert "finishing the session" in await refused()
        recorder._finishing = False

        recorder._roll_pending = True
        assert "retry finish_session" in await refused()
        recorder._roll_pending = False

        # Unknown name from the factory.
        assert "no session named" in await refused("2026-01-01_00-00-00")

        # Preflight refusal (e.g. no saved episodes).
        existing.preflight_error = ValueError("session has no saved episodes")
        assert "no saved episodes" in await refused()
        existing.preflight_error = None

        # Sources not live: schema capture fails.
        assert "not ready to resume" in await refused()

        # Compatibility refusal from Sink.resume.
        cache.links[ARM0] = fresh_joints()
        existing.resume_error = ValueError("dataset fps is 30, this recorder runs 100")
        assert "cannot resume" in await refused()
        existing.resume_error = None

        # A current session with saved episodes must be finished first.
        recorder._sink.created = True
        recorder._sink.saves = 2
        assert "finish it first" in await refused()

    asyncio.run(run())


def test_resume_refuses_the_current_session_name(tmp_path):
    recorder, _ = resume_recorder(tmp_path)
    recorder._open_existing = lambda name: (_ for _ in ()).throw(AssertionError)
    recorder._session_dir = tmp_path / "2026-07-27_10-00-00"

    async def run():
        _, _, error = await recorder.resume_session("2026-07-27_10-00-00", None)
        assert "currently recording" in error

    asyncio.run(run())


def test_resume_unavailable_without_a_factory(tmp_path):
    recorder, _ = resume_recorder(tmp_path)

    async def run():
        _, _, error = await recorder.resume_session("2026-07-27_10-00-00", None)
        assert "resume unavailable" in error

    asyncio.run(run())


def test_goals_and_finish_refused_while_resuming(tmp_path):
    recorder, _cache = resume_recorder(tmp_path)
    recorder._resuming = True
    decision = recorder._decide(goal_request())
    assert not decision.accepted
    assert "resuming a session" in decision.reason

    async def run():
        _, error = await recorder.finish_session()
        assert "resuming a session" in error
        _, _, error = await recorder.resume_session("2026-07-27_10-00-00", None)
        assert "already resuming" in error

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


def test_resume_serializes_against_a_warmup_create_in_flight(tmp_path, monkeypatch):
    monkeypatch.setattr(episode, "WARMUP_RETRY_S", 0.005)
    current = DelayedCreateSink()
    existing = FakeSink(created=False)
    existing.preflight_episodes = 1
    recorder, cache = resume_recorder(tmp_path, current=current, existing=existing)
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
        session, _episodes, error = await asyncio.wait_for(resume, timeout=5)
        assert error is None and session
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


def test_warmup_waits_for_a_live_source_then_creates(tmp_path, monkeypatch):
    monkeypatch.setattr(episode, "WARMUP_RETRY_S", 0.005)
    plan = arm_plan()
    sink = FakeSink(created=False)
    recorder, cache = recorder_with(plan, sink, tmp_path)
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


def test_warmup_rearms_the_session_finish_session_rolls_to(tmp_path, monkeypatch):
    monkeypatch.setattr(episode, "WARMUP_RETRY_S", 0.005)
    plan = arm_plan()
    first_sink = FakeSink(created=False)
    recorder, cache = recorder_with(plan, first_sink, tmp_path)
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
        _, error = await recorder.finish_session()
        assert error is None
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
        name, error = await recorder.finish_session()
        assert error is None and name
        assert first_sink.finalized
        assert recorder._sink is second_sink
        assert recorder._session_dir == next_dir
        assert recorder._schema is None

    asyncio.run(run())


def test_finish_session_refuses_when_empty_or_recording(tmp_path):
    recorder, _ = recorder_with(arm_plan(), FakeSink(), tmp_path)
    recorder._open_session = lambda: (tmp_path, FakeSink(created=False), None)

    async def run():
        _, error = await recorder.finish_session()
        assert "nothing recorded" in error
        recorder._sink.saves = 1
        recorder._recording = True
        _, error = await recorder.finish_session()
        assert "recording" in error
        recorder._recording = False
        recorder._finishing = True
        _, error = await recorder.finish_session()
        assert "already finishing" in error

    asyncio.run(run())


def test_finish_session_refuses_before_the_dataset_exists(tmp_path):
    """An uncreated dataset cannot answer episodes_saved, so the emptiness
    guard must lead with it rather than letting the assertion escape."""
    recorder, _ = recorder_with(arm_plan(), FakeSink(created=False), tmp_path)
    recorder._open_session = lambda: (tmp_path, FakeSink(created=False), None)

    async def run():
        name, error = await recorder.finish_session()
        assert name == ""
        assert "nothing recorded" in error

    asyncio.run(run())


def test_finish_session_refuses_without_a_session_factory(tmp_path):
    sink = FakeSink()
    sink.saves = 1
    recorder, _ = recorder_with(arm_plan(), sink, tmp_path)

    async def run():
        _, error = await recorder.finish_session()
        assert "session rolling unavailable" in error
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
        name, error = await recorder.finish_session()
        assert name == "" and "finish failed" in error
        assert sink.finalized
        # The finalized sink must not take another episode.
        decision = recorder._decide(goal_request())
        assert not decision.accepted
        assert "retry finish_session" in decision.reason
        assert not recorder._recording
        name, error = await recorder.finish_session()
        assert error is None and name
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
        name, error = await recorder.finish_session()
        assert name == ""
        assert "finish failed" in error
        assert recorder._sink is sink
        # Not left latched: a retry is allowed once the fault clears.
        assert not recorder._finishing

    asyncio.run(run())
