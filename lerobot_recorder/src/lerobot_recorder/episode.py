"""The record_episode peppy action: one episode per long-running goal.
(LeRobot's `action` feature is a different thing; recording.py assembles it.)

One goal-decide is pending at all times, so a goal sent mid-episode is
refused as busy, with `precondition_failed`, instead of queueing unanswered
(peppy holds undecided goals indefinitely). Feedback follows the sampling
grid, one message per recorded frame, with disk_free_bytes riding it once per
DISK_CHECK_PERIOD_S. The goal never completes: a cancel stops the episode and
saves it, and so does the recorder itself on a gap (stale source, silent
camera, shape change), the max-length bound, schedule lag, a disk floor or
its shutdown; the goal then ends cancelled with its result, for a reason
that names the cause. A goal whose episode cannot be written or saved fails
with `attempt_failed`, and its frames are dropped. The dataset is created
eagerly once every bound source has produced (the warm-up task), so the first
Record does not pay the creation cost; a goal arriving first still creates it
on demand.

The session services give an error reply, with a kind, to a finish or a
resume the recorder cannot carry out; after it, the open session stays as it
was.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import shutil
from dataclasses import dataclass

import peppygen.clock
from peppygen.exposed_actions.recording import record_episode
from peppygen.exposed_services.recording import finish_session, resume_session

from . import recording
from .plan import RecordingPlan
from .recording import Cache, NotReady, SampleGap, SourceSchema
from .sink import Sink

# Generous for one local service round-trip; a camera that cannot answer
# depth_stream_info in this long is not going to stream either.
DEPTH_INFO_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Timing:
    """The recorder's pacing, injectable so tests can tighten a bound
    without waiting out the production value."""

    # Cadence for retrying eager dataset creation while sources come up.
    warmup_retry_s: float = 1.0
    max_episode_s: float = recording.MAX_EPISODE_S
    disk_check_period_s: float = recording.DISK_CHECK_PERIOD_S
    max_schedule_lag_s: float = recording.MAX_SCHEDULE_LAG_S


async def poll_depth_units(node_runner, plan: RecordingPlan) -> list[float]:
    """Each bound rgbd camera's current depth unit, from the wire."""
    from peppygen.consumed_services.rgbd_cameras import (
        depth_stream_info as rgbd_cameras_depth_stream_info,
    )
    from peppygen.consumed_topics.rgbd_cameras import (
        video_stream as rgbd_cameras_video_stream,
    )

    units = [
        (
            await rgbd_cameras_depth_stream_info.poll(
                node_runner, producer, DEPTH_INFO_TIMEOUT_S
            )
        ).data.depth_unit
        for producer in rgbd_cameras_video_stream.bound_producers(node_runner)
    ]
    recording.validate_depth_units(plan, units)
    return units


def _log_sync_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        print(f"[recorder] mirror sync failed (next sync retries): {error!r}")


def _low_disk(free: int) -> str:
    return f"only {free // (1 << 20)} MB free under the session directory"


def _too_short(frames: int, ended_because: str) -> str:
    """Why an episode was too short to save, keeping whatever ended it: an
    episode cut off by low disk needs both halves to be diagnosable."""
    need = recording.MIN_EPISODE_FRAMES
    return f"too short to save: {frames} frame(s), need {need} ({ended_because})"


async def _end_goal(end, *args) -> None:
    """Ends the goal through `end`, the context's fail or complete_cancelled,
    and logs an end the runtime refused."""
    try:
        await end(*args)
    except Exception as e:
        print(f"[recorder] result delivery failed: {e!r}")


class _TickReport:
    """Per-second aggregate of frame-loop overruns, printed only when a tick
    exceeded the frame period; healthy episodes log nothing. `sample` is the
    on-loop cache read, `append` the off-loop decode and dataset write."""

    def __init__(self, period_s: float):
        self._period_s = period_s
        self._window_started: float | None = None
        self._overruns = 0
        self._worst = (0.0, 0.0, 0.0)

    def record(self, now_s: float, sample_s: float, append_s: float) -> None:
        if self._window_started is None:
            self._window_started = now_s
        total = sample_s + append_s
        if total > self._period_s:
            self._overruns += 1
            self._worst = max(self._worst, (total, sample_s, append_s))
        if now_s - self._window_started >= 1.0:
            self.flush()
            self._window_started = now_s

    def flush(self) -> None:
        if self._overruns:
            total, sample_s, append_s = self._worst
            print(
                f"[recorder] {self._overruns} tick overrun(s): worst "
                f"{total * 1000:.0f} ms (sample {sample_s * 1000:.0f} ms, "
                f"append {append_s * 1000:.0f} ms) against {self._period_s * 1000:.0f} ms"
            )
        self._overruns = 0
        self._worst = (0.0, 0.0, 0.0)


class Recorder:
    """Single owner of the recorder state machine; only the goal loop
    touches it, so goals cannot race."""

    def __init__(
        self,
        plan: RecordingPlan,
        cache: Cache,
        sink: Sink,
        session_dir,
        max_staleness_s: float,
        min_remaining_disk_bytes: int,
        mirror=None,
        open_session=None,
        open_existing=None,
        *,
        now_ns=peppygen.clock.now_ns,
        disk_usage=shutil.disk_usage,
        depth_probe=poll_depth_units,
        timing: Timing = Timing(),
    ):
        if not math.isfinite(max_staleness_s) or max_staleness_s <= 0:
            raise ValueError(f"max_staleness_s must be positive, got {max_staleness_s}")
        self._plan = plan
        self._cache = cache
        self._sink = sink
        self._session_dir = session_dir
        self._max_staleness_s = max_staleness_s
        self._min_remaining_disk_bytes = min_remaining_disk_bytes
        self._mirror = mirror
        # () -> (session_dir, sink, mirror): opens the next session when the
        # current one finishes; None disables finish_session.
        self._open_session = open_session
        # (name) -> (session_dir, sink, mirror) for an existing session,
        # raising resume_session.Failure: `rejected` for a name that names no
        # session of this recorder, `precondition_failed` for a session whose
        # sources differ from the ones this launch binds. None disables
        # resume_session.
        self._open_existing = open_existing
        self._now_ns = now_ns
        self._disk_usage = disk_usage
        self._depth_probe = depth_probe
        self._timing = timing
        self._finishing = False
        self._resuming = False
        # Set between a session's successful finalize and the roll to its
        # successor; while pending the finalized sink must not take frames.
        self._roll_pending = False
        self._sync_task: asyncio.Task | None = None
        self._schema: SourceSchema | None = None
        self._recording = False
        self._create_lock = asyncio.Lock()

    async def run(self, node_runner) -> None:
        token = node_runner.cancellation_token()
        action = await record_episode.ActionHandle.expose(node_runner)
        cancelled = asyncio.ensure_future(token.cancelled())
        pending = asyncio.ensure_future(action.handle_goal_next_request(self._decide))
        try:
            while not token.is_cancelled():
                await asyncio.wait(
                    [cancelled, pending], return_when=asyncio.FIRST_COMPLETED
                )
                if not pending.done():
                    break
                try:
                    ctx = pending.result()
                except Exception as e:
                    # Serving one goal must not end the goal server: without
                    # this, every later goal would hang undecided for the
                    # node's life with nothing logged until shutdown.
                    print(f"[recorder] goal request failed: {e!r}")
                    self._recording = False
                    pending = asyncio.ensure_future(
                        action.handle_goal_next_request(self._decide)
                    )
                    continue
                if ctx is None:
                    break  # goal stream closed: node shutting down
                # The next decide goes up before the episode starts, so goals
                # arriving mid-episode get an immediate busy refusal.
                pending = asyncio.ensure_future(
                    action.handle_goal_next_request(self._decide)
                )
                try:
                    await self._run_episode(node_runner, ctx, token)
                except Exception as e:
                    await self._contain_episode_failure(ctx, e)
                finally:
                    self._recording = False
        finally:
            pending.cancel()
            cancelled.cancel()

    async def warm_up(self, node_runner) -> None:
        """Create each session's dataset as soon as every bound source is
        live, so a Record starts capturing immediately instead of waiting out
        schema capture and writer setup. Runs for the node's life, not just
        the first session: finish_session clears the schema and the session it
        rolls to warms up the same way."""
        token = node_runner.cancellation_token()
        last_reason: str | None = None
        while not token.is_cancelled():
            if self._schema is not None:
                await asyncio.sleep(self._timing.warmup_retry_s)
                continue
            try:
                await self._ensure_created(node_runner)
                last_reason = None
            except Exception as e:
                # Expected until every source has produced. Printed once per
                # distinct reason, so a source that never arrives is visible
                # without a line every retry.
                if repr(e) != last_reason:
                    last_reason = repr(e)
                    print(f"[recorder] dataset not ready yet: {last_reason}")
                await asyncio.sleep(self._timing.warmup_retry_s)

    async def _ensure_created(self, node_runner) -> None:
        """Capture the schema and create the dataset exactly once; callers
        race (warm-up task vs an early goal), hence the lock."""
        async with self._create_lock:
            if self._schema is not None:
                return
            schema = await self._capture_schema(node_runner)
            await self._sink.create(schema, self._plan)
            self._schema = schema
            print("[recorder] dataset ready")

    async def finish_session(self) -> str:
        """Finalize and mirror the current dataset (the end of a lerobot
        recording run), then open a fresh session in place. Returns the
        finished session's name. Raises finish_session.Failure:
        `precondition_failed` when the recorder cannot finish the session
        now, `attempt_failed` when the finish fails. On a failed finalize or
        mirror nothing rolls, so a retry resumes where it failed (finalize is
        idempotent); on a failed roll after a successful finalize, goals are
        refused until a retry lands the roll."""
        refusal = self._finish_refusal()
        if refusal is not None:
            raise finish_session.Failure(finish_session.FailureKind.PRECONDITION_FAILED, refusal)
        self._finishing = True
        try:
            name = self._session_dir.name
            if not self._roll_pending:
                await self.drain_mirror()
                await self._sink.finalize()
                if self._mirror is not None:
                    await asyncio.to_thread(self._mirror.sync)
                # The old dataset is finalized from here on; until the roll
                # lands, goals must be refused rather than recorded into it.
                self._roll_pending = True
            self._session_dir, self._sink, self._mirror = self._open_session()
            self._roll_pending = False
            self._schema = None
            print(f"[recorder] session {name} finished and ready for replay")
            return name
        except Exception as e:
            raise finish_session.Failure(
                finish_session.FailureKind.ATTEMPT_FAILED, f"finish failed: {e}"
            ) from e
        finally:
            self._finishing = False

    def _finish_refusal(self) -> str | None:
        """Why the recorder cannot finish the session now, or None."""
        if self._recording:
            return "an episode is recording"
        if self._finishing:
            return "already finishing"
        if self._resuming:
            return "resuming a session"
        # An uncreated dataset holds no episodes and cannot answer
        # episodes_saved, so the emptiness check leads with it.
        if not self._sink.created or self._sink.episodes_saved == 0:
            return "nothing recorded this session"
        if self._open_session is None:
            return "session rolling unavailable"
        return None

    async def resume_session(self, name: str, node_runner) -> tuple[str, int]:
        """Swap the recorder onto a past session so new episodes append to
        its dataset (lerobot's record -> finalize -> resume lifecycle).
        Returns (session, episodes already saved). Raises
        resume_session.Failure, and the current session stays untouched:
        `rejected` for a name that names no session the recorder can resume,
        `precondition_failed` when the recorder cannot change sessions now or
        the live sources no longer match what the session recorded,
        `attempt_failed` when the resume fails. The abandoned fresh session
        directory stays on disk: nothing here deletes operator data."""
        refusal = self._resume_refusal(name)
        if refusal is not None:
            raise resume_session.Failure(resume_session.FailureKind.PRECONDITION_FAILED, refusal)
        self._resuming = True
        try:
            session_dir, new_sink, new_mirror = self._open_existing(name)
            try:
                manifest = await new_sink.preflight_resume()
            except ValueError as e:
                # The dataset on disk holds nothing this recorder can append
                # to: no dataset, no saved episode, unreadable metadata.
                raise resume_session.Failure(
                    resume_session.FailureKind.REJECTED, str(e)
                ) from None
            # The lock serializes against warm-up's create: a create already
            # in flight lands on the old sink before the swap, and afterwards
            # warm-up sees the schema set and idles.
            async with self._create_lock:
                await self.drain_mirror()
                try:
                    schema = await self._capture_schema(node_runner)
                except NotReady as e:
                    raise resume_session.Failure(
                        resume_session.FailureKind.PRECONDITION_FAILED,
                        f"not ready to resume: {e}",
                    ) from None
                try:
                    await new_sink.resume(manifest, schema, self._plan)
                except ValueError as e:
                    raise resume_session.Failure(
                        resume_session.FailureKind.PRECONDITION_FAILED,
                        f"cannot resume {name}: {e}",
                    ) from None
                old_sink = self._sink
                self._session_dir, self._sink, self._mirror = session_dir, new_sink, new_mirror
                self._schema = schema
            # The abandoned session's dataset (if warm-up created one) holds
            # image-writer threads only finalize stops; on a zero-episode
            # dataset this closes writers without touching operator data.
            if old_sink.created:
                try:
                    await old_sink.finalize()
                except Exception as e:
                    print(f"[recorder] abandoned session close failed: {e!r}")
            episodes = new_sink.episodes_saved
            print(f"[recorder] session {name} resumed with {episodes} episode(s)")
            return name, episodes
        except resume_session.Failure:
            raise
        except Exception as e:
            raise resume_session.Failure(
                resume_session.FailureKind.ATTEMPT_FAILED, f"resume failed: {e}"
            ) from e
        finally:
            self._resuming = False

    def _resume_refusal(self, name: str) -> str | None:
        """Why the recorder cannot change to the session `name` now, or
        None."""
        if self._recording:
            return "an episode is recording"
        if self._finishing:
            return "finishing the session"
        if self._resuming:
            return "already resuming"
        if self._roll_pending:
            return "session finished but not rolled; retry finish_session"
        if self._sink.created and self._sink.episodes_saved > 0:
            return (
                f"current session already holds "
                f"{self._sink.episodes_saved} episode(s); finish it first"
            )
        if name == self._session_dir.name:
            return "that session is currently recording"
        if self._open_existing is None:
            return "session resume unavailable"
        return None

    async def close_for_shutdown(self) -> None:
        """The shutdown tail: land in-flight uploads, close the writers so
        every parquet gets its footer, then the final mirror pass uploads the
        now-valid files. The storage root keeps the local dataset."""
        await self.drain_mirror()
        try:
            await self._sink.finalize()
        except Exception as e:
            # Without footers the files are not the "now-valid" set the final
            # pass exists to publish; syncing them would stamp a broken remote
            # copy as final. The incremental mirror state stays as-is.
            print(f"[recorder] dataset finalize failed; final mirror pass skipped: {e!r}")
            return
        if self._mirror is not None:
            # The whole shutdown hook runs inside the daemon's
            # shutdown_grace_secs window; the start line makes a cut-off pass
            # visible (no completion line) and the README documents recovery.
            # Flushed: a force-kill mid-pass must not eat the marker it exists
            # to leave behind.
            print("[recorder] final mirror pass starting", flush=True)
            try:
                files, size = await asyncio.to_thread(self._mirror.sync)
            except Exception as e:
                print(f"[recorder] final mirror sync failed, local copy kept: {e!r}", flush=True)
            else:
                print(
                    f"[recorder] mirror complete ({files} files, {size >> 20} MB); "
                    f"local copy kept at {self._session_dir}",
                    flush=True,
                )

    async def drain_mirror(self) -> None:
        """Wait out an in-flight background sync. Shutdown runs this before
        finalize, so a stale upload can never land after the final one."""
        if self._sync_task is not None:
            # The done-callback reports the failure; a failed sync must not
            # abort the shutdown sequence.
            await asyncio.gather(self._sync_task, return_exceptions=True)
            self._sync_task = None

    def _decide(self, request) -> record_episode.GoalDecision:
        try:
            reason = self._goal_refusal()
        except Exception as e:
            # A decider that raises kills the goal server, so anything
            # unexpected becomes a refusal the operator can read.
            return record_episode.GoalDecision.refuse(
                record_episode.FailureKind.ATTEMPT_FAILED,
                f"recorder cannot start an episode: {e!r}",
            )
        if reason is not None:
            return record_episode.GoalDecision.refuse(
                record_episode.FailureKind.PRECONDITION_FAILED, reason
            )
        # Claimed at admission, not when the run loop wakes: finish_session
        # must not slip between accepting a goal and starting its episode.
        self._recording = True
        return record_episode.GoalDecision.accept()

    def _goal_refusal(self) -> str | None:
        if self._recording:
            return "an episode is already recording"
        if self._finishing:
            return "finishing the session"
        if self._resuming:
            return "resuming a session"
        if self._roll_pending:
            return "session finished but not rolled; retry finish_session"
        return self._refuse_reason()

    def _refuse_reason(self) -> str | None:
        try:
            now_ns = self._now_ns()
        except RuntimeError as e:
            return f"recorder clock not ready: {e}"
        free = self._disk_usage(self._session_dir).free
        if free < self._min_remaining_disk_bytes:
            return _low_disk(free)
        # Probed on every goal, so a refusal names the first unmet
        # precondition and a source that died between episodes cannot start
        # a new one. Before creation no depth units exist to probe against;
        # inventing them would blame a camera for a number it never sent.
        depth_units = list(self._schema.depth_units) if self._schema is not None else None
        try:
            probe = recording.capture_schema(
                self._cache, self._plan, depth_units, now_ns, self._max_staleness_s
            )
        except NotReady as e:
            return str(e)
        if self._schema is not None:
            return recording.schema_mismatch(self._schema, probe)
        return None

    async def _run_episode(self, node_runner, ctx, token) -> None:
        created_before_goal = self._schema is not None
        try:
            await self._ensure_created(node_runner)
        except Exception as e:
            # Accepted but unable to start writing: the goal fails with the
            # reason rather than being abandoned.
            await _end_goal(
                ctx.fail, record_episode.FailureKind.ATTEMPT_FAILED, f"dataset create failed: {e}"
            )
            return
        if created_before_goal and self._plan.rgbd_cameras:
            # Re-probed each episode: a camera re-profiled between episodes
            # would otherwise silently mis-scale every depth frame.
            try:
                depth_units = await self._depth_probe(node_runner, self._plan)
            except Exception as e:
                await _end_goal(
                    ctx.fail,
                    record_episode.FailureKind.ATTEMPT_FAILED,
                    f"depth unit probe failed: {e}",
                )
                return
            if tuple(depth_units) != self._schema.depth_units:
                await _end_goal(
                    ctx.fail,
                    record_episode.FailureKind.PRECONDITION_FAILED,
                    "rgbd depth unit changed since the dataset was created",
                )
                return
        await self._record(ctx, token)

    async def _capture_schema(self, node_runner) -> SourceSchema:
        return recording.capture_schema(
            self._cache,
            self._plan,
            await self._depth_probe(node_runner, self._plan),
            self._now_ns(),
            self._max_staleness_s,
        )

    async def _record(self, ctx, token) -> None:
        assert self._schema is not None
        task = ctx.request().data.task
        episode_index = self._sink.episodes_saved
        period = 1.0 / self._sink.fps
        max_frames = self._timing.max_episode_s * self._sink.fps
        frames = 0
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        # Due immediately: the first frame both seeds disk_free and checks the
        # floor, so a disk that filled since goal-accept trips at frame one.
        next_disk_check = loop.time()
        report = _TickReport(period)
        cancel = asyncio.ensure_future(ctx.cancel_signal())
        shutdown = asyncio.ensure_future(token.cancelled())
        # Why the episode ends: the cause of a cancel, or the recorder's own
        # stop. The saved episode's goal ends cancelled for this reason.
        ended_because: str
        try:
            while True:
                if cancel.done() or ctx.is_cancelled():
                    ended_because = ctx.cancel_reason()
                    break
                if shutdown.done():
                    ended_because = "recorder shutting down"
                    break
                tick_started = loop.time()
                try:
                    row, pending = recording.sample(
                        self._cache,
                        self._schema,
                        self._plan,
                        self._now_ns(),
                        self._max_staleness_s,
                    )
                except (SampleGap, RuntimeError) as e:
                    ended_because = str(e)
                    break
                sampled = loop.time()
                # Off the loop: decode and dataset writes on the loop starve
                # the drains, age the cache, and end the episode as stale.
                try:
                    await asyncio.to_thread(self._append_frame, row, pending, task)
                except SampleGap as e:
                    ended_because = str(e)
                    break
                except Exception as e:
                    self._discard()
                    await _end_goal(
                        ctx.fail,
                        record_episode.FailureKind.ATTEMPT_FAILED,
                        f"frame write failed: {e}",
                    )
                    return
                frames += 1
                now = loop.time()
                report.record(now, sample_s=sampled - tick_started, append_s=now - sampled)
                # The slow field rides the per-frame feedback once per
                # DISK_CHECK_PERIOD_S and repeats in between.
                fresh_measurement = now >= next_disk_check
                if fresh_measurement:
                    next_disk_check = now + self._timing.disk_check_period_s
                    disk_free = self._disk_usage(self._session_dir).free
                # Progress reporting is best-effort: a subscriber that went
                # away must not end the episode.
                with contextlib.suppress(Exception):
                    await ctx.publish_feedback(frames, disk_free)
                if fresh_measurement and disk_free < self._min_remaining_disk_bytes:
                    ended_because = _low_disk(disk_free)
                    break
                if frames >= max_frames:
                    ended_because = "max episode length reached"
                    break
                next_tick += period
                delay = next_tick - loop.time()
                if delay < -self._timing.max_schedule_lag_s:
                    ended_because = (
                        f"cannot sustain {self._sink.fps} fps "
                        f"(sampling {-delay:.2f}s behind)"
                    )
                    break
                # Always yields, so the drain tasks run even on an overrun tick.
                await asyncio.sleep(max(delay, 0.0))
        finally:
            report.flush()
            cancel.cancel()
            shutdown.cancel()

        if frames < recording.MIN_EPISODE_FRAMES:
            self._discard()
            await _end_goal(
                ctx.fail,
                record_episode.FailureKind.ATTEMPT_FAILED,
                _too_short(frames, ended_because),
            )
            return
        try:
            shortfall = await self._sink.save_episode()
        except Exception as e:
            # The library's buffer is unusable after a failed save; drop it
            # so the next goal starts clean.
            self._discard()
            await _end_goal(
                ctx.fail, record_episode.FailureKind.ATTEMPT_FAILED, f"episode save failed: {e}"
            )
            return
        self._kick_mirror()
        warning = shortfall.warning()
        if warning is not None:
            # The episode saved, but its video does not line up with its
            # rows. The result gives the frames the shortest video misses;
            # the run log names each camera that came up short.
            print(f"[recorder] episode {episode_index}: {warning}", flush=True)
        await _end_goal(
            ctx.complete_cancelled,
            ended_because,
            episode_index,
            frames,
            shortfall.video_frames_missing,
        )

    def _append_frame(self, row, pending, task: str) -> None:
        """Decode the frame's images and hand the row to the sink. Runs in a
        worker thread; calls are sequential (one per tick), so the sink never
        sees concurrent frames."""
        recording.decode_images(row, pending)
        self._sink.add_frame(row, task)

    def _discard(self) -> None:
        if not self._sink.created:
            return
        try:
            self._sink.discard_open_frames()
        except Exception as e:
            print(f"[recorder] discard failed: {e!r}")

    async def _contain_episode_failure(self, ctx, error: Exception) -> None:
        """Last resort: an unexpected episode error must fail the goal, not
        end the goal loop (a dead setup task takes the whole node with it)."""
        print(f"[recorder] episode failed: {error!r}")
        self._discard()
        await _end_goal(
            ctx.fail, record_episode.FailureKind.ATTEMPT_FAILED, f"internal error: {error}"
        )

    def _kick_mirror(self) -> None:
        """Mirror the session in the background; a sync still running keeps
        going and the next episode's kick uploads whatever it missed."""
        if self._mirror is None:
            return
        if self._sync_task is not None and not self._sync_task.done():
            return
        self._sync_task = asyncio.create_task(asyncio.to_thread(self._mirror.sync))
        self._sync_task.add_done_callback(_log_sync_failure)
