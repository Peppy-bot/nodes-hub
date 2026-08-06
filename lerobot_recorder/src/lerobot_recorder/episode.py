"""The record_episode peppy action: one episode per long-running goal.
(LeRobot's `action` feature is a different thing; recording.py assembles it.)

One goal-decide is pending at all times, so a goal sent mid-episode is
rejected as busy instead of queueing unanswered (peppy holds undecided goals
indefinitely). Feedback follows the sampling grid, one message per recorded
frame, with disk_free_bytes riding it once per DISK_CHECK_PERIOD_S; a cancel
stops the episode and saves it; a gap (stale source, silent camera, shape
change), the max-length bound, schedule lag, or a disk floor ends the episode
with a save and the reason in the result's error field. The dataset is
created eagerly once every bound source has produced (the warm-up task),
so the first Record does not pay the creation cost; a goal arriving first
still creates it on demand.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import shutil

import peppygen.clock
from peppygen.consumed_services.rgbd_cameras import (
    depth_stream_info as rgbd_cameras_depth_stream_info,
)
from peppygen.consumed_topics.rgbd_cameras import (
    video_stream as rgbd_cameras_video_stream,
)
from peppygen.exposed_actions import record_episode

from . import recording
from .plan import RecordingPlan
from .recording import Cache, NotReady, SampleGap, SourceSchema
from .sink import Sink

# Generous for one local service round-trip; a camera that cannot answer
# depth_stream_info in this long is not going to stream either.
DEPTH_INFO_TIMEOUT_S = 5.0
# Cadence for retrying eager dataset creation while sources come up.
WARMUP_RETRY_S = 1.0


def _log_sync_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        print(f"[recorder] mirror sync failed (next sync retries): {error!r}")


def _low_disk(free: int) -> str:
    return f"only {free // (1 << 20)} MB free under the session directory"


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
        # raising ValueError for a bad or unknown name; None disables
        # resume_session.
        self._open_existing = open_existing
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
                # arriving mid-episode get an immediate busy rejection.
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
                await asyncio.sleep(WARMUP_RETRY_S)
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
                await asyncio.sleep(WARMUP_RETRY_S)

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

    async def finish_session(self) -> tuple[str, str | None]:
        """Finalize and mirror the current dataset (the end of a lerobot
        recording run), then open a fresh session in place. Returns the
        finished session's name, or an empty name with the refusal reason.
        On a failed finalize or mirror nothing rolls, so a retry resumes
        where it failed (finalize is idempotent); on a failed roll after a
        successful finalize, goals are refused until a retry lands the roll."""
        if self._recording:
            return "", "an episode is recording"
        if self._finishing:
            return "", "already finishing"
        if self._resuming:
            return "", "resuming a session"
        # An uncreated dataset holds no episodes and cannot answer
        # episodes_saved, so the emptiness check leads with it.
        if not self._sink.created or self._sink.episodes_saved == 0:
            return "", "nothing recorded this session"
        if self._open_session is None:
            return "", "session rolling unavailable"
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
            return name, None
        except Exception as e:
            return "", f"finish failed: {e}"
        finally:
            self._finishing = False

    async def resume_session(self, name: str, node_runner) -> tuple[str, int, str | None]:
        """Swap the recorder onto a past session so new episodes append to
        its dataset (lerobot's record -> finalize -> resume lifecycle).
        Returns (session, episodes already saved, error); on any refusal the
        current session is untouched. The abandoned fresh session directory
        stays on disk: nothing here deletes operator data."""
        if self._recording:
            return "", 0, "an episode is recording"
        if self._finishing:
            return "", 0, "finishing the session"
        if self._resuming:
            return "", 0, "already resuming"
        if self._roll_pending:
            return "", 0, "session finished but not rolled; retry finish_session"
        if self._sink.created and self._sink.episodes_saved > 0:
            return "", 0, (
                f"current session already holds "
                f"{self._sink.episodes_saved} episode(s); finish it first"
            )
        if name == self._session_dir.name:
            return "", 0, "that session is currently recording"
        if self._open_existing is None:
            return "", 0, "session resume unavailable"
        self._resuming = True
        try:
            try:
                session_dir, new_sink, new_mirror = self._open_existing(name)
                manifest = await new_sink.preflight_resume()
            except ValueError as e:
                return "", 0, str(e)
            # The lock serializes against warm-up's create: a create already
            # in flight lands on the old sink before the swap, and afterwards
            # warm-up sees the schema set and idles.
            async with self._create_lock:
                await self.drain_mirror()
                try:
                    schema = await self._capture_schema(node_runner)
                except NotReady as e:
                    return "", 0, f"not ready to resume: {e}"
                try:
                    await new_sink.resume(manifest, schema, self._plan)
                except ValueError as e:
                    return "", 0, f"cannot resume {name}: {e}"
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
            return name, episodes, None
        except Exception as e:
            return "", 0, f"resume failed: {e}"
        finally:
            self._resuming = False

    async def close_for_shutdown(self) -> None:
        """The shutdown tail: land in-flight uploads, close the writers so
        every parquet gets its footer, then the final mirror pass uploads the
        now-valid files. The staging copy stays as the local dataset."""
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
                print(f"[recorder] final mirror sync failed, staging kept: {e!r}", flush=True)
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
            reason = f"recorder cannot start an episode: {e!r}"
        if reason is not None:
            return record_episode.GoalDecision.reject(reason)
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
            now_ns = peppygen.clock.now_ns()
        except RuntimeError as e:
            return f"recorder clock not ready: {e}"
        free = shutil.disk_usage(self._session_dir).free
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
            # Accepted but unable to start writing: complete with the
            # reason and no frames rather than abandoning the goal.
            await self._complete(
                ctx,
                ctx.is_cancelled(),
                episode_index=-1,
                frames=0,
                discarded=True,
                error=f"dataset create failed: {e}",
            )
            return
        if created_before_goal and self._plan.rgbd_cameras:
            # Re-probed each episode: a camera re-profiled between episodes
            # would otherwise silently mis-scale every depth frame.
            try:
                depth_units = await self._poll_depth_units(node_runner)
            except Exception as e:
                await self._complete(
                    ctx,
                    ctx.is_cancelled(),
                    episode_index=-1,
                    frames=0,
                    discarded=True,
                    error=f"depth unit probe failed: {e}",
                )
                return
            if tuple(depth_units) != self._schema.depth_units:
                await self._complete(
                    ctx,
                    ctx.is_cancelled(),
                    episode_index=-1,
                    frames=0,
                    discarded=True,
                    error="rgbd depth unit changed since the dataset was created",
                )
                return
        await self._record(ctx, token)

    async def _poll_depth_units(self, node_runner) -> list[float]:
        units = [
            (
                await rgbd_cameras_depth_stream_info.poll(
                    node_runner, producer, DEPTH_INFO_TIMEOUT_S
                )
            ).data.depth_unit
            for producer in rgbd_cameras_video_stream.bound_producers(node_runner)
        ]
        recording.validate_depth_units(self._plan, units)
        return units

    async def _capture_schema(self, node_runner) -> SourceSchema:
        return recording.capture_schema(
            self._cache,
            self._plan,
            await self._poll_depth_units(node_runner),
            peppygen.clock.now_ns(),
            self._max_staleness_s,
        )

    async def _record(self, ctx, token) -> None:
        assert self._schema is not None
        task = ctx.request().data.task
        episode_index = self._sink.episodes_saved
        period = 1.0 / self._sink.fps
        max_frames = recording.MAX_EPISODE_S * self._sink.fps
        frames = 0
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        # Due immediately: the first frame both seeds disk_free and checks the
        # floor, so a disk that filled since goal-accept trips at frame one.
        next_disk_check = loop.time()
        cancel = asyncio.ensure_future(ctx.cancel_signal())
        shutdown = asyncio.ensure_future(token.cancelled())
        end_error: str | None = None
        cancelled = False
        try:
            while True:
                if cancel.done() or ctx.is_cancelled():
                    cancelled = True
                    break
                if shutdown.done():
                    end_error = "recorder shutting down"
                    break
                try:
                    row = recording.sample(
                        self._cache,
                        self._schema,
                        self._plan,
                        peppygen.clock.now_ns(),
                        self._max_staleness_s,
                    )
                except (SampleGap, RuntimeError) as e:
                    end_error = str(e)
                    break
                try:
                    self._sink.add_frame(row, task)
                except Exception as e:
                    self._discard()
                    await self._complete(
                        ctx,
                        ctx.is_cancelled(),
                        episode_index=-1,
                        frames=frames,
                        discarded=True,
                        error=f"frame write failed: {e}",
                    )
                    return
                frames += 1
                now = loop.time()
                # The slow field rides the per-frame feedback once per
                # DISK_CHECK_PERIOD_S and repeats in between.
                fresh_measurement = now >= next_disk_check
                if fresh_measurement:
                    next_disk_check = now + recording.DISK_CHECK_PERIOD_S
                    disk_free = shutil.disk_usage(self._session_dir).free
                # Progress reporting is best-effort: a subscriber that went
                # away must not end the episode.
                with contextlib.suppress(Exception):
                    await ctx.publish_feedback(frames, disk_free)
                if fresh_measurement and disk_free < self._min_remaining_disk_bytes:
                    end_error = _low_disk(disk_free)
                    break
                if frames >= max_frames:
                    end_error = "max episode length reached"
                    break
                next_tick += period
                delay = next_tick - loop.time()
                if delay < -recording.MAX_SCHEDULE_LAG_S:
                    end_error = (
                        f"cannot sustain {self._sink.fps} fps "
                        f"(sampling {-delay:.2f}s behind)"
                    )
                    break
                # Always yields, so the drain tasks run even on an overrun tick.
                await asyncio.sleep(max(delay, 0.0))
        finally:
            cancel.cancel()
            shutdown.cancel()

        if frames == 0:
            self._discard()
            await self._complete(
                ctx, cancelled, episode_index=-1, frames=0, discarded=True, error=end_error
            )
            return
        try:
            await self._sink.save_episode()
        except Exception as e:
            # The library's buffer is unusable after a failed save; drop it
            # so the next goal starts clean.
            self._discard()
            await self._complete(
                ctx,
                cancelled,
                episode_index=-1,
                frames=frames,
                discarded=True,
                error=f"episode save failed: {e}",
            )
            return
        self._kick_mirror()
        await self._complete(
            ctx, cancelled, episode_index=episode_index, frames=frames, discarded=False, error=end_error
        )

    def _discard(self) -> None:
        if not self._sink.created:
            return
        try:
            self._sink.discard_open_frames()
        except Exception as e:
            print(f"[recorder] discard failed: {e!r}")

    async def _contain_episode_failure(self, ctx, error: Exception) -> None:
        """Last resort: an unexpected episode error must end the goal, not
        the goal loop (a dead setup task takes the whole node with it)."""
        print(f"[recorder] episode failed: {error!r}")
        self._discard()
        await self._complete(
            ctx,
            ctx.is_cancelled(),
            episode_index=-1,
            frames=0,
            discarded=True,
            error=f"internal error: {error}",
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

    @staticmethod
    async def _complete(
        ctx,
        cancelled: bool,
        *,
        episode_index: int,
        frames: int,
        discarded: bool,
        error: str | None,
    ):
        try:
            if cancelled:
                await ctx.complete_cancelled(episode_index, frames, discarded, error)
            else:
                await ctx.complete(episode_index, frames, discarded, error)
        except Exception as e:
            print(f"[recorder] result delivery failed: {e!r}")
