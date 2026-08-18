import asyncio
import contextlib
import time

import pytest

from peppygen.consumed_actions.recorder import record_episode
from peppygen.consumed_services.recorder import finish_session

from tests.helpers import (
    POSE,
    FakeToken,
    boot,
    default_parameters,
    eventually,
    press_until_goal,
    stop_and_save,
    tap_x,
)
from xr_commander import config, publish, record
from xr_commander.bus import select_producer
from xr_commander.devices import HandSample, XrFrame

# The terminal payload of an ordinary stop-and-save.
SAVED = record_episode.ResultResponseData(
    episode_index=0, frames_recorded=7, discarded=False, error=None
)


class FakeSession:
    """A scripted headset (the FrameSource seam is a device surface, not a
    generated module)."""

    def __init__(self):
        self.hands = {}

    def press(self, *, x=False, y=False):
        self.hands = {
            "left": HandSample(
                pose=POSE,
                squeezing=False,
                trigger=0.0,
                primary_button=x,
                secondary_button=y,
            )
        }

    def latest(self):
        return XrFrame(received_monotonic_s=time.monotonic(), hands=self.hands)


def settings():
    return config.from_parameters(default_parameters())


@contextlib.asynccontextmanager
async def recorder_buttons(h, status=None):
    """`run_recorder_buttons` running against the real runner and the real
    recorder modules; yields the scripted headset session and the task."""
    session = FakeSession()
    token = FakeToken()
    task = asyncio.create_task(
        record.run_recorder_buttons(
            h.node_runner,
            action_module=record_episode,
            finish_module=finish_session,
            status=status or record.RecorderStatus(),
            session=session,
            settings=settings(),
            token=token,
        )
    )
    try:
        yield session, task
    finally:
        token.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _fired_episode(h):
    """One accepted record_episode goal fired outside the button loop: the
    node-side handle, the mock-side active goal, and the target."""
    target = select_producer(record_episode, h.node_runner, "recorder")
    fire = asyncio.create_task(
        publish.fire_goal(
            h.node_runner, record_episode, target, "record_episode", task=record.TASK
        )
    )
    pending = await h.mocks.deps.recorder[0].record_episode.next_goal(10.0)
    active = await pending.accept()
    handle = await asyncio.wait_for(fire, 10.0)
    assert handle is not None and handle.accepted
    return handle, active


async def test_x_fires_one_goal_per_rising_edge_with_the_task():
    async with boot(recorder_instances=1) as h:
        rec = h.mocks.deps.recorder[0].record_episode
        async with recorder_buttons(h) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            assert pending.request.task == record.TASK
            await pending.accept()
            # Still held: one edge, one goal.
            with pytest.raises(TimeoutError):
                await rec.next_goal(0.4)


async def test_x_already_held_at_first_tracking_does_not_start():
    async with boot(recorder_instances=1) as h:
        rec = h.mocks.deps.recorder[0].record_episode
        async with recorder_buttons(h) as (session, _task):
            session.press(x=True)
            with pytest.raises(TimeoutError):
                await rec.next_goal(0.4)
            # An observed release and then a press is the first real edge.
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            assert pending.request.task == record.TASK
            await pending.accept()


async def test_one_physical_y_hold_finishes_once_across_a_gap(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    async with boot(recorder_instances=1) as h:
        fin = h.mocks.deps.recorder[0].finish_session
        fin.enqueue_response(finish_session.ResponseData(session="s1", error=None))
        fin.enqueue_response(finish_session.ResponseData(session="s2", error=None))
        async with recorder_buttons(h) as (session, _task):
            session.press(y=True)
            await eventually(
                lambda: fin.captured_count() == 1, message="the held finish"
            )
            # Unknown frames cannot extend a hold, but one physical hold must
            # not finish twice: the fired latch survives the gap.
            session.hands = {}
            await asyncio.sleep(0.05)
            session.press(y=True)
            await asyncio.sleep(0.2)
            assert fin.captured_count() == 1
            # An observed release re-arms the finish.
            session.press(y=False)
            await asyncio.sleep(0.05)
            session.press(y=True)
            await eventually(
                lambda: fin.captured_count() == 2, message="the re-armed finish"
            )


async def test_x_again_stops_and_saves_instead_of_firing_twice():
    async with boot(recorder_instances=1) as h:
        rec = h.mocks.deps.recorder[0].record_episode
        status = record.RecorderStatus()
        async with recorder_buttons(h, status) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            active = await pending.accept()
            await eventually(lambda: status.recording, message="the REC line")
            await tap_x(session)  # stop-and-save, not a second goal
            await asyncio.wait_for(active.cancel_signal(), 10.0)
            with pytest.raises(TimeoutError):
                await rec.next_goal(0.4)
            await active.complete_cancelled(SAVED)
            # The watcher fetched the recorder's verdict once the stream closed.
            await eventually(
                lambda: status.note() == "saved episode 0", message="the verdict"
            )


async def test_a_finished_episode_rearms_the_button():
    async with boot(recorder_instances=1) as h:
        rec = h.mocks.deps.recorder[0].record_episode
        async with recorder_buttons(h) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            active = await pending.accept()
            await stop_and_save(session, active, SAVED)
            # The finished episode re-arms X for a second one.
            pending = await press_until_goal(session, rec, {"x": False}, {"x": True})
            assert pending.request.task == record.TASK
            await pending.accept()


async def test_a_rejected_goal_leaves_the_panel_idle_and_rearms():
    async with boot(recorder_instances=1) as h:
        rec = h.mocks.deps.recorder[0].record_episode
        status = record.RecorderStatus()
        async with recorder_buttons(h, status) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            await pending.reject("busy")
            # The refused goal re-arms the button immediately.
            pending = await press_until_goal(session, rec, {"x": False}, {"x": True})
            await pending.reject("busy")
            assert not status.recording


async def test_feedback_reaches_the_panel_and_clears_at_the_end():
    async with boot(recorder_instances=1) as h:
        rec = h.mocks.deps.recorder[0].record_episode
        status = record.RecorderStatus()
        async with recorder_buttons(h, status) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            active = await pending.accept()
            await eventually(lambda: status.recording, message="the REC line")
            for frames in (1, 2, 3):
                await active.publish_feedback(
                    record_episode.FeedbackMessage(
                        frames_recorded=frames, disk_free_bytes=10**12
                    )
                )
            # Feedback flowed while recording.
            await eventually(
                lambda: status.frames == 3 and status.recording,
                message="the frame counter",
            )
            await stop_and_save(session, active, SAVED)
            await eventually(
                lambda: not status.recording and status.frames == 0,
                message="the cleared row",
            )


async def test_y_must_be_held_to_finish(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.3)
    async with boot(recorder_instances=1) as h:
        fin = h.mocks.deps.recorder[0].finish_session
        fin.enqueue_response(finish_session.ResponseData(session="s1", error=None))
        async with recorder_buttons(h) as (session, _task):
            # A graze: released well before the hold threshold.
            session.press(y=True)
            await asyncio.sleep(0.05)
            session.press(y=False)
            await asyncio.sleep(0.4)
            assert fin.captured_count() == 0
            # A real hold finishes exactly once.
            session.press(y=True)
            await eventually(
                lambda: fin.captured_count() == 1, message="the held finish"
            )
            await asyncio.sleep(0.2)
            assert fin.captured_count() == 1


async def test_outcomes_reach_the_panel_as_notes(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    async with boot(recorder_instances=1) as h:
        rec = h.mocks.deps.recorder[0]
        status = record.RecorderStatus()
        async with recorder_buttons(h, status) as (session, _task):
            await tap_x(session)
            pending = await rec.record_episode.next_goal(10.0)
            active = await pending.accept()
            await stop_and_save(session, active, SAVED)
            await eventually(
                lambda: status.note() == "saved episode 0", message="the saved note"
            )
            session.press(x=False)
            await asyncio.sleep(0.05)
            rec.finish_session.enqueue_response(
                finish_session.ResponseData(session="s1", error=None)
            )
            session.press(y=True)
            await eventually(
                lambda: status.note() == "session finished",
                message="the finished note",
            )


async def test_a_refused_finish_notes_the_refusal(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    async with boot(recorder_instances=1) as h:
        fin = h.mocks.deps.recorder[0].finish_session
        fin.enqueue_response(
            finish_session.ResponseData(session="", error="an episode is recording")
        )
        status = record.RecorderStatus()
        async with recorder_buttons(h, status) as (session, _task):
            session.press(y=True)
            await eventually(
                lambda: status.note() == "finish refused", message="the refusal note"
            )


def test_notes_expire():
    status = record.RecorderStatus()
    status.set_note("saved episode 0")
    assert status.note()
    assert status.note(now_monotonic=time.monotonic() + record._NOTE_S + 1) == ""


async def test_a_refused_finish_is_survived(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    async with boot(recorder_instances=1) as h:
        fin = h.mocks.deps.recorder[0].finish_session
        fin.enqueue_response(
            finish_session.ResponseData(session="", error="an episode is recording")
        )
        async with recorder_buttons(h) as (session, task):
            session.press(y=True)
            await eventually(
                lambda: fin.captured_count() == 1, message="the refused finish"
            )
            await asyncio.sleep(0.2)
            # One hold, one attempt, and the loop is still alive.
            assert fin.captured_count() == 1
            assert not task.done()


async def test_a_stale_gap_does_not_stop_a_recording_with_x_still_held():
    """A dropout while the start press is still held must not read as a
    second press: the gap makes the input unknown, not released."""
    async with boot(recorder_instances=1) as h:
        rec = h.mocks.deps.recorder[0].record_episode
        async with recorder_buttons(h) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            active = await pending.accept()
            # The headset drops out with X still down, then comes back held.
            session.hands = {}
            await asyncio.sleep(0.05)
            session.press(x=True)
            await asyncio.sleep(0.2)
            assert not active.is_cancelled()
            # A real release and press still stops it.
            await stop_and_save(session, active, SAVED)


async def test_a_stop_racing_a_finished_watcher_does_not_claim_saving():
    async with boot(recorder_instances=1) as h:
        status = record.RecorderStatus()
        handle, active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await active.complete(SAVED)  # the episode ends on its own
        await asyncio.wait_for(watcher, 10.0)
        # A watcher that finished while the stop was in flight already
        # cleared the row; the late stop must not wedge it as saving.
        await record._stop_episode(record._Episode(handle, watcher), status)
        assert not status.saving


async def test_a_recorder_death_mid_episode_is_reported(capsys, monkeypatch):
    # The producer-gone result concludes only at the result deadline; keep
    # the test bounded without weakening the production value.
    monkeypatch.setattr(record, "_RESULT_TIMEOUT_S", 2.0)
    async with boot(recorder_instances=1) as h:
        status = record.RecorderStatus()
        handle, _active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await eventually(lambda: status.recording, message="the REC line")
        # The recorder process dies with the goal still active.
        await h.mocks.deps.recorder[0].stop()
        await asyncio.wait_for(watcher, 20.0)
    assert "recorder died mid-episode" in capsys.readouterr().out
    assert not status.recording


async def test_a_result_without_data_notes_the_goal_status(monkeypatch):
    # The one real terminal status that carries no data is ABANDONED (a
    # producer that died mid-goal); the note must still name it.
    monkeypatch.setattr(record, "_RESULT_TIMEOUT_S", 2.0)
    async with boot(recorder_instances=1) as h:
        status = record.RecorderStatus()
        handle, _active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await eventually(lambda: status.recording, message="the REC line")
        await h.mocks.deps.recorder[0].stop()
        await asyncio.wait_for(watcher, 20.0)
        assert status.note() == "episode abandoned"


async def test_a_discarded_episode_notes_the_discard():
    async with boot(recorder_instances=1) as h:
        status = record.RecorderStatus()
        handle, active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await active.complete(
            record_episode.ResultResponseData(
                episode_index=-1, frames_recorded=0, discarded=True, error="boom"
            )
        )
        await asyncio.wait_for(watcher, 10.0)
        assert status.note() == "episode discarded"


class SpyStatus(record.RecorderStatus):
    """Records every value written to finish_held, first write included."""

    @property
    def finish_held(self):
        return self.__dict__.get("value", False)

    @finish_held.setter
    def finish_held(self, value):
        self.__dict__.setdefault("seen", []).append(value)
        self.__dict__["value"] = value


async def test_a_grazed_y_never_shows_finishing(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.3)
    async with boot(recorder_instances=1) as h:
        fin = h.mocks.deps.recorder[0].finish_session
        grazed = SpyStatus()
        async with recorder_buttons(h, grazed) as (session, _task):
            session.press(y=True)
            await asyncio.sleep(0.05)
            session.press(y=False)
            await asyncio.sleep(0.4)
        assert True not in grazed.__dict__["seen"]
        fin.enqueue_response(finish_session.ResponseData(session="s1", error=None))
        held = SpyStatus()
        async with recorder_buttons(h, held) as (session, _task):
            session.press(y=True)
            await eventually(
                lambda: True in held.__dict__.get("seen", []),
                message="the finishing row",
            )


async def test_x_stop_shows_saving_until_the_result():
    async with boot(recorder_instances=1) as h:
        status = record.RecorderStatus()
        handle, active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await eventually(
            lambda: status.recording and not status.saving, message="the REC line"
        )
        # The stop is delivered; the recorder has not answered yet.
        await record._stop_episode(record._Episode(handle, watcher), status)
        assert status.saving  # result pending: the recorder is saving
        await active.complete_cancelled(SAVED)
        await asyncio.wait_for(watcher, 10.0)
        assert not status.saving and not status.recording
        assert status.note() == "saved episode 0"


async def test_a_failed_cancel_does_not_claim_saving(monkeypatch):
    monkeypatch.setattr(record, "GOAL_CANCEL_TIMEOUT_S", 0.5)
    monkeypatch.setattr(record, "_RESULT_TIMEOUT_S", 2.0)
    async with boot(recorder_instances=1) as h:
        status = record.RecorderStatus()
        handle, _active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await eventually(lambda: status.recording, message="the REC line")
        # The recorder dies before the stop: the cancel cannot be delivered,
        # so the row must not claim a save is in progress.
        await h.mocks.deps.recorder[0].stop()
        await record._stop_episode(record._Episode(handle, watcher), status)
        assert not status.saving
        await asyncio.wait_for(watcher, 20.0)


async def test_no_bound_recorder_is_inert():
    async with boot() as h:  # recorder_instances defaults to zero: unwired
        task = asyncio.create_task(
            record.run_recorder_buttons(
                h.node_runner,
                action_module=record_episode,
                finish_module=finish_session,
                status=record.RecorderStatus(),
                session=FakeSession(),
                settings=settings(),
                token=FakeToken(),
            )
        )
        # Nothing bound: the task returns on its own without firing.
        await asyncio.wait_for(task, 5.0)
