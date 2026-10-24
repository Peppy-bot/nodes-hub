import asyncio
import contextlib
import time
from types import SimpleNamespace

import pytest
from peppygen.consumed_actions.recorder import record_episode
from peppygen.consumed_services.recorder import finish_session

from tests.helpers import (
    CANCELLED_BY_CALLER,
    POSE,
    FakeToken,
    boot,
    default_parameters,
    eventually,
    press_until_goal,
    robots_of,
    stop_and_save,
    tap_x,
)
from xr_commander import config, publish, record
from xr_commander.bus import FLASH_S
from xr_commander.devices import HandSample, XrFrame

# The result of an ordinary stop-and-save.
SAVED = record_episode.ResultResponseData(
    episode_index=0, frames_recorded=7, video_frames_missing=0
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


# The label the task page hands the button loop unless a test scripts its own.
TASK = "wipe the table"


def boot_recording():
    """A stack whose one paired robot, outside any copy, carries the
    recorder: the same instance id names the pair and the producer."""
    return boot(
        recorder_instance_ids=["backbone_inst"],
        right_arm_members={"backbone_inst": None},
    )


@contextlib.asynccontextmanager
async def recorder_buttons(h, status=None, timing=None, read_task=None):
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
            robots=robots_of(h),
            settings=settings(),
            token=token,
            read_task=read_task or (lambda: TASK),
            timing=timing or record.Timing(),
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
    target = robots_of(h).producer_for(record_episode)
    fire = asyncio.create_task(
        publish.fire_goal(
            h.node_runner, record_episode, target, "record_episode", task=TASK
        )
    )
    pending = await h.mocks.deps.recorder[0].record_episode.next_goal(10.0)
    active = await pending.accept()
    handle = await asyncio.wait_for(fire, 10.0)
    assert handle is not None and handle.accepted
    return handle, active


async def test_x_fires_one_goal_per_rising_edge_with_the_task():
    async with boot_recording() as h:
        rec = h.mocks.deps.recorder[0].record_episode
        async with recorder_buttons(h) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            assert pending.request.task == TASK
            await pending.accept()
            # Still held: one edge, one goal.
            with pytest.raises(TimeoutError):
                await rec.next_goal(0.4)


async def test_x_already_held_at_first_tracking_does_not_start():
    async with boot_recording() as h:
        rec = h.mocks.deps.recorder[0].record_episode
        async with recorder_buttons(h) as (session, _task):
            session.press(x=True)
            with pytest.raises(TimeoutError):
                await rec.next_goal(0.4)
            # An observed release and then a press is the first real edge.
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            assert pending.request.task == TASK
            await pending.accept()


async def test_one_physical_y_hold_finishes_once_across_a_gap():
    async with boot_recording() as h:
        fin = h.mocks.deps.recorder[0].finish_session
        fin.enqueue_response(finish_session.ResponseData(session="s1"))
        fin.enqueue_response(finish_session.ResponseData(session="s2"))
        async with recorder_buttons(h, timing=record.Timing(finish_hold_s=0.05)) as (
            session,
            _task,
        ):
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
    async with boot_recording() as h:
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
            await active.complete_cancelled(CANCELLED_BY_CALLER, SAVED)
            # The watcher fetched the recorder's verdict once the stream closed.
            await eventually(
                lambda: status.note() == "saved episode 0", message="the verdict"
            )


async def test_a_finished_episode_rearms_the_button():
    async with boot_recording() as h:
        rec = h.mocks.deps.recorder[0].record_episode
        async with recorder_buttons(h) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            active = await pending.accept()
            await stop_and_save(session, active, SAVED)
            # The finished episode re-arms X for a second one.
            pending = await press_until_goal(session, rec, {"x": False}, {"x": True})
            assert pending.request.task == TASK
            await pending.accept()


PRECONDITION_FAILED = record_episode.FailureKind.PRECONDITION_FAILED


async def test_a_refused_goal_leaves_the_panel_idle_and_rearms():
    async with boot_recording() as h:
        rec = h.mocks.deps.recorder[0].record_episode
        status = record.RecorderStatus()
        async with recorder_buttons(h, status) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            await pending.refuse(PRECONDITION_FAILED, "busy; try later")
            await eventually(
                lambda: status.note() == "record refused: busy", message="the cause"
            )
            # The refused goal re-arms the button immediately.
            pending = await press_until_goal(session, rec, {"x": False}, {"x": True})
            await pending.refuse(PRECONDITION_FAILED, "busy")
            assert not status.recording


async def test_feedback_reaches_the_panel_and_clears_at_the_end():
    async with boot_recording() as h:
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


async def test_y_must_be_held_to_finish():
    async with boot_recording() as h:
        fin = h.mocks.deps.recorder[0].finish_session
        fin.enqueue_response(finish_session.ResponseData(session="s1"))
        async with recorder_buttons(h, timing=record.Timing(finish_hold_s=0.3)) as (
            session,
            _task,
        ):
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


async def test_outcomes_reach_the_panel_as_notes():
    async with boot_recording() as h:
        rec = h.mocks.deps.recorder[0]
        status = record.RecorderStatus()
        async with recorder_buttons(
            h, status, timing=record.Timing(finish_hold_s=0.05)
        ) as (session, _task):
            await tap_x(session)
            pending = await rec.record_episode.next_goal(10.0)
            active = await pending.accept()
            await stop_and_save(session, active, SAVED)
            await eventually(
                lambda: status.note() == "saved episode 0", message="the saved note"
            )
            session.press(x=False)
            await asyncio.sleep(0.05)
            rec.finish_session.enqueue_response(finish_session.ResponseData(session="s1"))
            session.press(y=True)
            await eventually(
                lambda: status.note() == "session finished",
                message="the finished note",
            )


async def answer_the_finish_with_an_error_reply(fin, kind, message):
    """Answers the next finish_session call with an error reply of `kind`."""
    responder = await fin.next_request(10.0)
    await responder.respond_failure(kind, message)


@pytest.mark.parametrize(
    ("kind", "note"),
    [
        (finish_session.FailureKind.PRECONDITION_FAILED, "finish refused"),
        (finish_session.FailureKind.ATTEMPT_FAILED, "finish failed"),
    ],
)
async def test_a_finish_error_reply_notes_the_refusal_or_the_failure(kind, note):
    async with boot_recording() as h:
        fin = h.mocks.deps.recorder[0].finish_session
        status = record.RecorderStatus()
        async with recorder_buttons(
            h, status, timing=record.Timing(finish_hold_s=0.05)
        ) as (session, _task):
            session.press(y=True)
            await answer_the_finish_with_an_error_reply(fin, kind, "an episode is recording")
            await eventually(lambda: status.note() == note, message="the error reply note")


def test_notes_expire():
    status = record.RecorderStatus()
    status.set_note("saved episode 0")
    assert status.note()
    assert status.note(now_monotonic=time.monotonic() + FLASH_S + 1) == ""


async def test_a_refused_finish_is_survived():
    async with boot_recording() as h:
        fin = h.mocks.deps.recorder[0].finish_session
        async with recorder_buttons(h, timing=record.Timing(finish_hold_s=0.05)) as (
            session,
            task,
        ):
            session.press(y=True)
            await answer_the_finish_with_an_error_reply(
                fin, finish_session.FailureKind.PRECONDITION_FAILED, "an episode is recording"
            )
            await asyncio.sleep(0.2)
            # One hold, one attempt, and the loop is still alive.
            assert fin.captured_count() == 1
            assert not task.done()


async def test_a_stale_gap_does_not_stop_a_recording_with_x_still_held():
    """A dropout while the start press is still held must not read as a
    second press: the gap makes the input unknown, not released."""
    async with boot_recording() as h:
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
    async with boot_recording() as h:
        status = record.RecorderStatus()
        handle, active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        # The recorder ends the episode on its own, with a save.
        await active.complete_cancelled("max episode length reached", SAVED)
        await asyncio.wait_for(watcher, 10.0)
        # A watcher that finished while the stop was in flight already
        # cleared the row; the late stop must not wedge it as saving.
        await record._stop_episode(record._Episode(handle, watcher, None), status)
        assert not status.saving


async def test_a_recorder_death_mid_episode_is_reported(capsys):
    # The producer-gone result concludes only at the result deadline; keep
    # the test bounded without weakening the production value.
    async with boot_recording() as h:
        status = record.RecorderStatus()
        handle, _active = await _fired_episode(h)
        watcher = asyncio.create_task(
            record._watch_episode(handle, status, record.Timing(result_timeout_s=2.0))
        )
        await eventually(lambda: status.recording, message="the REC line")
        # The recorder process dies with the goal still active.
        await h.mocks.deps.recorder[0].stop()
        await asyncio.wait_for(watcher, 20.0)
    assert "recorder died mid-episode" in capsys.readouterr().out
    assert not status.recording


async def test_a_result_without_data_notes_the_goal_status():
    # The one real terminal status that carries no data is ABANDONED (a
    # producer that died mid-goal); the note must still name it.
    async with boot_recording() as h:
        status = record.RecorderStatus()
        handle, _active = await _fired_episode(h)
        watcher = asyncio.create_task(
            record._watch_episode(handle, status, record.Timing(result_timeout_s=2.0))
        )
        await eventually(lambda: status.recording, message="the REC line")
        await h.mocks.deps.recorder[0].stop()
        await asyncio.wait_for(watcher, 20.0)
        assert status.note() == "episode abandoned"


async def test_a_failed_episode_notes_the_discard(capsys):
    async with boot_recording() as h:
        status = record.RecorderStatus()
        handle, active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await active.fail(
            record_episode.FailureKind.ATTEMPT_FAILED, "episode save failed: disk full"
        )
        await asyncio.wait_for(watcher, 10.0)
        assert status.note() == "episode discarded"
    assert "episode discarded: attempt_failed: episode save failed: disk full" in (
        capsys.readouterr().out
    )


async def test_an_episode_the_recorder_ends_itself_notes_the_save_and_logs_the_cause(capsys):
    async with boot_recording() as h:
        status = record.RecorderStatus()
        handle, active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await active.complete_cancelled("link backbone_inst/arm stale (0.30s behind)", SAVED)
        await asyncio.wait_for(watcher, 10.0)
        assert status.note() == "saved episode 0"
    assert "episode 0 saved, 7 frames: link backbone_inst/arm stale (0.30s behind)" in (
        capsys.readouterr().out
    )


async def test_a_saved_episode_whose_videos_miss_frames_says_so(capsys):
    """The videos of the episode miss frames: the note and the log give the
    count, and the log says not to train on the episode."""
    async with boot_recording() as h:
        status = record.RecorderStatus()
        handle, active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await active.complete_cancelled(
            "link backbone_inst/arm stale (0.30s behind)",
            record_episode.ResultResponseData(
                episode_index=2, frames_recorded=7, video_frames_missing=3
            ),
        )
        await asyncio.wait_for(watcher, 10.0)
        assert status.note() == "saved episode 2, 3 video frame(s) missing"
    assert (
        "episode 2 saved, 7 frames: link backbone_inst/arm stale (0.30s behind); "
        "3 video frame(s) missing, do not train on it"
    ) in capsys.readouterr().out


class SpyStatus(record.RecorderStatus):
    """Records every value written to finish_held, first write included."""

    @property
    def finish_held(self):
        return self.__dict__.get("value", False)

    @finish_held.setter
    def finish_held(self, value):
        self.__dict__.setdefault("seen", []).append(value)
        self.__dict__["value"] = value


async def test_a_grazed_y_never_shows_finishing():
    hold = record.Timing(finish_hold_s=0.3)
    async with boot_recording() as h:
        fin = h.mocks.deps.recorder[0].finish_session
        grazed = SpyStatus()
        async with recorder_buttons(h, grazed, timing=hold) as (session, _task):
            session.press(y=True)
            await asyncio.sleep(0.05)
            session.press(y=False)
            await asyncio.sleep(0.4)
        assert True not in grazed.__dict__["seen"]
        fin.enqueue_response(finish_session.ResponseData(session="s1"))
        held = SpyStatus()
        async with recorder_buttons(h, held, timing=hold) as (session, _task):
            session.press(y=True)
            await eventually(
                lambda: True in held.__dict__.get("seen", []),
                message="the finishing row",
            )


async def test_x_stop_shows_saving_until_the_result():
    async with boot_recording() as h:
        status = record.RecorderStatus()
        handle, active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await eventually(
            lambda: status.recording and not status.saving, message="the REC line"
        )
        # The stop is delivered; the recorder has not answered yet.
        await record._stop_episode(record._Episode(handle, watcher, None), status)
        assert status.saving  # result pending: the recorder is saving
        await active.complete_cancelled(CANCELLED_BY_CALLER, SAVED)
        await asyncio.wait_for(watcher, 10.0)
        assert not status.saving and not status.recording
        assert status.note() == "saved episode 0"


async def test_a_failed_cancel_does_not_claim_saving():
    timing = record.Timing(cancel_timeout_s=0.5, result_timeout_s=2.0)
    async with boot_recording() as h:
        status = record.RecorderStatus()
        handle, _active = await _fired_episode(h)
        watcher = asyncio.create_task(record._watch_episode(handle, status, timing))
        await eventually(lambda: status.recording, message="the REC line")
        # The recorder dies before the stop: the cancel cannot be delivered,
        # so the row must not claim a save is in progress.
        await h.mocks.deps.recorder[0].stop()
        await record._stop_episode(
            record._Episode(handle, watcher, None), status, timing
        )
        assert not status.saving
        await asyncio.wait_for(watcher, 20.0)


async def test_each_episode_carries_the_label_live_when_it_started():
    # The operator retitles between takes; the episode already running keeps
    # the label it was fired with.
    labels = ["wipe the table", "stack the blocks"]
    async with boot_recording() as h:
        rec = h.mocks.deps.recorder[0].record_episode
        async with recorder_buttons(
            h, read_task=lambda: labels.pop(0) if len(labels) > 1 else labels[0]
        ) as (session, _task):
            await tap_x(session)
            pending = await rec.next_goal(10.0)
            assert pending.request.task == "wipe the table"
            active = await pending.accept()
            await stop_and_save(session, active, SAVED)
            # The next take is fired with the label live by then.
            pending = await press_until_goal(session, rec, {"x": False}, {"x": True})
            assert pending.request.task == "stack the blocks"
            await pending.accept()


async def test_a_press_with_no_recorder_on_the_chosen_robot_is_inert(capsys):
    async with boot() as h:  # nothing paired, nothing bound: no robot to record
        session = FakeSession()
        token = FakeToken()
        task = asyncio.create_task(
            record.run_recorder_buttons(
                h.node_runner,
                action_module=record_episode,
                finish_module=finish_session,
                status=record.RecorderStatus(),
                session=session,
                robots=robots_of(h),
                settings=settings(),
                token=token,
                read_task=lambda: TASK,
            )
        )
        try:
            await tap_x(session)
            await eventually(
                lambda: "has no recorder" in capsys.readouterr().out,
                message="the inert press said once",
            )
        finally:
            token.cancel()
            await asyncio.wait_for(task, 5.0)


def test_the_row_shows_for_the_chosen_robots_recorder_and_for_an_episode_in_flight():
    status = record.RecorderStatus()
    assert record.shown_recorder(status, chosen_has_recorder=True) is status
    assert record.shown_recorder(status, chosen_has_recorder=False) is None
    # Recording on a robot left behind: the row stays until the save lands.
    status.recording = True
    assert record.shown_recorder(status, chosen_has_recorder=False) is status
    status.recording, status.saving = False, True
    assert record.shown_recorder(status, chosen_has_recorder=False) is status
    # Saved: the outcome stays on the row for its few seconds.
    status.saving = False
    status.set_note("saved episode 3")
    assert record.shown_recorder(status, chosen_has_recorder=False) is status


def test_y_finishes_the_session_of_the_episode_in_flight_whichever_robot_is_chosen():
    producers = {"alpha": "alpha's finish", "bravo": "bravo's finish"}
    robots = SimpleNamespace(
        producer_of=lambda _module, robot: producers.get(robot),
        producer_for=lambda _module: producers["bravo"],
    )
    recording_alpha = record._Episode(handle=None, watcher=None, robot="alpha")
    assert record.finish_target(robots, None, recording_alpha) == "alpha's finish"
    assert record.finish_target(robots, None, None) == "bravo's finish"
