import asyncio
import contextlib
import time
from types import SimpleNamespace

from tests.helpers import POSE, FakeToken, default_parameters
from xr_commander import config, record
from xr_commander.devices import HandSample, XrFrame


class FakeSession:
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


class FakeHandle:
    """One goal on the fake recorder: feedback until stopped, then a result."""

    accepted = True
    reason = ""

    def __init__(self, log, frames):
        self._log = log
        self._frames = list(frames)
        self._stopped = asyncio.Event()

    async def cancel_goal(self, _timeout):
        self._log.append("cancel")
        self._stopped.set()

    async def on_next_feedback_message(self):
        if self._frames:
            return SimpleNamespace(
                frames_recorded=self._frames.pop(0), disk_free_bytes=10**12
            )
        await self._stopped.wait()
        raise RuntimeError("stream closed")

    async def get_result(self, _timeout):
        self._log.append("result")
        return SimpleNamespace(
            status=SimpleNamespace(name="SUCCEEDED"),
            data=SimpleNamespace(
                episode_index=0, frames_recorded=7, discarded=False, error=None
            ),
        )


class FakeActionModule:
    """The consumed-action module surface run_recorder_buttons drives."""

    def __init__(self, log, frames=(), accepted=True):
        self.log = log
        self.GoalRequest = lambda task: task
        outer = self

        class Handle:
            @staticmethod
            async def fire_goal(_runner, _target, request, _timeout, _qos):
                outer.log.append(f"fire:{request}")
                handle = FakeHandle(outer.log, frames)
                handle.accepted = accepted
                outer.handles.append(handle)
                return handle

        self.ActionHandle = Handle
        self.handles = []

    def bound_producers(self, _runner):
        return [SimpleNamespace(instance_id="recorder_inst")]


class FakeFinishModule:
    def __init__(self, log, error=None):
        self._log = log
        self._error = error

    async def poll(self, _runner, _target, _timeout):
        self._log.append("finish")
        return SimpleNamespace(data=SimpleNamespace(session="s1", error=self._error))


def settings():
    return config.from_parameters(default_parameters())


TASK = "wipe the table"


def drive(script, *, action=None, finish=None, status=None, read_task=None):
    """Run the button task over scripted (x, y) presses, one per 30 ms; a
    None step is a headset gap (no tracked hands)."""
    log = []
    action = action or FakeActionModule(log)
    finish = finish or FakeFinishModule(log)
    status = status or record.RecorderStatus()
    session = FakeSession()
    token = FakeToken()

    async def run():
        task = asyncio.create_task(
            record.run_recorder_buttons(
                None,
                action_module=action,
                finish_module=finish,
                status=status,
                session=session,
                settings=settings(),
                token=token,
                read_task=read_task or (lambda: TASK),
            )
        )
        for step in script:
            if step is None:
                session.hands = {}
            else:
                x, y = step
                session.press(x=x, y=y)
            await asyncio.sleep(0.03)
        token.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())
    return log


def test_x_fires_one_goal_per_rising_edge_with_the_task():
    log = drive([(False, False), (True, False), (True, False)])
    assert log == [f"fire:{TASK}"]


def test_x_already_held_at_first_tracking_does_not_start():
    log = drive([(True, False), (True, False)])
    assert [e for e in log if e.startswith("fire")] == []
    # An observed release and then a press is the first real edge.
    log = drive([(True, False), (False, False), (True, False)])
    assert [e for e in log if e.startswith("fire")] == [f"fire:{TASK}"]


def test_one_physical_y_hold_finishes_once_across_a_gap(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    held = (False, True)
    log = drive([held, held, held, None, held, held, held])
    assert log.count("finish") == 1
    # An observed release re-arms the finish.
    log = drive([held, held, held, (False, False), held, held, held])
    assert log.count("finish") == 2


def test_x_again_stops_and_saves_instead_of_firing_twice():
    log = drive([(False, False), (True, False), (False, False), (True, False)])
    assert [e for e in log if e.startswith("fire")] == [f"fire:{TASK}"]
    assert "cancel" in log
    # The watcher fetched the recorder's verdict once the stream closed.
    assert "result" in log


def test_a_finished_episode_rearms_the_button():
    log = drive(
        [
            (False, False),
            (True, False),
            (False, False),
            (True, False),
            (False, False),
            (True, False),
        ]
    )
    assert len([e for e in log if e.startswith("fire")]) == 2


def test_a_rejected_goal_leaves_the_panel_idle_and_rearms():
    log = []
    action = FakeActionModule(log, accepted=False)
    status = record.RecorderStatus()
    drive(
        [(False, False), (True, False), (False, False), (True, False)],
        action=action,
        status=status,
    )
    assert len([e for e in log if e.startswith("fire")]) == 2
    assert not status.recording


def test_feedback_reaches_the_panel_and_clears_at_the_end(monkeypatch):
    log = []
    action = FakeActionModule(log, frames=(1, 2, 3))
    status = record.RecorderStatus()
    seen = []

    original = FakeHandle.on_next_feedback_message

    async def spy(self):
        message = await original(self)
        seen.append((status.recording, message.frames_recorded))
        return message

    monkeypatch.setattr(FakeHandle, "on_next_feedback_message", spy)
    drive(
        [(False, False), (True, False), (False, False), (True, False)],
        action=action,
        status=status,
    )
    assert seen  # feedback flowed while recording
    assert all(recording for recording, _ in seen)
    assert not status.recording
    assert status.frames == 0


def test_y_must_be_held_to_finish(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    graze = drive([(False, True), (False, False)])
    assert "finish" not in graze
    held = drive([(False, True), (False, True), (False, True), (False, True)])
    assert held.count("finish") == 1


def test_outcomes_reach_the_panel_as_notes(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    status = record.RecorderStatus()
    drive(
        [(False, False), (True, False), (False, False), (True, False), (False, False)],
        status=status,
    )
    assert status.note() == "saved episode 0"
    finished = record.RecorderStatus()
    drive(
        [(False, True), (False, True), (False, True), (False, True)],
        status=finished,
    )
    assert finished.note() == "session finished"


def test_a_refused_finish_notes_the_refusal(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    log = []
    status = record.RecorderStatus()
    drive(
        [(False, True), (False, True), (False, True), (False, True)],
        finish=FakeFinishModule(log, error="an episode is recording"),
        status=status,
    )
    assert status.note() == "finish refused"


def test_notes_expire():
    status = record.RecorderStatus()
    status.set_note("saved episode 0")
    assert status.note()
    assert status.note(now_monotonic=time.monotonic() + record._NOTE_S + 1) == ""


def test_a_refused_finish_is_survived(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    log = []
    finish = FakeFinishModule(log, error="an episode is recording")
    drive(
        [(False, True), (False, True), (False, True), (False, True)],
        finish=finish,
    )
    assert log.count("finish") == 1


def test_a_stale_gap_does_not_stop_a_recording_with_x_still_held():
    """A dropout while the start press is still held must not read as a
    second press: the gap makes the input unknown, not released."""
    log = drive(
        [(False, False), (True, False), (True, False), None, None, (True, False)]
    )
    assert [e for e in log if e.startswith("fire")] == [f"fire:{TASK}"]
    assert "cancel" not in log
    # A real release and press still stops it.
    log = drive(
        [
            (False, False),
            (True, False),
            None,
            (True, False),
            (False, False),
            (True, False),
        ]
    )
    assert "cancel" in log


def test_a_stop_racing_a_finished_watcher_does_not_claim_saving():
    log = []
    status = record.RecorderStatus()

    async def run():
        handle = FakeHandle(log, ())
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        handle._stopped.set()  # the episode ends on its own
        await watcher
        await record._stop_episode(record._Episode(handle, watcher), status)
        assert not status.saving

    asyncio.run(run())


class DyingHandle(FakeHandle):
    """Feedback dies like a crashed producer; the result still answers."""

    async def on_next_feedback_message(self):
        raise ConnectionError("producer gone")


def test_a_recorder_death_mid_episode_is_reported(capsys):
    log = []
    status = record.RecorderStatus()

    async def run():
        await record._watch_episode(DyingHandle(log, ()), status)

    asyncio.run(run())
    assert "recorder died mid-episode" in capsys.readouterr().out
    assert not status.recording


class ResultOnlyHandle(FakeHandle):
    def __init__(self, log, result):
        super().__init__(log, ())
        self._result = result

    async def on_next_feedback_message(self):
        raise RuntimeError("stream closed")

    async def get_result(self, _timeout):
        return self._result


def test_a_result_without_data_notes_the_goal_status():
    status = record.RecorderStatus()
    result = SimpleNamespace(status=SimpleNamespace(name="ABORTED"), data=None)

    asyncio.run(record._watch_episode(ResultOnlyHandle([], result), status))
    assert status.note() == "episode aborted"


def test_a_discarded_episode_notes_the_discard():
    status = record.RecorderStatus()
    result = SimpleNamespace(
        status=SimpleNamespace(name="SUCCEEDED"),
        data=SimpleNamespace(
            episode_index=-1, frames_recorded=0, discarded=True, error="boom"
        ),
    )

    asyncio.run(record._watch_episode(ResultOnlyHandle([], result), status))
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


def test_a_grazed_y_never_shows_finishing(monkeypatch):
    monkeypatch.setattr(record, "_FINISH_HOLD_S", 0.05)
    grazed = SpyStatus()
    drive([(False, True), (False, False), (False, False)], status=grazed)
    assert True not in grazed.__dict__["seen"]
    held = SpyStatus()
    drive([(False, True), (False, True), (False, True), (False, True)], status=held)
    assert True in held.__dict__["seen"]


def test_x_stop_shows_saving_until_the_result():
    log = []
    status = record.RecorderStatus()

    class SlowSaveHandle(FakeHandle):
        def __init__(self):
            super().__init__(log, ())
            self.release = asyncio.Event()

        async def get_result(self, timeout):
            await self.release.wait()
            return await super().get_result(timeout)

    async def run():
        handle = SlowSaveHandle()
        watcher = asyncio.create_task(record._watch_episode(handle, status))
        await asyncio.sleep(0.01)
        assert status.recording and not status.saving
        await record._stop_episode(record._Episode(handle, watcher), status)
        await asyncio.sleep(0.01)
        assert status.saving  # result pending: the recorder is saving
        handle.release.set()
        await watcher
        assert not status.saving and not status.recording
        assert status.note() == "saved episode 0"

    asyncio.run(run())


def test_a_failed_cancel_does_not_claim_saving():
    log = []
    status = record.RecorderStatus()

    class DeafHandle(FakeHandle):
        async def cancel_goal(self, _timeout):
            raise ConnectionError("recorder gone")

    async def run():
        watcher = asyncio.create_task(asyncio.sleep(0))
        await record._stop_episode(
            record._Episode(DeafHandle(log, ()), watcher), status
        )
        assert not status.saving
        await watcher

    asyncio.run(run())


def test_each_episode_carries_the_label_live_when_it_started():
    # The operator retitles between takes; the episode already running keeps
    # the label it was fired with.
    labels = ["wipe the table", "stack the blocks"]
    log = drive(
        [
            (False, False),
            (True, False),
            (False, False),
            (True, False),
            (False, False),
            (True, False),
        ],
        read_task=lambda: labels.pop(0) if len(labels) > 1 else labels[0],
    )
    assert [e for e in log if e.startswith("fire")] == [
        "fire:wipe the table",
        "fire:stack the blocks",
    ]


def test_no_bound_recorder_is_inert():
    log = []
    action = FakeActionModule(log)
    action.bound_producers = lambda _runner: []
    out = drive([(True, False)], action=action)
    assert [e for e in out if e.startswith("fire")] == []
