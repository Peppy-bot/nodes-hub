"""GPU-free iteration pacing and lifecycle tests with a wall clock and Event fakes."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
_PERIOD = 1.0 / 60
_PHASES = ["update", "bridge", "runtime", "scene", "forces", "targets"]
_MAIN_RATE_LIMIT_ENABLED = "/app/runLoops/main/rateLimitEnabled"
_RENDER_MODE = "/rtx/rendermode"
_ANTI_ALIASING_OP = "/rtx/post/aa/op"


@pytest.fixture
def loop(monkeypatch):
    bridge = Mock(is_ready=False)
    bridge_module = ModuleType("bridge_extension")
    bridge_module.IsaacBridgeExtension = Mock(return_value=bridge)
    monkeypatch.setitem(sys.modules, "bridge_extension", bridge_module)
    monkeypatch.syspath_prepend(str(_ROBOT_DIR))
    spec = importlib.util.spec_from_file_location("_launcher_under_test", _ROBOT_DIR / "_launcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    state = SimpleNamespace(
        now=100.0,
        frames=3,
        running=True,
        ready_after=1,
        starts=[],
        waits=[],
        trace=[],
        costs=dict.fromkeys(_PHASES, 0.0),
        module=module,
        bridge=bridge,
        stop=threading.Event(),
        ready=threading.Event(),
        app=Mock(),
        commander=Mock(),
        scene=Mock(),
        timeline=Mock(),
        settings=Mock(spec=["get", "get_as_bool", "set_bool"]),
        settings_values={_MAIN_RATE_LIMIT_ENABLED: False},
        # What Kit reports it renders with; the launch asked for exactly this.
        render_values={_RENDER_MODE: "RealTimePathTracing", _ANTI_ALIASING_OP: 3},
        settings_trace=[],
        settings_threads=[],
    )

    def get_settings():
        state.settings_threads.append(threading.current_thread())
        return state.settings

    def get_as_bool(key):
        value = state.settings_values[key]
        state.settings_trace.append(("read", value))
        return value

    def set_bool(key, value):
        state.settings_trace.append(("write", value))
        state.settings_values[key] = value

    # Install the SDK fake after importing the module, before running the loop.
    carb = ModuleType("carb")
    carb.settings = ModuleType("carb.settings")
    carb.settings.get_settings = Mock(side_effect=get_settings)
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.settings", carb.settings)
    state.get_settings = carb.settings.get_settings
    state.settings.get_as_bool.side_effect = get_as_bool
    state.settings.set_bool.side_effect = set_bool
    state.settings.get.side_effect = lambda key: state.render_values[key]

    monkeypatch.setattr(module.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(module.time, "sleep", Mock(side_effect=AssertionError("use Event.wait")))
    monkeypatch.setattr(module, "RuntimeCommanderServer", Mock(return_value=state.commander))

    def phase(name):
        state.trace.append(name)
        state.now += state.costs[name]

    def update():
        assert state.settings_values[_MAIN_RATE_LIMIT_ENABLED] is False
        state.settings_trace.append(("update",))
        state.starts.append(state.now)
        phase("update")

    def step():
        phase("bridge")
        bridge.is_ready = len(state.starts) >= state.ready_after

    def wait(delay):
        assert delay > 0
        state.trace.append("wait")
        state.waits.append(delay)
        state.now += delay
        return state.stop.is_set()

    state.wait = wait
    monkeypatch.setattr(state.stop, "wait", Mock(side_effect=wait))
    set_ready = state.ready.set

    def mark_ready():
        state.trace.append("ready")
        set_ready()

    monkeypatch.setattr(state.ready, "set", Mock(side_effect=mark_ready))
    state.app.is_running.side_effect = lambda: state.running and len(state.starts) < state.frames
    state.app.update.side_effect = update
    bridge.step.side_effect = step
    state.commander.process_pending.side_effect = lambda launcher: phase("runtime")
    state.scene.process_pending.side_effect = lambda launcher: phase("scene")
    state.launcher = module.SimLauncher(
        state.app,
        Path("/robot.usd"),
        state.ready,
        state.stop,
        object(),
        state.scene,
        state_rate_hz=17,
        cameras_enabled=False,
        frame_rate_hz=60,
        render_mode="RealTimePathTracing",
        anti_aliasing=3,
    )
    monkeypatch.setattr(state.launcher, "_update_runtime_forces", lambda: phase("forces"))
    monkeypatch.setattr(state.launcher, "_apply_runtime_arm_targets", lambda: phase("targets"))
    state.launcher._extension = bridge

    # Stub engine startup, not the loop or its cleanup path. No sockets or
    # threads start, and run() still constructs and shuts down its bridge.
    for name in ("_load_stage", "_setup_environment", "_warmup"):
        monkeypatch.setattr(state.launcher, name, Mock())
    monkeypatch.setattr(state.launcher, "_discover_isaac_props", Mock(return_value={}))

    def start_timeline():
        state.launcher._timeline = state.timeline

    monkeypatch.setattr(state.launcher, "_start_timeline", start_timeline)
    state.commander.stop.side_effect = lambda: state.trace.append("commander.stop")
    bridge.shutdown.side_effect = lambda: state.trace.append("bridge.shutdown")
    state.timeline.stop.side_effect = lambda: state.trace.append("timeline.stop")
    state.app.close.side_effect = lambda: state.trace.append("app.close")
    return state


def test_settings_interface_is_cached_on_the_loop_thread_without_redundant_writes(loop):
    loop.get_settings.assert_not_called()
    loop.launcher._run_loop()

    loop.get_settings.assert_called_once_with()
    assert loop.settings_threads == [threading.main_thread()]
    assert loop.settings_trace == [("read", False), ("update",)] * loop.frames
    assert loop.settings.get_as_bool.call_count == loop.frames
    loop.settings.set_bool.assert_not_called()


@pytest.mark.parametrize(
    "reset_after_frames",
    [(0,), (1,), (1, 3), (0, 1, 3)],
    ids=["startup", "connection", "reconnection", "startup-and-reconnections"],
)
def test_streamer_limiter_resets_are_cleared_before_the_next_update(loop, reset_after_frames):
    loop.frames = 5
    loop.settings_values[_MAIN_RATE_LIMIT_ENABLED] = 0 in reset_after_frames
    update = loop.app.update.side_effect

    def streaming_update():
        update()
        if len(loop.starts) in reset_after_frames:
            loop.settings_values[_MAIN_RATE_LIMIT_ENABLED] = True

    loop.app.update.side_effect = streaming_update
    loop.launcher.run()

    expected = []
    for frame in range(loop.frames):
        enabled = frame in reset_after_frames
        expected.append(("read", enabled))
        if enabled:
            expected.append(("write", False))
        expected.append(("update",))
    assert loop.settings_trace == expected
    # The render check before the loop fetches the interface; the loop once more.
    assert loop.settings_threads == [threading.main_thread()] * 2
    assert loop.settings.set_bool.call_count == len(reset_after_frames)
    loop.settings.set_bool.assert_called_with(_MAIN_RATE_LIMIT_ENABLED, False)
    assert loop.starts == pytest.approx([100.0 + i * _PERIOD for i in range(loop.frames)])
    assert loop.bridge.step.call_count == loop.frames


def test_initial_frame_is_immediate_and_orders_readiness_before_queues(loop):
    loop.frames = 1
    loop.launcher._run_loop()

    assert loop.starts == [100.0]
    assert loop.waits == []
    assert loop.trace == ["update", "bridge", "ready", "runtime", "scene", "forces", "targets"]
    loop.bridge.step.assert_called_once_with()
    loop.commander.process_pending.assert_called_once_with(loop.launcher)
    loop.scene.process_pending.assert_called_once_with(loop.launcher)
    assert loop.ready.is_set()


def test_whole_iteration_work_is_deducted_from_the_period(loop):
    loop.costs.update(update=0.004, bridge=0.003, runtime=0.001, scene=0.002, forces=0.001, targets=0.001)
    loop.launcher._run_loop()

    assert loop.starts == pytest.approx([100.0 + i * _PERIOD for i in range(3)])
    assert loop.waits == pytest.approx([_PERIOD - 0.012] * 2)
    assert loop.bridge.step.call_count == 3
    loop.ready.set.assert_called_once_with()
    for phase in _PHASES:
        assert loop.trace.count(phase) == 3
    assert loop.trace[-1] == "targets", "no sleep is appended to a finished frame"


def test_readiness_waits_for_bridge_setup_but_queues_run_each_frame(loop):
    loop.ready_after = 2
    loop.launcher._run_loop()

    assert loop.trace == (
        _PHASES + ["wait", "update", "bridge", "ready", "runtime", "scene", "forces", "targets"]
        + ["wait"] + _PHASES
    )
    loop.ready.set.assert_called_once_with()


@pytest.mark.parametrize("stalled_phase", ["update", "bridge", "scene"])
def test_stalled_frame_resynchronizes_without_replaying_missed_steps(loop, stalled_phase):
    update = loop.app.update.side_effect

    def stalled_update():
        loop.costs[stalled_phase] = 0.25 if not loop.starts else 0.0
        update()

    loop.app.update.side_effect = stalled_update
    loop.launcher._run_loop()

    assert loop.starts == pytest.approx([100.0, 100.25, 100.25 + _PERIOD])
    assert loop.waits == pytest.approx([_PERIOD])
    assert loop.bridge.step.call_count == 3
    assert loop.trace[6:8] == ["targets", "update"], "overdue work does not wait"


def test_early_wait_return_rechecks_the_deadline(loop):
    loop.frames = 2

    def early_wait(delay):
        if not loop.waits:
            loop.trace.append("wait")
            loop.waits.append(delay)
            loop.settings_values[_MAIN_RATE_LIMIT_ENABLED] = True
            loop.now += delay / 4
            return False
        # A connection during the wait does not cause an early update or check.
        assert loop.settings_trace == [("read", False), ("update",)]
        assert loop.settings_values[_MAIN_RATE_LIMIT_ENABLED] is True
        return loop.wait(delay)

    loop.stop.wait.side_effect = early_wait
    loop.launcher._run_loop()

    assert loop.starts == pytest.approx([100.0, 100.0 + _PERIOD])
    assert loop.waits == pytest.approx([_PERIOD, _PERIOD * 3 / 4])
    assert loop.trace[6:10] == ["targets", "wait", "wait", "update"]
    assert loop.bridge.step.call_count == 2
    assert loop.settings_trace == [
        ("read", False), ("update",), ("read", True), ("write", False), ("update",),
    ]
    loop.settings.set_bool.assert_called_once_with(_MAIN_RATE_LIMIT_ENABLED, False)


def test_late_wait_return_does_not_shift_the_absolute_grid(loop):
    def late_wait(delay):
        result = loop.wait(delay)
        if len(loop.waits) == 1:
            loop.now += 0.002
        return result

    loop.stop.wait.side_effect = late_wait
    loop.launcher._run_loop()

    assert loop.starts == pytest.approx([100.0, 100.0 + _PERIOD + 0.002, 100.0 + 2 * _PERIOD])
    assert loop.waits == pytest.approx([_PERIOD, _PERIOD - 0.002])
    assert loop.bridge.step.call_count == 3


@pytest.mark.parametrize("exit_reason", ["stop", "app_closed"])
def test_shutdown_during_wait_prevents_an_extra_frame_and_closes_orderly(loop, exit_reason):
    def interrupted_wait(delay):
        loop.waits.append(delay)
        loop.settings_values[_MAIN_RATE_LIMIT_ENABLED] = True
        if exit_reason == "stop":
            loop.stop.set()
        else:
            loop.running = False
        return loop.stop.is_set()

    loop.stop.wait.side_effect = interrupted_wait
    loop.launcher.run()

    assert loop.starts == [100.0]
    assert loop.waits == pytest.approx([_PERIOD])
    loop.bridge.step.assert_called_once_with()
    assert not loop.ready.is_set()
    assert loop.trace[-4:] == ["commander.stop", "bridge.shutdown", "timeline.stop", "app.close"]
    loop.app.close.assert_called_once_with()
    loop.module.IsaacBridgeExtension.assert_called_once_with(loop.launcher._io, 17, False)
    assert loop.settings_trace == [("read", False), ("update",)]
    assert loop.settings_values[_MAIN_RATE_LIMIT_ENABLED] is True
    loop.settings.set_bool.assert_not_called()


@pytest.mark.parametrize("exit_reason", ["stop", "app_closed"])
def test_no_frame_when_already_stopped(loop, exit_reason):
    if exit_reason == "stop":
        loop.stop.set()
    else:
        loop.running = False
    loop.launcher._run_loop()

    loop.app.update.assert_not_called()
    loop.bridge.step.assert_not_called()
    loop.stop.wait.assert_not_called()


def test_deliberate_wait_is_excluded_from_slow_work_diagnostic(loop, caplog):
    loop.frames = 2

    def overslept_wait(delay):
        result = loop.wait(delay)
        loop.now += 0.3
        return result

    loop.stop.wait.side_effect = overslept_wait
    loop.launcher._run_loop()

    assert len(loop.waits) == 1
    assert loop.bridge.step.call_count == 2
    assert not [record for record in caplog.records if record.msg.startswith("slow loop")]


def test_slow_work_diagnostic_keeps_its_phase_breakdown(loop, caplog):
    loop.frames = 1
    loop.costs.update(update=0.04, bridge=0.025, runtime=0.005, scene=0.01)
    loop.launcher._run_loop()

    (record,) = [record for record in caplog.records if record.msg.startswith("slow loop")]
    assert record.args == pytest.approx((80.0, 40.0, 40.0))


def test_bridge_failure_still_closes_the_app(loop):
    loop.bridge.step.side_effect = RuntimeError("bridge failed")
    with pytest.raises(RuntimeError, match="bridge failed"):
        loop.launcher.run()

    assert loop.trace[-4:] == ["commander.stop", "bridge.shutdown", "timeline.stop", "app.close"]
    loop.app.close.assert_called_once_with()
    assert not loop.ready.is_set()
    loop.commander.process_pending.assert_not_called()
    loop.scene.process_pending.assert_not_called()


def test_render_profile_is_read_after_the_warmup_and_before_the_timeline(loop, monkeypatch):
    # Kit swaps the renderer or anti-aliasing on the first rendered frames, so
    # a read before the warmup would pass on a launch that then streams noise.
    order = []
    loop.launcher._warmup.side_effect = lambda: order.append("warmup")
    loop.settings.get.side_effect = lambda key: (order.append(key), loop.render_values[key])[1]
    start_timeline = loop.launcher._start_timeline

    def traced_start_timeline():
        order.append("timeline")
        start_timeline()

    monkeypatch.setattr(loop.launcher, "_start_timeline", traced_start_timeline)
    loop.launcher.run()

    assert order == ["warmup", _RENDER_MODE, _ANTI_ALIASING_OP, "timeline"]
    assert loop.bridge.step.call_count == loop.frames


@pytest.mark.parametrize(
    "reported",
    [{_ANTI_ALIASING_OP: 1}, {_RENDER_MODE: "PathTracing"}],
    ids=["dlss-dropped-for-taa", "renderer-substituted"],
)
def test_substituted_render_profile_stops_before_the_timeline_and_closes_isaac(loop, reported):
    loop.render_values.update(reported)
    with pytest.raises(RuntimeError, match="NGX core library") as failure:
        loop.launcher.run()

    assert "'RealTimePathTracing' and 3" in str(failure.value)
    assert loop.launcher._timeline is None
    loop.app.update.assert_not_called()
    loop.bridge.step.assert_not_called()
    assert loop.trace[-3:] == ["commander.stop", "bridge.shutdown", "app.close"]
    loop.app.close.assert_called_once_with()
    assert not loop.ready.is_set()
