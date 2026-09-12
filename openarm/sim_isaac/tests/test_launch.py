"""Exercise startup through fake runtimes and parse the packaged Kit experience."""

import builtins
import importlib.util
import logging
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
_LAUNCH_PATH = _ROBOT_DIR / "openarm" / "launch.py"
_KIT_PATH = _ROBOT_DIR / "config" / "openarm.sim.kit"
_STREAM_PREFIX = "--/exts/omni.kit.livestream.app/primaryStream/"


@pytest.fixture
def startup(monkeypatch):
    for name in (
        "PEPPY_ISAAC_PUBLIC_IP", "PEPPY_ISAAC_SIGNAL_PORT", "PEPPY_ISAAC_STREAM_PORT",
        "PEPPY_ROBOT_ASSETS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", [str(_LAUNCH_PATH), "--/app/window/title=Operator"])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(logging, "basicConfig", Mock())
    runtime = ModuleType("peppylib.runtime")
    runtime.NodeBuilder = Mock()
    monkeypatch.setitem(sys.modules, "peppylib", ModuleType("peppylib"))
    monkeypatch.setitem(sys.modules, "peppylib.runtime", runtime)

    state = SimpleNamespace(constructed=False, trace=[], argv=[], app=Mock())

    def construct(config, *, experience):
        state.constructed = True
        state.trace.append("app")
        state.argv = sys.argv.copy()
        return state.app

    isaacsim = ModuleType("isaacsim")
    isaacsim.SimulationApp = Mock(side_effect=construct)
    launcher = ModuleType("_launcher")
    launcher.SimLauncher = Mock()
    launcher.SimLauncher.return_value.run.side_effect = lambda: state.trace.append("run")
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "_launcher", launcher)
    state.simulation_app = isaacsim.SimulationApp
    state.sim_launcher = launcher.SimLauncher
    import_module = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in ("carb", "omni", "_launcher") or name.startswith(("carb.", "omni.")):
            assert state.constructed, f"{name} imported before SimulationApp construction"
        if name == "isaacsim":
            state.trace.append("import isaacsim")
        if name == "_launcher":
            state.trace.append("import launcher")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    def load(*, headless=True, cameras_enabled=False, hardware_version="v2"):
        spec = importlib.util.spec_from_file_location("_openarm_launch_under_test", _LAUNCH_PATH)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        assert state.trace == [], "importing launch must not initialize Isaac"
        module._handoff["value"] = module._SimHandoff(
            io=object(),
            scene_actions=object(),
            state_rate_hz=17,
            headless=headless,
            hardware_version=hardware_version,
            cameras_enabled=cameras_enabled,
        )
        module._handoff_ready.set()
        state.thread = Mock()
        state.thread.return_value.start.side_effect = lambda: state.trace.append("node thread")
        monkeypatch.setattr(module.threading, "Thread", state.thread)
        state.module = module
        return state

    return load


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("cameras_enabled", [False, True])
@pytest.mark.parametrize("overrides", [False, True])
def test_launch_selects_extensions_before_construction_and_preserves_handoff(
    startup, monkeypatch, headless, cameras_enabled, overrides,
):
    if overrides:
        monkeypatch.setenv("PEPPY_ISAAC_PUBLIC_IP", " 192.0.2.5 ")
        monkeypatch.setenv("PEPPY_ISAAC_SIGNAL_PORT", " 49200 ")
        monkeypatch.setenv("PEPPY_ISAAC_STREAM_PORT", " 48098 ")
    state = startup(headless=headless, cameras_enabled=cameras_enabled)
    state.module.main()

    state.simulation_app.assert_called_once_with(
        {
            "headless": headless,
            "renderer": "RealTimePathTracing",
            "anti_aliasing": 3,
            "width": 1280,
            "height": 720,
        },
        experience=str(_KIT_PATH),
    )
    assert _KIT_PATH.is_file()
    assert state.argv[:2] == [str(_LAUNCH_PATH), "--/app/window/title=Operator"]
    extensions = [state.argv[i + 1] for i, arg in enumerate(state.argv) if arg == "--enable"]
    assert extensions == (
        (["omni.kit.livestream.app"] if headless else [])
        + (["omni.replicator.core"] if cameras_enabled else [])
    )
    stream_settings = dict(
        arg.removeprefix(_STREAM_PREFIX).split("=", 1)
        for arg in state.argv if arg.startswith(_STREAM_PREFIX)
    )
    if headless:
        assert stream_settings == {
            "targetFps": "60",
            "signalPort": "49200" if overrides else "49100",
            "streamPort": "48098" if overrides else "47998",
            **({"publicIp": "192.0.2.5"} if overrides else {}),
        }
    else:
        assert stream_settings == {}
    handoff = state.module._handoff["value"]
    state.sim_launcher.assert_called_once_with(
        state.app,
        _LAUNCH_PATH.parent / "assets" / "openarm_bimanual_v2.usd",
        state.module._ready,
        state.module._stop,
        handoff.io,
        handoff.scene_actions,
        17,
        cameras_enabled,
        frame_rate_hz=60,
        render_mode="RealTimePathTracing",
        anti_aliasing=3,
    )
    state.thread.assert_called_once_with(target=state.module._run_node_builder, daemon=True)
    state.thread.return_value.start.assert_called_once_with()
    assert state.trace == [
        "node thread", "import isaacsim", "app", "import launcher", "run",
    ]


def test_render_profile_is_real_time_2_with_dlss(startup):
    # RTX Real-Time 2.0 denoises only through DLSS Ray Reconstruction, which
    # runs on the NGX core library the base image carries; the launcher checks
    # after its warmup that Kit kept the profile.
    config = startup().module._RENDER_CONFIG
    assert config["renderer"] == "RealTimePathTracing"
    assert config["anti_aliasing"] == 3


def test_blank_public_ip_leaves_ice_address_selection_automatic(startup, monkeypatch):
    monkeypatch.setenv("PEPPY_ISAAC_PUBLIC_IP", "   ")
    state = startup()
    state.module.main()

    assert not any(arg.startswith(_STREAM_PREFIX + "publicIp=") for arg in state.argv)


@pytest.mark.parametrize("hardware_version", ["v1", "v2"])
def test_robot_asset_root_override_is_preserved(startup, monkeypatch, hardware_version):
    monkeypatch.setenv("PEPPY_ROBOT_ASSETS_DIR", "/opt/robot_assets/openarm/isaac")
    state = startup(hardware_version=hardware_version)
    state.module.main()

    assert state.sim_launcher.call_args.args[1] == Path(
        f"/opt/robot_assets/openarm/isaac/openarm_bimanual_{hardware_version}.usd"
    )


def test_setup_failure_does_not_construct_isaac(startup):
    state = startup()
    failure = ValueError("invalid parameters")
    state.module._setup_error["value"] = failure
    with pytest.raises(RuntimeError, match="node setup failed") as raised:
        state.module.main()

    assert raised.value.__cause__ is failure
    state.simulation_app.assert_not_called()
    state.sim_launcher.assert_not_called()


@pytest.fixture
def kit():
    return tomllib.loads(_KIT_PATH.read_text())


def test_kit_has_physics_rendering_and_viewport_dependencies_without_full_experience(kit):
    deps = kit["dependencies"]
    assert {
        "omni.kit.loop-isaac", "omni.isaac.ml_archive", "isaacsim.core.prims",
        "isaacsim.storage.native", "isaacsim.core.simulation_manager",
        "omni.physics.physx", "omni.physics.stageupdate", "omni.physx.tensors",
        "omni.usd", "omni.hydra.rtx", "omni.hydra.rtx.shadercache.vulkan",
        "omni.gpu_foundation.shadercache.vulkan", "omni.kit.renderer.core",
        "omni.kit.mainwindow", "omni.kit.viewport.window", "omni.kit.viewport.utility",
        "omni.kit.manipulator.camera", "omni.kit.manipulator.selection", "omni.kit.manipulator.prim",
    } <= deps.keys()
    assert not any(name.startswith(("isaacsim.exp.", "omni.isaac.sim.")) for name in deps)
    assert "isaacsim.core.api" not in deps
    assert "omni.kit.livestream.app" not in deps
    assert "omni.replicator.core" not in deps
    # The storage extension supplies its asset root, not the robot's USD directory.
    assert "isaacsim.storage.native" not in kit["settings"]["exts"]
    assert "isaac" not in kit["settings"]["persistent"]


def test_kit_camera_defaults_use_meter_scale_distances(kit):
    app_defaults = kit["settings"]["persistent"]["app"]
    assert app_defaults["viewport"] == {
        "camMoveVelocity": 0.05,
        "camVelocityMin": 0.0001,
        "camVelocityMax": 0.2,
    }
    assert app_defaults["primCreation"]["typedDefaults"]["camera"]["clippingRange"] == [
        0.01, 10000000.0,
    ]


def test_kit_uses_installed_extension_roots_and_disables_persistence_and_registry(kit):
    app = kit["settings"]["app"]
    assert app["exts"]["folders"]["++"] == [
        "/isaac-sim/apps", "/isaac-sim/exts", "/isaac-sim/extscache", "/isaac-sim/extsDeprecated",
    ]
    assert app["settings"]["persistent"] is False
    assert app["extensions"]["registryEnabled"] is False


def test_kit_leaves_main_pacing_to_python_and_keeps_synchronous_fixed_steps(kit):
    app = kit["settings"]["app"]
    assert app["runLoops"]["main"]["manualModeEnabled"] is True
    assert app["runLoops"]["main"]["rateLimitEnabled"] is False
    assert app["runLoopsGlobal"]["syncToPresent"] is False
    assert app["player"]["useFixedTimeStepping"] is True
    assert app["asyncRendering"] is False
    assert app["asyncRenderingLowLatency"] is False
    assert app["gatherRenderResults"] is True
    for name in ("rendering_0", "rendering_1"):
        assert app["runLoops"][name] == {
            "rateLimitEnabled": True, "rateLimitFrequency": 120, "syncToPresent": True,
        }
    assert app["runLoops"]["present"] == {"rateLimitEnabled": True, "rateLimitFrequency": 60}
    assert kit["settings"]["exts"]["omni.kit.renderer.core"]["present"] == {
        "enabled": True, "presentAfterRendering": True,
    }


def test_kit_drives_sensor_annotator_frame_gates_from_the_timeline(kit):
    settings = kit["settings"]
    assert settings["omni"]["replicator"]["asyncRendering"] is False
    assert settings["persistent"]["omni"]["replicator"]["captureOnPlay"] is True


def test_kit_leaves_renderer_and_ngx_to_the_launch_config(kit):
    # SimulationApp's launch config selects the renderer and anti-aliasing;
    # the experience neither overrides them nor touches NGX initialization.
    settings = kit["settings"]
    assert "rtx" not in settings
    assert "ngx" not in settings
    assert "rtx" not in settings["persistent"]


def test_kit_disables_dlss_frame_generation(kit):
    assert kit["settings"]["rtx-transient"]["dlssg"]["enabled"] is False
