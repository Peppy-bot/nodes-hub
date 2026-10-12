"""Exercise startup through fake runtimes and parse the packaged Kit experience."""

import asyncio
import builtins
import ctypes
import importlib.util
import json
import logging
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, Mock

import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
_LAUNCH_PATH = _ENGINE_DIR / "launch.py"
_KIT_PATH = _ENGINE_DIR / "config" / "sim_isaac.kit"
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

    def nvml_init():
        state.trace.append("nvml init")
        return state.nvml.nvmlInit_v2.return_value

    def nvml_shutdown():
        state.trace.append("nvml shutdown")
        return state.nvml.nvmlShutdown.return_value

    state.nvml = SimpleNamespace(
        nvmlInit_v2=Mock(side_effect=nvml_init, return_value=0),
        nvmlErrorString=Mock(return_value=b"Unknown Error"),
        nvmlShutdown=Mock(side_effect=nvml_shutdown, return_value=0),
    )
    state.cdll = Mock(return_value=state.nvml)
    monkeypatch.setattr(ctypes, "CDLL", state.cdll)

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

    def load(*, headless=True, cameras=False):
        spec = importlib.util.spec_from_file_location("_launch_under_test", _LAUNCH_PATH)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        assert state.trace == [], "importing launch must not initialize NVML or Isaac"
        state.cdll.assert_not_called()
        module._handoff["value"] = module._SimHandoff(
            io=object(),
            scene_actions=object(),
            robots=object(),
            robots_io=Mock(),
            world=Mock(),
            edits=Mock(),
            state_rate_hz=17,
            headless=headless,
            renders=cameras,
        )
        module._handoff_ready.set()
        state.thread = Mock()
        state.thread.return_value.start.side_effect = lambda: state.trace.append("node thread")
        monkeypatch.setattr(module.threading, "Thread", state.thread)
        state.module = module
        return state

    return load


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("cameras", [False, True])
@pytest.mark.parametrize("overrides", [False, True])
def test_launch_selects_extensions_before_construction_and_preserves_handoff(
    startup, monkeypatch, headless, cameras, overrides,
):
    if overrides:
        monkeypatch.setenv("PEPPY_ISAAC_PUBLIC_IP", " 192.0.2.5 ")
        monkeypatch.setenv("PEPPY_ISAAC_SIGNAL_PORT", " 49200 ")
        monkeypatch.setenv("PEPPY_ISAAC_STREAM_PORT", " 48098 ")
    state = startup(headless=headless, cameras=cameras)
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
        + (["omni.replicator.core"] if cameras else [])
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
        handoff.world,
        handoff.edits,
        # The bridge the node builds around the world; which object it is
        # belongs to the bridge's own tests.
        ANY,
        state.module._ready,
        state.module._stop,
        handoff.io,
        handoff.scene_actions,
        frame_rate_hz=60,
        render_mode="RealTimePathTracing",
        anti_aliasing=3,
        boot=state.module._boot,
    )
    state.thread.assert_called_once_with(target=state.module._run_node_builder, daemon=True)
    state.thread.return_value.start.assert_called_once_with()
    assert state.trace == [
        "nvml init", "nvml shutdown", "node thread", "import isaacsim", "app",
        "import launcher", "run",
    ]
    state.cdll.assert_called_once_with("libnvidia-ml.so.1")
    for operation in (state.nvml.nvmlInit_v2, state.nvml.nvmlShutdown):
        operation.assert_called_once_with()
        assert operation.argtypes == []
        assert operation.restype is ctypes.c_int
    assert state.nvml.nvmlErrorString.argtypes == [ctypes.c_int]
    assert state.nvml.nvmlErrorString.restype is ctypes.c_char_p
    state.nvml.nvmlErrorString.assert_not_called()


def _assert_no_startup(state):
    state.thread.assert_not_called()
    state.simulation_app.assert_not_called()
    state.sim_launcher.assert_not_called()
    assert "import isaacsim" not in state.trace


@pytest.mark.parametrize(
    ("init_result", "shutdown_result", "detail"),
    [
        (18, 0, b"Driver/library version mismatch"),
        (9, 0, b"Driver Not Loaded"),
        (0, 999, b"Unknown Error"),
        (999, 0, None),
    ],
)
def test_nvml_failure_stops_before_startup(startup, init_result, shutdown_result, detail):
    state = startup()
    state.nvml.nvmlInit_v2.return_value = init_result
    state.nvml.nvmlShutdown.return_value = shutdown_result
    state.nvml.nvmlErrorString.return_value = detail

    with pytest.raises(RuntimeError) as raised:
        state.module.main()

    message = str(raised.value)
    operation = "nvmlInit_v2" if init_result else "nvmlShutdown"
    result = init_result or shutdown_result
    expected_detail = detail.decode("utf-8") if detail else "unknown NVML error"
    assert f"{operation} returned NVML error {result} ({expected_detail})" in message
    assert "nvidia-smi" in message
    assert "host NVIDIA driver" in message
    assert "--nv" in message
    if result == 18:
        assert "user-space library and loaded kernel driver do not match" in message
        assert "Reboot the host after a driver update" in message
    else:
        assert "Reboot" not in message
    state.cdll.assert_called_once_with("libnvidia-ml.so.1")
    state.nvml.nvmlInit_v2.assert_called_once_with()
    state.nvml.nvmlErrorString.assert_called_once_with(result)
    if init_result:
        state.nvml.nvmlShutdown.assert_not_called()
        assert state.trace == ["nvml init"]
    else:
        state.nvml.nvmlShutdown.assert_called_once_with()
        assert state.trace == ["nvml init", "nvml shutdown"]
    _assert_no_startup(state)


def test_missing_nvml_library_stops_before_startup(startup):
    state = startup()
    failure = OSError("libnvidia-ml.so.1: cannot open shared object file")
    state.cdll.side_effect = failure

    with pytest.raises(RuntimeError, match="cannot load libnvidia-ml.so.1") as raised:
        state.module.main()

    assert raised.value.__cause__ is failure
    assert str(failure) in str(raised.value)
    assert "host NVIDIA driver" in str(raised.value)
    assert "--nv" in str(raised.value)
    state.cdll.assert_called_once_with("libnvidia-ml.so.1")
    for operation in vars(state.nvml).values():
        operation.assert_not_called()
    assert state.trace == []
    _assert_no_startup(state)


@pytest.mark.parametrize("symbol", ["nvmlInit_v2", "nvmlErrorString", "nvmlShutdown"])
def test_missing_nvml_api_symbol_stops_before_initialization(startup, symbol):
    state = startup()
    operations = list(vars(state.nvml).values())
    delattr(state.nvml, symbol)

    with pytest.raises(RuntimeError, match="missing required NVML API symbol") as raised:
        state.module.main()

    assert isinstance(raised.value.__cause__, AttributeError)
    assert symbol in str(raised.value)
    assert "host NVIDIA driver" in str(raised.value)
    assert "--nv" in str(raised.value)
    state.cdll.assert_called_once_with("libnvidia-ml.so.1")
    for operation in operations:
        operation.assert_not_called()
    assert state.trace == []
    _assert_no_startup(state)


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


@pytest.mark.parametrize(
    "model, filename",
    [
        ("openarm_v1", "openarm_bimanual.usd"),
        ("openarm_v2", "openarm_bimanual_v2.usd"),
    ],
)
def test_every_openarm_entry_names_a_stage_of_the_bundle_the_image_bakes(model, filename):
    """An entry names its stage under its robot's directory of the baked
    assets, and the image is built to carry exactly those files there."""
    sys.path.insert(0, str(_ENGINE_DIR))
    from isaac_models import IsaacModels

    assert IsaacModels.read().of(model).stage == f"openarm/{filename}"
    manifest = json.loads((_ENGINE_DIR.parent / "scripts/visual_sources.json").read_text())
    assert filename in manifest["robot"]["files"]


def test_setup_reads_every_models_entry_and_the_head_camera_where_the_image_stages_it():
    """Setup parses every model's entry before the stage opens, and reads the
    pack apptainer.def stages beside launch.py for the models that draw it."""
    launch = _LAUNCH_PATH.read_text()
    assert '_HEAD_CAMERA_DIR = Path(__file__).parent / "assets" / "head_camera"' in launch
    assert "models = IsaacModels.read()" in launch
    assert "World(head_camera.load_for(models, _HEAD_CAMERA_DIR))" in launch
    assert "renders=params.cameras_enabled" in launch


def test_the_engines_modules_and_its_experience_sit_beside_the_launch_script():
    launch = _LAUNCH_PATH.read_text()
    assert "_ENGINE_DIR = Path(__file__).resolve().parent\n" in launch
    assert '_EXPERIENCE_PATH = _ENGINE_DIR / "config" / "sim_isaac.kit"' in launch
    assert _KIT_PATH.is_file()
    for module in ("_launcher.py", "bridge_extension.py", "isaac_models.py", "sim_topics.py"):
        assert (_LAUNCH_PATH.parent / module).is_file()


def test_setup_failure_does_not_construct_isaac(startup):
    state = startup()
    failure = ValueError("invalid parameters")
    state.module._setup_error["value"] = failure
    with pytest.raises(RuntimeError, match="node setup failed") as raised:
        state.module.main()

    assert raised.value.__cause__ is failure
    state.simulation_app.assert_not_called()
    state.sim_launcher.assert_not_called()


def _setup_handing_off(state, monkeypatch):
    """The node's setup with its IO stubbed: the handoff ends at once and
    answers the node's tasks, so what the setup waits for after it is the
    boot alone."""
    tasks = [object()]

    async def handed_off(_params, _node_runner):
        return tasks

    monkeypatch.setattr(state.module, "_node_setup", handed_off)
    return tasks


def _run_setup_until_it_waits(module, settle_the_boot):
    """Runs the node's setup until it waits on the boot, settles the boot
    with `settle_the_boot`, and answers what the setup ends with."""

    async def scenario():
        setup = asyncio.ensure_future(module.setup(object(), object()))
        # The stubbed handoff does not suspend, so one step of the setup
        # takes it to the first wait it has.
        await asyncio.sleep(0)
        assert not setup.done(), "the setup waits for Isaac Sim to take robots"
        settle_the_boot(setup)
        return await setup

    return asyncio.run(scenario())


def test_setup_ends_once_isaac_takes_robots(startup, monkeypatch):
    state = startup()
    tasks = _setup_handing_off(state, monkeypatch)

    ended = _run_setup_until_it_waits(state.module, lambda _setup: state.module._boot.takes_robots())

    assert ended is tasks


def test_a_boot_that_fails_ends_the_setup_with_its_error(startup, monkeypatch):
    state = startup()
    _setup_handing_off(state, monkeypatch)
    failure = RuntimeError("Kit could not open the stage")

    with pytest.raises(RuntimeError, match="Isaac Sim did not boot: Kit could not open the stage") as raised:
        _run_setup_until_it_waits(state.module, lambda _setup: state.module._boot.fails(failure))

    assert raised.value.__cause__ is failure


def test_a_stop_during_the_boot_ends_the_setup(startup, monkeypatch):
    state = startup()
    _setup_handing_off(state, monkeypatch)

    # peppylib cancels an async setup when the node is stopped.
    with pytest.raises(asyncio.CancelledError):
        _run_setup_until_it_waits(state.module, lambda setup: setup.cancel())

    # The main thread still finishes its boot, and nothing waits for it.
    state.module._boot.takes_robots()
    state.module._boot.fails(RuntimeError("Isaac Sim closed before it took robots"))


def _boot_outcome(module):
    """What the node's setup reads of the boot: None once Isaac Sim takes
    robots, else the error that ended the boot."""

    async def read():
        try:
            await module._boot.wait()
        except Exception as error:  # pylint: disable=W0718
            return error
        return None

    return asyncio.run(read())


def _record_the_wait_for_the_node(state):
    """Records, each time the main thread waits for the node's thread,
    whether the boot had failed by then."""
    waits = []

    def join(timeout):
        assert timeout == state.module._SETUP_REPORT_TIMEOUT_S
        waits.append(state.module._boot.failed())

    state.thread.return_value.join.side_effect = join
    return waits


def test_isaac_takes_robots_when_the_launcher_says_so(startup):
    state = startup()
    waits = _record_the_wait_for_the_node(state)

    def run():
        state.sim_launcher.call_args.kwargs["boot"].takes_robots()
        state.trace.append("run")

    state.sim_launcher.return_value.run.side_effect = run
    state.module.main()

    # The loop ran and ended, as on a stop, after Isaac Sim took robots: the
    # node's setup had ended, so the process does not wait for it.
    assert _boot_outcome(state.module) is None
    assert waits == []


def test_a_boot_error_ends_the_setup_with_it_before_the_process_ends(startup, monkeypatch):
    state = startup()
    _setup_handing_off(state, monkeypatch)
    waits = _record_the_wait_for_the_node(state)
    failure = RuntimeError("no GPU")
    state.simulation_app.side_effect = failure

    def boot():
        with pytest.raises(RuntimeError, match="no GPU"):
            state.module.main()

    with pytest.raises(RuntimeError, match="Isaac Sim did not boot: no GPU") as raised:
        _run_setup_until_it_waits(state.module, lambda _setup: boot())

    assert raised.value.__cause__ is failure
    # The main thread waited for the node's thread once the boot had failed,
    # so the node reports the setup's error before the process ends.
    assert waits == [True]


def test_a_missing_file_that_the_launcher_reports_ends_the_setup_with_it(startup, monkeypatch):
    state = startup()
    _setup_handing_off(state, monkeypatch)
    waits = _record_the_wait_for_the_node(state)
    missing = FileNotFoundError("the stage's USD is missing")
    # The launcher logs a missing file, reports it to the boot and returns.
    state.sim_launcher.return_value.run.side_effect = (
        lambda: state.sim_launcher.call_args.kwargs["boot"].fails(missing)
    )

    with pytest.raises(
        RuntimeError, match="Isaac Sim did not boot: the stage's USD is missing"
    ) as raised:
        _run_setup_until_it_waits(state.module, lambda _setup: state.module.main())

    assert raised.value.__cause__ is missing
    assert waits == [True]


def test_a_launcher_that_ends_before_it_takes_robots_ends_the_boot(startup):
    state = startup()
    waits = _record_the_wait_for_the_node(state)
    state.module.main()

    outcome = _boot_outcome(state.module)
    assert isinstance(outcome, RuntimeError)
    assert str(outcome) == "Isaac Sim closed before it took robots"
    assert waits == [True]


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
