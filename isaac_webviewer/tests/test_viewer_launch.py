"""Test the Peppy wrapper with a fake server and thread, without importing Peppy."""

import asyncio
import errno
import importlib.util
import logging
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

_VIEWER_DIR = Path(__file__).resolve().parents[1]
# The port a launcher prefers for the viewer in these tests, other than the
# manifest's default, so a viewer that ignores its parameter fails them.
PREFERRED_PORT = 8311
# The port the fake operating system hands a bind to port 0.
OS_CHOSEN_PORT = 49731


@pytest.fixture
def startup(monkeypatch):
    state = SimpleNamespace(
        events=[], bind_errors=[], thread_error=None,
        start_error=None, registration_error=None,
        params=SimpleNamespace(http_port=PREFERRED_PORT),
    )

    def record(event):
        state.events.append((event, threading.get_ident()))

    def bind(address):
        record("bind")
        if state.bind_errors:
            raise state.bind_errors.pop(0)
        host, port = address
        state.server.server_address = (host, port or OS_CHOSEN_PORT)
        return state.server

    def make_thread(**kwargs):
        record("thread")
        if state.thread_error is not None:
            raise state.thread_error
        return state.thread

    def start():
        record("start")
        if state.start_error is not None:
            raise state.start_error

    def register(callback):
        record("register")
        if state.registration_error is not None:
            raise state.registration_error

    def announce(label, scheme, host, port):
        record("announce")
        if state.announce_error is not None:
            raise state.announce_error

    state.announce_error = None
    state.server = Mock(spec=["serve_forever", "shutdown", "server_close", "server_address"])
    state.server.server_address = None
    state.server.shutdown.side_effect = lambda: record("shutdown")
    state.server.server_close.side_effect = lambda: record("close")
    state.server_factory = Mock(side_effect=bind)
    state.thread = Mock(spec=["start", "join"])
    state.thread.start.side_effect = start
    state.thread.join.side_effect = lambda: record("join")
    state.thread_factory = Mock(side_effect=make_thread)
    state.runner = Mock(spec=["on_shutdown", "announce_endpoint"])
    state.runner.on_shutdown.side_effect = register
    state.runner.announce_endpoint.side_effect = announce

    runtime = ModuleType("peppylib.runtime")
    state.node_builder = runtime.NodeBuilder = Mock()
    monkeypatch.setitem(sys.modules, "peppylib", ModuleType("peppylib"))
    monkeypatch.setitem(sys.modules, "peppylib.runtime", runtime)
    server_module = ModuleType("viewer_server")
    server_module.ViewerServer = state.server_factory
    monkeypatch.setitem(sys.modules, "viewer_server", server_module)
    state.log_config = Mock()
    monkeypatch.setattr(logging, "basicConfig", state.log_config)

    spec = importlib.util.spec_from_file_location(
        "_isaac_viewer_launch_under_test", _VIEWER_DIR / "launch.py",
    )
    state.module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(state.module)
    # Replace only the module's references, not the threading/asyncio modules used
    # by the real executor that runs shutdown.
    monkeypatch.setattr(state.module, "threading", SimpleNamespace(Thread=state.thread_factory))
    state.to_thread = AsyncMock(wraps=asyncio.to_thread)
    monkeypatch.setattr(state.module, "asyncio", SimpleNamespace(to_thread=state.to_thread))
    return state


def _assert_cleanup_off_loop(state, loop_thread):
    assert [event for event, _ in state.events][-3:] == ["shutdown", "close", "join"]
    cleanup_threads = {thread for _, thread in state.events[-3:]}
    assert len(cleanup_threads) == 1
    assert loop_thread not in cleanup_threads
    state.to_thread.assert_awaited_once()
    state.server.shutdown.assert_called_once_with()
    state.server.server_close.assert_called_once_with()
    state.thread.join.assert_called_once_with()


def test_setup_registers_shutdown_after_start_and_cleans_up_off_event_loop(startup, caplog):
    caplog.set_level(logging.INFO, logger=startup.module.logger.name)
    loop_thread = threading.get_ident()

    async def run():
        assert await startup.module.setup(startup.params, startup.runner) == []
        assert [event for event, _ in startup.events] == [
            "bind", "thread", "start", "register", "announce",
        ]
        startup.server.shutdown.assert_not_called()
        startup.server.server_close.assert_not_called()
        startup.thread.join.assert_not_called()
        startup.runner.on_shutdown.assert_called_once()
        # The viewer is announced with the address the server bound, after
        # the server is up, so the daemon reports a socket that answers.
        startup.runner.announce_endpoint.assert_called_once_with(
            "viewer", "http", "0.0.0.0", PREFERRED_PORT,
        )
        shutdown = startup.runner.on_shutdown.call_args.args[0]
        await shutdown()

    asyncio.run(run())
    startup.server_factory.assert_called_once_with(("0.0.0.0", PREFERRED_PORT))
    startup.thread_factory.assert_called_once_with(
        target=startup.server.serve_forever, name="isaac-viewer", daemon=True,
    )
    startup.thread.start.assert_called_once_with()
    assert [event for event, _ in startup.events] == [
        "bind", "thread", "start", "register", "announce", "shutdown", "close", "join",
    ]
    _assert_cleanup_off_loop(startup, loop_thread)
    messages = [record.getMessage() for record in caplog.records]
    assert any(f"listening on 0.0.0.0:{PREFERRED_PORT}" in message for message in messages)
    assert any("Stopping Isaac Sim browser viewer" in message for message in messages)


def test_a_taken_port_is_not_a_failure_but_a_port_of_its_own(startup, caplog):
    """Two viewers run on one host: the configured port is a preference, and a
    copy whose port is held takes one the operating system picks."""
    caplog.set_level(logging.INFO, logger=startup.module.logger.name)
    startup.bind_errors = [OSError(errno.EADDRINUSE, "port occupied")]

    async def run():
        assert await startup.module.setup(startup.params, startup.runner) == []

    asyncio.run(run())

    assert startup.server_factory.call_args_list == [
        call(("0.0.0.0", PREFERRED_PORT)),
        call(("0.0.0.0", 0)),
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any(f"{PREFERRED_PORT} is already taken" in message for message in messages)
    startup.thread.start.assert_called_once_with()
    # The daemon is told the port the viewer holds, the one the operating
    # system chose.
    startup.runner.announce_endpoint.assert_called_once_with(
        "viewer", "http", "0.0.0.0", OS_CHOSEN_PORT,
    )
    assert any(f"listening on 0.0.0.0:{OS_CHOSEN_PORT}" in message for message in messages)


def test_a_bind_failure_that_is_not_a_taken_port_reaches_the_operator(startup, caplog):
    caplog.set_level(logging.INFO, logger=startup.module.logger.name)
    refusal = OSError(errno.EADDRNOTAVAIL, "address unavailable")
    startup.bind_errors = [refusal]

    with pytest.raises(OSError) as raised:
        asyncio.run(startup.module.setup(startup.params, startup.runner))

    assert raised.value is refusal
    assert [event for event, _ in startup.events] == ["bind"]
    startup.thread_factory.assert_not_called()
    startup.server.shutdown.assert_not_called()
    startup.server.server_close.assert_not_called()
    startup.runner.on_shutdown.assert_not_called()
    startup.to_thread.assert_not_called()
    assert caplog.records == []


@pytest.mark.parametrize("failure", ["thread_error", "start_error"], ids=["construction", "start"])
def test_thread_failure_closes_listener_without_shutting_down_unstarted_server(startup, caplog, failure):
    caplog.set_level(logging.INFO, logger=startup.module.logger.name)
    error = RuntimeError("thread unavailable")
    setattr(startup, failure, error)

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(startup.module.setup(startup.params, startup.runner))

    assert raised.value is error
    expected = ["bind", "thread"]
    if failure == "start_error":
        expected.append("start")
    assert [event for event, _ in startup.events] == expected + ["close"]
    startup.server.server_close.assert_called_once_with()
    startup.server.shutdown.assert_not_called()
    startup.thread.join.assert_not_called()
    startup.runner.on_shutdown.assert_not_called()
    startup.to_thread.assert_not_called()
    assert caplog.records == []


def test_registration_failure_awaits_off_loop_cleanup_before_propagating(startup, caplog):
    caplog.set_level(logging.INFO, logger=startup.module.logger.name)
    loop_thread = threading.get_ident()
    startup.registration_error = RuntimeError("shutdown callback rejected")

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(startup.module.setup(startup.params, startup.runner))

    assert raised.value is startup.registration_error
    assert [event for event, _ in startup.events] == [
        "bind", "thread", "start", "register", "shutdown", "close", "join",
    ]
    _assert_cleanup_off_loop(startup, loop_thread)
    assert not any("listening" in record.getMessage() for record in caplog.records)


def test_a_refused_announcement_awaits_off_loop_cleanup_before_propagating(startup, caplog):
    caplog.set_level(logging.INFO, logger=startup.module.logger.name)
    loop_thread = threading.get_ident()
    startup.announce_error = ValueError("endpoint `viewer` is not declared")

    with pytest.raises(ValueError) as raised:
        asyncio.run(startup.module.setup(startup.params, startup.runner))

    assert raised.value is startup.announce_error
    assert [event for event, _ in startup.events] == [
        "bind", "thread", "start", "register", "announce", "shutdown", "close", "join",
    ]
    _assert_cleanup_off_loop(startup, loop_thread)
    assert not any("listening" in record.getMessage() for record in caplog.records)


def test_main_passes_setup_to_node_builder_and_configures_stream_logging(startup):
    startup.node_builder.assert_not_called()
    startup.server_factory.assert_not_called()
    startup.module.main()

    startup.node_builder.assert_called_once_with()
    startup.node_builder.return_value.run.assert_called_once_with(startup.module.setup)
    startup.server_factory.assert_not_called()
    startup.log_config.assert_called_once_with(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True,
    )


def test_container_packages_server_and_browser_diagnostics_beside_launch():
    recipe = (_VIEWER_DIR / "apptainer.def").read_text()
    files_section = recipe.split("\n%files\n", 1)[1].split("\n%", 1)[0]
    packaged_files = {tuple(line.split()) for line in files_section.splitlines() if line.strip()}
    for filename in ("launch.py", "viewer_server.py", "browser_logs.js"):
        assert (filename, f"/opt/isaac_webviewer/{filename}") in packaged_files
        assert (_VIEWER_DIR / filename).is_file()
    runscript = recipe.split("\n%runscript\n", 1)[1].split("\n%", 1)[0]
    assert 'exec python3 /opt/isaac_webviewer/launch.py "$@"' in runscript
