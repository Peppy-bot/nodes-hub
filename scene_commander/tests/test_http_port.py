"""The commander takes a port it can serve on, and gives it back when it stops."""

import asyncio
import errno
import logging
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import pytest
from aiohttp import web

from scene_commander import listen


COPIES = 4

# A barrier that never releases must fail the test, not hang the run.
BARRIER_TIMEOUT_S = 10

# An address of TEST-NET-1 (RFC 5737), which no host holds, so binding it fails
# for a reason no other port can fix.
UNBINDABLE = ("192.0.2.1", 34567)


@pytest.fixture
def holder():
    """A listener on a free port, standing in for whatever else on the host
    holds the port a launcher asked for."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", listen.ANY_PORT))
    listener.listen(1)

    yield listener

    listener.close()


def port_of(listener):
    return listener.getsockname()[1]


def test_the_preferred_port_is_the_one_taken(holder):
    # Free a port this process just held, so the preference under test is a
    # port this host allowed a moment ago.
    preferred = port_of(holder)
    holder.close()

    listener = listen.bind_listener("127.0.0.1", preferred)

    try:
        assert port_of(listener) == preferred

    finally:
        listener.close()


def test_a_held_port_moves_the_commander_and_leaves_the_holder_serving(holder, caplog):
    caplog.set_level(logging.WARNING)
    listener = listen.bind_listener("127.0.0.1", port_of(holder))

    assert port_of(listener) != port_of(holder), (
        "the commander must not claim the port another process holds"
    )
    assert listener.getsockname()[0] == "127.0.0.1", "only the port moves"
    assert f"127.0.0.1:{port_of(holder)} is already in use" in caplog.text, (
        "the operator must be told the configured port was taken"
    )

    # The panel really answers where it moved to.
    assert _serve_and_get(listener) == "scene"

    # The holder is untouched: it still accepts on the port it owns.
    with socket.create_connection(holder.getsockname()) as client:
        accepted, _ = holder.accept()
        accepted.close()
        assert client.getpeername() == holder.getsockname()


def test_two_copies_racing_for_one_port_both_survive():
    # Both copies reach the bind together, as two instances of one launcher do.
    # Listening settles the race: Linux lets a second SO_REUSEADDR socket bind
    # a port no socket has listened on, so the loser's bind raises EADDRINUSE
    # and it falls back.
    preferred = _free_port()
    listeners = _bind_together(COPIES, preferred)

    try:
        ports = {port_of(listener) for listener in listeners}
        assert len(ports) == COPIES, f"every copy needs its own socket: {ports}"
        assert preferred in ports, (
            f"one copy must win the port they all prefer: {ports}"
        )

    finally:
        for listener in listeners:
            listener.close()


def test_concurrent_copies_each_bind_their_own_socket(holder):
    preferred = port_of(holder)
    # Every copy prefers the held port, as copies of one robot launcher do.
    listeners = _bind_together(COPIES, preferred)

    try:
        ports = {port_of(listener) for listener in listeners}
        assert len(ports) == COPIES, f"every copy needs its own socket: {ports}"
        assert preferred not in ports, "no copy may take the held port"

        # Each copy serves its own interface on the socket it took.
        for listener in listeners:
            served = port_of(listener)
            assert _serve_and_get(listener) == "scene", (
                f"the copy on port {served} must serve its own panel"
            )

    finally:
        for listener in listeners:
            listener.close()


def test_an_address_this_host_does_not_hold_is_raised():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        probe.bind(UNBINDABLE)
        pytest.skip(f"this host permits binding {UNBINDABLE[0]} (ip_nonlocal_bind)")

    except OSError:
        pass

    finally:
        probe.close()

    with pytest.raises(OSError, match="http_host and http_port") as refused:
        listen.bind_listener(*UNBINDABLE)

    assert refused.value.errno == errno.EADDRNOTAVAIL


def test_a_failure_another_port_would_fix_is_still_raised(monkeypatch):
    # A privileged port is the everyday case: the operator has a launch
    # parameter to fix, and the refusal is what tells them so.
    attempts = []

    def only_the_preferred_port_fails(address, port):
        attempts.append(port)
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(listen, "_bind", only_the_preferred_port_fails)

    with pytest.raises(OSError) as refused:
        listen.bind_listener("127.0.0.1", 80)

    assert refused.value.errno == errno.EACCES
    assert attempts == [80], "a refusal must not be retried on another port"


def test_a_fallback_that_cannot_bind_names_the_conflict(monkeypatch):
    def every_port_fails(address, port):
        raise OSError(
            errno.EADDRINUSE if port != listen.ANY_PORT else errno.EMFILE,
            "Too many open files",
        )

    monkeypatch.setattr(listen, "_bind", every_port_fails)

    with pytest.raises(OSError, match="was already in use.*set http_port"):
        listen.bind_listener("127.0.0.1", 8766)


def test_a_host_that_is_not_an_ip_is_refused_by_name():
    # Only a literal IP names one socket to bind.
    with pytest.raises(ValueError, match="http_host"):
        listen.bind_listener("localhost", 8766)


def test_port_zero_is_refused_by_name():
    # Port 0 asks for an operating-system port on every launch, so no operator
    # could be sent to the panel. The node takes one only when the launcher's
    # port is already held, and logs where it landed.
    with pytest.raises(ValueError, match="http_port"):
        listen.bind_listener("127.0.0.1", listen.ANY_PORT)


def test_served_url_reads_the_bound_socket(holder):
    assert listen.served_url(holder) == f"http://127.0.0.1:{port_of(holder)}"


def test_served_url_brackets_an_ipv6_host():
    probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)

    try:
        probe.bind(("::1", listen.ANY_PORT))

    except OSError as error:
        pytest.skip(f"this host has no IPv6 loopback: {error}")

    finally:
        probe.close()

    listener = listen.bind_listener("::1", _free_port())

    try:
        assert listen.served_url(listener) == f"http://[::1]:{port_of(listener)}"

    finally:
        listener.close()


def test_the_commander_serves_on_the_address_it_reports_and_releases_it():
    preferred = _free_port()
    listener = listen.bind_listener("127.0.0.1", preferred)
    url = listen.served_url(listener)

    async def serve_then_stop():
        server = await listen.start_serving(_page(), listener)

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                assert response.status == 200
                assert await response.text() == "scene"

        server.cancel()
        await asyncio.gather(server, return_exceptions=True)

    asyncio.run(serve_then_stop())

    # Stopping releases the socket, so a copy that rejoins can take its port
    # back.
    rebound = listen.bind_listener("127.0.0.1", preferred)

    try:
        assert listen.served_url(rebound) == url

    finally:
        rebound.close()


def test_a_commander_stopped_at_once_still_releases_its_port():
    preferred = _free_port()

    async def serve_then_stop():
        listener = listen.bind_listener("127.0.0.1", preferred)
        server = await listen.start_serving(_page(), listener)
        # A launch that fails right after this node comes up cancels it before
        # it has served anything.
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)

    asyncio.run(serve_then_stop())

    rebound = listen.bind_listener("127.0.0.1", preferred)

    try:
        assert port_of(rebound) == preferred

    finally:
        rebound.close()


def test_a_start_cancelled_while_the_site_owns_the_socket_releases_it(monkeypatch):
    preferred = _free_port()
    reached = None

    async def cancel_while_starting():
        nonlocal reached
        reached = asyncio.Event()
        original = listen._hold_until_cancelled

        async def hold_without_signalling(runner, holding):
            # The site owns the socket by now. Leaving `holding` unset keeps
            # start_serving at its wait, which is the window a node-startup
            # timeout lands in.
            reached.set()
            await original(runner, asyncio.Event())

        monkeypatch.setattr(listen, "_hold_until_cancelled", hold_without_signalling)

        # Cleanup runs to its end only if the socket it closes is still open,
        # so the last receiver firing is the signal the release went through
        # the runner rather than around it.
        app = _page()

        async def cleaned(_app):
            _done.set()

        app.on_cleanup.append(cleaned)

        listener = listen.bind_listener("127.0.0.1", preferred)
        starting = asyncio.create_task(listen.start_serving(app, listener))
        await reached.wait()
        starting.cancel()

        outcome = await asyncio.gather(starting, return_exceptions=True)

        # Taken before the loop closes, so it is start_serving that released
        # the port and not the teardown of the run.
        return outcome, _rebindable(preferred)

    _done = threading.Event()
    ([outcome], rebound_in_loop) = asyncio.run(cancel_while_starting())

    assert isinstance(outcome, asyncio.CancelledError), (
        f"a cancelled start must not fail another way: {outcome!r}"
    )
    assert _done.is_set(), "the runner must finish its cleanup, not fail partway"
    assert rebound_in_loop, "the port must be free once the cancelled start returns"


def test_a_site_that_cannot_serve_gives_the_port_back(monkeypatch):
    # The site never takes the socket, so releasing it and the runner falls to
    # start_serving itself.
    async def refuse(_self):
        raise OSError(errno.EADDRNOTAVAIL, "cannot serve")

    monkeypatch.setattr(web.SockSite, "start", refuse)

    listener = listen.bind_listener("127.0.0.1", _free_port())
    preferred = port_of(listener)
    done = threading.Event()

    async def start_a_site_that_refuses():
        app = _page()

        async def cleaned(_app):
            done.set()

        app.on_cleanup.append(cleaned)

        with pytest.raises(OSError):
            await listen.start_serving(app, listener)

        return _rebindable(preferred)

    assert asyncio.run(start_a_site_that_refuses()), "a failed start must free the port"
    assert done.is_set(), "a failed start must clean up the runner it set up"


def test_served_url_names_localhost_for_a_panel_on_every_interface():
    listener = listen.bind_listener("0.0.0.0", _free_port())

    try:
        assert listen.served_url(listener) == f"http://localhost:{port_of(listener)}"

    finally:
        listener.close()


def _rebindable(port):
    """Whether `port` itself can be taken again, which is what release means.

    A held port sends `bind_listener` to another one, so the port it returns
    is the answer.
    """
    listener = listen.bind_listener("127.0.0.1", port)

    try:
        return port_of(listener) == port

    finally:
        listener.close()


def _bind_together(count, preferred):
    """Bind `count` copies that all reach the bind at the same moment."""
    together = threading.Barrier(count)

    def copy():
        together.wait(timeout=BARRIER_TIMEOUT_S)
        return listen.bind_listener("127.0.0.1", preferred)

    with ThreadPoolExecutor(max_workers=count) as pool:
        return [future.result() for future in [pool.submit(copy) for _ in range(count)]]


def _serve_and_get(listener):
    """Serve a page on the bound socket and read it back, closing the socket."""

    async def serve_then_stop():
        url = listen.served_url(listener)
        server = await listen.start_serving(_page(), listener)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    assert response.status == 200
                    return await response.text()

        finally:
            server.cancel()
            await asyncio.gather(server, return_exceptions=True)

    return asyncio.run(serve_then_stop())


def _page():
    async def index(_request):
        return web.Response(text="scene")

    app = web.Application()
    app.router.add_get("/", index)

    return app


def _free_port():
    """A port that was bindable a moment ago, released before it is preferred."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", listen.ANY_PORT))

        return listener.getsockname()[1]
