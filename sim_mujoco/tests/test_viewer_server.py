"""The viewer's server takes a port it can serve on while the node sets up,
and holds one scene at a time for as long as the node runs.

viser is faked: the suites render nothing and it is not installed where they
run. The ports are real ones of this host's loopback.
"""

from __future__ import annotations

import errno
import logging
import socket
import sys
import types
from pathlib import Path

import pytest
import viewer_fakes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import viewer_server  # noqa: E402  pylint: disable=C0413
from viewer_server import ViewerServer  # noqa: E402  pylint: disable=C0413

# An address of TEST-NET-1 (RFC 5737), which no host holds, so binding it fails
# for a reason no other port can fix.
UNBINDABLE = ("192.0.2.1", 34567)


@pytest.fixture(name="served")
def served_fixture(monkeypatch):
    """Every server viser was asked for, in order. viser binds the address
    it is given, so each one reports that address back."""
    servers = []

    def serve(host, port):
        servers.append(viewer_fakes.Viser(host, port))
        return servers[-1]

    monkeypatch.setitem(sys.modules, "viser", types.SimpleNamespace(ViserServer=serve))
    return servers


@pytest.fixture(name="holder")
def holder_fixture():
    """A listener on a free port, standing in for whatever else on the host
    holds the port a launcher asked for. It binds the way the server of
    another viewer does, with SO_REUSEADDR."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", viewer_server.ANY_PORT))
    listener.listen(1)

    yield listener

    listener.close()


def _port_of(listener) -> int:
    return listener.getsockname()[1]


class TestThePortTheViewerServesOn:
    def test_the_preferred_port_is_the_one_taken(self, served, holder):
        # Free a port this process just held, so the preference under test
        # is a port this host allowed a moment ago.
        preferred = _port_of(holder)
        holder.close()

        server = ViewerServer.serve("127.0.0.1", preferred)

        assert (server.host, server.port) == ("127.0.0.1", preferred)
        assert len(served) == 1

    def test_a_held_port_moves_the_viewer_to_one_the_operating_system_picks(
        self, served, holder, caplog
    ):
        with caplog.at_level(logging.WARNING):
            server = ViewerServer.serve("127.0.0.1", _port_of(holder))

        assert server.host == "127.0.0.1", "only the port moves"
        assert server.port not in (_port_of(holder), viewer_server.ANY_PORT)
        assert f"127.0.0.1:{_port_of(holder)} is already in use" in caplog.text, (
            "the operator must be told the configured port was taken"
        )
        assert len(served) == 1

    def test_the_address_is_the_one_the_server_reports(self, monkeypatch):
        """viser binds its own socket, so the port it took is the one it
        reports, whatever port it was asked for."""
        monkeypatch.setitem(
            sys.modules,
            "viser",
            types.SimpleNamespace(ViserServer=lambda host, port: viewer_fakes.Viser(host, port + 1)),
        )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as free:
            free.bind(("127.0.0.1", viewer_server.ANY_PORT))
            preferred = _port_of(free)

        server = ViewerServer.serve("127.0.0.1", preferred)

        assert server.port == preferred + 1

    def test_an_address_this_host_does_not_hold_is_refused_before_viser_is_asked(self, served):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(UNBINDABLE)
            except OSError:
                pass
            else:
                pytest.skip(f"this host permits binding {UNBINDABLE[0]} (ip_nonlocal_bind)")

        with pytest.raises(OSError, match="viewer_host and viewer_port") as refused:
            ViewerServer.serve(*UNBINDABLE)

        assert refused.value.errno == errno.EADDRNOTAVAIL
        assert f"{UNBINDABLE[0]}:{UNBINDABLE[1]}" in str(refused.value)
        assert not served

    def test_a_failure_another_port_would_fix_is_still_raised(self, served, monkeypatch):
        # A privileged port is the everyday case: the operator has a launch
        # parameter to fix, and the refusal is what tells them so.
        attempts = []

        def only_the_preferred_port_fails(_address, port):
            attempts.append(port)
            raise PermissionError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(viewer_server, "_bindable_port", only_the_preferred_port_fails)

        with pytest.raises(OSError) as refused:
            ViewerServer.serve("127.0.0.1", 80)

        assert refused.value.errno == errno.EACCES
        assert attempts == [80], "a refusal must not be retried on another port"
        assert not served

    def test_a_fallback_that_cannot_bind_names_the_conflict(self, served, monkeypatch):
        def every_port_fails(_address, port):
            raise OSError(
                errno.EADDRINUSE if port != viewer_server.ANY_PORT else errno.EMFILE,
                "Too many open files",
            )

        monkeypatch.setattr(viewer_server, "_bindable_port", every_port_fails)

        with pytest.raises(OSError, match="was already in use.*set viewer_port"):
            ViewerServer.serve("127.0.0.1", 8080)

        assert not served

    def test_a_host_that_is_not_an_ip_is_refused_by_name(self, served):
        # Only a literal IP names one socket to bind, and the node announces
        # the address it bound as one.
        with pytest.raises(ValueError, match="viewer_host"):
            ViewerServer.serve("localhost", 8080)

        assert not served

    def test_port_zero_is_refused_by_name(self, served):
        # Port 0 asks for an operating-system port on every launch. The
        # viewer takes one only when the launcher's port is already held.
        with pytest.raises(ValueError, match="viewer_port"):
            ViewerServer.serve("127.0.0.1", viewer_server.ANY_PORT)

        assert not served

    def test_an_ipv6_host_is_served_as_the_literal_it_is(self, served):
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("::1", viewer_server.ANY_PORT))
            except OSError as error:
                pytest.skip(f"this host has no IPv6 loopback: {error}")
            preferred = _port_of(probe)

        server = ViewerServer.serve("::1", preferred)

        # The announcement takes an IP literal, which carries no brackets.
        assert (server.host, server.port) == ("::1", preferred)


class TestTheSceneOnTheServer:
    def test_a_wipe_takes_everything_a_scene_left_off_the_server(self):
        viser = viewer_fakes.Viser()
        viser.nodes.append("/fixed_bodies")
        viser.gui.add_button("Reset scene objects")
        viser.scene.on_click()(lambda _event: None)
        viser.on_client_connect(lambda _client: None)
        server = ViewerServer(viser)

        server.wipe()

        assert viser.left_on_it() == []
        assert not viser.stopped, "the server serves the scene drawn next"

    def test_a_stop_stops_the_server(self):
        viser = viewer_fakes.Viser()

        ViewerServer(viser).stop()

        assert viser.stopped
