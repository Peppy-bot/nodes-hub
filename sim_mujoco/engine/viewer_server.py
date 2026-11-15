#!/usr/bin/env python3
"""The viser server that serves the engine's viewer to a browser.

The server is bound once, while the node sets up, and serves for as long as
the node runs, so the address the node announces is the one a browser reaches
whatever scene stands. Each scene is drawn on it and wiped off it when the
scene is composed again, and a browser open on the viewer stays connected
from one scene to the next.
"""

from __future__ import annotations

import errno
import ipaddress
import logging
import socket

logger = logging.getLogger(__name__)

# Binding this port has the operating system choose a free one, which is what
# the viewer falls back to when the port it prefers is already taken.
ANY_PORT = 0

_Address = ipaddress.IPv4Address | ipaddress.IPv6Address


class ViewerServer:
    """The viewer's server, bound, with the scene that is drawn on it."""

    def __init__(self, server) -> None:
        # The viser server itself, which the view of a scene draws on.
        self.viser = server

    @classmethod
    def serve(cls, host: str, preferred_port: int) -> "ViewerServer":
        """Binds the viewer to `host`, on `preferred_port` or on a port the
        operating system picks when another socket already holds it.

        Only a port conflict falls back: a host that is not an IP address, an
        address this machine does not have, a port it may not bind, or any
        other bind failure is raised, so it reaches the operator as a refusal
        naming what to fix.
        """
        import viser  # pylint: disable=C0415

        address = _parse_host(host)
        if preferred_port == ANY_PORT:
            raise ValueError("parameter viewer_port must name a port to serve on, not 0")
        port = _port_to_serve_on(address, preferred_port)
        return cls(viser.ViserServer(host=str(address), port=port))

    @property
    def host(self) -> str:
        """The IP literal the server is bound to."""
        return self.viser.get_host()

    @property
    def port(self) -> int:
        """The port the server is bound to, as the server reports it."""
        return self.viser.get_port()

    def wipe(self) -> None:
        """Takes the scene drawn on the server off it: what it shows, its
        controls, and what it asked to be told of the browsers (their clicks,
        their connecting). The scene drawn next starts on a server that holds
        nothing of this one."""
        self.viser.scene.reset()
        self.viser.scene.remove_click_callback("all")
        self.viser.gui.reset()
        # viser holds the callbacks it runs for a browser that connects in
        # this list and has no call that takes one back. A callback of a
        # scene left here would point the camera of every browser that
        # connects at a scene that is gone.
        self.viser._client_connect_cb.clear()  # pylint: disable=W0212

    def stop(self) -> None:
        """Stops serving and gives the address back."""
        self.viser.stop()


def _parse_host(host: str) -> _Address:
    try:
        return ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError(f"parameter viewer_host must be an IP address, not {host!r}") from error


def _port_to_serve_on(address: _Address, preferred_port: int) -> int:
    """`preferred_port` where this process can bind it at `address`, or a
    port the operating system picks there when another socket holds it.

    viser binds its own socket and takes a port number, not a socket, so the
    port is tried here first: a bind that viser cannot make is refused here,
    naming the address, before viser is asked for it.
    """
    try:
        return _bindable_port(address, preferred_port)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise _unservable(address, preferred_port, error) from error
    taken = _authority(address, preferred_port)
    logger.warning("%s is already in use; the viewer takes a port from the operating system", taken)
    try:
        return _bindable_port(address, ANY_PORT)
    except OSError as error:
        raise OSError(
            error.errno,
            f"bind the viewer to a port chosen by the operating system, after {taken} was "
            f"already in use: {error.strerror}. Free ports on this host, or set viewer_port "
            "to one this machine can bind.",
        ) from error


def _bindable_port(address: _Address, port: int) -> int:
    """The port a socket bound to `address` and `port` takes, which is `port`
    itself unless `port` asks the operating system for one. The socket is
    closed before this returns, so the port is free for the server."""
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        # The option viser's own listener binds with: a port whose last
        # connections sit in TIME_WAIT after a restart binds for both, and a
        # port another socket listens on is refused for both.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((str(address), port))
        return probe.getsockname()[1]


def _unservable(address: _Address, port: int, error: OSError) -> OSError:
    return OSError(
        error.errno,
        f"serve the viewer on {_authority(address, port)}: {error.strerror}. "
        "Set viewer_host and viewer_port to an address this machine can serve.",
    )


def _authority(address: _Address, port: int) -> str:
    return f"[{address}]:{port}" if address.version == 6 else f"{address}:{port}"
