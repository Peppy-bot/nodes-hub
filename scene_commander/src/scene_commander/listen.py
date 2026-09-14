"""The scene panel's HTTP listener: taking a port and serving on it.

The socket is bound before the node reports ready and stays owned until the
server stops, so the address the operator is told is the address the panel is
answering on.
"""

from __future__ import annotations

import asyncio
import errno
import ipaddress
import logging
import socket

from aiohttp import web

logger = logging.getLogger(__name__)

# Binding this port has the operating system choose a free one, which is what
# the scene panel falls back to when the launcher's port is already taken.
ANY_PORT = 0

# Connections the kernel queues before the server accepts them, matching the
# default aiohttp's own site passes to `create_server`.
BACKLOG = 128


def bind_listener(host: str, preferred_port: int) -> socket.socket:
    """Own the scene panel's listening socket.

    Takes `preferred_port`, or a port the operating system picks on the same
    host address when another process already holds it. Only a port conflict
    falls back: a mistyped host, an unservable address, or any other bind
    failure is raised, so it reaches the operator as a refusal naming what to
    fix.
    """

    address = _parse_host(host)

    if preferred_port == ANY_PORT:
        raise ValueError("parameter http_port must name a port to serve on, not 0")

    try:
        return _bind(address, preferred_port)

    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise _unservable(host, preferred_port, error) from error

    taken = _authority(host, preferred_port)
    logger.warning(
        "%s is already in use; taking a port from the operating system", taken
    )

    try:
        return _bind(address, ANY_PORT)

    except OSError as error:
        raise OSError(
            error.errno,
            f"bind the scene panel to a port chosen by the operating system, "
            f"after {taken} was already in use: {error.strerror}. "
            "Free ports on this host, or set http_port to one this machine can bind.",
        ) from error


def served_url(listener: socket.socket) -> str:
    """The address an operator opens to reach the scene panel on this socket.

    A panel bound to every interface answers on loopback too, which is the
    name that works in a browser on every platform.
    """

    host, port = listener.getsockname()[:2]

    if ipaddress.ip_address(host).is_unspecified:
        return f"http://localhost:{port}"

    return f"http://{_authority(host, port)}"


def bound_address(listener: socket.socket) -> str:
    """The address the socket is bound to, which a wildcard keeps as it is."""

    host, port = listener.getsockname()[:2]

    return _authority(host, port)


async def start_serving(
    app: web.Application,
    listener: socket.socket,
) -> asyncio.Task[None]:
    """Serve `app` on the bound socket.

    Returns once the server is accepting, so a node is reported ready only
    when its panel answers. The returned task holds the socket until it is
    cancelled, and releasing it is what lets a copy that rejoins the stack
    take its port back.
    """
    # Browser requests stay out of the node log: what the log records is the
    # Peppy traffic, one line per goal and per catalogue state change.
    runner = web.AppRunner(app, access_log=None)
    holder = None

    try:
        await runner.setup()
        await web.SockSite(runner, listener).start()

        holding = asyncio.Event()
        holder = asyncio.create_task(_hold_until_cancelled(runner, holding))
        # The task releases the socket from its own cleanup, which it reaches
        # only once it has started running. Waiting for that here is what puts
        # the release under the returned task from the moment it is returned.
        await holding.wait()

        return holder

    except BaseException:
        if holder is None:
            # The site never took the socket, so releasing it falls to here.
            listener.close()
            await runner.cleanup()

        else:
            # The site owns the socket and closes it from the runner's
            # cleanup, which the task reaches when it is cancelled.
            holder.cancel()
            await asyncio.gather(holder, return_exceptions=True)

        raise


async def _hold_until_cancelled(runner: web.AppRunner, holding: asyncio.Event) -> None:
    holding.set()

    try:
        await asyncio.Event().wait()

    finally:
        await runner.cleanup()


def _parse_host(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        return ipaddress.ip_address(host)

    except ValueError as error:
        raise ValueError(
            f"parameter http_host must be an IP address, not {host!r}"
        ) from error


def _bind(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address, port: int
) -> socket.socket:
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)

    try:
        # Rebinding the port after a restart succeeds while the last
        # connections sit in TIME_WAIT.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((str(address), port))
        # Linux lets a second SO_REUSEADDR socket bind a port while no socket
        # on it has listened, so listening here is what makes the port this
        # process's own and what makes a rival's bind raise EADDRINUSE.
        listener.listen(BACKLOG)

    except OSError:
        listener.close()
        raise

    return listener


def _unservable(host: str, port: int, error: OSError) -> OSError:
    return OSError(
        error.errno,
        f"serve the scene panel on {_authority(host, port)}: {error.strerror}. "
        f"Set http_host and http_port to an address this machine can serve.",
    )


def _authority(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
