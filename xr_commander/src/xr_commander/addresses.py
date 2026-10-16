"""The addresses this machine answers on: the names of the TLS identity and
the URLs the headset's browser opens the page at.

The server binds the address the launcher names, `0.0.0.0` by default, which
no other machine can dial: the page's URLs carry the machine's own addresses
in its place, the outbound route first, as the daemon prints them under
`Web pages:`.
"""

from __future__ import annotations

import ipaddress
import socket

Address = ipaddress.IPv4Address | ipaddress.IPv6Address

# RFC 5737 TEST-NET; the datagram socket is never written, only routed.
_PROBE_ENDPOINT = ("192.0.2.1", 1)


def outbound_ip() -> Address | None:
    """The outbound-facing address, or None when routing is unknowable."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(_PROBE_ENDPOINT)
            return ipaddress.ip_address(probe.getsockname()[0])
    except OSError:
        return None


def host_addresses() -> list[Address]:
    """Every address this host answers on but loopback: the outbound route
    first, then the hostname's others in address order."""
    routed = outbound_ip()
    named: set[Address] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            named.add(ipaddress.ip_address(info[4][0].split("%")[0]))
    except (OSError, ValueError):
        pass
    named.discard(routed)
    others = sorted((a for a in named if not a.is_loopback), key=str)
    return ([routed] if routed is not None else []) + others


def _host_literal(address: Address) -> str:
    return f"[{address}]" if address.version == 6 else str(address)


def page_urls(bound_host: str, port: int) -> list[str]:
    """The page's URLs as a browser on another machine reaches it, from the
    address the server bound: `0.0.0.0` stands for every IPv4 address the
    machine answers on and `::` for every address (dual stack), a loopback
    address reaches no other machine, and any other address is itself."""
    host = ipaddress.ip_address(bound_host)
    if host.is_loopback:
        return []
    if host.is_unspecified:
        addresses = [
            a
            for a in host_addresses()
            if a.version == host.version or host.version == 6
        ]
    else:
        addresses = [host]
    return [f"https://{_host_literal(a)}:{port}" for a in addresses]
