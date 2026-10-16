import ipaddress
import socket

from xr_commander import addresses

V4 = ipaddress.ip_address("198.51.100.7")
V6 = ipaddress.ip_address("2001:db8::1")


def test_the_page_is_reachable_on_every_address_of_an_unspecified_binds_family(
    monkeypatch,
):
    monkeypatch.setattr(addresses, "host_addresses", lambda: [V4, V6])
    assert addresses.page_urls("0.0.0.0", 4443) == ["https://198.51.100.7:4443"]
    assert addresses.page_urls("::", 4443) == [
        "https://198.51.100.7:4443",
        "https://[2001:db8::1]:4443",
    ]


def test_a_loopback_bind_reaches_no_other_machine(monkeypatch):
    monkeypatch.setattr(addresses, "host_addresses", lambda: [V4])
    assert addresses.page_urls("127.0.0.1", 4443) == []
    assert addresses.page_urls("::1", 4443) == []


def test_a_literal_bind_is_the_page_itself(monkeypatch):
    monkeypatch.setattr(addresses, "host_addresses", lambda: [V4, V6])
    assert addresses.page_urls("10.0.0.5", 8443) == ["https://10.0.0.5:8443"]
    assert addresses.page_urls("2001:db8::9", 4443) == ["https://[2001:db8::9]:4443"]


def test_the_outbound_route_comes_first_then_the_hostnames_others_once_each(
    monkeypatch,
):
    monkeypatch.setattr(addresses, "outbound_ip", lambda: V4)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_: [
            (None, None, None, None, ("127.0.0.1", 0)),
            (None, None, None, None, ("198.51.100.7", 0)),
            (None, None, None, None, ("100.64.0.7", 0)),
            (None, None, None, None, ("2001:db8::1%eth0", 0, 0, 0)),
            (None, None, None, None, ("100.64.0.7", 0)),
        ],
    )
    assert addresses.host_addresses() == [
        V4,
        ipaddress.ip_address("100.64.0.7"),
        V6,
    ]


def test_an_unroutable_host_still_lists_what_its_name_resolves_to(monkeypatch):
    monkeypatch.setattr(addresses, "outbound_ip", lambda: None)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_: [(None, None, None, None, ("198.51.100.7", 0))],
    )
    assert addresses.host_addresses() == [V4]


def test_a_host_that_resolves_nothing_lists_the_route_alone(monkeypatch):
    monkeypatch.setattr(addresses, "outbound_ip", lambda: V4)

    def refuse(*_):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert addresses.host_addresses() == [V4]
