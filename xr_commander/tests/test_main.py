from xr_commander import __main__ as main
from xr_commander import config
from tests.helpers import default_parameters


def hint(host, routed=None):
    settings = config.from_parameters(default_parameters(https_host=host))
    return main._headset_url_hint(settings, outbound_ip=lambda: routed)


def test_an_unspecified_bind_advertises_the_outbound_address():
    assert "https://192.168.0.5:4443" in hint("0.0.0.0", "192.168.0.5")


def test_a_pinned_bind_advertises_itself():
    # Even with a different outbound route, a pinned bind is the reachable one.
    assert "https://10.0.0.9:4443" in hint("10.0.0.9", "192.168.0.5")


def test_an_ipv6_address_is_bracketed_for_the_url():
    assert "https://[2001:db8::7]:4443" in hint("2001:db8::7")


def test_no_route_still_yields_a_usable_hint():
    text = hint("0.0.0.0", None)
    assert "<this machine>" in text
    assert "adb reverse" in text


def test_hand_wiring_covers_both_hands_with_distinct_modules():
    assert set(main._HANDS) == {"left", "right"}
    modules = [
        module
        for wiring in main._HANDS.values()
        for module in (wiring.pose_setpoints, wiring.pose_states, wiring.gripper_setpoints)
    ]
    assert len(set(map(id, modules))) == len(modules)
