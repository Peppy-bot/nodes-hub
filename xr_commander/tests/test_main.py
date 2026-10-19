from xr_commander import __main__ as main


def test_the_hint_names_the_pages_urls_and_the_usb_route():
    text = main._headset_url_hint(
        ["https://198.51.100.7:4443", "https://100.64.0.7:4443"], 4443
    )
    assert "open https://198.51.100.7:4443 or https://100.64.0.7:4443" in text
    assert "choose Start or Start Passthrough." in text
    assert "adb reverse tcp:4443 tcp:4443" in text
    assert "https://localhost:4443" in text


def test_the_hint_says_when_the_page_reaches_no_other_machine():
    text = main._headset_url_hint([], 4443)
    assert "no network address" in text
    assert "adb reverse tcp:4443 tcp:4443" in text


def test_hand_wiring_covers_both_hands_with_distinct_modules():
    assert set(main._HANDS) == {"left", "right"}
    modules = [
        module
        for wiring in main._HANDS.values()
        for module in (
            wiring.pose_setpoints,
            wiring.pose_states,
            wiring.gripper_setpoints,
        )
    ]
    assert len(set(map(id, modules))) == len(modules)
