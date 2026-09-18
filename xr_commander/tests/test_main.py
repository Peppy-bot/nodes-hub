from xr_commander import __main__ as main


def test_the_hint_points_at_the_printed_pages_and_the_usb_route():
    text = main._headset_url_hint(4443)
    assert "Web pages:" in text
    assert "adb reverse tcp:4443 tcp:4443" in text
    assert "https://localhost:4443" in text


def test_hand_wiring_covers_both_hands_with_distinct_modules():
    assert set(main._HANDS) == {"left", "right"}
    modules = [
        module
        for wiring in main._HANDS.values()
        for module in (wiring.pose_setpoints, wiring.pose_states, wiring.gripper_setpoints)
    ]
    assert len(set(map(id, modules))) == len(modules)
