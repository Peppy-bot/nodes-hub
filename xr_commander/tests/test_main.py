from tests.helpers import POSE
from xr_commander import __main__ as main
from xr_commander.devices import HandSample


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


def test_the_face_buttons_wire_home_low_and_ready_high():
    # The wiring table is the one place a swap could hide: pin module to
    # button, so exchanging the two entries fails here and nowhere else.
    (home_module, home_pressed), (ready_module, ready_pressed) = main._POSTURE_BUTTONS
    assert home_module.TARGET_ACTION_NAME == "move_to_home"
    assert ready_module.TARGET_ACTION_NAME == "move_to_ready"
    a_only = HandSample(pose=POSE, squeezing=False, trigger=0.0, primary_button=True)
    b_only = HandSample(pose=POSE, squeezing=False, trigger=0.0, secondary_button=True)
    assert home_pressed(a_only) and not home_pressed(b_only)
    assert ready_pressed(b_only) and not ready_pressed(a_only)
