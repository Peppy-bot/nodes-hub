import numpy as np
import pytest

from zed_camera_sdk import conversions


def test_resolution_member_maps_and_is_case_insensitive():
    assert conversions.resolution_member("hd720") == "HD720"
    assert conversions.resolution_member("HD1080") == "HD1080"
    assert conversions.resolution_member("vga") == "VGA"
    assert conversions.resolution_member("hd2k") == "HD2K"


def test_resolution_member_rejects_unknown():
    with pytest.raises(ValueError):
        conversions.resolution_member("4k")


@pytest.mark.parametrize(
    "resolution,rate",
    [("vga", 100), ("hd720", 60), ("hd1080", 30), ("hd2k", 15)],
)
def test_validate_frame_rate_accepts_legal_rates(resolution, rate):
    assert conversions.validate_frame_rate(resolution, rate) == rate


@pytest.mark.parametrize(
    "resolution,rate",
    [("vga", 25), ("hd720", 100), ("hd1080", 60), ("hd2k", 30)],
)
def test_validate_frame_rate_rejects_illegal_rates(resolution, rate):
    with pytest.raises(ValueError):
        conversions.validate_frame_rate(resolution, rate)


def test_depth_mode_member_maps_and_uppercases():
    assert conversions.depth_mode_member("neural") == "NEURAL"
    assert conversions.depth_mode_member("NEURAL_LIGHT") == "NEURAL_LIGHT"
    assert conversions.depth_mode_member("Neural_Plus") == "NEURAL_PLUS"


def test_depth_mode_member_rejects_non_neural():
    with pytest.raises(ValueError):
        conversions.depth_mode_member("ULTRA")


def test_color_mode_accepts_and_rejects():
    assert conversions.color_mode("auto") == "auto"
    assert conversions.color_mode("manual") == "manual"
    with pytest.raises(ValueError):
        conversions.color_mode("semi")


def test_min_depth_mm_converts_metres():
    assert conversions.min_depth_mm(0.3) == pytest.approx(300.0)


def test_min_depth_mm_rejects_non_positive():
    with pytest.raises(ValueError):
        conversions.min_depth_mm(0.0)
    with pytest.raises(ValueError):
        conversions.min_depth_mm(-1.0)


def test_bgra_to_bgr_bytes_drops_alpha_in_order():
    image = np.array(
        [[[10, 20, 30, 255], [40, 50, 60, 128]]], dtype=np.uint8
    )  # 1x2 BGRA
    assert conversions.bgra_to_bgr_bytes(image) == bytes([10, 20, 30, 40, 50, 60])


def test_bgra_to_bgr_bytes_rejects_wrong_channels():
    with pytest.raises(ValueError):
        conversions.bgra_to_bgr_bytes(np.zeros((2, 2, 3), dtype=np.uint8))


def test_depth_mm_to_z16_rounds_and_is_little_endian():
    depth = np.array([[1000.4, 2000.6]], dtype=np.float32)
    result = np.frombuffer(conversions.depth_mm_to_z16_bytes(depth), dtype="<u2")
    np.testing.assert_array_equal(result, [1000, 2001])


def test_depth_mm_to_z16_marks_invalid_as_zero():
    depth = np.array(
        [[np.nan, np.inf, -np.inf, -5.0, 70000.0, 500.0]], dtype=np.float32
    )
    result = np.frombuffer(conversions.depth_mm_to_z16_bytes(depth), dtype="<u2")
    np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 500])


def test_depth_mm_to_z16_rejects_non_2d():
    with pytest.raises(ValueError):
        conversions.depth_mm_to_z16_bytes(np.zeros((2, 2, 2), dtype=np.float32))
