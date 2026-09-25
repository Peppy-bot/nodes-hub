"""The camera model and the core Perceiver: pixels to world positions,
duplicate boxes, and the ways a scan is refused."""

import asyncio
import math

import pytest

from conftest import FakeDetector, depth_frame, rgb_frame
from openarm_ai_brain_vla.perception.camera import (
    INVERSE_PLUMB_BOB,
    NONE,
    PLUMB_BOB,
    CameraModel,
    distort,
    pixel_to_ray,
    rotate,
    undistort,
)
from openarm_ai_brain_vla.perception.frames import FrameStore, decode_color, decode_depth, depth_at
from openarm_ai_brain_vla.perception.perceiver import Perceiver, merge_duplicates
from openarm_ai_brain_vla.ports import Box, CancelToken, Refusal

IDENTITY = CameraModel.from_parameters(90.0, "0 0 0 0 0 0 1")


def close(a, b, tolerance=1e-6):
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def test_the_camera_pose_parameter_is_parsed_and_normalised():
    with pytest.raises(ValueError):
        CameraModel.from_parameters(52.0, "0 0 0")
    with pytest.raises(ValueError):
        CameraModel.from_parameters(0.0, "0 0 0 0 0 0 1")
    camera = CameraModel.from_parameters(52.0, "1, 2, 3, 0, 0, 0, 2")
    assert camera.position == (1.0, 2.0, 3.0)
    assert close(camera.orientation, (0.0, 0.0, 0.0, 1.0))


def test_an_identity_camera_deprojects_along_minus_z_with_x_right_and_y_up():
    width, height = 16, 12
    centre = IDENTITY.deproject(8.0, 6.0, 1.0, width, height)
    assert close(centre, (0.0, 0.0, -1.0))
    # 90 degree vertical field of view: the top edge is one focal length up.
    focal = IDENTITY.focal_px(height)
    assert close((focal,), (6.0,))
    right = IDENTITY.deproject(8.0 + focal, 6.0, 2.0, width, height)
    assert close(right, (2.0, 0.0, -2.0))
    up = IDENTITY.deproject(8.0, 6.0 - focal, 2.0, width, height)
    assert close(up, (0.0, 2.0, -2.0))


# A D455-like colour lens, the magnitudes a real unit reports.
LENS = (-0.055, 0.066, -0.0007, 0.0005, -0.021)


def test_a_pixel_becomes_a_ray_through_the_intrinsics_with_no_lens():
    # Under "none" the ray is the pixel through fx, fy, cx, cy and nothing
    # else, so the principal point looks straight ahead and one focal
    # length to the right is 45 degrees.
    assert close(pixel_to_ray(639.5, 359.5, 738.1, 738.1, 639.5, 359.5, NONE, ()), (0.0, 0.0))
    assert close(pixel_to_ray(639.5 + 738.1, 359.5, 738.1, 738.1, 639.5, 359.5, NONE, ()), (1.0, 0.0))
    # deproject is the same path with the field of view for the focal length.
    assert close(IDENTITY.deproject(8.0 + 6.0, 6.0, 2.0, 16, 12), (2.0, 0.0, -2.0))


def test_plumb_bob_round_trips_through_distort_and_undistort():
    # The forward polynomial distorts an ideal point; undistorting by
    # iteration brings it back, out to the image corners.
    for point in [(0.0, 0.0), (0.3, -0.2), (-0.6, 0.35), (0.8, 0.45)]:
        distorted = distort(PLUMB_BOB, LENS, *point)
        assert not close(distorted, point, 1e-4) or point == (0.0, 0.0)
        assert close(undistort(PLUMB_BOB, LENS, *distorted), point, 1e-9)
    # And the same round trip the other way for the inverse model.
    for point in [(0.3, -0.2), (-0.6, 0.35)]:
        distorted = distort(INVERSE_PLUMB_BOB, LENS, *point)
        assert close(undistort(INVERSE_PLUMB_BOB, LENS, *distorted), point, 1e-9)


def test_the_inverse_model_applies_the_polynomial_once_from_the_distorted_point():
    # inverse_plumb_bob is one application of the polynomial to the
    # distorted point, nothing iterated: the result is the closed form.
    xd, yd = 0.3, -0.2
    k1, k2, p1, p2, k3 = LENS
    r2 = xd * xd + yd * yd
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    expected = (
        xd * radial + 2.0 * p1 * xd * yd + p2 * (r2 + 2.0 * xd * xd),
        yd * radial + p1 * (r2 + 2.0 * yd * yd) + 2.0 * p2 * xd * yd,
    )
    assert close(undistort(INVERSE_PLUMB_BOB, LENS, xd, yd), expected, 1e-12)


def test_inverse_and_forward_models_agree_for_small_coefficients():
    # To first order the inverse of the polynomial with coefficients c is
    # the polynomial with -c, so for a small lens the direct inverse model
    # with -c and the iterated forward model with c undistort a pixel to
    # the same ray, within the second-order term.
    for scale in (0.01, 0.003):
        small = tuple(c * scale for c in (-0.5, 0.6, -0.07, 0.05, -0.2))
        negated = tuple(-c for c in small)
        for xd, yd in [(0.3, -0.2), (-0.6, 0.35), (0.8, 0.45)]:
            forward = undistort(PLUMB_BOB, small, xd, yd)
            direct = undistort(INVERSE_PLUMB_BOB, negated, xd, yd)
            tolerance = 20.0 * scale * scale
            assert close(forward, direct, tolerance), (scale, xd, yd, forward, direct)
        # The agreement is second order: a lens ten times smaller agrees a
        # hundred times better.
    gaps = []
    for scale in (0.01, 0.001):
        small = tuple(c * scale for c in (-0.5, 0.6, -0.07, 0.05, -0.2))
        negated = tuple(-c for c in small)
        forward = undistort(PLUMB_BOB, small, 0.8, 0.45)
        direct = undistort(INVERSE_PLUMB_BOB, negated, 0.8, 0.45)
        gaps.append(math.hypot(forward[0] - direct[0], forward[1] - direct[1]))
    assert gaps[0] / gaps[1] > 50.0


def test_any_other_distortion_model_is_refused_by_name():
    with pytest.raises(ValueError, match="kannala_brandt"):
        pixel_to_ray(10.0, 10.0, 500.0, 500.0, 320.0, 240.0, "kannala_brandt", (0.1, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="modified_plumb_bob"):
        undistort("modified_plumb_bob", LENS, 0.1, 0.1)
    # The right name with the wrong number of coefficients is refused too.
    with pytest.raises(ValueError, match="k1, k2, p1, p2, k3"):
        undistort(PLUMB_BOB, (0.1, 0.0), 0.1, 0.1)
    with pytest.raises(ValueError, match="no coefficients"):
        undistort(NONE, (0.1,), 0.1, 0.1)
    with pytest.raises(ValueError, match="finite"):
        undistort(INVERSE_PLUMB_BOB, (math.nan, 0.0, 0.0, 0.0, 0.0), 0.1, 0.1)


def test_the_default_chest_pose_looks_ahead_and_down():
    # The OpenArm v2 chest camera faces world +X, 62 degrees below the horizon.
    chest = CameraModel.from_parameters(52.0, "0.0792 0.0315 0.7941 0.1710647 -0.1710647 -0.6861027 0.6861027")
    forward = chest.forward()
    assert close(forward, (math.cos(math.radians(62)), 0.0, -math.sin(math.radians(62))), 1e-3)
    assert close(rotate(chest.orientation, (0.0, 1.0, 0.0)), (math.sin(math.radians(62)), 0.0, math.cos(math.radians(62))), 1e-3)
    # One meter along the axis from the chest lands on the table in front.
    point = chest.deproject(640.0, 360.0, 1.0, 1280, 720)
    assert close(point, (0.0792 + forward[0], 0.0315, 0.7941 + forward[2]), 1e-3)


def test_frames_decode_and_depth_reads_the_median_under_a_box():
    colour = decode_color(rgb_frame(16, 12))
    assert colour.shape == (12, 16, 3)
    depth = decode_depth(depth_frame(0.75, 8, 6), 0.001)
    assert depth.shape == (6, 8) and abs(float(depth[0, 0]) - 0.75) < 1e-6
    # The box is in colour pixels; the depth frame is half the size.
    box = Box("cup", 0.9, 6.0, 4.0, 10.0, 8.0)
    assert abs(depth_at(depth, box, 16, 12) - 0.75) < 1e-6
    depth[:, :] = 0.0
    assert depth_at(depth, box, 16, 12) is None
    with pytest.raises(Refusal, match="unsupported colour encoding"):
        message = rgb_frame(4, 4)
        message.encoding = "yuyv"
        decode_color(message)


def test_duplicate_boxes_of_one_label_keep_the_most_confident():
    a = Box("cup", 0.9, 0, 0, 10, 10)
    b = Box("cup", 0.8, 1, 1, 11, 11)
    other = Box("bowl", 0.7, 1, 1, 11, 11)
    far = Box("cup", 0.6, 50, 50, 60, 60)
    kept = merge_duplicates([b, a, other, far])
    assert kept == [a, other, far]


async def test_a_scan_turns_boxes_into_world_detections():
    frames = FrameStore()
    frames.color = rgb_frame(16, 12)
    frames.depth = depth_frame(2.0, 16, 12)
    frames.depth_unit = 0.001
    detector = FakeDetector([Box("cup", 0.9, 6, 4, 10, 8), Box("cup", 0.85, 6.5, 4.5, 10.5, 8.5)])
    perceiver = Perceiver(detector, frames, IDENTITY)
    await perceiver.load("weights.pt")
    assert detector.loaded == "weights.pt"
    detections = await perceiver.scan(["cup"], CancelToken(), timeout_s=0.0)
    assert detector.vocabulary == ["cup"]
    assert len(detections) == 1
    assert detections[0].label == "cup" and detections[0].confidence == 0.9
    assert close(detections[0].position, (0.0, 0.0, -2.0))


async def test_a_scan_is_refused_without_a_detector_or_without_frames():
    from openarm_ai_brain_vla.perception.none import NoneDetector

    frames = FrameStore()
    none = Perceiver(NoneDetector(), frames, IDENTITY)
    with pytest.raises(Refusal, match="perception_backend is 'none'"):
        await none.scan([], CancelToken(), 0.0)
    fake = Perceiver(FakeDetector(), frames, IDENTITY)
    with pytest.raises(Refusal, match="no camera frame received"):
        await fake.scan([], CancelToken(), 0.0)


async def test_a_backend_loads_in_the_background_and_searches_wait_on_it():
    # The node's start must not wait on a model that takes a minute to
    # load, so the load runs beside it and a search meanwhile is refused as
    # still loading, not as no backend.
    class SlowDetector(FakeDetector):
        def __init__(self) -> None:
            super().__init__()
            self.ready = False
            self.release = threading.Event()

        @property
        def available(self) -> bool:
            return self.ready

        def load(self, model: str) -> None:
            self.release.wait(5.0)
            self.loaded = model
            self.ready = True

    import threading

    detector = SlowDetector()
    frames = FrameStore()
    perceiver = Perceiver(detector, frames, IDENTITY)
    task = perceiver.start_loading("gallery")
    await asyncio.sleep(0.05)
    assert not task.done()
    assert perceiver.why_unavailable() == "no perception source: perception_backend 'fake' is still loading"
    with pytest.raises(Refusal, match="still loading"):
        await perceiver.scan([], CancelToken(), timeout_s=0.0)
    detector.release.set()
    await perceiver.loaded()
    assert detector.loaded == "gallery" and detector.available
    assert perceiver.why_unavailable() == "no perception source: no camera frame received"


async def test_a_load_that_fails_becomes_the_reason_every_search_is_refused_with():
    class BrokenDetector(FakeDetector):
        @property
        def available(self) -> bool:
            return False

        def load(self, model: str) -> None:
            raise ValueError(f"no gallery at {model}")

    perceiver = Perceiver(BrokenDetector(), FrameStore(), IDENTITY)
    await perceiver.start_loading("/nowhere")
    assert perceiver.why_unavailable() == (
        "no perception source: perception_backend 'fake' could not load '/nowhere': no gallery at /nowhere"
    )
    with pytest.raises(Refusal, match="could not load '/nowhere'"):
        await perceiver.scan([], CancelToken(), timeout_s=0.0)


def test_registries_import_only_the_chosen_backend_and_refuse_unknown_names():
    from openarm_ai_brain_vla.manipulation import make_manipulator
    from openarm_ai_brain_vla.perception import make_detector

    assert make_detector("none").name == "none"
    assert make_manipulator("none").name == "none"
    with pytest.raises(ValueError, match="unknown perception_backend 'sam9'"):
        make_detector("sam9")
    with pytest.raises(ValueError, match="unknown manipulation_backend"):
        make_manipulator("policy")
