"""The camera model and the core Perceiver: pixels to world positions,
duplicate boxes, and the ways a scan is refused."""

import math

import pytest

from conftest import FakeDetector, depth_frame, rgb_frame
from openarm_ai_brain_vla.perception.camera import CameraModel, rotate
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


def test_registries_import_only_the_chosen_backend_and_refuse_unknown_names():
    from openarm_ai_brain_vla.manipulation import make_manipulator
    from openarm_ai_brain_vla.perception import make_detector

    assert make_detector("none").name == "none"
    assert make_manipulator("none").name == "none"
    with pytest.raises(ValueError, match="unknown perception_backend 'sam9'"):
        make_detector("sam9")
    with pytest.raises(ValueError, match="unknown manipulation_backend"):
        make_manipulator("policy")
