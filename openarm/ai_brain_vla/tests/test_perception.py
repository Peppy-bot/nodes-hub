"""The camera model, the frame store and the core Perceiver: pixels to
world positions, frames paired by capture, the camera's answers asked
until they come, duplicate boxes, and the ways a scan is refused."""

import asyncio
import math
import threading
from types import SimpleNamespace

import pytest

from conftest import NEVER, FakeDetector, FakeToken, camera_poses, depth_frame, rgb_frame
from openarm_ai_brain_vla.perception.camera import (
    INVERSE_PLUMB_BOB,
    NONE,
    PLUMB_BOB,
    CameraModel,
    Intrinsics,
    distort,
    pixel_to_ray,
    rotate,
    undistort,
)
from openarm_ai_brain_vla.perception import frames as frames_module
from openarm_ai_brain_vla.perception import geometry
from openarm_ai_brain_vla.perception.frames import FRAME_BUFFER, FrameStore, decode_color, decode_depth, depth_at
from openarm_ai_brain_vla.perception.perceiver import DEFAULT_TIMEOUT_S, Perceiver, in_daemon_thread, merge_duplicates
from openarm_ai_brain_vla.ports import Box, CancelToken, Coverage, Refusal

# At the robot frame's origin, looking along its +Z: the identity pose.
AT_ORIGIN = CameraModel().with_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
IDENTITY = AT_ORIGIN.with_intrinsics(Intrinsics.from_fovy(90.0, 16, 12))


def close(a, b, tolerance=1e-6):
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def test_the_camera_pose_the_robot_reports_is_checked_and_normalised():
    with pytest.raises(ValueError, match="three coordinates"):
        CameraModel().with_pose((0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        CameraModel().with_pose((math.nan, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="zero"):
        CameraModel().with_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    camera = CameraModel().with_pose((1, 2, 3), (0, 0, 0, 2))
    assert camera.pose.position == (1.0, 2.0, 3.0)
    assert close(camera.pose.orientation, (0.0, 0.0, 0.0, 1.0))
    # Without intrinsics, or without a pose, the camera cannot place a
    # pixel, and says which.
    assert camera.placed and not camera.ready
    with pytest.raises(ValueError, match="intrinsics are not known yet"):
        camera.deproject(1.0, 1.0, 1.0, 16, 12)
    unplaced = CameraModel().with_intrinsics(Intrinsics.from_fovy(90.0, 16, 12))
    assert not unplaced.placed and not unplaced.ready
    with pytest.raises(ValueError, match="pose is not known yet"):
        unplaced.deproject(1.0, 1.0, 1.0, 16, 12)
    with pytest.raises(ValueError, match="pose is not known yet"):
        unplaced.forward()
    assert camera.with_intrinsics(Intrinsics.from_fovy(90.0, 16, 12)).ready


def test_intrinsics_are_checked_and_scale_with_the_image():
    with pytest.raises(ValueError):
        Intrinsics.from_fovy(0.0, 16, 12)
    with pytest.raises(ValueError, match="positive finite focal"):
        Intrinsics(16, 12, 0.0, 6.0, 8.0, 6.0)
    with pytest.raises(ValueError, match="unknown distortion model"):
        Intrinsics(16, 12, 6.0, 6.0, 8.0, 6.0, "fisheye")
    # The ZED's colour intrinsics at 1280x720, used on a 640x360 stream:
    # focal lengths halve, the principal point follows the pixel centres.
    k = Intrinsics(1280, 720, 700.0, 700.0, 639.5, 359.5, PLUMB_BOB, (0.1, 0.0, 0.0, 0.0, 0.0))
    half = k.scaled_to(640, 360)
    assert (half.fx, half.fy) == (350.0, 350.0)
    assert close((half.cx, half.cy), (319.5, 179.5))
    assert half.distortion_model == PLUMB_BOB and half.distortion == (0.1, 0.0, 0.0, 0.0, 0.0)
    assert k.scaled_to(1280, 720) is k


def test_an_identity_camera_deprojects_along_plus_z_with_x_right_and_y_down():
    # The optical frame camera_geometry and camera_mounts share: +X to the
    # right of the image, +Y down it, +Z along the view.
    width, height = 16, 12
    centre = IDENTITY.deproject(8.0, 6.0, 1.0, width, height)
    assert close(centre, (0.0, 0.0, 1.0))
    assert close(IDENTITY.forward(), (0.0, 0.0, 1.0))
    # 90 degree vertical field of view: the top edge is one focal length up.
    focal = IDENTITY.intrinsics.fy
    assert close((focal,), (6.0,))
    right = IDENTITY.deproject(8.0 + focal, 6.0, 2.0, width, height)
    assert close(right, (2.0, 0.0, 2.0))
    down = IDENTITY.deproject(8.0, 6.0 + focal, 2.0, width, height)
    assert close(down, (0.0, 2.0, 2.0))


# A D455-like colour lens, the magnitudes a real unit reports.
LENS = (-0.055, 0.066, -0.0007, 0.0005, -0.021)


def test_a_pixel_becomes_a_ray_through_the_intrinsics_with_no_lens():
    # Under "none" the ray is the pixel through fx, fy, cx, cy and nothing
    # else, so the principal point looks straight ahead and one focal
    # length to the right is 45 degrees.
    assert close(pixel_to_ray(639.5, 359.5, 738.1, 738.1, 639.5, 359.5, NONE, ()), (0.0, 0.0))
    assert close(pixel_to_ray(639.5 + 738.1, 359.5, 738.1, 738.1, 639.5, 359.5, NONE, ()), (1.0, 0.0))
    # deproject is the same path with the field of view for the focal length.
    assert close(IDENTITY.deproject(8.0 + 6.0, 6.0, 2.0, 16, 12), (2.0, 0.0, 2.0))


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


def quaternion_product(a, b):
    """`a` then `b` as one rotation, both [x, y, z, w]: the rotation `a * b`."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# The OpenArm v2 chest camera as the backbone reports it: the design's mount
# (looking along its own -Z with +Y up) turned half a turn about X into the
# optical frame.
CHEST_POSITION = (0.0792, 0.0315, 0.7941)
CHEST_ORIENTATION = quaternion_product((0.1710647, -0.1710647, -0.6861027, 0.6861027), (1.0, 0.0, 0.0, 0.0))


def test_the_chest_pose_the_backbone_reports_looks_ahead_and_down():
    # The OpenArm v2 chest camera faces the robot's +X, 62 degrees below the
    # horizon, and the image's +Y (down the picture) leans forward and down.
    chest = CameraModel().with_pose(CHEST_POSITION, CHEST_ORIENTATION).with_intrinsics(Intrinsics.from_fovy(52.0, 1280, 720))
    forward = chest.forward()
    assert close(forward, (math.cos(math.radians(62)), 0.0, -math.sin(math.radians(62))), 1e-3)
    assert close(rotate(chest.pose.orientation, (0.0, 1.0, 0.0)), (-math.sin(math.radians(62)), 0.0, -math.cos(math.radians(62))), 1e-3)
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


def store_with(colors=(), depths=(), depth_unit=0.001) -> FrameStore:
    store = FrameStore()
    for message in colors:
        store.add_color(message)
    for message in depths:
        store.add_depth(message)
    store.depth_unit = depth_unit
    return store


def test_the_frame_store_pairs_colour_and_depth_by_capture():
    # Colour runs a frame ahead of depth: the newest pair is capture 2.
    store = store_with([rgb_frame(frame_id=1), rgb_frame(frame_id=2), rgb_frame(frame_id=3)], [depth_frame(1.0, frame_id=1), depth_frame(2.0, frame_id=2)])
    frame = store.latest()
    assert frame.color.header.frame_id == 2 and frame.depth.header.frame_id == 2
    assert store.why_unavailable() == "" and store.frames_seen == 3
    # Only the last FRAME_BUFFER frames of a stream are kept.
    for frame_id in range(10, 10 + FRAME_BUFFER + 2):
        store.add_color(rgb_frame(frame_id=frame_id))
    assert [m.header.frame_id for m in store.colors] == list(range(12, 12 + FRAME_BUFFER))


def test_the_frame_store_says_what_is_missing():
    assert FrameStore().why_unavailable() == "no camera frame received"
    assert store_with([rgb_frame()]).why_unavailable() == "no depth frame received"
    assert store_with([rgb_frame()], [depth_frame(1.0)], depth_unit=None).why_unavailable() == "the camera has not answered depth_stream_info yet"
    disjoint = store_with([rgb_frame(frame_id=1), rgb_frame(frame_id=2)], [depth_frame(1.0, frame_id=3)])
    assert disjoint.why_unavailable() == "no colour and depth frame of one capture received (latest colour frame_id 2, depth 3)"
    with pytest.raises(Refusal, match="no colour and depth frame of one capture"):
        disjoint.latest()


def test_frames_that_are_not_aligned_to_each_other_are_refused():
    unaligned = store_with([rgb_frame(align_mode="none")], [depth_frame(1.0, align_mode="none")])
    assert unaligned.why_unavailable() == "the camera's depth is not aligned to its colour (align_mode 'none'); the brain reads depth at colour pixels"
    mixed = store_with([rgb_frame(align_mode="depth_to_color")], [depth_frame(1.0, align_mode="none")])
    assert mixed.why_unavailable() == "the colour and depth frames name different alignments ('depth_to_color' and 'none')"
    for mode in ("depth_to_color", "color_to_depth"):
        assert store_with([rgb_frame(align_mode=mode)], [depth_frame(1.0, align_mode=mode)]).why_unavailable() == ""
    unknown = store_with([rgb_frame(align_mode="rectified")], [depth_frame(1.0, align_mode="rectified")])
    assert "align_mode 'rectified'" in unknown.why_unavailable()


class Feed:
    """A subscription that delivers its messages, then waits until it is
    cancelled; `delivered` says the messages are all out."""

    def __init__(self, messages) -> None:
        self.messages = list(messages)
        self.delivered = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return ("camera", self.messages.pop(0))
        self.delivered.set()
        await asyncio.Event().wait()


async def test_the_frame_store_asks_the_depth_unit_until_answered_then_follows_both_streams(monkeypatch):
    answers = [TimeoutError("no answer yet"), SimpleNamespace(data=SimpleNamespace(depth_unit=0.001))]

    async def poll(node_runner, producer, timeout):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    colours = Feed([rgb_frame(frame_id=1), rgb_frame(frame_id=2)])
    depths = Feed([depth_frame(1.0, frame_id=2)])

    async def subscribe_colour(node_runner):
        return colours

    async def subscribe_depth(node_runner):
        return depths

    monkeypatch.setattr(frames_module, "DEPTH_INFO_RETRY_S", 0.0)
    monkeypatch.setattr(frames_module.depth_stream_info, "poll", poll)
    monkeypatch.setattr(frames_module.video_stream, "bound_producer", lambda node_runner: "camera")
    monkeypatch.setattr(frames_module.video_stream, "subscribe", subscribe_colour)
    monkeypatch.setattr(frames_module.depth_stream, "subscribe", subscribe_depth)
    store = FrameStore()
    token = FakeToken()
    running = asyncio.create_task(store.run(None, token))
    await asyncio.wait_for(colours.delivered.wait(), 1.0)
    await asyncio.wait_for(depths.delivered.wait(), 1.0)
    assert answers == [] and store.depth_unit == 0.001 and store.frames_seen == 2
    frame = store.latest()
    assert frame.color.header.frame_id == 2 and frame.depth.header.frame_id == 2
    token.cancel()
    await asyncio.wait_for(running, 1.0)


async def test_a_vacant_camera_slot_leaves_the_store_empty(monkeypatch):
    monkeypatch.setattr(frames_module.video_stream, "bound_producer", lambda node_runner: None)
    store = FrameStore()
    await asyncio.wait_for(store.run(None, FakeToken()), 1.0)
    assert store.why_unavailable() == "no camera frame received"


def test_duplicate_boxes_of_one_label_keep_the_most_confident():
    a = Box("cup", 0.9, 0, 0, 10, 10)
    b = Box("cup", 0.8, 1, 1, 11, 11)
    other = Box("bowl", 0.7, 1, 1, 11, 11)
    far = Box("cup", 0.6, 50, 50, 60, 60)
    kept = merge_duplicates([b, a, other, far])
    assert kept == [a, other, far]


async def test_a_scan_turns_boxes_into_world_detections():
    frames = store_with([rgb_frame(16, 12)], [depth_frame(2.0, 16, 12)])
    detector = FakeDetector([Box("cup", 0.9, 6, 4, 10, 8), Box("cup", 0.85, 6.5, 4.5, 10.5, 8.5)])
    perceiver = Perceiver(detector, frames, IDENTITY)
    await perceiver.load("weights.pt", "/gallery")
    assert detector.loaded == ("weights.pt", "/gallery")
    look = await perceiver.scan(["cup"], CancelToken(), timeout_s=0.0)
    assert detector.vocabulary == ["cup"]
    detections = look.detections
    assert len(detections) == 1
    assert detections[0].label == "cup" and detections[0].confidence == 0.9
    assert close(detections[0].position, (0.0, 0.0, 2.0))
    # The look reports the box in the picture it used, and the picture.
    assert detections[0].region == (6, 4, 10, 8)
    assert (look.image_width, look.image_height, look.frame_timestamp) == (16, 12, 1.0)
    # The detector is handed the search's deadline: the default budget for
    # a zero timeout, the caller's otherwise.
    assert detector.deadlines[-1].budget_s == DEFAULT_TIMEOUT_S
    await perceiver.scan([], CancelToken(), timeout_s=2.5)
    assert detector.deadlines[-1].budget_s == 2.5


async def test_a_detectors_calls_run_in_daemon_threads_of_their_own():
    threads: dict[str, threading.Thread] = {}

    class Recording(FakeDetector):
        def load(self, model, gallery):
            threads["load"] = threading.current_thread()

        def detect(self, image, deadline):
            threads["detect"] = threading.current_thread()
            return []

    perceiver = Perceiver(Recording(), store_with([rgb_frame()], [depth_frame(1.0)]), IDENTITY)
    await perceiver.load("", "")
    await perceiver.scan([], CancelToken(), 0.0)
    assert threads["load"].daemon and threads["detect"].daemon
    assert threads["load"] is not threading.main_thread()
    assert threads["load"].name == "brain-load" and threads["detect"].name == "brain-detect"

    def boom():
        raise ValueError("in the thread")

    with pytest.raises(ValueError, match="in the thread"):
        await in_daemon_thread(boom)


async def test_a_scan_is_refused_without_a_detector_or_without_frames():
    from openarm_ai_brain_vla.perception.none import NoneDetector

    frames = FrameStore()
    assert NoneDetector().scan_coverage() == Coverage()
    assert NoneDetector().detect(None, NEVER) == []
    none = Perceiver(NoneDetector(), frames, IDENTITY)
    with pytest.raises(Refusal, match="perception_backend is 'none'"):
        await none.scan([], CancelToken(), 0.0)
    fake = Perceiver(FakeDetector(), frames, IDENTITY)
    with pytest.raises(Refusal, match="no camera frame received"):
        await fake.scan([], CancelToken(), 0.0)
    assert not fake.available
    fake.frames.add_color(rgb_frame(align_mode="none"))
    fake.frames.add_depth(depth_frame(1.0, align_mode="none"))
    fake.frames.depth_unit = 0.001
    with pytest.raises(Refusal, match="no perception source: the camera's depth is not aligned"):
        await fake.scan([], CancelToken(), 0.0)


async def test_the_none_backend_refuses_a_model_or_a_gallery_it_cannot_read():
    from openarm_ai_brain_vla.perception.none import NoneDetector

    NoneDetector().load("", "")
    for model, gallery in (("weights", ""), ("", "/gallery")):
        with pytest.raises(ValueError, match="perception_backend 'none' loads no model"):
            NoneDetector().load(model, gallery)


async def test_a_backend_loads_in_the_background_and_searches_wait_on_it():
    # The node's start must not wait on a model that takes a minute to
    # load, so the load runs beside it and a search meanwhile is refused as
    # still loading, not as no backend.
    class SlowDetector(FakeDetector):
        def __init__(self) -> None:
            super().__init__()
            self.ready = False
            self.entered = threading.Event()
            self.release = threading.Event()

        @property
        def available(self) -> bool:
            return self.ready

        def load(self, model: str, gallery: str) -> None:
            self.entered.set()
            self.release.wait(5.0)
            self.loaded = (model, gallery)
            self.ready = True

    detector = SlowDetector()
    frames = FrameStore()
    perceiver = Perceiver(detector, frames, IDENTITY)
    task = perceiver.start_loading("", "gallery")
    await asyncio.wait_for(asyncio.to_thread(detector.entered.wait), 5.0)
    assert not task.done()
    assert perceiver.why_unavailable() == "no perception source: perception_backend 'fake' is still loading"
    with pytest.raises(Refusal, match="still loading"):
        await perceiver.scan([], CancelToken(), timeout_s=0.0)
    detector.release.set()
    await perceiver.loaded()
    assert detector.loaded == ("", "gallery") and detector.available
    assert perceiver.why_unavailable() == "no perception source: no camera frame received"


async def test_a_load_that_fails_becomes_the_reason_every_search_is_refused_with():
    class BrokenDetector(FakeDetector):
        @property
        def available(self) -> bool:
            return False

        def load(self, model: str, gallery: str) -> None:
            raise ValueError(f"gallery {gallery} is not a directory the node can see")

    perceiver = Perceiver(BrokenDetector(), FrameStore(), IDENTITY)
    await perceiver.start_loading("", "/nowhere")
    assert perceiver.why_unavailable() == (
        "no perception source: perception_backend 'fake' could not load: gallery /nowhere is not a directory the node can see"
    )
    with pytest.raises(Refusal, match="could not load: gallery /nowhere"):
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


def test_the_confidence_parameter_reaches_the_detector_unless_it_is_zero():
    from openarm_ai_brain_vla.brain import Brain
    from conftest import PARAMS

    detector = FakeDetector()
    detector.min_confidence = 0.25
    params = SimpleNamespace(**{**PARAMS, "perception_confidence": 0.10})
    assert Brain(params, node_runner=None, detector=detector).perceiver.detector.min_confidence == 0.10
    detector.min_confidence = 0.25
    params = SimpleNamespace(**{**PARAMS, "perception_confidence": 0.0})
    assert Brain(params, node_runner=None, detector=detector).perceiver.detector.min_confidence == 0.25


async def test_a_scan_waits_for_the_cameras_intrinsics_and_names_the_slot():
    from openarm_ai_brain_vla.perception.geometry import VACANT, intrinsics_from

    frames = store_with([rgb_frame(16, 12)], [depth_frame(1.0, 16, 12)])
    perceiver = Perceiver(FakeDetector([Box("mug", 0.9, 4, 4, 8, 8)]), frames, AT_ORIGIN)
    assert not perceiver.available
    with pytest.raises(Refusal, match="intrinsics have not been received"):
        await perceiver.scan([], CancelToken(), 1.0)
    perceiver.intrinsics_reason = VACANT
    with pytest.raises(Refusal, match="geometry slot is vacant"):
        await perceiver.scan([], CancelToken(), 1.0)
    answer = SimpleNamespace(width=16, height=12, fx=6.0, fy=6.0, cx=8.0, cy=6.0, distortion_model="none", distortion=[])
    perceiver.set_camera(perceiver.camera.with_intrinsics(intrinsics_from(answer)))
    assert perceiver.available and perceiver.intrinsics_reason == ""
    found = await perceiver.scan([], CancelToken(), 1.0)
    assert [d.label for d in found.detections] == ["mug"]
    with pytest.raises(ValueError, match="unknown distortion model"):
        intrinsics_from(SimpleNamespace(width=16, height=12, fx=6.0, fy=6.0, cx=8.0, cy=6.0, distortion_model="fisheye", distortion=[]))


def colour_answer(success: bool = True, message: str = "", fx: float = 6.0):
    return SimpleNamespace(success=success, message=message, width=16, height=12, fx=fx, fy=6.0, cx=8.0, cy=6.0, distortion_model="none", distortion=[])


def depth_answer(success: bool = True, message: str = "", depth_model: str = "z"):
    return SimpleNamespace(success=success, message=message, depth_model=depth_model)


async def test_the_camera_is_asked_again_until_it_knows_its_geometry(monkeypatch):
    # The reasons a search is refused with, one per round, until the
    # camera answers both questions with something usable: not answered,
    # colour not known yet (what a sim relay says before its simulation
    # speaks), depth not known, a depth model the brain does not read, an
    # unusable pinhole, then the answer.
    perceiver = Perceiver(FakeDetector(), FrameStore(), AT_ORIGIN)
    colour = [
        TimeoutError("no answer"),
        colour_answer(False, "no camera geometry received from the simulation yet"),
        colour_answer(),
        colour_answer(),
        colour_answer(fx=0.0),
        colour_answer(),
    ]
    depth = [depth_answer(), depth_answer(False, "no depth stream"), depth_answer(depth_model="range"), depth_answer(), depth_answer()]
    reasons: list[str] = []

    async def poll_colour(node_runner, producer, timeout):
        reasons.append(perceiver.intrinsics_reason)
        answer = colour.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(data=answer)

    async def poll_depth(node_runner, producer, timeout):
        return SimpleNamespace(data=depth.pop(0))

    monkeypatch.setattr(geometry, "INTRINSICS_RETRY_S", 0.0)
    monkeypatch.setattr(geometry.get_color_intrinsics, "bound_producer", lambda node_runner: "camera")
    monkeypatch.setattr(geometry.get_color_intrinsics, "poll", poll_colour)
    monkeypatch.setattr(geometry.get_depth_intrinsics, "poll", poll_depth)
    await asyncio.wait_for(geometry.learn_intrinsics(None, FakeToken(), perceiver), 5.0)
    assert reasons == [
        "the camera's intrinsics have not been received yet",
        "the camera's geometry is not answered yet (TimeoutError('no answer'))",
        "the camera does not know its colour intrinsics: no camera geometry received from the simulation yet",
        "the camera does not know its depth intrinsics: no depth stream",
        "the camera's depth samples are 'range' distances; the brain reads distances along the optical axis ('z')",
        "the camera's intrinsics are unusable: intrinsics need positive finite focal lengths, got fx 0.0 fy 6.0",
    ]
    assert colour == [] and depth == []
    assert perceiver.camera.ready and perceiver.intrinsics_reason == ""


async def test_a_camera_that_never_knows_its_geometry_is_asked_until_the_node_stops(monkeypatch):
    perceiver = Perceiver(FakeDetector(), FrameStore(), AT_ORIGIN)
    token = FakeToken()
    asked = 0

    async def poll_colour(node_runner, producer, timeout):
        nonlocal asked
        asked += 1
        if asked == 3:
            token.cancel()
        return SimpleNamespace(data=colour_answer(False, "this camera has no calibration"))

    async def poll_depth(node_runner, producer, timeout):
        return SimpleNamespace(data=depth_answer())

    monkeypatch.setattr(geometry, "INTRINSICS_RETRY_S", 0.0)
    monkeypatch.setattr(geometry.get_color_intrinsics, "bound_producer", lambda node_runner: "camera")
    monkeypatch.setattr(geometry.get_color_intrinsics, "poll", poll_colour)
    monkeypatch.setattr(geometry.get_depth_intrinsics, "poll", poll_depth)
    await asyncio.wait_for(geometry.learn_intrinsics(None, token, perceiver), 5.0)
    assert asked == 3
    assert not perceiver.camera.ready
    assert perceiver.intrinsics_reason == "the camera does not know its colour intrinsics: this camera has no calibration"


async def test_a_vacant_geometry_slot_is_the_reason_at_once(monkeypatch):
    perceiver = Perceiver(FakeDetector(), FrameStore(), AT_ORIGIN)
    monkeypatch.setattr(geometry.get_color_intrinsics, "bound_producer", lambda node_runner: None)
    await asyncio.wait_for(geometry.learn_intrinsics(None, FakeToken(), perceiver), 1.0)
    assert perceiver.intrinsics_reason == geometry.VACANT


def poses_answer(success: bool = True, message: str = "", names=("wrist_left", "chest"), chest_position=(0.1, 0.2, 0.3)):
    """The robot's answer as conftest's camera_poses gives it, the last
    camera named, the chest, standing at `chest_position`."""
    answer = camera_poses(success, message, names)
    if success:
        answer.positions[-3:] = list(chest_position)
    return answer


async def test_the_robot_is_asked_again_until_it_places_the_camera(monkeypatch):
    # The reasons a search is refused with, one per round, until the robot
    # answers with success and names the camera: not answered, not measured
    # yet, a robot without that camera, then the answer.
    frames = store_with([rgb_frame(16, 12)], [depth_frame(1.0, 16, 12)])
    perceiver = Perceiver(FakeDetector(), frames, CameraModel().with_intrinsics(Intrinsics.from_fovy(90.0, 16, 12)))
    assert not perceiver.available
    with pytest.raises(Refusal, match="pose has not been received from the robot"):
        await perceiver.scan([], CancelToken(), 1.0)
    answers = [
        TimeoutError("no answer"),
        poses_answer(False, "the robot has not measured its joints yet"),
        poses_answer(names=("wrist_left", "wrist_right")),
        poses_answer(),
    ]
    reasons: list[str] = []

    async def poll(node_runner, producer, timeout):
        reasons.append(perceiver.pose_reason)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(data=answer)

    monkeypatch.setattr(geometry, "INTRINSICS_RETRY_S", 0.0)
    monkeypatch.setattr(geometry.get_camera_poses, "bound_producer", lambda node_runner: "robot")
    monkeypatch.setattr(geometry.get_camera_poses, "poll", poll)
    await asyncio.wait_for(geometry.learn_camera_pose(None, FakeToken(), perceiver, "chest"), 5.0)
    assert reasons == [
        "the camera's pose has not been received from the robot yet",
        "the robot's camera mounts are not answered yet (TimeoutError('no answer'))",
        "the robot does not know where its cameras stand: the robot has not measured its joints yet",
        "the robot carries no camera named 'chest' (it carries: wrist_left, wrist_right)",
    ]
    assert answers == []
    assert perceiver.camera.ready and perceiver.pose_reason == ""
    assert perceiver.camera.pose.position == (0.1, 0.2, 0.3)
    assert perceiver.available


async def test_a_vacant_camera_mounts_slot_is_the_reason_at_once(monkeypatch):
    perceiver = Perceiver(FakeDetector(), FrameStore(), CameraModel())
    monkeypatch.setattr(geometry.get_camera_poses, "bound_producer", lambda node_runner: None)
    await asyncio.wait_for(geometry.learn_camera_pose(None, FakeToken(), perceiver, "chest"), 1.0)
    assert perceiver.pose_reason == geometry.MOUNTS_VACANT
    assert not perceiver.camera.placed
