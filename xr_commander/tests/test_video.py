import asyncio
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from peppygen.consumed_topics.color_cameras import video_stream as color_video_stream

from tests.helpers import (
    FakeToken,
    RecordingSink,
    StubTopic,
    boot,
    eventually,
    frame_message,
    running_drain,
)
from xr_commander.video import (
    TRACK_HZ,
    CameraPanels,
    CameraSource,
    LastFrame,
    camera_source,
    chosen_panels,
    decode_to_bgr,
    drain_frames,
    feed_panels,
    follow_camera_panels,
    no_signal_frame,
    panel_of,
    shows_for,
    shows_live_video,
    shrink_to_width,
    watch_camera_silence,
)


def test_bgr8_passes_through_with_its_geometry_intact():
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(3, 2, 3)
    decoded = decode_to_bgr("bgr8", 2, 3, frame.tobytes())
    assert decoded.shape == (3, 2, 3)
    assert np.array_equal(decoded, frame)
    # An owned, writable copy: a view would alias the recycled wire buffer.
    assert decoded.flags["OWNDATA"] and decoded.flags["WRITEABLE"]


def test_rgb8_is_reordered_because_the_encoder_expects_bgr():
    # teleop_xr's track runs a BGR-to-RGB conversion before encoding, so a frame
    # handed over in RGB would reach the headset with red and blue swapped.
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)  # pure red, RGB order
    decoded = decode_to_bgr("rgb8", 1, 1, rgb.tobytes())
    assert tuple(decoded[0, 0]) == (0, 0, 255)  # pure red, BGR order


def test_a_contiguous_array_is_produced_for_the_encoder():
    decoded = decode_to_bgr("rgb8", 4, 2, np.zeros((2, 4, 3), dtype=np.uint8).tobytes())
    assert decoded.flags["C_CONTIGUOUS"]


def test_yuyv_decodes_to_the_declared_geometry():
    raw = np.full((2, 4, 2), 128, dtype=np.uint8)
    decoded = decode_to_bgr("yuyv", 4, 2, raw.tobytes())
    assert decoded.shape == (2, 4, 3)


def test_mjpeg_round_trips_through_the_jpeg_decoder():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 2] = 200
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    # The declared geometry is deliberately wrong: a compressed frame carries
    # its own, and the header's is not used to reshape it.
    decoded = decode_to_bgr("mjpeg", 999, 999, buffer.tobytes())
    assert decoded.shape == (8, 8, 3)


def test_an_undecodable_mjpeg_payload_is_rejected():
    with pytest.raises(ValueError, match="failed to decode"):
        decode_to_bgr("mjpeg", 8, 8, b"not a jpeg")


def test_an_unknown_encoding_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unsupported encoding"):
        decode_to_bgr("z16", 2, 2, b"\x00" * 8)


def test_a_payload_that_contradicts_its_header_is_refused():
    # A silently wrong reshape would either throw somewhere far less
    # informative or reinterpret the wrong bytes as an image.
    with pytest.raises(ValueError, match="expected 12"):
        decode_to_bgr("rgb8", 2, 2, b"\x00" * 11)


def member(instance_id, copy=None, in_copy_id=None, core_node="core"):
    """A bound camera as the generated slot answers it."""
    tag = None if copy is None else SimpleNamespace(name=copy, instance_id=in_copy_id)
    producer = SimpleNamespace(core_node=core_node, instance_id=instance_id)
    return SimpleNamespace(copy=tag, producer=producer)


def test_a_camera_is_named_by_its_robot_and_the_id_its_copy_wrote():
    assert camera_source(
        member("alpha_wrist_left", "alpha", "wrist_left")
    ) == CameraSource(robot="alpha", role="wrist_left")
    assert camera_source(member("chest")) == CameraSource(robot=None, role="chest")


def test_the_chosen_robots_cameras_show_and_a_copyless_camera_always_does():
    alpha = CameraSource(robot="alpha", role="chest")
    assert shows_for(alpha, "alpha")
    assert not shows_for(alpha, "bravo")
    assert not shows_for(alpha, None)
    assert shows_for(CameraSource(robot=None, role="chest"), "bravo")


def recording_panels(reserved=frozenset()):
    """Camera panels over recording sinks, with the sinks by panel name and
    every set of views told to the session."""
    sinks: dict[str, RecordingSink] = {}
    told: list[dict] = []

    def show_views(views):
        sinks.update(views)
        told.append(dict(views))

    panels = CameraPanels(
        make_sink=RecordingSink, show_views=show_views, reserved=reserved
    )
    return panels, sinks, told


def sides(**by_robot):
    """An arm_sides answering the hands each named robot pairs an arm with."""
    return lambda robot: frozenset(by_robot.get(robot, ()))


def robots_choosing(chosen, **by_robot):
    """The paired robots as the camera code reads them."""
    return SimpleNamespace(chosen=lambda: chosen, arm_sides=sides(**by_robot))


def test_a_camera_fills_the_panel_of_its_role():
    both = sides(alpha=["left", "right"])
    assert panel_of(CameraSource("alpha", "chest"), both) == "chest"
    assert panel_of(CameraSource(None, "chest"), sides()) == "chest"


def test_a_one_armed_robots_wrist_camera_fills_the_wrist_panel_of_its_hand():
    wrist = CameraSource("bravo", "wrist")
    assert panel_of(wrist, sides(bravo=["right"])) == "wrist_right"
    assert panel_of(wrist, sides(bravo=["left"])) == "wrist_left"
    assert panel_of(wrist, sides(bravo=["left", "right"])) == "wrist"


def test_a_camera_of_a_robot_pairing_no_arm_fills_no_panel():
    assert panel_of(CameraSource("charlie", "wrist"), sides()) is None
    assert panel_of(CameraSource("charlie", "chest"), sides()) is None


def cameras_of(*members):
    """A camera slot module answering `members` as bound."""
    return SimpleNamespace(bound_members=lambda _runner: list(members))


OPENARM_AND_SO101 = cameras_of(
    member("alpha_wrist_left", "alpha", "wrist_left"),
    member("alpha_wrist_right", "alpha", "wrist_right"),
    member("alpha_chest", "alpha", "chest"),
    member("bravo_wrist", "bravo", "wrist"),
)


def test_the_chosen_robots_cameras_alone_are_its_panels():
    pairs = {"alpha": ["left", "right"], "bravo": ["right"]}
    openarm = robots_choosing("alpha", **pairs)
    so101 = robots_choosing("bravo", **pairs)
    assert chosen_panels(None, [OPENARM_AND_SO101], openarm) == {
        "wrist_left",
        "wrist_right",
        "chest",
    }
    assert chosen_panels(None, [OPENARM_AND_SO101], so101) == {"wrist_right"}
    # A camera outside any copy is every robot's.
    copyless = cameras_of(member("overhead"))
    assert chosen_panels(None, [copyless], so101) == {"overhead"}


def test_the_session_is_told_the_shown_panels_once_per_change(capsys):
    panels, _sinks, told = recording_panels(reserved=frozenset({"status"}))
    panels.show_only(["wrist_right", "chest"])
    panels.show_only(["chest", "wrist_right"])
    panels.show_only(["wrist_right"])
    assert [sorted(views) for views in told] == [
        ["chest", "wrist_right"],
        ["wrist_right"],
    ]
    # A panel keeps its sink across sets: a page in session holds its track.
    assert told[0]["wrist_right"] is told[1]["wrist_right"]
    assert panels.shown() == ("wrist_right",)
    out = capsys.readouterr().out
    assert "camera panels shown: chest, wrist_right" in out
    assert "camera panels shown: wrist_right" in out


def test_a_reserved_name_is_refused_once_and_never_shown(capsys):
    panels, _sinks, told = recording_panels(reserved=frozenset({"status"}))
    assert not panels.ensure("status")
    panels.show_only(["status", "chest"])
    assert told == [{"chest": told[0]["chest"]}]
    assert capsys.readouterr().out.count("status panel's name") == 1


def test_a_feed_hands_each_shown_panel_the_chosen_robots_newest_frame():
    panels, sinks, _told = recording_panels()
    first, second = np.zeros((1, 1, 3), np.uint8), np.ones((1, 1, 3), np.uint8)
    alpha = CameraSource("alpha", "chest")
    panels.show_only(["chest", "wrist_left"])
    panels.show("chest", first, alpha)
    panels.show("chest", second, alpha)
    panels.feed("alpha")
    panels.feed("alpha")
    assert [frame is second for frame in sinks["chest"].frames] == [True, True]
    # A shown panel nothing was shown on is fed nothing.
    assert sinks["wrist_left"].frames == []
    # Another robot's frame is held back once the choice moves on.
    panels.feed("bravo")
    assert len(sinks["chest"].frames) == 2
    # A hidden panel is fed nothing.
    panels.show_only(["wrist_left"])
    panels.feed("alpha")
    assert len(sinks["chest"].frames) == 2


def test_the_panels_are_fed_at_the_tracks_rate():
    panels, sinks, _told = recording_panels()
    panels.show_only(["chest"])
    panels.show("chest", np.zeros((1, 1, 3), np.uint8))
    token = FakeToken()
    window_s = 3 / TRACK_HZ

    async def run():
        task = asyncio.create_task(feed_panels(panels, lambda: None, token))
        await asyncio.sleep(window_s)
        token.cancel()
        await asyncio.wait_for(task, 1.0)

    asyncio.run(run())
    assert 2 <= len(sinks["chest"].frames) <= 4


async def test_following_the_slots_shows_a_panel_per_bound_camera_role():
    async with boot(color_cameras_instance_ids=["wrist_left", "chest"]) as h:
        panels, _sinks, _told = recording_panels()
        async with running_drain(
            lambda token: follow_camera_panels(
                h.node_runner,
                [color_video_stream],
                panels,
                robots_choosing(None),
                token,
                period_s=0.01,
            )
        ):
            await eventually(
                lambda: panels.shown() == ("chest", "wrist_left"),
                message="a panel per role",
            )


async def test_the_panels_follow_the_choice_and_the_arms_as_they_pair():
    arms: dict[str, list[str]] = {}
    chosen = ["bravo"]
    robots = SimpleNamespace(
        chosen=lambda: chosen[0],
        arm_sides=lambda robot: frozenset(arms.get(robot, ())),
    )
    panels, _sinks, _told = recording_panels()
    async with running_drain(
        lambda token: follow_camera_panels(
            None, [OPENARM_AND_SO101], panels, robots, token, period_s=0.01
        )
    ):
        await asyncio.sleep(0.05)
        assert panels.shown() == ()
        arms.update(bravo=["right"], alpha=["left", "right"])
        await eventually(
            lambda: panels.shown() == ("wrist_right",), message="bravo's wrist panel"
        )
        chosen[0] = "alpha"
        await eventually(
            lambda: panels.shown() == ("chest", "wrist_left", "wrist_right"),
            message="alpha's three panels",
        )


def bgr_message(width=1, height=1):
    data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
    return color_video_stream.Message(
        header=color_video_stream.MessageHeader(timestamp=0.0, frame_id=0),
        encoding="bgr8",
        width=width,
        height=height,
        frame=data,
    )


# No robot chosen and none pairing an arm: the harness's cameras are
# outside any copy, which every robot shows.
NO_ROBOTS = robots_choosing(None)


async def test_an_undecodable_frame_is_dropped_and_logged_once(capsys):
    async with boot(color_cameras_instance_ids=["chest"]) as h:
        camera = h.mocks.deps.color_cameras[0]
        panels, sinks, _told = recording_panels()
        panels.show_only(["chest"])
        health: dict[str, LastFrame] = {}
        async with running_drain(
            lambda token: drain_frames(
                h.node_runner,
                color_video_stream,
                panels,
                NO_ROBOTS,
                token,
                "test",
                0,
                health,
            )
        ):
            bad = color_video_stream.Message(
                header=color_video_stream.MessageHeader(timestamp=0.0, frame_id=0),
                encoding="z16",
                width=1,
                height=1,
                frame=b"xx",
            )
            await camera.video_stream.publish(bad)
            await camera.video_stream.publish(bad)
            await camera.video_stream.publish(bgr_message())
            # Per-producer order holds, so the landed good frame proves both
            # bad ones were already processed.
            await eventually(lambda: "chest" in health, message="the good frame")
        panels.feed(None)
        assert len(sinks["chest"].frames) == 1
        assert capsys.readouterr().out.count("frame unusable") == 1  # latched
        assert health["chest"].source == CameraSource(robot=None, role="chest")


class StubCameras(StubTopic):
    """A camera slot module answering `members` as bound, whose subscription
    yields queued (producer, message) items."""

    def __init__(self, items, *members):
        super().__init__(items)
        self._members = members

    def bound_members(self, _runner):
        return list(self._members)


async def test_one_instance_id_on_two_core_nodes_is_two_producers(capsys):
    # Two stacks on one mesh may each run a `chest` camera. Every harness
    # mock shares the harness's core node, so the producers are scripted.
    # core-a's chest sends a bad frame, core-b's chest a good one, then
    # core-a's chest a bad one; the last frame, from a third camera, lands
    # once the rest are processed. The test checks that core-a's bad frames
    # log one "frame unusable" line.
    here = member("chest", core_node="core-a")
    there = member("chest", core_node="core-b")
    wrist = member("wrist_left", core_node="core-a")
    bad = frame_message(encoding="z16", data=b"xx")
    cameras = StubCameras(
        [
            (here.producer, bad),
            (there.producer, bgr_message()),
            (here.producer, bad),
            (wrist.producer, bgr_message()),
        ],
        here,
        there,
        wrist,
    )
    panels, _sinks, _told = recording_panels()
    panels.show_only(["chest", "wrist_left"])
    health: dict[str, LastFrame] = {}
    async with running_drain(
        lambda token: drain_frames(
            None, cameras, panels, NO_ROBOTS, token, "test", 0, health
        )
    ):
        await eventually(lambda: "wrist_left" in health, message="the last frame")
    assert capsys.readouterr().out.count("frame unusable") == 1


def test_a_failed_camera_subscribe_is_loud_and_fatal_to_the_drain(capsys):
    # A refusing runner cannot be scripted over the real wire; the stub
    # stands in for the transport failing the subscribe.
    class ExplodingTopic:
        async def subscribe(self, _runner):
            raise RuntimeError("no transport")

    async def run():
        await asyncio.wait_for(
            drain_frames(
                None,
                ExplodingTopic(),
                recording_panels()[0],
                NO_ROBOTS,
                FakeToken(),
                "camera",
            ),
            1.0,
        )

    asyncio.run(run())
    assert "subscribe failed" in capsys.readouterr().out


def test_no_signal_frame_is_drawn_and_sized():
    frame = no_signal_frame("wrist_left")
    assert frame.shape == (180, 320, 3)
    assert frame.any()  # the text actually rendered


COPYLESS = CameraSource(robot=None, role="cam")


def test_a_panel_shows_live_video_while_its_chosen_robots_camera_delivers():
    now = 100.0
    fresh = LastFrame(CameraSource("alpha", "cam"), now - 0.5)
    assert shows_live_video(fresh, "alpha", now, 0.0, 2.0)
    assert not shows_live_video(fresh, "alpha", now + 2.0, 0.0, 2.0)
    # A camera outside any copy is every robot's.
    assert shows_live_video(LastFrame(COPYLESS, now), "bravo", now, 0.0, 2.0)
    # A panel holding no frame of the chosen robot is timed from the start
    # given, however fresh another robot's frame is.
    for last in (None, fresh):
        assert shows_live_video(last, "bravo", now, now - 1.0, 2.0)
        assert not shows_live_video(last, "bravo", now, now - 3.0, 2.0)


def watched_panel(name):
    panels, sinks, _told = recording_panels()
    panels.show_only([name])
    return panels, sinks[name]


def run_watch(panels, health, chosen, steps):
    """Watch `panels` for silence, running each step in turn while it does,
    and stop one poll after the last."""
    token = FakeToken()

    async def run():
        task = asyncio.create_task(
            watch_camera_silence(
                panels, health, chosen, token, silent_after_s=0.03, poll_s=0.01
            )
        )
        for step in steps:
            await asyncio.sleep(0.09)
            step()
        await asyncio.sleep(0.02)
        token.cancel()
        await asyncio.wait_for(task, 1.0)

    asyncio.run(run())


def fed_no_signal(panels, sink, name) -> bool:
    panels.feed(None)
    return np.array_equal(sink.frames[-1], no_signal_frame(name))


def test_a_silent_camera_is_blanked_once_and_recovery_logged(capsys):
    panels, sink = watched_panel("cam")
    health = {"cam": LastFrame(COPYLESS, time.monotonic())}
    fed = []

    def resume():
        fed.append(fed_no_signal(panels, sink, "cam"))
        health["cam"] = LastFrame(COPYLESS, time.monotonic())

    run_watch(panels, health, lambda: None, [resume])
    assert fed == [True]
    out = capsys.readouterr().out
    assert out.count("went silent") == 1  # blanked once, not per poll
    assert "recovered" in out


def test_a_camera_that_never_produces_is_blanked_timed_from_its_panel_showing(
    capsys,
):
    # A dead camera and no camera look identical as a blank panel; the timer
    # starts when the panel shows so the panel says NO SIGNAL instead.
    panels, sink = watched_panel("cam")
    run_watch(panels, {}, lambda: None, [lambda: None])
    assert fed_no_signal(panels, sink, "cam")
    assert capsys.readouterr().out.count("went silent") == 1


def test_a_panel_shown_late_is_timed_from_its_showing(capsys):
    panels, sinks, _told = recording_panels()
    panels.show_only(["cam"])
    seen = []

    def show_late_panel():
        # The watcher has run longer than the silence span already.
        panels.show_only(["cam", "late"])

    def check():
        panels.feed(None)
        frames = sinks["late"].frames
        seen.append(
            bool(frames) and np.array_equal(frames[-1], no_signal_frame("late"))
        )

    token = FakeToken()

    async def run():
        task = asyncio.create_task(
            watch_camera_silence(
                panels, {}, lambda: None, token, silent_after_s=0.1, poll_s=0.01
            )
        )
        await asyncio.sleep(0.2)
        show_late_panel()
        await asyncio.sleep(0.05)
        check()
        await asyncio.sleep(0.15)
        check()
        token.cancel()
        await asyncio.wait_for(task, 1.0)

    asyncio.run(run())
    # Fed nothing until its silence span passes from its showing.
    assert seen == [False, True]


def test_a_newly_chosen_robots_silent_camera_is_blanked_once_timed_from_the_choice():
    panels, sink = watched_panel("chest")
    panels.show("chest", np.zeros((1, 1, 3), np.uint8))
    health = {"chest": LastFrame(CameraSource("alpha", "chest"), time.monotonic())}
    chosen = ["alpha"]
    seen = []
    token = FakeToken()

    async def run():
        task = asyncio.create_task(
            watch_camera_silence(
                panels,
                health,
                lambda: chosen[0],
                token,
                silent_after_s=0.2,
                poll_s=0.01,
            )
        )
        await asyncio.sleep(0.15)
        chosen[0] = "bravo"
        # Alpha's frame is old by now, but bravo is timed from its choice.
        await asyncio.sleep(0.1)
        seen.append(fed_no_signal(panels, sink, "chest"))
        await asyncio.sleep(0.2)
        seen.append(fed_no_signal(panels, sink, "chest"))
        token.cancel()
        await asyncio.wait_for(task, 1.0)

    asyncio.run(run())
    assert seen == [False, True]


def test_shrink_keeps_aspect_and_caps_width():
    tall = np.zeros((1200, 1920, 3), dtype=np.uint8)
    small = shrink_to_width(tall, 640)
    assert small.shape == (400, 640, 3)


def test_shrink_never_upscales_and_zero_disables():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert shrink_to_width(frame, 640) is frame
    assert shrink_to_width(frame, 0) is frame


async def test_a_frame_decoded_across_a_robot_change_is_dropped(monkeypatch):
    async with boot(color_cameras_instance_ids=["chest"]) as h:
        camera = h.mocks.deps.color_cameras[0]
        panels, sinks, _told = recording_panels()
        panels.show_only(["chest"])
        # The harness's camera is outside any copy, so it shows for every
        # robot; the switch is to no robot at all, which shows it still, so
        # the test flips the camera's robot under the decode instead.
        chosen = ["alpha"]
        robots = SimpleNamespace(
            chosen=lambda: chosen[0], arm_sides=sides(alpha=["left", "right"])
        )
        monkeypatch.setattr(
            "xr_commander.video.camera_source",
            lambda member: CameraSource("alpha", "chest"),
        )
        decoding = asyncio.Event()
        switched = asyncio.Event()

        def slow_decode(message, view_max_width):
            decoding.set()
            # Blocks the worker, never the loop, until the choice has moved on.
            while not switched.is_set():
                time.sleep(0.005)
            return np.zeros((1, 1, 3), np.uint8)

        monkeypatch.setattr("xr_commander.video._decode_and_shrink", slow_decode)
        async with running_drain(
            lambda token: drain_frames(
                h.node_runner, color_video_stream, panels, robots, token, "test"
            )
        ):
            await camera.video_stream.publish(bgr_message())
            await asyncio.wait_for(decoding.wait(), 10.0)
            chosen[0] = "bravo"
            switched.set()
            await asyncio.sleep(0.2)
        panels.feed("bravo")
        panels.feed("alpha")
        assert sinks["chest"].frames == []


async def test_drain_frames_routes_each_camera_into_the_panel_of_its_role():
    async with boot(color_cameras_instance_ids=["wrist_left", "chest"]) as h:
        wrist_mock, chest_mock = h.mocks.deps.color_cameras
        sinks: dict[str, RecordingSink] = {}
        panels = CameraPanels(
            make_sink=RecordingSink, show_views=sinks.update, reserved=frozenset()
        )
        panels.show_only(["wrist_left", "chest"])
        robots = SimpleNamespace(
            chosen=lambda: None, arm_sides=lambda _robot: frozenset()
        )
        async with running_drain(
            lambda token: drain_frames(
                h.node_runner, color_video_stream, panels, robots, token, "test"
            )
        ) as drain:
            # Cameras outside any copy belong to every robot, chosen or not.
            await wrist_mock.video_stream.publish(frame_message())
            await chest_mock.video_stream.publish(frame_message())
            await eventually(
                lambda: (
                    panels.feed(None)
                    or (sinks["wrist_left"].frames and sinks["chest"].frames)
                ),
                message="a frame in each panel",
            )
            assert not drain.done()
