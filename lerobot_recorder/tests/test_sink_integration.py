"""End-to-end against the real lerobot library: create a dataset from a
discovered-style schema, write frames, save the episode, and load it back.
Needs lerobot + ffmpeg; skipped where they are absent."""

import asyncio
import importlib

import numpy as np
import pytest

from lerobot_recorder.recording import FrameRow, LinkLayout, SourceSchema
from lerobot_recorder.sink import Sink
from tests.test_recording import action_entry, joint_entry, make_plan

lerobot = pytest.importorskip("lerobot")


def _video_decode_available() -> bool:
    try:
        importlib.import_module("torchcodec.decoders")
    except Exception:
        return False
    return True


requires_video_decode = pytest.mark.skipif(
    not _video_decode_available(),
    reason="torchcodec cannot load its ffmpeg libraries on this host",
)

FPS = 10
FRAMES = 10
W, H = 32, 16
ARM_LAYOUT = LinkLayout(dims=7, has_velocities=True, has_efforts=False)


def schema_with_camera() -> SourceSchema:
    return SourceSchema(
        layouts=(ARM_LAYOUT,),
        action_layouts=(ARM_LAYOUT,),
        color_geometry=((W, H),),
        rgbd_geometry=(),
        depth_geometry=(),
        depth_units=(),
    )


# Keeps every recorded action clear of every recorded state.
ACTION_OFFSET = 100.0


def camera_plan():
    return make_plan(state=(joint_entry(),), action=(action_entry(),), color=1)


def row(i: int) -> FrameRow:
    image = np.full((H, W, 3), (i * 20) % 255, dtype=np.uint8)
    # State and action carry disjoint ranges so a feature that ends up holding
    # the other one's values is visible on reload.
    return FrameRow(
        state=np.linspace(0, 1, 7, dtype=np.float32) + i,
        action=np.linspace(0, 1, 7, dtype=np.float32) + i + ACTION_OFFSET,
        velocities=np.zeros(7, dtype=np.float32),
        efforts=None,
        images={"cam0": image},
    )


@requires_video_decode
def test_record_two_episodes_finalize_reload(tmp_path):
    plan = camera_plan()
    sink = Sink(root=tmp_path / "dataset", repo_id="test/session", robot_type="bot", fps=FPS, image_writer_threads=2)

    async def record():
        await sink.create(schema_with_camera(), plan)
        for _ in range(2):
            for i in range(FRAMES):
                sink.add_frame(row(i), task="smoke")
            await sink.save_episode()
        # Once per session: parquet footers are written here.
        await sink.finalize()

    asyncio.run(record())
    assert sink.episodes_saved == 2

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("test/session", root=str(tmp_path / "dataset"))
    assert ds.meta.total_episodes == 2
    assert len(ds) == 2 * FRAMES
    item = ds[0]
    assert item["observation.state"].shape == (7,)
    assert ds.meta.features["action"]["shape"] == (7,)
    assert "observation.velocities" in ds.meta.features
    assert tuple(item["observation.images.cam0"].shape) in {(3, H, W), (H, W, 3)}
    assert ds.meta.tasks is not None


def test_add_frame_refused_after_finalize(tmp_path):
    plan = camera_plan()
    sink = Sink(root=tmp_path / "dataset", repo_id="test/session", robot_type="bot", fps=FPS, image_writer_threads=2)

    async def run():
        await sink.create(schema_with_camera(), plan)
        for i in range(FRAMES):
            sink.add_frame(row(i), task="smoke")
        await sink.save_episode()
        await sink.finalize()

    asyncio.run(run())
    with pytest.raises(AssertionError, match="finalized"):
        sink.add_frame(row(0), task="smoke")


def test_failed_save_recovers_via_discard(tmp_path):
    """The library's episode buffer is unusable after a failed save;
    discard_open_frames must bring the sink back for the next episode."""
    plan = camera_plan()
    sink = Sink(root=tmp_path / "dataset", repo_id="test/session", robot_type="bot", fps=FPS, image_writer_threads=2)

    async def run():
        await sink.create(schema_with_camera(), plan)
        for i in range(FRAMES):
            sink.add_frame(row(i), task="smoke")

        def boom():
            raise OSError("simulated encode failure")

        real_save = sink._dataset.save_episode
        sink._dataset.save_episode = boom
        with pytest.raises(OSError):
            await sink.save_episode()
        sink._dataset.save_episode = real_save

        sink.discard_open_frames()
        for i in range(FRAMES):
            sink.add_frame(row(i), task="smoke")
        await sink.save_episode()
        await sink.finalize()

    asyncio.run(run())
    assert sink.episodes_saved == 1

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("test/session", root=str(tmp_path / "dataset"))
    assert ds.meta.total_episodes == 1
    assert len(ds) == FRAMES


def test_zero_frame_episode_discards(tmp_path):
    plan = make_plan(state=(joint_entry(),))
    sink = Sink(root=tmp_path / "dataset", repo_id="test/session", robot_type="bot", fps=FPS, image_writer_threads=2)
    schema = SourceSchema(
        layouts=(ARM_LAYOUT,),
        action_layouts=(),
        color_geometry=(),
        rgbd_geometry=(),
        depth_geometry=(),
        depth_units=(),
    )

    async def run():
        await sink.create(schema, plan)
        sink.discard_open_frames()

    asyncio.run(run())
    assert sink.episodes_saved == 0


async def _resume_now(sink: Sink, schema, plan) -> None:
    """Resume reads its manifest through the preflight, so tests do too."""
    await sink.resume(await sink.preflight_resume(), schema, plan)


def _sink(tmp_path) -> Sink:
    return Sink(
        root=tmp_path / "dataset",
        repo_id="test/session",
        robot_type="bot",
        fps=FPS,
        image_writer_threads=2,
    )


def _record_one_episode_and_finalize(sink: Sink) -> None:
    async def run():
        await sink.create(schema_with_camera(), camera_plan())
        for i in range(FRAMES):
            sink.add_frame(row(i), task="smoke")
        await sink.save_episode()
        await sink.finalize()

    asyncio.run(run())


def test_resume_appends_to_a_finalized_session(tmp_path):
    """The core resume proof: record, finalize, reopen with a fresh Sink,
    record again, finalize again, and load both episodes back."""
    first = _sink(tmp_path)
    _record_one_episode_and_finalize(first)

    resumed = _sink(tmp_path)

    async def resume_and_record():
        manifest = await resumed.preflight_resume()
        assert manifest.total_episodes == 1
        await resumed.resume(manifest, schema_with_camera(), camera_plan())
        assert resumed.episodes_saved == 1
        # The private buffer-size knob resume() cannot set; pinned so an
        # upstream change fails loudly instead of silently reverting to 10.
        assert resumed._dataset.meta._metadata_buffer_size == 1
        for i in range(FRAMES):
            resumed.add_frame(row(i), task="smoke2")
        await resumed.save_episode()
        await resumed.finalize()

    asyncio.run(resume_and_record())
    assert resumed.episodes_saved == 2

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("test/session", root=str(tmp_path / "dataset"), video_backend="pyav")
    assert ds.meta.total_episodes == 2
    assert len(ds) == 2 * FRAMES
    # Frames from both the original and the resumed episode decode (pyav
    # backend: decode must work wherever encode did).
    assert ds[0]["observation.state"].shape == (7,)
    assert ds[len(ds) - 1]["observation.state"].shape == (7,)
    # Each feature holds its own vector, not the other's.
    assert float(ds[0]["observation.state"][0]) < ACTION_OFFSET
    assert float(ds[0]["action"][0]) >= ACTION_OFFSET


def test_resume_refuses_a_session_without_episodes(tmp_path, monkeypatch):
    """A zero-episode dataset must be refused BEFORE the library's metadata
    loader runs: its missing-table fallback is a Hugging Face Hub download."""
    sink = _sink(tmp_path)

    async def create_only():
        await sink.create(schema_with_camera(), camera_plan())
        await sink.finalize()

    asyncio.run(create_only())

    def explode(*args, **kwargs):
        raise AssertionError("resume preflight must not reach the HF Hub")

    monkeypatch.setattr("huggingface_hub.snapshot_download", explode)
    fresh = _sink(tmp_path)
    with pytest.raises(ValueError, match="no saved episodes"):
        asyncio.run(fresh.preflight_resume())


def test_resume_refuses_disagreeing_metadata(tmp_path):
    import json

    first = _sink(tmp_path)
    _record_one_episode_and_finalize(first)
    info_path = tmp_path / "dataset" / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] += 1
    info_path.write_text(json.dumps(info))

    fresh = _sink(tmp_path)
    with pytest.raises(ValueError, match=r"disagrees with info\.json"):
        asyncio.run(fresh.preflight_resume())


def test_resume_refuses_an_unusable_fps(tmp_path):
    """A parseable info.json without a usable fps must still refuse with a
    named reason, not escape as a KeyError."""
    import json

    first = _sink(tmp_path)
    _record_one_episode_and_finalize(first)
    info_path = tmp_path / "dataset" / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["fps"]
    info_path.write_text(json.dumps(info))

    fresh = _sink(tmp_path)
    with pytest.raises(ValueError, match="no usable fps"):
        asyncio.run(fresh.preflight_resume())


def test_resume_clears_stale_frame_staging(tmp_path):
    first = _sink(tmp_path)
    _record_one_episode_and_finalize(first)
    # An unclean death leaves the next episode's staged frames behind; they
    # would be mixed into that episode's video on the next save.
    stale = tmp_path / "dataset" / "images" / "observation.images.cam0" / "episode-000001"
    stale.mkdir(parents=True)
    (stale / "frame-000000.png").write_bytes(b"png")

    resumed = _sink(tmp_path)

    async def resume_then_close():
        await resumed.resume(await resumed.preflight_resume(), schema_with_camera(), camera_plan())
        cleared = not stale.exists()
        # Stop the resumed dataset's image-writer threads; they are
        # non-daemon and would otherwise outlive the test.
        await resumed.finalize()
        return cleared

    assert asyncio.run(resume_then_close())


def test_resume_removes_orphaned_encoder_scratch(tmp_path):
    """A hard stop mid-episode leaves the streaming encoder's scratch
    directory behind. Nothing in the library sweeps it, so resume does: it
    belongs to an episode that never saved, and the s3 mirror would upload it."""
    first = _sink(tmp_path)
    _record_one_episode_and_finalize(first)
    orphan = tmp_path / "dataset" / "tmpdeadbeef"
    orphan.mkdir(parents=True)
    (orphan / "observation.images.cam0_streaming.mp4").write_bytes(b"partial")
    # A scratch directory without an encoder file is not ours to delete.
    bystander = tmp_path / "dataset" / "tmpsomethingelse"
    bystander.mkdir(parents=True)
    (bystander / "notes.txt").write_bytes(b"keep me")

    resumed = _sink(tmp_path)

    async def resume_then_close():
        await resumed.resume(await resumed.preflight_resume(), schema_with_camera(), camera_plan())
        swept = not orphan.exists()
        await resumed.finalize()
        return swept

    assert asyncio.run(resume_then_close())
    assert bystander.exists(), "only the encoder's own scratch is swept"


def test_save_episode_reports_dropped_video_frames(tmp_path):
    """A full encoder queue drops video frames while the tabular row is still
    written, so the episode ends with fewer video frames than state rows. The
    save has to say so; nothing else in the dataset records it."""
    sink = _sink(tmp_path)

    async def record_with_a_drop():
        await sink.create(schema_with_camera(), camera_plan())
        for i in range(FRAMES):
            sink.add_frame(row(i), task="smoke")
        encoder = sink._dataset.writer._streaming_encoder
        assert encoder is not None, "the default sink streams"
        encoder._dropped_frames["observation.images.cam0"] = 3
        dropped = await sink.save_episode()
        await sink.finalize()
        return dropped

    reported = asyncio.run(record_with_a_drop())
    assert reported is not None
    assert "observation.images.cam0 3" in reported
    assert "should not be trained on" in reported


def test_save_episode_reports_nothing_when_no_frames_are_dropped(tmp_path):
    sink = _sink(tmp_path)

    async def record_clean():
        await sink.create(schema_with_camera(), camera_plan())
        for i in range(FRAMES):
            sink.add_frame(row(i), task="smoke")
        dropped = await sink.save_episode()
        await sink.finalize()
        return dropped

    assert asyncio.run(record_clean()) is None


def test_staged_encoding_has_no_streaming_encoder(tmp_path):
    """Turning the option off puts the session back on the staged-PNG path,
    whose writer queue is unbounded and therefore cannot drop a frame."""
    sink = Sink(
        root=tmp_path / "dataset",
        repo_id="test/session",
        robot_type="bot",
        fps=FPS,
        image_writer_threads=2,
        streaming_encoding=False,
    )

    async def record_staged():
        await sink.create(schema_with_camera(), camera_plan())
        for i in range(FRAMES):
            sink.add_frame(row(i), task="smoke")
        staged = sink._dataset.writer._streaming_encoder is None
        dropped = await sink.save_episode()
        await sink.finalize()
        return staged, dropped

    staged, dropped = asyncio.run(record_staged())
    assert staged
    assert dropped is None


def test_resume_refuses_incompatible_live_sources(tmp_path):
    first = _sink(tmp_path)
    _record_one_episode_and_finalize(first)

    wrong_geometry = SourceSchema(
        layouts=(ARM_LAYOUT,),
        action_layouts=(ARM_LAYOUT,),
        color_geometry=((W * 2, H),),
        rgbd_geometry=(),
        depth_geometry=(),
        depth_units=(),
    )
    fresh = _sink(tmp_path)
    with pytest.raises(ValueError, match=r"observation\.images\.cam0 shape"):
        asyncio.run(_resume_now(fresh, wrong_geometry, camera_plan()))
    assert not fresh.created

    wrong_robot = Sink(
        root=tmp_path / "dataset", repo_id="test/session",
        robot_type="other-bot", fps=FPS, image_writer_threads=2,
    )
    with pytest.raises(ValueError, match="robot_type"):
        asyncio.run(_resume_now(wrong_robot, schema_with_camera(), camera_plan()))

    wrong_fps = Sink(
        root=tmp_path / "dataset", repo_id="test/session",
        robot_type="bot", fps=FPS * 2, image_writer_threads=2,
    )
    with pytest.raises(ValueError, match="fps"):
        asyncio.run(_resume_now(wrong_fps, schema_with_camera(), camera_plan()))


def test_dataset_added_features_mirror_upstream():
    """DATASET_ADDED_FEATURES is a hand copy of lerobot's DEFAULT_FEATURES
    keys (importing the real one drags torch into unit tests); fail loudly
    if upstream drifts."""
    from lerobot.utils.constants import DEFAULT_FEATURES

    from lerobot_recorder.sink import DATASET_ADDED_FEATURES

    assert set(DEFAULT_FEATURES.keys()) == DATASET_ADDED_FEATURES
