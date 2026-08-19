import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot_recorder.plan import assign_camera_names, sanitize_key
from lerobot_recorder.recording import GRIPPER_LAYOUT, LinkLayout, SourceSchema
from lerobot_recorder.sink import (
    Sink,
    build_features,
    features_mismatch,
    missing_video_frames,
)
from lerobot_recorder.storage import (
    FRAME_STAGING_DIR,
    Mirror,
    S3Destination,
    StorageTarget,
    ensure_mounted,
    parse_storage,
    probe_credentials,
    resolved_endpoint,
    validate_session_name,
)
from tests.test_recording import action_entry, gripper_entry, joint_entry, make_plan


def test_parse_local_only():
    target = parse_storage("/data/lerobot", "")
    assert target == StorageTarget(root=Path("/data/lerobot"), s3=None)


def test_parse_with_s3_mirror():
    target = parse_storage("/data/lerobot", "s3://robot-datasets/openarm/sessions")
    assert target.root == Path("/data/lerobot")
    assert target.s3 == S3Destination(bucket="robot-datasets", prefix="openarm/sessions")


def test_ensure_mounted_requires_the_bound_directory(tmp_path):
    """A missing root means the manifest's bind mount is not in effect;
    creating it would write datasets into the container's own filesystem,
    gone when the instance stops."""
    ensure_mounted(StorageTarget(root=tmp_path, s3=None))
    with pytest.raises(ValueError, match="mounted"):
        ensure_mounted(StorageTarget(root=tmp_path / "missing", s3=None))


def test_validate_session_name():
    validate_session_name("2026-07-27_14-03-22")
    for bad in ("", "../2026-07-27_14-03-22", "a/b", "/etc/passwd",
                "2026-1-2_3-4-5", "2026-07-27_14-03-22x", "2026-07-27 14:03:22"):
        with pytest.raises(ValueError):
            validate_session_name(bad)


@pytest.mark.parametrize(
    ("root", "s3_uri"),
    [
        ("relative/path", ""),
        ("", ""),
        ("/data", "http://x"),
        ("/data", "s3://"),
        ("/data", "plainpath"),
    ],
)
def test_parse_rejects_bad_input(root, s3_uri):
    with pytest.raises(ValueError):
        parse_storage(root, s3_uri)


def test_sanitize_key():
    assert sanitize_key("Left-Arm 01") == "left_arm_01"
    assert sanitize_key("---") == "___"


def test_assign_camera_names_reserves_derived_depth_keys():
    used = set()
    color = assign_camera_names(used, ["wrist_depth"], with_depth=False)
    rgbd = assign_camera_names(used, ["wrist"], with_depth=True)
    assert color == ["wrist_depth"]
    assert rgbd == ["wrist_cam"]
    keys = color + rgbd + [f"{n}_depth" for n in rgbd]
    assert len(keys) == len(set(keys))


def test_assign_camera_names_two_rgbd_collision():
    names = assign_camera_names(set(), ["cam", "cam_depth"], with_depth=True)
    assert names[0] == "cam"
    keys = names + [f"{n}_depth" for n in names]
    assert len(keys) == len(set(keys))


def test_probe_credentials_without_mirror_is_noop(tmp_path):
    probe_credentials(StorageTarget(root=tmp_path, s3=None))


def test_probe_credentials_fails_without_creds(tmp_path):
    """The injected lookup keeps this independent of botocore's provider
    chain; ambient credentials (env, ~/.boto, IMDS) cannot flip the result."""
    target = parse_storage(str(tmp_path), "s3://bucket/x")
    with pytest.raises(RuntimeError, match="credentials"):
        probe_credentials(target, lookup=lambda: None)
    probe_credentials(target, lookup=lambda: object())


def test_build_features_shapes():
    plan = make_plan(
        state=(joint_entry(), gripper_entry()),
        action=(action_entry(),),
        color=1,
        rgbd=1,
    )
    schema = SourceSchema(
        layouts=(
            LinkLayout(dims=7, has_velocities=True, has_efforts=False),
            GRIPPER_LAYOUT,
        ),
        action_layouts=(LinkLayout(dims=7, has_velocities=True, has_efforts=False),),
        color_geometry=((640, 480),),
        rgbd_geometry=((640, 480),),
        depth_geometry=((320, 240),),
        depth_units=(0.001,),
    )
    features = build_features(schema, plan)
    assert features["observation.state"]["shape"] == (8,)
    assert features["observation.state"]["names"][-1] == "grip0_opening"
    assert features["action"]["shape"] == (7,)
    assert features["action"]["names"][0] == "arm0_j0"
    # Optional vectors cover exactly the sources whose wire delivers them.
    assert features["observation.velocities"]["shape"] == (7,)
    assert features["observation.efforts"]["names"] == ["grip0_effort"]
    assert features["observation.images.cam0"]["shape"] == (480, 640, 3)
    assert features["observation.images.rgbd0"]["shape"] == (480, 640, 3)
    depth = features["observation.images.rgbd0_depth"]
    assert depth["shape"] == (240, 320, 1)
    assert depth["info"] == {"is_depth_map": True}


def test_resolved_endpoint_prefers_service_scoped_var():
    both = {
        "AWS_ENDPOINT_URL": "https://generic.example",
        "AWS_ENDPOINT_URL_S3": "https://scoped.example",
    }
    assert resolved_endpoint(both) == "https://scoped.example"


def test_resolved_endpoint_falls_back_to_generic_then_default():
    assert (
        resolved_endpoint({"AWS_ENDPOINT_URL": "https://generic.example"})
        == "https://generic.example"
    )
    assert resolved_endpoint({}) == "AWS default endpoints"


class StubS3:
    """Records uploads the way boto3's client would perform them."""

    def __init__(self, vanish: set[str] = frozenset()):
        self.uploads: list[tuple[str, str]] = []
        self._vanish = vanish

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        if key in self._vanish:
            raise FileNotFoundError(filename)
        self.uploads.append((bucket, key))


def mirror_over(tmp_path, client, prefix="pre") -> Mirror:
    mirror = Mirror(
        dest=S3Destination(bucket="buck", prefix=prefix),
        session_dir=tmp_path,
        session_name="sess",
    )
    mirror._client = client
    return mirror


def test_mirror_uploads_new_files_under_prefix_and_session(tmp_path):
    (tmp_path / "dataset" / "meta").mkdir(parents=True)
    (tmp_path / "dataset" / "meta" / "info.json").write_text("{}")
    client = StubS3()
    mirror_over(tmp_path, client).sync()
    assert client.uploads == [("buck", "pre/sess/dataset/meta/info.json")]


def test_mirror_omits_the_prefix_when_the_uri_has_none(tmp_path):
    (tmp_path / "info.json").write_text("{}")
    client = StubS3()
    mirror_over(tmp_path, client, prefix="").sync()
    assert client.uploads == [("buck", "sess/info.json")]


def test_mirror_reuploads_only_what_changed(tmp_path):
    stable = tmp_path / "stable.json"
    stable.write_text("{}")
    growing = tmp_path / "grows.parquet"
    growing.write_text("a")
    client = StubS3()
    mirror = mirror_over(tmp_path, client)
    mirror.sync()
    assert len(client.uploads) == 2

    growing.write_text("aa")
    client.uploads.clear()
    mirror.sync()
    assert client.uploads == [("buck", "pre/sess/grows.parquet")]


def test_mirror_skips_the_frame_staging_tree(tmp_path):
    staging = tmp_path / "dataset" / FRAME_STAGING_DIR / "cam0" / "episode-000000"
    staging.mkdir(parents=True)
    (staging / "frame-000000.png").write_bytes(b"png")
    (tmp_path / "dataset" / "info.json").write_text("{}")
    client = StubS3()
    mirror_over(tmp_path, client).sync()
    # Staged frames are deleted once encoded, so mirroring them would upload
    # garbage keys that nothing ever removes.
    assert client.uploads == [("buck", "pre/sess/dataset/info.json")]


def test_mirror_survives_a_file_deleted_mid_pass(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "z.json").write_text("{}")
    client = StubS3(vanish={"pre/sess/a.json"})
    mirror = mirror_over(tmp_path, client)
    mirror.sync()
    assert client.uploads == [("buck", "pre/sess/z.json")]
    # The vanished file is not marked uploaded, so a later sync retries it.
    client._vanish = frozenset()
    client.uploads.clear()
    mirror.sync()
    assert client.uploads == [("buck", "pre/sess/a.json")]


def _feature(names):
    return {"dtype": "float32", "shape": (len(names),), "names": names}


def test_features_mismatch_identical_is_none():
    live = {"observation.state": _feature(["j0"]), "action": _feature(["j0"])}
    disk = {
        **{k: dict(v) for k, v in live.items()},
        # The library's bookkeeping features exist on disk only.
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    # JSON round-trips shapes to lists; still equal.
    disk["action"]["shape"] = [1]
    # Per-feature info blobs are ignored, matching lerobot's own check.
    disk["action"]["info"] = {"video.codec": "libsvtav1"}
    assert features_mismatch(disk, live) is None


def test_features_mismatch_names_the_first_difference():
    live = {"observation.state": _feature(["j0", "j1"])}
    disk = {"observation.state": _feature(["j0"])}
    assert "shape" in features_mismatch(disk, live)

    disk = {"observation.state": _feature(["j0", "j1"]), "observation.images.cam0": _feature(["x"])}
    assert "do not provide" in features_mismatch(disk, live)

    disk = {}
    assert "does not record" in features_mismatch(disk, live)


def test_a_failed_finalize_can_be_retried(tmp_path):
    """Latching before the close would turn a failed close into a permanent
    no-op reporting footer-less parquet as a finished session."""
    sink = Sink(
        root=tmp_path / "dataset",
        repo_id="bot/session",
        robot_type="bot",
        fps=30,
        image_writer_threads=1,
        streaming_encoding=True,
    )
    closes = {"n": 0}

    def close():
        closes["n"] += 1
        if closes["n"] == 1:
            raise OSError("disk full")

    sink._dataset = SimpleNamespace(finalize=close)
    with pytest.raises(OSError):
        asyncio.run(sink.finalize())
    asyncio.run(sink.finalize())
    asyncio.run(sink.finalize())
    assert closes["n"] == 2, "retried after the failure, latched after the success"


def episode_metadata(rows: int, spans: dict[str, tuple[float, float]]) -> dict:
    """An episode's metadata row in the shape the library buffers it: every
    value a single-element list."""
    meta = {"length": [rows]}
    for key, (start, end) in spans.items():
        meta[f"videos/{key}/from_timestamp"] = [start]
        meta[f"videos/{key}/to_timestamp"] = [end]
    return meta


def test_missing_video_frames_accepts_a_whole_episode():
    meta = episode_metadata(30, {"cam0": (0.0, 1.0)})
    assert missing_video_frames(meta, ["cam0"], 30) == {}


def test_missing_video_frames_counts_the_shortfall():
    """27 frames encoded against the 30 rows written."""
    meta = episode_metadata(30, {"cam0": (0.0, 0.9)})
    assert missing_video_frames(meta, ["cam0"], 30) == {"cam0": 3}


def test_missing_video_frames_measures_the_span_not_the_end():
    """Every episode after the first starts partway into its chunk file, so
    reading the end timestamp alone would call a whole episode short."""
    meta = episode_metadata(30, {"cam0": (14.6, 15.6)})
    assert missing_video_frames(meta, ["cam0"], 30) == {}


def test_missing_video_frames_reports_each_camera_separately():
    meta = episode_metadata(30, {"cam0": (2.0, 3.0), "cam1": (2.0, 2.8)})
    assert missing_video_frames(meta, ["cam0", "cam1"], 30) == {"cam1": 6}


def test_missing_video_frames_counts_an_absent_video_as_wholly_missing():
    """A camera the save left no video metadata for is the same defect in its
    most complete form, not a camera to skip."""
    meta = episode_metadata(30, {"cam0": (0.0, 1.0)})
    assert missing_video_frames(meta, ["cam0", "cam1"], 30) == {"cam1": 30}
