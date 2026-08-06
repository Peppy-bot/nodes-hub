import asyncio
import os
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot_recorder import storage
from lerobot_recorder.plan import assign_camera_names, sanitize_key
from lerobot_recorder.recording import GRIPPER_LAYOUT, LinkLayout, SourceSchema
from lerobot_recorder.sink import Sink, build_features, features_mismatch
from lerobot_recorder.storage import (
    FRAME_STAGING_DIR,
    LocalTarget,
    Mirror,
    S3Location,
    S3Target,
    cleanup_staging,
    parse_storage_uri,
    prepare_target,
    probe_credentials,
    resolved_endpoint,
    staging_key,
    validate_session_name,
)
from tests.test_recording import action_entry, gripper_entry, joint_entry, make_plan


def test_parse_file_uri():
    target = parse_storage_uri("file:///data/lerobot")
    assert isinstance(target, LocalTarget)
    assert target.root == Path("/data/lerobot")


def test_parse_s3_uri_is_pure():
    # Parsing allocates nothing; the staging directory belongs to prepare.
    location = parse_storage_uri("s3://robot-datasets/openarm/sessions")
    assert location == S3Location(bucket="robot-datasets", prefix="openarm/sessions")


def test_prepare_target_allocates_stable_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    # gettempdir caches its answer; clearing it forces a re-read of TMPDIR.
    monkeypatch.setattr(tempfile, "tempdir", None)
    first = prepare_target(parse_storage_uri("s3://robot-datasets/openarm/sessions"))
    again = prepare_target(parse_storage_uri("s3://robot-datasets/openarm/sessions"))
    other = prepare_target(parse_storage_uri("s3://robot-datasets/openarm/other"))
    assert isinstance(first, S3Target)
    assert (first.bucket, first.prefix) == ("robot-datasets", "openarm/sessions")
    assert first.staging_root.exists()
    # Same URI resolves to the same root across runs: resume by name depends
    # on it. A different prefix gets its own root.
    assert again.staging_root == first.staging_root
    assert other.staging_root != first.staging_root
    assert first.staging_root.is_relative_to(tmp_path)


def test_staging_key_separates_sanitize_collisions():
    a = staging_key("bucket", "a/b")
    b = staging_key("bucket", "a_b")
    assert a != b
    # The readable part still leads for operators.
    assert a.startswith("bucket_a_b_")


def test_validate_session_name():
    validate_session_name("2026-07-27_14-03-22")
    for bad in ("", "../2026-07-27_14-03-22", "a/b", "/etc/passwd",
                "2026-1-2_3-4-5", "2026-07-27_14-03-22x", "2026-07-27 14:03:22"):
        with pytest.raises(ValueError):
            validate_session_name(bad)


def test_prepare_target_passes_local_through(tmp_path):
    local = LocalTarget(root=tmp_path)
    assert prepare_target(local) is local


@pytest.mark.parametrize("uri", ["http://x", "s3://", "file://host/x", "plainpath"])
def test_parse_rejects_bad_uris(uri):
    with pytest.raises(ValueError):
        parse_storage_uri(uri)


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


def test_cleanup_staging_only_removes_an_empty_root(tmp_path):
    empty = S3Target(bucket="b", prefix="x", staging_root=tmp_path / "empty")
    empty.staging_root.mkdir()
    cleanup_staging(empty)
    assert not empty.staging_root.exists()

    # A root holding a past session (possibly the only local copy of one
    # whose final mirror pass was cut off) must survive a failed start.
    occupied = S3Target(bucket="b", prefix="x", staging_root=tmp_path / "occupied")
    (occupied.staging_root / "2026-07-27_14-03-22").mkdir(parents=True)
    cleanup_staging(occupied)
    assert occupied.staging_root.exists()


def test_cleanup_staging_keeps_local_root(tmp_path):
    cleanup_staging(LocalTarget(root=tmp_path))
    assert tmp_path.exists()


def test_probe_credentials_local_is_noop(tmp_path):
    probe_credentials(LocalTarget(root=tmp_path))


def test_probe_credentials_fails_without_creds(tmp_path, monkeypatch):
    """The injected lookup keeps this independent of botocore's provider
    chain; ambient credentials (env, ~/.boto, IMDS) cannot flip the result.
    Staging roots are stable across runs now, so this one is redirected into
    the test's own directory rather than shared with every other run on the
    host."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    target = prepare_target(parse_storage_uri("s3://bucket/x"))
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


def test_resolved_endpoint_prefers_service_scoped_var(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://generic.example")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://scoped.example")
    assert resolved_endpoint() == "https://scoped.example"


def test_resolved_endpoint_falls_back_to_generic_then_default(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://generic.example")
    assert resolved_endpoint() == "https://generic.example"
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    assert resolved_endpoint() == "AWS default endpoints"


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
        target=S3Target(bucket="buck", prefix=prefix, staging_root=tmp_path),
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


def test_staging_refuses_a_root_it_does_not_own(tmp_path, monkeypatch):
    """The path is derivable from the storage_uri, so anyone can create it
    first; recording through someone else's directory is refused."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    location = parse_storage_uri("s3://bucket/x")
    planted = tmp_path / "lerobot_recorder_staging" / staging_key("bucket", "x")
    planted.parent.mkdir(parents=True)
    planted.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="not a directory"):
        prepare_target(location)


def test_staging_refuses_a_planted_base_directory(tmp_path, monkeypatch):
    """The base level is as plantable as the leaf: mkdir(exist_ok=True)
    succeeds through a symlink-to-directory, so a checked leaf inside an
    unchecked base would still write datasets through the attacker's link."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "lerobot_recorder_staging").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="not a directory"):
        prepare_target(parse_storage_uri("s3://bucket/x"))


def test_staging_refuses_a_root_owned_by_another_user(tmp_path, monkeypatch):
    """A symlink is not the only planting: a real directory someone else owns
    must be refused too, and the base directory is as plantable as the leaf."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    not_me = os.getuid() + 1
    monkeypatch.setattr(storage.os, "getuid", lambda: not_me)
    with pytest.raises(ValueError, match="belongs to another user"):
        prepare_target(parse_storage_uri("s3://bucket/x"))


def test_staging_directories_are_private(tmp_path, monkeypatch):
    """0o700 on both levels: the base sits in a shared /tmp, and a
    group-readable staging tree would hand other users the datasets."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    target = prepare_target(parse_storage_uri("s3://bucket/x"))
    for directory in (target.staging_root, target.staging_root.parent):
        assert stat.S_IMODE(directory.lstat().st_mode) == 0o700, directory
