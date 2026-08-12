"""The LeRobot dataset this session writes, via the official library.

The dataset is created once every bound source has produced (the feature
schema needs each source's first message) and lives for the session. Blocking
library calls (create, save) run in worker threads so the goal loop stays
responsive; add_frame runs in the per-frame worker beside the image decode,
with image encoding handled by the library's writer threads.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import recording
from .plan import RecordingPlan
from .recording import FrameRow, SourceSchema

# The bookkeeping features lerobot's create() merges into every dataset's
# info.json beside ours. Mirrored here instead of importing the library's
# DEFAULT_FEATURES, which would pull torch into every unit test; the
# integration suite pins this copy against upstream drift.
DATASET_ADDED_FEATURES = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)


def build_features(schema: SourceSchema, plan: RecordingPlan) -> dict:
    features: dict = {
        "observation.state": _vector_feature(list(recording.state_dim_names(plan, schema))),
        "action": _vector_feature(list(recording.action_dim_names(plan, schema))),
    }
    # Optional vectors cover exactly the sources whose wire delivers them,
    # named per dimension so the subset is self-describing.
    velocities = recording.velocities_dim_names(plan, schema)
    if velocities:
        features["observation.velocities"] = _vector_feature(list(velocities))
    efforts = recording.efforts_dim_names(plan, schema)
    if efforts:
        features["observation.efforts"] = _vector_feature(list(efforts))
    for entry, (w, h) in zip(plan.color_cameras, schema.color_geometry, strict=True):
        features[f"observation.images.{entry.name}"] = _video_feature(w, h)
    for i, entry in enumerate(plan.rgbd_cameras):
        w, h = schema.rgbd_geometry[i]
        features[f"observation.images.{entry.name}"] = _video_feature(w, h)
        dw, dh = schema.depth_geometry[i]
        features[f"observation.images.{entry.name}_depth"] = {
            "dtype": "video",
            "shape": (dh, dw, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        }
    return features


def features_mismatch(on_disk: dict, live: dict) -> str | None:
    """Why a dataset's recorded features cannot take frames from the live
    sources, or None when they can. Mirrors lerobot's own resume sanity
    check: per-feature `info` blobs are ignored, as are the library's
    bookkeeping features on the disk side."""
    disk = {k: v for k, v in on_disk.items() if k not in DATASET_ADDED_FEATURES}
    if unprovided := sorted(disk.keys() - live.keys()):
        return f"dataset records {unprovided[0]}; the live sources do not provide it"
    if unrecorded := sorted(live.keys() - disk.keys()):
        return f"live sources provide {unrecorded[0]}; the dataset does not record it"
    for key, want in sorted(disk.items()):
        have = live[key]
        for field in ("dtype", "shape", "names"):
            recorded = _normalized(want.get(field))
            offered = _normalized(have.get(field))
            if recorded != offered:
                return f"{key} {field} is {recorded!r} in the dataset, live sources offer {offered!r}"
    return None


def _normalized(value):
    """JSON round-trips tuples to lists; compare shape-like fields as lists."""
    return list(value) if isinstance(value, tuple) else value


@dataclass(frozen=True)
class ResumeManifest:
    robot_type: str | None
    fps: int
    features: dict
    total_episodes: int


def read_resume_manifest(dataset_root: Path) -> ResumeManifest:
    """Everything resume must know about a session's dataset, read without
    constructing it. Raises ValueError with an operator-grade reason when the
    session cannot be resumed. Blocking; run in a worker thread.

    The zero-episode check matters beyond a nicer message: the library's
    metadata loader reacts to a missing episodes table by downloading meta/
    from the Hugging Face Hub, so resuming an empty session would otherwise
    become a network call ending in an HF error."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError("session has no dataset")
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"dataset info.json unreadable: {e}") from e

    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise ValueError("session has no saved episodes; nothing to resume")
    try:
        import pyarrow.parquet as pq

        table_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in episode_files)
    except Exception as e:
        raise ValueError(f"episodes metadata unreadable (unclean stop?): {e}") from e
    total = int(info.get("total_episodes", 0))
    if table_rows != total or total < 1:
        raise ValueError(
            f"episode metadata disagrees with info.json (table {table_rows} vs "
            f"info {total}); the session needs manual repair"
        )

    try:
        fps = int(info["fps"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"dataset info.json has no usable fps: {e!r}") from e

    _warn_on_unreadable_tail(dataset_root)
    return ResumeManifest(
        robot_type=info.get("robot_type"),
        fps=fps,
        features=info.get("features", {}),
        total_episodes=total,
    )


def _warn_on_unreadable_tail(dataset_root: Path) -> None:
    """An unclean stop leaves the open data chunk without its footer; those
    rows are already lost and stay lost after a resume. Resumable regardless
    (the library appends in fresh chunk files), so this only warns."""
    chunks = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not chunks:
        return
    try:
        import pyarrow.parquet as pq

        _ = pq.ParquetFile(chunks[-1]).metadata
    except Exception:
        print(
            "[recorder] resuming after an unclean stop; episodes since the "
            "last chunk rotation may be unreadable",
            flush=True,
        )


def _remove_orphaned_encoder_dirs(dataset_root: Path) -> None:
    """Drop the per-episode encoder scratch directories a hard stop leaves
    behind. The streaming encoder writes the open episode into a mkdtemp
    directory under the dataset root and moves it into the chunk file at save,
    so any that survive belong to an episode that never saved: dead weight the
    library's own interrupted-episode cleanup (staged images only) does not
    cover, and bytes the s3 mirror would otherwise upload."""
    for path in sorted(dataset_root.glob("tmp*")):
        if not path.is_dir() or not any(path.glob("*_streaming.mp4")):
            continue
        try:
            shutil.rmtree(path)
            print(f"[recorder] removed orphaned encoder scratch {path.name}", flush=True)
        except OSError as e:
            print(f"[recorder] could not remove {path}: {e!r}", flush=True)


def missing_video_frames(latest_episode: dict, video_keys: list[str], fps: int) -> dict[str, int]:
    """Per camera, how many of the episode's rows ended up with no video frame.

    The encoder drops frames it cannot keep up with while the tabular row for
    each one is still written, so an episode can end holding a video shorter
    than its own state rows. The library records both numbers as it saves:
    `length` counts the rows, and the video timestamps span the encoded file,
    measured off the mp4 rather than derived from the row count. They disagree
    exactly when frames went missing, whatever the cause, a full encoder queue
    and a dead encoder thread alike.

    Values arrive as single-element lists, the shape the library buffers
    episode metadata in."""
    rows = int(latest_episode["length"][0])
    covered = ((key, _encoded_frame_count(latest_episode, key, fps)) for key in video_keys)
    return {key: rows - frames for key, frames in covered if frames < rows}


def _encoded_frame_count(latest_episode: dict, video_key: str, fps: int) -> int:
    """Frames this episode contributed to its camera's video file. Zero when
    the save left no video metadata at all for the camera, which is the same
    defect in its most complete form."""
    start = latest_episode.get(f"videos/{video_key}/from_timestamp")
    end = latest_episode.get(f"videos/{video_key}/to_timestamp")
    if start is None or end is None:
        return 0
    # Exact: the encoder timestamps frame n at n/fps, so the span a whole
    # episode covers is its frame count over fps.
    return round((end[0] - start[0]) * fps)


def _vector_feature(names: list[str]) -> dict:
    return {"dtype": "float32", "shape": (len(names),), "names": names}


def _video_feature(width: int, height: int) -> dict:
    return {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }


class Sink:
    def __init__(
        self,
        root: Path,
        repo_id: str,
        robot_type: str,
        fps: int,
        image_writer_threads: int,
        streaming_encoding: bool = True,
    ):
        self._root = root
        self._repo_id = repo_id
        self._robot_type = robot_type
        self._fps = fps
        self._image_writer_threads = image_writer_threads
        self._streaming_encoding = streaming_encoding
        self._dataset = None
        self._finalized = False

    @property
    def created(self) -> bool:
        return self._dataset is not None

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def episodes_saved(self) -> int:
        assert self._dataset is not None, "created at the first accepted goal"
        return int(self._dataset.meta.total_episodes)

    async def create(self, schema: SourceSchema, plan: RecordingPlan) -> None:
        assert self._dataset is None, "one dataset per session"
        features = build_features(schema, plan)

        def _create():
            # lerobot pulls in torch; deferred so module import stays cheap
            # (node startup, unit tests) and the cost lands in this worker.
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            return LeRobotDataset.create(
                repo_id=self._repo_id,
                fps=self._fps,
                features=features,
                root=str(self._root),
                robot_type=self._robot_type,
                use_videos=True,
                image_writer_threads=self._image_writer_threads,
                # Metadata rows flush to the writer per save instead of
                # buffering. Parquet footers still land only at finalize, so
                # an unclean death loses the tabular data written since the
                # last chunk rotation; the videos survive.
                metadata_buffer_size=1,
                streaming_encoding=self._streaming_encoding,
            )

        self._dataset = await asyncio.to_thread(_create)

    async def preflight_resume(self) -> ResumeManifest:
        """The cheap disk-only checks, exposed so a caller can refuse a bad
        session before gating on live sources. Returns the manifest to hand
        back to `resume`, so the session is read once; raises ValueError with
        the refusal reason."""
        return await asyncio.to_thread(read_resume_manifest, self._root)

    async def resume(
        self, manifest: ResumeManifest, schema: SourceSchema, plan: RecordingPlan
    ) -> None:
        """Reopen this root's existing dataset so new episodes append to it,
        lerobot's record -> finalize -> resume lifecycle. The compatibility
        gate runs against the on-disk metadata BEFORE construction, so an
        incompatible dataset instance never exists."""
        assert self._dataset is None, "one dataset per session"
        features = build_features(schema, plan)

        def _resume():
            if manifest.robot_type != self._robot_type:
                raise ValueError(
                    f"dataset robot_type is {manifest.robot_type!r}, "
                    f"this recorder runs {self._robot_type!r}"
                )
            if manifest.fps != self._fps:
                raise ValueError(f"dataset fps is {manifest.fps}, this recorder runs {self._fps}")
            mismatch = features_mismatch(manifest.features, features)
            if mismatch is not None:
                raise ValueError(mismatch)

            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset.resume(
                repo_id=self._repo_id,
                root=str(self._root),
                image_writer_threads=self._image_writer_threads,
                # Same encoder mode as `create`: resume() defaults to the
                # staged-PNG path, so omitting this would silently make a
                # resumed session slower than the one it continues.
                streaming_encoding=self._streaming_encoding,
            )
            # resume() exposes no metadata_buffer_size knob and defaults to
            # buffering 10 episodes; create() flushes per save so info.json
            # and the episodes table cannot drift apart across a kill. Keep
            # the resumed session on the same contract.
            dataset.meta._metadata_buffer_size = 1
            # An unclean death leaves the next episode's staged frame images
            # behind; recording into that index would mix them into its
            # video. The library's own cleanup routine covers every camera
            # key, depth included.
            dataset.writer.cleanup_interrupted_episode(int(dataset.meta.total_episodes))
            _remove_orphaned_encoder_dirs(self._root)
            return dataset

        self._dataset = await asyncio.to_thread(_resume)

    async def finalize(self) -> None:
        """Close the writers so every parquet gets its footer. Once per
        session: the library's finalize is a no-op the second time, so a chunk
        opened after it would never become valid; add_frame refuses instead."""
        if self._dataset is None or self._finalized:
            return
        # Latched only on success: the library's finalize is retry-safe, and
        # latching first would turn a failed close into a permanent no-op
        # that reports footer-less parquet as a finished session.
        await asyncio.to_thread(self._dataset.finalize)
        self._finalized = True

    def add_frame(self, row: FrameRow, task: str) -> None:
        assert self._dataset is not None, "created before the first frame"
        assert not self._finalized, "session finalized; no further episodes"
        frame = {
            "observation.state": row.state,
            "action": row.action,
            "task": task,
        }
        if row.velocities is not None:
            frame["observation.velocities"] = row.velocities
        if row.efforts is not None:
            frame["observation.efforts"] = row.efforts
        for key, image in row.images.items():
            frame[f"observation.images.{key}"] = image
        self._dataset.add_frame(frame)

    async def save_episode(self) -> str | None:
        """Save the open episode. Returns why its video is untrustworthy, or
        None when it is whole."""
        assert self._dataset is not None
        assert not self._finalized, "session finalized; no further episodes"
        await asyncio.to_thread(self._dataset.save_episode)
        return self._short_video_report()

    def _short_video_report(self) -> str | None:
        """Name the cameras whose video came up short, or None when every one
        of them covers the episode."""
        meta = self._dataset.meta
        latest = meta.latest_episode
        assert latest is not None, "metadata_buffer_size=1 flushes the episode on every save"
        missing = missing_video_frames(latest, meta.video_keys, meta.fps)
        if not missing:
            return None
        detail = ", ".join(f"{key} {count}" for key, count in sorted(missing.items()))
        return (
            f"video is shorter than the state rows ({detail} frame(s) missing); "
            f"this episode should not be trained on"
        )

    def discard_open_frames(self) -> None:
        """Drop whatever the open episode buffered. Also the recovery path
        after a failed save: the library's buffer is unusable until reset."""
        assert self._dataset is not None
        self._dataset.clear_episode_buffer()
