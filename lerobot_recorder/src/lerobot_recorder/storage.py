"""Where sessions land: a host directory the manifest bind-mounts into the
container, plus an optional s3 mirror.

`storage_root` is mounted same-path via the manifest's `mount_paths`, so
sessions persist on the host by contract rather than by runtime accident.
With `s3_uri` set, the same root doubles as the staging area: the dataset is
mirrored to the bucket after every saved episode plus once at shutdown, and
the root keeps the local copy. Credentials and a custom endpoint (R2) come
from the standard AWS environment variables, never from launch parameters.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

# Session directories are named by their UTC creation time; the name is the
# wire identity resume_session resolves, so it is a strict format, not a hint.
SESSION_NAME_FORMAT = "%Y-%m-%d_%H-%M-%S"
# The same shape spelled for the operators and clients who read a refusal;
# strftime is this node's business, not theirs.
SESSION_NAME_SHAPE = "YYYY-MM-DD_HH-MM-SS"


def validate_session_name(name: str) -> None:
    """Reject anything that is not exactly a session directory name. The
    round-trip check refuses non-canonical spellings, and with them every
    path-traversal shape (separators, dots, absolute paths)."""
    try:
        canonical = datetime.datetime.strptime(name, SESSION_NAME_FORMAT).strftime(
            SESSION_NAME_FORMAT
        )
    except ValueError:
        canonical = None
    if canonical != name:
        raise ValueError(f"{name!r} is not a session name (UTC {SESSION_NAME_SHAPE})")


@dataclass(frozen=True)
class S3Destination:
    bucket: str
    prefix: str


@dataclass(frozen=True)
class StorageTarget:
    """The mounted directory sessions are written under, and the optional s3
    destination they mirror to."""

    root: Path
    s3: S3Destination | None


def parse_storage(storage_root: str, s3_uri: str) -> StorageTarget:
    if not storage_root.startswith("/"):
        raise ValueError(f"storage_root must be an absolute path, got {storage_root!r}")
    if not s3_uri:
        return StorageTarget(root=Path(storage_root), s3=None)
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"s3_uri must be s3://bucket[/prefix], got {s3_uri!r}")
    if not parsed.netloc:
        raise ValueError("s3_uri has no bucket")
    s3 = S3Destination(bucket=parsed.netloc, prefix=parsed.path.strip("/"))
    return StorageTarget(root=Path(storage_root), s3=s3)


def ensure_mounted(target: StorageTarget) -> None:
    """The manifest bind-mounts storage_root into the container, and peppy
    creates the host side before the start. A root that does not exist means
    the mount is not in effect, and creating it here would write datasets
    into the container's own filesystem, gone when the instance stops."""
    if not target.root.is_dir():
        raise ValueError(
            f"storage_root {target.root} does not exist; it should have been "
            f"mounted from the host before the node started"
        )


def resolved_endpoint() -> str:
    """The S3 endpoint boto3 will use, for the startup destination log.
    The service-scoped variable wins over the generic one, mirroring boto3's
    own precedence; with neither set boto3 talks to AWS proper."""
    return (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
        or "AWS default endpoints"
    )


def _boto_credentials():
    import boto3

    return boto3.session.Session().get_credentials()


def probe_credentials(target: StorageTarget, lookup=_boto_credentials) -> None:
    """Fail an s3 mirror at startup, not at the first upload mid-session.
    The lookup is injectable so tests exercise the refusal without depending
    on botocore's full provider chain (env vars, files, IMDS)."""
    if target.s3 is None:
        return
    if lookup() is None:
        raise RuntimeError(
            "s3_uri needs credentials in the environment "
            "(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)"
        )


# The dataset library stages one image per frame here and deletes the tree
# once the episode is encoded to video. Datasets are written with videos, so
# nothing under it belongs in the mirror.
FRAME_STAGING_DIR = "images"

# Upload bounds; without them botocore's defaults let one hung endpoint
# consume the whole shutdown grace window. total_max_attempts counts the
# initial request, so this is three requests, not three retries.
S3_CONNECT_TIMEOUT_S = 10
S3_READ_TIMEOUT_S = 60
S3_TOTAL_MAX_ATTEMPTS = 3


@dataclass
class Mirror:
    """Incremental uploader to an s3 destination: re-walks the session
    directory and uploads files whose size or mtime changed since the last
    sync. Runs in a worker thread (boto3 is blocking); best-effort, failures
    are retried by the next sync because the manifest entry is only recorded
    on success.

    Consistency model: still-growing files (the open parquet, the current
    video chunk) re-upload after every saved episode, bounded per sync by the
    library's chunk-rotation sizes; their remote copies are unreadable until
    the final sync after finalize, which mirrors the completed dataset. A file
    that changes while its upload is in flight gets a torn remote copy whose
    signature no longer matches, so the next sync re-uploads it."""

    dest: S3Destination
    session_dir: Path
    session_name: str
    _uploaded: dict[Path, tuple[int, float]] = field(default_factory=dict)
    _client: object | None = field(default=None, repr=False)

    def sync(self) -> tuple[int, int]:
        """Upload everything that changed; returns (files, bytes) uploaded."""
        if self._client is None:
            import boto3
            from botocore.config import Config

            # Bounded: the final pass runs inside the daemon's shutdown grace
            # window, so a hung endpoint must fail (local copy kept, logged)
            # rather than stall until the daemon kills the node mid-pass.
            self._client = boto3.client(
                "s3",
                config=Config(
                    connect_timeout=S3_CONNECT_TIMEOUT_S,
                    read_timeout=S3_READ_TIMEOUT_S,
                    retries={"total_max_attempts": S3_TOTAL_MAX_ATTEMPTS, "mode": "standard"},
                ),
            )
        files = 0
        size = 0
        for path in sorted(self.session_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.session_dir)
            if FRAME_STAGING_DIR in rel.parts:
                continue
            try:
                status = path.stat()
                signature = (status.st_size, status.st_mtime)
                if self._uploaded.get(path) == signature:
                    continue
                key_parts = [self.dest.prefix, self.session_name, str(rel)]
                key = "/".join(part for part in key_parts if part)
                self._client.upload_file(str(path), self.dest.bucket, key)
                files += 1
                size += status.st_size
            except FileNotFoundError:
                # The library deletes staged files as it encodes; a file that
                # vanished mid-walk must not abort the rest of the pass.
                continue
            self._uploaded[path] = signature
        return files, size
