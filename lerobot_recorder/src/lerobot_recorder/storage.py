"""Where sessions land, parsed from the storage_uri parameter.

file:// writes the dataset directly under the given root. s3:// stages the
dataset in a local directory and mirrors it to the bucket after every saved
episode plus once at shutdown; the staging copy is kept at shutdown as the
local dataset (under /tmp, so it lasts until reboot). Credentials and a
custom endpoint (R2) come from the standard AWS environment variables,
never from launch parameters.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import os
import re
import stat
import tempfile
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
class LocalTarget:
    root: Path


@dataclass(frozen=True)
class S3Location:
    """A parsed s3:// URI; carries no local state, so parsing stays pure."""

    bucket: str
    prefix: str


@dataclass(frozen=True)
class S3Target:
    bucket: str
    prefix: str
    staging_root: Path


ParsedTarget = LocalTarget | S3Location
StorageTarget = LocalTarget | S3Target


def parse_storage_uri(uri: str) -> ParsedTarget:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"file:// URI must be local, got host {parsed.netloc!r}")
        if not parsed.path:
            raise ValueError("file:// URI has no path")
        return LocalTarget(root=Path(parsed.path))
    if parsed.scheme == "s3":
        if not parsed.netloc:
            raise ValueError("s3:// URI has no bucket")
        return S3Location(bucket=parsed.netloc, prefix=parsed.path.strip("/"))
    raise ValueError(f"unsupported storage_uri scheme {parsed.scheme!r} (file:// or s3://)")


def staging_key(bucket: str, prefix: str) -> str:
    """Stable directory name for one s3 destination: a sanitized readable part
    for operators plus a short hash so prefixes that sanitize identically
    (a/b vs a_b) still get distinct roots."""
    readable = re.sub(r"[^a-z0-9_-]", "_", f"{bucket}_{prefix}".lower())[:60]
    digest = hashlib.sha256(f"{bucket}\n{prefix}".encode()).hexdigest()[:8]
    return f"{readable}_{digest}"


def prepare_target(parsed: ParsedTarget) -> StorageTarget:
    """Allocate the runtime side of a parsed target: an s3 location gains its
    local staging directory here, next to the owner that cleans it up. The
    root is stable across runs (same URI, same directory), which is what lets
    resume_session find a previous run's sessions by name."""
    if isinstance(parsed, S3Location):
        base = Path(tempfile.gettempdir()) / "lerobot_recorder_staging"
        staging = base / staging_key(parsed.bucket, parsed.prefix)
        # Both levels are checked: a leaf created inside somebody else's base
        # directory can be renamed out from under its owner after the check.
        for directory in (base, staging):
            directory.mkdir(exist_ok=True, mode=0o700)
            _own_directory(directory)
        return S3Target(bucket=parsed.bucket, prefix=parsed.prefix, staging_root=staging)
    return parsed


def _own_directory(path: Path) -> None:
    """A stable staging path under a shared /tmp is one anybody can create
    first, so datasets would be written through whatever they left there.
    Recording only ever writes into a real directory this user owns. One
    lstat answers both questions, so a symlink cannot slip in between a
    symlink check and an ownership check."""
    status = path.lstat()
    if not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"staging path {path} is not a directory")
    if status.st_uid != os.getuid():
        raise ValueError(f"staging directory {path} belongs to another user")


def session_root(target: StorageTarget) -> Path:
    """The local directory this session's dataset is written under."""
    if isinstance(target, LocalTarget):
        return target.root
    return target.staging_root


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
    """Fail an s3 target at startup, not at the first upload mid-session.
    The lookup is injectable so tests exercise the refusal without depending
    on botocore's full provider chain (env vars, files, IMDS)."""
    if not isinstance(target, S3Target):
        return
    if lookup() is None:
        raise RuntimeError(
            "s3 storage_uri needs credentials in the environment "
            "(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)"
        )


def cleanup_staging(target: StorageTarget) -> None:
    """Drop an s3 target's staging root only if it holds nothing. The root is
    stable across runs and may hold previous sessions (including the only
    local copy of one whose final mirror pass was cut off), so a failed start
    must never remove it recursively."""
    if isinstance(target, S3Target):
        with contextlib.suppress(OSError):
            target.staging_root.rmdir()


# The dataset library stages one image per frame here and deletes the tree
# once the episode is encoded to video. Datasets are written with videos, so
# nothing under it belongs in the mirror.
FRAME_STAGING_DIR = "images"

# Upload bounds, per attempt; without them botocore's defaults let one hung
# endpoint consume the whole shutdown grace window.
S3_CONNECT_TIMEOUT_S = 10
S3_READ_TIMEOUT_S = 60
S3_MAX_ATTEMPTS = 3


@dataclass
class Mirror:
    """Incremental uploader for an s3 target: re-walks the session directory
    and uploads files whose size or mtime changed since the last sync. Runs in
    a worker thread (boto3 is blocking); best-effort, failures are retried by
    the next sync because the manifest entry is only recorded on success.

    Consistency model: still-growing files (the open parquet, the current
    video chunk) re-upload after every saved episode, bounded per sync by the
    library's chunk-rotation sizes; their remote copies are unreadable until
    the final sync after finalize, which mirrors the completed dataset. A file
    that changes while its upload is in flight gets a torn remote copy whose
    signature no longer matches, so the next sync re-uploads it."""

    target: S3Target
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
            # window, so a hung endpoint must fail (staging kept, logged)
            # rather than stall until the daemon kills the node mid-pass.
            self._client = boto3.client(
                "s3",
                config=Config(
                    connect_timeout=S3_CONNECT_TIMEOUT_S,
                    read_timeout=S3_READ_TIMEOUT_S,
                    retries={"max_attempts": S3_MAX_ATTEMPTS, "mode": "standard"},
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
                stat = path.stat()
                signature = (stat.st_size, stat.st_mtime)
                if self._uploaded.get(path) == signature:
                    continue
                key_parts = [self.target.prefix, self.session_name, str(rel)]
                key = "/".join(part for part in key_parts if part)
                self._client.upload_file(str(path), self.target.bucket, key)
                files += 1
                size += stat.st_size
            except FileNotFoundError:
                # The library deletes staged files as it encodes; a file that
                # vanished mid-walk must not abort the rest of the pass.
                continue
            self._uploaded[path] = signature
        return files, size
