"""The weights of the sam3_siglip backend: what is pinned, where a model is
kept, and what a download does when it is stopped, when the link drops and
when the server sends other bytes. The Hub is a fake, or a server of the
test's own on the loopback interface: the suite downloads nothing from the
network."""

import fcntl
import hashlib
import http.server
import io
import logging
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from openarm_ai_brain_vla.perception import weights
from openarm_ai_brain_vla.perception.weights import (
    DEFAULT_DIRECTORY,
    FETCHING_SUFFIX,
    LOCK_FILE,
    PARTIAL_SUFFIX,
    SAM3,
    SIGLIP,
    SOURCES,
    STALLED_ATTEMPTS,
    WEIGHTS_DIRECTORY_VARIABLE,
    Body,
    File,
    Source,
    WeightsError,
)

# A model of two files, the second of many chunks at the chunk size the
# tests set.
FILES = {"config.json": b"{}", "model.safetensors": bytes(range(100))}
CHUNK_BYTES = 10
# How long a test waits for another thread or process before it fails. No
# test passes because this time has gone by.
FAILURE_S = 60.0


def source_of(files: dict[str, bytes]) -> Source:
    pins = tuple(File(path, len(content), hashlib.sha256(content).hexdigest()) for path, content in files.items())
    return Source("model", "someone/model", "0123456789abcdef0123456789abcdef01234567", pins)


MODEL = source_of(FILES)
CONFIG, SAFETENSORS = MODEL.files


class Killed(BaseException):
    """Stands in for the end of the node's process in the middle of a read:
    nothing of the download catches it."""


@dataclass(frozen=True)
class Step:
    """One answer of the fake Hub: `after` bytes, then the end of the answer,
    or `then` raised by the read."""

    after: int
    then: Optional[BaseException] = None


class FakeHub:
    """Stands in for the Hub: answers each request with the bytes of its URL
    from the offset asked for, and records the request. `plan` gives a file
    the answers to its next requests, each a `Step` or an exception the
    request raises; `ignores_ranges` makes every answer the whole file."""

    def __init__(self, files: dict[str, bytes] = FILES) -> None:
        self.content = {file.path: files[file.path] for file in MODEL.files}
        self.asked: list[tuple[str, int]] = []
        self.plan: dict[str, list] = {}
        self.ignores_ranges = False
        self.on_ask = lambda: None

    @contextmanager
    def open_at(self, url: str, offset: int):
        path = url.removeprefix(f"{weights.HUB}/{MODEL.repository}/resolve/{MODEL.revision}/")
        self.asked.append((path, offset))
        self.on_ask()
        step = self.plan[path].pop(0) if self.plan.get(path) else None
        if isinstance(step, BaseException):
            raise step
        starts_at = 0 if self.ignores_ranges else offset
        content = self.content[path][starts_at:]
        if step is None:
            yield Body(starts_at, io.BytesIO(content).read)
            return
        stream = io.BytesIO(content[: step.after])

        def read(size: int) -> bytes:
            chunk = stream.read(size)
            if not chunk and step.then is not None:
                raise step.then
            return chunk

        yield Body(starts_at, read)


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    monkeypatch.setattr(weights, "CHUNK_BYTES", CHUNK_BYTES)


@pytest.fixture
def hub() -> FakeHub:
    return FakeHub()


@pytest.fixture
def directory(tmp_path) -> Path:
    return tmp_path / "weights"


def names_in(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def content_of(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_every_model_is_pinned_by_a_commit_and_every_file_by_its_size_and_hash():
    assert SOURCES == (SAM3, SIGLIP)
    assert len({source.name for source in SOURCES}) == len(SOURCES)
    for source in SOURCES:
        assert re.fullmatch(r"[0-9a-f]{40}", source.revision), source
        assert source.directory_name == f"{source.name}-{source.revision}"
        paths = [file.path for file in source.files]
        assert len(set(paths)) == len(paths) and {"config.json", "model.safetensors"} <= set(paths)
        for file in source.files:
            assert file.size > 0 and re.fullmatch(r"[0-9a-f]{64}", file.sha256), file
    # The checkpoint of SAM 3's original code base is not one transformers reads.
    assert "sam3.pt" not in [file.path for file in SAM3.files]
    assert SAM3.url(SAM3.files[0]) == f"https://huggingface.co/jetjodh/sam3/resolve/{SAM3.revision}/config.json"


def test_a_model_is_downloaded_whole_under_its_name_and_revision(directory, hub):
    staged_at = weights.stage(MODEL, directory, hub.open_at)
    assert staged_at == directory / MODEL.directory_name
    assert content_of(staged_at) == FILES
    assert hub.asked == [("config.json", 0), ("model.safetensors", 0)]
    assert names_in(directory) == sorted([LOCK_FILE, MODEL.directory_name])


def test_a_model_on_the_machine_is_not_downloaded_again(directory, hub, no_network):
    weights.stage(MODEL, directory, hub.open_at)
    asked = list(hub.asked)
    assert weights.stage(MODEL, directory, hub.open_at) == directory / MODEL.directory_name
    assert hub.asked == asked
    # The node's own way to the Hub is not taken either.
    assert weights.stage(MODEL, directory) == directory / MODEL.directory_name


def test_a_node_killed_in_a_download_leaves_a_partial_file_that_the_next_start_continues(directory, hub):
    hub.plan["model.safetensors"] = [Step(after=30, then=Killed())]
    with pytest.raises(Killed):
        weights.stage(MODEL, directory, hub.open_at)
    # Nothing is under the model's name, so nothing reads half a model: the
    # file that was whole has its name, the other is partial and holds
    # every chunk that was read.
    fetching = directory / f"{MODEL.directory_name}{FETCHING_SUFFIX}"
    assert names_in(directory) == sorted([LOCK_FILE, fetching.name])
    assert content_of(fetching) == {"config.json": b"{}", f"model.safetensors{PARTIAL_SUFFIX}": FILES["model.safetensors"][:30]}
    hub.asked.clear()
    staged_at = weights.stage(MODEL, directory, hub.open_at)
    assert hub.asked == [("model.safetensors", 30)]
    assert content_of(staged_at) == FILES
    assert names_in(directory) == sorted([LOCK_FILE, MODEL.directory_name])


@pytest.mark.parametrize("then", [None, ConnectionResetError("the link went down")])
def test_a_link_that_drops_is_asked_again_from_the_last_byte(directory, hub, then):
    hub.plan["model.safetensors"] = [Step(after=30, then=then), Step(after=30, then=then), Step(after=30, then=then)]
    staged_at = weights.stage(MODEL, directory, hub.open_at)
    assert hub.asked == [("config.json", 0), ("model.safetensors", 0), ("model.safetensors", 30), ("model.safetensors", 60), ("model.safetensors", 90)]
    assert content_of(staged_at) == FILES


def test_a_download_fails_when_attempts_in_a_row_add_nothing_and_the_next_one_continues(directory, hub):
    assert STALLED_ATTEMPTS == 3
    down = ConnectionRefusedError("the link is down")
    # An attempt that adds bytes does not count: the three that fail follow it.
    hub.plan["model.safetensors"] = [Step(after=30), down, Step(after=20), down, down, down]
    url = MODEL.url(SAFETENSORS)
    with pytest.raises(WeightsError, match=re.escape(f"the download of {url} stopped at 50 of 100 bytes: the link is down")):
        weights.stage(MODEL, directory, hub.open_at)
    assert [offset for path, offset in hub.asked if path == "model.safetensors"] == [0, 30, 30, 50, 50, 50]
    assert MODEL.directory_name not in names_in(directory)
    hub.asked.clear()
    assert content_of(weights.stage(MODEL, directory, hub.open_at)) == FILES
    assert hub.asked == [("model.safetensors", 50)]


def test_a_server_whose_file_is_shorter_than_the_pin_fails_the_download(directory):
    hub = FakeHub({**FILES, "model.safetensors": FILES["model.safetensors"][:90]})
    url = MODEL.url(SAFETENSORS)
    with pytest.raises(WeightsError, match=re.escape(f"the download of {url} stopped at 90 of 100 bytes: the server ended its answer")):
        weights.stage(MODEL, directory, hub.open_at)
    assert [offset for path, offset in hub.asked if path == "model.safetensors"] == [0, 90, 90, 90]


def test_a_server_that_sends_the_whole_file_starts_the_partial_file_again(directory, hub):
    hub.ignores_ranges = True
    hub.plan["model.safetensors"] = [Step(after=30)]
    staged_at = weights.stage(MODEL, directory, hub.open_at)
    assert hub.asked == [("config.json", 0), ("model.safetensors", 0), ("model.safetensors", 30)]
    assert content_of(staged_at) == FILES


def test_an_answer_that_starts_past_the_byte_asked_for_is_not_appended(directory, hub):
    @contextmanager
    def ahead(url, offset):
        with hub.open_at(url, offset) as body:
            yield Body(body.starts_at + 5, body.read) if url.endswith("model.safetensors") else body

    url = MODEL.url(SAFETENSORS)
    with pytest.raises(WeightsError, match=re.escape(f"the download of {url} stopped at 0 of 100 bytes: the answer starts at byte 5, past byte 0")):
        weights.stage(MODEL, directory, ahead)
    assert content_of(directory / f"{MODEL.directory_name}{FETCHING_SUFFIX}") == {"config.json": b"{}", f"model.safetensors{PARTIAL_SUFFIX}": b""}


def test_bytes_that_are_not_the_pinned_ones_are_removed_and_never_named(directory, hub):
    other = FakeHub({**FILES, "model.safetensors": bytes(100)})
    found = hashlib.sha256(bytes(100)).hexdigest()
    url = MODEL.url(SAFETENSORS)
    with pytest.raises(WeightsError, match=re.escape(f"{url} is not the pinned file: its SHA-256 is {found}, the pin is {SAFETENSORS.sha256}")):
        weights.stage(MODEL, directory, other.open_at)
    fetching = directory / f"{MODEL.directory_name}{FETCHING_SUFFIX}"
    assert names_in(directory) == sorted([LOCK_FILE, fetching.name]) and names_in(fetching) == ["config.json"]
    # The next fetch downloads the file from its start.
    assert content_of(weights.stage(MODEL, directory, hub.open_at)) == FILES
    assert hub.asked == [("model.safetensors", 0)]


def test_no_more_than_the_pinned_size_of_a_file_is_read(directory):
    longer = FakeHub({**FILES, "model.safetensors": FILES["model.safetensors"] + b"more than the pin"})
    assert content_of(weights.stage(MODEL, directory, longer.open_at)) == FILES


def test_a_partial_file_of_other_bytes_fails_the_fetch_once(directory, hub):
    fetching = directory / f"{MODEL.directory_name}{FETCHING_SUFFIX}"
    fetching.mkdir(parents=True)
    (fetching / f"model.safetensors{PARTIAL_SUFFIX}").write_bytes(bytes(120))
    with pytest.raises(WeightsError, match="is not the pinned file"):
        weights.stage(MODEL, directory, hub.open_at)
    assert hub.asked == [("config.json", 0)]
    assert content_of(weights.stage(MODEL, directory, hub.open_at)) == FILES


def test_a_disk_without_room_for_what_is_missing_is_refused_before_the_download(directory, hub, monkeypatch):
    asked_of = []

    def free_bytes(path):
        asked_of.append(path)
        return 101

    monkeypatch.setattr(weights, "free_bytes", free_bytes)
    with pytest.raises(WeightsError, match=re.escape(f"the weights of someone/model need 102 more bytes in {directory}, which has 101 free")):
        weights.stage(MODEL, directory, hub.open_at)
    fetching = directory / f"{MODEL.directory_name}{FETCHING_SUFFIX}"
    assert hub.asked == [] and asked_of == [fetching]
    # What a stopped download left counts: one byte of it makes the room.
    (fetching / f"model.safetensors{PARTIAL_SUFFIX}").write_bytes(FILES["model.safetensors"][:1])
    assert content_of(weights.stage(MODEL, directory, hub.open_at)) == FILES


def test_the_free_room_is_the_disk_of_the_directory(tmp_path):
    assert weights.free_bytes(tmp_path) > 0


def test_the_lock_is_held_for_the_whole_download_and_released_after_it(directory, hub):
    def lock_is_free() -> bool:
        with (directory / LOCK_FILE).open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

    held_at_each_request = []
    hub.on_ask = lambda: held_at_each_request.append(not lock_is_free())
    weights.stage(MODEL, directory, hub.open_at)
    assert held_at_each_request == [True, True]
    assert lock_is_free()


class Lines(logging.Handler):
    """Keeps the lines a logger says, and sets `waited_for` when one of
    them is `line`."""

    def __init__(self, line: str = "") -> None:
        super().__init__()
        self.line = line
        self.messages: list[str] = []
        self.waited_for = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())
        if record.getMessage() == self.line:
            self.waited_for.set()


@contextmanager
def lines_of_the_download(line: str = ""):
    """What the weights' logger says, whatever the loggers above it keep."""
    lines = Lines(line)
    level = weights.logger.level
    weights.logger.addHandler(lines)
    weights.logger.setLevel(logging.INFO)
    try:
        yield lines
    finally:
        weights.logger.setLevel(level)
        weights.logger.removeHandler(lines)


def test_a_second_node_waits_for_the_one_that_downloads_and_then_finds_the_model(directory, hub):
    directory.mkdir()
    outcome = []
    with lines_of_the_download(f"another node downloads into {directory}: waiting for it") as lines, (directory / LOCK_FILE).open("w") as lock_of_the_first_node:
        fcntl.flock(lock_of_the_first_node, fcntl.LOCK_EX)
        second_node = threading.Thread(target=lambda: outcome.append(weights.stage(MODEL, directory, hub.open_at)), daemon=True)
        second_node.start()
        assert lines.waited_for.wait(FAILURE_S)
        # The first node ends its download while the second waits.
        (directory / MODEL.directory_name).mkdir()
    second_node.join(FAILURE_S)
    assert outcome == [directory / MODEL.directory_name] and hub.asked == []


def test_the_download_logs_where_it_goes_and_each_tenth_of_a_file(directory, hub):
    with lines_of_the_download() as lines:
        weights.stage(MODEL, directory, hub.open_at)
    url = MODEL.url(SAFETENSORS)
    assert lines.messages == [
        f"downloading the weights of someone/model into {directory}: 0 MB to go",
        f"{MODEL.url(CONFIG)}: 100% of 0.0 MB",
        *(f"{url}: {percent}% of 0.0 MB" for percent in range(10, 101, 10)),
    ]


def test_the_node_keeps_its_weights_where_the_variable_says_else_under_the_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert str(DEFAULT_DIRECTORY) == "~/.cache/openarm_ai_brain_vla/weights"
    for unset in (None, "", "  "):
        if unset is None:
            monkeypatch.delenv(WEIGHTS_DIRECTORY_VARIABLE, raising=False)
        else:
            monkeypatch.setenv(WEIGHTS_DIRECTORY_VARIABLE, unset)
        assert weights.node_directory() == tmp_path / ".cache/openarm_ai_brain_vla/weights"
    monkeypatch.setenv(WEIGHTS_DIRECTORY_VARIABLE, " ~/elsewhere ")
    assert weights.node_directory() == tmp_path / "elsewhere"
    monkeypatch.setenv(WEIGHTS_DIRECTORY_VARIABLE, "/disk/weights")
    assert weights.node_directory() == Path("/disk/weights")


def test_a_response_starts_at_the_offset_its_content_range_names():
    assert weights.range_start(200, None) == 0
    # A server that ignores the range sends the whole file.
    assert weights.range_start(200, "bytes 30-99/100") == 0
    assert weights.range_start(206, "bytes 30-99/100") == 30
    assert weights.range_start(206, " bytes 30-99/* ") == 30
    for unreadable in (None, "", "30-99/100", "bytes */100"):
        with pytest.raises(OSError, match="a partial answer with the Content-Range"):
            weights.range_start(206, unreadable)


class LoopbackHub(http.server.ThreadingHTTPServer):
    """A server on the loopback interface that answers as the Hub does: the
    URL of a file redirects to the one that holds its bytes, which honours a
    range. Records the path and the Range header of every request.
    `stalls_after` makes the next answer stop after that many bytes and
    hold its connection open until `release` is set."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), LoopbackHandler)
        self.content = {f"/{MODEL.repository}/resolve/{MODEL.revision}/{path}": content for path, content in FILES.items()}
        self.seen: list[tuple[str, Optional[str]]] = []
        self.ignores_ranges = False
        self.stalls_after: Optional[int] = None
        self.stalled = threading.Event()
        self.release = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def handle_error(self, request, client_address) -> None:
        """A client that went away in the middle of an answer is not an error
        of the test."""


class LoopbackHandler(http.server.BaseHTTPRequestHandler):
    server: LoopbackHub

    def do_GET(self) -> None:
        self.server.seen.append((self.path, self.headers.get("Range")))
        if not self.path.startswith("/bytes"):
            self.send_response(302)
            self.send_header("Location", f"/bytes{self.path}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        content = self.server.content.get(self.path.removeprefix("/bytes"))
        if content is None:
            self.send_error(404)
            return
        asked = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range") or "")
        start = int(asked.group(1)) if asked and not self.server.ignores_ranges else 0
        if start >= len(content):
            self.send_error(416)
            return
        if asked and not self.server.ignores_ranges:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(content) - 1}/{len(content)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(content) - start))
        self.end_headers()
        stalls_after, self.server.stalls_after = self.server.stalls_after, None
        if stalls_after is None:
            self.wfile.write(content[start:])
            return
        self.wfile.write(content[start : start + stalls_after])
        self.wfile.flush()
        self.server.stalled.set()
        self.server.release.wait()

    def log_message(self, format, *args) -> None:
        return None


@pytest.fixture
def loopback(monkeypatch):
    """The node's own way to the Hub, pointed at a server of the test's."""
    server = LoopbackHub()
    serving = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    serving.start()
    monkeypatch.setattr(weights, "HUB", server.url)
    yield server
    server.release.set()
    server.shutdown()
    server.server_close()
    serving.join(FAILURE_S)


def bytes_path(file: File) -> str:
    return f"/bytes/{MODEL.repository}/resolve/{MODEL.revision}/{file.path}"


def test_the_node_downloads_over_http_through_the_redirect_of_the_hub(directory, loopback):
    assert content_of(weights.stage(MODEL, directory)) == FILES
    assert loopback.seen == [
        (f"/{MODEL.repository}/resolve/{MODEL.revision}/config.json", None),
        (bytes_path(CONFIG), None),
        (f"/{MODEL.repository}/resolve/{MODEL.revision}/model.safetensors", None),
        (bytes_path(SAFETENSORS), None),
    ]


def test_a_partial_file_is_continued_over_http_with_a_range_the_redirect_keeps(directory, loopback):
    fetching = directory / f"{MODEL.directory_name}{FETCHING_SUFFIX}"
    fetching.mkdir(parents=True)
    (fetching / "config.json").write_bytes(FILES["config.json"])
    (fetching / f"model.safetensors{PARTIAL_SUFFIX}").write_bytes(FILES["model.safetensors"][:30])
    assert content_of(weights.stage(MODEL, directory)) == FILES
    assert loopback.seen == [
        (f"/{MODEL.repository}/resolve/{MODEL.revision}/model.safetensors", "bytes=30-"),
        (bytes_path(SAFETENSORS), "bytes=30-"),
    ]


def test_a_server_that_ignores_the_range_over_http_gives_the_whole_file(directory, loopback):
    loopback.ignores_ranges = True
    fetching = directory / f"{MODEL.directory_name}{FETCHING_SUFFIX}"
    fetching.mkdir(parents=True)
    # Not the file's first bytes: they are gone once the whole file comes.
    (fetching / f"model.safetensors{PARTIAL_SUFFIX}").write_bytes(bytes(30))
    assert content_of(weights.stage(MODEL, directory)) == FILES


def test_a_file_the_server_does_not_have_fails_the_download_with_its_answer(directory, loopback):
    del loopback.content[f"/{MODEL.repository}/resolve/{MODEL.revision}/model.safetensors"]
    url = MODEL.url(SAFETENSORS)
    with pytest.raises(WeightsError, match=re.escape(f"the download of {url} stopped at 0 of 100 bytes: HTTP Error 404: Not Found")):
        weights.stage(MODEL, directory)
    assert loopback.seen.count((bytes_path(SAFETENSORS), None)) == STALLED_ATTEMPTS


def test_the_command_downloads_every_model_into_the_directory_it_is_given(directory, loopback, monkeypatch):
    monkeypatch.setattr(weights, "SOURCES", (MODEL,))
    weights.main(["fetch", str(directory)])
    assert content_of(directory / MODEL.directory_name) == FILES


def test_the_command_downloads_into_the_directory_of_the_node_when_given_none(directory, loopback, monkeypatch):
    monkeypatch.setattr(weights, "SOURCES", (MODEL,))
    monkeypatch.setenv(WEIGHTS_DIRECTORY_VARIABLE, str(directory))
    weights.main(["fetch"])
    assert content_of(directory / MODEL.directory_name) == FILES


# A node that loads: the download runs in the daemon thread the perceiver
# gives a load, and the process ends when its first line of input comes,
# while the download still waits for bytes.
NODE_THAT_STOPS_IN_A_DOWNLOAD = """
import asyncio, sys
from pathlib import Path
from openarm_ai_brain_vla.perception import weights
from openarm_ai_brain_vla.perception.perceiver import in_daemon_thread

weights.HUB, directory = sys.argv[1], Path(sys.argv[2])
weights.CHUNK_BYTES = int(sys.argv[3])
file = weights.File(sys.argv[4], int(sys.argv[5]), sys.argv[6])
source = weights.Source("model", "someone/model", "0123456789abcdef0123456789abcdef01234567", (file,))

async def node():
    load = asyncio.ensure_future(in_daemon_thread(weights.stage, source, directory))
    await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
    assert not load.done()

asyncio.run(node())
"""


def test_a_node_that_stops_in_a_download_ends_at_once_and_the_next_start_continues(directory, loopback):
    model = Source(MODEL.name, MODEL.repository, MODEL.revision, (SAFETENSORS,))
    loopback.stalls_after = 30
    arguments = [loopback.url, str(directory), str(CHUNK_BYTES), SAFETENSORS.path, str(SAFETENSORS.size), SAFETENSORS.sha256]
    node = subprocess.Popen([sys.executable, "-c", NODE_THAT_STOPS_IN_A_DOWNLOAD, *arguments], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert loopback.stalled.wait(FAILURE_S)
        # The node is told to stop while its download waits for the next bytes.
        _, errors = node.communicate(b"stop\n", timeout=FAILURE_S)
    finally:
        node.kill()
    assert node.returncode == 0, errors.decode()
    fetching = directory / f"{MODEL.directory_name}{FETCHING_SUFFIX}"
    assert content_of(fetching) == {f"model.safetensors{PARTIAL_SUFFIX}": FILES["model.safetensors"][:30]}
    # The lock went with the process, and the next start asks for the rest.
    loopback.seen.clear()
    assert content_of(weights.stage(model, directory)) == {"model.safetensors": FILES["model.safetensors"]}
    assert loopback.seen[-1] == (bytes_path(SAFETENSORS), "bytes=30-")
