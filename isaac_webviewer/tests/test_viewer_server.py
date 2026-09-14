"""Exercise the viewer over HTTP without its container, browser, or GPU."""

import errno
import importlib.util
import json
import logging
import socket
import threading
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_VIEWER_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_isaac_viewer_server_under_test", _VIEWER_DIR / "viewer_server.py",
)
viewer_server = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(viewer_server)
_SCRIPT_TAG = '<script src="/browser-logs.js"></script>'
_ENTRY = {"level": "warn", "message": "WebRTC connection failed", "page": "/index.html"}


@pytest.fixture
def dist(tmp_path):
    directory = tmp_path / "dist"
    directory.mkdir()
    (directory / "index.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        '<script type="module" src="/assets/app.js"></script>'
        '</head><body>Robot café</body></html>',
        encoding="utf-8",
    )
    assets = directory / "assets"
    assets.mkdir()
    (assets / "app.js").write_bytes(b'console.warn("viewer warning");\n')
    (assets / "app.css").write_bytes(b"body { color: black; }\n")
    (assets / "logo.svg").write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>')
    (assets / "robot arm.txt").write_bytes("OpenArm café".encode())
    (assets / "data.unknown-viewer-extension").write_bytes(b"\x00\x01\xff")
    (directory / "empty").mkdir()
    return directory


@pytest.fixture
def server(dist):
    instance = viewer_server.ViewerServer(("127.0.0.1", 0), dist)
    thread = threading.Thread(
        target=instance.serve_forever, kwargs={"poll_interval": 0.01},
        name="test-viewer-http",
    )
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join()


def _request(server, method, path, *, body=b"", headers=None, end_body=False):
    """Send explicit headers so malformed framing is not repaired by http.client."""
    host = "{}:{}".format(*server.server_address)
    request_headers = {"Host": host}
    if method == "POST":
        request_headers.update({
            "Origin": f"http://{host}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
    request_headers.update(headers or {})
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for name, value in request_headers.items():
            if value is not None:
                connection.putheader(name, value)
        connection.endheaders(body)
        if end_body:
            connection.sock.shutdown(socket.SHUT_WR)
        response = connection.getresponse()
        return SimpleNamespace(
            status=response.status, headers=response.headers, body=response.read(),
        )
    finally:
        connection.close()


def _assert_get_and_head(server, path, body, content_types):
    for method in ("GET", "HEAD"):
        response = _request(server, method, path)
        assert response.status == 200
        assert response.headers["Content-Type"] in content_types
        assert response.headers["Content-Length"] == str(len(body))
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.body == (body if method == "GET" else b"")


@pytest.mark.parametrize("path", [
    "/", "/index.html", "/?view=robot", "/assets/../index.html",
    "/%69ndex.html?view=robot", "/assets/%2e%2e/",
])
def test_index_injects_diagnostics_before_app_without_changing_dist(server, dist, path):
    original_files = {file: file.read_bytes() for file in dist.rglob("*") if file.is_file()}
    response = _request(server, "GET", path)
    assert response.status == 200
    html = response.body.decode("utf-8")
    assert html.count(_SCRIPT_TAG) == 1
    assert html.index("<head>") < html.index(_SCRIPT_TAG) < html.index('type="module"')
    assert html.replace("\n    " + _SCRIPT_TAG, "", 1).encode() == original_files[dist / "index.html"]
    _assert_get_and_head(server, path, response.body, {"text/html; charset=utf-8"})
    assert {file: file.read_bytes() for file in dist.rglob("*") if file.is_file()} == original_files


@pytest.mark.parametrize("path", [
    "/browser-logs.js", "/browser-logs.js?version=1", "/%62rowser-logs.js",
])
def test_diagnostics_script_is_served_from_node_not_dist(server, dist, path):
    shadow = dist / "browser-logs.js"
    shadow.write_bytes(b"not the diagnostics script")
    _assert_get_and_head(
        server, path, (_VIEWER_DIR / "browser_logs.js").read_bytes(),
        {"text/javascript; charset=utf-8"},
    )
    assert shadow.read_bytes() == b"not the diagnostics script"


@pytest.mark.parametrize(("path", "filename", "content_types"), [
    ("/assets/app.js", "app.js", {"text/javascript", "application/javascript"}),
    ("/assets/app.css?theme=dark", "app.css", {"text/css"}),
    ("/assets/logo.svg", "logo.svg", {"image/svg+xml"}),
    ("/assets/robot%20arm.txt", "robot arm.txt", {"text/plain"}),
    ("/assets/data.unknown-viewer-extension", "data.unknown-viewer-extension", {"application/octet-stream"}),
])
def test_static_assets_preserve_bytes_mime_and_length(server, dist, path, filename, content_types):
    _assert_get_and_head(server, path, (dist / "assets" / filename).read_bytes(), content_types)


@pytest.mark.parametrize("path", ["/missing.js", "/assets/", "/empty", "/browser-logs"])
def test_missing_assets_and_directory_listings_are_not_served(server, path):
    for method in ("GET", "HEAD"):
        response = _request(server, method, path)
        assert response.status == 404
        assert response.body == (b"Viewer asset not found" if method == "GET" else b"")


@pytest.mark.parametrize("path", [
    "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/assets/%2e%2e/%2e%2e/secret.txt",
])
def test_decoded_traversal_is_forbidden(server, dist, path):
    (dist.parent / "secret.txt").write_bytes(b"outside secret")
    for method in ("GET", "HEAD"):
        response = _request(server, method, path)
        assert response.status == 403
        assert b"outside secret" not in response.body


def test_symlinks_cannot_escape_dist_but_internal_targets_are_served(server, dist):
    outside = dist.parent / "dist-sibling"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"outside secret")
    (dist / "escape.txt").symlink_to(outside / "secret.txt")
    (dist / "escape-dir").symlink_to(outside, target_is_directory=True)
    (dist / "internal.css").symlink_to(dist / "assets" / "app.css")

    for path in ("/escape.txt", "/escape-dir/secret.txt"):
        for method in ("GET", "HEAD"):
            response = _request(server, method, path)
            assert response.status == 403
            assert b"outside secret" not in response.body
    _assert_get_and_head(server, "/internal.css", (dist / "assets" / "app.css").read_bytes(), {"text/css"})


@pytest.mark.parametrize(("layout", "error", "message"), [
    ("missing-dist", FileNotFoundError, "index.html"),
    ("missing-index", FileNotFoundError, "index.html"),
    ("missing-head", RuntimeError, "no <head>"),
])
def test_invalid_dist_fails_before_binding(tmp_path, monkeypatch, layout, error, message):
    directory = tmp_path / "dist"
    if layout != "missing-dist":
        directory.mkdir()
    if layout == "missing-head":
        (directory / "index.html").write_text("<html><body>viewer</body></html>")
    bind = Mock()
    monkeypatch.setattr(viewer_server.ThreadingHTTPServer, "__init__", bind)

    with pytest.raises(error, match=message):
        viewer_server.ViewerServer(("127.0.0.1", 0), directory)
    bind.assert_not_called()


def test_escaping_index_symlink_is_rejected_before_reading_or_binding(dist, monkeypatch):
    outside = dist.parent / "private.html"
    outside.write_text("<html><head></head><body>private content</body></html>")
    index = dist / "index.html"
    index.unlink()
    index.symlink_to(outside)
    read_text = Mock(side_effect=AssertionError("must not read an escaping index"))
    bind = Mock()
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(viewer_server.ThreadingHTTPServer, "__init__", bind)

    with pytest.raises(RuntimeError, match="outside the viewer directory"):
        viewer_server.ViewerServer(("127.0.0.1", 0), dist)
    read_text.assert_not_called()
    bind.assert_not_called()


def test_internal_index_symlink_keeps_diagnostics_on_index_routes(dist, request):
    index = dist / "index.html"
    target = dist / "assets" / "viewer.html"
    index.rename(target)
    original = target.read_bytes()
    index.symlink_to(target)
    # Construct the server only after the fixture's index becomes a symlink.
    server = request.getfixturevalue("server")
    injected = _request(server, "GET", "/").body
    assert _SCRIPT_TAG.encode() in injected
    for path in ("/", "/index.html", "/assets/../index.html", "/%69ndex.html", "/assets/viewer.html"):
        _assert_get_and_head(server, path, injected, {"text/html; charset=utf-8"})
    assert target.read_bytes() == original
    assert index.is_symlink()


def test_request_socket_has_a_bounded_read_timeout(server, monkeypatch):
    timeouts = []
    setup = viewer_server.ViewerHandler.setup

    def observe_setup(handler):
        setup(handler)
        timeouts.append(handler.connection.gettimeout())

    monkeypatch.setattr(viewer_server.ViewerHandler, "setup", observe_setup)
    assert _request(server, "GET", "/").status == 200
    assert timeouts == [5]


def test_server_close_terminates_incomplete_upload_worker_and_socket(server, monkeypatch):
    entered = threading.Event()
    finished = threading.Event()
    workers = []
    post = viewer_server.ViewerHandler.do_POST

    def observe_post(handler):
        # Only closing the socket, not an idle read deadline, can end this upload.
        handler.connection.settimeout(None)
        workers.append(threading.current_thread())
        entered.set()
        try:
            post(handler)
        finally:
            finished.set()

    monkeypatch.setattr(viewer_server.ViewerHandler, "do_POST", observe_post)
    host = "{}:{}".format(*server.server_address)
    request = (
        "POST /browser-logs HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Origin: http://{host}\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 100\r\n\r\n{"
    ).encode()
    try:
        with socket.create_connection(server.server_address, timeout=5) as connection:
            connection.sendall(request)
            entered.wait()
            assert len(workers) == 1
            assert not workers[0].daemon
            server.shutdown()
            server.server_close()
            assert finished.is_set()
            assert not workers[0].is_alive()
            assert connection.recv(1) == b""
    finally:
        for worker in workers:
            worker.join()


def test_body_read_timeout_is_rejected_without_diagnostics(server, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=viewer_server.logger.name)
    post = viewer_server.ViewerHandler.do_POST

    def interrupt_body_read(handler):
        handler.rfile = Mock(wraps=handler.rfile)
        handler.rfile.read.side_effect = TimeoutError
        post(handler)

    monkeypatch.setattr(viewer_server.ViewerHandler, "do_POST", interrupt_body_read)
    response = _request(server, "POST", "/browser-logs", body=json.dumps(_ENTRY).encode())
    assert response.status == 408
    assert caplog.records == []


def test_occupied_port_fails_at_server_construction(dist):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(OSError) as raised:
            viewer_server.ViewerServer(occupied.getsockname(), dist)
        assert raised.value.errno == errno.EADDRINUSE


@pytest.mark.parametrize(("level", "severity", "scheme"), [
    ("info", logging.INFO, "http"),
    ("warn", logging.WARNING, "http"),
    ("error", logging.ERROR, "https"),
])
def test_browser_logs_are_accepted_at_matching_severity_and_escaped(server, caplog, level, severity, scheme):
    caplog.set_level(logging.INFO, logger=viewer_server.logger.name)
    entry = {
        "level": level,
        "message": "disconnect\nforged\rlevel\t\x1b[31m\x00\x85\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}",
        "page": "/viewer\n\x1b\N{LINE SEPARATOR}",
    }
    response = _request(
        server, "POST", "/browser-logs", body=json.dumps(entry).encode(),
        headers={"Origin": f"{scheme}://{'{}:{}'.format(*server.server_address)}"},
    )
    assert response.status == 204
    assert response.body == b""
    assert response.headers["Content-Length"] == "0"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == viewer_server.logger.name
    assert record.levelno == severity
    assert record.getMessage() == f"Browser 127.0.0.1: {json.dumps(entry, ensure_ascii=True)}"
    assert record.getMessage().isascii()
    assert record.getMessage().splitlines() == [record.getMessage()]


def test_browser_log_is_emitted_to_stderr_as_one_escaped_line(server, capsys):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    viewer_server.logger.addHandler(handler)
    entry = {**_ENTRY, "message": "failed\nFORGED\r\x1b[31m"}
    try:
        response = _request(server, "POST", "/browser-logs", body=json.dumps(entry).encode())
    finally:
        viewer_server.logger.removeHandler(handler)
        handler.close()
    assert response.status == 204
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"WARNING Browser 127.0.0.1: {json.dumps(entry)}\n"


@pytest.mark.parametrize("page", ["", "p" * 256], ids=["empty-page", "maximum-page"])
def test_browser_log_limits_are_inclusive(server, caplog, page):
    caplog.set_level(logging.INFO, logger=viewer_server.logger.name)
    entry = {"level": "info", "message": "m" * 2048, "page": page}
    body = json.dumps(entry).encode()
    body += b" " * (16_384 - len(body))
    response = _request(
        server, "POST", "/browser-logs", body=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status == 204
    assert len(caplog.records) == 1
    assert json.loads(caplog.records[0].getMessage().split(": ", 1)[1]) == entry


@pytest.mark.parametrize("body", [
    pytest.param(b"{", id="malformed-json"),
    pytest.param(b"\xff", id="invalid-utf8"),
    pytest.param(b"null", id="null"),
    pytest.param(b"[]", id="array"),
    pytest.param(b"[" * 1100 + b"]" * 1100, id="excessive-nesting"),
    pytest.param(json.dumps({"level": "warn", "message": "failed"}).encode(), id="missing-field"),
    pytest.param(json.dumps({**_ENTRY, "extra": "unexpected"}).encode(), id="extra-field"),
    pytest.param(json.dumps({**_ENTRY, "level": "debug"}).encode(), id="unsupported-level"),
    pytest.param(json.dumps({**_ENTRY, "level": []}).encode(), id="nonstring-level"),
    pytest.param(json.dumps({**_ENTRY, "message": 42}).encode(), id="nonstring-message"),
    pytest.param(json.dumps({**_ENTRY, "page": {}}).encode(), id="nonstring-page"),
    pytest.param(json.dumps({**_ENTRY, "message": ""}).encode(), id="empty-message"),
    pytest.param(json.dumps({**_ENTRY, "message": "m" * 2049}).encode(), id="oversized-message"),
    pytest.param(json.dumps({**_ENTRY, "page": "p" * 257}).encode(), id="oversized-page"),
])
def test_invalid_browser_log_bodies_are_rejected_without_diagnostics(server, caplog, body):
    caplog.set_level(logging.INFO, logger=viewer_server.logger.name)
    response = _request(server, "POST", "/browser-logs", body=body)
    assert response.status == 400
    assert caplog.records == []


def test_incomplete_body_containing_valid_json_is_rejected_without_diagnostics(server, caplog):
    caplog.set_level(logging.INFO, logger=viewer_server.logger.name)
    body = json.dumps(_ENTRY).encode()
    response = _request(
        server, "POST", "/browser-logs", body=body,
        headers={"Content-Length": str(len(body) + 1)}, end_body=True,
    )
    assert response.status == 400
    assert caplog.records == []


def test_oversized_body_is_rejected_without_diagnostics(server, caplog):
    caplog.set_level(logging.INFO, logger=viewer_server.logger.name)
    body = json.dumps(_ENTRY).encode()
    body += b" " * (16_385 - len(body))
    response = _request(server, "POST", "/browser-logs", body=body)
    assert response.status == 413
    assert caplog.records == []


@pytest.mark.parametrize(("name", "value", "status"), [
    ("Content-Type", None, 415),
    ("Content-Type", "text/plain", 415),
    ("Origin", None, 403),
    ("Origin", "null", 403),
    ("Origin", "http://other.example", 403),
    ("Host", None, 403),
    ("Transfer-Encoding", "chunked", 400),
    ("Transfer-Encoding", "", 400),
    ("Content-Length", None, 411),
    ("Content-Length", "invalid", 400),
    ("Content-Length", "1.5", 400),
    ("Content-Length", "-1", 400),
    ("Content-Length", "0", 400),
])
def test_invalid_browser_log_headers_are_rejected_without_diagnostics(server, caplog, name, value, status):
    caplog.set_level(logging.INFO, logger=viewer_server.logger.name)
    response = _request(
        server, "POST", "/browser-logs", body=json.dumps(_ENTRY).encode(),
        headers={name: value},
    )
    assert response.status == status
    assert caplog.records == []


def test_same_host_with_different_origin_port_is_rejected(server, caplog):
    caplog.set_level(logging.INFO, logger=viewer_server.logger.name)
    host, port = server.server_address
    other_port = port % 65535 + 1
    response = _request(
        server, "POST", "/browser-logs", body=json.dumps(_ENTRY).encode(),
        headers={"Origin": f"http://{host}:{other_port}"},
    )
    assert response.status == 403
    assert caplog.records == []


@pytest.mark.parametrize(("method", "path", "status"), [
    ("POST", "/", 404),
    ("POST", "/browser-logs.js", 404),
    ("POST", "/browser-logs?extra=1", 404),
    ("HEAD", "/browser-logs", 404),
    ("PUT", "/browser-logs", 501),
    ("DELETE", "/browser-logs", 501),
    ("OPTIONS", "/browser-logs", 501),
])
def test_other_routes_and_methods_do_not_accept_browser_logs(server, caplog, method, path, status):
    caplog.set_level(logging.INFO, logger=viewer_server.logger.name)
    response = _request(server, method, path, body=json.dumps(_ENTRY).encode())
    assert response.status == status
    assert caplog.records == []
