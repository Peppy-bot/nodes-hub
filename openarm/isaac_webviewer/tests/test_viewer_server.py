"""Exercise the viewer over HTTP without its container, browser, or GPU."""

import errno
import importlib.util
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
def test_index_routes_serve_the_dist_index_unchanged(server, dist, path):
    original_files = {file: file.read_bytes() for file in dist.rglob("*") if file.is_file()}
    _assert_get_and_head(server, path, original_files[dist / "index.html"], {"text/html; charset=utf-8"})
    assert {file: file.read_bytes() for file in dist.rglob("*") if file.is_file()} == original_files


@pytest.mark.parametrize(("path", "filename", "content_types"), [
    ("/assets/app.js", "app.js", {"text/javascript", "application/javascript"}),
    ("/assets/app.css?theme=dark", "app.css", {"text/css"}),
    ("/assets/logo.svg", "logo.svg", {"image/svg+xml"}),
    ("/assets/robot%20arm.txt", "robot arm.txt", {"text/plain"}),
    ("/assets/data.unknown-viewer-extension", "data.unknown-viewer-extension", {"application/octet-stream"}),
])
def test_static_assets_preserve_bytes_mime_and_length(server, dist, path, filename, content_types):
    _assert_get_and_head(server, path, (dist / "assets" / filename).read_bytes(), content_types)


@pytest.mark.parametrize("path", ["/missing.js", "/assets/", "/empty"])
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


@pytest.mark.parametrize("layout", ["missing-dist", "missing-index"])
def test_invalid_dist_fails_before_binding(tmp_path, monkeypatch, layout):
    directory = tmp_path / "dist"
    if layout != "missing-dist":
        directory.mkdir()
    bind = Mock()
    monkeypatch.setattr(viewer_server.ThreadingHTTPServer, "__init__", bind)

    with pytest.raises(FileNotFoundError, match="index.html"):
        viewer_server.ViewerServer(("127.0.0.1", 0), directory)
    bind.assert_not_called()


def test_escaping_index_symlink_is_rejected_before_binding(dist, monkeypatch):
    outside = dist.parent / "private.html"
    outside.write_text("<html><head></head><body>private content</body></html>")
    index = dist / "index.html"
    index.unlink()
    index.symlink_to(outside)
    bind = Mock()
    monkeypatch.setattr(viewer_server.ThreadingHTTPServer, "__init__", bind)

    with pytest.raises(RuntimeError, match="outside the viewer directory"):
        viewer_server.ViewerServer(("127.0.0.1", 0), dist)
    bind.assert_not_called()


def test_internal_index_symlink_serves_its_target_on_index_routes(dist, request):
    index = dist / "index.html"
    target = dist / "assets" / "viewer.html"
    index.rename(target)
    original = target.read_bytes()
    index.symlink_to(target)
    # Construct the server only after the fixture's index becomes a symlink.
    server = request.getfixturevalue("server")
    for path in ("/", "/index.html", "/assets/../index.html", "/%69ndex.html", "/assets/viewer.html"):
        _assert_get_and_head(server, path, original, {"text/html; charset=utf-8"})
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


def test_server_close_terminates_a_stalled_request_worker_and_socket(server, monkeypatch):
    entered = threading.Event()
    finished = threading.Event()
    workers = []
    setup = viewer_server.ViewerHandler.setup
    finish = viewer_server.ViewerHandler.finish

    def observe_setup(handler):
        setup(handler)
        # Only closing the socket, not an idle read deadline, can end this request.
        handler.connection.settimeout(None)
        workers.append(threading.current_thread())
        entered.set()

    def observe_finish(handler):
        try:
            finish(handler)
        finally:
            finished.set()

    monkeypatch.setattr(viewer_server.ViewerHandler, "setup", observe_setup)
    monkeypatch.setattr(viewer_server.ViewerHandler, "finish", observe_finish)
    host = "{}:{}".format(*server.server_address)
    # Headers never end, so the worker sits in the request parser.
    request = f"GET / HTTP/1.1\r\nHost: {host}\r\n".encode()
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


def test_occupied_port_fails_at_server_construction(dist):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(OSError) as raised:
            viewer_server.ViewerServer(occupied.getsockname(), dist)
        assert raised.value.errno == errno.EADDRINUSE


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "OPTIONS"])
def test_only_get_and_head_are_served(server, method):
    response = _request(server, method, "/", body=b"{}")
    assert response.status == 501
