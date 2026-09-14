"""Serve the compiled Isaac viewer and receive its browser diagnostics."""

from __future__ import annotations

import json
import logging
import mimetypes
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)

_MAX_BODY_BYTES = 16_384
_MAX_MESSAGE_CHARS = 2_048
_MAX_PAGE_CHARS = 256
_LEVELS = {"info": logging.INFO, "warn": logging.WARNING, "error": logging.ERROR}
_SCRIPT_TAG = '<script src="/browser-logs.js"></script>'


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, dist_dir: Path = Path("/app/dist")):
        self._requests = set()
        self._requests_lock = threading.Lock()
        self._closing = False
        self.dist_dir = dist_dir.resolve()
        index_path = (self.dist_dir / "index.html").resolve()
        if not index_path.is_relative_to(self.dist_dir):
            raise RuntimeError("Viewer index.html is outside the viewer directory")
        self.index_path = index_path
        index = index_path.read_text(encoding="utf-8")
        if "<head>" not in index:
            raise RuntimeError("Viewer index.html has no <head> for browser diagnostics")
        self.index = index.replace("<head>", "<head>\n    " + _SCRIPT_TAG, 1).encode()
        self.browser_script = Path(__file__).with_name("browser_logs.js").read_bytes()
        super().__init__(address, ViewerHandler)

    def process_request(self, request, client_address):
        with self._requests_lock:
            self._requests.add(request)
        super().process_request(request, client_address)

    def shutdown_request(self, request):
        try:
            super().shutdown_request(request)
        finally:
            with self._requests_lock:
                self._requests.discard(request)

    def server_close(self):
        self._closing = True
        with self._requests_lock:
            requests = tuple(self._requests)
        # Interrupt uploads and idle sockets before joining the request workers.
        for request in requests:
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        super().server_close()

    def handle_error(self, request, client_address):
        if not self._closing:
            super().handle_error(request, client_address)


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer

    def setup(self):
        self.request.settimeout(5)
        super().setup()

    def log_message(self, fmt, *args):
        logger.debug("Viewer HTTP %s: %s", self.client_address[0], json.dumps(fmt % args))

    def _respond(self, status, body=b"", content_type="text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            path = unquote(urlsplit(self.path).path)
            if path == "/browser-logs.js":
                self._respond(200, self.server.browser_script, "text/javascript; charset=utf-8")
                return
            asset = (self.server.dist_dir / path.lstrip("/")).resolve()
            if not asset.is_relative_to(self.server.dist_dir):
                self._respond(403, b"Asset path is outside the viewer directory")
                return
            if asset in (self.server.dist_dir, self.server.index_path):
                self._respond(200, self.server.index, "text/html; charset=utf-8")
                return
            if not asset.is_file():
                self._respond(404, b"Viewer asset not found")
                return
            content_type = mimetypes.guess_type(asset)[0] or "application/octet-stream"
            self._respond(200, asset.read_bytes(), content_type)
        except (OSError, ValueError):
            self._respond(404, b"Viewer asset not found")

    def do_POST(self):
        if self.path != "/browser-logs":
            self._respond(404, b"Unknown endpoint")
            return
        host = self.headers.get("Host")
        if not host or self.headers.get("Origin") not in (f"http://{host}", f"https://{host}"):
            self._respond(403, b"Browser logs require a same-origin request")
            return
        if self.headers.get_content_type() != "application/json":
            self._respond(415, b"Browser logs require application/json")
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._respond(400, b"Transfer-Encoding is not supported")
            return
        if self.headers.get("Content-Length") is None:
            self._respond(411, b"Content-Length is required")
            return
        try:
            length = int(self.headers["Content-Length"])
        except ValueError:
            self._respond(400, b"Invalid Content-Length")
            return
        if length > _MAX_BODY_BYTES:
            self._respond(413, b"Browser log is too large")
            return
        if length <= 0:
            self._respond(400, b"Browser log is empty")
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                self._respond(400, b"Incomplete browser log body")
                return
            entry = json.loads(body)
        except TimeoutError:
            self._respond(408, b"Browser log body timed out")
            return
        except (ValueError, RecursionError):
            self._respond(400, b"Invalid browser log JSON")
            return
        if not (
            isinstance(entry, dict)
            and entry.keys() == {"level", "message", "page"}
            and isinstance(entry["level"], str)
            and entry["level"] in _LEVELS
            and isinstance(entry["message"], str)
            and 0 < len(entry["message"]) <= _MAX_MESSAGE_CHARS
            and isinstance(entry["page"], str)
            and len(entry["page"]) <= _MAX_PAGE_CHARS
        ):
            self._respond(400, b"Invalid browser log fields")
            return
        # Browser input stays quoted on one line, including control characters.
        logger.log(
            _LEVELS[entry["level"]],
            "Browser %s: %s",
            self.client_address[0],
            json.dumps(entry, ensure_ascii=True),
        )
        self._respond(204)
