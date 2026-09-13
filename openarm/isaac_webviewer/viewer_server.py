"""Serve the compiled Isaac viewer from the base image's dist directory."""

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
        if not index_path.is_file():
            raise FileNotFoundError(f"Viewer index.html not found at {index_path}")
        self.index_path = index_path
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
        # Interrupt stalled and idle sockets before joining the request workers.
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
        # Browser requests are not what the node log is for; keep them at debug.
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
            asset = (self.server.dist_dir / path.lstrip("/")).resolve()
            if not asset.is_relative_to(self.server.dist_dir):
                self._respond(403, b"Asset path is outside the viewer directory")
                return
            if asset == self.server.dist_dir:
                asset = self.server.index_path
            if not asset.is_file():
                self._respond(404, b"Viewer asset not found")
                return
            if asset == self.server.index_path:
                content_type = "text/html; charset=utf-8"
            else:
                content_type = mimetypes.guess_type(asset)[0] or "application/octet-stream"
            self._respond(200, asset.read_bytes(), content_type)
        except (OSError, ValueError):
            self._respond(404, b"Viewer asset not found")
