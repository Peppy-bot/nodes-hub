"""An OpenAI-compatible API inside the test process, on 127.0.0.1, for the
node's real any-llm client: `http.server` in a thread of its own, so it
needs no async code.

It records each request. It answers a chat completion of a model it serves
at once, with the next of the answers a test queued, `{"boxes": []}` once
they run out, and a chat completion of any other model with a 404 and an
error in OpenAI's form. The poll interval of its thread only bounds how
long `close` waits for the thread to stop: no assertion waits on it.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable

NOTHING = '{"boxes": []}'
CHAT_COMPLETIONS = "/v1/chat/completions"
# How often the server's thread looks whether `close` asked it to stop.
POLL_INTERVAL_S = 0.05


@dataclass(frozen=True)
class Request:
    """One request the API received: its path, its headers by lower-case
    name, and its JSON body."""

    path: str
    headers: dict[str, str]
    body: dict

    @property
    def question(self) -> str:
        """The text of the request's one user message."""
        [message] = self.body["messages"]
        return next(part["text"] for part in message["content"] if part["type"] == "text")


class FakeApi:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: list[Request] = []
        self._answers: list[str] = []
        self._served: frozenset[str] = frozenset()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever, args=(POLL_INTERVAL_S,), daemon=True)
        self._thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def requests(self) -> list[Request]:
        with self._lock:
            return list(self._requests)

    def reset(self, served: Iterable[str]) -> None:
        """No request recorded, no answer queued, and `served` the models
        the API serves."""
        with self._lock:
            self._requests.clear()
            self._answers.clear()
            self._served = frozenset(served)

    def answer(self, *texts: str) -> None:
        """Queues `texts`, the contents of the next answers, in order."""
        with self._lock:
            self._answers.extend(texts)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def reply(self, request: Request) -> tuple[int, dict]:
        """Records `request` and gives its status and its JSON answer."""
        with self._lock:
            self._requests.append(request)
            model = request.body.get("model")
            if request.path != CHAT_COMPLETIONS:
                return 404, _error(f"no route for {request.path}", None)
            if model not in self._served:
                return 404, _error(f"model '{model}' not found", "model_not_found")
            text = self._answers.pop(0) if self._answers else NOTHING
        return 200, {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }


def _error(message: str, code) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}}


def _handler(api: FakeApi) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            headers = {name.lower(): value for name, value in self.headers.items()}
            status, payload = api.reply(Request(self.path, headers, body))
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args) -> None:  # pylint: disable=redefined-builtin
            """Quiet: each request is recorded, not printed."""

    return Handler
