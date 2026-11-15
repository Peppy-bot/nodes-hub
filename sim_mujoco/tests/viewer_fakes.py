"""viser and mjviser, faked for the suites that show a scene: the suites
render nothing, and neither library is installed where they run.

The server holds what a scene leaves on it (what it shows, its controls, the
callbacks it registers), so a suite reads what stays on a server once a scene
is wiped off it.
"""

from __future__ import annotations

import types


class Viser:
    """viser's server as the engine uses it, serving nobody."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._host, self._port = host, port
        self.stopped = False
        # What a scene leaves on the server.
        self.nodes: list[str] = []
        self.controls: list[str] = []
        self.click_callbacks: list = []
        # The callbacks viser runs for a browser that connects, under the
        # name viser itself holds them by.
        self.connect_callbacks: list = []
        self._client_connect_cb = self.connect_callbacks
        # The browsers open on the viewer: viser runs a connect callback for
        # each of them as soon as the callback is registered.
        self.clients: list = []
        # What was done on the server, in order, for a suite that holds one
        # step to come after another.
        self.events: list[str] = []
        self.scene = types.SimpleNamespace(
            reset=self._reset_scene,
            on_click=self._on_click,
            remove_click_callback=self._remove_click_callback,
        )
        self.gui = types.SimpleNamespace(reset=self.controls.clear, add_button=self._add_button)

    def get_host(self) -> str:
        return self._host

    def get_port(self) -> int:
        return self._port

    def on_client_connect(self, handler):
        self.connect_callbacks.append(handler)
        for client in self.clients:
            handler(client)
        return handler

    def stop(self) -> None:
        self.events.append("stop")
        self.stopped = True

    def left_on_it(self) -> list:
        """Everything a scene left on the server."""
        return [*self.nodes, *self.controls, *self.click_callbacks, *self.connect_callbacks]

    def _reset_scene(self) -> None:
        self.events.append("wipe")
        self.nodes.clear()

    def _on_click(self):
        def register(callback):
            self.click_callbacks.append(callback)
            return callback

        return register

    def _remove_click_callback(self, callback) -> None:
        assert callback == "all", "the engine takes every click callback of a scene back"
        self.click_callbacks.clear()

    def _add_button(self, label: str):
        self.controls.append(label)
        return types.SimpleNamespace(on_click=lambda handler: handler)


class Viewer:
    """mjviser's viewer as the browser view drives it. It leaves on the
    server what mjviser's leaves there: the bodies it shows, its controls, a
    click callback and a connect callback. A tick steps the scene through the
    callback the view handed it, and nothing is drawn."""

    def __init__(self, model, data, server, step_fn) -> None:
        self._model, self._data, self._server, self._step_fn = model, data, server, step_fn
        server.events.append("draw")
        server.nodes.append("/fixed_bodies")
        server.scene.on_click()(lambda _event: None)

    def _tick(self) -> None:
        self._step_fn(self._model, self._data)

    def _setup_gui(self) -> None:
        self._server.controls.append("Controls")
        self._server.on_client_connect(lambda _client: None)

    def _render(self) -> None:
        self._server.events.append("render")

    def _refresh_scene_from_gui(self) -> None:
        self._server.events.append("refresh")
