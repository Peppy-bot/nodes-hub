"""g1_audio_node entry point: exposes the SDK AudioClient as peppy services.

The AudioClient is synchronous, so each service call runs through one
single-worker executor (the sole audio writer). setup() spawns one serve loop
per service and returns them so the runtime owns the tasks.
"""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor

from peppygen import NodeBuilder, NodeRunner
from peppygen.exposed_services import (
    get_volume,
    play_audio,
    play_stop,
    set_led,
    set_volume,
    tts,
)
from peppygen.parameters import Parameters

from .backend import AudioBackend, AudioClientBackend

SDK_CALL_TIMEOUT = 12.0


class AudioController:
    """Serializes AudioClient access through one worker."""

    def __init__(self, backend: AudioBackend) -> None:
        self._backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="g1-audio")

    @property
    def backend(self) -> AudioBackend:
        return self._backend

    def call_sync(self, fn, *args):
        return self._executor.submit(fn, *args).result(timeout=SDK_CALL_TIMEOUT)

    def close(self) -> None:
        self._executor.shutdown(wait=False)


def _build_service_handlers(controller: AudioController):
    b = controller.backend

    def issued(service, fn, *args):
        controller.call_sync(fn, *args)
        return service.Response(ok=True)

    def do_play(request):
        pcm = base64.b64decode(request.data.pcm_base64)
        controller.call_sync(b.play_stream, request.data.app_name, request.data.stream_id, pcm)
        return play_audio.Response(ok=True)

    return [
        (tts, lambda r: issued(tts, b.tts, r.data.text, r.data.speaker_id)),
        (set_volume, lambda r: issued(set_volume, b.set_volume, r.data.volume)),
        (set_led, lambda r: issued(set_led, b.set_led, r.data.r, r.data.g, r.data.b)),
        (play_stop, lambda r: issued(play_stop, b.play_stop, r.data.app_name)),
        (play_audio, do_play),
        (get_volume, lambda _r: get_volume.Response(volume=controller.call_sync(b.get_volume))),
    ]


async def _serve(node_runner: NodeRunner, service, handler, name: str, token):
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            req = asyncio.ensure_future(service.handle_next_request(node_runner, handler))
            await asyncio.wait([cancelled, req], return_when=asyncio.FIRST_COMPLETED)
            if not req.done():
                req.cancel()
                break
            try:
                req.result()
            except Exception as exc:
                print(f"[g1-audio] service {name} error: {exc!r}")
    finally:
        cancelled.cancel()


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    backend = AudioClientBackend(params.network_interface, params.dds_domain_id)
    controller = AudioController(backend)

    async def on_shutdown():
        controller.close()

    node_runner.on_shutdown(on_shutdown)

    token = node_runner.cancellation_token()
    tasks = []
    for service, handler in _build_service_handlers(controller):
        name = service.__name__.rsplit(".", 1)[-1]
        tasks.append(asyncio.create_task(_serve(node_runner, service, handler, name, token)))
    return tasks


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
