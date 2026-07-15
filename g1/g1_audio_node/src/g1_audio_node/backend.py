"""The G1 audio backend: an abstract seam plus the real AudioClient wrapper.

The node logic depends only on `AudioBackend`, so it is exercised in tests with
an in-memory fake and, on hardware, with `AudioClientBackend`. The Unitree SDK is
imported lazily and guarded, so the module loads (and the node builds) without
the SDK or its CycloneDDS backend present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AudioBackend(ABC):
    @abstractmethod
    def tts(self, text: str, speaker_id: int) -> None: ...

    @abstractmethod
    def get_volume(self) -> int: ...

    @abstractmethod
    def set_volume(self, volume: int) -> None: ...

    @abstractmethod
    def set_led(self, r: int, g: int, b: int) -> None: ...

    @abstractmethod
    def play_stream(self, app_name: str, stream_id: str, pcm: bytes) -> None: ...

    @abstractmethod
    def play_stop(self, app_name: str) -> None: ...


class SdkUnavailable(RuntimeError):
    """Raised when the real backend is requested but the Unitree SDK is absent."""


class AudioClientBackend(AudioBackend):
    """Wraps the Unitree SDK AudioClient over CycloneDDS."""

    def __init__(self, network_interface: str, domain_id: int) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        except ImportError as exc:
            raise SdkUnavailable(
                "unitree_sdk2py is not installed; build the node with the "
                "`hardware` extra (`uv sync --extra hardware`) to reach a robot"
            ) from exc

        ChannelFactoryInitialize(domain_id, network_interface)
        self._audio = AudioClient()
        self._audio.SetTimeout(10.0)
        self._audio.Init()

    def tts(self, text: str, speaker_id: int) -> None:
        self._audio.TtsMaker(text, speaker_id)

    def get_volume(self) -> int:
        volume = self._audio.GetVolume()
        # Some SDK builds return (code, data); normalize to the level.
        if isinstance(volume, tuple):
            volume = volume[1]
        if isinstance(volume, dict):
            volume = volume.get("volume", 0)
        return int(volume)

    def set_volume(self, volume: int) -> None:
        self._audio.SetVolume(volume)

    def set_led(self, r: int, g: int, b: int) -> None:
        self._audio.LedControl(r, g, b)

    def play_stream(self, app_name: str, stream_id: str, pcm: bytes) -> None:
        self._audio.PlayStream(app_name, stream_id, pcm)

    def play_stop(self, app_name: str) -> None:
        self._audio.PlayStop(app_name)
