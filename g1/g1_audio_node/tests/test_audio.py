from g1_audio_node.backend import AudioBackend, AudioClientBackend


class FakeAudio:
    """Stands in for the SDK AudioClient: records every call."""

    def __init__(self, volume=30):
        self.calls: list[tuple] = []
        self._volume = volume

    def TtsMaker(self, text, speaker_id):
        self.calls.append(("TtsMaker", text, speaker_id))

    def GetVolume(self):
        return (0, {"volume": self._volume})

    def SetVolume(self, volume):
        self.calls.append(("SetVolume", volume))

    def LedControl(self, r, g, b):
        self.calls.append(("LedControl", r, g, b))

    def PlayStream(self, app_name, stream_id, pcm):
        self.calls.append(("PlayStream", app_name, stream_id, pcm))

    def PlayStop(self, app_name):
        self.calls.append(("PlayStop", app_name))


def _audio_backend(fake):
    backend = object.__new__(AudioClientBackend)
    backend._audio = fake
    return backend


def test_audio_backend_dispatches_every_method():
    fake = FakeAudio()
    backend = _audio_backend(fake)
    backend.tts("hello", 1)
    backend.set_volume(50)
    backend.set_led(255, 0, 128)
    backend.play_stream("app", "s1", b"\x00\x01")
    backend.play_stop("app")
    assert ("TtsMaker", "hello", 1) in fake.calls
    assert ("SetVolume", 50) in fake.calls
    assert ("LedControl", 255, 0, 128) in fake.calls
    assert ("PlayStream", "app", "s1", b"\x00\x01") in fake.calls
    assert ("PlayStop", "app") in fake.calls


def test_get_volume_normalizes_tuple_and_dict():
    backend = _audio_backend(FakeAudio(volume=42))
    assert backend.get_volume() == 42


def test_get_volume_plain_int():
    class PlainAudio(FakeAudio):
        def GetVolume(self):
            return 17

    assert _audio_backend(PlainAudio()).get_volume() == 17
