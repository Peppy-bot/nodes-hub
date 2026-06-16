import queue
import threading

from uvc_camera_python_mock.frames import force_put, offer_latest_frame

_SENTINEL = object()


def test_enqueues_when_room_available():
    q: queue.Queue = queue.Queue(maxsize=2)

    offer_latest_frame(q, b"a")
    offer_latest_frame(q, b"b")

    assert q.get_nowait() == b"a"
    assert q.get_nowait() == b"b"


def test_drops_oldest_and_keeps_newest_when_full():
    q: queue.Queue = queue.Queue(maxsize=2)

    for frame in (b"1", b"2", b"3", b"4"):
        offer_latest_frame(q, frame)

    # Only the two most recent frames survive; the older ones were dropped.
    assert q.get_nowait() == b"3"
    assert q.get_nowait() == b"4"
    assert q.empty()


def test_never_blocks_when_consumer_is_stalled():
    # A stalled consumer never drains the queue. The decoder must keep making
    # progress regardless: offering far more frames than the queue can hold must
    # not block and must leave the queue bounded at its maxsize.
    q: queue.Queue = queue.Queue(maxsize=2)

    for i in range(1000):
        offer_latest_frame(q, i.to_bytes(2, "big"))

    assert q.qsize() == 2
    assert q.get_nowait() == (998).to_bytes(2, "big")
    assert q.get_nowait() == (999).to_bytes(2, "big")


def test_force_put_delivers_on_empty_queue():
    q: queue.Queue = queue.Queue(maxsize=2)

    force_put(q, _SENTINEL)

    assert q.get_nowait() is _SENTINEL


def test_force_put_delivers_even_when_full():
    # The sentinel must reach the consumer no matter how full the queue is, so
    # force_put empties it first. A plain put_nowait would raise queue.Full and
    # the sentinel would be lost, leaving the consumer parked forever.
    q: queue.Queue = queue.Queue(maxsize=2)
    q.put_nowait(b"frame-1")
    q.put_nowait(b"frame-2")

    force_put(q, _SENTINEL)

    assert q.get_nowait() is _SENTINEL
    assert q.empty()


def test_force_put_wakes_a_parked_consumer():
    # Mirrors the shutdown path: a consumer blocked in frame_queue.get must be
    # woken by the sentinel rather than blocking until interpreter exit.
    q: queue.Queue = queue.Queue(maxsize=2)
    received: list = []

    def consumer():
        received.append(q.get(timeout=5))

    t = threading.Thread(target=consumer)
    t.start()
    force_put(q, _SENTINEL)
    t.join(timeout=5)

    assert not t.is_alive()
    assert received == [_SENTINEL]
