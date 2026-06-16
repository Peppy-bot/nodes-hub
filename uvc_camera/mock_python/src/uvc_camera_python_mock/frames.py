"""Frame hand-off helpers shared between the decoder thread and the emitter.

Kept dependency-free (standard library only) so the drop-oldest logic, which is
what keeps a stalled consumer from wedging the decoder, can be unit tested
without pulling in PyAV or the peppy runtime.
"""

import queue


def offer_latest_frame(frame_queue: "queue.Queue", data) -> None:
    """Enqueue `data`, dropping the oldest frame if the queue is full.

    The decoder uses this instead of a blocking `put` so a slow or stalled
    consumer can never apply backpressure that wedges the decoder. The newest
    frame always wins, which is the right trade-off for a live camera stream.
    """
    while True:
        try:
            frame_queue.put_nowait(data)
            return
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass


def force_put(frame_queue: "queue.Queue", item) -> None:
    """Enqueue `item`, first emptying the queue so the put can never be lost to
    a `queue.Full`.

    Used to hand a consumer parked in `frame_queue.get` the shutdown sentinel:
    the consumer must be woken immediately regardless of how full the queue is,
    so that its (non-daemon) worker thread is not left blocked at interpreter
    exit. Dropping the queued frames is harmless: the newest frame always wins
    for a live stream, and at shutdown they are discarded anyway.
    """
    while True:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    try:
        frame_queue.put_nowait(item)
    except queue.Full:
        # A racing producer refilled the queue between the drain and the put.
        # The consumer's bounded get re-checks the cancellation token, so it
        # still exits promptly even if this sentinel is dropped.
        pass
