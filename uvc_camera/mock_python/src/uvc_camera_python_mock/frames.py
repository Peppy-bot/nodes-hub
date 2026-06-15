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
