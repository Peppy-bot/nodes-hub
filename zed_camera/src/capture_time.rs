//! The instant a frame was captured, on the node's clock. V4L2 stamps each buffer by the
//! clock its flags name, the monotonic one for a UVC camera such as the ZED (see
//! `capture`). The node reads its own clock and the monotonic clock together when it
//! dequeues the buffer, and carries the stamp across.

use std::time::{Duration, SystemTime};

/// The node clock's reading when the monotonic clock read `captured`, from both clocks
/// read together: the node clock at `now`, the monotonic clock at `monotonic_now`. A
/// stamp after `monotonic_now` is by another clock, such as uvcvideo's under its
/// `clock=realtime`, and gives none.
pub fn captured_at(
    now: SystemTime,
    monotonic_now: Duration,
    captured: Duration,
) -> Result<SystemTime, String> {
    let age = monotonic_now.checked_sub(captured).ok_or_else(|| {
        format!(
            "the driver stamps a frame {:?} after the monotonic clock's reading at its \
             dequeue: it stamps by another clock than the monotonic one",
            captured - monotonic_now
        )
    })?;
    now.checked_sub(age)
        .ok_or_else(|| format!("the frame's age {age:?} reaches before the node clock's range"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A frame captured 40 ms before the dequeue is stamped 40 ms before the node's
    // clock read at the dequeue; a stamp from after the dequeue gives none, and says how
    // far after it lies.
    #[test]
    fn the_capture_instant_is_the_dequeue_less_the_time_since_capture() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
        let monotonic_now = Duration::from_secs(500);
        let delay = Duration::from_millis(40);
        let early = monotonic_now - delay;
        let late = monotonic_now + Duration::from_millis(5);
        assert_eq!(captured_at(now, monotonic_now, early), Ok(now - delay));
        assert_eq!(
            captured_at(now, monotonic_now, late),
            Err(
                "the driver stamps a frame 5ms after the monotonic clock's reading at its \
                 dequeue: it stamps by another clock than the monotonic one"
                    .to_string()
            )
        );
    }
}
