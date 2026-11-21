//! The node's clock: the wall clock, or the simulation clock the launcher binds the
//! node to, fed the ticks of its domain, which the limiter's steps and the wait of each
//! placement by the robot's tree read, in integer nanoseconds as control_core's limiter
//! compares times.

use std::sync::Arc;
use std::time::SystemTime;

use control_core::fault::Fault;
use control_core::node_clock::{NodeClock, follow_ticks};
use frame_tree::Stamp;
use peppygen::NodeRunner;
use peppylib::messaging::TopicBuffer;
use peppylib::runtime::CancellationToken;

/// Why the node ends when its simulation clock closes.
const CLOCK_CLOSED: &str = "the simulation clock of the node closed: no placement of a command \
                            waits past its last tick, so the node exits for its restart policy \
                            to subscribe again";

/// The clock the launcher binds the node to. The end of a simulation clock's ticks raises
/// `fault`, since the placement of each command reads that clock.
pub(crate) async fn node_clock(
    runner: &Arc<NodeRunner>,
    fault: Fault,
) -> peppygen::Result<NodeClock> {
    if runner.processor().clock().domain().is_none() {
        return Ok(NodeClock::wall());
    }
    let mut subscription = peppylib::clock::subscribe(runner, TopicBuffer::KeepLatest).await?;
    let (clock, feed) = NodeClock::simulation();
    tokio::spawn(follow_ticks(
        async move || {
            subscription
                .on_next_tick()
                .await
                .map(|tick| tick.map(|tick| tick.time()))
        },
        feed,
        clock_closed(fault, runner.cancellation_token().clone()),
    ));
    Ok(clock)
}

/// What the end of the simulation clock's ticks does: it raises `fault` and stops the node
/// that `stopping` cancels.
fn clock_closed(fault: Fault, stopping: CancellationToken) -> impl FnOnce() {
    move || fault.raise(CLOCK_CLOSED, || stopping.cancel())
}

/// The node's clock now, in nanoseconds, or `None` while it reads no time:
/// on a simulated domain before its first tick.
pub fn now_ns() -> Option<i64> {
    peppygen::clock::now_ns()
        .ok()
        .and_then(|now| i64::try_from(now).ok())
}

/// A wire time as nanoseconds on the node's clock, as its [`Stamp`] counts
/// them, or `None` for a time no stamp holds: one before the clock's origin.
pub fn ns_from_wire(time: SystemTime) -> Option<i64> {
    Stamp::from_time(time)
        .ok()
        .and_then(|stamp| i64::try_from(stamp.as_nanos()).ok())
}

/// Nanoseconds on the node's clock as a wire time: the time of their
/// [`Stamp`], the clock's origin for a time before it.
pub fn wire_from_ns(time_ns: i64) -> SystemTime {
    Stamp::from_nanos(time_ns.max(0).unsigned_abs()).time()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wire_time_and_its_nanoseconds_round_trip() {
        let time_ns = 1_234_567_891_011;
        assert_eq!(ns_from_wire(wire_from_ns(time_ns)), Some(time_ns));
    }

    #[test]
    fn a_time_before_the_clock_origin_reads_none() {
        let before = SystemTime::UNIX_EPOCH - std::time::Duration::from_secs(1);
        assert_eq!(ns_from_wire(before), None);
    }

    #[tokio::test]
    async fn the_end_of_the_simulation_clocks_ticks_raises_the_fault_and_stops_the_node() {
        let fault = Fault::default();
        let stopping = CancellationToken::new();
        let (clock, feed) = NodeClock::simulation();
        let mut ticks = vec![Ok::<_, String>(Some(5)), Ok(None)].into_iter();
        follow_ticks(
            async || ticks.next().expect("a receive past the end"),
            feed,
            clock_closed(fault.clone(), stopping.clone()),
        )
        .await;
        assert_eq!(clock.now_ns(), 5);
        assert!(stopping.is_cancelled(), "the node stops");
        assert_eq!(
            fault.take().map(|reason| reason.to_string()),
            Some(CLOCK_CLOSED.to_string())
        );
    }
}
