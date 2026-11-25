//! The node's clock: the wall clock, or the simulation clock the launcher binds the
//! node to, fed the ticks of its domain. Every time the node reads is this clock: the
//! steps of the control loop, the arrivals of drive_states and of each sensor's
//! readings, the waits of the placements by the robot's tree and of the admission, in
//! integer nanoseconds as navigation_core's checks compare times. Each setpoint the node
//! streams carries its step's time on this clock.

use std::sync::Arc;

use control_core::fault::Fault;
use control_core::node_clock::{NodeClock, follow_ticks};
use peppygen::NodeRunner;
use peppylib::messaging::TopicBuffer;
use peppylib::runtime::CancellationToken;

/// Why the node ends when its simulation clock closes.
const CLOCK_CLOSED: &str = "the simulation clock of the node closed: no step of the navigator \
                            runs past its last tick, so the node exits for its restart policy \
                            to subscribe again";

/// The clock the launcher binds the node to. The end of a simulation clock's ticks raises
/// `fault`, since every step of the navigator reads that clock.
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

/// `clock` now, in nanoseconds as the node compares times.
pub fn now_ns(clock: &NodeClock) -> i64 {
    i64::try_from(clock.now_ns()).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_simulation_clock_reads_its_newest_tick() {
        let (clock, feed) = NodeClock::simulation();
        assert_eq!(now_ns(&clock), 0, "before its first tick");
        feed.tick(5_000);
        assert_eq!(now_ns(&clock), 5_000);
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
