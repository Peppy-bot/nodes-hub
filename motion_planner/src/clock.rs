//! The node's clock, which the wait of each depth frame for the robot's tree reads: the
//! wall clock, or the simulation clock the launcher binds the node to, fed the ticks of its
//! domain.

use std::sync::Arc;

use control_core::fault::Fault;
use control_core::node_clock::{NodeClock, follow_ticks};
use peppygen::NodeRunner;
use peppylib::messaging::TopicBuffer;
use peppylib::runtime::CancellationToken;

/// Why the node ends when its simulation clock closes.
const CLOCK_CLOSED: &str = "the simulation clock of the node closed: no wait of a depth frame \
                            ends past its last tick, so the node exits for its restart policy \
                            to subscribe again";

/// The clock the launcher binds the node to. The end of a simulation clock's ticks raises
/// `fault`, since the wait of each depth frame reads that clock.
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

#[cfg(test)]
mod tests {
    use super::*;

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
