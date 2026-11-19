//! The run clock of the director: wall time, or the simulation clock the
//! launcher binds it to, and the watch for a simulation clock that stops
//! while a run goes.

use std::collections::BTreeSet;
use std::time::Duration;

use std::sync::Arc;

use control_core::node_clock::{NodeClock, follow_ticks};
use peppygen::NodeRunner;
use peppylib::messaging::TopicBuffer;
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, watch};
use workflow_documents::Name;

use crate::Fault;
use crate::engine::failure::{DirectorKind, Failure};
use crate::engine::{Phase, RunCommand};

/// How long the simulation clock may give no tick, in wall time, before a
/// running lane fails with `clock_stopped`, and before a run that waits for
/// its first tick is refused.
const CLOCK_STOP: Duration = Duration::from_secs(10);

/// The clock of the director.
#[derive(Clone)]
pub enum ClockSource {
    Wall,
    /// The simulation clock, 0 before its first tick, and its domain as the
    /// fleet listing names it.
    Simulation {
        clock: NodeClock,
        domain: String,
    },
}

impl ClockSource {
    /// The clock the launcher binds the director to, with its domain for a simulation
    /// clock. The end of a simulation clock's ticks raises `fault`, since the runs of the
    /// node read that clock.
    pub async fn of(runner: &Arc<NodeRunner>, fault: Fault) -> peppygen::Result<ClockSource> {
        let clock = node_clock(runner, fault).await?;
        Ok(match runner.processor().clock().domain() {
            None => ClockSource::Wall,
            Some(domain) => ClockSource::Simulation {
                clock,
                domain: domain.to_string(),
            },
        })
    }
}

/// Why the node ends when its simulation clock closes.
const CLOCK_CLOSED: &str = "the simulation clock of the node closed: no run that reads it goes \
                            on past its last tick, so the node exits for its restart policy to \
                            subscribe again";

/// The clock the launcher binds the node to. The end of a simulation clock's ticks raises
/// `fault`, since the runs of the node read that clock.
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
    move || fault.raise(&stopping, CLOCK_CLOSED)
}

/// Waits up to [`CLOCK_STOP`] of wall time for the first tick of `clock`; the
/// wall clock reads its time at once.
pub async fn first_tick(clock: &NodeClock) -> Result<(), String> {
    let Some(mut ticks) = clock.ticks() else {
        return Ok(());
    };
    if ticks.has_changed().is_err() {
        return Err("the simulation clock closed".to_owned());
    }
    match tokio::time::timeout(CLOCK_STOP, ticks.wait_for(|now| *now > 0)).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(_)) => Err("the simulation clock closed".to_owned()),
        Err(_) => Err(format!(
            "the simulation clock gave no tick in {} s",
            CLOCK_STOP.as_secs()
        )),
    }
}

/// Fails each lane of `roles` with `clock_stopped` once the clock has given
/// no tick for [`CLOCK_STOP`] while the run is running, and again after
/// each new stop.
pub async fn watch_for_stops(
    mut ticks: watch::Receiver<u64>,
    phase: watch::Receiver<Phase>,
    roles: BTreeSet<Name>,
    commands: mpsc::UnboundedSender<RunCommand>,
) {
    let mut stopped = false;
    loop {
        match tokio::time::timeout(CLOCK_STOP, ticks.changed()).await {
            Ok(Ok(())) => stopped = false,
            Ok(Err(_)) => return,
            Err(_) if stopped || *phase.borrow() != Phase::Running => {}
            Err(_) => {
                stopped = true;
                let message = format!(
                    "the simulation clock gave no tick for {} s of wall time",
                    CLOCK_STOP.as_secs()
                );
                for role in &roles {
                    let failure = Failure::of(DirectorKind::ClockStopped, message.clone());
                    if commands
                        .send(RunCommand::Fault {
                            role: role.clone(),
                            failure,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::name;

    /// The `(role, kind)` of each fault that the watch sent.
    fn faults(commands: &mut mpsc::UnboundedReceiver<RunCommand>) -> Vec<(String, String)> {
        std::iter::from_fn(|| commands.try_recv().ok())
            .map(|command| match command {
                RunCommand::Fault { role, failure } => (role.to_string(), failure.kind),
                _ => panic!("the watch sends faults alone"),
            })
            .collect()
    }

    /// While the run runs, a clock that gives no tick for [`CLOCK_STOP`]
    /// fails each lane once with `clock_stopped`; a tick ends the stop, and
    /// the next stop fails them again. A stop while the run does not run
    /// fails none.
    #[tokio::test(start_paused = true)]
    async fn a_clock_that_stops_while_the_run_runs_fails_each_lane_once_a_stop() {
        let (ticks, ticks_rx) = watch::channel(1u64);
        let (phase, phase_rx) = watch::channel(Phase::Running);
        let (commands, mut sent) = mpsc::unbounded_channel();
        let roles = BTreeSet::from([name("left"), name("right")]);
        let stopped = || {
            ["left", "right"]
                .map(|role| (role.to_owned(), "clock_stopped".to_owned()))
                .to_vec()
        };
        let watching = tokio::spawn(watch_for_stops(ticks_rx, phase_rx, roles, commands));
        let second = Duration::from_secs(1);

        tokio::time::sleep(CLOCK_STOP - second).await;
        ticks.send_replace(2);
        tokio::time::sleep(CLOCK_STOP - second).await;
        assert!(
            faults(&mut sent).is_empty(),
            "a tick within the bound keeps the clock going"
        );
        tokio::time::sleep(2 * second).await;
        assert_eq!(faults(&mut sent), stopped());
        tokio::time::sleep(3 * CLOCK_STOP).await;
        assert!(
            faults(&mut sent).is_empty(),
            "one stop fails each lane once"
        );

        ticks.send_replace(3);
        tokio::time::sleep(CLOCK_STOP + second).await;
        assert_eq!(faults(&mut sent), stopped());

        phase.send_replace(Phase::Preparing);
        ticks.send_replace(4);
        tokio::time::sleep(3 * CLOCK_STOP).await;
        assert!(
            faults(&mut sent).is_empty(),
            "a run that does not run has no stopped clock"
        );

        drop(ticks);
        watching.await.unwrap();
    }

    /// The end of the simulation clock's ticks raises the fault and stops the
    /// node.
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
        let ended = fault.end().expect_err("the fault ends the node");
        assert!(ended.to_string().contains(CLOCK_CLOSED), "{ended}");
    }

    /// A clock that closed gives no first tick: a run refuses it. A clock
    /// that ticked gives its first tick at once, and the wall clock needs none.
    #[tokio::test(start_paused = true)]
    async fn a_closed_clock_gives_no_first_tick_and_a_ticked_one_does() {
        let (clock, feed) = NodeClock::simulation();
        feed.tick(5);
        assert_eq!(first_tick(&clock).await, Ok(()));
        drop(feed);
        assert_eq!(
            first_tick(&clock).await,
            Err("the simulation clock closed".to_owned())
        );
        assert_eq!(first_tick(&NodeClock::wall()).await, Ok(()));
    }

    /// A simulation clock that gives no tick within [`CLOCK_STOP`] of wall
    /// time gives no first tick.
    #[tokio::test(start_paused = true)]
    async fn a_clock_with_no_tick_within_the_bound_gives_no_first_tick() {
        let (clock, _feed) = NodeClock::simulation();
        assert_eq!(
            first_tick(&clock).await,
            Err(format!(
                "the simulation clock gave no tick in {} s",
                CLOCK_STOP.as_secs()
            ))
        );
    }
}
