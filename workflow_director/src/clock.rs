//! The run clock of the director: wall time, or the simulation clock the
//! launcher binds it to, and the watch for a simulation clock that stops
//! while a run goes.

use std::collections::BTreeSet;
use std::time::Duration;

use std::sync::Arc;

use futures::{Stream, StreamExt};

use peppygen::NodeRunner;
use peppylib::messaging::TopicBuffer;
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
    /// The simulation clock: the time of its last tick, 0 before the first,
    /// and its domain as the fleet listing names it.
    Simulation {
        ticks: watch::Receiver<u64>,
        domain: String,
    },
}

impl ClockSource {
    /// The clock the launcher binds the director to. A tick that does not
    /// decode is left out; the end of the subscription raises `fault`, since
    /// the runs of the node read that clock.
    pub async fn of(runner: &Arc<NodeRunner>, fault: Fault) -> peppygen::Result<ClockSource> {
        let Some(domain) = runner.processor().clock().domain().map(ToString::to_string) else {
            return Ok(ClockSource::Wall);
        };
        let subscription = peppylib::clock::subscribe(runner, TopicBuffer::KeepLatest).await?;
        let (sender, ticks) = watch::channel(0);
        let runner = Arc::clone(runner);
        let tick_times = futures::stream::unfold(subscription, |mut subscription| async move {
            let next = subscription
                .on_next_tick()
                .await
                .map(|tick| tick.map(|tick| tick.time()))
                .map_err(|error| error.to_string());
            Some((next, subscription))
        });
        tokio::spawn(follow_ticks(Box::pin(tick_times), sender, move || {
            fault.raise(
                &runner,
                "the simulation clock of the director closed; the node exits for its restart policy to subscribe again",
            );
        }));
        Ok(ClockSource::Simulation { ticks, domain })
    }
}

/// Gives the time of each tick of `tick_times` on `sender`, leaving out a
/// tick that does not decode, up to the end of the clock (`Ok(None)`), where
/// it calls `closed`.
async fn follow_ticks(
    mut tick_times: impl Stream<Item = Result<Option<u64>, String>> + Unpin,
    sender: watch::Sender<u64>,
    closed: impl FnOnce(),
) {
    while let Some(next) = tick_times.next().await {
        match next {
            Ok(Some(time)) => {
                sender.send_replace(time);
            }
            Ok(None) => break,
            Err(error) => tracing::warn!("a tick of the simulation clock did not decode: {error}"),
        }
    }
    closed();
}

/// Waits up to [`CLOCK_STOP`] of wall time for the first tick of `ticks`.
pub async fn first_tick(ticks: &watch::Receiver<u64>) -> Result<(), String> {
    let mut ticks = ticks.clone();
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

/// Resolves at the first tick of `ticks` at or after `at`; a clock that
/// closed ends no wait past its last tick.
pub async fn until_tick(ticks: &watch::Receiver<u64>, at: u64) {
    let mut ticks = ticks.clone();
    if ticks.wait_for(|now| *now >= at).await.is_err() {
        std::future::pending::<()>().await;
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

    /// A tick that does not decode is left out, the ticks after it go on,
    /// and the end of the clock calls its close once.
    #[tokio::test]
    async fn a_tick_that_does_not_decode_is_left_out_and_the_end_closes_the_clock() {
        let ticks = futures::stream::iter(vec![
            Ok(Some(1)),
            Err("a tick that does not decode".to_owned()),
            Ok(Some(2)),
            Ok(None),
            Ok(Some(3)),
        ]);
        let (sender, times) = watch::channel(0);
        let closes = std::sync::atomic::AtomicU32::new(0);
        follow_ticks(ticks, sender, || {
            closes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })
        .await;
        assert_eq!(
            *times.borrow(),
            2,
            "the tick after the bad one came, and none after the end"
        );
        assert_eq!(closes.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    /// A clock that closed gives no first tick and no instant to wait for: a
    /// run refuses it, and a wait on it does not end.
    #[tokio::test(start_paused = true)]
    async fn a_closed_clock_gives_no_first_tick_and_ends_no_wait() {
        let (ticks, ticks_rx) = watch::channel(5u64);
        drop(ticks);
        assert_eq!(
            first_tick(&ticks_rx).await,
            Err("the simulation clock closed".to_owned())
        );
        let waited = tokio::time::timeout(CLOCK_STOP, until_tick(&ticks_rx, 9)).await;
        assert!(waited.is_err(), "a wait on a closed clock stands");
        let reached = tokio::time::timeout(CLOCK_STOP, until_tick(&ticks_rx, 5)).await;
        assert!(
            reached.is_ok(),
            "an instant the clock reached ends the wait"
        );
    }
}
