//! The state that the lanes of one run share: the state of each lane, the
//! pending events of each signal, the observations read at the current
//! instant, the counts of the loops and the incidents, with a version that
//! each change advances so a waiter sees every change.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use serde_json::Value as Json;
use tokio::sync::watch;
use workflow_documents::Name;
use workflow_documents::result::{IncidentRecord, LaneState, LoopCount};

use super::failure::Failure;

/// The state of the lanes of a run and what they share.
pub(crate) struct Shared {
    state: Mutex<State>,
    version: watch::Sender<u64>,
}

#[derive(Default)]
pub(crate) struct State {
    pub lanes: BTreeMap<Name, LaneStatus>,
    pub pending: BTreeMap<Name, u32>,
    pub emitters: BTreeMap<Name, BTreeSet<Name>>,
    /// The instant of the reads of `reads`.
    pub read_at: Option<u64>,
    /// Each observation read at `read_at`: its answer once it came.
    pub reads: BTreeMap<Name, watch::Receiver<Option<Result<Json, Failure>>>>,
    /// How many reads of each observation the run made.
    pub observed: BTreeMap<Name, u64>,
    pub last_values: BTreeMap<Name, Json>,
    pub loops: BTreeMap<String, LoopCount>,
    pub incidents: Vec<IncidentRecord>,
    /// The incident alerts that stand on the robot of each role, by the
    /// producer that gave them.
    pub alerts: BTreeMap<Name, BTreeMap<String, Vec<String>>>,
}

/// Where a lane stands.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LaneStatus {
    pub state: LaneState,
    /// `<role>.<id>` of its current or last step.
    pub step: Option<String>,
    pub failure_kind: Option<String>,
}

impl LaneStatus {
    pub fn is_ended(&self) -> bool {
        matches!(
            self.state,
            LaneState::Completed | LaneState::Failed | LaneState::Aborted | LaneState::Interrupted
        )
    }
}

impl State {
    /// Whether the lane of `role` stands suspended, its lease open for a
    /// take-over.
    pub fn is_suspended(&self, role: &Name) -> bool {
        self.lanes
            .get(role)
            .is_some_and(|lane| lane.state == LaneState::Suspended)
    }
}

impl Shared {
    pub fn new(emitters: BTreeMap<Name, BTreeSet<Name>>) -> Shared {
        Shared {
            state: Mutex::new(State {
                emitters,
                ..State::default()
            }),
            version: watch::Sender::new(0),
        }
    }

    /// Reads the state.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        f(&self.state.lock().expect("no lane panics holding the state"))
    }

    /// Changes the state and wakes each waiter.
    pub fn change<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        let out = f(&mut self.state.lock().expect("no lane panics holding the state"));
        self.version.send_modify(|version| *version += 1);
        out
    }

    /// Takes one pending event of the first of `signals` that has one; the
    /// waiters wake only when it takes one.
    pub fn take_signal(&self, signals: &[Name]) -> Option<Name> {
        let taken = {
            let mut state = self.state.lock().expect("no lane panics holding the state");
            let signal = signals
                .iter()
                .find(|signal| state.pending.get(*signal).is_some_and(|count| *count > 0))?
                .clone();
            *state.pending.get_mut(&signal).expect("a pending signal") -= 1;
            signal
        };
        self.version.send_modify(|version| *version += 1);
        Some(taken)
    }

    /// A receiver that wakes at each change after it was made.
    pub fn changes(&self) -> watch::Receiver<u64> {
        self.version.subscribe()
    }

    pub fn set_lane(
        &self,
        role: &Name,
        state: LaneState,
        step: Option<String>,
        failure_kind: Option<String>,
    ) {
        self.change(|s| {
            s.lanes.insert(
                role.clone(),
                LaneStatus {
                    state,
                    step,
                    failure_kind,
                },
            )
        });
    }
}

/// The time of one lane: run time without the spans the lane stood
/// suspended, so the bound of each of its steps counts no suspended time.
pub(crate) struct LaneClock {
    state: Mutex<ClockState>,
    version: watch::Sender<u64>,
}

#[derive(Clone, Copy, Default)]
struct ClockState {
    offset: u64,
    paused_since: Option<u64>,
}

impl LaneClock {
    pub fn new() -> LaneClock {
        LaneClock {
            state: Mutex::new(ClockState::default()),
            version: watch::Sender::new(0),
        }
    }

    fn get(&self) -> ClockState {
        *self.state.lock().expect("no lane panics holding its clock")
    }

    /// Lane time at run time `run_now`.
    pub fn lane_time(&self, run_now: u64) -> u64 {
        let state = self.get();
        state
            .paused_since
            .unwrap_or(run_now)
            .saturating_sub(state.offset)
    }

    /// The run time at which lane time reaches `lane_at`, or `None` while the
    /// clock stands.
    pub fn run_time_of(&self, lane_at: u64) -> Option<u64> {
        let state = self.get();
        match state.paused_since {
            Some(_) => None,
            None => Some(lane_at + state.offset),
        }
    }

    pub fn pause(&self, run_now: u64) {
        self.state
            .lock()
            .expect("no lane panics holding its clock")
            .paused_since = Some(run_now);
        self.version.send_modify(|version| *version += 1);
    }

    pub fn resume(&self, run_now: u64) {
        {
            let mut state = self.state.lock().expect("no lane panics holding its clock");
            if let Some(since) = state.paused_since.take() {
                state.offset += run_now.saturating_sub(since);
            }
        }
        self.version.send_modify(|version| *version += 1);
    }

    pub fn changes(&self) -> watch::Receiver<u64> {
        self.version.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lane_counts_no_time_while_it_stands_suspended() {
        let clock = LaneClock::new();
        assert_eq!(clock.lane_time(100), 100);
        clock.pause(100);
        assert_eq!(clock.lane_time(250), 100);
        assert_eq!(clock.run_time_of(150), None);
        clock.resume(300);
        assert_eq!(clock.lane_time(350), 150);
        assert_eq!(clock.run_time_of(160), Some(360));
    }

    /// Over any schedule of suspensions, lane time is run time less the time
    /// the lane stood suspended, and a lane time ahead comes at the run time
    /// that adds that time back.
    #[hegel::test]
    fn lane_time_leaves_out_every_suspension(tc: hegel::TestCase) {
        use hegel::generators as gs;
        let clock = LaneClock::new();
        let mut now = 0u64;
        let mut suspended = 0u64;
        for _ in 0..tc.draw(gs::integers::<usize>().max_value(6)) {
            now += tc.draw(gs::integers::<u64>().max_value(1_000));
            clock.pause(now);
            let paused = tc.draw(gs::integers::<u64>().max_value(1_000));
            now += paused;
            suspended += paused;
            clock.resume(now);
        }
        now += tc.draw(gs::integers::<u64>().max_value(1_000));
        let lane_now = clock.lane_time(now);
        assert_eq!(lane_now, now - suspended);
        let ahead = tc.draw(gs::integers::<u64>().max_value(1_000));
        assert_eq!(clock.run_time_of(lane_now + ahead), Some(now + ahead));
    }
}
