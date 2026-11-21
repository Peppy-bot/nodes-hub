//! What the goal handlers admit a goal on: whether the wheels report, when
//! the leader's newest setpoint arrived and whether the base comes to rest
//! from an earlier goal, all written by the control loop, and the base's move
//! slot. A goal handler claims the slot when it admits a goal, at the time the
//! admission read; the goal releases it when it ends.

use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use control_core::move_slot::{MoveSlot, SlotRelease};
use control_core::sync::lock_ignoring_poison;

use crate::failure::{Failure, FailureKind};
use crate::follower;

const WHEELS_SILENT: &str = "the base's wheels are silent: retry once base_states reports";
const LEADER_STREAMS: &str = "the leader of drive_link streams to the base: stop the leader, or wait until 0.5 s pass after its last setpoint, then retry";
const GOAL_RUNS: &str =
    "the base runs another goal, or still comes to rest from one: wait for it to end, then retry";

#[derive(Debug, Default)]
struct BaseStatusState {
    wheels_report: bool,
    /// When the leader's newest setpoint arrived on the node's clock (ns),
    /// followed or dropped.
    setpoint_ns: Option<i64>,
    /// The odometry_start_timestamp of the odometry the base reports.
    odometry_start: Option<SystemTime>,
    /// Whether the base comes to rest from an earlier goal: from the end of a
    /// goal that ended while the limiter's command was not 0, completed,
    /// cancelled or failed, until the command reaches 0.
    resting: bool,
}

/// The base's state as the goal handlers read it.
#[derive(Clone, Debug, Default)]
pub struct BaseStatus(Arc<Mutex<BaseStatusState>>);

impl BaseStatus {
    /// Admit a goal at `now_ns` (`None` while the clock reads no time) and
    /// claim `slot` for it from then, or the refusal, in base_motion's order:
    /// the wheels are silent (`resource_unavailable`), the leader streams, as
    /// drive_link defines it (`leader_streaming`), the base comes to rest from
    /// an earlier goal, or another goal holds the slot, running or coming to
    /// rest (`precondition_failed`).
    pub fn admit(&self, now_ns: Option<i64>, slot: &MoveSlot) -> Result<Admitted, Failure> {
        let state = lock_ignoring_poison(&self.0);
        let Some(now_ns) = now_ns.filter(|_| state.wheels_report) else {
            return Err(Failure::new(
                FailureKind::ResourceUnavailable,
                WHEELS_SILENT,
            ));
        };
        if follower::LEADER_STREAM.streams(state.setpoint_ns, now_ns) {
            return Err(Failure::new(FailureKind::LeaderStreaming, LEADER_STREAMS));
        }
        if state.resting || !slot.claim() {
            return Err(Failure::new(FailureKind::PreconditionFailed, GOAL_RUNS));
        }
        Ok(Admitted {
            _release: slot.release_on_drop(),
            admitted_ns: now_ns,
            odometry_start: state.odometry_start,
        })
    }

    /// The odometry started at `start`, its odometry_start_timestamp.
    pub fn set_odometry_start(&self, start: SystemTime) {
        lock_ignoring_poison(&self.0).odometry_start = Some(start);
    }

    pub fn set_wheels_report(&self, report: bool) {
        lock_ignoring_poison(&self.0).wheels_report = report;
    }

    /// A setpoint of the leader arrived at `received_ns`.
    pub fn note_setpoint(&self, received_ns: i64) {
        lock_ignoring_poison(&self.0).setpoint_ns = Some(received_ns);
    }

    /// The base's clock went back to `now_ns`: the leader's newest setpoint,
    /// when it came after it, takes `now_ns` as its time, as the follower's
    /// does ([`follower::Follower::clock_went_back`]).
    pub fn clock_went_back(&self, now_ns: i64) {
        let mut state = lock_ignoring_poison(&self.0);
        state.setpoint_ns = state.setpoint_ns.map(|received_ns| received_ns.min(now_ns));
    }

    /// Whether the base comes to rest from an earlier goal.
    pub fn set_resting(&self, resting: bool) {
        lock_ignoring_poison(&self.0).resting = resting;
    }
}

/// An admitted goal: the release of the move slot it holds, which frees the
/// slot on every end of the goal and when the goal never reaches the control
/// loop, the time it was admitted at, and the odometry it was placed by.
pub struct Admitted {
    _release: SlotRelease,
    admitted_ns: i64,
    odometry_start: Option<SystemTime>,
}

impl Admitted {
    /// When the admission admitted the goal, on the node's clock (ns).
    pub fn admitted_ns(&self) -> i64 {
        self.admitted_ns
    }

    /// The odometry_start_timestamp of the odometry the base reported when
    /// the goal was admitted.
    pub fn odometry_start(&self) -> Option<SystemTime> {
        self.odometry_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::follower::LEADER_STREAMING_WINDOW_NS;

    fn kind(admitted: Result<Admitted, Failure>) -> Option<FailureKind> {
        admitted.err().map(|failure| failure.kind)
    }

    #[test]
    fn a_goal_is_refused_by_the_wheels_first_then_the_leader_then_the_goal() {
        let (status, slot) = (BaseStatus::default(), MoveSlot::default());
        status.note_setpoint(1_000);
        assert_eq!(
            kind(status.admit(Some(1_000), &slot)),
            Some(FailureKind::ResourceUnavailable)
        );
        status.set_wheels_report(true);
        assert_eq!(
            kind(status.admit(None, &slot)),
            Some(FailureKind::ResourceUnavailable)
        );
        assert_eq!(
            kind(status.admit(Some(1_000 + LEADER_STREAMING_WINDOW_NS), &slot)),
            Some(FailureKind::LeaderStreaming),
            "the leader streams until more than 0.5 s pass"
        );
        let claim = status
            .admit(Some(1_000 + LEADER_STREAMING_WINDOW_NS + 1), &slot)
            .expect("the base is free");
        assert_eq!(claim.admitted_ns(), 1_000 + LEADER_STREAMING_WINDOW_NS + 1);
        assert!(slot.is_claimed(), "the admitted goal holds the slot");
        assert_eq!(
            kind(status.admit(Some(1_000 + LEADER_STREAMING_WINDOW_NS + 1), &slot)),
            Some(FailureKind::PreconditionFailed)
        );
        drop(claim);
        assert!(!slot.is_claimed(), "the dropped release frees the slot");
    }

    #[test]
    fn a_setpoint_after_a_clock_that_went_back_streams_from_the_new_time() {
        let (status, slot) = (BaseStatus::default(), MoveSlot::default());
        status.set_wheels_report(true);
        status.note_setpoint(2_000);
        assert_eq!(
            kind(status.admit(Some(1_000), &slot)),
            Some(FailureKind::LeaderStreaming),
            "a setpoint later than the time streams"
        );
        status.clock_went_back(1_000);
        assert_eq!(
            kind(status.admit(Some(1_000 + LEADER_STREAMING_WINDOW_NS), &slot)),
            Some(FailureKind::LeaderStreaming),
            "the stream lasts 0.5 s from the new time"
        );
        assert!(
            status
                .admit(Some(1_000 + LEADER_STREAMING_WINDOW_NS + 1), &slot)
                .is_ok(),
            "then the leader streams no more"
        );
    }

    #[test]
    fn a_base_coming_to_rest_from_an_earlier_goal_refuses_a_goal() {
        let (status, slot) = (BaseStatus::default(), MoveSlot::default());
        status.set_wheels_report(true);
        status.set_resting(true);
        assert_eq!(
            kind(status.admit(Some(1_000), &slot)),
            Some(FailureKind::PreconditionFailed)
        );
        status.set_resting(false);
        assert!(status.admit(Some(1_000), &slot).is_ok(), "the base rests");
    }
}
