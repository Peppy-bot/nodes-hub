//! Drive policy for the gripper motor: what one tick's status requires of
//! the follow loop. The arm follower's policy sized down to one motor, so
//! both ends of the robot respond to a limp motor the same way.

use openarm_can::MotorStatus;
use tracing::warn;

/// What one tick's motor status requires of the follow loop.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Continue,
    /// The motor reports Disabled without a fault: send a re-enable this tick.
    Reenable,
    HardFault(String),
}

/// Classify the tick against the not-driving streak, which this advances
/// (warning at the onset).
///
/// A decoded fault stops the node immediately, stale or not: it can only
/// stop the node, which is the safe direction. "Not driving" covers the
/// same three states as the arm: the motor reports Disabled, reports a
/// status nibble without a defined meaning (such a frame cannot confirm the
/// motor is driving), or is `silent` (no state frame within the stale
/// window, so the cached status is not a measurement and is disregarded).
/// Each earns a re-enable only where that is the answer, Disabled, and
/// escalates to a hard fault after `escalate_after_ticks` consecutive
/// not-driving ticks.
pub(crate) fn assess(
    status: MotorStatus,
    silent: bool,
    streak: &mut u32,
    escalate_after_ticks: u32,
) -> Verdict {
    if let Some(kind) = status.fault() {
        return Verdict::HardFault(format!("motor fault: {}", kind.name()));
    }

    let state = status.drive_state(silent);
    *streak = if state.is_driving() { 0 } else { *streak + 1 };
    if *streak == 1 {
        warn!("motor {}: not driving", state.words());
    }

    if *streak >= escalate_after_ticks {
        return Verdict::HardFault(format!(
            "motor not driving for {streak} consecutive ticks, currently {}",
            state.words()
        ));
    }

    match status {
        MotorStatus::Disabled if !silent => Verdict::Reenable,
        _ => Verdict::Continue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openarm_can::FaultKind;

    /// Drives `status` for `ticks` with `silent` fixed, returning the last
    /// verdict and leaving the streak inspectable.
    fn run_ticks(
        status: MotorStatus,
        silent: bool,
        streak: &mut u32,
        escalate_after: u32,
        ticks: u32,
    ) -> Verdict {
        (0..ticks)
            .map(|_| assess(status, silent, streak, escalate_after))
            .last()
            .expect("at least one tick")
    }

    #[test]
    fn an_enabled_fresh_motor_continues_and_clears_the_streak() {
        let mut streak = 5;
        assert_eq!(
            assess(MotorStatus::Enabled, false, &mut streak, 10),
            Verdict::Continue
        );
        assert_eq!(streak, 0);
    }

    #[test]
    fn a_fault_stops_the_node_immediately_and_names_the_protection() {
        let mut streak = 0;
        let verdict = assess(
            MotorStatus::Fault(FaultKind::Overload),
            false,
            &mut streak,
            10,
        );
        match verdict {
            Verdict::HardFault(reason) => assert!(reason.contains("overload"), "{reason}"),
            other => panic!("a fault must stop the node, got {other:?}"),
        }
    }

    #[test]
    fn a_stale_cached_fault_still_stops_the_node() {
        // The motor said fault and then went quiet; acting on the cached
        // fault can only stop the node, which is the safe direction.
        let mut streak = 0;
        assert!(matches!(
            assess(
                MotorStatus::Fault(FaultKind::CoilOvertemp),
                true,
                &mut streak,
                10
            ),
            Verdict::HardFault(_)
        ));
    }

    #[test]
    fn a_disabled_motor_is_reenabled_until_the_window_runs_out() {
        let mut streak = 0;
        assert_eq!(
            run_ticks(MotorStatus::Disabled, false, &mut streak, 10, 9),
            Verdict::Reenable
        );
        let escalated = assess(MotorStatus::Disabled, false, &mut streak, 10);
        match escalated {
            Verdict::HardFault(reason) => assert_eq!(
                reason, "motor not driving for 10 consecutive ticks, currently disabled",
                "the escalation names the streak and the current condition"
            ),
            other => panic!("persistent disabled must escalate, got {other:?}"),
        }
    }

    #[test]
    fn an_unrecognised_status_escalates_without_reenables() {
        let mut streak = 0;
        assert_eq!(
            run_ticks(MotorStatus::Unknown(0x7), false, &mut streak, 10, 9),
            Verdict::Continue
        );
        let escalated = assess(MotorStatus::Unknown(0x7), false, &mut streak, 10);
        match escalated {
            Verdict::HardFault(reason) => assert!(reason.contains("unrecognised"), "{reason}"),
            other => panic!("a stuck unrecognised status must escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_silent_motor_escalates_and_its_cached_status_is_disregarded() {
        // Silence means the cached Disabled is not a measurement: no
        // re-enable spam at an unhearing motor, and the escalation names
        // what is actually known.
        let mut streak = 0;
        assert_eq!(
            run_ticks(MotorStatus::Disabled, true, &mut streak, 10, 9),
            Verdict::Continue
        );
        let escalated = assess(MotorStatus::Disabled, true, &mut streak, 10);
        match escalated {
            Verdict::HardFault(reason) => assert!(reason.contains("silent"), "{reason}"),
            other => panic!("a silent motor must escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_flapping_motor_never_escalates_while_it_keeps_recovering() {
        let mut streak = 0;
        for _ in 0..50 {
            assert_eq!(
                assess(MotorStatus::Disabled, false, &mut streak, 10),
                Verdict::Reenable
            );
            assert_eq!(
                assess(MotorStatus::Enabled, false, &mut streak, 10),
                Verdict::Continue
            );
        }
    }

    #[test]
    fn flapping_between_not_driving_states_still_escalates() {
        // Disabled and silent are the same thing to a held load; alternating
        // between them must not read as recovery.
        let mut streak = 0;
        for _ in 0..200 {
            if let Verdict::HardFault(_) = assess(MotorStatus::Disabled, false, &mut streak, 10) {
                return;
            }
            if let Verdict::HardFault(_) = assess(MotorStatus::Enabled, true, &mut streak, 10) {
                return;
            }
        }
        panic!("a motor that is never driving must escalate however it flaps");
    }
}
