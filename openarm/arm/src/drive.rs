//! Drive policy for the arm's motors: what one tick's statuses require of
//! the control loop. The gripper's policy, across the arm's joints, so both
//! ends of the robot respond to a limp motor the same way.

use openarm_can::MotorStatus;
use openarm_description::ARM_DOF;
use tracing::warn;

/// What one tick's motor statuses require of the loop.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Continue,
    /// The named joint (first of any) reports Disabled without a fault: send
    /// a re-enable this tick.
    Reenable(usize),
    HardFault(String),
}

/// Classify the tick's statuses and silent joints against the per-joint
/// not-driving streaks, which this advances (warning at each joint's onset).
///
/// A fault stops the node immediately, from a fresh frame or the cache: a
/// faulted motor latches until power-cycled, so a cached fault is still the
/// joint's condition. Any other joint that is not driving gets a re-enable
/// each tick it freshly reports Disabled, and escalates to a hard fault
/// after `escalate_after_ticks` consecutive not-driving ticks.
///
/// "Not driving" covers three states, because they are the same thing to a
/// held load: the motor reports Disabled, reports a status nibble this
/// driver does not recognise (such a frame cannot confirm the motor is
/// driving), or has nothing current to report (`silent`, or never heard from).
///
/// The streaks are per joint, so escalation names the joint that ran out of
/// patience, and a flapping joint's good ticks reset only its own streak.
pub(crate) fn assess(
    statuses: &[MotorStatus; ARM_DOF],
    silent: &[bool; ARM_DOF],
    not_driving: &mut [u32; ARM_DOF],
    escalate_after_ticks: u32,
) -> Verdict {
    let fault = statuses
        .iter()
        .enumerate()
        .find_map(|(i, s)| s.fault().map(|kind| (i, kind)));
    if let Some((joint, kind)) = fault {
        return Verdict::HardFault(format!("motor j{} fault: {}", joint + 1, kind.name()));
    }

    for (joint, streak) in not_driving.iter_mut().enumerate() {
        let state = statuses[joint].drive_state(silent[joint]);
        *streak = if state.is_driving() { 0 } else { *streak + 1 };
        if *streak == 1 {
            warn!("motor j{} {}: not driving", joint + 1, state.words());
        }
    }

    let escalated = (0..ARM_DOF).find(|&j| not_driving[j] >= escalate_after_ticks);
    if let Some(joint) = escalated {
        let state = statuses[joint].drive_state(silent[joint]);
        return Verdict::HardFault(format!(
            "motor j{} not driving for {} consecutive ticks, currently {}",
            joint + 1,
            not_driving[joint],
            state.words()
        ));
    }

    // Re-enabling is the answer to a fresh Disabled report; a silent motor is
    // not listening and an unrecognised status never said the motor was
    // disabled, so those only run out the escalation window.
    match (0..ARM_DOF).find(|&j| !silent[j] && statuses[j] == MotorStatus::Disabled) {
        Some(joint) => Verdict::Reenable(joint),
        None => Verdict::Continue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openarm_can::FaultKind;

    const FRESH: [bool; ARM_DOF] = [false; ARM_DOF];

    /// assess with fresh streaks and every joint's state current.
    fn verdict(statuses: &[MotorStatus; ARM_DOF], escalate_after: u32) -> Verdict {
        let mut streaks = [0u32; ARM_DOF];
        assess(statuses, &FRESH, &mut streaks, escalate_after)
    }

    #[test]
    fn assess_verdicts_by_status_table() {
        let all = |s: MotorStatus| -> [MotorStatus; ARM_DOF] { [s; ARM_DOF] };
        let one = |i: usize, s: MotorStatus| -> [MotorStatus; ARM_DOF] {
            let mut v = all(MotorStatus::Enabled);
            v[i] = s;
            v
        };
        assert_eq!(verdict(&all(MotorStatus::Enabled), 100), Verdict::Continue);
        // One frame with an undefined nibble is noise; only persistence
        // escalates, and it is no grounds for a re-enable.
        assert_eq!(
            verdict(&one(2, MotorStatus::Unknown(0x3)), 100),
            Verdict::Continue
        );
        assert_eq!(
            verdict(&one(4, MotorStatus::Disabled), 100),
            Verdict::Reenable(4)
        );
        // A fault outranks a concurrent disabled joint.
        let mut mixed = one(1, MotorStatus::Disabled);
        mixed[5] = MotorStatus::Fault(FaultKind::Overload);
        assert!(matches!(verdict(&mixed, 100), Verdict::HardFault(_)));
    }

    #[test]
    fn a_cached_fault_on_a_silent_joint_still_stops_the_node() {
        // A faulted motor latches until power-cycled, so the fault is the
        // joint's condition even when its frames have since stopped.
        let mut statuses = [MotorStatus::Enabled; ARM_DOF];
        statuses[3] = MotorStatus::Fault(FaultKind::Overload);
        let mut silent = FRESH;
        silent[3] = true;
        let mut streaks = [0u32; ARM_DOF];
        assert!(matches!(
            assess(&statuses, &silent, &mut streaks, 100),
            Verdict::HardFault(_)
        ));
    }

    #[test]
    fn a_silent_joint_escalates_whatever_its_cache_says() {
        // Past the stale window the cached Enabled is not a measurement; the
        // joint is silent and runs out the escalation window like any other.
        let mut silent = FRESH;
        silent[4] = true;
        let statuses = [MotorStatus::Enabled; ARM_DOF];
        let mut streaks = [0u32; ARM_DOF];
        for tick in 1..100 {
            assert_eq!(
                assess(&statuses, &silent, &mut streaks, 100),
                Verdict::Continue,
                "tick {tick}"
            );
        }
        match assess(&statuses, &silent, &mut streaks, 100) {
            Verdict::HardFault(reason) => {
                assert!(reason.contains("j5"), "{reason}");
                assert!(reason.contains("silent"), "{reason}");
            }
            other => panic!("a silent joint must escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_motor_that_goes_silent_under_load_escalates_to_a_hard_fault() {
        // A silent joint hangs limp while six others drive around it; its
        // streak runs the escalation window out and the node stops.
        let mut streaks = [0u32; ARM_DOF];
        let mut statuses = [MotorStatus::Enabled; ARM_DOF];
        statuses[4] = MotorStatus::Unreported;
        for tick in 1..100 {
            assert_eq!(
                assess(&statuses, &FRESH, &mut streaks, 100),
                Verdict::Continue,
                "tick {tick}"
            );
        }
        let escalated = assess(&statuses, &FRESH, &mut streaks, 100);
        match escalated {
            Verdict::HardFault(reason) => {
                assert!(reason.contains("j5"), "{reason}");
                assert!(reason.contains("silent"), "{reason}");
            }
            other => panic!("a silent motor must escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_motor_stuck_in_an_unrecognised_status_escalates_without_reenables() {
        // Frames are arriving but their status nibble is undefined, so the
        // motor cannot be confirmed driving. It never said Disabled, so no
        // re-enable; it runs out the window like a silent one.
        let mut streaks = [0u32; ARM_DOF];
        let mut statuses = [MotorStatus::Enabled; ARM_DOF];
        statuses[3] = MotorStatus::Unknown(0x7);
        for tick in 1..10 {
            assert_eq!(
                assess(&statuses, &FRESH, &mut streaks, 10),
                Verdict::Continue,
                "tick {tick}"
            );
        }
        let escalated = assess(&statuses, &FRESH, &mut streaks, 10);
        match escalated {
            Verdict::HardFault(reason) => {
                assert!(reason.contains("j4"), "{reason}");
                assert!(reason.contains("unrecognised"), "{reason}");
            }
            other => panic!("a stuck unrecognised status must escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_flapping_joint_still_escalates() {
        // The streak is per joint and reset only by its own driving ticks, so
        // a joint limp half the time still runs out its window however the
        // not-driving states alternate.
        let mut streaks = [0u32; ARM_DOF];
        let mut disabled = [MotorStatus::Enabled; ARM_DOF];
        disabled[3] = MotorStatus::Disabled;
        let mut silent = [MotorStatus::Enabled; ARM_DOF];
        silent[3] = MotorStatus::Unreported;
        for _ in 0..200 {
            let v = assess(&disabled, &FRESH, &mut streaks, 10);
            if let Verdict::HardFault(reason) = v {
                assert!(reason.contains("j4"), "{reason}");
                return;
            }
            let v = assess(&silent, &FRESH, &mut streaks, 10);
            if let Verdict::HardFault(reason) = v {
                assert!(reason.contains("j4"), "{reason}");
                return;
            }
        }
        panic!("a joint that is never driving must escalate however it flaps");
    }

    #[test]
    fn escalation_names_the_joint_that_ran_out_not_the_first_one_seen() {
        // j2 spends most of the window disabled and recovers; j5 disables
        // later and escalates only when its own streak runs out.
        let mut streaks = [0u32; ARM_DOF];
        let mut early = [MotorStatus::Enabled; ARM_DOF];
        early[1] = MotorStatus::Disabled;
        for _ in 0..50 {
            assess(&early, &FRESH, &mut streaks, 100);
        }
        let mut late = [MotorStatus::Enabled; ARM_DOF];
        late[4] = MotorStatus::Disabled;
        for _ in 0..99 {
            assert_eq!(
                assess(&late, &FRESH, &mut streaks, 100),
                Verdict::Reenable(4)
            );
        }
        match assess(&late, &FRESH, &mut streaks, 100) {
            Verdict::HardFault(reason) => assert!(reason.contains("j5"), "{reason}"),
            other => panic!("expected j5 to escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_recovered_joint_starts_its_patience_over() {
        let mut streaks = [0u32; ARM_DOF];
        let mut disabled = [MotorStatus::Enabled; ARM_DOF];
        disabled[0] = MotorStatus::Disabled;
        for _ in 0..99 {
            assess(&disabled, &FRESH, &mut streaks, 100);
        }
        assert_eq!(
            assess(&[MotorStatus::Enabled; ARM_DOF], &FRESH, &mut streaks, 100),
            Verdict::Continue
        );
        assert_eq!(streaks[0], 0, "recovery clears that joint's streak");
    }

    #[test]
    fn a_stale_disabled_cache_gets_no_reenable() {
        // A stale joint's cached Disabled is not a fresh report; it is
        // silent, and re-enabling answers only a motor that just said so.
        let mut statuses = [MotorStatus::Enabled; ARM_DOF];
        statuses[2] = MotorStatus::Disabled;
        let mut silent = FRESH;
        silent[2] = true;
        let mut streaks = [0u32; ARM_DOF];
        assert_eq!(
            assess(&statuses, &silent, &mut streaks, 100),
            Verdict::Continue
        );
    }
}
