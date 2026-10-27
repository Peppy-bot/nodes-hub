//! The operator's end-effector speed cap, as a limiter.
//!
//! A streamed jog is capped by how fast it moves the hand, not just by the
//! per-joint rates: the same joint step swings the hand slowly near the body
//! and fast at full reach. The cap scales that side's whole arm step uniformly
//! so the hand's linear speed stays at or under the commanded cap, which
//! preserves the step's direction.
//!
//! The basis is per side and present only while that side follows the
//! operator's stream: a planned move was rolled out against the cap at
//! admission and tracks its own schedule, so capping it again per tick would
//! stretch it past the duration its budget was sized from. Openings are never
//! capped here; a gripper does not move the hand.

use srs_model::Jacobian;
use srs_model::nalgebra::SVector;

use super::super::{ARM_DOF, ArmPair, DUAL_DOF, dof_side, step_change};
use super::allowance::Allowance;
use super::{Limiter, Step};

pub(in crate::governor) struct EeSpeed<'tick> {
    /// The commanded cap on the hand's linear speed (m/s), validated positive.
    pub cap_m_s: f64,
    /// Each side's end-effector Jacobian at the measured pose, present only
    /// while that side's target is the operator's stream.
    pub hands: &'tick ArmPair<Option<Jacobian>>,
    pub dt: f64,
}

impl EeSpeed<'_> {
    /// The uniform fraction that keeps one side's hand at or under the cap:
    /// 1.0 for a planned move (no basis) or a step already under it.
    ///
    /// The stream boundary rejects non-finite measured states, so the Jacobian
    /// and with it the hand speed are finite by construction; if that ever
    /// stops holding, a basis that cannot be evaluated is not a licence to run
    /// uncapped, so it freezes the side like every other fail-safe here.
    fn side_fraction(&self, jac: &Option<Jacobian>, arm_step: &[f64; ARM_DOF]) -> f64 {
        let Some(jac) = jac else { return 1.0 };
        let twist = jac * SVector::<f64, ARM_DOF>::from_column_slice(arm_step);
        let hand_speed = twist.fixed_rows::<3>(0).norm() / self.dt;
        if !hand_speed.is_finite() {
            0.0
        } else if hand_speed > self.cap_m_s {
            self.cap_m_s / hand_speed
        } else {
            1.0
        }
    }
}

impl Limiter for EeSpeed<'_> {
    fn name(&self) -> &'static str {
        "ee-speed"
    }

    fn allow(&self, step: &Step) -> Allowance {
        let arm_steps = step_change(&step.prev, &step.target).arms;
        let fractions =
            ArmPair::from_fn(|side| self.side_fraction(&self.hands[side], &arm_steps[side]));
        Allowance::new(std::array::from_fn(|i| {
            if i >= DUAL_DOF {
                1.0
            } else {
                fractions[dof_side(i)]
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{GOV_DOF, Side, Step, split};
    use super::*;

    const DT: f64 = 0.01;

    fn step_with_left_arm(arm: [f64; ARM_DOF]) -> Step {
        let mut target = [0.0; GOV_DOF];
        target[..ARM_DOF].copy_from_slice(&arm);
        Step {
            prev: [0.0; GOV_DOF],
            target,
        }
    }

    /// Joint 0 moves the hand 1 m per rad along x; the rest do not move it.
    fn x_only_jacobian() -> Jacobian {
        let mut jac = Jacobian::zeros();
        jac[(0, 0)] = 1.0;
        jac
    }

    #[test]
    fn a_hand_moving_step_is_scaled_to_the_cap() {
        let hands = ArmPair::new(Some(x_only_jacobian()), None);
        let ee = EeSpeed {
            cap_m_s: 0.25,
            hands: &hands,
            dt: DT,
        };
        let mut arm = [0.0; ARM_DOF];
        arm[0] = 0.1; // 0.1 rad over 0.01 s is 10 m/s of hand speed, over the cap.
        let step = step_with_left_arm(arm);
        let governed = ee.allow(&step).apply(&step);
        assert!(
            (governed[0] - 0.25 * DT).abs() < 1e-12,
            "joint 0 not scaled to the cap: {}",
            governed[0]
        );
        assert_eq!(governed[1..], [0.0; GOV_DOF - 1]);
    }

    #[test]
    fn a_step_that_does_not_move_the_hand_passes_bit_exact() {
        let hands = ArmPair::new(Some(Jacobian::zeros()), None);
        let ee = EeSpeed {
            cap_m_s: 0.25,
            hands: &hands,
            dt: DT,
        };
        let mut arm = [0.0; ARM_DOF];
        arm[3] = 0.2;
        let step = step_with_left_arm(arm);
        assert_eq!(ee.allow(&step).apply(&step), step.target);
    }

    #[test]
    fn a_planned_move_side_is_never_capped() {
        // No basis (a planned move): the same over-cap step passes untouched.
        let hands = ArmPair::new(None, None);
        let ee = EeSpeed {
            cap_m_s: 0.25,
            hands: &hands,
            dt: DT,
        };
        let mut arm = [0.0; ARM_DOF];
        arm[0] = 0.1;
        let step = step_with_left_arm(arm);
        assert_eq!(ee.allow(&step).apply(&step), step.target);
    }

    #[test]
    fn an_unevaluable_basis_freezes_the_side_not_uncaps_it() {
        // Unreachable through the stream boundary (non-finite measured states
        // are rejected at parse), but the fail-safe direction must match the
        // rest of the pipeline: restrictive, never permissive.
        let mut jac = Jacobian::zeros();
        jac[(0, 0)] = f64::NAN;
        let hands = ArmPair::new(Some(jac), None);
        let ee = EeSpeed {
            cap_m_s: 0.25,
            hands: &hands,
            dt: DT,
        };
        let mut arm = [0.0; ARM_DOF];
        arm[0] = 0.1;
        let step = step_with_left_arm(arm);
        assert_eq!(
            ee.allow(&step).apply(&step),
            step.prev,
            "a NaN basis must freeze the side at prev"
        );
    }

    #[test]
    fn the_cap_is_per_side() {
        // Each side commands its own hand-moving step, 0.1 rad on the
        // streamed (basis-carrying) side and 0.2 rad on the planned one: the
        // streamed side alone is scaled to the cap, whichever side streams.
        for streamed in Side::BOTH {
            let hands = ArmPair::from_fn(|side| (side == streamed).then(x_only_jacobian));
            let ee = EeSpeed {
                cap_m_s: 0.25,
                hands: &hands,
                dt: DT,
            };
            let commanded = ArmPair::from_fn(|side| if side == streamed { 0.1 } else { 0.2 });
            let mut target = [0.0; GOV_DOF];
            target[0] = commanded.left;
            target[ARM_DOF] = commanded.right;
            let step = Step {
                prev: [0.0; GOV_DOF],
                target,
            };
            let governed = split(&ee.allow(&step).apply(&step));
            for side in Side::BOTH {
                let first_joint = governed.arms[side][0];
                let label = side.label();
                if side == streamed {
                    assert!(
                        (first_joint - 0.25 * DT).abs() < 1e-12,
                        "streamed {label} side not scaled to the cap: {first_joint}"
                    );
                } else {
                    assert_eq!(first_joint, 0.2, "planned {label} side is not capped");
                }
            }
        }
    }
}
