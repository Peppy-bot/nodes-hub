//! The two collision stages that shape a step: the closing-velocity barrier and
//! the exact floor scan.
//!
//! These are the only stages that are not independent per-DOF limits, and they
//! run in this order for a reason.
//!
//! [`Governor::project_closing`] is a *directional* transform: it removes just
//! enough of the gap-closing component that the clearance loses no more than
//! `allowed_closing(d) * dt`, and leaves tangential and separating motion at
//! full speed. That cannot be expressed as a per-DOF fraction, so it cannot be
//! a limiter. It runs after the limiters so its guarantee holds on the step that is
//! actually published rather than on an intermediate one.
//!
//! [`Governor::clip_to_floor_along`] is the exact backstop, and it runs last. Surface
//! distance is not monotone along a joint-space segment, so no upstream stage
//! can prove the realized path stays clear; this one walks the segment and
//! retracts to the furthest point that does. It is what makes the limiters'
//! order-independence safe rather than merely convenient.

use robot_description::{ArmId, PerArm};

use super::{
    APPROACH_VELOCITY_AT_SAFE_M_S, Clip, GOV_DOF, Governor, MAX_PROBE_ARC_RAD,
    MAX_PROBE_GRIPPER_FRAC, MIN_GRADIENT_NORM_SQ, PathArms, RECOVERY_LOSS_M_PER_S,
    SEGMENT_SAMPLES_MIN, dof_arm, dot, step_change,
};
use crate::robot::{ARMS, DUAL_DOF};

/// The clearance a step of `dt` from a clearance of `d_now` must not drop
/// below, under a guard that stops at `d_stop`.
///
/// `d_stop` on an approach, or the current clearance if the arms are inside
/// it, so closing further is refused. Once the bodies actually overlap the
/// rule relaxes to a bounded rate of loss: escaping an interpenetration
/// routinely sweeps deeper before it separates, and a floor that forbids any
/// loss refuses that whole segment every tick. See [`RECOVERY_LOSS_M_PER_S`].
pub(super) fn step_floor(d_stop: f64, d_now: f64, dt: f64) -> f64 {
    if d_now >= 0.0 {
        return d_now.min(d_stop);
    }
    d_now - RECOVERY_LOSS_M_PER_S * dt
}

impl Governor {
    /// The closing-velocity barrier: scale back only the gap-closing component of the
    /// step (minimum-norm, along the distance gradient) so the clearance loses no more
    /// than `allowed_closing(d_now) * dt`, then clamp each DOF's step into
    /// `[0, commanded]` so the barrier can only slow motion, never add motion a DOF
    /// was not commanded nor reverse one it was. Returns the governed configuration
    /// and whether it limited the step.
    pub(super) fn project_closing(
        &self,
        prev_q: &[f64; GOV_DOF],
        cand_q: &[f64; GOV_DOF],
        grad: &[f64; GOV_DOF],
        d_now: f64,
        dt: f64,
    ) -> ([f64; GOV_DOF], bool) {
        let step: [f64; GOV_DOF] = std::array::from_fn(|i| cand_q[i] - prev_q[i]);
        // Predicted change in clearance over this tick if the full step is taken, and
        // the most clearance the barrier permits losing.
        let predicted_delta_d = dot(grad, &step);
        let max_loss = self.allowed_closing(d_now) * dt;
        let norm_sq = dot(grad, grad);
        if predicted_delta_d >= -max_loss || norm_sq <= MIN_GRADIENT_NORM_SQ {
            // Unrestricted motion must carry the commanded value itself: the
            // clamp below reconstructs each DOF from `prev + delta`, which is
            // not bit-identical to the candidate.
            return (*cand_q, false);
        }
        // Subtract just enough of the closing component (along the gradient) to
        // land on the barrier `grad . step = -max_loss`.
        let excess = (predicted_delta_d + max_loss) / norm_sq;
        let projected: [f64; GOV_DOF] =
            std::array::from_fn(|i| prev_q[i] + step[i] - excess * grad[i]);
        // The minimum-norm correction spreads the closing reduction along the
        // gradient, which can jog a DOF the operator did not drive or reverse one
        // they did. Clamp each DOF's governed step into [0, commanded step]: a held
        // DOF stays put, none reverses, separating motion is untouched.
        let governed = std::array::from_fn(|i| {
            prev_q[i] + (projected[i] - prev_q[i]).clamp(step[i].min(0.0), step[i].max(0.0))
        });
        (governed, true)
    }

    /// Permitted closing speed (m/s) at signed surface distance `d`: zero at or
    /// under `d_stop`, ramping linearly to the full approach at `d_safe`, and
    /// unbounded beyond it, where the barrier is not an active constraint.
    ///
    /// Inside an actual overlap this is a recovery budget rather than an
    /// approach one. Leaving it at zero there makes the projection itself the
    /// trap: an escape whose first move goes deeper is a closing step, so it
    /// would be projected away and the arms could never leave the collision.
    fn allowed_closing(&self, d: f64) -> f64 {
        if d < 0.0 {
            RECOVERY_LOSS_M_PER_S
        } else if d <= self.d_stop {
            0.0
        } else if d >= self.d_safe {
            f64::INFINITY
        } else {
            APPROACH_VELOCITY_AT_SAFE_M_S * (d - self.d_stop) / (self.d_safe - self.d_stop)
        }
    }

    /// The per-DOF hold mask for [`Self::clip_to_floor_along`]. When exactly
    /// one arm's own motion (the other held at `prev`) opens the clearance,
    /// that separating arm's [`holdable`] DOF are held at `target` while the
    /// floor scan clips the rest, so the separating arm keeps its escape while
    /// the approaching arm is clipped. Two operators can then retreat
    /// independently even while one pushes in. When both arms approach
    /// (nothing separates), or both separate alone yet may jointly close,
    /// nothing is held and the shared-segment backstop governs both. An arm's
    /// motion is the motion of its holdable DOF: on a path of both arms, its
    /// gripper alone.
    ///
    /// A hold pins that arm at `target` for the whole scan, so the exempted
    /// scan never probes the held arm's own sweep. What proves that sweep is
    /// the re-scan in [`Self::clip_to_floor_along`], which walks the true
    /// `prev`-to-published line with nothing held and therefore covers both
    /// arms moving together: the point that goes out is proved on the path the
    /// arms travel, however the hold was decided. So the hold only has to leave
    /// the exempted scan a legal starting point, which the endpoint check does
    /// by requiring the solo configuration to be at or above `d_prev`.
    ///
    /// An arm whose holdable DOF do not move is never held: pinning them is a
    /// no-op that only disables the scan's Lipschitz skip.
    fn separating_hold(
        &mut self,
        prev: &[f64; GOV_DOF],
        target: &[f64; GOV_DOF],
        d_prev: f64,
        paths: &[PathArms],
    ) -> [bool; GOV_DOF] {
        let holdable = holdable(paths);
        let arm_dofs =
            move |arm: ArmId| (0..GOV_DOF).filter(move |&i| holdable[i] && dof_arm(i) == arm);
        let solo = |arm: ArmId| -> [f64; GOV_DOF] {
            let mut q = *prev;
            for i in arm_dofs(arm) {
                q[i] = target[i];
            }
            q
        };
        let moves = |arm: ArmId| arm_dofs(arm).any(|i| target[i] != prev[i]);
        let separates =
            |g: &mut Self, q: &[f64; GOV_DOF]| g.distance_at(q).is_some_and(|d| d >= d_prev);
        let held = held_arm(PerArm::from_fn(|arm| {
            moves(arm) && separates(self, &solo(arm))
        }));
        std::array::from_fn(|i| holdable[i] && Some(dof_arm(i)) == held)
    }

    /// Retract `target` to the furthest point along the segment from `prev`
    /// that stays at or above the step floor.
    ///
    /// The strict scan runs first: its Lipschitz skip decides most ticks
    /// without a single probe, and a clear strict scan means nothing would be
    /// clipped, so there is nothing for an exemption to protect. Only a strict
    /// clip pays for the separating-arm machinery.
    ///
    /// A granted exemption scans a two-leg path (the separating arm already at
    /// `target`, the other interpolating from there), but the arms travel the
    /// straight line from `prev` to whatever is published. So the exemption's
    /// result is re-scanned on that line: the point that goes out is always one
    /// this stage proved on the path the arms will actually take, and the
    /// exemption survives as the endpoint it aims for rather than as an
    /// unverified shortcut. When the exemption changes nothing, the strict
    /// clip stands.
    ///
    /// The arms of each path in `paths` step along their path: the hold
    /// never parts the joints of a path of both arms ([`holdable`]), since
    /// the scan moves them by one share of their step, so a clip keeps both
    /// arms on their path.
    pub(super) fn clip_to_floor_along(
        &mut self,
        prev: &[f64; GOV_DOF],
        target: &[f64; GOV_DOF],
        paths: &[PathArms],
        d_now: f64,
        dt: f64,
    ) -> Clip {
        const NO_HOLD: [bool; GOV_DOF] = [false; GOV_DOF];
        let strict = match self.scan_to_floor(prev, target, &NO_HOLD, d_now, dt) {
            Clip::Clear => return Clip::Clear,
            Clip::Clipped(q) => q,
        };
        let hold = self.separating_hold(prev, target, d_now, paths);
        if hold == NO_HOLD {
            return Clip::Clipped(strict);
        }
        let exempted = match self.scan_to_floor(prev, target, &hold, d_now, dt) {
            // The exemption reaches `target`, so the re-scan below would repeat
            // the strict scan argument for argument, and that scan clipped.
            Clip::Clear => return Clip::Clipped(strict),
            Clip::Clipped(q) => q,
        };
        match self.scan_to_floor(prev, &exempted, &NO_HOLD, d_now, dt) {
            Clip::Clipped(q) => Clip::Clipped(q),
            Clip::Clear => Clip::Clipped(exempted),
        }
    }

    /// Walk from `prev` toward `target` and return [`Clip::Clipped`] at the first
    /// point where the straight segment drops below the step floor, or
    /// [`Clip::Clear`] if every probed point stays at or above it. `d_now` is the
    /// clearance at `prev`; the floor is [`step_floor`]`(d_stop, d_now, dt)`,
    /// so `prev` itself is at or above it by construction. Bimanual distance is
    /// not monotone along a joint-space segment, so this probes interior points
    /// (one per `MAX_PROBE_ARC_RAD` of joint motion, at least
    /// `SEGMENT_SAMPLES_MIN`) to find the first breach (an endpoint check alone
    /// can step over a pocket, and a fixed grid can step over one on a large
    /// jump) and retracts to the last point it proved clear. A failed query
    /// counts as a breach (so a model-rejected configuration is never
    /// returned), retracting conservatively.
    ///
    /// The retraction is to a proven point, never an interpolated one, so
    /// refining the breach any further would only recover motion the next tick
    /// takes anyway: the chase re-issues its command every cycle, so a step cut
    /// a fraction of a probe short is made up immediately, and the arms park in
    /// the same place either way.
    ///
    /// A clear probe vouches for the span it covers: the model's Lipschitz
    /// step bound caps how fast the clearance can change along the segment, so
    /// clearance `d` at blend `t` proves everything up to
    /// `t + (d - floor) / bound` cannot cross the floor, and those probes are
    /// skipped. Ample clearance clears the whole segment from its starting
    /// clearance alone (holding still, slow motion far apart), while in-band
    /// margins are small, so the region where clipping actually happens keeps
    /// its full probe density. The vouching base at `t = 0` is `d_now` only
    /// when nothing is held (a hold starts the scan from a different
    /// configuration); every later base is a probe on the scanned path itself,
    /// so advancement is sound under a hold too.
    pub(super) fn scan_to_floor(
        &mut self,
        prev: &[f64; GOV_DOF],
        target: &[f64; GOV_DOF],
        hold: &[bool; GOV_DOF],
        d_now: f64,
        dt: f64,
    ) -> Clip {
        let floor = step_floor(self.d_stop, d_now, dt);
        // Held DOF (a separating arm) sit at `target` for the whole scan; the
        // rest interpolate, so the clip retracts only the approaching arm.
        let point_at = |t: f64| -> [f64; GOV_DOF] {
            std::array::from_fn(|i| {
                if hold[i] {
                    target[i]
                } else {
                    prev[i] + t * (target[i] - prev[i])
                }
            })
        };
        // Probe spacing: one probe per resolution unit of the fastest-moving
        // DOF, so no DOF moves more than its probe resolution between probes.
        // The step reaching here is bounded by `DofSpeed` in the limit stage,
        // which is what keeps this count finite.
        let mut max_probe_ratio = 0.0_f64;
        for i in 0..GOV_DOF {
            let excursion = (target[i] - prev[i]).abs();
            let probe_resolution = if i < DUAL_DOF {
                MAX_PROBE_ARC_RAD
            } else {
                MAX_PROBE_GRIPPER_FRAC
            };
            max_probe_ratio = max_probe_ratio.max(excursion / probe_resolution);
        }
        // The bound scales linearly along the straight segment, so clearance
        // `d` vouches for `(d - floor) / bound` of blend beyond its own point.
        // A zero bound means the step cannot change the clearance at all, and
        // vouches for everything.
        let bound = self.model.step_bound(&step_change(prev, target));
        let vouches_until = |t: f64, d: f64| -> f64 {
            if bound > 0.0 {
                t + (d - floor) / bound
            } else {
                f64::INFINITY
            }
        };
        // One probe per resolution unit of the fastest-moving DOF (floored for
        // tiny steps); no fixed ceiling, so the spacing guarantee holds for any
        // step the limit stage admits.
        let samples = (max_probe_ratio.ceil() as usize).max(SEGMENT_SAMPLES_MIN);
        let unheld_base = hold.iter().all(|&h| !h);
        // `last_clear` is the furthest blend PROVEN at or above the floor, by a
        // probe or by a vouched span; the retraction below may only return a
        // proven point.
        let mut vouched = if unheld_base {
            vouches_until(0.0, d_now)
        } else {
            0.0
        };
        if vouched >= 1.0 {
            return Clip::Clear;
        }
        let mut last_clear = vouched;
        let mut s = 1;
        while s <= samples {
            let t = s as f64 / samples as f64;
            if t <= vouched {
                s += 1;
                continue;
            }
            match self.distance_at(&point_at(t)) {
                Some(d) if d >= floor => {
                    vouched = vouches_until(t, d);
                    if vouched >= 1.0 {
                        return Clip::Clear;
                    }
                    last_clear = t.max(vouched);
                    s += 1;
                }
                _ => return Clip::Clipped(point_at(last_clear)),
            }
        }
        Clip::Clear
    }
}

/// The DOF the separating hold may pin at their target: every DOF but the
/// joints of a path of both arms in `paths`, which move by one share of
/// their step together. A path of one arm has its joints on that arm, which
/// a hold pins or leaves whole.
fn holdable(paths: &[PathArms]) -> [bool; GOV_DOF] {
    let on_a_path_of_both =
        |i: usize| paths.contains(&PathArms::Both) && PathArms::Both.joints().any(|j| j == i);
    std::array::from_fn(|i| !on_a_path_of_both(i))
}

/// The arm the separating hold pins at its target: the one arm whose own
/// motion separates the arms, and none when no arm or more than one does.
fn held_arm(separating: PerArm<bool, ARMS>) -> Option<ArmId> {
    let mut separating_arms = PerArm::<(), ARMS>::ids().filter(|&arm| separating[arm]);
    match (separating_arms.next(), separating_arms.next()) {
        (Some(arm), None) => Some(arm),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::NO_PATHS;
    use crate::test_support::{LEFT, RIGHT};

    /// The separating hold's choice of arm, pinned as a table without any
    /// geometry.
    #[test]
    fn the_hold_pins_the_one_arm_that_separates_alone() {
        assert_eq!(held_arm(PerArm::new([true, false])), Some(LEFT));
        assert_eq!(held_arm(PerArm::new([false, true])), Some(RIGHT));
        assert_eq!(held_arm(PerArm::new([true, true])), None);
        assert_eq!(held_arm(PerArm::new([false, false])), None);
    }

    /// On a path of both arms the hold may pin the grippers alone; on paths
    /// of one arm, or none, it may pin every DOF.
    #[test]
    fn a_path_of_both_arms_leaves_only_the_grippers_holdable() {
        let grippers_only: [bool; GOV_DOF] = std::array::from_fn(|i| i >= DUAL_DOF);
        assert_eq!(holdable(&[PathArms::Both]), grippers_only);
        let every_dof = [true; GOV_DOF];
        assert_eq!(holdable(NO_PATHS), every_dof);
        let one_arm_each = [PathArms::One(LEFT), PathArms::One(RIGHT)];
        assert_eq!(holdable(&one_arm_each), every_dof);
    }
}
