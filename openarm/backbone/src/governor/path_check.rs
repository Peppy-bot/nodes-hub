//! The check of a joint path against the guard before the arms move along
//! it: the path, probed at the floor scan's resolution from its start, where
//! the arms it moves stand, must stay at or above the floor the step floor
//! holds each step at from where the step starts, and must not close the gap
//! along it from a probe at or inside the stop distance, where the closing
//! barrier holds every closing step. The check runs on a copy of the
//! governor's collision model ([`PathChecker`]), so it runs beside the
//! control loop, against the guard as the governor held it when the path
//! started ([`GuardAt`]).

use std::sync::atomic::{AtomicBool, Ordering};

use control_core::path_timing::RoundedPath;
use control_core::positive_finite::PositiveFinite;

use super::barrier::step_floor;
use super::model::ConfiguredModel;
use super::{
    DUAL_DOF, GOV_DOF, GovState, Governor, MAX_PROBE_ARC_RAD, MIN_GRADIENT_NORM_SQ, NearestPair,
    PathArms, RECOVERY_LOSS_M_PER_S, Step, along_paths, concat, dot, governed_gradient, split,
    step_change,
};

/// The guard a joint path is checked against: where the governor holds the
/// limbs when the path starts, where the check keeps each limb the path does
/// not move, and the governor's stop distance then, none while the governor
/// is disabled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuardAt {
    held: GovState,
    d_stop: Option<f64>,
}

impl GuardAt {
    /// A disabled guard with the limbs at `held`: the guard of a test whose
    /// checker judges the path alone.
    #[cfg(test)]
    pub fn disabled(held: GovState) -> Self {
        Self { held, d_stop: None }
    }
}

impl Governor {
    /// The guard a joint path that starts with the limbs at `held` is
    /// checked against: the governor's stop distance now, none while it is
    /// disabled.
    pub fn guard_at(&self, held: GovState) -> GuardAt {
        GuardAt {
            held,
            d_stop: self.enabled.then_some(self.d_stop),
        }
    }

    /// The checker of joint paths on a copy of the governor's collision
    /// model.
    pub fn path_checker(&self) -> PathChecker {
        PathChecker {
            model: self.model.duplicate(),
        }
    }
}

/// The check of joint paths against the robot's own body, on a copy of the
/// governor's collision model: a check takes as long as its path is long,
/// and the governor's model serves the control tick.
pub struct PathChecker {
    model: ConfiguredModel,
}

/// Why the check of a path fails: the guard would hold an arm short of it,
/// or the check stopped.
#[derive(Debug)]
pub enum PathBlocked {
    /// The check stopped before its end: the goal of the path ended.
    Stopped,
    /// The clearance at a point of the path, its start among them, could
    /// not be measured.
    Unmeasured,
    /// The path brings `pair` under the guard's stop distance.
    UnderStop { pair: NearestPair, d_stop: f64 },
    /// The limbs start under the stop distance, `start` apart, and the path
    /// closes the gap where `pair` stands its distance apart, inside the stop
    /// distance: from a probe before it, or along the path there.
    ClosesInsideStop { pair: NearestPair, start: f64 },
    /// The limbs start in contact, `depth` deep, and the path closes the gap
    /// before it leaves the contact, where `pair` stands its distance apart:
    /// the check allows no loss of clearance there, where the guard allows a
    /// bounded rate of it while the arm escapes.
    DeeperInContact { pair: NearestPair, depth: f64 },
}

/// The clearance `d` (m) in the words of a refusal: how far apart, or how
/// deep in each other.
fn apart(d: f64) -> String {
    if d >= 0.0 {
        format!("{:.1} mm apart", d * 1000.0)
    } else {
        format!("{:.1} mm deep in each other", -d * 1000.0)
    }
}

impl std::fmt::Display for PathBlocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mm = |m: f64| m * 1000.0;
        match self {
            Self::Stopped => write!(f, "the check of the path stopped: its goal ended"),
            Self::Unmeasured => write!(
                f,
                "the collision guard cannot measure the clearance at a point of the path"
            ),
            Self::UnderStop { pair, d_stop } => write!(
                f,
                "the path brings {} and {} to {:.1} mm apart, inside the collision guard's \
                 {:.1} mm stop distance, where the guard would hold the arm short",
                pair.link_a,
                pair.link_b,
                mm(pair.distance),
                mm(*d_stop)
            ),
            Self::ClosesInsideStop { pair, start } => write!(
                f,
                "the robot starts with two of its links {}, inside the collision guard's stop \
                 distance, where the guard holds every step that closes the gap, and the path \
                 closes the gap between {} and {} where they stand {}, so the guard would hold \
                 the arm short",
                apart(*start),
                pair.link_a,
                pair.link_b,
                apart(pair.distance)
            ),
            Self::DeeperInContact { pair, depth } => write!(
                f,
                "the robot starts {:.1} mm deep in contact with itself, where the collision \
                 guard lets a step lose clearance at up to {:.0} mm/s while the arm escapes and \
                 the check lets it lose none, and the path closes the gap between {} and {} \
                 where they stand {}",
                mm(*depth),
                mm(RECOVERY_LOSS_M_PER_S),
                pair.link_a,
                pair.link_b,
                apart(pair.distance)
            ),
        }
    }
}

impl PathChecker {
    /// Whether `guard` lets `arms` along `path`, whose points carry both arms'
    /// joints, from its start, with every other governed DOF where the guard
    /// holds it. The path is probed at most [`MAX_PROBE_ARC_RAD`] apart along
    /// it, as the floor scan probes a step, each probe against the floor the
    /// guard holds a step at from where the step starts: the stop distance, or
    /// the clearance there when that lies under it. Inside an overlap the floor
    /// is the clearance there: the check runs on no clock, so it allows none of
    /// the loss the guard's recovery rate allows each tick. The probes hold
    /// each rest of the path ([`RoundedPath::arcs_every`]), where the path
    /// turns. From each probe at or inside the stop distance, the clearance
    /// must not close along the way the path heads there to first order, as the
    /// closing barrier judges a step there (`PathChecker::barrier_holds`): the
    /// barrier holds such a step however short, so a path that closes the gap
    /// there, though it opens it again before the next probe, stalls there.
    /// Inside the stop distance the floor is the clearance itself, so a probe
    /// vouches for no stretch after it: the check reads closing at its probes
    /// alone there, and a stretch that closes the gap strictly between two
    /// probes, both of which open it, the later no closer than the earlier,
    /// passes. A clear probe beyond the stop
    /// distance vouches, by the model's step bound, for the probes after it
    /// until the bound could reach the floor, so ample clearance probes little.
    /// A disabled guard lets every path through. Once `stop` is raised, the
    /// check ends at its next measured probe.
    pub fn check_path(
        &mut self,
        arms: PathArms,
        guard: &GuardAt,
        path: &RoundedPath,
        stop: &AtomicBool,
    ) -> Result<(), PathBlocked> {
        let Some(d_stop) = guard.d_stop else {
            return Ok(());
        };
        let floor_from = |d: f64| step_floor(d_stop, d, 0.0);
        let spacing = PositiveFinite::parse(MAX_PROBE_ARC_RAD).expect("a positive resolution");
        let arcs = path.arcs_every(spacing);
        let probes: Vec<[f64; GOV_DOF]> = arcs
            .iter()
            .map(|&s| concat(&arms.placed(&guard.held, &path.at(s))))
            .collect();
        let d_start = self
            .model
            .distance(&probes[0])
            .ok_or(PathBlocked::Unmeasured)?;
        // The floor of the next probe, from the clearance of the last probe
        // measured, and how much clearance that probe had over it, less what
        // the steps since could have taken of it. A probe the margin vouches
        // for stands past the stop distance, so the floor after it is the
        // stop distance still. `inside` holds while the probe a window starts
        // at was measured at or inside the stop distance.
        let mut floor = floor_from(d_start);
        let mut margin = d_start - floor;
        let mut inside = d_start <= d_stop;
        for (w, &s) in probes.windows(2).zip(&arcs) {
            if inside && self.barrier_holds(arms, &w[0], &path.heading_at(s)) {
                return Err(self.blocked_at(&w[0], floor, d_start, d_stop));
            }
            margin -= self.model.step_bound(&step_change(&w[0], &w[1]));
            if margin > 0.0 {
                continue;
            }
            if stop.load(Ordering::Acquire) {
                return Err(PathBlocked::Stopped);
            }
            match self.model.distance(&w[1]) {
                Some(d) if d >= floor => {
                    floor = floor_from(d);
                    margin = d - floor;
                    inside = d <= d_stop;
                }
                _ => return Err(self.blocked_at(&w[1], floor, d_start, d_stop)),
            }
        }
        Ok(())
    }

    /// Whether the guard's closing barrier holds a step of `arms` from `q`,
    /// a probe at or inside the stop distance, along `heading`, the path's
    /// unit tangent there through both arms' joints, as it judges a step
    /// there: the clearance closes along the step to first order, and its
    /// gradient, reduced to the step on the path's joints, is one the barrier
    /// steers on. At a touch the clearance has no gradient, and the barrier
    /// stands down.
    fn barrier_holds(&mut self, arms: PathArms, q: &[f64; GOV_DOF], heading: &[f64]) -> bool {
        let Ok(gradient) = self.model.gradient(q) else {
            return false;
        };
        let change: [f64; GOV_DOF] =
            std::array::from_fn(|i| if i < DUAL_DOF { heading[i] } else { 0.0 });
        let next: [f64; GOV_DOF] = std::array::from_fn(|i| q[i] + change[i]);
        let Some(step) = Step::new(&split(q), &split(&next)) else {
            return false;
        };
        let along = along_paths(&governed_gradient(&gradient), &step, &[arms]);
        dot(&along, &change) < 0.0 && dot(&along, &along) > MIN_GRADIENT_NORM_SQ
    }

    /// The refusal of a path at its probe `q`, under `floor` or closing the
    /// gap from it, the limbs starting `d_start` apart under a guard that
    /// stops at `d_stop`. A floor under 0 is the clearance of a contact the
    /// path has stayed in since its start.
    fn blocked_at(
        &mut self,
        q: &[f64; GOV_DOF],
        floor: f64,
        d_start: f64,
        d_stop: f64,
    ) -> PathBlocked {
        let Some(pair) = self.model.proximity(q) else {
            return PathBlocked::Unmeasured;
        };
        if floor < 0.0 {
            PathBlocked::DeeperInContact {
                pair,
                depth: -d_start,
            }
        } else if floor < d_stop {
            PathBlocked::ClosesInsideStop {
                pair,
                start: d_start,
            }
        } else {
            PathBlocked::UnderStop { pair, d_stop }
        }
    }
}
