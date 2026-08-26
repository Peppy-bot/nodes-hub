//! Self-collision governor: a closing-velocity barrier over the bimanual
//! collision model. Each tick it limits only the component of the commanded joint
//! step that closes the nearest gap between the two arms, leaving tangential and
//! separating motion at full speed. The permitted approach speed ramps from a full
//! approach at `d_safe` down to zero at `d_stop` (a Faverjon-Tournassoud velocity
//! damper / exponential control-barrier form), so the surface clearance never
//! crosses `d_stop` in continuous time. URDF-based, so it governs sim and hardware
//! identically. Throttle/stop/clear transitions are logged.
//!
//! The barrier needs the gradient of the min surface distance with respect to the
//! 14 joints; the collision model computes it analytically from the nearest pair's
//! witness points (one distance query, no finite differencing). The residual
//! approximation is the per-tick linearization of the step, bounded by the small
//! control-rate step and absorbed by the exact line-search backstop: each tick the
//! realized clearance is held at or above the step floor, so an approach never
//! crosses `d_stop`. Inside an actual overlap the floor instead decays at a
//! bounded rate, so a recovery whose path sweeps deeper before it separates is
//! throttled rather than refused outright.
//!
//! The barrier above shapes only the commanded stream and is blind to how well the
//! arms track it. A second, independent measured-state monitor (defense in depth)
//! holds closing motion, judged per arm, whenever the real clearance, from the
//! measured joint state, has closed past `MONITOR_TRIP_FRACTION * d_stop`, until
//! it recovers past `d_stop` (hysteresis, so jitter at the wall cannot chatter
//! the hold); an arm whose own motion opens the real gap always stays free. It
//! shares the governor enable, so the operator toggle gates the commanded
//! barrier and this tripwire together.
//!
//! The governor also runs the two speed caps (the per-DOF rate bound and the
//! operator's end-effector speed cap) as limiters ahead of the collision
//! stages. They are motion shaping, not collision avoidance: they run in both
//! modes and never touch the collision readout, so the toggle gates collision
//! avoidance and nothing else.

use bimanual_collision_model::BimanualCollisionModel;
use srs_model::Jacobian;
use tracing::{info, warn};

use crate::arm_pair::ArmPair;
use crate::torso::{TORSO_BODY, torso_regions};
use crate::types::{ARM_DOF, JointVec};

mod barrier;
mod limiters;
mod model;
mod sense;

use limiters::{DofSpeed, EeSpeed, Limiter, Limits, MeasuredTripwire};
use model::ConfiguredModel;
use sense::Sensed;

/// Why the governor refused to build.
///
/// The node parses these same launcher values by name before it gets here, so
/// in a launch these fire only if the two ever disagree. The governor still
/// checks them: it is the component that has to hold the caps, and it is built
/// directly by tests and by any future caller.
#[derive(Debug, thiserror::Error)]
pub enum GovernorError {
    #[error(
        "collision band invalid: require 0 < d_stop_m ({d_stop_m}) < d_safe_m ({d_safe_m}), both finite"
    )]
    Band { d_stop_m: f64, d_safe_m: f64 },

    #[error("{name} must be finite and > 0, got {got}")]
    NotPositiveFinite { name: &'static str, got: f64 },

    #[error(
        "max_gripper_rate_frac_s must be finite and at least {MIN_GRIPPER_RATE_FRAC_S}, got {0}"
    )]
    GripperRate(f64),

    #[error(transparent)]
    TorsoRegion(#[from] crate::torso::InvalidTorsoRegion),

    #[error("build the collision model")]
    CollisionModel(#[from] bimanual_collision_model::CollisionError),
}

/// Joints across both arms, left (0..7) then right (7..14).
const DUAL_DOF: usize = 2 * ARM_DOF;

/// Every governed degree of freedom: both arms' joints, then the left and
/// right grippers. One vector so the barrier, the floor scan, the separating
/// hold, and the measured-state monitor treat a gripper exactly like a joint.
const GOV_DOF: usize = DUAL_DOF + 2;
const LEFT_GRIPPER: usize = DUAL_DOF;
const RIGHT_GRIPPER: usize = DUAL_DOF + 1;

/// One governed configuration: both arms' joints and both grippers. The single
/// state [`Governor::govern`] throttles, scans, and monitors, so every
/// guarantee the arms get covers the fingers identically.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GovState {
    pub arms: ArmPair<JointVec>,
    /// Gripper opening fraction per side: 0 closed, 1 fully open.
    pub grippers: ArmPair<f64>,
}

impl GovState {
    pub fn new(arms: ArmPair<JointVec>, grippers: ArmPair<f64>) -> Self {
        Self { arms, grippers }
    }
}

/// A proposed motion of the whole governed configuration over one tick: where
/// the DOF are now (the last governed setpoint) and where the commanded stream
/// would put them. Flat over [`GOV_DOF`] because every stage of the pipeline
/// treats a gripper exactly like a joint.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Step {
    prev: [f64; GOV_DOF],
    target: [f64; GOV_DOF],
}

impl Step {
    /// Parse a proposed motion. A non-finite endpoint is an upstream fault to
    /// hold on, never a step to shape, so it has no representation in this
    /// type and every later stage may assume finite arithmetic.
    fn new(prev: &GovState, target: &GovState) -> Option<Self> {
        let step = Self {
            prev: concat(prev),
            target: concat(target),
        };
        let finite = |q: &[f64; GOV_DOF]| q.iter().all(|x| x.is_finite());
        (finite(&step.prev) && finite(&step.target)).then_some(step)
    }
}

/// Approach speed (m/s) the barrier permits at the outer edge of the band
/// (`d_safe`); it ramps linearly to zero at `d_stop`, so the clearance decays no
/// faster than this as the arms close. A module constant (not a node parameter) so
/// the node builds without regenerating peppygen; promote it to a parameter when
/// tuning on hardware.
const APPROACH_VELOCITY_AT_SAFE_M_S: f64 = 0.15;

/// Smallest opening rate worth accepting (fraction per second): below this a
/// full stroke takes over a minute, which is a misconfiguration rather than a
/// slow gripper.
const MIN_GRIPPER_RATE_FRAC_S: f64 = 0.01;

/// Probe resolution of the floor scan on an opening DOF (fraction), the opening
/// analog of `MAX_PROBE_ARC_RAD`: one probe per this much opening travel, ~0.7 mm
/// of gripper motion, comparable surface resolution to the joint arc.
const MAX_PROBE_GRIPPER_FRAC: f64 = 0.01;

/// Clearance an already-overlapping configuration may give up per second while
/// recovering. Escaping an interpenetration routinely sweeps deeper before it
/// separates, so a floor that forbids any loss refuses that escape every tick.
/// Another joint may still get the operator out, but the obvious command is
/// silently held with the readout reading `stopped`, and the natural next move
/// is to switch the guard off while the arms are in contact. This bounds how
/// fast an overlap may worsen instead of forbidding it: at the shipped 100 Hz,
/// 2 mm per tick, enough to cross the pockets seen in sim and slow enough to
/// stay controlled. It applies ONLY at negative clearance; the guarded band
/// above zero keeps the strict floor, so an approach still parks at `d_stop`.
const RECOVERY_LOSS_M_PER_S: f64 = 0.2;

/// A squared gradient norm at or below this (m/rad)² means the clearance is locally
/// insensitive to motion (no closing direction exists), so the step passes
/// unconstrained instead of dividing by a near-zero norm.
const MIN_GRADIENT_NORM_SQ: f64 = 1e-18;

/// Floor-scan resolution: the backstop walks a per-tick segment and probes at least
/// once every `MAX_PROBE_ARC_RAD` of joint motion, so the spatial resolution is
/// bounded regardless of how large the step is. Bimanual surface distance is not
/// monotone along a joint-space segment, so a fixed grid would step over a thin
/// pocket on a large jump; scaling the probe count to the segment length keeps the
/// resolution fixed. Tied to the smallest hull feature the scan must resolve (a few
/// mm of surface motion).
const MAX_PROBE_ARC_RAD: f64 = 0.01;
/// Probe-count floor for near-zero steps, where the `MAX_PROBE_ARC_RAD` count
/// rounds down to almost nothing. The spacing guarantee itself comes from the
/// per-arc count (a full-speed step still gets `excursion / MAX_PROBE_ARC_RAD`
/// probes); this floor only keeps a handful of probes on the smallest steps, so
/// it is sized for per-tick cost at high control rates rather than density.
/// There is no fixed ceiling; the count scales with the step, and the
/// `DofSpeed` limiter clamps every step to its velocity-limited bound, which
/// is what caps the count.
const SEGMENT_SAMPLES_MIN: usize = 4;

/// Disposition of the last governed cycle: the commanded motion passed
/// unrestricted, was scaled down to hold the band, or was denied entirely
/// (stop floor, measured-state monitor hold, or a fault hold). Ordered by
/// severity; transitions are logged once, not at the control rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Guard {
    Clear,
    Throttling,
    Stopped,
}

/// Outcome of walking a per-tick segment against the floor.
enum Clip {
    /// Every sampled point along the segment stayed at or above the floor.
    Clear,
    /// The segment crossed below the floor; carries the furthest point reached
    /// before the first crossing.
    Clipped([f64; GOV_DOF]),
}

/// The nearest checked pair at a configuration: signed surface distance (m;
/// positive is clearance, negative is penetration) and the two link names. The
/// operator proximity readout.
pub struct NearestPair {
    pub distance: f64,
    pub link_a: String,
    pub link_b: String,
}

pub struct Governor {
    model: ConfiguredModel,
    /// Closing is fully stopped at or under this signed surface distance (m).
    d_stop: f64,
    /// Outside this signed surface distance (m) the barrier is inactive.
    d_safe: f64,
    /// Largest single-joint speed (rad/s). The `DofSpeed` limiter clamps every
    /// step to it, which is what bounds the floor scan's probe count.
    max_joint_velocity_rad_s: f64,
    /// The operator's cap on a streamed hand's linear speed (m/s), applied by
    /// the EE-speed limiter to each side that carries a stream basis.
    max_ee_velocity_m_s: f64,
    /// Largest rate (opening fraction per second) the coordinator's chase may
    /// drive an opening: the opening analog of the joint speed cap, bounding
    /// each tick's opening step. The `DofSpeed` clamp and the floor scan's
    /// probe sizing key off it, so raising it buys stroke speed at the cost of
    /// probes per tick. The gripper node and its motor own the real opening
    /// speed; this only bounds what the governor has to scan.
    max_gripper_rate_frac_s: f64,
    enabled: bool,
    guard: Guard,
    /// Whether the measured-state monitor is currently holding. Latched with
    /// hysteresis: set when the real clearance closes past `MONITOR_TRIP_FRACTION *
    /// d_stop`, cleared once it recovers past `d_stop`.
    monitor_tripped: bool,
}

impl Governor {
    /// Build the bimanual model (with the tight torso proxy) and validate the band.
    /// Fails loudly on a bad URDF / mesh dir / base link or an invalid band, so a
    /// misconfigured backbone aborts at bringup instead of running ungoverned.
    #[allow(clippy::too_many_arguments)] // distinct model inputs + band + speed bound + toggle
    pub fn build(
        urdf: &str,
        meshes_dir: &str,
        left_base: &str,
        right_base: &str,
        d_stop: f64,
        d_safe: f64,
        max_joint_velocity_rad_s: f64,
        max_ee_velocity_m_s: f64,
        max_gripper_rate_frac_s: f64,
        enabled: bool,
    ) -> Result<Self, GovernorError> {
        if !valid_band(d_stop, d_safe) {
            return Err(GovernorError::Band {
                d_stop_m: d_stop,
                d_safe_m: d_safe,
            });
        }
        for (name, got) in [
            ("max_joint_velocity_rad_s", max_joint_velocity_rad_s),
            ("max_ee_velocity_m_s", max_ee_velocity_m_s),
        ] {
            if !(got.is_finite() && got > 0.0) {
                return Err(GovernorError::NotPositiveFinite { name, got });
            }
        }
        if !valid_gripper_rate(max_gripper_rate_frac_s) {
            return Err(GovernorError::GripperRate(max_gripper_rate_frac_s));
        }
        let model = ConfiguredModel::new(build_collision_model(
            urdf, meshes_dir, left_base, right_base,
        )?);
        Ok(Self {
            model,
            d_stop,
            d_safe,
            max_joint_velocity_rad_s,
            max_ee_velocity_m_s,
            max_gripper_rate_frac_s,
            enabled,
            guard: Guard::Clear,
            monitor_tripped: false,
        })
    }

    /// Flip the governor on/off at runtime (the operator toggle). Disabling resets
    /// the transition state so the next throttle re-logs from clear.
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled == self.enabled {
            return;
        }
        info!(
            "collision avoidance {}",
            if enabled {
                "ENABLED"
            } else {
                "DISABLED (speed caps only)"
            }
        );
        self.enabled = enabled;
        if !enabled {
            // Only the log/readout state resets. The tripwire latch is a fact
            // about where the bodies are, not about the toggle: dropping it here
            // would re-arm at the weaker untripped threshold and free closing
            // commands while the measured clearance is still inside the band.
            self.guard = Guard::Clear;
        }
    }

    /// The nearest checked pair's signed surface distance and link names at this
    /// configuration (fingers placed at its grippers), for the operator readout.
    /// Excluded pairs are never returned (the model drops them), and this is
    /// independent of the enabled state so the readout is live even in
    /// passthrough. `None` if the distance query fails.
    pub fn proximity(&mut self, state: &GovState) -> Option<NearestPair> {
        self.model.proximity(&concat(state))
    }

    /// Disposition of the last governed cycle, for the status readout. Clear
    /// while disabled (the collision machinery stands down; the speed caps
    /// still shape motion but are not collision events), except the non-finite
    /// candidate hold, which reports Stopped in either mode.
    pub fn guard(&self) -> Guard {
        self.guard
    }

    /// Retune the band at runtime (the operator's stop/safe controls). Rejects an
    /// invalid band (`0 < d_stop < d_safe` required), keeping the current one, and
    /// is a no-op when unchanged so it can be called every tick.
    pub fn set_band(&mut self, d_stop: f64, d_safe: f64) {
        if d_stop == self.d_stop && d_safe == self.d_safe {
            return;
        }
        if !valid_band(d_stop, d_safe) {
            warn!("collision: ignoring invalid band (d_stop={d_stop}, d_safe={d_safe})");
            return;
        }
        info!("collision band set to d_stop={d_stop} d_safe={d_safe}");
        self.d_stop = d_stop;
        self.d_safe = d_safe;
    }

    /// Retune the end-effector speed cap at runtime (the operator's speed
    /// control). Rejects a non-positive or non-finite value, keeping the
    /// current cap, and is a no-op when unchanged so it can be called every tick.
    pub fn set_ee_cap(&mut self, max_ee_velocity_m_s: f64) {
        if max_ee_velocity_m_s == self.max_ee_velocity_m_s {
            return;
        }
        if !(max_ee_velocity_m_s.is_finite() && max_ee_velocity_m_s > 0.0) {
            warn!("collision: ignoring invalid EE speed cap ({max_ee_velocity_m_s})");
            return;
        }
        info!("EE speed cap set to {max_ee_velocity_m_s} m/s");
        self.max_ee_velocity_m_s = max_ee_velocity_m_s;
    }

    /// Govern one bimanual step from `prev` to `cand` over `dt`, returning the
    /// governed configuration. Arms and grippers ride the same vector,
    /// so every guarantee the joints get covers the fingers identically.
    ///
    /// Six stages, in this order, each contractive with respect to the last:
    ///
    /// 1. **Parse.** A non-finite endpoint is an upstream fault, not a step. It
    ///    has no representation in [`Step`], so every stage below may assume
    ///    finite arithmetic.
    /// 2. **Limit (speed).** The per-DOF rate bound and the operator's
    ///    end-effector speed cap, as independent allowances combined by keeping
    ///    the most restrictive. These are motion-shaping controls, not
    ///    collision guards: they run in both modes, so the operator toggle
    ///    gates collision avoidance and nothing else.
    /// 3. **Sense.** One immutable snapshot of everything the limiters and
    ///    the projection decide on, so neither can perturb the other's view.
    ///    Only the clip stage queries the model after this, and every query
    ///    goes through [`model::ConfiguredModel`], which takes the whole
    ///    configuration per call: a read at a stale finger placement is
    ///    unrepresentable.
    /// 4. **Limit (tripwire).** The measured-state tripwire's allowance joins
    ///    the combination. Order-free with stage 2 by construction, and the
    ///    binding limiter falls out of the combination rather than being
    ///    threaded out of it.
    /// 5. **Project.** The closing-velocity barrier removes just enough of the
    ///    gap-closing component that the clearance loses no more than
    ///    `allowed_closing(d) * dt`, leaving tangential and separating motion
    ///    untouched. Directional, so no per-DOF fraction can express it and it
    ///    is not a limiter; it runs *after* them so its guarantee holds on the
    ///    step that is actually published.
    /// 6. **Clip.** The exact floor scan. Surface distance is not monotone
    ///    along a joint-space segment, so this is the only stage that can prove
    ///    the realized path stays clear, and it is what makes the limiters'
    ///    order-independence safe rather than merely convenient.
    ///
    /// `measured` is the real joint state and opening fractions, which only the
    /// measured-state tripwire reads. `hands` carries each side's end-effector
    /// Jacobian at the measured pose while that side follows the operator's
    /// stream; a planned move rides with `None` there, because it was budgeted
    /// against the cap at admission and capping it again per tick would stretch
    /// it past its validated duration. A disabled governor skips stages 3
    /// through 6; the stage 1 fault hold and the stage 2 speed caps still
    /// apply, so a non-finite candidate is never streamed and the speed
    /// controls do not vanish with the collision toggle.
    pub fn govern(
        &mut self,
        prev: &GovState,
        cand: &GovState,
        measured: &GovState,
        hands: &ArmPair<Option<Jacobian>>,
        dt: f64,
    ) -> GovState {
        // The tick period is an input like any other: a non-finite or
        // non-positive dt would make every speed budget infinite and the
        // overlap floor unbounded, so it holds exactly as a non-finite
        // candidate does. Production cannot produce one (the control rate is
        // bringup-asserted), which is no reason for the boundary to trust it.
        if !(dt.is_finite() && dt > 0.0) {
            self.report(Guard::Stopped, None, None);
            return *prev;
        }
        let Some(step) = Step::new(prev, cand) else {
            self.report(Guard::Stopped, None, None);
            return *prev;
        };
        let speed_limits = self.limit_speed(&step, hands, dt);
        if !self.enabled {
            self.report(Guard::Clear, None, None);
            return split(&speed_limits.apply(&step));
        }
        let Some(sensed) = self.sense(prev, cand, measured) else {
            self.report(Guard::Stopped, None, None);
            return *prev;
        };

        // The tripwire's allowance is folded into the same combination as the
        // speed caps, but judged separately: it is the only limiter that is a
        // collision guard, so it alone feeds the operator readout below.
        let tripwire = MeasuredTripwire {
            d_prev: sensed.d_prev,
            tripwire: sensed.tripwire.as_ref(),
        };
        let trip_allowance = tripwire.allow(&step);
        let trip_limits = Limits::unrestricted().add(tripwire.name(), trip_allowance);
        let limited = speed_limits
            .add(tripwire.name(), trip_allowance)
            .apply(&step);

        // The barrier stands down in deep penetration: with the witness points
        // coincident there is no separating direction to steer on. The floor
        // scan below still lets the operator drive out and still refuses to let
        // the penetration deepen, so standing down never traps them inside it.
        let (projected, throttled) = match sensed.grad {
            Some(grad) => self.project_closing(&step.prev, &limited, &grad, sensed.d_prev, dt),
            None => (limited, false),
        };

        let (governed, clipped) =
            match self.clip_to_floor(&step.prev, &projected, sensed.d_prev, dt) {
                Clip::Clear => (projected, false),
                Clip::Clipped(q) => (q, true),
            };

        let guard = self.disposition(&sensed, &trip_limits, throttled || clipped);
        self.report(guard, Some(&sensed.pair), trip_limits.tightest());
        split(&governed)
    }

    /// The always-on limiters: the per-DOF rate bound and the end-effector
    /// speed cap. Nothing here reads the model or another limiter's output, so
    /// the order of this list changes only which name is recorded on a tie,
    /// never the governed step.
    fn limit_speed(&self, step: &Step, hands: &ArmPair<Option<Jacobian>>, dt: f64) -> Limits {
        let dof_speed = DofSpeed {
            max_step: std::array::from_fn(|i| self.dof_speed_limit(i) * dt),
        };
        let ee_speed = EeSpeed {
            cap_m_s: self.max_ee_velocity_m_s,
            hands,
            dt,
        };
        let limiters: [&dyn Limiter; 2] = [&dof_speed, &ee_speed];
        limiters.iter().fold(Limits::unrestricted(), |limits, l| {
            limits.add(l.name(), l.allow(step))
        })
    }

    /// This tick's collision disposition, judged from the collision mechanisms
    /// alone (the tripwire's limits plus the barrier and floor scan): the speed
    /// caps shape ordinary motion and must not read as a collision event. A
    /// side denied outright, or a step already at the wall, reads as a stop;
    /// anything else the collision machinery restricted reads as throttling.
    fn disposition(&self, sensed: &Sensed, trip_limits: &Limits, shaped: bool) -> Guard {
        if !(shaped || trip_limits.restricted()) {
            return Guard::Clear;
        }
        if trip_limits.frozen() || sensed.d_prev <= self.d_stop {
            Guard::Stopped
        } else {
            Guard::Throttling
        }
    }

    /// Largest rate (fraction/s) the coordinator's chase may drive an opening
    /// candidate; the floor scan's probe sizing and the `DofSpeed` clamp are
    /// keyed to the same value.
    pub fn max_gripper_rate_frac_s(&self) -> f64 {
        self.max_gripper_rate_frac_s
    }

    /// Retune the opening rate at runtime, mirroring [`set_ee_cap`]. Rejects a
    /// value the governor cannot scan against, keeping the current rate, and is
    /// a no-op when unchanged so it can be called every tick.
    pub fn set_gripper_rate(&mut self, max_gripper_rate_frac_s: f64) {
        if max_gripper_rate_frac_s == self.max_gripper_rate_frac_s {
            return;
        }
        if !valid_gripper_rate(max_gripper_rate_frac_s) {
            warn!("collision: ignoring invalid gripper opening rate ({max_gripper_rate_frac_s})");
            return;
        }
        info!("gripper opening rate set to {max_gripper_rate_frac_s} /s");
        self.max_gripper_rate_frac_s = max_gripper_rate_frac_s;
    }

    /// Largest per-tick step governed DOF `i` may take, from the chase's arm
    /// velocity limit or the opening rate limit. The floor scan's probe count
    /// and the `DofSpeed` clamp both key off this.
    fn dof_speed_limit(&self, i: usize) -> f64 {
        if i < DUAL_DOF {
            self.max_joint_velocity_rad_s
        } else {
            self.max_gripper_rate_frac_s()
        }
    }

    /// Signed clearance at a governed configuration. `None` on a query error
    /// so callers fail safe.
    fn distance_at(&mut self, q: &[f64; GOV_DOF]) -> Option<f64> {
        self.model.distance(q)
    }

    /// Record and log this tick's disposition, once per transition rather than
    /// at the control rate.
    ///
    /// The only place `guard` is assigned while running, so a fault taken on one
    /// path can never strand the readout on another. Every return recomputes it,
    /// including the passthrough one.
    fn report(&mut self, next: Guard, pair: Option<&NearestPair>, by: Option<&'static str>) {
        if next == self.guard {
            return;
        }
        let by = by.unwrap_or("barrier");
        match (next, pair) {
            (Guard::Stopped, Some(p)) => warn!(
                "collision: STOP - motion halted by {by} at d={:+.4} m between {} and {}",
                p.distance, p.link_a, p.link_b
            ),
            (Guard::Stopped, None) => warn!("collision: STOP - motion halted on a fault hold"),
            (Guard::Throttling, Some(p)) => warn!(
                "collision: throttling approach ({by}), d={:+.4} m, pair {}/{}",
                p.distance, p.link_a, p.link_b
            ),
            (Guard::Throttling, None) => warn!("collision: throttling approach ({by})"),
            (Guard::Clear, _) => info!("collision: clear, resuming full speed"),
        }
        self.guard = next;
    }
}

/// Build the bimanual collision model with the shared tight torso proxy, from
/// the embedded URDF and its materialized meshes.
fn build_collision_model(
    urdf: &str,
    meshes_dir: &str,
    left_base: &str,
    right_base: &str,
) -> Result<BimanualCollisionModel, GovernorError> {
    Ok(
        BimanualCollisionModel::builder(urdf, meshes_dir, left_base, right_base)
            .regions(TORSO_BODY, torso_regions()?)
            .build()?,
    )
}

/// A valid band requires finite `0 < d_stop < d_safe` (the ramp denominator
/// `d_safe - d_stop` is then positive).
pub(crate) fn valid_band(d_stop: f64, d_safe: f64) -> bool {
    d_stop.is_finite() && d_safe.is_finite() && d_stop > 0.0 && d_safe > d_stop
}

/// A valid opening rate is finite and at least [`MIN_GRIPPER_RATE_FRAC_S`], so
/// the chase always makes progress and the floor scan's probe count stays
/// finite.
pub(crate) fn valid_gripper_rate(rate_frac_s: f64) -> bool {
    rate_frac_s.is_finite() && rate_frac_s >= MIN_GRIPPER_RATE_FRAC_S
}

/// Pack a governed configuration into one vector: left joints, right joints,
/// left opening, right opening.
fn concat(s: &GovState) -> [f64; GOV_DOF] {
    std::array::from_fn(|i| match i {
        LEFT_GRIPPER => s.grippers.left,
        RIGHT_GRIPPER => s.grippers.right,
        i if i < ARM_DOF => s.arms.left[i],
        i => s.arms.right[i - ARM_DOF],
    })
}

/// Split a governed vector back into the configuration.
fn split(q: &[f64; GOV_DOF]) -> GovState {
    GovState {
        arms: ArmPair::new(
            std::array::from_fn(|i| q[i]),
            std::array::from_fn(|i| q[ARM_DOF + i]),
        ),
        grippers: ArmPair::new(q[LEFT_GRIPPER], q[RIGHT_GRIPPER]),
    }
}

/// True if governed DOF `i` belongs to the left half (arm joints or opening).
fn is_left_dof(i: usize) -> bool {
    i < ARM_DOF || i == LEFT_GRIPPER
}

fn dot(a: &[f64; GOV_DOF], b: &[f64; GOV_DOF]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::limiters::measured_tripwire::MONITOR_TRIP_FRACTION;
    use super::*;
    use crate::chase::rate_limited;

    /// Materialize a generation's bundled collision meshes so the file-based collision
    /// builder can fit hulls; the URDF itself comes from the same `HardwareVersion`.
    /// Written once per generation into a unique tempdir held for the test process:
    /// `cargo test` runs these in parallel, so re-writing per call would let one test
    /// truncate a mesh mid-read of another's `build()`; a unique path also avoids
    /// clashing with a concurrent test process on the same host.
    fn fixture_meshes_dir(version: openarm_description::HardwareVersion) -> std::path::PathBuf {
        static V1_DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
        static V2_DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
        let cell = match version {
            openarm_description::HardwareVersion::V1 => &V1_DIR,
            openarm_description::HardwareVersion::V2 => &V2_DIR,
        };
        cell.get_or_init(|| {
            let dir = tempfile::tempdir().expect("create scratch dir for collision meshes");
            version
                .write_meshes_to(dir.path())
                .expect("materialize collision meshes");
            dir
        })
        .path()
        .to_path_buf()
    }

    const D_STOP: f64 = 0.005;
    const D_SAFE: f64 = 0.02;
    const DT: f64 = 0.01;
    /// No side carries a stream basis: collision scenarios exercise the
    /// collision stages, not the EE cap.
    const NO_HANDS: &ArmPair<Option<Jacobian>> = &ArmPair {
        left: None,
        right: None,
    };
    /// Generous so the `DofSpeed` clamp never binds on the synthetic
    /// direct-jump configs these tests use; the clamp itself is covered separately.
    const MAX_JOINT_VELOCITY_RAD_S: f64 = 1000.0;
    /// Positive as build requires; irrelevant to these tests, which pass no
    /// stream basis, so the EE limiter never engages.
    const TEST_EE_CAP_M_S: f64 = 0.5;
    /// Fast enough that a test never sees the opening rate bind first.
    const TEST_GRIPPER_RATE_FRAC_S: f64 = 6.0;

    /// In-limit home; the elbow's one-sided lower limit is 0.05.
    fn home() -> ArmPair<JointVec> {
        ArmPair::new(
            [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
        )
    }

    /// Arms at `arms` with both grippers fully open: the widest finger envelope, which
    /// is what the arm-barrier scenarios governed against before grippers were
    /// governed DOF, so their tuned poses and distances carry over unchanged.
    fn at(arms: ArmPair<JointVec>) -> GovState {
        GovState::new(arms, ArmPair::new(1.0, 1.0))
    }

    fn governor_for(version: openarm_description::HardwareVersion, enabled: bool) -> Governor {
        let meshes_dir = fixture_meshes_dir(version);
        Governor::build(
            version.urdf(),
            meshes_dir.to_str().expect("meshes dir path is valid UTF-8"),
            version.base_link(openarm_description::Side::Left),
            version.base_link(openarm_description::Side::Right),
            D_STOP,
            D_SAFE,
            MAX_JOINT_VELOCITY_RAD_S,
            TEST_EE_CAP_M_S,
            TEST_GRIPPER_RATE_FRAC_S,
            enabled,
        )
        .expect("build governor from bundled description")
    }

    fn governor(enabled: bool) -> Governor {
        governor_for(openarm_description::HardwareVersion::V1, enabled)
    }

    fn v2_governor(enabled: bool) -> Governor {
        governor_for(openarm_description::HardwareVersion::V2, enabled)
    }

    #[test]
    fn v2_governor_builds_with_the_revolute_gripper() {
        // The v2 revolute finger joints must parse into live-placed finger
        // bodies; a build failure here means the revolute finger path regressed.
        v2_governor(true);
    }

    #[test]
    fn v2_finger_opening_changes_live_clearance_on_both_sides() {
        // The real v2 assets, not a fixture: sweep the wrists inward until a
        // finger (ee_link) is the nearest body with the grippers open, then pin
        // that closing the grippers strictly recovers clearance. v2 mirrors its
        // right gripper by flipping the finger joint limit range, so this fails
        // if either side's open/closed sense is read off the URDF limit order
        // instead of the meshes.
        let mut g = v2_governor(true);
        let pose_at = |t: f64| {
            let mut p = home();
            p.left[2] = t;
            p.right[2] = -t;
            p
        };
        let pose = (0..=40)
            .map(|i| pose_at(i as f64 * 0.05))
            .find(|p| {
                g.proximity(&at(*p))
                    .is_some_and(|n| n.link_a.contains("ee_link") || n.link_b.contains("ee_link"))
            })
            .expect("some wrists-inward pose has a finger as the nearest body when open");
        let d_open = g.proximity(&at(pose)).expect("query").distance;
        let d_closed = g
            .proximity(&GovState::new(pose, ArmPair::new(0.0, 0.0)))
            .expect("query")
            .distance;
        assert!(
            d_closed > d_open + 1e-4,
            "closing the v2 grippers should recover clearance: open {d_open:+.4}, closed {d_closed:+.4}"
        );
    }

    #[test]
    fn a_step_beyond_the_velocity_limit_is_clamped_not_panicked() {
        // A tiny velocity makes the bound (max_joint_velocity * DT) 5e-4 rad,
        // so any real step exceeds it. The scan sizes its probe count from that
        // bound, so an over-limit step would under-resolve the segment; this
        // `DofSpeed` clamps to the bound, so the scan keeps its resolution
        // precondition without the node going down over an upstream bug.
        let velocity = 0.05;
        let meshes_dir = fixture_meshes_dir(openarm_description::HardwareVersion::V1);
        let mut g = Governor::build(
            openarm_description::HardwareVersion::V1.urdf(),
            meshes_dir.to_str().expect("meshes dir path is valid UTF-8"),
            openarm_description::HardwareVersion::V1.base_link(openarm_description::Side::Left),
            openarm_description::HardwareVersion::V1.base_link(openarm_description::Side::Right),
            D_STOP,
            D_SAFE,
            velocity,
            TEST_EE_CAP_M_S,
            TEST_GRIPPER_RATE_FRAC_S,
            true,
        )
        .expect("build governor from bundled description");
        let prev = at(home());
        let mut cand = prev;
        cand.arms.left[0] += 0.5; // 0.5 rad >> the 5e-4 rad velocity-limited bound

        let governed = g.govern(&prev, &cand, &prev, NO_HANDS, DT);
        let bound = velocity * DT;
        let taken = governed.arms.left[0] - prev.arms.left[0];
        assert!(
            taken > 0.0,
            "an over-limit command must still move, just slower"
        );
        assert!(
            taken <= bound * (1.0 + 1e-9),
            "step {taken:.6} exceeded its velocity-limited bound {bound:.6}"
        );
        for (i, (&q, &p)) in governed
            .arms
            .right
            .iter()
            .zip(prev.arms.right.iter())
            .enumerate()
        {
            assert_eq!(q, p, "uncommanded right joint {i} moved");
        }
    }

    /// Both arms elbow-bent, j3 wrapping the wrists toward the centerline by `t`.
    fn wrists_inward(t: f64) -> ArmPair<JointVec> {
        ArmPair::new(
            [0.0, 0.0, t, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, -t, 0.4, 0.0, 0.0, 0.0],
        )
    }

    /// Signed clearance at a governed configuration (fingers placed at its grippers).
    fn distance(g: &mut Governor, s: &GovState) -> f64 {
        g.distance_at(&concat(s)).expect("finite config")
    }

    /// Step `from` toward `to` by at most `max` rad on each joint (a stand-in for
    /// the velocity-limited chase that feeds the governor in the real loop).
    fn chase(from: &ArmPair<JointVec>, to: &ArmPair<JointVec>, max: f64) -> ArmPair<JointVec> {
        let one = |f: &JointVec, t: &JointVec| {
            std::array::from_fn(|i| f[i] + (t[i] - f[i]).clamp(-max, max))
        };
        ArmPair::new(one(&from.left, &to.left), one(&from.right, &to.right))
    }

    /// Govern a chase from home toward a deeply folded pose until the clearance
    /// first drops just inside the band, so a test starts in the positive-clearance
    /// regime the barrier is designed for (not deep penetration).
    fn drive_into_band(g: &mut Governor) -> GovState {
        let target = wrists_inward(1.2);
        let mut q = at(home());
        for _ in 0..400 {
            if distance(g, &q) < D_SAFE {
                break;
            }
            let cand = at(chase(&q.arms, &target, 0.05));
            q = g.govern(&q, &cand, &q, NO_HANDS, DT);
        }
        q
    }

    #[test]
    fn disabled_passes_a_speed_legal_step_bit_exact() {
        // With the collision machinery down and no speed limiter binding (the
        // test velocity bound is huge; no side carries a stream basis), the
        // candidate must come back bit-exact: the followers require an
        // unrestricted DOF to carry the commanded value itself.
        let mut g = governor(false);
        let deep = at(wrists_inward(1.2));
        assert_eq!(
            g.govern(&at(home()), &deep, &at(home()), NO_HANDS, DT),
            deep
        );
        assert_eq!(g.guard(), Guard::Clear, "disabled restricts no collision");
    }

    #[test]
    fn far_apart_is_unthrottled() {
        let mut g = governor(true);
        // Home clearance is outside the band, so any step passes untouched.
        let cand = at(wrists_inward(0.2));
        assert!(
            distance(&mut g, &at(home())) >= D_SAFE,
            "home should sit outside the band"
        );
        assert_eq!(
            g.govern(&at(home()), &cand, &at(home()), NO_HANDS, DT),
            cand
        );
        assert_eq!(g.guard(), Guard::Clear, "unrestricted motion reads clear");
    }

    #[test]
    fn a_fast_approach_outside_the_band_is_not_throttled() {
        // The band's outer edge is where the barrier stops shaping motion:
        // past d_safe it is not an active constraint, and the exact floor scan
        // is what backstops the steps it passes.
        let mut g = governor(true);
        let start = at(wrists_inward(0.50));
        let cand = at(wrists_inward(0.62));
        let d_start = distance(&mut g, &start);
        assert!(d_start > D_SAFE, "the step must start outside the band");
        let closing = d_start - distance(&mut g, &cand);
        assert!(
            closing / DT > APPROACH_VELOCITY_AT_SAFE_M_S,
            "the step must close faster than the in-band allowance or it proves \
             nothing: closed {closing:.4} m in {DT} s"
        );
        assert_eq!(g.govern(&start, &cand, &start, NO_HANDS, DT), cand);
        assert_eq!(g.guard(), Guard::Clear, "outside the band reads clear");
    }

    #[test]
    fn separating_motion_always_passes() {
        let mut g = governor(true);
        // Drive just into the band, then step back toward home: separating motion
        // (clearance increasing) is never throttled.
        let q = drive_into_band(&mut g);
        let cand = at(chase(&q.arms, &home(), 0.02));
        assert_eq!(g.govern(&q, &cand, &q, NO_HANDS, DT), cand);
    }

    #[test]
    fn the_clip_stage_alone_allows_escape_and_never_deepens() {
        // In deep penetration the witnesses coincide, no gradient exists, and
        // the barrier stands down: `govern` runs the clip stage alone. That
        // stage by itself must still let the operator drive out and still
        // refuse to let the penetration deepen, so exercise it directly.
        let mut g = governor(true);
        let deep = at(wrists_inward(1.5));
        let d0 = distance(&mut g, &deep);
        let floor = d0.min(D_STOP);
        let clip_only = |g: &mut Governor, from: &GovState, to: &GovState| match g.clip_to_floor(
            &concat(from),
            &concat(to),
            d0,
            DT,
        ) {
            Clip::Clear => *to,
            Clip::Clipped(q) => split(&q),
        };
        // Escape toward home increases clearance: allowed, never frozen.
        let escape = at(chase(&deep.arms, &home(), 0.02));
        // While overlapping the floor decays at RECOVERY_LOSS_M_PER_S so a
        // recovery can cross a pocket; the step may lose at most that much.
        let slack = RECOVERY_LOSS_M_PER_S * DT + 1e-3;
        let out = clip_only(&mut g, &deep, &escape);
        assert_ne!(out, deep, "escape was frozen in place");
        assert!(
            distance(&mut g, &out) >= floor - slack,
            "escape dropped below the floor"
        );
        // A deeper command is still bounded: it may not run away past the slack.
        let deeper = at(chase(&deep.arms, &wrists_inward(2.0), 0.02));
        let held = clip_only(&mut g, &deep, &deeper);
        assert!(
            distance(&mut g, &held) >= floor - slack,
            "guard let penetration deepen without bound"
        );
    }

    #[test]
    fn held_arm_is_not_jogged_and_commanded_joints_never_reverse() {
        let mut g = governor(true);
        let q = drive_into_band(&mut g);
        // Command only the left arm further toward the centerline (closing); hold
        // the right exactly where it is.
        let pushed = chase(&q.arms, &wrists_inward(1.5), 0.02);
        let cand = GovState::new(ArmPair::new(pushed.left, q.arms.right), q.grippers);
        let governed = g.govern(&q, &cand, &q, NO_HANDS, DT);
        // The held right arm must not be jogged by the barrier's correction.
        assert_eq!(
            governed.arms.right, q.arms.right,
            "held right arm was jogged"
        );
        // Each commanded left joint's governed step stays within [0, commanded]:
        // same sign as the command, never larger, never reversed.
        for i in 0..ARM_DOF {
            let cmd = cand.arms.left[i] - q.arms.left[i];
            let gov = governed.arms.left[i] - q.arms.left[i];
            assert!(
                gov * cmd >= -1e-12 && gov.abs() <= cmd.abs() + 1e-12,
                "left joint {i}: governed step {gov} outside [0, {cmd}]"
            );
        }
    }

    #[test]
    fn tangential_motion_passes_unthrottled() {
        let mut g = governor(true);
        let q = drive_into_band(&mut g);
        // Build a step orthogonal to the distance gradient (purely tangential): it
        // does not change clearance, so the barrier must pass it unthrottled.
        let grad_pair = g.model.gradient(&concat(&q)).expect("gradient");
        let mut grad = [0.0; GOV_DOF];
        grad[..ARM_DOF].copy_from_slice(&grad_pair.grad_left);
        grad[ARM_DOF..DUAL_DOF].copy_from_slice(&grad_pair.grad_right);
        grad[LEFT_GRIPPER] = grad_pair.grad_openings[0];
        grad[RIGHT_GRIPPER] = grad_pair.grad_openings[1];
        let raw: [f64; GOV_DOF] = std::array::from_fn(|i| {
            if i < DUAL_DOF {
                ((i % 3) as f64 - 1.0) * 0.01
            } else {
                0.0
            }
        });
        let comp = dot(&raw, &grad) / dot(&grad, &grad);
        let tangential: [f64; GOV_DOF] = std::array::from_fn(|i| raw[i] - comp * grad[i]);
        let q16 = concat(&q);
        let cand = split(&std::array::from_fn(|i| q16[i] + tangential[i]));
        let governed = g.govern(&q, &cand, &q, NO_HANDS, DT);
        for i in 0..ARM_DOF {
            assert!(
                (governed.arms.left[i] - cand.arms.left[i]).abs() < 1e-9,
                "left tangential joint {i} was throttled"
            );
            assert!(
                (governed.arms.right[i] - cand.arms.right[i]).abs() < 1e-9,
                "right tangential joint {i} was throttled"
            );
        }
    }

    #[test]
    fn floor_holds_on_both_sides_of_the_scan_skip_boundary() {
        // The Lipschitz early-out skips the floor scan when the margin above
        // the floor exceeds the model's step bound. Engineer one closing step
        // whose bound sits just under that margin (skip may fire) and one just
        // over (the scan must run), and require the floor contract to hold on
        // both sides, so an off-by-margin or an underestimating bound cannot
        // silently reintroduce endpoint-trusting.
        let mut g = governor(true);
        let q = drive_into_band(&mut g);
        let d_now = distance(&mut g, &q);
        let margin = d_now - D_STOP;
        assert!(margin > 0.0, "setup: in band, above the stop");

        let prev16 = concat(&q);
        let toward16 = concat(&at(chase(&q.arms, &wrists_inward(1.5), 0.02)));
        let step16: [f64; GOV_DOF] = std::array::from_fn(|i| toward16[i] - prev16[i]);
        let dq = split(&step16);
        let bound = g.model.step_bound(
            &dq.arms.left,
            &dq.arms.right,
            &[dq.grippers.left, dq.grippers.right],
        );
        assert!(bound > 0.0, "setup: a closing step has a positive bound");

        for (scale, expect_skip) in [(0.9 * margin / bound, true), (1.1 * margin / bound, false)] {
            let target16: [f64; GOV_DOF] = std::array::from_fn(|i| prev16[i] + scale * step16[i]);
            // The two cases must actually straddle the skip predicate
            // (margin > bound of the scaled step), or a reshaped bound would
            // quietly turn this into a generic floor sweep.
            let scaled = split(&std::array::from_fn(|i| target16[i] - prev16[i]));
            let scaled_bound = g.model.step_bound(
                &scaled.arms.left,
                &scaled.arms.right,
                &[scaled.grippers.left, scaled.grippers.right],
            );
            assert_eq!(
                margin > scaled_bound,
                expect_skip,
                "scale {scale} does not straddle the skip predicate (margin {margin:.5}, bound {scaled_bound:.5})"
            );
            // No hold: this exercises the shared-segment skip predicate directly.
            match g.scan_to_floor(&prev16, &target16, &[false; GOV_DOF], d_now, DT) {
                Clip::Clear => {
                    // Whether cleared by the skip or by the scan, no point of
                    // the accepted segment may sit below the stop floor.
                    assert!(
                        segment_min(&mut g, &q, &split(&target16), 32) >= D_STOP - 1e-3,
                        "cleared segment dips below the stop at scale {scale}"
                    );
                }
                Clip::Clipped(p) => {
                    assert!(
                        distance(&mut g, &split(&p)) >= D_STOP - 1e-9,
                        "clipped point below the stop at scale {scale}"
                    );
                }
            }
        }
    }

    #[test]
    fn barrier_keeps_clearance_above_stop() {
        let mut g = governor(true);
        let target = wrists_inward(1.5);
        let mut q = at(home());
        let mut entered_band = false;
        for _ in 0..250 {
            let prev = q;
            let cand = at(chase(&prev.arms, &target, 0.02));
            q = g.govern(&prev, &cand, &prev, NO_HANDS, DT);
            let d = distance(&mut g, &q);
            entered_band |= d < D_SAFE;
            // The exact backstop holds the realized clearance at the floor, so the
            // stop distance is never breached on any tick.
            assert!(d >= D_STOP, "barrier breached: d={d:+.5}");
            // The whole realized path prev->governed, not just its endpoint, stays at
            // or above the floor (the step is small, so a coarse sweep resolves it).
            assert!(
                segment_min(&mut g, &prev, &q, 16) >= D_STOP - 1e-3,
                "the prev->governed path dipped below the stop"
            );
        }
        assert!(entered_band, "arms never approached into the band");
        // It should converge near the stop boundary, not stall far away.
        assert!(
            distance(&mut g, &q) < D_STOP + 4e-3,
            "did not settle near the stop distance"
        );
    }

    // The band is entered at whatever the always-on speed limiters allow, which
    // is well above the in-band allowance, so the floor has to hold for an
    // approach that arrives fast: the ramp shapes the crossing and the exact
    // scan is what guarantees it. The chase here is an order of magnitude
    // quicker than the one above.
    #[test]
    fn a_fast_approach_into_the_band_still_holds_the_floor() {
        let mut g = governor(true);
        let target = wrists_inward(1.5);
        let mut q = at(home());
        let mut crossed_fast = false;
        for _ in 0..250 {
            let prev = q;
            let d_prev = distance(&mut g, &prev);
            let cand = at(chase(&prev.arms, &target, 0.25));
            q = g.govern(&prev, &cand, &prev, NO_HANDS, DT);
            let d = distance(&mut g, &q);
            // What makes this a fast entry rather than a slower rerun: the tick
            // that crosses d_safe closes faster than the in-band allowance.
            crossed_fast |=
                d_prev > D_SAFE && d < D_SAFE && (d_prev - d) / DT > APPROACH_VELOCITY_AT_SAFE_M_S;
            assert!(d >= D_STOP, "barrier breached: d={d:+.5}");
            assert!(
                segment_min(&mut g, &prev, &q, 64) >= D_STOP - 1e-3,
                "the prev->governed path dipped below the stop"
            );
        }
        assert!(
            crossed_fast,
            "never crossed d_safe faster than the in-band allowance"
        );
    }

    /// Deterministic pseudo-random configurations, so a walk that finds
    /// something can be replayed from its seed.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn in_range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next_f64()
        }
        fn pose(&mut self) -> ArmPair<JointVec> {
            let mut one = || -> JointVec {
                [
                    self.in_range(-1.6, 1.6),
                    self.in_range(-1.6, 1.6),
                    self.in_range(-1.6, 1.6),
                    self.in_range(0.05, 2.2),
                    self.in_range(-1.2, 1.2),
                    self.in_range(-1.2, 1.2),
                    self.in_range(-1.2, 1.2),
                ]
            };
            ArmPair::new(one(), one())
        }
    }

    /// The floor scan proves its probes clear, not the space between them, so a
    /// realized path may dip this far under the floor before the next probe
    /// catches it. Sized from the measured worst case with headroom: at the
    /// shipped 5 mm stop it is a 5% erosion, and it is the residue the scan's
    /// probe spacing buys back only by spending queries per tick.
    const SCAN_PATH_RESIDUE_M: f64 = 2.5e-4;

    /// Whether a faster control loop is affordable. A jog is a speed, so a
    /// shorter period means a shorter segment and fewer probes per tick, while
    /// the budget shrinks in step: this reports which side wins. Run by hand:
    /// `cargo test --release control_rate_sweep -- --ignored --nocapture`
    #[test]
    #[ignore = "rate sweep, run by hand in release"]
    fn control_rate_sweep() {
        use std::time::Instant;

        /// The fastest joint-velocity limit the chase drives (J1/J2, rad/s), so
        /// each rate is measured at the longest segment it can be asked for.
        const J12_LIMIT_RAD_S: f64 = 16.754666;
        for rate_hz in [100.0_f64, 250.0, 500.0, 1000.0] {
            let dt = 1.0 / rate_hz;
            let budget_us = 1e6 / rate_hz;
            let mut g = v2_governor(true);
            let mut rng = Lcg(0x5eed_1729);
            let mut q = GovState::new(home(), ArmPair::new(0.0, 0.0));
            let mut target = rng.pose();
            let ticks = (rate_hz as usize) * 20;
            let mut times = Vec::with_capacity(ticks);
            for tick in 0..ticks {
                if tick % (ticks / 20) == 0 {
                    target = rng.pose();
                }
                let cand = GovState::new(chase(&q.arms, &target, J12_LIMIT_RAD_S * dt), q.grippers);
                let t0 = Instant::now();
                q = g.govern(&q, &cand, &q, NO_HANDS, dt);
                times.push(t0.elapsed().as_micros());
            }
            let over = times.iter().filter(|&&t| t as f64 > budget_us).count();
            times.sort_unstable();
            let pct = |p: usize| times[(times.len() - 1) * p / 100];
            println!(
                "TIMING rate {rate_hz:.0} Hz (budget {budget_us:.0} us): p50={} p95={} p99={} \
max={} us | over budget {over}/{}",
                pct(50),
                pct(95),
                pct(99),
                times[times.len() - 1],
                times.len()
            );
        }
    }

    /// Hand-run timing report (ignored in normal runs; timing asserts nothing):
    /// `cargo test --release governor_tick_timing_report -- --ignored --nocapture`
    #[test]
    #[ignore = "timing report, run by hand in release"]
    fn governor_tick_timing_report() {
        use std::time::Instant;

        let time_us = |label: &str, iters: u32, mut f: Box<dyn FnMut() + '_>| {
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("TIMING {label}: {us:.1} us/call");
            us
        };

        // The raw queries.
        let far = concat(&at(home()));
        let mut g2 = v2_governor(true);
        time_us(
            "distance (far)",
            2000,
            Box::new(move || {
                g2.model.distance(&far).unwrap();
            }),
        );
        let near = concat(&at(wrists_inward(1.05)));
        let mut g3 = v2_governor(true);
        time_us(
            "distance (in band)",
            2000,
            Box::new(move || {
                g3.model.distance(&near).unwrap();
            }),
        );
        let mut g4 = v2_governor(true);
        time_us(
            "gradient (in band)",
            2000,
            Box::new(move || {
                g4.model.gradient(&near).map(|_| ()).unwrap();
            }),
        );

        // The speed limiters alone (no model).
        let hands = ArmPair::new(
            Some(srs_model::Jacobian::zeros()),
            Some(srs_model::Jacobian::zeros()),
        );
        let prev = at(home());
        let mut cand = prev;
        cand.arms.left[2] += 0.01;
        let step = Step::new(&prev, &cand).unwrap();
        let gl = governor(true);
        time_us(
            "speed limiters fold",
            20000,
            Box::new(move || {
                std::hint::black_box(gl.limit_speed(&step, &hands, DT));
            }),
        );

        // Whole ticks by regime. `closed` keeps the v2 home pose out of the
        // wide-gripper torso overlap, so "far apart" really is far apart; the
        // overlap gets its own regime at the end.
        let closed = |arms: ArmPair<JointVec>| GovState::new(arms, ArmPair::new(0.0, 0.0));
        let regimes: [(&str, GovState, GovState, GovState); 5] = [
            (
                "tick: far apart (skip)",
                closed(home()),
                closed(chase(&home(), &wrists_inward(0.4), 0.02)),
                closed(home()),
            ),
            (
                "tick: in-band approach",
                at(wrists_inward(1.05)),
                at(chase(&wrists_inward(1.05), &wrists_inward(1.5), 0.02)),
                at(wrists_inward(1.05)),
            ),
            (
                "tick: hold at the wall",
                at(wrists_inward(1.093)),
                at(chase(&wrists_inward(1.093), &wrists_inward(1.5), 0.02)),
                at(wrists_inward(1.093)),
            ),
            (
                "tick: tripwire latched",
                at(wrists_inward(1.05)),
                at(chase(&wrists_inward(1.05), &wrists_inward(1.5), 0.02)),
                at(wrists_inward(1.15)),
            ),
            (
                "tick: penetration escape",
                at(home()),
                at(chase(&home(), &wrists_inward(0.4), 0.02)),
                at(home()),
            ),
        ];
        // The disabled tick: parse + speed limiters + apply, no model at all.
        {
            let mut g = v2_governor(false);
            let prev = at(wrists_inward(1.05));
            let cand = at(chase(&wrists_inward(1.05), &wrists_inward(1.5), 0.02));
            time_us(
                "tick: governor disabled",
                20000,
                Box::new(move || {
                    std::hint::black_box(g.govern(&prev, &cand, &prev, NO_HANDS, DT));
                }),
            );
        }
        // The planner's per-tick kinematics outside the governor, for the
        // whole-loop picture: the Jacobian a streamed tick hands out.
        {
            let version = openarm_description::HardwareVersion::V2;
            let mut arm =
                crate::arm_model(version, openarm_description::Side::Left).expect("arm model");
            let q = [0.1, -0.2, 0.3, 0.5, 0.1, -0.1, 0.2];
            time_us(
                "planner: jacobian at measured",
                5000,
                Box::new(move || {
                    std::hint::black_box(arm.at(&q).jacobian());
                }),
            );
        }
        // The exemption path stacks the most queries: parked at the wall,
        // one side escaping (its solo scan must prove itself) while the other
        // keeps pushing (its clip bisects). Built from the live fixture
        // rather than a synthetic pose so the hold is actually granted.
        {
            let mut g = v2_governor(true);
            // Park AT the stop (not merely in-band): govern an approach until
            // it settles, so the pushing side's clip and the escaping side's
            // exemption both actually engage.
            let mut parked = at(wrists_inward(1.05));
            for _ in 0..250 {
                let cand = at(chase(&parked.arms, &wrists_inward(1.5), 0.02));
                parked = g.govern(&parked, &cand, &parked, NO_HANDS, DT);
            }
            let mut cand = parked;
            cand.arms.left = chase(&parked.arms, &home(), 0.02).left;
            cand.arms.right = chase(&parked.arms, &wrists_inward(1.5), 0.02).right;
            time_us(
                "tick: wall, one side escaping (exemption)",
                300,
                Box::new(move || {
                    std::hint::black_box(g.govern(&parked, &cand, &parked, NO_HANDS, DT));
                }),
            );
        }
        for (label, prev, cand, measured) in regimes {
            let mut g = v2_governor(true);
            time_us(
                label,
                500,
                Box::new(move || {
                    std::hint::black_box(g.govern(&prev, &cand, &measured, NO_HANDS, DT));
                }),
            );
        }

        // The regimes above are steady states: each holds one pose pair, so the
        // step is small and the scan clears in a handful of probes. A jog does
        // neither. This walk commands a fresh pose every tick, so segments are
        // long enough to need their full probe density and the clip stage runs
        // its exemption machinery, which is where the tick's real tail lives.
        // Percentiles, not a mean: the deadline is missed by the tail, and the
        // tail is what a mean hides.
        //
        // Speeds are stated in rad/s and turned into a per-tick excursion here,
        // because that is the rate-independent quantity: the probe count a tick
        // pays is `speed * dt / MAX_PROBE_ARC_RAD`, so the same jog costs fewer
        // probes per tick on a faster loop (against a proportionally smaller
        // budget). The fastest entry is the shipped J1/J2 limit, the fastest
        // any joint is chased at; J3/J4 are clamped to their own lower limits
        // by `DofSpeed` before the scan sees them.
        for jog_rad_s in [2.0_f64, 5.0, 16.754666] {
            let chase_rad = jog_rad_s * DT;
            let mut g = v2_governor(true);
            let mut rng = Lcg(0x5eed_1729);
            // Closed grippers: the v2 open-gripper home overlaps, and the
            // penetration-recovery floor is a different regime than an approach.
            let mut q = GovState::new(home(), ArmPair::new(0.0, 0.0));
            let mut target = rng.pose();
            let mut times = Vec::with_capacity(4000);
            for tick in 0..4000 {
                if tick % 150 == 0 {
                    target = rng.pose();
                }
                let cand = GovState::new(chase(&q.arms, &target, chase_rad), q.grippers);
                let t0 = Instant::now();
                q = g.govern(&q, &cand, &q, NO_HANDS, DT);
                times.push(t0.elapsed().as_micros());
            }
            times.sort_unstable();
            let pct = |p: usize| times[(times.len() - 1) * p / 100];
            let max = *times.last().expect("4000 ticks timed");
            println!(
                "TIMING jog {jog_rad_s:.1} rad/s ({chase_rad:.4} rad/tick at {:.0} Hz): \
                 p50={} p95={} p99={} max={} us -> {:.0} Hz ceiling",
                1.0 / DT,
                pct(50),
                pct(95),
                pct(99),
                max,
                1e6 / max as f64
            );
        }
    }

    /// Captured live (MuJoCo v2): wrists swept in with the grippers wide, parked
    /// against the stop with an open finger near the torso. An operator's
    /// "retreat" to a tucked pose from here is actually a closing command for
    /// the binding pair, and the governor refused it for 19 s straight.
    fn wedged_open_gripper_at_the_torso() -> GovState {
        GovState::new(
            ArmPair::new(
                [-0.0743, -0.0735, 0.8353, 0.3973, 0.0009, -0.0074, 0.1502],
                [0.0682, 0.0724, -0.8067, 0.3973, 0.0006, 0.0076, -0.1422],
            ),
            ArmPair::new(1.0, 1.0),
        )
    }

    /// The field wedge above is not a trap: the refused command was genuinely
    /// closing (a tucked "safe" pose swings the open fingers toward the
    /// torso), and both real escapes pass. Closing the grippers alone opens the
    /// binding gap immediately, and swinging the wrists outward with the grippers
    /// closing drives the whole configuration to clear air.
    #[test]
    fn the_open_gripper_torso_wedge_is_refused_but_never_a_trap() {
        let mut g = v2_governor(true);
        let wedge = wedged_open_gripper_at_the_torso();
        let d0 = distance(&mut g, &wedge);
        assert!(
            (0.0..D_SAFE).contains(&d0),
            "setup: the wedge parks inside the band, got {d0:+.6}"
        );

        // The escape that failed in the field: a tucked pose with the wrists
        // inward. It closes the torso-finger gap, so whatever the governor
        // passes must not breach the floor.
        let tucked = ArmPair::new(
            [0.0, 0.0, 0.2, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.2, 0.5, 0.0, 0.0, 0.0],
        );
        let mut q = wedge;
        for _ in 0..100 {
            let cand = GovState::new(chase(&q.arms, &tucked, 0.02), q.grippers);
            q = g.govern(&q, &cand, &q, NO_HANDS, DT);
            let d = distance(&mut g, &q);
            assert!(
                d >= d0.min(D_STOP) - 1e-9,
                "tucked escape breached: {d:+.6}"
            );
        }

        // Real escape 1: closing the grippers alone opens the binding gap.
        let rate = g.max_gripper_rate_frac_s();
        q = wedge;
        for _ in 0..200 {
            let cand = GovState::new(
                q.arms,
                ArmPair::new(
                    rate_limited(q.grippers.left, 0.0, rate, DT),
                    rate_limited(q.grippers.right, 0.0, rate, DT),
                ),
            );
            q = g.govern(&q, &cand, &q, NO_HANDS, DT);
        }
        let grippers_closed = distance(&mut g, &q);
        assert!(
            grippers_closed > d0 + 0.005,
            "closing the grippers must open the torso gap materially: {grippers_closed:+.6}"
        );

        // Real escape 2, from where escape 1 leaves off: with the grippers closed
        // and the gap reopened, swinging the wrists outward reaches clear air.
        // (Sequencing matters: bundling the outward swing with the still-open
        // grippers is itself a closing command here, since the chase's first step
        // toward a distant pose sweeps the fingers through the torso.)
        let outward = ArmPair::new(
            [0.0, 0.0, -0.4, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.5, 0.0, 0.0, 0.0],
        );
        for _ in 0..300 {
            let cand = GovState::new(chase(&q.arms, &outward, 0.02), q.grippers);
            q = g.govern(&q, &cand, &q, NO_HANDS, DT);
        }
        let out = distance(&mut g, &q);
        assert!(
            out >= D_SAFE,
            "the outward escape must reach clear air: {out:+.6}"
        );
    }

    /// Random-walk the governor the way an operator jogs it, and check every
    /// in-band tick against the invariants the whole design rests on:
    ///
    /// 1. the governed configuration never sits below the step floor;
    /// 2. the straight line the arms actually travel to reach it never dips
    ///    more than [`SCAN_PATH_RESIDUE_M`] under that floor;
    /// 3. a commanded step is never frozen outright while its own endpoint is
    ///    admissible, so no jog can be refused with somewhere to go.
    ///
    /// Invariant 2 is what caught the separating-side exemption scanning a
    /// two-leg path the arms never travel; a scalar endpoint check cannot see
    /// that, so the path is sampled rather than assumed.
    #[test]
    fn the_realized_path_holds_the_floor_under_a_random_walk() {
        for version in [
            openarm_description::HardwareVersion::V1,
            openarm_description::HardwareVersion::V2,
        ] {
            let mut g = governor_for(version, true);
            let rate = g.max_gripper_rate_frac_s();
            let mut rng = Lcg(0x5EED_1234_ABCD_0001);
            let mut q = at(home());
            let mut target = q;
            let mut in_band = 0u32;
            let mut worst_dip = 0.0_f64;
            for k in 0..5_000 {
                // Re-aim periodically, so the walk both approaches and retreats
                // rather than parking on one wall.
                if k % 60 == 0 {
                    target = GovState::new(
                        rng.pose(),
                        ArmPair::new(rng.in_range(0.0, 1.0), rng.in_range(0.0, 1.0)),
                    );
                }
                let prev = q;
                let Some(d_prev) = g.distance_at(&concat(&prev)) else {
                    q = at(home());
                    continue;
                };
                let cand = GovState::new(
                    chase(&prev.arms, &target.arms, 0.02),
                    ArmPair::new(
                        rate_limited(prev.grippers.left, target.grippers.left, rate, DT),
                        rate_limited(prev.grippers.right, target.grippers.right, rate, DT),
                    ),
                );
                q = g.govern(&prev, &cand, &prev, NO_HANDS, DT);
                if d_prev >= D_SAFE {
                    continue;
                }
                in_band += 1;
                let floor = if d_prev >= 0.0 {
                    d_prev.min(D_STOP)
                } else {
                    d_prev - RECOVERY_LOSS_M_PER_S * DT
                };

                let d_governed = g.distance_at(&concat(&q)).unwrap_or(f64::NEG_INFINITY);
                assert!(
                    d_governed >= floor - 1e-9,
                    "{version:?} k={k}: governed below the floor, {d_governed:+.6} < {floor:+.6}"
                );

                worst_dip = worst_dip.max(floor - segment_min(&mut g, &prev, &q, 16));

                let moved: f64 = (0..GOV_DOF)
                    .map(|i| (concat(&q)[i] - concat(&prev)[i]).abs())
                    .fold(0.0, f64::max);
                let commanded: f64 = (0..GOV_DOF)
                    .map(|i| (concat(&cand)[i] - concat(&prev)[i]).abs())
                    .fold(0.0, f64::max);
                let d_cand = g.distance_at(&concat(&cand)).unwrap_or(f64::NEG_INFINITY);
                assert!(
                    moved > 0.0 || commanded <= 1e-12 || d_cand < floor,
                    "{version:?} k={k}: frozen with a way out, d_cand={d_cand:+.6} floor={floor:+.6}"
                );
            }
            assert!(
                in_band > 1_000,
                "{version:?}: the walk barely entered the band ({in_band} ticks); it proves nothing"
            );
            assert!(
                worst_dip <= SCAN_PATH_RESIDUE_M,
                "{version:?}: the realized path dipped {worst_dip:.6} m under the floor, over the {SCAN_PATH_RESIDUE_M:.6} m scan residue"
            );
        }
    }

    /// An operator who opens the grippers with the arms down and tucked drives a
    /// fully-splayed finger into the torso proxy. What must hold is that the
    /// governor never makes that overlap a trap: two independent escapes work.
    ///
    /// The overlap is *constructed* rather than read off a fixed pose. How deep
    /// any given configuration sits inside the proxy depends on how tightly the
    /// collision meshes are fitted, which is a property of the model and not of
    /// this invariant, so the test bisects to a definite overlap and proves the
    /// escapes from there. Pinning a pose instead makes this test silently stop
    /// exercising its own escapes the day the fit changes.
    #[test]
    fn a_wide_gripper_overlapping_the_torso_is_never_a_trap() {
        let mut g = v2_governor(true);
        let wide = GovState::new(home(), ArmPair::new(1.0, 1.0));
        // Tucking the shoulders swings the splayed fingers into the torso; stop
        // short of the depth where a whole link, not a finger, becomes nearest.
        let tucked = GovState::new(
            {
                let mut arms = home();
                arms.left[1] = 0.1;
                arms.right[1] = 0.1;
                arms
            },
            ArmPair::new(1.0, 1.0),
        );
        let overlapped = config_at_distance(&mut g, &wide, &tucked, -0.003);
        let d0 = distance(&mut g, &overlapped);
        assert!(
            d0 < 0.0,
            "setup: the constructed pose should overlap, got {d0:+.6}"
        );
        let pair = g.proximity(&overlapped).expect("a nearest pair");
        let names = format!("{} <-> {}", pair.link_a, pair.link_b);
        assert!(
            names.contains("body") && names.contains("ee_link"),
            "setup: the overlap should be a finger against the torso, got {names}"
        );

        // Closing the grippers is the direct way out, and it must not be refused.
        let rate = g.max_gripper_rate_frac_s();
        let mut q = overlapped;
        for _ in 0..200 {
            let cand = GovState::new(
                q.arms,
                ArmPair::new(
                    rate_limited(q.grippers.left, 0.0, rate, DT),
                    rate_limited(q.grippers.right, 0.0, rate, DT),
                ),
            );
            q = g.govern(&q, &cand, &q, NO_HANDS, DT);
        }
        assert!(
            distance(&mut g, &q) > 0.0,
            "closing the grippers must clear the torso overlap"
        );

        // So is swinging the arms out with the grippers left wide.
        let mut q = overlapped;
        let mut out = home();
        out.left[1] = -0.6;
        out.right[1] = 0.6;
        for _ in 0..200 {
            let cand = GovState::new(chase(&q.arms, &out, 0.02), q.grippers);
            q = g.govern(&q, &cand, &q, NO_HANDS, DT);
        }
        assert!(
            distance(&mut g, &q) > 0.0,
            "swinging the arms clear must also resolve the overlap"
        );
    }

    #[test]
    fn outside_band_large_jump_is_floored() {
        let mut g = governor(true);
        // Start clear (outside the band) and command a single oversized step that
        // would vault straight past the stop floor in one tick. The outside-band
        // fast path must still run the backstop and retract to the floor.
        let start = at(home());
        assert!(
            distance(&mut g, &start) >= D_SAFE,
            "start should sit outside the band"
        );
        let deep = at(wrists_inward(1.5));
        assert!(
            distance(&mut g, &deep) < D_STOP,
            "target should be past the stop floor"
        );
        let governed = g.govern(&start, &deep, &start, NO_HANDS, DT);
        assert_ne!(governed, deep, "oversized step passed unfloored");
        assert!(
            distance(&mut g, &governed) >= D_STOP,
            "large jump breached the stop floor"
        );
    }

    #[test]
    fn an_invalid_tick_period_holds_prev() {
        let mut g = governor(true);
        let prev = at(home());
        let cand = at(wrists_inward(0.4));
        for dt in [0.0, -0.01, f64::NAN, f64::INFINITY] {
            assert_eq!(
                g.govern(&prev, &cand, &prev, NO_HANDS, dt),
                prev,
                "dt={dt} must hold"
            );
            assert_eq!(g.guard(), Guard::Stopped);
        }
        // A valid tick immediately after is unaffected.
        assert_eq!(g.govern(&prev, &cand, &prev, NO_HANDS, DT), cand);
    }

    #[test]
    fn non_finite_candidate_holds_prev() {
        let mut g = governor(true);
        let prev = at(home());
        let mut bad = at(wrists_inward(0.2));
        bad.arms.left[0] = f64::NAN;
        // Enabled: the up-front guard holds prev rather than steering on NaN.
        assert_eq!(g.govern(&prev, &bad, &prev, NO_HANDS, DT), prev);
        assert_eq!(g.guard(), Guard::Stopped, "a fault hold reads stopped");
        // A non-finite OPENING is the same class of upstream glitch.
        let mut bad_opening = at(wrists_inward(0.2));
        bad_opening.grippers.left = f64::NAN;
        assert_eq!(g.govern(&prev, &bad_opening, &prev, NO_HANDS, DT), prev);
        // Disabled fast path: still never passes a non-finite candidate through.
        g.set_enabled(false);
        assert_eq!(g.govern(&prev, &bad, &prev, NO_HANDS, DT), prev);
    }

    #[test]
    fn set_enabled_toggles_barrier() {
        let mut g = governor(true);
        // An in-band closing step is throttled when enabled, passed when disabled.
        let near = at(wrists_inward(1.0));
        let closer = at(wrists_inward(1.3));
        assert!(
            distance(&mut g, &near) < D_SAFE,
            "near pose should be in the band"
        );
        assert_ne!(g.govern(&near, &closer, &near, NO_HANDS, DT), closer);
        assert_ne!(
            g.guard(),
            Guard::Clear,
            "a limited step must read restricted"
        );
        g.set_enabled(false);
        assert_eq!(g.govern(&near, &closer, &near, NO_HANDS, DT), closer);
        assert_eq!(g.guard(), Guard::Clear, "disabling resets the readout");
    }

    /// Blends the segment from `lo_pose` to `hi_pose` and returns the
    /// configuration where the real clearance first falls to `target`: a
    /// measured pose at a chosen clearance, for the scenarios whose setup needs
    /// one.
    ///
    /// Searching is a scan, and only the refinement is a bisection. Clearance
    /// along a joint-space segment is not monotone (a finger sweeps through the
    /// other arm and back out, so both endpoints can sit above a target an
    /// interior stretch dips below), and a bisection handed the whole segment
    /// is not a search at all: it only converges on a crossing when the
    /// midpoints it happens to pick straddle one, and it steps over anything
    /// narrower. The scan states its resolution instead of leaving it to that
    /// accident, and the bisection then runs inside a bracket it was given
    /// rather than one it assumed.
    const CROSSING_SCAN_STEPS: usize = 256;

    fn config_at_distance(
        g: &mut Governor,
        lo_pose: &GovState,
        hi_pose: &GovState,
        target: f64,
    ) -> GovState {
        let lo = concat(lo_pose);
        let hi = concat(hi_pose);
        let point_at = |t: f64| split(&std::array::from_fn(|i| lo[i] + t * (hi[i] - lo[i])));
        // Walk the segment for the first adjacent pair straddling the target.
        // Anything thinner than one scan cell is invisible to this, which is
        // what the resolution above buys and the panic below reports.
        let cell = 1.0 / CROSSING_SCAN_STEPS as f64;
        let mut left = (0.0_f64, distance(g, &point_at(0.0)));
        let bracket = (1..=CROSSING_SCAN_STEPS).find_map(|k| {
            let right = (k as f64 * cell, distance(g, &point_at(k as f64 * cell)));
            let straddles = left.1 >= target && right.1 < target;
            let found = straddles.then_some((left.0, right.0));
            left = right;
            found
        });
        let Some((mut a, mut b)) = bracket else {
            panic!(
                "config_at_distance found no crossing of {target:+.6} at {CROSSING_SCAN_STEPS} \
                 steps: endpoints are {:+.6} and {:+.6}",
                distance(g, lo_pose),
                distance(g, hi_pose)
            );
        };
        for _ in 0..50 {
            let m = 0.5 * (a + b);
            if distance(g, &point_at(m)) >= target {
                a = m
            } else {
                b = m
            }
        }
        point_at(a)
    }

    /// The fixture builder's own guard, both ways round. Every collision
    /// scenario's setup rests on it, so it has to find an interior crossing its
    /// endpoints do not reveal, and it has to fail loudly rather than hand back
    /// an endpoint at some unrelated clearance when there is none to find.
    #[test]
    fn config_at_distance_finds_a_crossing_neither_endpoint_reveals() {
        let mut g = v2_governor(true);
        let arms = finger_into_other_arm();
        let (shut, wide) = (
            GovState::new(arms, ArmPair::new(0.0, 0.0)),
            GovState::new(arms, ArmPair::new(1.0, 0.0)),
        );
        // The finger sweeps through the other arm and back out, so the dip is
        // strictly interior: a bisection over the whole segment can miss it.
        let target = 0.5 * MONITOR_TRIP_FRACTION * D_STOP;
        assert!(
            distance(&mut g, &shut) > target && distance(&mut g, &wide) > target,
            "setup: both ends must sit above the target for this to mean anything"
        );
        let found = config_at_distance(&mut g, &shut, &wide, target);
        assert!(
            (distance(&mut g, &found) - target).abs() < 1e-6,
            "the scan should land on the target, got {:+.6}",
            distance(&mut g, &found)
        );
    }

    #[test]
    #[should_panic(expected = "found no crossing")]
    fn config_at_distance_refuses_a_segment_that_never_crosses() {
        let mut g = v2_governor(true);
        let clear = at(home());
        let also_clear = at(wrists_inward(0.1));
        assert!(
            distance(&mut g, &clear) > 0.0 && distance(&mut g, &also_clear) > 0.0,
            "setup: both poses are clear, so no point between them overlaps"
        );
        config_at_distance(&mut g, &clear, &also_clear, -0.05);
    }

    #[test]
    fn monitor_always_allows_separation_when_measured_breaches() {
        let mut g = governor(true);
        // The MEASURED arms are past the monitor floor (a real near-collision). A
        // command that opens the gap must ALWAYS pass: the operator can never be
        // trapped inside a near-collision, even while the monitor is tripped.
        let measured = at(wrists_inward(2.0));
        assert!(
            distance(&mut g, &measured) < MONITOR_TRIP_FRACTION * D_STOP,
            "measured pose must breach the monitor floor"
        );
        let prev = measured;
        let retreat = at(wrists_inward(1.4)); // a more-open configuration
        assert!(
            distance(&mut g, &retreat) > distance(&mut g, &measured),
            "retreat opens the gap"
        );
        let governed = g.govern(&prev, &retreat, &measured, NO_HANDS, DT);
        assert_ne!(
            governed, prev,
            "separation was blocked while the monitor was tripped"
        );
        assert!(
            distance(&mut g, &governed) > distance(&mut g, &measured),
            "the governed step did not open the gap"
        );
    }

    /// The left shoulder swung in is a deep self-collision; interpolating home toward
    /// it gives configurations at any chosen clearance for the monitor tests.
    fn deep_collision() -> GovState {
        let mut p = at(home());
        p.arms.left[1] = 1.4;
        p
    }

    #[test]
    fn barrier_frees_a_retreating_arm_while_the_other_pushes() {
        // The field scenario at the BARRIER (monitor untripped): parked at the
        // wall, one operator keeps pushing an arm in (its solo motion closes)
        // while the other retreats (its solo motion opens). The shared-segment
        // clip once retracted both to the same parameter, freezing the escape;
        // the separating-side hold must let the retreating arm move nearly its
        // full commanded step while the pushing arm is held at the floor.
        let mut g = governor(true);
        let q = drive_into_band(&mut g);
        let d_wall = distance(&mut g, &q);
        assert!(
            (D_STOP..D_SAFE).contains(&d_wall),
            "setup: parked in-band near the wall, got {d_wall:+.5}"
        );
        // Search a push/retreat pair that is genuinely mixed: solo-left closes
        // below the floor while solo-right opens.
        let mut found = None;
        for (push, pull) in [
            (0.05, 0.02),
            (0.1, 0.02),
            (0.15, 0.03),
            (0.2, 0.02),
            (0.2, 0.05),
        ] {
            let cl = chase(&q.arms, &wrists_inward(1.6), push).left;
            let cr = chase(&q.arms, &home(), pull).right;
            let solo_left = distance(
                &mut g,
                &GovState::new(ArmPair::new(cl, q.arms.right), q.grippers),
            );
            let solo_right = distance(
                &mut g,
                &GovState::new(ArmPair::new(q.arms.left, cr), q.grippers),
            );
            if solo_left < d_wall && solo_right > d_wall {
                found = Some((GovState::new(ArmPair::new(cl, cr), q.grippers), solo_right));
                break;
            }
        }
        let (cand, solo_right) = found.expect("setup: some push/retreat pair is mixed");

        let governed = g.govern(&q, &cand, &q, NO_HANDS, DT);
        // The retreating (right) arm must actually move; the governed config
        // stays at or above the stop floor.
        assert_ne!(
            governed.arms.right, q.arms.right,
            "the retreating arm was frozen"
        );
        assert!(
            distance(&mut g, &governed) >= D_STOP - 1e-6,
            "the governed step breached the stop floor"
        );
        // It should recover most of the way to the solo-retreat clearance, not a
        // token sliver (the frozen-arm bug clipped it to a few percent).
        let opened = distance(
            &mut g,
            &GovState::new(ArmPair::new(q.arms.left, governed.arms.right), q.grippers),
        );
        assert!(
            opened >= d_wall + 0.5 * (solo_right - d_wall),
            "retreat barely moved: opened to {opened:+.5} of solo {solo_right:+.5} from wall {d_wall:+.5}"
        );
    }

    #[test]
    fn monitor_frees_a_retreating_arm_while_the_other_pushes() {
        // The field scenario: with the monitor tripped, one operator keeps
        // pushing arm A into the breach while the other retreats arm B. The
        // joint candidate closes (A dominates), but B's own motion opens the
        // real gap, so the per-side gate must hold A and let B escape; a
        // whole-candidate hold would freeze B for as long as A pushes.
        let mut g = governor(true);
        // A shallow breach (positive clearance under the trip floor) keeps the
        // distance field smooth, and this asymmetric converging pose binds a
        // CROSS-ARM pair, so both arms' motion genuinely moves the gap (a
        // torso-bound pair would make the retreating arm irrelevant).
        let deep = at(ArmPair::new(
            [0.0, 0.0, 1.15, 0.4, 0.1, 0.0, 0.2],
            [0.0, 0.0, -1.25, 0.4, -0.1, 0.1, 0.0],
        ));
        assert!(
            distance(&mut g, &deep) < 0.5 * MONITOR_TRIP_FRACTION * D_STOP,
            "setup: the deep pose must pass the breach target"
        );
        let measured = config_at_distance(
            &mut g,
            &at(home()),
            &deep,
            0.5 * MONITOR_TRIP_FRACTION * D_STOP,
        );
        let d_measured = distance(&mut g, &measured);
        assert!(
            d_measured < MONITOR_TRIP_FRACTION * D_STOP,
            "setup: measured pose must breach the monitor floor"
        );
        let prev = measured;
        // Find a push/pull step pair where the joint candidate closes (the push
        // dominates) while the retreating arm alone opens: the mixed case the
        // per-side gate exists for.
        let mut found = None;
        for (push_step, pull_step) in [(0.05, 0.01), (0.1, 0.01), (0.15, 0.02), (0.2, 0.02)] {
            let push = chase(&measured.arms, &deep.arms, push_step);
            let pull = chase(&measured.arms, &home(), pull_step);
            let cand = GovState::new(ArmPair::new(push.left, pull.right), measured.grippers);
            let solo_right = GovState::new(
                ArmPair::new(prev.arms.left, cand.arms.right),
                measured.grippers,
            );
            let solo_left = GovState::new(
                ArmPair::new(cand.arms.left, prev.arms.right),
                measured.grippers,
            );
            if distance(&mut g, &cand) <= d_measured
                && distance(&mut g, &solo_right) > d_measured
                && distance(&mut g, &solo_left) <= d_measured
            {
                found = Some(cand);
                break;
            }
        }
        let cand = found.expect("setup: some push/pull pair produces the mixed case");

        let governed = g.govern(&prev, &cand, &measured, NO_HANDS, DT);
        assert_eq!(
            governed.arms.left, prev.arms.left,
            "the pushing arm must be held"
        );
        assert_ne!(
            governed.arms.right, prev.arms.right,
            "the retreating arm was frozen"
        );
        assert!(
            distance(&mut g, &governed) > d_measured,
            "the freed motion must open the real gap"
        );
    }

    #[test]
    fn monitor_holds_a_closing_command_when_measured_breaches() {
        let mut g = governor(true);
        // Same breach, but the command would close the gap further: that is held.
        let deep = deep_collision();
        assert!(
            distance(&mut g, &deep) < 0.0,
            "deep pose must be in penetration"
        );
        let measured = config_at_distance(
            &mut g,
            &at(home()),
            &deep,
            0.5 * MONITOR_TRIP_FRACTION * D_STOP,
        );
        assert!(
            distance(&mut g, &measured) < MONITOR_TRIP_FRACTION * D_STOP,
            "measured must breach the floor"
        );
        let prev = measured;
        // `deep` is more closed than the measured pose: a closing command, held at prev.
        assert!(
            distance(&mut g, &deep) < distance(&mut g, &measured),
            "deep is a closing command"
        );
        assert_eq!(
            g.govern(&prev, &deep, &measured, NO_HANDS, DT),
            prev,
            "a closing command was not held on a measured breach"
        );
    }

    #[test]
    fn monitor_judges_closing_in_the_commanded_space_under_a_tracking_offset() {
        let mut g = governor(true);
        // A systematic tracking offset: the measured arms breach the monitor
        // floor while the commanded setpoints still read ~15 mm clear. Judged
        // against the measured clearance (across spaces), every
        // velocity-limited closing candidate would read as "opening" (~15 mm vs
        // ~2 mm) and pass; judged in the commanded space, a candidate that
        // closes on the held setpoint is held.
        let deep = deep_collision();
        let measured = config_at_distance(
            &mut g,
            &at(home()),
            &deep,
            0.5 * MONITOR_TRIP_FRACTION * D_STOP,
        );
        let prev = config_at_distance(&mut g, &at(home()), &deep, 0.015);
        let cand = at(chase(&prev.arms, &deep.arms, 0.02));
        assert!(
            distance(&mut g, &cand) < distance(&mut g, &prev),
            "setup: the candidate closes on the held setpoint"
        );
        assert_eq!(
            g.govern(&prev, &cand, &measured, NO_HANDS, DT),
            prev,
            "a closing command passed the tripped monitor under a tracking offset"
        );
        // The escape is still free under the same offset: opening on the held
        // setpoint passes even though its clearance is far above the measured.
        let retreat = at(chase(&prev.arms, &home(), 0.02));
        assert!(
            distance(&mut g, &retreat) > distance(&mut g, &prev),
            "setup: the retreat opens on the held setpoint"
        );
        assert_ne!(
            g.govern(&prev, &retreat, &measured, NO_HANDS, DT),
            prev,
            "a separating command was frozen by the cross-space baseline"
        );
    }

    #[test]
    fn monitor_inert_under_good_tracking() {
        let mut g = governor(true);
        // Measured == commanded (perfect tracking), far apart: the monitor never
        // trips and the commanded step passes as it would without it.
        let prev = at(home());
        let cand = at(wrists_inward(0.2));
        assert!(
            distance(&mut g, &prev) >= D_SAFE,
            "precondition: home sits outside the band"
        );
        assert_eq!(
            g.govern(&prev, &cand, &prev, NO_HANDS, DT),
            cand,
            "monitor tripped under good tracking"
        );
    }

    #[test]
    fn monitor_hysteresis_holds_a_closing_command_until_recovered_past_d_stop() {
        let mut g = governor(true);
        // `deep` is a closing command vs every measured pose below, so the monitor's
        // hold, not a separation pass, is what is under test.
        let deep = deep_collision();
        let prev = at(home());
        let breaching = config_at_distance(
            &mut g,
            &at(home()),
            &deep,
            0.5 * MONITOR_TRIP_FRACTION * D_STOP,
        );
        assert!(distance(&mut g, &breaching) < MONITOR_TRIP_FRACTION * D_STOP);

        // A measured pose whose real clearance sits inside the hysteresis band
        // [trip floor, d_stop): below the commanded stop but above the trip floor.
        let in_band = config_at_distance(
            &mut g,
            &at(home()),
            &deep,
            0.5 * (MONITOR_TRIP_FRACTION * D_STOP + D_STOP),
        );
        assert!(
            (MONITOR_TRIP_FRACTION * D_STOP..D_STOP).contains(&distance(&mut g, &in_band)),
            "setup: in_band not in the hysteresis band"
        );

        // Breach trips the latch: the closing command is held at prev.
        assert_eq!(
            g.govern(&prev, &deep, &breaching, NO_HANDS, DT),
            prev,
            "closing command not held on a breach"
        );
        // In-band measurement (above the trip floor, below d_stop): still held (hysteresis).
        assert_eq!(
            g.govern(&prev, &deep, &in_band, NO_HANDS, DT),
            prev,
            "released before recovering past d_stop"
        );
        // Recovered past d_stop: the latch releases, so the command is governed
        // normally (clipped toward the floor), not force-held at prev.
        assert_ne!(
            g.govern(&prev, &deep, &at(home()), NO_HANDS, DT),
            prev,
            "did not release after recovery"
        );
        // The release must actually clear the LATCH, not just pass this call: a
        // later in-band measurement (inside the hysteresis band) must be judged
        // from the untripped threshold and not re-hold. A latch stuck set would
        // force-hold here forever.
        assert_ne!(
            g.govern(&prev, &deep, &in_band, NO_HANDS, DT),
            prev,
            "the latch did not clear on recovery"
        );
    }

    #[test]
    fn monitor_inert_when_disabled() {
        let mut g = governor(false);
        // Tied to the operator toggle: disabled is passthrough even though the
        // measured arms breach the floor.
        let cand = at(wrists_inward(0.2));
        let breaching = at(wrists_inward(2.0));
        assert_eq!(g.govern(&at(home()), &cand, &breaching, NO_HANDS, DT), cand);
    }

    #[test]
    fn monitor_defers_when_the_measured_query_fails_so_separation_is_never_blocked() {
        let mut g = governor(true);
        // A non-finite measured state makes the distance query fail. The monitor must
        // not hold (which would block escape); it defers to the main governing, so the
        // command is never force-held at prev.
        let prev = at(home());
        let cand = at(wrists_inward(0.2));
        let mut measured = at(home());
        measured.arms.left[0] = f64::NAN;
        assert_eq!(
            g.govern(&prev, &cand, &measured, NO_HANDS, DT),
            cand,
            "monitor blocked a command on a failed measured query"
        );
    }

    #[test]
    fn monitor_does_not_trip_from_clear_on_an_in_band_measurement() {
        let mut g = governor(true);
        // Hysteresis asymmetry: from an untripped state the trip threshold is the trip
        // floor, not d_stop, so a measurement in [trip floor, d_stop) must NOT trip. A
        // closing command is then governed normally, not force-held at prev.
        let deep = deep_collision();
        let in_band = config_at_distance(
            &mut g,
            &at(home()),
            &deep,
            0.5 * (MONITOR_TRIP_FRACTION * D_STOP + D_STOP),
        );
        assert!(
            (MONITOR_TRIP_FRACTION * D_STOP..D_STOP).contains(&distance(&mut g, &in_band)),
            "setup: in_band not in the hysteresis band"
        );
        assert_ne!(
            g.govern(&at(home()), &deep, &in_band, NO_HANDS, DT),
            at(home()),
            "an in-band measurement tripped from a clear state"
        );
    }

    #[test]
    fn a_penetrated_arm_can_still_drive_out_when_the_way_out_dips_deeper() {
        // Field repro (MuJoCo v2): with the wrists swept in, the arms sat
        // interpenetrating and every escape command was refused, because the
        // way out sweeps DEEPER before it separates and the floor scan rejected
        // the whole segment on every tick. The operator's only recourse was to
        // switch the governor off, which is the worst moment to have no guard.
        //
        // `wrists_inward` has exactly that shape: t = 1.20 is a local clearance
        // maximum inside a penetrating pocket, so retreating toward home makes
        // the clearance worse before it makes it better.
        let mut g = v2_governor(true);
        let trapped = at(wrists_inward(1.20));
        let d0 = distance(&mut g, &trapped);
        assert!(d0 < 0.0, "setup: the repro pose must overlap, got {d0:+.5}");
        let backed_off = distance(&mut g, &at(wrists_inward(1.15)));
        assert!(
            backed_off < d0,
            "setup: retreating must dip deeper first ({backed_off:+.5} vs {d0:+.5})"
        );

        // One velocity-limited tick toward home must not be frozen.
        let one = at(chase(&trapped.arms, &home(), 0.02));
        let governed = g.govern(&trapped, &one, &trapped, NO_HANDS, DT);
        assert_ne!(
            governed, trapped,
            "escape frozen: a penetrated arm must never be trapped by the governor"
        );

        // And driving out must actually work: the operator reaches clear air.
        let mut state = trapped;
        let budget = RECOVERY_LOSS_M_PER_S * DT + 1e-6;
        for _ in 0..400 {
            let before = distance(&mut g, &state);
            let cand = GovState::new(chase(&state.arms, &home(), 0.02), state.grippers);
            state = g.govern(&state, &cand, &state, NO_HANDS, DT);
            let after = distance(&mut g, &state);
            assert!(
                after >= before - budget,
                "recovery gave up {:.5} m in one tick, over the {budget:.5} m budget",
                before - after
            );
        }
        let out = distance(&mut g, &state);
        assert!(
            out > 0.0,
            "the operator could not drive out of the collision: d={out:+.5}"
        );
    }

    #[test]
    fn concat_split_round_trip() {
        let state = GovState::new(
            ArmPair::new(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            ),
            ArmPair::new(0.25, 0.75),
        );
        assert_eq!(split(&concat(&state)), state);
        let flat: [f64; GOV_DOF] = std::array::from_fn(|i| i as f64);
        assert_eq!(concat(&split(&flat)), flat);
    }

    /// Minimum clearance sampled along the straight segment `prev`->`cand`.
    fn segment_min(g: &mut Governor, prev: &GovState, cand: &GovState, n: usize) -> f64 {
        let p = concat(prev);
        let c = concat(cand);
        let mut m = f64::INFINITY;
        for i in 0..=n {
            let t = i as f64 / n as f64;
            m = m.min(distance(
                g,
                &split(&std::array::from_fn(|j| p[j] + t * (c[j] - p[j]))),
            ));
        }
        m
    }

    #[test]
    fn outside_band_segment_is_scanned_even_when_both_ends_are_clear() {
        let mut g = governor(true);
        // Bimanual distance is not monotone along a joint-space segment, and the
        // governor must not trust the endpoints even when both are clear of the band.
        // Sweeping the left shoulder (j1) swings the left arm around the right one:
        // from home the clearance dives into deep penetration near j1=1.4 and
        // resurfaces by j1~3.15. So home and a far shoulder angle are both clear of
        // d_safe, yet the straight segment between them crosses well below the stop.
        let prev = at(home());
        let cand = {
            let mut p = at(home());
            p.arms.left[1] = 3.2;
            p
        };
        assert!(
            distance(&mut g, &prev) >= D_SAFE,
            "home end is clear of the band"
        );
        assert!(
            distance(&mut g, &cand) >= D_SAFE,
            "far-shoulder end is clear of the band"
        );
        assert!(
            segment_min(&mut g, &prev, &cand, 128) < D_STOP,
            "the segment dips below the stop"
        );

        // Trusting the (clear) endpoints would pass `cand` through; the segment scan
        // must clip it to a setpoint that is itself clear of the stop.
        let governed = g.govern(&prev, &cand, &prev, NO_HANDS, DT);
        assert_ne!(
            governed, cand,
            "a clear-ended segment with a sub-stop interior was passed unclipped"
        );
        assert!(
            distance(&mut g, &governed) >= D_STOP - 1e-6,
            "the clipped setpoint is below the stop"
        );
    }

    // --- Gripper scenarios (v2, whose fingers travel the farthest) -----------
    //
    // The grippers are ordinary governed DOF, so every arm guarantee above
    // already applies to them; these pin the finger-specific geometry paths:
    // opening into the other arm, bilateral grippers sharing one clearance,
    // closing as separation, and the monitor judging measured fingers.

    /// A wrists-inward v2 pose whose left finger sweeps toward the right palm as
    /// the gripper opens: with the grippers closed the palms are ~23 mm clear, fully open
    /// the finger penetrates. The gripper scenarios open into it.
    fn finger_into_other_arm() -> ArmPair<JointVec> {
        ArmPair::new(
            [0.0, 0.0, 0.6, 0.4, -0.4, 0.4, -0.4],
            [0.0, 0.0, -0.6, 0.4, 0.0, 0.0, 0.0],
        )
    }

    /// Drive `start` toward `target` through the governor tick by tick with
    /// velocity-limited candidates (arms and grippers alike), asserting the
    /// realized clearance never crosses the stop floor on any tick; measured
    /// tracks commanded (perfect tracking). Returns the settled state.
    fn drive(g: &mut Governor, start: GovState, target: &GovState, ticks: usize) -> GovState {
        let gripper_step = g.max_gripper_rate_frac_s() * DT;
        let chase_frac = |from: f64, to: f64| from + (to - from).clamp(-gripper_step, gripper_step);
        let mut s = start;
        for _ in 0..ticks {
            let cand = GovState::new(
                chase(&s.arms, &target.arms, 0.02),
                ArmPair::new(
                    chase_frac(s.grippers.left, target.grippers.left),
                    chase_frac(s.grippers.right, target.grippers.right),
                ),
            );
            s = g.govern(&s, &cand, &s, NO_HANDS, DT);
            let d = distance(g, &s);
            assert!(d >= D_STOP - 1e-9, "floor breached mid-drive: d={d:+.6}");
        }
        s
    }

    #[test]
    fn gripper_opening_into_the_other_arm_settles_at_the_stop() {
        let mut g = v2_governor(true);
        let arms = finger_into_other_arm();
        let start = GovState::new(arms, ArmPair::new(0.0, 0.0));
        assert!(
            distance(&mut g, &start) >= D_SAFE,
            "setup: closed grippers start clear of the band"
        );
        // Command the left gripper fully open; the barrier admits a useful partial
        // opening and parks it at the floor, holding the floor on every tick.
        let target = GovState::new(arms, ArmPair::new(1.0, 0.0));
        let settled = drive(&mut g, start, &target, 300);
        assert!(
            (1e-4..1.0 - 1e-4).contains(&settled.grippers.left),
            "opening into the other arm should settle to a safe partial, got {}",
            settled.grippers.left
        );
        assert_eq!(
            settled.grippers.right, 0.0,
            "uncommanded right gripper opened"
        );
        // Settled NEAR the stop, not stalled far above it.
        let d = distance(&mut g, &settled);
        assert!(
            d < D_STOP + 4e-3,
            "did not settle near the stop distance: d={d:+.5}"
        );
        assert_ne!(
            g.guard(),
            Guard::Clear,
            "parked at the floor with open still commanded must read restricted"
        );
    }

    #[test]
    fn gripper_closing_recovers_clearance_under_the_same_barrier() {
        let mut g = v2_governor(true);
        let arms = finger_into_other_arm();
        // Park the left opening at the floor, then command fully closed. Closing
        // is NOT exempt from governing: a gripper has two fingers, and at an angled
        // pose retracting the near finger advances the far one, so a closing
        // sub-motion can itself reduce clearance and gets budgeted like any
        // other. The invariants are the floor holding on every tick (asserted
        // inside `drive`), steady progress to fully closed, and the clearance
        // recovering past the parked value.
        let parked = drive(
            &mut g,
            GovState::new(arms, ArmPair::new(0.0, 0.0)),
            &GovState::new(arms, ArmPair::new(1.0, 0.0)),
            300,
        );
        let d_parked = distance(&mut g, &parked);
        let closed = drive(
            &mut g,
            parked,
            &GovState::new(arms, ArmPair::new(0.0, 0.0)),
            300,
        );
        assert!(
            closed.grippers.left < 1e-6,
            "the gripper did not close: settled at {}",
            closed.grippers.left
        );
        assert!(
            distance(&mut g, &closed) > d_parked,
            "closing the gripper should recover clearance"
        );
    }

    #[test]
    fn bilateral_grippers_share_one_clearance_budget() {
        let mut g = v2_governor(true);
        // Two grippers facing each other across the centerline: with the grippers
        // closed the pose is clear of the band, with both fully open the fingers
        // interpenetrate, so opening EITHER gripper closes the same gap.
        let pose = wrists_inward(0.55);
        let start = GovState::new(pose, ArmPair::new(0.0, 0.0));
        assert!(
            distance(&mut g, &start) >= D_SAFE,
            "setup: closed grippers start clear of the band"
        );
        assert!(
            distance(&mut g, &GovState::new(pose, ArmPair::new(1.0, 1.0))) < 0.0,
            "setup: both grippers open must interpenetrate"
        );
        // Command BOTH grippers fully open at once. One shared barrier budgets the
        // joint motion, so the combined opening still holds the floor on every
        // tick (asserted inside drive); two independent barriers would each
        // spend the full budget and jointly breach it.
        let settled = drive(
            &mut g,
            start,
            &GovState::new(pose, ArmPair::new(1.0, 1.0)),
            300,
        );
        let d = distance(&mut g, &settled);
        assert!(
            (D_STOP - 1e-9..D_STOP + 4e-3).contains(&d),
            "bilateral grippers should park the shared clearance at the stop, got {d:+.5}"
        );
        // Both grippers made real progress: the budget was shared, not starved onto
        // one side.
        assert!(
            settled.grippers.left > 1e-3 && settled.grippers.right > 1e-3,
            "both grippers should open partially, got {:?}",
            settled.grippers
        );
    }

    #[test]
    fn mixed_arm_push_and_opening_same_tick_holds_the_floor() {
        let mut g = v2_governor(true);
        let arms = finger_into_other_arm();
        let start = GovState::new(arms, ArmPair::new(0.0, 0.0));
        // Drive the left arm further inward AND its gripper open in the same ticks:
        // the shared barrier budgets the combined closing motion, and the floor
        // holds on every tick (asserted inside drive).
        let mut pushed = arms;
        pushed.left[2] += 0.3;
        let target = GovState::new(pushed, ArmPair::new(1.0, 0.0));
        let settled = drive(&mut g, start, &target, 300);
        let d = distance(&mut g, &settled);
        assert!(
            d < D_STOP + 4e-3,
            "the mixed push should converge near the stop, got {d:+.5}"
        );
    }

    #[test]
    fn opening_below_the_floor_recovers_by_closing_never_deepens() {
        let mut g = v2_governor(true);
        let arms = finger_into_other_arm();
        // Force a sub-floor state (finger opened into the other arm, as an
        // upstream fault or disabled-period motion would leave it). The stuck
        // opening must sit on the near side of the finger's sweep minimum
        // (~0.55 at this pose) so closing is the motion that retracts it;
        // asserted below so a geometry change fails as setup, not as a
        // governor regression.
        let stuck = GovState::new(arms, ArmPair::new(0.5, 0.0));
        let d_stuck = distance(&mut g, &stuck);
        assert!(
            d_stuck < D_STOP,
            "setup: the forced state breaches the floor"
        );
        let gripper_step = g.max_gripper_rate_frac_s() * DT;
        let deeper = GovState::new(arms, ArmPair::new(0.5 + gripper_step, 0.0));
        let closing = GovState::new(arms, ArmPair::new(0.5 - gripper_step, 0.0));
        assert!(
            distance(&mut g, &deeper) < d_stuck && distance(&mut g, &closing) > d_stuck,
            "setup: at this pose opening must deepen and closing must recover"
        );
        // Opening further is fully frozen (the floor is the current clearance).
        let held = g.govern(&stuck, &deeper, &stuck, NO_HANDS, DT);
        assert!(
            distance(&mut g, &held) >= d_stuck - 1e-9,
            "an opening below the floor deepened the breach"
        );
        // Closing recovers: the escape passes and clearance increases.
        let governed = g.govern(&stuck, &closing, &stuck, NO_HANDS, DT);
        assert_eq!(governed, closing, "the closing escape was throttled");
        assert!(
            distance(&mut g, &governed) > d_stuck,
            "closing did not recover clearance"
        );
    }

    #[test]
    fn a_streamed_hand_is_capped_and_the_toggle_does_not_uncap_it() {
        // A basis-carrying side is capped through the whole pipeline, enabled
        // or not, and the guard does not read a comfort cap as a collision
        // event. Joint 0 moves the hand 1 m/rad, so a 0.1 rad step over DT is
        // 10 m/s against the 0.5 m/s test cap: the governed step must be the
        // cap's fraction of the command, on the streamed side only.
        for enabled in [true, false] {
            let mut g = governor(enabled);
            let mut jac = Jacobian::zeros();
            jac[(0, 0)] = 1.0;
            let hands = ArmPair::new(Some(jac), None);
            let prev = at(home());
            let mut cand = prev;
            cand.arms.left[0] += 0.1;
            cand.arms.right[0] += 0.1;
            let governed = g.govern(&prev, &cand, &prev, &hands, DT);
            let expected = prev.arms.left[0] + TEST_EE_CAP_M_S * DT;
            assert!(
                (governed.arms.left[0] - expected).abs() < 1e-12,
                "streamed side not at the cap (enabled={enabled}): {}",
                governed.arms.left[0]
            );
            assert_eq!(
                governed.arms.right[0], cand.arms.right[0],
                "the basis-free side is not capped"
            );
            assert_eq!(
                g.guard(),
                Guard::Clear,
                "a speed cap must not read as a collision event"
            );
        }
    }

    #[test]
    fn disabled_passes_a_colliding_opening_through() {
        // The toggle stands down the collision machinery and nothing else: a
        // rate-legal opening into a collision passes bit-exact, while an
        // over-rate jump is still clamped to the opening rate (the speed
        // limiters are motion shaping, not collision avoidance, so they do not
        // vanish with the toggle).
        let mut g = v2_governor(false);
        let arms = finger_into_other_arm();
        let legal_step = g.max_gripper_rate_frac_s() * DT;
        let prev = GovState::new(arms, ArmPair::new(0.0, 0.0));
        let open = GovState::new(arms, ArmPair::new(legal_step, 0.0));
        assert_eq!(
            g.govern(&prev, &open, &prev, NO_HANDS, DT),
            open,
            "disabled governor throttled a rate-legal colliding opening"
        );
        let jump = GovState::new(arms, ArmPair::new(1.0, 0.0));
        assert_eq!(
            g.govern(&prev, &jump, &prev, NO_HANDS, DT).grippers.left,
            legal_step,
            "an over-rate jump is rate-limited even while disabled"
        );
    }

    #[test]
    fn monitor_blocks_an_opening_on_a_measured_finger_breach() {
        let mut g = v2_governor(true);
        let arms = finger_into_other_arm();
        // The MEASURED fingers are open into the other arm past the trip floor
        // (tracking divergence: the commanded gripper is still nearly closed).
        let measured = config_at_distance(
            &mut g,
            &GovState::new(arms, ArmPair::new(0.0, 0.0)),
            &GovState::new(arms, ArmPair::new(1.0, 0.0)),
            0.5 * MONITOR_TRIP_FRACTION * D_STOP,
        );
        assert!(
            distance(&mut g, &measured) < MONITOR_TRIP_FRACTION * D_STOP,
            "setup: measured fingers breach the monitor floor"
        );
        let prev = GovState::new(arms, ArmPair::new(0.1, 0.0));
        let gripper_step = g.max_gripper_rate_frac_s() * DT;
        // Opening further closes the commanded-space gap: held.
        let open_more = GovState::new(arms, ArmPair::new(0.1 + gripper_step, 0.0));
        assert!(
            distance(&mut g, &open_more) < distance(&mut g, &prev),
            "setup: opening closes the gap"
        );
        assert_eq!(
            g.govern(&prev, &open_more, &measured, NO_HANDS, DT),
            prev,
            "the monitor passed an opening during a measured finger breach"
        );
        assert_eq!(g.guard(), Guard::Stopped, "a monitor hold reads stopped");
        // Closing (separation) still passes: the operator is never trapped.
        let close = GovState::new(arms, ArmPair::new(0.1 - gripper_step, 0.0));
        assert_ne!(
            g.govern(&prev, &close, &measured, NO_HANDS, DT),
            prev,
            "the monitor froze the closing escape"
        );
    }
}
