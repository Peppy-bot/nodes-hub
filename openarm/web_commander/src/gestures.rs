//! The gesture library: named choreography pre-baked to dense joint-space
//! trajectories and streamed over the same governed wire as live jogs.
//!
//! Gestures are authored in two forms: joint keyframes (interpolated with a C1
//! cubic Hermite spline) and Cartesian parametric curves (resolved to joints by
//! seed-chained IK against the same model the jog uses). `Registry::bake` runs
//! every definition through that resolution once at bringup and asserts
//! feasibility (reachability, joint limits, branch continuity, velocity budget),
//! so playback never validates: an infeasible definition aborts the node like a
//! bad URDF would, and whatever the registry serves is streamable as-is.

use std::sync::Arc;

use control_core::minimum_jerk::{self, QUINTIC_PEAK_VELOCITY};
use srs_model::nalgebra::{Quaternion, UnitQuaternion};

use crate::pose::ArmModels;
use openarm_description::{READY_R, Side as ModelSide, mirror};

use crate::state::{ARM_DOF, BySide, GesturePhase, GesturePlayback, SIDES, Side};

/// Dense bake grid (s). Playback interpolates linearly between grid samples, so
/// the command tick rate need not match.
const BAKE_DT_S: f64 = 0.01;
/// Largest joint step allowed between adjacent baked samples: the backbone's
/// branch-flip guard (a bigger step means IK jumped solution branches).
const MAX_BAKE_STEP_RAD: f64 = 0.35;
/// Fraction of the URDF velocity budget a baked step may use, leaving headroom
/// for the backbone's chase to track without lagging.
const VELOCITY_BUDGET_FRACTION: f64 = 0.8;
/// FK-vs-curve tolerance for Cartesian bakes: catches an IK "success" that
/// landed on a different branch than the curve point it was asked for.
const CURVE_TRACK_TOL_M: f64 = 0.005;
/// Slack on the joint-limit assert for keyframe gestures (the IK path is
/// in-limit by construction).
const LIMIT_SLACK_RAD: f64 = 1e-9;

/// Lead-in sizing: a quintic blend from the held target to the first sample,
/// paced at half the slowest involved joint's velocity budget and clamped to a
/// comfortable band.
const LEAD_IN_SPEED_FRACTION: f64 = 0.5;
const LEAD_IN_MIN_S: f64 = 0.8;
const LEAD_IN_MAX_S: f64 = 4.0;

/// The Ready workspace pose per side, from the description's canonical
/// postures (the backbone's move_to_ready lands on the same values). The
/// panel's Ready Pose button drives the same pose, served to it through the
/// snapshot, so the whole system has exactly one definition.
pub(crate) const READY: BySide<[f64; ARM_DOF]> = BySide::new(
    openarm_description::ready(ModelSide::Left),
    openarm_description::ready(ModelSide::Right),
);

/// The Home park pose per side, from the same canonical postures: the elbow
/// sits on the description's floor rather than the straight-arm singularity.
/// The panel's Home Pose button and the backbone's move_to_home land here.
pub(crate) const HOME: BySide<[f64; ARM_DOF]> = BySide::new(
    openarm_description::home(ModelSide::Left),
    openarm_description::home(ModelSide::Right),
);

// --------------------------- authoring types ---------------------------

/// One authored joint keyframe: time (s), the seven joint targets, and an
/// optional gripper opening key (fraction, 0..1).
struct Keyframe {
    t: f64,
    joints: [f64; ARM_DOF],
    gripper: Option<f64>,
}

/// A parametric Cartesian curve for one side: world-frame offsets (m) from the
/// side's anchor as a function of progress `s` in [0, 1].
type Curve = fn(f64) -> [f64; 3];

enum Motion {
    Joint(BySide<Option<Vec<Keyframe>>>),
    Cartesian(BySide<Option<Curve>>),
}

struct GestureDef {
    name: &'static str,
    label: &'static str,
    duration_s: f64,
    motion: Motion,
}

// --------------------------- baked types ---------------------------

/// One side's baked track: joint samples on the [`BAKE_DT_S`] grid, plus the
/// gripper opening per sample when the gesture drives it (`None` holds the
/// opening measured at start).
#[derive(Debug)]
struct BakedTrack {
    joints: Vec<[f64; ARM_DOF]>,
    gripper: Option<Vec<f64>>,
}

/// A gesture resolved to streamable joint trajectories, one shared clock across
/// its involved sides.
#[derive(Debug)]
pub struct BakedGesture {
    pub name: &'static str,
    pub label: &'static str,
    pub duration_s: f64,
    tracks: BySide<Option<BakedTrack>>,
}

impl BakedGesture {
    pub fn involves(&self, side: Side) -> bool {
        self.tracks[side].is_some()
    }

    /// The joint pose the lead-in blends toward.
    pub fn first_joints(&self, side: Side) -> Option<[f64; ARM_DOF]> {
        self.tracks[side].as_ref().map(|t| t.joints[0])
    }

    /// The opening the gripper track starts at, when the gesture drives the jaw.
    pub fn first_gripper(&self, side: Side) -> Option<f64> {
        self.tracks[side]
            .as_ref()
            .and_then(|t| t.gripper.as_ref())
            .map(|g| g[0])
    }

    /// Sample the track at `t` seconds (clamped to the trajectory), linearly
    /// interpolated between grid samples. `None` for an uninvolved side.
    pub fn sample(&self, side: Side, t: f64) -> Option<([f64; ARM_DOF], Option<f64>)> {
        let track = self.tracks[side].as_ref()?;
        let last = track.joints.len() - 1;
        let x = (t / BAKE_DT_S).clamp(0.0, last as f64);
        let (lo, frac) = (x.floor() as usize, x.fract());
        let hi = (lo + 1).min(last);
        let joints = std::array::from_fn(|i| lerp(track.joints[lo][i], track.joints[hi][i], frac));
        let gripper = track
            .gripper
            .as_ref()
            .map(|g| lerp(g[lo], g[hi], frac).clamp(0.0, 1.0));
        Some((joints, gripper))
    }
}

/// The baked gesture roster, resolved once at bringup.
#[derive(Clone)]
pub struct Registry(Vec<Arc<BakedGesture>>);

impl Registry {
    /// Bake every definition against the live arm models. Panics on an
    /// infeasible definition: a gesture that cannot be streamed safely is a
    /// build error surfaced at bringup, not a runtime condition.
    pub fn bake(models: &ArmModels) -> Self {
        Self(
            defs()
                .iter()
                .map(|def| Arc::new(bake(models, def)))
                .collect(),
        )
    }

    pub fn find(&self, name: &str) -> Option<Arc<BakedGesture>> {
        self.0.iter().find(|g| g.name == name).cloned()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<BakedGesture>> {
        self.0.iter()
    }
}

// --------------------------- playback ---------------------------

/// Per-side playback output for one tick: the joint setpoint and, when the
/// gesture drives it, the gripper opening.
pub type GestureSamples = BySide<Option<([f64; ARM_DOF], Option<f64>)>>;

/// The shared quintic profile held at its endpoints: playback evaluates past
/// the end of a segment, where the pose is the one the segment finished on.
pub fn quintic_blend(tau: f64) -> f64 {
    minimum_jerk::quintic(tau.clamp(0.0, 1.0)).0
}

/// Size the lead-in blend so the fastest joint peaks at
/// [`LEAD_IN_SPEED_FRACTION`] of its velocity budget, within the comfort band.
pub fn lead_in_duration(
    from: &[f64; ARM_DOF],
    to: &[f64; ARM_DOF],
    velocity_limits: &[f64; ARM_DOF],
) -> f64 {
    let slowest = (0..ARM_DOF)
        .map(|i| (to[i] - from[i]).abs() / (LEAD_IN_SPEED_FRACTION * velocity_limits[i]))
        .fold(0.0_f64, f64::max);
    (QUINTIC_PEAK_VELOCITY * slowest).clamp(LEAD_IN_MIN_S, LEAD_IN_MAX_S)
}

/// Advance playback one tick: the retained playback (`None` once complete) and
/// each involved side's setpoints to apply. Pure, so the tick decision is
/// testable without an owner.
pub fn advance_gesture_step(
    pb: &GesturePlayback,
    dt_s: f64,
) -> (Option<GesturePlayback>, GestureSamples) {
    let retained = |phase| {
        Some(GesturePlayback {
            gesture: pb.gesture.clone(),
            phase,
        })
    };
    match pb.phase {
        GesturePhase::LeadIn {
            from,
            gripper_from,
            t,
            duration_s,
        } => {
            let t = t + dt_s;
            if t >= duration_s {
                return (
                    retained(GesturePhase::Playing { t: 0.0 }),
                    sample_all(&pb.gesture, 0.0),
                );
            }
            let s = quintic_blend(t / duration_s);
            let mut out = BySide::new(None, None);
            for side in SIDES {
                let (Some(start), Some(first)) = (from[side], pb.gesture.first_joints(side)) else {
                    continue;
                };
                let joints = std::array::from_fn(|i| lerp(start[i], first[i], s));
                // A driven gripper eases from its measured opening to the
                // track's first value over the same blend; an undriven one
                // holds (the owner keeps streaming the seeded opening).
                let gripper = pb
                    .gesture
                    .first_gripper(side)
                    .and_then(|g0| gripper_from[side].map(|seed| lerp(seed, g0, s)));
                out[side] = Some((joints, gripper));
            }
            (
                retained(GesturePhase::LeadIn {
                    from,
                    gripper_from,
                    t,
                    duration_s,
                }),
                out,
            )
        }
        GesturePhase::Playing { t } => {
            let t = t + dt_s;
            let samples = sample_all(&pb.gesture, t);
            if t >= pb.gesture.duration_s {
                (None, samples)
            } else {
                (retained(GesturePhase::Playing { t }), samples)
            }
        }
    }
}

fn sample_all(gesture: &BakedGesture, t: f64) -> GestureSamples {
    BySide::new(
        gesture.sample(Side::Left, t),
        gesture.sample(Side::Right, t),
    )
}

fn lerp(a: f64, b: f64, s: f64) -> f64 {
    a + (b - a) * s
}

// --------------------------- baking ---------------------------

fn bake(models: &ArmModels, def: &GestureDef) -> BakedGesture {
    assert!(
        def.duration_s.is_finite() && def.duration_s > 0.0,
        "gesture '{}': duration must be positive",
        def.name
    );
    let clock_s = snap_to_grid(def.duration_s);
    let track = |side: Side| match &def.motion {
        Motion::Joint(tracks) => tracks[side]
            .as_ref()
            .map(|keys| bake_joint(models, def, side, keys, clock_s)),
        Motion::Cartesian(curves) => {
            curves[side].map(|curve| bake_cartesian(models, def, side, curve, clock_s))
        }
    };
    let mut tracks = BySide::new(track(Side::Left), track(Side::Right));
    assert!(
        SIDES.iter().any(|&s| tracks[s].is_some()),
        "gesture '{}': at least one side must have a track",
        def.name
    );
    let coda_s = append_ready_coda(models, &mut tracks);
    // The coda samples answer to the same guards as the gesture proper.
    for side in SIDES {
        if let Some(track) = tracks[side].as_ref() {
            assert_steps_within_budget(models, def, side, &track.joints, &|k| k as f64 * BAKE_DT_S);
        }
    }
    BakedGesture {
        name: def.name,
        label: def.label,
        duration_s: clock_s + coda_s,
        tracks,
    }
}

/// Every gesture ends at the Ready pose: when a track's last sample is not
/// already there, append a quintic joint blend back to Ready (paced like the
/// lead-in), holding any gripper track at its final value. Returns the coda
/// duration added to the shared clock, 0 when every track already ends at
/// Ready.
fn append_ready_coda(models: &ArmModels, tracks: &mut BySide<Option<BakedTrack>>) -> f64 {
    let coda_s = SIDES
        .iter()
        .filter_map(|&side| {
            let track = tracks[side].as_ref()?;
            let last = track.joints.last().expect("baked tracks are never empty");
            let moved = (0..ARM_DOF).any(|j| (last[j] - READY[side][j]).abs() > 1e-9);
            moved.then(|| lead_in_duration(last, &READY[side], models.velocity_limits(side)))
        })
        .fold(0.0_f64, f64::max);
    if coda_s <= 0.0 {
        return 0.0;
    }
    let coda_s = snap_to_grid(coda_s);
    let (n, at) = grid(coda_s);
    for side in SIDES {
        let Some(track) = tracks[side].as_mut() else {
            continue;
        };
        let from = *track.joints.last().expect("baked tracks are never empty");
        for k in 1..n {
            let blend = quintic_blend(at(k) / coda_s);
            track.joints.push(std::array::from_fn(|j| {
                lerp(from[j], READY[side][j], blend)
            }));
        }
        if let Some(gripper) = track.gripper.as_mut() {
            let hold = *gripper.last().expect("gripper tracks are never empty");
            gripper.extend(std::iter::repeat_n(hold, n - 1));
        }
    }
    coda_s
}

/// Round a duration up to a whole number of grid steps, so every baked track
/// is uniformly spaced and playback's index arithmetic is exact.
fn snap_to_grid(duration_s: f64) -> f64 {
    (duration_s / BAKE_DT_S).ceil() * BAKE_DT_S
}

/// The bake grid over a snapped clock: sample count and the time of sample `k`.
fn grid(clock_s: f64) -> (usize, impl Fn(usize) -> f64) {
    let steps = (clock_s / BAKE_DT_S).round() as usize;
    (steps + 1, move |k: usize| k as f64 * BAKE_DT_S)
}

fn bake_joint(
    models: &ArmModels,
    def: &GestureDef,
    side: Side,
    keys: &[Keyframe],
    clock_s: f64,
) -> BakedTrack {
    let name = def.name;
    assert!(
        keys.len() >= 2,
        "gesture '{name}': need at least two keyframes"
    );
    assert!(
        keys.windows(2).all(|w| w[0].t < w[1].t),
        "gesture '{name}': keyframe times must be strictly increasing"
    );
    assert!(
        keys[0].t == 0.0 && keys[keys.len() - 1].t == def.duration_s,
        "gesture '{name}': keyframes must span [0, duration]"
    );

    let tangents = monotone_tangents(keys);
    let (n, at) = grid(clock_s);
    let joints: Vec<[f64; ARM_DOF]> = (0..n)
        .map(|k| hermite_sample(keys, &tangents, at(k)))
        .collect();

    let ranges = models.joint_ranges(side);
    for (k, q) in joints.iter().enumerate() {
        for j in 0..ARM_DOF {
            assert!(
                q[j] >= ranges[j][0] - LIMIT_SLACK_RAD && q[j] <= ranges[j][1] + LIMIT_SLACK_RAD,
                "gesture '{name}' {side_l}: j{j1} = {v:.4} outside [{lo:.4}, {hi:.4}] at sample {k}",
                side_l = side.label(),
                j1 = j + 1,
                v = q[j],
                lo = ranges[j][0],
                hi = ranges[j][1],
            );
        }
    }
    assert_steps_within_budget(models, def, side, &joints, &at);

    BakedTrack {
        joints,
        gripper: bake_gripper_track(name, keys, clock_s),
    }
}

/// Monotone Catmull-Rom tangents: zero at the ends (start and stop at rest),
/// zero wherever a joint plateaus or reverses (repeated keyframes hold truly
/// still), and magnitude-clamped to three times the smaller adjacent secant
/// (Fritsch-Carlson), so a segment never overshoots its authored endpoints.
fn monotone_tangents(keys: &[Keyframe]) -> Vec<[f64; ARM_DOF]> {
    (0..keys.len())
        .map(|i| {
            if i == 0 || i == keys.len() - 1 {
                return [0.0; ARM_DOF];
            }
            let (prev, here, next) = (&keys[i - 1], &keys[i], &keys[i + 1]);
            std::array::from_fn(|j| {
                let secant_in = (here.joints[j] - prev.joints[j]) / (here.t - prev.t);
                let secant_out = (next.joints[j] - here.joints[j]) / (next.t - here.t);
                if secant_in * secant_out <= 0.0 {
                    return 0.0;
                }
                let tangent = (next.joints[j] - prev.joints[j]) / (next.t - prev.t);
                let bound = 3.0 * secant_in.abs().min(secant_out.abs());
                tangent.clamp(-bound, bound)
            })
        })
        .collect()
}

/// Sample the keyframe spline at `t`: cubic Hermite on each segment.
fn hermite_sample(keys: &[Keyframe], tangents: &[[f64; ARM_DOF]], t: f64) -> [f64; ARM_DOF] {
    let seg = keys.partition_point(|k| k.t <= t).clamp(1, keys.len() - 1) - 1;
    let (a, b) = (&keys[seg], &keys[seg + 1]);
    let dt = b.t - a.t;
    let h = ((t - a.t) / dt).clamp(0.0, 1.0);
    let (h2, h3) = (h * h, h * h * h);
    let (b00, b10, b01, b11) = (
        2.0 * h3 - 3.0 * h2 + 1.0,
        h3 - 2.0 * h2 + h,
        -2.0 * h3 + 3.0 * h2,
        h3 - h2,
    );
    std::array::from_fn(|j| {
        b00 * a.joints[j]
            + b10 * dt * tangents[seg][j]
            + b01 * b.joints[j]
            + b11 * dt * tangents[seg + 1][j]
    })
}

/// The gripper track from the keyframes that carry a gripper key: cubic
/// smoothstep between consecutive keys, held flat outside them. `None` when no
/// keyframe drives the gripper (playback then holds the measured opening).
fn bake_gripper_track(name: &str, keys: &[Keyframe], clock_s: f64) -> Option<Vec<f64>> {
    let gkeys: Vec<(f64, f64)> = keys
        .iter()
        .filter_map(|k| k.gripper.map(|g| (k.t, g)))
        .collect();
    if gkeys.is_empty() {
        return None;
    }
    for &(t, g) in &gkeys {
        assert!(
            (0.0..=1.0).contains(&g),
            "gesture '{name}': gripper key {g} at t={t} outside [0, 1]"
        );
    }
    let (n, at) = grid(clock_s);
    Some((0..n).map(|k| sample_gripper(&gkeys, at(k))).collect())
}

fn sample_gripper(gkeys: &[(f64, f64)], t: f64) -> f64 {
    if t <= gkeys[0].0 {
        return gkeys[0].1;
    }
    let Some(next) = gkeys.iter().position(|&(kt, _)| kt > t) else {
        return gkeys[gkeys.len() - 1].1;
    };
    let (t0, g0) = gkeys[next - 1];
    let (t1, g1) = gkeys[next];
    let h = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
    lerp(g0, g1, h * h * (3.0 - 2.0 * h))
}

fn bake_cartesian(
    models: &ArmModels,
    def: &GestureDef,
    side: Side,
    curve: Curve,
    clock_s: f64,
) -> BakedTrack {
    let name = def.name;
    let anchor_pos = anchor_position(models, side);
    // Hold the Ready orientation across the whole trace: only the position
    // draws (fixed world aims proved unreachable across these curves; the
    // wrist pitch limit is tight).
    let q4 = models.ee_quat_world(side, &READY[side]);
    let rotation = UnitQuaternion::from_quaternion(Quaternion::new(q4[3], q4[0], q4[1], q4[2]));

    let (n, at) = grid(clock_s);
    let mut prev = READY[side];
    let joints: Vec<[f64; ARM_DOF]> = (0..n)
        .map(|k| {
            let off = curve(quintic_blend(at(k) / def.duration_s));
            let p = [
                anchor_pos[0] + off[0],
                anchor_pos[1] + off[1],
                anchor_pos[2] + off[2],
            ];
            let q = models
                .solve_ik(side, p, rotation, &prev)
                .unwrap_or_else(|| {
                    panic!(
                        "gesture '{name}' {}: unreachable curve point {p:?} at sample {k}",
                        side.label()
                    )
                });
            let ee = models.ee_pose_world(side, &q);
            let err = crate::pose::dist3([ee[0], ee[1], ee[2]], p);
            assert!(
                err < CURVE_TRACK_TOL_M,
                "gesture '{name}' {}: IK landed {err:.4} m off the curve at sample {k}",
                side.label()
            );
            prev = q;
            q
        })
        .collect();
    assert_steps_within_budget(models, def, side, &joints, &at);

    BakedTrack {
        joints,
        gripper: None,
    }
}

/// Curve offsets anchor on the side's own Ready end-effector position.
fn anchor_position(models: &ArmModels, side: Side) -> [f64; 3] {
    let p = models.ee_pose_world(side, &READY[side]);
    [p[0], p[1], p[2]]
}

/// Every adjacent baked pair must respect the branch-flip guard and leave the
/// backbone's chase velocity headroom; a violation is an authoring error.
fn assert_steps_within_budget(
    models: &ArmModels,
    def: &GestureDef,
    side: Side,
    joints: &[[f64; ARM_DOF]],
    at: &impl Fn(usize) -> f64,
) {
    let budget = models.velocity_limits(side);
    for k in 1..joints.len() {
        let dt = at(k) - at(k - 1);
        if dt <= 0.0 {
            continue;
        }
        for j in 0..ARM_DOF {
            let step = (joints[k][j] - joints[k - 1][j]).abs();
            assert!(
                step < MAX_BAKE_STEP_RAD,
                "gesture '{}' {}: j{} steps {step:.3} rad at sample {k} (adjacent-sample guard; on an IK path this means a branch flip)",
                def.name,
                side.label(),
                j + 1
            );
            let cap = VELOCITY_BUDGET_FRACTION * budget[j] * dt;
            assert!(
                step <= cap,
                "gesture '{}' {}: j{} needs {:.2} rad/s at sample {k}, budget {:.2}",
                def.name,
                side.label(),
                j + 1,
                step / dt,
                VELOCITY_BUDGET_FRACTION * budget[j]
            );
        }
    }
}

// --------------------------- the roster ---------------------------

fn kf(t: f64, joints: [f64; ARM_DOF]) -> Keyframe {
    Keyframe {
        t,
        joints,
        gripper: None,
    }
}

fn kfg(t: f64, joints: [f64; ARM_DOF], gripper: f64) -> Keyframe {
    Keyframe {
        t,
        joints,
        gripper: Some(gripper),
    }
}

fn defs() -> Vec<GestureDef> {
    vec![wave(), spiral(), figure_eight(), shrug(), clap()]
}

/// The right hand raised in front at head height, forearm up, waving side to
/// side with the gripper flapping open and closed on the beats.
fn wave() -> GestureDef {
    let up = |j3: f64| [1.00, 0.35, j3, 1.90, 0.0, 0.0, 0.0];
    GestureDef {
        name: "wave",
        label: "Wave",
        duration_s: 8.0,
        motion: Motion::Joint(BySide::new(
            None,
            Some(vec![
                kfg(0.0, READY_R, 0.0),
                kfg(1.5, up(0.0), 0.35),
                kfg(2.25, up(0.40), 0.5),
                kfg(3.0, up(-0.40), 0.1),
                kfg(3.75, up(0.40), 0.5),
                kfg(4.5, up(-0.40), 0.1),
                kfg(5.25, up(0.40), 0.5),
                kfg(6.1, up(0.0), 0.25),
                kfg(8.0, READY_R, 0.0),
            ]),
        )),
    }
}

/// The right arm draws an Archimedean spiral in the vertical plane in front of
/// it, winding out from its Ready point; arc length advances uniformly in the
/// eased progress, so the outer rings do not sprint.
fn spiral() -> GestureDef {
    GestureDef {
        name: "spiral",
        label: "Spiral",
        duration_s: 11.0,
        motion: Motion::Cartesian(BySide::new(None, Some(spiral_curve as Curve))),
    }
}

fn spiral_curve(s: f64) -> [f64; 3] {
    const TURNS: f64 = 3.0;
    const R_MAX: f64 = 0.10;
    // The envelope at the held Ready orientation thins a few centimetres below
    // the anchor, so the spiral winds around a lifted centre.
    const Z_LIFT_M: f64 = 0.07;
    let theta_max = TURNS * std::f64::consts::TAU;
    // theta grows with sqrt(s) so the arc length advances uniformly in s and
    // the outer rings do not sprint.
    let theta = theta_max * s.sqrt();
    let r = R_MAX * theta / theta_max;
    [0.0, r * theta.cos(), Z_LIFT_M + r * theta.sin()]
}

/// The right arm traces an upright figure eight twice, long axis vertical.
fn figure_eight() -> GestureDef {
    GestureDef {
        name: "figure_eight",
        label: "Figure Eight",
        duration_s: 10.0,
        motion: Motion::Cartesian(BySide::new(None, Some(figure_eight_curve as Curve))),
    }
}

fn figure_eight_curve(s: f64) -> [f64; 3] {
    const LOOPS: f64 = 2.0;
    // Lifted like the spiral: the figure rides above the anchor where the
    // held-orientation envelope is deep.
    const Z_LIFT_M: f64 = 0.07;
    let u = LOOPS * std::f64::consts::TAU * s;
    [0.0, 0.06 * u.sin() * u.cos(), Z_LIFT_M + 0.11 * u.sin()]
}

/// Both arms out with palms up and a double shoulder bounce: a shrug.
fn shrug() -> GestureDef {
    let open = |lift: f64| [0.55, 0.85 + lift, 0.0, 1.30, 0.0, 0.0, 0.0];
    let right = vec![
        kf(0.0, READY_R),
        kf(1.2, open(0.0)),
        kf(1.7, open(0.30)),
        kf(2.2, open(0.0)),
        kf(2.7, open(0.30)),
        kf(3.4, open(0.0)),
        kf(4.4, open(0.0)),
        kf(6.0, READY_R),
    ];
    let left = right.iter().map(|k| kf(k.t, mirror(k.joints))).collect();
    GestureDef {
        name: "shrug",
        label: "Shrug",
        duration_s: 6.0,
        motion: Motion::Joint(BySide::new(Some(left), Some(right))),
    }
}

/// Hands brought together in front at mid height, then clapped with a rising
/// tempo.
fn clap() -> GestureDef {
    let clap = |j3: f64| [0.70, 0.50, j3, 1.45, 0.0, 0.0, 0.0];
    const APART: f64 = -0.55;
    const TOGETHER: f64 = -0.82;
    const CLOSED: f64 = 0.0;
    let beats = [1.4, 1.2, 1.0, 0.85, 0.75, 0.65];
    let mut right = vec![kfg(0.0, READY_R, CLOSED), kfg(1.5, clap(APART), CLOSED)];
    let mut t = 1.5;
    for beat in beats {
        t += beat;
        right.push(kfg(t, clap(TOGETHER), CLOSED));
        t += beat * 0.6;
        right.push(kfg(t, clap(APART), CLOSED));
    }
    right.push(kfg(t + 2.0, READY_R, CLOSED));
    let duration_s = t + 2.0;
    let left = right
        .iter()
        .map(|k| Keyframe {
            t: k.t,
            joints: mirror(k.joints),
            gripper: k.gripper,
        })
        .collect();
    GestureDef {
        name: "clap",
        label: "Clap",
        duration_s,
        motion: Motion::Joint(BySide::new(Some(left), Some(right))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openarm_description::HardwareVersion;

    fn models() -> ArmModels {
        ArmModels::from_version(HardwareVersion::V2)
    }

    fn test_keys() -> Vec<Keyframe> {
        vec![
            kf(0.0, [0.0; ARM_DOF]),
            kfg(1.0, [0.4, -0.2, 0.1, 1.0, 0.0, 0.3, -0.1], 1.0),
            kf(2.0, [-0.2, 0.3, -0.1, 0.6, 0.2, -0.3, 0.1]),
            kfg(3.0, [0.0; ARM_DOF], 0.0),
        ]
    }

    #[test]
    fn hermite_passes_through_every_keyframe() {
        let keys = test_keys();
        let tangents = monotone_tangents(&keys);
        for k in &keys {
            let q = hermite_sample(&keys, &tangents, k.t);
            for (sampled, authored) in q.iter().zip(k.joints.iter()) {
                assert!(
                    (sampled - authored).abs() < 1e-12,
                    "spline misses keyframe at t={}",
                    k.t
                );
            }
        }
    }

    #[test]
    fn hermite_velocity_is_continuous_at_knots() {
        let keys = test_keys();
        let tangents = monotone_tangents(&keys);
        let eps = 1e-6;
        for k in &keys[1..keys.len() - 1] {
            let before = hermite_sample(&keys, &tangents, k.t - eps);
            let after = hermite_sample(&keys, &tangents, k.t + eps);
            let at = hermite_sample(&keys, &tangents, k.t);
            for j in 0..ARM_DOF {
                let v_in = (at[j] - before[j]) / eps;
                let v_out = (after[j] - at[j]) / eps;
                assert!(
                    (v_in - v_out).abs() < 1e-4,
                    "velocity jumps {v_in} -> {v_out} at knot t={}",
                    k.t
                );
            }
        }
    }

    #[test]
    fn hermite_starts_and_ends_at_rest() {
        let keys = test_keys();
        let tangents = monotone_tangents(&keys);
        let eps = 1e-6;
        let end = keys[keys.len() - 1].t;
        let v0 = hermite_sample(&keys, &tangents, eps);
        let v1 = hermite_sample(&keys, &tangents, end - eps);
        for j in 0..ARM_DOF {
            assert!(
                (v0[j] - keys[0].joints[j]).abs() / eps < 1e-3,
                "moving at t=0"
            );
            assert!(
                (v1[j] - keys[keys.len() - 1].joints[j]).abs() / eps < 1e-3,
                "moving at t=end"
            );
        }
    }

    #[test]
    fn repeated_keyframes_hold_perfectly_still() {
        // The monotone clamp zeroes tangents at plateaus, so a held pose never
        // wobbles or overshoots between its repeated keyframes.
        let hold = [0.5, -0.3, 0.2, 1.4, 0.0, -0.2, 0.6];
        let keys = vec![
            kf(0.0, [0.0; ARM_DOF]),
            kf(1.0, hold),
            kf(2.0, hold),
            kf(3.0, [0.1; ARM_DOF]),
        ];
        let tangents = monotone_tangents(&keys);
        for step in 0..=20 {
            let t = 1.0 + step as f64 / 20.0;
            let q = hermite_sample(&keys, &tangents, t);
            for j in 0..ARM_DOF {
                assert!(
                    (q[j] - hold[j]).abs() < 1e-12,
                    "held pose drifts at t={t}: j{} = {} vs {}",
                    j + 1,
                    q[j],
                    hold[j]
                );
            }
        }
    }

    #[test]
    fn the_blend_holds_at_its_endpoints() {
        // The profile itself is control_core's; what this wrapper owns is the
        // hold, since playback evaluates past the end of a segment.
        assert_eq!(quintic_blend(0.0), 0.0);
        assert_eq!(quintic_blend(1.0), 1.0);
        assert_eq!(quintic_blend(-5.0), 0.0);
        assert_eq!(quintic_blend(5.0), 1.0);
    }

    #[test]
    fn registry_bakes_every_gesture_on_v2() {
        let registry = Registry::bake(&models());
        let names: Vec<&str> = registry.iter().map(|g| g.name).collect();
        assert_eq!(names, ["wave", "spiral", "figure_eight", "shrug", "clap"]);
    }

    #[test]
    fn registry_bakes_every_gesture_on_v1() {
        // The commander launches on either generation and the bake panics on an
        // infeasible definition, so v1 feasibility is a build gate too.
        Registry::bake(&ArmModels::from_version(HardwareVersion::V1));
    }

    #[test]
    fn every_baked_step_respects_velocity_budget_and_limits() {
        // Re-assert outside the constructor so a relaxed bake cannot silently pass.
        let m = models();
        for g in Registry::bake(&m).iter() {
            for side in SIDES {
                let Some(track) = g.tracks[side].as_ref() else {
                    continue;
                };
                let budget = m.velocity_limits(side);
                let ranges = m.joint_ranges(side);
                for (k, w) in track.joints.windows(2).enumerate() {
                    for j in 0..ARM_DOF {
                        let step = (w[1][j] - w[0][j]).abs();
                        assert!(step < MAX_BAKE_STEP_RAD, "{}: branch flip at {k}", g.name);
                        assert!(
                            step <= VELOCITY_BUDGET_FRACTION * budget[j] * BAKE_DT_S + 1e-12,
                            "{} {} j{}: over budget at sample {k}",
                            g.name,
                            side.label(),
                            j + 1
                        );
                    }
                }
                for q in &track.joints {
                    for j in 0..ARM_DOF {
                        assert!(
                            q[j] >= ranges[j][0] - 1e-9 && q[j] <= ranges[j][1] + 1e-9,
                            "{} {}: j{} out of limits",
                            g.name,
                            side.label(),
                            j + 1
                        );
                    }
                }
                if let Some(grip) = &track.gripper {
                    assert_eq!(grip.len(), track.joints.len());
                    assert!(grip.iter().all(|v| (0.0..=1.0).contains(v)));
                }
            }
        }
    }

    #[test]
    fn cartesian_bakes_track_their_curves() {
        let m = models();
        let registry = Registry::bake(&m);
        for name in ["spiral", "figure_eight"] {
            let g = registry.find(name).unwrap();
            for side in SIDES {
                let Some(track) = g.tracks[side].as_ref() else {
                    continue;
                };
                // Adjacent samples stay close in task space too: no Cartesian
                // teleports between grid points.
                for w in track.joints.windows(2) {
                    let a = m.ee_pose_world(side, &w[0]);
                    let b = m.ee_pose_world(side, &w[1]);
                    let d = crate::pose::dist3([a[0], a[1], a[2]], [b[0], b[1], b[2]]);
                    assert!(d < 0.01, "{name} {}: {d:.4} m EE jump", side.label());
                }
            }
        }
    }

    #[test]
    fn every_gesture_ends_at_ready() {
        for g in Registry::bake(&models()).iter() {
            for side in SIDES {
                let Some((last, _)) = g.sample(side, g.duration_s) else {
                    continue;
                };
                for j in 0..ARM_DOF {
                    assert!(
                        (last[j] - READY[side][j]).abs() < 1e-9,
                        "gesture '{}' {} j{} ends at {} not Ready",
                        g.name,
                        side.label(),
                        j + 1,
                        last[j]
                    );
                }
            }
        }
    }

    #[test]
    fn dual_arm_gestures_keep_the_hands_apart() {
        // The governor throttles inside d_safe (2 cm) and stops inside d_stop;
        // baked choreography must keep the end-effector frames clear of that
        // band with margin for the finger geometry around each frame. The
        // 0.06 m floor comes from streaming candidate poses at the sim governor
        // and reading its proximity feed: the closest untouched pair (the clap)
        // settles at 23 mm surface clearance with EE frames ~0.26 m apart.
        let m = models();
        for g in Registry::bake(&m).iter() {
            let (Some(l), Some(r)) = (
                g.tracks[Side::Left].as_ref(),
                g.tracks[Side::Right].as_ref(),
            ) else {
                continue;
            };
            let mut min_gap = f64::INFINITY;
            for (ql, qr) in l.joints.iter().zip(r.joints.iter()) {
                let (pl, pr) = (
                    m.ee_pose_world(Side::Left, ql),
                    m.ee_pose_world(Side::Right, qr),
                );
                let d = crate::pose::dist3([pl[0], pl[1], pl[2]], [pr[0], pr[1], pr[2]]);
                min_gap = min_gap.min(d);
            }
            assert!(
                min_gap > 0.06,
                "gesture '{}': hands close to {min_gap:.3} m",
                g.name
            );
        }
    }

    #[test]
    fn sampling_interpolates_and_clamps() {
        let g = Registry::bake(&models()).find("wave").unwrap();
        let (q0, _) = g.sample(Side::Right, 0.0).unwrap();
        assert_eq!(q0, g.first_joints(Side::Right).unwrap());
        assert!(
            g.sample(Side::Left, 0.0).is_none(),
            "wave is right-arm only"
        );
        // Past the end holds the final pose.
        let (qe, _) = g.sample(Side::Right, g.duration_s + 5.0).unwrap();
        let (qd, _) = g.sample(Side::Right, g.duration_s).unwrap();
        assert_eq!(qe, qd);
        // Between grid points, sampling lands between the neighbors.
        let (qa, _) = g.sample(Side::Right, 1.005).unwrap();
        let (ql, _) = g.sample(Side::Right, 1.0).unwrap();
        let (qh, _) = g.sample(Side::Right, 1.01).unwrap();
        for j in 0..ARM_DOF {
            let (lo, hi) = (ql[j].min(qh[j]), ql[j].max(qh[j]));
            assert!(qa[j] >= lo - 1e-12 && qa[j] <= hi + 1e-12);
        }
    }

    #[test]
    fn lead_in_duration_scales_and_clamps() {
        let limits = [2.0; ARM_DOF];
        let zero = [0.0; ARM_DOF];
        assert_eq!(lead_in_duration(&zero, &zero, &limits), LEAD_IN_MIN_S);
        let mut far = zero;
        far[0] = 3.0;
        assert_eq!(lead_in_duration(&zero, &far, &limits), LEAD_IN_MAX_S);
        let mut mid = zero;
        mid[2] = 0.8;
        let expected = QUINTIC_PEAK_VELOCITY * 0.8 / (LEAD_IN_SPEED_FRACTION * 2.0);
        assert!((lead_in_duration(&zero, &mid, &limits) - expected).abs() < 1e-12);
    }

    #[test]
    fn playback_leads_in_plays_and_completes() {
        let gesture = Registry::bake(&models()).find("wave").unwrap();
        let start = [0.1, 0.2, -0.1, 0.8, 0.0, 0.1, 0.0];
        let mut pb = GesturePlayback {
            gesture: gesture.clone(),
            phase: GesturePhase::LeadIn {
                from: BySide::new(None, Some(start)),
                gripper_from: BySide::new(None, Some(0.7)),
                t: 0.0,
                duration_s: 1.0,
            },
        };
        let dt = 0.01;
        let mut prev = start;
        let mut prev_grip = 0.7;
        let mut ticks = 0;
        loop {
            let (next, samples) = advance_gesture_step(&pb, dt);
            assert!(samples[Side::Left].is_none());
            if let Some((q, grip)) = samples[Side::Right] {
                // The wave drives the jaw, so every tick carries an opening and
                // it eases rather than snapping (0.7 seed to the 0.0 track start).
                let grip = grip.expect("wave's gripper track streams throughout");
                assert!(
                    (grip - prev_grip).abs() < 0.05,
                    "tick {ticks} snapped the gripper"
                );
                prev_grip = grip;
                // No snap anywhere: every tick's step stays small.
                for j in 0..ARM_DOF {
                    assert!(
                        (q[j] - prev[j]).abs() < 0.1,
                        "tick {ticks} snapped j{}",
                        j + 1
                    );
                }
                prev = q;
            }
            match next {
                Some(p) => pb = p,
                None => break,
            }
            ticks += 1;
            assert!(ticks < 10_000, "playback never completed");
        }
        let (q_end, _) = gesture.sample(Side::Right, gesture.duration_s).unwrap();
        assert_eq!(prev, q_end, "playback ends on the final sample");
    }
}
