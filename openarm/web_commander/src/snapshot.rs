//! The browser snapshot: the view model the owner serializes on its snapshot
//! tick and the panel renders, with every severity and staleness judged here
//! so the browser draws it without re-deriving anything.

use std::time::{Duration, Instant};

use openarm_description::{ARM_DOF, Side};
use serde::Serialize;
use srs_model::JointVec;

use crate::gestures::{HOME, READY, Registry};
use crate::limits::{arm_limits, gripper_limits};
use crate::pose::{ArmModels, Pose};
use crate::state::{
    Alert, ArmTarget, Disposition, GesturePhase, GovernorSettings, GripperTarget, HealthLevel,
    HealthReport, Limb, Proximity, UiState,
};

// The backbone publishes the proximity readout at ~20 Hz; treat it as stale after this
// long with no update (a dead backbone) so the panel falls back to n/a instead of
// latching the last distance.
const PROXIMITY_STALE_AFTER: Duration = Duration::from_millis(500);
// A launch's producers need a beat to start reporting: until this long after
// state creation, a health component that has never reported renders as
// pending rather than raising a not-reporting problem, so a healthy launch
// does not flash a false alarm.
const STARTUP_GRACE: Duration = Duration::from_secs(3);

/// Serialize the browser snapshot from the owner's state; called on the owner's
/// snapshot tick, so it holds no lock and each connection forwards the same bytes.
pub(crate) fn build_snapshot_json(
    s: &UiState,
    now: Instant,
    models: &ArmModels,
    registry: &Registry,
) -> serde_json::Result<String> {
    serde_json::to_string(&Snapshot::build(s, now, models, registry))
}

#[derive(Serialize)]
struct Snapshot {
    left_arm: ArmView,
    right_arm: ArmView,
    left_gripper: GripperView,
    right_gripper: GripperView,
    // Streaming deadman per side, shared by that side's arm and gripper.
    left_enabled: bool,
    right_enabled: bool,
    // Operator's self-collision governor controls (streamed to the backbone).
    #[serde(flatten)]
    governor: GovernorSettings,
    // Live nearest-pair proximity from the backbone (null until the first report).
    proximity: Option<ProximityView>,
    // Motor health with severity fully computed server-side; the browser's
    // chips, banner, and motors tables render it without re-deriving.
    health: HealthPanelView,
    // Active operator alerts, most severe first.
    alerts: Vec<AlertView>,
    // Whether anything is bound to the alerts slot; an empty list means "no
    // active alerts" only when this is true.
    alerts_bound: bool,
    // The baked gesture roster (name, label, involved sides); the browser builds
    // its buttons from this, so gestures live in exactly one place.
    gestures: Vec<GestureView>,
    // The gesture in flight, if any.
    gesture: Option<GestureStatusView>,
    // The dataset recorder panel; null when the deployment binds no recorder
    // (the browser hides the panel entirely).
    recorder: Option<RecorderView>,
    status: String,
}

#[derive(Serialize)]
struct RecorderView {
    recording: bool,
    // finish_session in flight (finalize + mirror); gates the Finish button.
    finishing: bool,
    // Stop was requested and the episode is encoding its videos; the goal
    // stays in flight (and `recording` true) until the save lands.
    saving: bool,
    // Frames written to the in-flight episode; 0 between episodes.
    frames: u64,
}

#[derive(Serialize)]
struct GestureView {
    name: &'static str,
    label: &'static str,
    left: bool,
    right: bool,
}

#[derive(Serialize)]
struct GestureStatusView {
    name: &'static str,
    phase: GesturePhaseView,
    // Fraction of the current phase completed, 0..1.
    progress: f64,
}

/// A gesture's phase as the browser reads it.
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum GesturePhaseView {
    /// Blending from the held target to the gesture's first pose.
    LeadIn,
    Playing,
}

#[derive(Serialize)]
struct ProximityView {
    distance: f64,
    link_a: String,
    link_b: String,
    throttled: bool,
    stopped: bool,
}

#[derive(Serialize)]
struct AlertView {
    source: String,
    // Alert severity on the contract's scale: 1 warning, 2 critical, 3
    // fault. The browser indexes its colour and label tables by this number.
    severity: u8,
    message: String,
}

/// Everything the browser's health surfaces render: per-side per-motor rows,
/// the combined severities, and the operator-facing problems.
#[derive(Serialize)]
struct HealthPanelView {
    // Whether any producer is bound to the motor_health slot; false renders
    // every surface as "not wired" rather than implying a healthy robot.
    bound: bool,
    // Worst wire level across both sides, grippers included; null when
    // nothing is judged yet (unwired, or every component still pending).
    worst: Option<u8>,
    // Problems at critical and above (consumer-detected silence included),
    // most severe first; the banner shows exactly these.
    problems: Vec<ProblemView>,
    left: SideHealthView,
    right: SideHealthView,
}

#[derive(Serialize)]
struct ProblemView {
    side: Side,
    // The wire level behind the problem; the banner colors on the first
    // (most severe) entry.
    level: u8,
    text: String,
}

/// One side's health surfaces: its render state, the combined worst across
/// arm joints and gripper, and one row per motor (j1..j7 then the gripper).
#[derive(Serialize)]
struct SideHealthView {
    status: SideHealthStatus,
    // Null while pending or unwired; a quiet side reads NotReporting.
    worst: Option<u8>,
    // Empty unless live.
    motors: Vec<MotorRowView>,
}

/// What a side's health surfaces render.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SideHealthStatus {
    /// The rows are current.
    Live,
    /// Nothing reported yet, inside the startup grace.
    Pending,
    /// Bound but quiet.
    NotReporting,
    /// No producer is bound to the motor_health slot.
    NotWired,
}

/// One motor's table row, the same shape for arm joints and the gripper.
#[derive(Serialize)]
struct MotorRowView {
    name: String,
    // The wire level; null for a motor whose component is still pending.
    level: Option<u8>,
    effort_fraction_rated: Option<f64>,
    effort_fraction_rated_sustained: Option<f64>,
    effort_fraction_peak: Option<f64>,
    driver_temp_c: Option<f64>,
    winding_temp_c: Option<f64>,
}

#[derive(Serialize)]
struct ArmView {
    joints: JointVec,
    feedback: Option<JointVec>,
    in_flight: bool,
    // Per-joint [min, max] (rad); the browser bounds its sliders with these.
    limits: [[f64; 2]; ARM_DOF],
    // The side's Ready pose, the single definition the gesture library anchors
    // on; the browser's Ready Pose button fires these joints.
    ready: JointVec,
    // The side's Home pose, from the same canonical postures; the browser's
    // Home Pose button fires these joints.
    home: JointVec,
    // World-frame x/y/z reachable bounds [[min, max]; 3]; the browser bounds its
    // position sliders with these (per generation, from the arm's FK envelope).
    pos_bounds: [[f64; 2]; 3],
    // World-frame end-effector pose [x, y, z, roll, pitch, yaw] of the joint target
    // (FK), so moving a joint updates the panel's pose fields.
    pose: Pose,
    // Same for the measured joints, `null` until the first state arrives; the panel
    // shows it beside the target pose the way it does per-joint feedback.
    pose_feedback: Option<Pose>,
    // World-frame end-effector orientation as a quaternion [x, y, z, w] for the target
    // (FK of the joint target) and the measured pose. The arcball composes on these,
    // so orientation never round-trips through euler on the wire.
    orientation: [f64; 4],
    orientation_feedback: Option<[f64; 4]>,
}

#[derive(Serialize)]
struct GripperView {
    position: f64,
    // Measured opening fraction from the gripper_states stream.
    feedback: Option<f64>,
    min: f64,
    max: f64,
    // The operator's effort cap (null = no preference: the ceiling applies).
    max_effort: Option<f64>,
    // The gripper's reported effort ceiling: null until the first state
    // report, 0 = no effort control (the panel hides the effort slider).
    effort_ceiling: Option<f64>,
    // A discrete move_gripper is in flight (drives the gripper card's badge).
    in_flight: bool,
}

impl Snapshot {
    fn build(s: &UiState, now: Instant, models: &ArmModels, registry: &Registry) -> Self {
        Self {
            gestures: registry
                .iter()
                .map(|g| GestureView {
                    name: g.name,
                    label: g.label,
                    left: g.involves(Side::Left),
                    right: g.involves(Side::Right),
                })
                .collect(),
            gesture: s.gesture.as_ref().map(|pb| match pb.phase {
                GesturePhase::LeadIn { t, duration_s, .. } => GestureStatusView {
                    name: pb.gesture.name,
                    phase: GesturePhaseView::LeadIn,
                    progress: (t / duration_s).clamp(0.0, 1.0),
                },
                GesturePhase::Playing { t } => GestureStatusView {
                    name: pb.gesture.name,
                    phase: GesturePhaseView::Playing,
                    progress: (t / pb.gesture.duration_s).clamp(0.0, 1.0),
                },
            }),
            left_arm: arm_view(&s.arms[Side::Left], Side::Left, models),
            right_arm: arm_view(&s.arms[Side::Right], Side::Right, models),
            left_gripper: gripper_view(&s.grippers[Side::Left]),
            right_gripper: gripper_view(&s.grippers[Side::Right]),
            left_enabled: s.enabled[Side::Left],
            right_enabled: s.enabled[Side::Right],
            governor: s.governor,
            recorder: s.recorder.available.then(|| RecorderView {
                recording: s.recorder.episode.is_some(),
                finishing: s.recorder.finishing,
                saving: s
                    .recorder
                    .episode
                    .as_ref()
                    .is_some_and(|e| e.stop.is_cancelled()),
                frames: s.recorder.episode.as_ref().map_or(0, |e| e.frames),
            }),
            proximity: live_proximity(s, now).map(|p| ProximityView {
                distance: p.distance,
                link_a: p.link_a.clone(),
                link_b: p.link_b.clone(),
                throttled: p.disposition == Disposition::Throttled,
                stopped: p.disposition == Disposition::Stopped,
            }),
            health: health_panel_view(s, now),
            alerts: alert_views(s, now),
            alerts_bound: s.alerts_bound,
            status: s.status.clone(),
        }
    }
}

/// The proximity readout if it is still fresh, else `None` (the backbone stopped
/// reporting), so the UI falls back to n/a instead of latching a stale distance.
fn live_proximity(s: &UiState, now: Instant) -> Option<&Proximity> {
    s.proximity
        .as_ref()
        .filter(|p| now.duration_since(p.received_at) < PROXIMITY_STALE_AFTER)
}

/// The alerts of every producer whose set is still live, most severe first.
///
/// Producers are ordered by identity and each producer lists its own alerts
/// in a fixed order, and the sort is stable, so the list holds its order
/// between redraws while a producer republishes an unchanged set.
fn alert_views(s: &UiState, now: Instant) -> Vec<AlertView> {
    let mut live: Vec<&Alert> = s
        .alerts
        .values()
        .filter(|set| set.validity.is_live_at(now))
        .flat_map(|set| set.alerts.iter())
        .collect();
    live.sort_by_key(|a| std::cmp::Reverse(a.severity));
    live.into_iter()
        .map(|a| AlertView {
            source: a.source.clone(),
            severity: a.severity.wire(),
            message: a.message.clone(),
        })
        .collect()
}

/// One health component's consumer-side condition at `now`.
enum ComponentState<'a, const MOTORS: usize> {
    /// A report inside its validity window.
    Live(&'a HealthReport<MOTORS>),
    /// Bound but nothing renderable: it reported once and aged out, or it
    /// has never reported and the startup grace is spent.
    NotReporting,
    /// Never reported, still inside the startup grace.
    Pending,
}

fn component_state<'a, const MOTORS: usize>(
    stored: &'a Option<HealthReport<MOTORS>>,
    now: Instant,
    in_grace: bool,
) -> ComponentState<'a, MOTORS> {
    match stored {
        Some(report) if report.validity.is_live_at(now) => ComponentState::Live(report),
        Some(_) => ComponentState::NotReporting,
        None if in_grace => ComponentState::Pending,
        None => ComponentState::NotReporting,
    }
}

/// A component's contribution to its side: the worst level it pins (`None`
/// while pending) and the operator-facing problem it raises, if any. A live
/// component raises its worst at critical and above; a quiet one raises
/// not-reporting, which the level order ranks above every spoken condition.
fn component_severity<const MOTORS: usize>(
    state: &ComponentState<'_, MOTORS>,
    side: Side,
    limb: Limb,
) -> (Option<HealthLevel>, Option<ProblemView>) {
    match state {
        ComponentState::Live(report) => {
            let worst = report.worst();
            let problem = (worst >= HealthLevel::Critical).then(|| ProblemView {
                side,
                level: worst.wire(),
                text: format!("{side} {limb} {}", level_label(worst)),
            });
            (Some(worst), problem)
        }
        ComponentState::NotReporting => (
            Some(HealthLevel::NotReporting),
            Some(ProblemView {
                side,
                level: HealthLevel::NotReporting.wire(),
                text: format!("{side} {limb} not reporting"),
            }),
        ),
        ComponentState::Pending => (None, None),
    }
}

/// One row per motor of a component: live readings verbatim, a quiet
/// component as not-reporting rows, a pending one as level-less rows.
fn component_rows<const MOTORS: usize>(
    state: &ComponentState<'_, MOTORS>,
    name: impl Fn(usize) -> String,
) -> Vec<MotorRowView> {
    (0..MOTORS)
        .map(|i| match state {
            ComponentState::Live(report) => {
                let reading = &report.readings[i];
                MotorRowView {
                    name: name(i),
                    level: Some(reading.level.wire()),
                    effort_fraction_rated: reading.effort_fraction_rated,
                    effort_fraction_rated_sustained: reading.effort_fraction_rated_sustained,
                    effort_fraction_peak: reading.effort_fraction_peak,
                    driver_temp_c: reading.driver_temp_c,
                    winding_temp_c: reading.winding_temp_c,
                }
            }
            ComponentState::NotReporting => absent_row(name(i), Some(HealthLevel::NotReporting)),
            ComponentState::Pending => absent_row(name(i), None),
        })
        .collect()
}

fn absent_row(name: String, level: Option<HealthLevel>) -> MotorRowView {
    MotorRowView {
        name,
        level: level.map(HealthLevel::wire),
        effort_fraction_rated: None,
        effort_fraction_rated_sustained: None,
        effort_fraction_peak: None,
        driver_temp_c: None,
        winding_temp_c: None,
    }
}

/// Operator-facing name of a level, matching the browser's label table.
fn level_label(level: HealthLevel) -> &'static str {
    match level {
        HealthLevel::Nominal => "nominal",
        HealthLevel::Warning => "WARNING",
        HealthLevel::Critical => "CRITICAL",
        HealthLevel::Fault => "FAULT",
        HealthLevel::NotReporting => "NOT REPORTING",
    }
}

/// One side's view plus the problems its components raise. The side is live
/// while either component is (the other renders as its own rows); with
/// nothing live it reads not-reporting, or pending when no component has
/// ever reported and the startup grace still runs.
fn side_health_view(
    s: &UiState,
    side: Side,
    now: Instant,
    in_grace: bool,
) -> (SideHealthView, Vec<ProblemView>) {
    if !s.health_bound {
        let view = SideHealthView {
            status: SideHealthStatus::NotWired,
            worst: None,
            motors: Vec::new(),
        };
        return (view, Vec::new());
    }
    let arm = component_state(&s.arm_health[side], now, in_grace);
    let gripper = component_state(&s.gripper_health[side], now, in_grace);
    let live = matches!(arm, ComponentState::Live(_)) || matches!(gripper, ComponentState::Live(_));
    let (arm_worst, arm_problem) = component_severity(&arm, side, Limb::Arm);
    let (gripper_worst, gripper_problem) = component_severity(&gripper, side, Limb::Gripper);
    let worst = [arm_worst, gripper_worst].into_iter().flatten().max();
    let motors = if live {
        let mut rows = component_rows(&arm, |i| format!("j{}", i + 1));
        rows.extend(component_rows(&gripper, |_| Limb::Gripper.to_string()));
        rows
    } else {
        Vec::new()
    };
    let status = match (live, worst) {
        (true, _) => SideHealthStatus::Live,
        (false, Some(_)) => SideHealthStatus::NotReporting,
        (false, None) => SideHealthStatus::Pending,
    };
    let view = SideHealthView {
        status,
        worst: worst.map(HealthLevel::wire),
        motors,
    };
    (
        view,
        arm_problem.into_iter().chain(gripper_problem).collect(),
    )
}

/// The health panel: both sides' views, the overall worst, and the problems
/// list sorted most severe first (the banner colors on the first entry).
fn health_panel_view(s: &UiState, now: Instant) -> HealthPanelView {
    let in_grace = now.duration_since(s.created_at) < STARTUP_GRACE;
    let (left, left_problems) = side_health_view(s, Side::Left, now, in_grace);
    let (right, right_problems) = side_health_view(s, Side::Right, now, in_grace);
    let mut problems: Vec<ProblemView> = left_problems.into_iter().chain(right_problems).collect();
    problems.sort_by_key(|p| std::cmp::Reverse(p.level));
    HealthPanelView {
        bound: s.health_bound,
        worst: [left.worst, right.worst].into_iter().flatten().max(),
        problems,
        left,
        right,
    }
}

fn arm_view(a: &ArmTarget, side: Side, models: &ArmModels) -> ArmView {
    ArmView {
        joints: a.joints,
        feedback: a.last_feedback,
        in_flight: a.in_flight,
        limits: *arm_limits(side),
        ready: READY[side],
        home: HOME[side],
        pos_bounds: models.pos_bounds(side),
        pose: models.ee_pose_world(side, &a.joints),
        pose_feedback: a.last_feedback.map(|fb| models.ee_pose_world(side, &fb)),
        orientation: models.ee_quat_world(side, &a.joints),
        orientation_feedback: a.last_feedback.map(|fb| models.ee_quat_world(side, &fb)),
    }
}

fn gripper_view(g: &GripperTarget) -> GripperView {
    let [min, max] = gripper_limits();
    GripperView {
        position: g.opening,
        feedback: g.last_feedback,
        min,
        max,
        max_effort: g.max_effort,
        effort_ceiling: g.effort_ceiling,
        in_flight: g.in_flight,
    }
}

#[cfg(test)]
mod tests {
    use peppylib::messaging::ProducerRef;

    use super::*;
    use crate::consumer::{ALERT_STALE_AFTER, HEALTH_STALE_AFTER};
    use crate::limits::init_limits_for_tests;
    use crate::state::{AlertSet, AlertSeverity, ArmHealth, GripperHealth, Validity};
    use crate::test_support::{models, reading, registry, ui_state};

    /// A state with health producers bound, as every wired deployment has.
    fn bound_state() -> UiState {
        let mut s = ui_state();
        s.health_bound = true;
        s
    }

    fn producer(instance: &str) -> ProducerRef {
        ProducerRef::new("core", instance)
    }

    fn left_arm() -> ProducerRef {
        producer("left_arm_inst")
    }

    fn alert(source: &str, severity: AlertSeverity) -> Alert {
        Alert {
            source: source.to_string(),
            severity,
            message: "holding 93% of rated torque".to_string(),
        }
    }

    /// A set received at `received_at`, so it renders until its window ends.
    fn alert_set(alerts: Vec<Alert>, received_at: Instant) -> AlertSet {
        AlertSet {
            alerts,
            validity: Validity::new(received_at, ALERT_STALE_AFTER),
        }
    }

    fn arm_health(levels: [HealthLevel; ARM_DOF], received_at: Instant) -> ArmHealth {
        HealthReport {
            readings: levels.map(reading),
            validity: Validity::new(received_at, HEALTH_STALE_AFTER),
        }
    }

    fn gripper_health(level: HealthLevel, received_at: Instant) -> GripperHealth {
        HealthReport {
            readings: [reading(level)],
            validity: Validity::new(received_at, HEALTH_STALE_AFTER),
        }
    }

    /// The problem texts, in banner order.
    fn problem_texts(view: &HealthPanelView) -> Vec<&str> {
        view.problems.iter().map(|p| p.text.as_str()).collect()
    }

    #[test]
    fn a_producers_set_replaces_its_whole_entry_and_clears_by_absence() {
        let mut s = ui_state();
        let t0 = s.created_at;
        s.apply_alert_set(
            left_arm(),
            alert_set(
                vec![
                    alert("left arm j2", AlertSeverity::Warning),
                    alert("left arm j5", AlertSeverity::Warning),
                ],
                t0,
            ),
        );
        s.apply_alert_set(
            left_arm(),
            alert_set(vec![alert("left arm j2", AlertSeverity::Critical)], t0),
        );
        assert_eq!(s.alerts.len(), 1, "one set per producer");
        let views = alert_views(&s, t0);
        assert_eq!(views.len(), 1, "the omitted alert is gone");
        assert_eq!(views[0].source, "left arm j2");
        assert_eq!(views[0].severity, AlertSeverity::Critical.wire());

        s.apply_alert_set(left_arm(), alert_set(Vec::new(), t0));
        assert!(
            alert_views(&s, t0).is_empty(),
            "a set that omits every alert clears them"
        );
    }

    #[test]
    fn a_producer_that_stops_refreshing_its_set_ages_out() {
        // The contract's cadence is what lets the panel stop presenting a
        // quiet producer's conditions as current.
        let mut s = ui_state();
        let t0 = s.created_at;
        s.apply_alert_set(
            left_arm(),
            alert_set(vec![alert("left arm j2", AlertSeverity::Fault)], t0),
        );
        assert_eq!(
            alert_views(&s, t0 + ALERT_STALE_AFTER - Duration::from_millis(1)).len(),
            1,
            "live for the whole window"
        );
        assert!(
            alert_views(&s, t0 + ALERT_STALE_AFTER).is_empty(),
            "a quiet producer's set stops rendering"
        );
    }

    #[test]
    fn a_producer_cannot_replace_or_clear_anothers_alert() {
        let mut s = ui_state();
        let t0 = s.created_at;
        s.apply_alert_set(
            left_arm(),
            alert_set(vec![alert("left arm j2", AlertSeverity::Critical)], t0),
        );
        s.apply_alert_set(
            producer("imposter"),
            alert_set(vec![alert("left arm j2", AlertSeverity::Warning)], t0),
        );
        assert_eq!(
            alert_views(&s, t0).len(),
            2,
            "same wire strings, distinct producers"
        );
        s.apply_alert_set(producer("imposter"), alert_set(Vec::new(), t0));
        let views = alert_views(&s, t0);
        assert_eq!(views.len(), 1, "the empty set cleared only its own entry");
        assert_eq!(views[0].severity, AlertSeverity::Critical.wire());
    }

    #[test]
    fn a_producer_that_leaves_the_slot_takes_its_alerts_with_it() {
        let mut s = ui_state();
        let t0 = s.created_at;
        let right = producer("right_arm_inst");
        s.apply_alert_set(
            left_arm(),
            alert_set(vec![alert("left arm j2", AlertSeverity::Warning)], t0),
        );
        s.apply_alert_set(
            right.clone(),
            alert_set(vec![alert("right arm j1", AlertSeverity::Critical)], t0),
        );
        s.retain_alert_producers(&[right]);
        let views = alert_views(&s, t0);
        assert_eq!(views.len(), 1, "the departed producer's set is gone");
        assert_eq!(views[0].source, "right arm j1");
        assert!(s.alerts_bound, "the slot still holds a producer");
    }

    #[test]
    fn an_empty_bound_set_clears_every_alert_and_reads_as_unwired() {
        // The startup state and the last-producer-leaves state. A panel that
        // kept the rows would show a robot nobody is watching as a healthy
        // one.
        let mut s = ui_state();
        let t0 = s.created_at;
        s.alerts_bound = true;
        s.apply_alert_set(
            left_arm(),
            alert_set(vec![alert("left arm j2", AlertSeverity::Fault)], t0),
        );
        s.retain_alert_producers(&[]);
        assert!(alert_views(&s, t0).is_empty());
        assert!(!s.alerts_bound, "an empty slot is not wired");
    }

    #[test]
    fn alert_views_sort_most_severe_first_and_hold_their_order() {
        let mut s = ui_state();
        let t0 = s.created_at;
        let listed = || {
            vec![
                alert("left arm j2", AlertSeverity::Critical),
                alert("left arm j5", AlertSeverity::Fault),
                alert("left arm j7", AlertSeverity::Warning),
            ]
        };
        s.apply_alert_set(left_arm(), alert_set(listed(), t0));
        s.apply_alert_set(
            producer("right_arm_inst"),
            alert_set(vec![alert("right arm j1", AlertSeverity::Critical)], t0),
        );
        let sources: Vec<String> = alert_views(&s, t0)
            .iter()
            .map(|v| v.source.clone())
            .collect();
        assert_eq!(
            sources,
            ["left arm j5", "left arm j2", "right arm j1", "left arm j7"],
            "severity descending, then producer, then the producer's own order"
        );

        // A producer republishing an unchanged set must not reshuffle the
        // list under an operator reading it.
        s.apply_alert_set(left_arm(), alert_set(listed(), t0));
        let after: Vec<String> = alert_views(&s, t0)
            .iter()
            .map(|v| v.source.clone())
            .collect();
        assert_eq!(after, sources);
    }

    #[test]
    fn unwired_health_renders_not_wired_everywhere() {
        let s = ui_state();
        let view = health_panel_view(&s, s.created_at);
        assert!(!view.bound);
        assert_eq!(view.worst, None);
        assert!(view.problems.is_empty());
        assert_eq!(view.left.status, SideHealthStatus::NotWired);
        assert_eq!(view.right.status, SideHealthStatus::NotWired);
        assert!(view.left.motors.is_empty());
    }

    #[test]
    fn per_side_worst_combines_gripper_severity() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.arm_health[Side::Left] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Fault, t0));
        let view = health_panel_view(&s, t0);
        assert_eq!(view.left.status, SideHealthStatus::Live);
        assert_eq!(
            view.left.worst,
            Some(HealthLevel::Fault.wire()),
            "a faulted gripper under a nominal arm is the side's worst"
        );
        assert_eq!(view.worst, Some(HealthLevel::Fault.wire()));
        assert_eq!(problem_texts(&view)[0], "left gripper FAULT");
    }

    #[test]
    fn side_rows_name_each_joint_then_the_gripper() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.arm_health[Side::Left] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Warning, t0));
        let view = health_panel_view(&s, t0);
        let names: Vec<&str> = view.left.motors.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]);
        let gripper = &view.left.motors[ARM_DOF];
        assert_eq!(gripper.level, Some(HealthLevel::Warning.wire()));
        assert_eq!(gripper.effort_fraction_rated, Some(0.4));
        assert_eq!(gripper.effort_fraction_peak, Some(0.2));
    }

    #[test]
    fn not_reporting_outranks_fault_at_the_side_level() {
        let mut s = bound_state();
        let t0 = s.created_at;
        let mut levels = [HealthLevel::Nominal; ARM_DOF];
        levels[3] = HealthLevel::Fault;
        s.arm_health[Side::Left] = Some(arm_health(levels, t0));
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Nominal, t0));
        // The gripper's window expires while the arm keeps reporting.
        let later = t0 + HEALTH_STALE_AFTER;
        s.arm_health[Side::Left] = Some(arm_health(levels, later));
        let view = health_panel_view(&s, later);
        assert_eq!(
            view.left.status,
            SideHealthStatus::Live,
            "the arm still reports"
        );
        assert_eq!(view.left.worst, Some(HealthLevel::NotReporting.wire()));
        assert_eq!(
            problem_texts(&view),
            ["left gripper not reporting", "left arm FAULT"],
            "silence sorts above the spoken fault"
        );
        let gripper_row = &view.left.motors[ARM_DOF];
        assert_eq!(gripper_row.level, Some(HealthLevel::NotReporting.wire()));
        assert_eq!(
            gripper_row.effort_fraction_rated, None,
            "a quiet component's stale readings do not render as current"
        );
    }

    #[test]
    fn within_grace_a_silent_launch_reads_pending_not_a_problem() {
        let s = bound_state();
        let view = health_panel_view(&s, s.created_at + STARTUP_GRACE - Duration::from_millis(1));
        assert_eq!(view.left.status, SideHealthStatus::Pending);
        assert_eq!(view.right.status, SideHealthStatus::Pending);
        assert_eq!(view.worst, None);
        assert!(
            view.problems.is_empty(),
            "a healthy launch must not flash a false alarm"
        );
    }

    #[test]
    fn bound_but_silent_past_grace_raises_not_reporting() {
        let s = bound_state();
        let view = health_panel_view(&s, s.created_at + STARTUP_GRACE);
        assert_eq!(view.left.status, SideHealthStatus::NotReporting);
        assert_eq!(view.left.worst, Some(HealthLevel::NotReporting.wire()));
        assert_eq!(view.worst, Some(HealthLevel::NotReporting.wire()));
        assert_eq!(
            problem_texts(&view),
            [
                "left arm not reporting",
                "left gripper not reporting",
                "right arm not reporting",
                "right gripper not reporting",
            ]
        );
    }

    #[test]
    fn gripper_silence_after_first_report_is_a_problem_even_inside_grace() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Nominal, t0));
        // Its window expires before the startup grace does; having reported
        // once, the gripper must escalate rather than vanish.
        let now = t0 + HEALTH_STALE_AFTER;
        assert!(now < t0 + STARTUP_GRACE, "the test must land inside grace");
        let view = health_panel_view(&s, now);
        assert_eq!(problem_texts(&view), ["left gripper not reporting"]);
        assert_eq!(view.left.status, SideHealthStatus::NotReporting);
        assert_eq!(
            view.right.status,
            SideHealthStatus::Pending,
            "the never-reported side stays pending"
        );
    }

    #[test]
    fn health_ages_out_for_arm_and_gripper_instead_of_latching() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.arm_health[Side::Left] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Nominal, t0));
        let live = health_panel_view(&s, t0 + HEALTH_STALE_AFTER - Duration::from_millis(1));
        assert_eq!(live.left.status, SideHealthStatus::Live);
        assert!(live.problems.is_empty());
        let aged = health_panel_view(&s, t0 + HEALTH_STALE_AFTER);
        assert_eq!(aged.left.status, SideHealthStatus::NotReporting);
        assert_eq!(
            problem_texts(&aged),
            ["left arm not reporting", "left gripper not reporting"]
        );
    }

    #[test]
    fn statuses_and_phases_serialize_to_the_words_the_browser_reads() {
        for (status, word) in [
            (SideHealthStatus::Live, "live"),
            (SideHealthStatus::Pending, "pending"),
            (SideHealthStatus::NotReporting, "not_reporting"),
            (SideHealthStatus::NotWired, "not_wired"),
        ] {
            assert_eq!(serde_json::to_value(status).unwrap(), word);
        }
        for (phase, word) in [
            (GesturePhaseView::LeadIn, "lead_in"),
            (GesturePhaseView::Playing, "playing"),
        ] {
            assert_eq!(serde_json::to_value(phase).unwrap(), word);
        }
    }

    /// An object's key set, sorted so the pin is independent of map order.
    fn keys(v: &serde_json::Value) -> Vec<&str> {
        let mut keys: Vec<&str> = v
            .as_object()
            .expect("an object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn snapshot_serializes_the_keys_the_browser_reads() {
        init_limits_for_tests();
        let models = models();
        let registry = registry();
        let mut s = bound_state();
        s.alerts_bound = true;
        let t0 = s.created_at;
        s.arm_health[Side::Left] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Fault, t0));
        s.apply_alert_set(
            left_arm(),
            alert_set(vec![alert("left arm j2", AlertSeverity::Critical)], t0),
        );
        let json = build_snapshot_json(&s, t0, &models, registry).unwrap();
        let snap: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(
            keys(&snap),
            [
                "alerts",
                "alerts_bound",
                "collision_enabled",
                "d_safe",
                "d_stop",
                "gesture",
                "gestures",
                "health",
                "left_arm",
                "left_enabled",
                "left_gripper",
                "max_ee_velocity_m_s",
                "max_gripper_rate_frac_s",
                "proximity",
                "recorder",
                "right_arm",
                "right_enabled",
                "right_gripper",
                "status",
            ]
        );
        let health = &snap["health"];
        assert_eq!(
            keys(health),
            ["bound", "left", "problems", "right", "worst"]
        );
        assert_eq!(health["bound"], serde_json::json!(true));
        assert_eq!(health["worst"], serde_json::json!(3));
        assert_eq!(keys(&health["left"]), ["motors", "status", "worst"]);
        assert_eq!(health["left"]["status"], "live");
        assert_eq!(health["right"]["status"], "pending");
        let row = &health["left"]["motors"][0];
        assert_eq!(
            keys(row),
            [
                "driver_temp_c",
                "effort_fraction_peak",
                "effort_fraction_rated",
                "effort_fraction_rated_sustained",
                "level",
                "name",
                "winding_temp_c",
            ]
        );
        assert_eq!(row["name"], "j1");
        assert_eq!(health["left"]["motors"][7]["name"], "gripper");
        let problem = &health["problems"][0];
        assert_eq!(keys(problem), ["level", "side", "text"]);
        assert_eq!(problem["side"], "left");
        assert_eq!(problem["level"], serde_json::json!(3));
        assert_eq!(problem["text"], "left gripper FAULT");
        let alert_view = &snap["alerts"][0];
        assert_eq!(keys(alert_view), ["message", "severity", "source"]);
    }
}
