//! The browser snapshot: the view model the owner serializes on its snapshot
//! tick and the panel renders, with every severity and staleness judged here
//! so the browser draws it without re-deriving anything.

use std::time::{Duration, Instant};

use robot_description::{ArmId, PerArm};
use serde::Serialize;
use srs_model::{ARM_DOF, JointVec};

use crate::gestures::Registry;
use crate::limits::{arm_limits, gripper_limits};
use crate::pose::{ArmModels, Pose};
use crate::robot::{self, ARMS};
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
    // Each arm with its gripper, in place order; the browser builds one set of
    // panels per entry and names them by these names.
    arms: Vec<ArmPanelView>,
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
    // The baked gesture roster (name, label, involved arms); the browser builds
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

/// One arm and its gripper, as the browser's panels for them render.
#[derive(Serialize)]
struct ArmPanelView {
    // The arm's name and its gripper's, the names the wire carries; the
    // browser labels the panels with them and names the arm in each command.
    name: &'static str,
    gripper_name: &'static str,
    // Streaming deadman, shared by the arm and its gripper.
    enabled: bool,
    arm: ArmView,
    gripper: GripperView,
    health: ArmHealthView,
}

#[derive(Serialize)]
struct GestureView {
    name: &'static str,
    label: &'static str,
    // The names of the arms the gesture drives.
    arms: Vec<&'static str>,
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

/// The robot-wide health surfaces the browser renders: the combined severity
/// and the operator-facing problems. Each arm's own surfaces ride its
/// [`ArmPanelView`].
#[derive(Serialize)]
struct HealthPanelView {
    // Whether any producer is bound to the motor_health slot; false renders
    // every surface as "not wired" rather than implying a healthy robot.
    bound: bool,
    // Worst wire level across every arm, grippers included; null when
    // nothing is judged yet (unwired, or every component still pending).
    worst: Option<u8>,
    // Problems at critical and above (consumer-detected silence included),
    // most severe first; the banner shows exactly these.
    problems: Vec<ProblemView>,
}

#[derive(Serialize)]
struct ProblemView {
    // The wire level behind the problem; the banner colors on the first
    // (most severe) entry.
    level: u8,
    // The limb's name and its condition.
    text: String,
}

/// One arm's health surfaces: its render state, the combined worst across
/// its joints and its gripper, and one row per motor (j1..j7 then the gripper).
#[derive(Serialize)]
struct ArmHealthView {
    status: ArmHealthStatus,
    // Null while pending or unwired; a quiet arm reads NotReporting.
    worst: Option<u8>,
    // Empty unless live.
    motors: Vec<MotorRowView>,
}

/// What an arm's health surfaces render.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ArmHealthStatus {
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
    // The arm's ready posture, the SRDF's, which the gesture library anchors
    // on; the browser's Ready Pose button fires these joints.
    ready: JointVec,
    // The arm's home posture, the SRDF's; the browser's Home Pose button fires
    // these joints.
    home: JointVec,
    // World-frame x/y/z reachable bounds [[min, max]; 3]; the browser bounds its
    // position sliders with these (per model, from the arm's FK envelope).
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
        let (health, arm_health) = health_panel_view(s, now);
        Self {
            arms: PerArm::<(), ARMS>::ids()
                .zip(arm_health)
                .map(|(arm, health)| ArmPanelView {
                    name: robot::LIMBS.arm_name(arm),
                    gripper_name: robot::LIMBS.gripper_name(arm),
                    enabled: s.enabled[arm],
                    arm: arm_view(&s.arms[arm], arm, models),
                    gripper: gripper_view(&s.grippers[arm]),
                    health,
                })
                .collect(),
            gestures: registry
                .iter()
                .map(|g| GestureView {
                    name: g.name,
                    label: g.label,
                    arms: PerArm::<(), ARMS>::ids()
                        .filter(|&arm| g.involves(arm))
                        .map(|arm| robot::LIMBS.arm_name(arm))
                        .collect(),
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
            health,
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

/// A component's contribution to its arm: the worst level it pins (`None`
/// while pending) and the operator-facing problem it raises, if any. A live
/// component raises its worst at critical and above; a quiet one raises
/// not-reporting, which the level order ranks above every spoken condition.
fn component_severity<const MOTORS: usize>(
    state: &ComponentState<'_, MOTORS>,
    arm: ArmId,
    limb: Limb,
) -> (Option<HealthLevel>, Option<ProblemView>) {
    let name = limb.name_of(arm);
    match state {
        ComponentState::Live(report) => {
            let worst = report.worst();
            let problem = (worst >= HealthLevel::Critical).then(|| ProblemView {
                level: worst.wire(),
                text: format!("{name} {}", level_label(worst)),
            });
            (Some(worst), problem)
        }
        ComponentState::NotReporting => (
            Some(HealthLevel::NotReporting),
            Some(ProblemView {
                level: HealthLevel::NotReporting.wire(),
                text: format!("{name} not reporting"),
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

/// One arm's view plus the problems its components raise. The arm is live
/// while either component is (the other renders as its own rows); with
/// nothing live it reads not-reporting, or pending when no component has
/// ever reported and the startup grace still runs.
fn arm_health_view(
    s: &UiState,
    arm: ArmId,
    now: Instant,
    in_grace: bool,
) -> (ArmHealthView, Vec<ProblemView>) {
    if !s.health_bound {
        let view = ArmHealthView {
            status: ArmHealthStatus::NotWired,
            worst: None,
            motors: Vec::new(),
        };
        return (view, Vec::new());
    }
    let joints = component_state(&s.arm_health[arm], now, in_grace);
    let gripper = component_state(&s.gripper_health[arm], now, in_grace);
    let live =
        matches!(joints, ComponentState::Live(_)) || matches!(gripper, ComponentState::Live(_));
    let (arm_worst, arm_problem) = component_severity(&joints, arm, Limb::Arm);
    let (gripper_worst, gripper_problem) = component_severity(&gripper, arm, Limb::Gripper);
    let worst = [arm_worst, gripper_worst].into_iter().flatten().max();
    let motors = if live {
        let mut rows = component_rows(&joints, |i| format!("j{}", i + 1));
        rows.extend(component_rows(&gripper, |_| Limb::Gripper.to_string()));
        rows
    } else {
        Vec::new()
    };
    let status = match (live, worst) {
        (true, _) => ArmHealthStatus::Live,
        (false, Some(_)) => ArmHealthStatus::NotReporting,
        (false, None) => ArmHealthStatus::Pending,
    };
    let view = ArmHealthView {
        status,
        worst: worst.map(HealthLevel::wire),
        motors,
    };
    (
        view,
        arm_problem.into_iter().chain(gripper_problem).collect(),
    )
}

/// The health panel: the overall worst and the problems list sorted most
/// severe first (the banner colors on the first entry), and each arm's view.
fn health_panel_view(s: &UiState, now: Instant) -> (HealthPanelView, PerArm<ArmHealthView, ARMS>) {
    let in_grace = now.duration_since(s.created_at) < STARTUP_GRACE;
    let (views, problems) = PerArm::from_fn(|arm| arm_health_view(s, arm, now, in_grace)).unzip();
    let mut problems: Vec<ProblemView> = problems.into_iter().flatten().collect();
    problems.sort_by_key(|p| std::cmp::Reverse(p.level));
    let panel = HealthPanelView {
        bound: s.health_bound,
        worst: views.iter().filter_map(|view| view.worst).max(),
        problems,
    };
    (panel, views)
}

fn arm_view(a: &ArmTarget, arm: ArmId, models: &ArmModels) -> ArmView {
    ArmView {
        joints: a.joints,
        feedback: a.last_feedback,
        in_flight: a.in_flight,
        limits: *arm_limits(arm),
        ready: models.ready(arm),
        home: models.home(arm),
        pos_bounds: models.pos_bounds(arm),
        pose: models.ee_pose_world(arm, &a.joints),
        pose_feedback: a.last_feedback.map(|fb| models.ee_pose_world(arm, &fb)),
        orientation: models.ee_quat_world(arm, &a.joints),
        orientation_feedback: a.last_feedback.map(|fb| models.ee_quat_world(arm, &fb)),
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
    use crate::test_support::{LEFT, RIGHT, models, reading, registry, ui_state};

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
                    alert("left_arm j2", AlertSeverity::Warning),
                    alert("left_arm j5", AlertSeverity::Warning),
                ],
                t0,
            ),
        );
        s.apply_alert_set(
            left_arm(),
            alert_set(vec![alert("left_arm j2", AlertSeverity::Critical)], t0),
        );
        assert_eq!(s.alerts.len(), 1, "one set per producer");
        let views = alert_views(&s, t0);
        assert_eq!(views.len(), 1, "the omitted alert is gone");
        assert_eq!(views[0].source, "left_arm j2");
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
            alert_set(vec![alert("left_arm j2", AlertSeverity::Fault)], t0),
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
            alert_set(vec![alert("left_arm j2", AlertSeverity::Critical)], t0),
        );
        s.apply_alert_set(
            producer("imposter"),
            alert_set(vec![alert("left_arm j2", AlertSeverity::Warning)], t0),
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
            alert_set(vec![alert("left_arm j2", AlertSeverity::Warning)], t0),
        );
        s.apply_alert_set(
            right.clone(),
            alert_set(vec![alert("right_arm j1", AlertSeverity::Critical)], t0),
        );
        s.retain_alert_producers(&[right]);
        let views = alert_views(&s, t0);
        assert_eq!(views.len(), 1, "the departed producer's set is gone");
        assert_eq!(views[0].source, "right_arm j1");
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
            alert_set(vec![alert("left_arm j2", AlertSeverity::Fault)], t0),
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
                alert("left_arm j2", AlertSeverity::Critical),
                alert("left_arm j5", AlertSeverity::Fault),
                alert("left_arm j7", AlertSeverity::Warning),
            ]
        };
        s.apply_alert_set(left_arm(), alert_set(listed(), t0));
        s.apply_alert_set(
            producer("right_arm_inst"),
            alert_set(vec![alert("right_arm j1", AlertSeverity::Critical)], t0),
        );
        let sources: Vec<String> = alert_views(&s, t0)
            .iter()
            .map(|v| v.source.clone())
            .collect();
        assert_eq!(
            sources,
            ["left_arm j5", "left_arm j2", "right_arm j1", "left_arm j7"],
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
        let (view, arms) = health_panel_view(&s, s.created_at);
        assert!(!view.bound);
        assert_eq!(view.worst, None);
        assert!(view.problems.is_empty());
        assert_eq!(arms[LEFT].status, ArmHealthStatus::NotWired);
        assert_eq!(arms[RIGHT].status, ArmHealthStatus::NotWired);
        assert!(arms[LEFT].motors.is_empty());
    }

    #[test]
    fn per_arm_worst_combines_gripper_severity() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.arm_health[LEFT] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[LEFT] = Some(gripper_health(HealthLevel::Fault, t0));
        let (view, arms) = health_panel_view(&s, t0);
        assert_eq!(arms[LEFT].status, ArmHealthStatus::Live);
        assert_eq!(
            arms[LEFT].worst,
            Some(HealthLevel::Fault.wire()),
            "a faulted gripper under a nominal arm is the arm's worst"
        );
        assert_eq!(view.worst, Some(HealthLevel::Fault.wire()));
        assert_eq!(problem_texts(&view)[0], "left_gripper FAULT");
    }

    #[test]
    fn arm_rows_name_each_joint_then_the_gripper() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.arm_health[LEFT] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[LEFT] = Some(gripper_health(HealthLevel::Warning, t0));
        let (_, arms) = health_panel_view(&s, t0);
        let names: Vec<&str> = arms[LEFT].motors.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]);
        let gripper = &arms[LEFT].motors[ARM_DOF];
        assert_eq!(gripper.level, Some(HealthLevel::Warning.wire()));
        assert_eq!(gripper.effort_fraction_rated, Some(0.4));
        assert_eq!(gripper.effort_fraction_peak, Some(0.2));
    }

    #[test]
    fn not_reporting_outranks_fault_at_the_arm_level() {
        let mut s = bound_state();
        let t0 = s.created_at;
        let mut levels = [HealthLevel::Nominal; ARM_DOF];
        levels[3] = HealthLevel::Fault;
        s.arm_health[LEFT] = Some(arm_health(levels, t0));
        s.gripper_health[LEFT] = Some(gripper_health(HealthLevel::Nominal, t0));
        // The gripper's window expires while the arm keeps reporting.
        let later = t0 + HEALTH_STALE_AFTER;
        s.arm_health[LEFT] = Some(arm_health(levels, later));
        let (view, arms) = health_panel_view(&s, later);
        assert_eq!(
            arms[LEFT].status,
            ArmHealthStatus::Live,
            "the arm still reports"
        );
        assert_eq!(arms[LEFT].worst, Some(HealthLevel::NotReporting.wire()));
        assert_eq!(
            problem_texts(&view),
            ["left_gripper not reporting", "left_arm FAULT"],
            "silence sorts above the spoken fault"
        );
        let gripper_row = &arms[LEFT].motors[ARM_DOF];
        assert_eq!(gripper_row.level, Some(HealthLevel::NotReporting.wire()));
        assert_eq!(
            gripper_row.effort_fraction_rated, None,
            "a quiet component's stale readings do not render as current"
        );
    }

    #[test]
    fn within_grace_a_silent_launch_reads_pending_not_a_problem() {
        let s = bound_state();
        let (view, arms) =
            health_panel_view(&s, s.created_at + STARTUP_GRACE - Duration::from_millis(1));
        assert_eq!(arms[LEFT].status, ArmHealthStatus::Pending);
        assert_eq!(arms[RIGHT].status, ArmHealthStatus::Pending);
        assert_eq!(view.worst, None);
        assert!(
            view.problems.is_empty(),
            "a healthy launch must not flash a false alarm"
        );
    }

    #[test]
    fn bound_but_silent_past_grace_raises_not_reporting() {
        let s = bound_state();
        let (view, arms) = health_panel_view(&s, s.created_at + STARTUP_GRACE);
        assert_eq!(arms[LEFT].status, ArmHealthStatus::NotReporting);
        assert_eq!(arms[LEFT].worst, Some(HealthLevel::NotReporting.wire()));
        assert_eq!(view.worst, Some(HealthLevel::NotReporting.wire()));
        assert_eq!(
            problem_texts(&view),
            [
                "left_arm not reporting",
                "left_gripper not reporting",
                "right_arm not reporting",
                "right_gripper not reporting",
            ]
        );
    }

    #[test]
    fn gripper_silence_after_first_report_is_a_problem_even_inside_grace() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.gripper_health[LEFT] = Some(gripper_health(HealthLevel::Nominal, t0));
        // Its window expires before the startup grace does; having reported
        // once, the gripper must escalate rather than vanish.
        let now = t0 + HEALTH_STALE_AFTER;
        assert!(now < t0 + STARTUP_GRACE, "the test must land inside grace");
        let (view, arms) = health_panel_view(&s, now);
        assert_eq!(problem_texts(&view), ["left_gripper not reporting"]);
        assert_eq!(arms[LEFT].status, ArmHealthStatus::NotReporting);
        assert_eq!(
            arms[RIGHT].status,
            ArmHealthStatus::Pending,
            "the never-reported arm stays pending"
        );
    }

    #[test]
    fn health_ages_out_for_arm_and_gripper_instead_of_latching() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.arm_health[LEFT] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[LEFT] = Some(gripper_health(HealthLevel::Nominal, t0));
        let (live, live_arms) =
            health_panel_view(&s, t0 + HEALTH_STALE_AFTER - Duration::from_millis(1));
        assert_eq!(live_arms[LEFT].status, ArmHealthStatus::Live);
        assert!(live.problems.is_empty());
        let (aged, aged_arms) = health_panel_view(&s, t0 + HEALTH_STALE_AFTER);
        assert_eq!(aged_arms[LEFT].status, ArmHealthStatus::NotReporting);
        assert_eq!(
            problem_texts(&aged),
            ["left_arm not reporting", "left_gripper not reporting"]
        );
    }

    #[test]
    fn statuses_and_phases_serialize_to_the_words_the_browser_reads() {
        for (status, word) in [
            (ArmHealthStatus::Live, "live"),
            (ArmHealthStatus::Pending, "pending"),
            (ArmHealthStatus::NotReporting, "not_reporting"),
            (ArmHealthStatus::NotWired, "not_wired"),
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
        s.arm_health[LEFT] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[LEFT] = Some(gripper_health(HealthLevel::Fault, t0));
        s.apply_alert_set(
            left_arm(),
            alert_set(vec![alert("left_arm j2", AlertSeverity::Critical)], t0),
        );
        let json = build_snapshot_json(&s, t0, &models, registry).unwrap();
        let snap: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(
            keys(&snap),
            [
                "alerts",
                "alerts_bound",
                "arms",
                "collision_enabled",
                "d_safe",
                "d_stop",
                "gesture",
                "gestures",
                "health",
                "max_gripper_rate_frac_s",
                "max_stream_ee_velocity_m_s",
                "proximity",
                "recorder",
                "status",
            ]
        );
        let arms = snap["arms"].as_array().expect("the arms in place order");
        let names: Vec<(&str, &str)> = arms
            .iter()
            .map(|entry| {
                (
                    entry["name"].as_str().expect("a name"),
                    entry["gripper_name"].as_str().expect("a gripper name"),
                )
            })
            .collect();
        assert_eq!(
            names,
            PerArm::<(), ARMS>::ids()
                .map(|arm| (robot::LIMBS.arm_name(arm), robot::LIMBS.gripper_name(arm)))
                .collect::<Vec<_>>()
        );
        let left = &arms[LEFT.place()];
        assert_eq!(
            keys(left),
            [
                "arm",
                "enabled",
                "gripper",
                "gripper_name",
                "health",
                "name"
            ]
        );
        assert_eq!(
            keys(&left["arm"]),
            [
                "feedback",
                "home",
                "in_flight",
                "joints",
                "limits",
                "orientation",
                "orientation_feedback",
                "pos_bounds",
                "pose",
                "pose_feedback",
                "ready",
            ]
        );
        let health = &snap["health"];
        assert_eq!(keys(health), ["bound", "problems", "worst"]);
        assert_eq!(health["bound"], serde_json::json!(true));
        assert_eq!(health["worst"], serde_json::json!(3));
        assert_eq!(keys(&left["health"]), ["motors", "status", "worst"]);
        assert_eq!(left["health"]["status"], "live");
        assert_eq!(arms[RIGHT.place()]["health"]["status"], "pending");
        let row = &left["health"]["motors"][0];
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
        assert_eq!(left["health"]["motors"][7]["name"], "gripper");
        let problem = &health["problems"][0];
        assert_eq!(keys(problem), ["level", "text"]);
        assert_eq!(problem["level"], serde_json::json!(3));
        assert_eq!(problem["text"], "left_gripper FAULT");
        let wave = snap["gestures"]
            .as_array()
            .expect("the roster")
            .iter()
            .find(|g| g["name"] == "wave")
            .expect("the wave");
        assert_eq!(keys(wave), ["arms", "label", "name"]);
        assert_eq!(wave["arms"], serde_json::json!(["right_arm"]));
        let alert_view = &snap["alerts"][0];
        assert_eq!(keys(alert_view), ["message", "severity", "source"]);
    }
}
