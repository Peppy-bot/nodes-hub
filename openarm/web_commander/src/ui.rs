// HTTP+WS UI on 0.0.0.0:PEPPY_JC_PORT (default 8765). The WS exposes
// unauthenticated motion control, so only run on a trusted network; set
// PEPPY_JC_BIND_IP=127.0.0.1 to restrict to loopback.
//
// This is only the transport: every text frame is decoded to a [`Command`] and sent to
// the state owner, and every snapshot the owner publishes is forwarded to the browser.
// The owner (see [`crate::owner`]) is the sole reader/writer of `UiState`.

use std::env;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use openarm_description::HardwareVersion;
use peppylib::runtime::CancellationToken;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};

use crate::gestures::Registry;
use crate::owner::UiMsg;
use crate::pose::{ArmModels, JogMode, Pose};

/// A second `init_limits` call, which would be a second generation's ranges
/// arriving after the first were already handed out.
#[derive(Debug, thiserror::Error)]
#[error("init_limits must run exactly once")]
pub struct LimitsAlreadySet;

/// Why the operator panel stopped serving. Both are `io::Error`, so each names
/// which of the two steps produced it rather than sharing one label.
#[derive(Debug, thiserror::Error)]
pub enum UiError {
    #[error("bind the commander UI to {addr}: {source}")]
    Bind {
        addr: SocketAddr,
        #[source]
        source: std::io::Error,
    },

    #[error("serve the commander UI: {0}")]
    Serve(#[source] std::io::Error),
}
use crate::state::{
    ARM_DOF, Alert, ArmTarget, Disposition, GesturePhase, GripperTarget, HealthLevel, HealthReport,
    Proximity, Side, UiState,
};

const DEFAULT_PORT: u16 = 8765;
// The backbone publishes the proximity readout at ~20 Hz; treat it as stale after this
// long with no update (a dead backbone) so the panel falls back to n/a instead of
// latching the last distance.
const PROXIMITY_STALE_AFTER: Duration = Duration::from_millis(500);
// A launch's producers need a beat to start reporting: until this long after
// state creation, a health component that has never reported renders as
// pending rather than raising a not-reporting problem, so a healthy launch
// does not flash a false alarm.
const STARTUP_GRACE: Duration = Duration::from_secs(3);

const INDEX_HTML: &str = include_str!("../static/index.html");

// Joint ranges from the generation's bundled URDF plus the unitless gripper
// opening range; the single source for slider bounds (via the WS snapshot) and
// for clamping incoming commands. Resolved once at startup by [`init_limits`].
#[derive(Clone, Copy)]
struct JointLimits {
    gripper: [f64; 2],
    left: [[f64; 2]; ARM_DOF],
    right: [[f64; 2]; ARM_DOF],
}

impl JointLimits {
    fn arm(&self, side: Side) -> &[[f64; 2]; ARM_DOF] {
        match side {
            Side::Left => &self.left,
            Side::Right => &self.right,
        }
    }

    fn resolve(version: HardwareVersion) -> Self {
        Self {
            gripper: [0.0, 1.0],
            left: version.joint_limits(openarm_description::Side::Left),
            right: version.joint_limits(openarm_description::Side::Right),
        }
    }
}

static LIMITS: std::sync::OnceLock<JointLimits> = std::sync::OnceLock::new();

/// Resolve the panel's clamp/display ranges from the generation's description:
/// arm joints via its `joint_limits` (URDF limits with the elbow held off its
/// singularity floor, matching the backbone's clamp); the gripper is the
/// unitless opening fraction [0, 1] on every generation. Must run before the
/// UI serves, and exactly once: a second call would be a second generation's
/// ranges arriving after the first were already handed out.
pub fn init_limits(version: HardwareVersion) -> std::result::Result<(), LimitsAlreadySet> {
    LIMITS
        .set(JointLimits::resolve(version))
        .map_err(|_| LimitsAlreadySet)
}

fn joint_limits() -> &'static JointLimits {
    LIMITS
        .get()
        .expect("init_limits must run before the UI serves")
}

/// The gripper opening range `[closed, open]` as a fraction of full jaw travel;
/// the owner clamps gripper commands into it, the same single source the
/// sliders bound against.
pub(crate) fn gripper_limits() -> [f64; 2] {
    joint_limits().gripper
}

#[derive(Clone)]
struct AppState {
    // Operator input to the owner: decoded commands and the disconnect signal.
    command_tx: mpsc::Sender<UiMsg>,
    // The owner's latest pre-serialized snapshot; forwarded verbatim to the browser.
    snapshot_rx: watch::Receiver<String>,
    token: CancellationToken,
}

pub async fn run(
    command_tx: mpsc::Sender<UiMsg>,
    snapshot_rx: watch::Receiver<String>,
    token: CancellationToken,
) -> std::result::Result<(), UiError> {
    let port = env::var("PEPPY_JC_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(DEFAULT_PORT);
    let bind_ip = env::var("PEPPY_JC_BIND_IP")
        .ok()
        .and_then(|s| s.parse::<IpAddr>().ok())
        .unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
    let addr = SocketAddr::new(bind_ip, port);

    let app_state = AppState {
        command_tx,
        snapshot_rx,
        token: token.clone(),
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/ws", get(ws_upgrade))
        .with_state(app_state);

    let listener = TcpListener::bind(addr)
        .await
        .map_err(|source| UiError::Bind { addr, source })?;
    info!("commander UI at http://localhost:{port}");

    let shutdown_token = token.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown_token.cancelled().await })
        .await
        .map_err(UiError::Serve)?;
    Ok(())
}

async fn index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(app): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_handle(socket, app))
}

async fn ws_handle(mut socket: WebSocket, app: AppState) {
    let mut snapshots = app.snapshot_rx.clone();
    // Send the latest snapshot immediately so a fresh connection paints at once, then
    // follow the owner's updates. `borrow_and_update` marks it seen, so `changed` next
    // waits for the following snapshot.
    let initial = snapshots.borrow_and_update().clone();
    if !initial.is_empty()
        && socket
            .send(Message::Text(Utf8Bytes::from(initial)))
            .await
            .is_err()
    {
        return;
    }
    loop {
        tokio::select! {
            _ = app.token.cancelled() => break,
            changed = snapshots.changed() => {
                if changed.is_err() {
                    break; // the owner is gone
                }
                let json = snapshots.borrow_and_update().clone();
                if !json.is_empty()
                    && socket.send(Message::Text(Utf8Bytes::from(json))).await.is_err()
                {
                    break;
                }
            }
            msg = socket.recv() => match msg {
                Some(Ok(Message::Text(text))) => match serde_json::from_str::<Command>(text.as_str()) {
                    Ok(cmd) => { let _ = app.command_tx.send(UiMsg::Command(cmd)).await; }
                    // Payload withheld: it carries operator free text (task
                    // names) and arbitrary bytes from the wire.
                    Err(e) => warn!(error = %e, payload_bytes = text.len(), "ws: bad command"),
                },
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(e)) => { warn!(error = %e, "ws: recv"); break; }
                _ => {}
            }
        }
    }
    // Releasing the panel drops the deadman and restores the governor default.
    let _ = app.command_tx.send(UiMsg::Disconnect).await;
}

// A governor band the UI may stream: all finite and positive, with d_stop below
// d_safe. The backbone validates again before applying.
pub(crate) fn valid_governor_band(d_stop: f64, d_safe: f64, max_ee_velocity_m_s: f64) -> bool {
    [d_stop, d_safe, max_ee_velocity_m_s]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
        && d_stop < d_safe
}

// Clamp each joint setpoint into its configured [min, max]. The single clamp
// path for every operator-driven arm command; the arm clamps again on its side.
pub(crate) fn clamp_to_limits(joints: &mut [f64; ARM_DOF], side: Side) {
    for (j, &[lo, hi]) in joints.iter_mut().zip(joint_limits().arm(side).iter()) {
        *j = j.clamp(lo, hi);
    }
}

// Clamp a requested move duration to a sane range (finite, 0..=30 s). 0 = fastest.
pub(crate) fn sane_duration(duration_s: f64) -> f64 {
    if duration_s.is_finite() {
        duration_s.clamp(0.0, 30.0)
    } else {
        0.0
    }
}

// A discrete move's duration: the operator's request, floored so the straight-line
// EE speed never exceeds the governor cap (time >= distance / cap). The backbone floors
// again at its joint-velocity limit, so this only ever slows a move, never speeds it.
pub(crate) fn ee_speed_floored(user_s: f64, ee_distance_m: f64, max_ee_velocity_m_s: f64) -> f64 {
    (ee_distance_m / max_ee_velocity_m_s).max(user_s)
}

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

// --------------------------- wire protocol ---------------------------

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
    collision_enabled: bool,
    d_stop: f64,
    d_safe: f64,
    max_ee_velocity_m_s: f64,
    max_gripper_rate_frac_s: f64,
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
    // "lead_in" while blending from the held target, then "playing".
    phase: &'static str,
    // Fraction of the current phase completed, 0..1.
    progress: f64,
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
    // Alert severity: 1 warning, 2 critical, 3 fault (0 never renders; a
    // clear removes its entry instead).
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
    side: &'static str,
    // The wire level behind the problem; the banner colors on the first
    // (most severe) entry.
    level: u8,
    text: String,
}

/// One side's health surfaces: its render state, the combined worst across
/// arm joints and gripper, and one row per motor (j1..j7 then the gripper).
#[derive(Serialize)]
struct SideHealthView {
    // "live" (rows are current), "pending" (nothing yet, inside the startup
    // grace), "not_reporting" (bound but quiet), or "not_wired".
    status: &'static str,
    // Null while pending or unwired; a quiet side reads NotReporting.
    worst: Option<u8>,
    // Empty unless live.
    motors: Vec<MotorRowView>,
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
    joints: [f64; ARM_DOF],
    feedback: Option<[f64; ARM_DOF]>,
    in_flight: bool,
    // Per-joint [min, max] (rad); the browser bounds its sliders with these.
    limits: [[f64; 2]; ARM_DOF],
    // The side's Ready pose, the single definition the gesture library anchors
    // on; the browser's Ready Pose button fires these joints.
    ready: [f64; ARM_DOF],
    // The side's Home pose, from the same canonical postures; the browser's
    // Home Pose button fires these joints.
    home: [f64; ARM_DOF],
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
    // Arm angle psi (elbow swivel, rad) of the target and the measured pose; `null` at
    // the straight-arm singularity (kept off by the elbow floor). Drives the elbow slider.
    arm_angle: Option<f64>,
    arm_angle_feedback: Option<f64>,
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
                    phase: "lead_in",
                    progress: (t / duration_s).clamp(0.0, 1.0),
                },
                GesturePhase::Playing { t } => GestureStatusView {
                    name: pb.gesture.name,
                    phase: "playing",
                    progress: (t / pb.gesture.duration_s).clamp(0.0, 1.0),
                },
            }),
            left_arm: arm_view(&s.arms[Side::Left], Side::Left, models),
            right_arm: arm_view(&s.arms[Side::Right], Side::Right, models),
            left_gripper: gripper_view(&s.grippers[Side::Left]),
            right_gripper: gripper_view(&s.grippers[Side::Right]),
            left_enabled: s.enabled[Side::Left],
            right_enabled: s.enabled[Side::Right],
            collision_enabled: s.collision_enabled,
            d_stop: s.d_stop,
            d_safe: s.d_safe,
            max_ee_velocity_m_s: s.max_ee_velocity_m_s,
            max_gripper_rate_frac_s: s.max_gripper_rate_frac_s,
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
            alerts: live_alerts(s, now),
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

/// The still-live alerts, most severe first (receipt order within a
/// severity); entries whose producer stopped re-emitting age out.
fn live_alerts(s: &UiState, now: Instant) -> Vec<AlertView> {
    let mut live: Vec<&Alert> = s
        .alerts
        .iter()
        .filter(|a| a.validity.is_live_at(now))
        .collect();
    live.sort_by_key(|a| std::cmp::Reverse(a.severity));
    live.into_iter()
        .map(|a| AlertView {
            source: a.source.clone(),
            severity: a.severity,
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
    part: &'static str,
) -> (Option<HealthLevel>, Option<ProblemView>) {
    match state {
        ComponentState::Live(report) => {
            let worst = report.worst();
            let problem = (worst >= HealthLevel::Critical).then(|| ProblemView {
                side: side.label(),
                level: worst.wire(),
                text: format!("{} {part} {}", side.label(), level_label(worst)),
            });
            (Some(worst), problem)
        }
        ComponentState::NotReporting => (
            Some(HealthLevel::NotReporting),
            Some(ProblemView {
                side: side.label(),
                level: HealthLevel::NotReporting.wire(),
                text: format!("{} {part} not reporting", side.label()),
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
            status: "not_wired",
            worst: None,
            motors: Vec::new(),
        };
        return (view, Vec::new());
    }
    let arm = component_state(&s.health[side], now, in_grace);
    let gripper = component_state(&s.gripper_health[side], now, in_grace);
    let live = matches!(arm, ComponentState::Live(_)) || matches!(gripper, ComponentState::Live(_));
    let (arm_worst, arm_problem) = component_severity(&arm, side, "arm");
    let (gripper_worst, gripper_problem) = component_severity(&gripper, side, "gripper");
    let worst = [arm_worst, gripper_worst].into_iter().flatten().max();
    let motors = if live {
        let mut rows = component_rows(&arm, |i| format!("j{}", i + 1));
        rows.extend(component_rows(&gripper, |_| "gripper".to_string()));
        rows
    } else {
        Vec::new()
    };
    let status = match (live, worst) {
        (true, _) => "live",
        (false, Some(_)) => "not_reporting",
        (false, None) => "pending",
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
        limits: *joint_limits().arm(side),
        ready: crate::gestures::READY[side],
        home: crate::gestures::HOME[side],
        pos_bounds: models.pos_bounds(side),
        pose: models.ee_pose_world(side, &a.joints),
        pose_feedback: a.last_feedback.map(|fb| models.ee_pose_world(side, &fb)),
        orientation: models.ee_quat_world(side, &a.joints),
        orientation_feedback: a.last_feedback.map(|fb| models.ee_quat_world(side, &fb)),
        arm_angle: models.arm_angle(side, &a.joints),
        arm_angle_feedback: a.last_feedback.and_then(|fb| models.arm_angle(side, &fb)),
    }
}

fn gripper_view(g: &GripperTarget) -> GripperView {
    let [min, max] = joint_limits().gripper;
    GripperView {
        position: g.position,
        feedback: g.last_feedback,
        min,
        max,
        max_effort: g.max_effort,
        effort_ceiling: g.effort_ceiling,
        in_flight: g.in_flight,
    }
}

#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub(crate) enum Command {
    FireArm {
        side: SideWire,
        joints: [f64; ARM_DOF],
        // Requested move duration (s); 0 = fastest safe.
        duration_s: f64,
    },
    // Toggle the streaming deadman for one side. While enabled, the command stream
    // emits that side's arm target and gripper opening on its pairing slots;
    // while disabled both track the measured pose and emit nothing.
    SetEnabled {
        side: SideWire,
        on: bool,
    },
    // Update an enabled arm's streamed target. Ignored while disabled, where the
    // target follows the measured pose so enabling never steps the arm.
    SetArmTarget {
        side: SideWire,
        joints: [f64; ARM_DOF],
    },
    // Arm a jog: position (metres) + orientation quaternion [x, y, z, w] + arm angle
    // (elbow swivel, rad), plus which one the jog drives (`mode` = the touched control;
    // the rest is held). The command stream walks the joint target toward it and holds
    // at the boundary. Ignored while disabled, like set_arm_target.
    SetArmPose {
        side: SideWire,
        position: [f64; 3],
        orientation: [f64; 4],
        arm_angle: f64,
        mode: JogModeWire,
    },
    // Fire the backbone's planned Cartesian move_arm to a composed world-frame pose
    // (Actions-mode Execute): a governed straight-line move, not a jog. Refused
    // while the side streams, like fire_arm.
    FireArmPose {
        side: SideWire,
        position: [f64; 3],
        orientation: [f64; 4],
        // Requested move duration (s); 0 = fastest safe.
        duration_s: f64,
    },
    // Update an enabled gripper's streamed opening. Ignored while disabled.
    SetGripperTarget {
        side: SideWire,
        position: f64,
    },
    // Set the gripper's effort cap, applied to both the streamed opening and
    // discrete moves. Not deadman-gated: Actions mode fires while disabled.
    SetGripperEffort {
        side: SideWire,
        max_effort: f64,
    },
    // Fire the backbone's discrete move_gripper (Actions-mode gripper Execute): a governed
    // open/close to `position` (m), not the streamed opening. Refused while the side
    // streams or a prior gripper move is in flight.
    FireGripper {
        side: SideWire,
        position: f64,
    },
    // Set the backbone's self-collision-avoidance toggle (streamed continuously).
    SetCollision {
        enabled: bool,
    },
    // Retune the backbone's governor band and stream speed cap (streamed continuously).
    SetGovernorParams {
        d_stop: f64,
        d_safe: f64,
        max_ee_velocity_m_s: f64,
        max_gripper_rate_frac_s: f64,
    },
    // Play a named gesture from the baked library. Refused while its sides
    // stream, a move is in flight, or another gesture is playing.
    RunGesture {
        name: String,
    },
    // Stop the playing gesture; the involved sides hold where they are.
    StopGesture,
    // Start recording a dataset episode labeled with the operator's task text.
    // Refused when no recorder is bound or an episode is already recording.
    StartRecording {
        task: String,
    },
    // End the in-flight episode with a save (the recorder's cancel semantics).
    StopRecording,
    // Finalize and mirror the current session's dataset and open a fresh one;
    // the finished directory becomes replayable. Refused while recording.
    FinishSession,
}

#[derive(Deserialize, Copy, Clone)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SideWire {
    Left,
    Right,
}

impl From<SideWire> for Side {
    fn from(s: SideWire) -> Self {
        match s {
            SideWire::Left => Side::Left,
            SideWire::Right => Side::Right,
        }
    }
}

// Which component a jog drives, as sent by the panel: "position" from the x/y/z
// sliders, "orientation" from the arcball, or "arm_angle" from the elbow slider.
#[derive(Deserialize, Copy, Clone)]
#[serde(rename_all = "lowercase")]
pub(crate) enum JogModeWire {
    Position,
    Orientation,
    #[serde(rename = "arm_angle")]
    ArmAngle,
}

impl From<JogModeWire> for JogMode {
    fn from(m: JogModeWire) -> Self {
        match m {
            JogModeWire::Position => JogMode::Position,
            JogModeWire::Orientation => JogMode::Orientation,
            JogModeWire::ArmAngle => JogMode::ArmAngle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{
        ArmHealth, GripperHealth, HEALTH_STALE_AFTER, MotorHealthReading, Validity,
    };

    /// Tests have no main() to run init_limits, so resolve the v2 limits on
    /// first use; concurrent tests settle benignly through get_or_init.
    fn init_limits_for_tests() {
        LIMITS.get_or_init(|| JointLimits::resolve(HardwareVersion::V2));
    }

    fn ui_state() -> UiState {
        UiState::new(true, 0.005, 0.02, 0.25, 6.0, 10.0)
    }

    /// A state with health producers bound, as every wired deployment has.
    fn bound_state() -> UiState {
        let mut s = ui_state();
        s.health_bound = true;
        s
    }

    const TEST_VALID_FOR: Duration = Duration::from_secs(5);

    fn alert(source: &str, severity: u8, received_at: Instant) -> Alert {
        alert_from("core/left_arm_inst", source, severity, received_at)
    }

    fn alert_from(producer: &str, source: &str, severity: u8, received_at: Instant) -> Alert {
        Alert {
            producer: producer.to_string(),
            source: source.to_string(),
            kind: "motor_overload".to_string(),
            severity,
            message: "holding 93% of rated torque".to_string(),
            validity: Validity::new(received_at, TEST_VALID_FOR),
        }
    }

    fn reading(level: HealthLevel) -> MotorHealthReading {
        MotorHealthReading {
            level,
            effort_fraction_rated: Some(0.4),
            effort_fraction_rated_sustained: Some(0.3),
            effort_fraction_peak: Some(0.2),
            driver_temp_c: Some(41.0),
            winding_temp_c: Some(37.0),
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
    fn alerts_upsert_by_identity_and_clear_on_severity_zero() {
        let mut s = ui_state();
        let t0 = Instant::now();
        s.apply_alert(alert("left arm j2", 1, t0));
        s.apply_alert(alert("left arm j2", 2, t0));
        assert_eq!(s.alerts.len(), 1, "one entry per (producer, source, kind)");
        assert_eq!(s.alerts[0].severity, 2);
        s.apply_alert(alert("left arm j2", 0, t0));
        assert!(s.alerts.is_empty(), "severity 0 clears the entry");
    }

    #[test]
    fn a_producer_cannot_replace_or_clear_anothers_alert() {
        let mut s = ui_state();
        let t0 = Instant::now();
        s.apply_alert(alert_from("core/left_arm_inst", "left arm j2", 2, t0));
        s.apply_alert(alert_from("core/imposter", "left arm j2", 1, t0));
        assert_eq!(s.alerts.len(), 2, "same wire strings, distinct producers");
        s.apply_alert(alert_from("core/imposter", "left arm j2", 0, t0));
        assert_eq!(s.alerts.len(), 1, "the clear removed only its own entry");
        assert_eq!(s.alerts[0].producer, "core/left_arm_inst");
        assert_eq!(s.alerts[0].severity, 2);
    }

    #[test]
    fn a_clear_for_an_unknown_alert_is_a_no_op() {
        let mut s = ui_state();
        let t0 = Instant::now();
        s.apply_alert(alert("right arm j1", 2, t0));
        s.apply_alert(alert("left arm j2", 0, t0));
        assert_eq!(s.alerts.len(), 1, "an unknown clear removes nothing");
        assert_eq!(s.alerts[0].source, "right arm j1");
    }

    #[test]
    fn alerts_purge_entries_whose_producer_went_quiet() {
        let mut s = ui_state();
        let t0 = Instant::now();
        s.apply_alert(alert("left arm j2", 1, t0));
        s.apply_alert(alert("right arm j1", 2, t0 + TEST_VALID_FOR));
        assert_eq!(s.alerts.len(), 1, "the quiet producer's entry is purged");
        assert_eq!(s.alerts[0].source, "right arm j1");
    }

    #[test]
    fn live_alerts_age_out_and_sort_most_severe_first() {
        let mut s = ui_state();
        let t0 = Instant::now();
        s.apply_alert(alert("left arm j2", 1, t0));
        s.apply_alert(alert("right arm j1", 3, t0));
        let views = live_alerts(&s, t0 + Duration::from_secs(1));
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].source, "right arm j1", "most severe first");
        assert_eq!(views[1].source, "left arm j2");
        assert!(
            live_alerts(&s, t0 + TEST_VALID_FOR).is_empty(),
            "a quiet producer's alerts stop rendering"
        );
    }

    #[test]
    fn unwired_health_renders_not_wired_everywhere() {
        let s = ui_state();
        let view = health_panel_view(&s, s.created_at);
        assert!(!view.bound);
        assert_eq!(view.worst, None);
        assert!(view.problems.is_empty());
        assert_eq!(view.left.status, "not_wired");
        assert_eq!(view.right.status, "not_wired");
        assert!(view.left.motors.is_empty());
    }

    #[test]
    fn per_side_worst_combines_gripper_severity() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.health[Side::Left] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Fault, t0));
        let view = health_panel_view(&s, t0);
        assert_eq!(view.left.status, "live");
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
        s.health[Side::Left] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
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
        s.health[Side::Left] = Some(arm_health(levels, t0));
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Nominal, t0));
        // The gripper's window expires while the arm keeps reporting.
        let later = t0 + HEALTH_STALE_AFTER;
        s.health[Side::Left] = Some(arm_health(levels, later));
        let view = health_panel_view(&s, later);
        assert_eq!(view.left.status, "live", "the arm still reports");
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
        assert_eq!(view.left.status, "pending");
        assert_eq!(view.right.status, "pending");
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
        assert_eq!(view.left.status, "not_reporting");
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
        assert_eq!(view.left.status, "not_reporting");
        assert_eq!(
            view.right.status, "pending",
            "the never-reported side stays pending"
        );
    }

    #[test]
    fn health_ages_out_for_arm_and_gripper_instead_of_latching() {
        let mut s = bound_state();
        let t0 = s.created_at;
        s.health[Side::Left] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Nominal, t0));
        let live = health_panel_view(&s, t0 + HEALTH_STALE_AFTER - Duration::from_millis(1));
        assert_eq!(live.left.status, "live");
        assert!(live.problems.is_empty());
        let aged = health_panel_view(&s, t0 + HEALTH_STALE_AFTER);
        assert_eq!(aged.left.status, "not_reporting");
        assert_eq!(
            problem_texts(&aged),
            ["left arm not reporting", "left gripper not reporting"]
        );
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
        let models = ArmModels::from_version(HardwareVersion::V2);
        let registry = Registry::bake(&models);
        let mut s = bound_state();
        s.alerts_bound = true;
        let t0 = s.created_at;
        s.health[Side::Left] = Some(arm_health([HealthLevel::Nominal; ARM_DOF], t0));
        s.gripper_health[Side::Left] = Some(gripper_health(HealthLevel::Fault, t0));
        s.apply_alert(alert("left arm j2", 2, t0));
        let json = build_snapshot_json(&s, t0, &models, &registry).unwrap();
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

    #[test]
    fn clamp_pins_each_joint_into_its_range() {
        init_limits_for_tests();
        for side in [Side::Left, Side::Right] {
            let limits = joint_limits().arm(side);

            let mut high = [f64::INFINITY; ARM_DOF];
            clamp_to_limits(&mut high, side);
            for (v, &[_, hi]) in high.iter().zip(limits.iter()) {
                assert_eq!(*v, hi);
            }

            let mut low = [f64::NEG_INFINITY; ARM_DOF];
            clamp_to_limits(&mut low, side);
            for (v, &[lo, _]) in low.iter().zip(limits.iter()) {
                assert_eq!(*v, lo);
            }
        }
    }

    #[test]
    fn clamp_leaves_in_range_values_untouched() {
        init_limits_for_tests();
        for side in [Side::Left, Side::Right] {
            let limits = joint_limits().arm(side);
            let mut mid = [0.0; ARM_DOF];
            for (m, &[lo, hi]) in mid.iter_mut().zip(limits.iter()) {
                *m = (lo + hi) / 2.0;
            }
            let before = mid;
            clamp_to_limits(&mut mid, side);
            assert_eq!(mid, before);
        }
    }

    #[test]
    fn valid_governor_band_boundaries() {
        assert!(valid_governor_band(0.005, 0.02, 1.0));
        assert!(
            !valid_governor_band(0.02, 0.02, 1.0),
            "d_stop == d_safe is degenerate"
        );
        assert!(
            !valid_governor_band(0.03, 0.02, 1.0),
            "d_stop > d_safe is inverted"
        );
        assert!(!valid_governor_band(0.0, 0.02, 1.0), "non-positive d_stop");
        assert!(
            !valid_governor_band(0.005, 0.02, 0.0),
            "non-positive speed cap"
        );
        assert!(
            !valid_governor_band(f64::NAN, 0.02, 1.0),
            "non-finite d_stop"
        );
    }

    #[test]
    fn config_joint_limits_are_well_formed() {
        init_limits_for_tests();
        // Each range must be non-empty so clamp and the slider bounds are valid.
        for side in [Side::Left, Side::Right] {
            for &[lo, hi] in joint_limits().arm(side).iter() {
                assert!(lo < hi, "joint range [{lo}, {hi}] must be non-empty");
            }
        }
        let [lo, hi] = joint_limits().gripper;
        assert!(lo < hi, "gripper range [{lo}, {hi}] must be non-empty");
    }
}
