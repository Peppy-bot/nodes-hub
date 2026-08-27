// Node composition: parses every parameter up front, bakes the arm models and
// gesture registry, then wires the state owner, feedback streams, command
// publisher, and UI server together. All the real logic lives in the sibling
// modules; this is only the assembly.

use std::net::{AddrParseError, IpAddr, SocketAddr};
use std::sync::Arc;

use control_core::positive_finite::{NotPositiveFinite, PositiveFinite};
use control_core::time::{RateOutOfRange, period_from_hz};
use openarm_description::HardwareVersion;
use peppygen::{NodeRunner, Parameters, Result};
use tokio::sync::{mpsc, watch};
use tracing::error;

use crate::alerts;
use crate::collision_status;
use crate::command_stream;
use crate::gestures;
use crate::gripper_states;
use crate::joint_states;
use crate::motor_health;
use crate::owner;
use crate::pose;
use crate::state;
use crate::ui;

// Channel depths: commands are operator-paced (small); feedback bursts across the
// state streams and goal tasks (larger), but the owner drains both far faster than
// they fill.
const COMMAND_CAP: usize = 64;
const FEEDBACK_CAP: usize = 256;

/// The fastest the panel streams command frames. The backbone consuming them
/// runs no faster than 1 kHz.
const MAX_RATE_HZ: u32 = 1_000;

/// The UI fault that stopped this node, if one did; read by `main` through
/// [`ui_failed`] after the runtime returns.
static UI_FAILED: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Everything this node refuses to run on.
///
/// Named rather than stringly typed so each refusal keeps its source, and so
/// this list is the one place to read what a launch can be rejected for. It
/// exists because returning a refusal, rather than panicking it, is what runs
/// the runtime's shutdown hooks: a panic in `setup` unwinds straight past them.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("parameter command_rate_hz")]
    CommandRate(#[source] RateOutOfRange),

    #[error("parameter http_host is not an IP address")]
    HttpHost(#[source] AddrParseError),

    #[error("parameter http_port must name a port to serve on, not 0")]
    HttpPort,

    #[error("parameter max_ee_velocity_m_s")]
    EeVelocity(#[source] NotPositiveFinite),

    #[error("parameter joint_jog_acceleration_rad_s2")]
    JogAcceleration(#[source] NotPositiveFinite),

    #[error("parameter max_gripper_rate_frac_s")]
    GripperRate(#[source] NotPositiveFinite),

    #[error(transparent)]
    HardwareVersion(#[from] openarm_description::UnknownHardwareVersion),

    #[error(transparent)]
    Limits(#[from] ui::LimitsAlreadySet),

    #[error("the operator panel stopped: {0}")]
    Ui(&'static str),

    #[error("governor band must satisfy 0 < d_stop ({d_stop}) < d_safe ({d_safe}), both finite")]
    GovernorBand { d_stop: f64, d_safe: f64 },

    /// The runtime's own failures pass through unchanged rather than being
    /// re-wrapped, so a messaging or config error keeps the variant it was
    /// raised as.
    #[error(transparent)]
    Runtime(#[from] peppygen::Error),
}

/// This node's own results, distinct from `peppygen::Result`, which the runtime
/// takes at the boundary and which is what bare `Result` means in this crate.
type NodeResult<T = ()> = std::result::Result<T, NodeError>;

impl From<NodeError> for peppygen::Error {
    /// The one place this node's refusals meet the runtime's error type.
    ///
    /// A runtime error passes back unchanged; everything else is this node's own
    /// and travels as `Error::Node`, which keeps the wrapped error reachable
    /// through `Error::source` rather than flattening it to a message.
    fn from(e: NodeError) -> Self {
        match e {
            NodeError::Runtime(e) => e,
            other => peppygen::Error::Node(Box::new(other)),
        }
    }
}

/// The UI fault that stopped this node, if one did; read by `main` after the
/// runtime returns, so a dead panel is recorded as a failure, not a finish.
pub fn ui_failed() -> Option<&'static str> {
    UI_FAILED.get().map(|s| s.as_str())
}

pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    assemble(params, node_runner).await.map_err(Into::into)
}

async fn assemble(params: Parameters, node_runner: Arc<NodeRunner>) -> NodeResult {
    // Pairing timestamps read the daemon-resolved clock (sim time under a
    // simulated clock), so the backbone ages setpoints on one timeline.
    peppygen::clock::init(&node_runner).await?;
    let token = node_runner.cancellation_token().clone();
    // The generation picks the panel's arm joint ranges (URDF limits); the
    // gripper axis is the unitless opening fraction and everything else in
    // the commander is version-blind.
    let version: HardwareVersion = params.hardware_version.parse()?;
    ui::init_limits(version)?;
    // Arm models for the panel's Cartesian pose fields (pose <-> joints), built
    // from the same generation the ranges came from so FK/IK match the backbone's chain.
    let models = pose::ArmModels::from_version(version);
    // Bake the gesture roster against those models: every gesture is resolved
    // to a feasible joint trajectory here or the node aborts bringup.
    let registry = gestures::Registry::bake(&models);
    // The operator streams the governor controls live; their launch defaults are
    // node parameters, kept in step with the backbone's, so the real arm starts
    // conservative (tight band, slow cap) and the sim launchers start fast.
    let max_ee_velocity =
        PositiveFinite::parse(params.max_ee_velocity_m_s).map_err(NodeError::EeVelocity)?;
    let jog_acceleration = PositiveFinite::parse(params.joint_jog_acceleration_rad_s2)
        .map_err(NodeError::JogAcceleration)?;
    // The panel streams this rate to the backbone, which refuses the same
    // values; refusing here names the launcher parameter instead of surfacing
    // as a rejected frame after the node is already up.
    let gripper_rate =
        PositiveFinite::parse(params.max_gripper_rate_frac_s).map_err(NodeError::GripperRate)?;
    // The panel and the backbone's governor must reject the same bands, or the
    // operator can stream one the backbone will not take.
    if !(params.d_stop.is_finite()
        && params.d_safe.is_finite()
        && params.d_stop > 0.0
        && params.d_stop < params.d_safe)
    {
        return Err(NodeError::GovernorBand {
            d_stop: params.d_stop,
            d_safe: params.d_safe,
        });
    }
    let state = state::UiState::new(
        params.collision_governor_enabled,
        params.d_stop,
        params.d_safe,
        max_ee_velocity.get(),
        gripper_rate.get(),
        jog_acceleration.get(),
    );

    let command_period =
        period_from_hz(params.command_rate_hz, MAX_RATE_HZ).map_err(NodeError::CommandRate)?;

    // Where the panel listens. Resolved before anything is spawned so a
    // mistyped address is a refusal naming the parameter, not a bind failure
    // from a task the daemon has already called ready.
    let panel_addr = panel_address(&params)?;

    // The state owner is the one task that touches UiState; everything else holds a
    // channel end. Commands flow in from the WS, feedback in from the state streams
    // and the goal tasks, and the owner publishes the browser snapshot and the
    // per-tick command frame the publishers stream.
    let (command_tx, command_rx) = mpsc::channel::<owner::UiMsg>(COMMAND_CAP);
    let (feedback_tx, feedback_rx) = mpsc::channel::<owner::Feedback>(FEEDBACK_CAP);
    let (frame_tx, frame_rx) = watch::channel(owner::CommandFrame::from_state(&state));
    let (snapshot_tx, snapshot_rx) = watch::channel(String::new());

    // Feed the owner live arm + gripper + proximity state off the always-on streams.
    tokio::spawn(joint_states::run(
        node_runner.clone(),
        feedback_tx.clone(),
        token.clone(),
    ));
    tokio::spawn(gripper_states::run(
        node_runner.clone(),
        feedback_tx.clone(),
        token.clone(),
    ));
    tokio::spawn(collision_status::run(
        node_runner.clone(),
        feedback_tx.clone(),
        token.clone(),
    ));
    tokio::spawn(motor_health::run(
        node_runner.clone(),
        feedback_tx.clone(),
        token.clone(),
    ));
    tokio::spawn(alerts::run(
        node_runner.clone(),
        feedback_tx.clone(),
        token.clone(),
    ));

    // The always-on publisher: streams each enabled side's governed setpoint from
    // the owner's command frame at command_rate_hz. A disabled side has None in the
    // frame, so nothing is published and the backbone holds its last setpoint.
    tokio::spawn(command_stream::run(
        node_runner.clone(),
        command_period,
        token.clone(),
        frame_rx,
    ));

    // The owner: advances jogs, reduces commands + feedback, and publishes both
    // watches. It owns `state` and `models` and holds the runner to spawn goal tasks.
    tokio::spawn(owner::run(
        state,
        models,
        registry,
        node_runner,
        command_period,
        token.clone(),
        owner::Channels {
            command_rx,
            feedback_rx,
            feedback_tx,
            frame_tx,
            snapshot_tx,
        },
    ));

    // ui::run is the long-lived HTTP + WebSocket server. It must be spawned rather
    // than awaited here: peppylib registers `node_health` only after the setup
    // closure returns, so awaiting a forever-task starves the health probe and the
    // daemon SIGKILLs the instance after ~10s.
    // A commander without its panel is not degraded, it is dead: nothing can
    // reach the owner. Record the fault and cancel the node so the daemon
    // restarts it, instead of standing ready with nothing listening.
    tokio::spawn(async move {
        if let Err(e) = ui::run(panel_addr, command_tx, snapshot_rx, token.clone()).await {
            error!("operator panel stopped: {e}; cancelling the node");
            let _ = UI_FAILED.set(e.to_string());
            token.cancel();
        }
    });
    Ok(())
}

/// The panel's listen address. Port 0 is refused: an operator panel on an
/// ephemeral port is one nobody can reach at the address they were given.
fn panel_address(params: &Parameters) -> NodeResult<SocketAddr> {
    let host: IpAddr = params.http_host.parse().map_err(NodeError::HttpHost)?;
    (params.http_port != 0)
        .then(|| SocketAddr::new(host, params.http_port))
        .ok_or(NodeError::HttpPort)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A launch the node accepts, for the cases that spoil one field of it.
    fn params() -> Parameters {
        Parameters {
            collision_governor_enabled: true,
            command_rate_hz: 50,
            d_safe: 0.02,
            d_stop: 0.005,
            hardware_version: "v2".to_string(),
            http_host: "0.0.0.0".to_string(),
            http_port: 8765,
            joint_jog_acceleration_rad_s2: 10.0,
            max_ee_velocity_m_s: 0.5,
            max_gripper_rate_frac_s: 6.0,
        }
    }

    #[test]
    fn the_panel_serves_the_launcher_s_address() {
        let addr = panel_address(&Parameters {
            http_host: "127.0.0.1".to_string(),
            http_port: 18765,
            ..params()
        })
        .expect("a host and port the launcher may set must be accepted");
        assert_eq!(addr.to_string(), "127.0.0.1:18765");
    }

    #[test]
    fn a_host_that_is_not_an_ip_is_refused_by_name() {
        // A hostname would resolve to any number of addresses, none of them a
        // choice this node gets to make.
        let refused = panel_address(&Parameters {
            http_host: "localhost".to_string(),
            ..params()
        })
        .expect_err("only an IP address can be bound");
        assert!(
            refused.to_string().contains("http_host"),
            "the refusal must name http_host, got: {refused}"
        );
    }

    #[test]
    fn port_zero_is_refused_by_name() {
        // Port 0 binds an ephemeral port: the panel would come up somewhere
        // the operator was never told about.
        let refused = panel_address(&Parameters {
            http_port: 0,
            ..params()
        })
        .expect_err("an ephemeral panel port must be refused");
        assert!(
            refused.to_string().contains("http_port"),
            "the refusal must name http_port, got: {refused}"
        );
    }

    #[test]
    fn init_limits_refuses_a_second_call() {
        // The panel hands out one generation's ranges for the process; a second
        // call would mean a second generation arriving after the first were read.
        // Sibling tests resolve the same process-wide ranges, so this call may
        // be the one that wins or one that already lost; either way the ranges
        // are resolved afterwards and the next call must be refused.
        ui::init_limits(HardwareVersion::V2).ok();
        assert!(
            ui::init_limits(HardwareVersion::V1).is_err(),
            "a second init_limits must be refused, not accepted silently"
        );
    }
}
