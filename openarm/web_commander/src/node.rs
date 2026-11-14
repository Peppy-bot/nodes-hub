//! Node composition: setup parses every parameter up front and binds the panel,
//! then, once the robot's initializer names the model, the node bakes the arm models
//! and gesture registry and wires the state owner, feedback streams, command
//! publisher, and UI server together. All the real logic lives in the sibling
//! modules; this is only the assembly.

use std::net::{AddrParseError, IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use control_core::fault::Fault;
use control_core::positive_finite::{NotPositiveFinite, PositiveFinite};
use control_core::time::{RateOutOfRange, period_from_hz};
use peppygen::{EndpointBinding, NodeRunner, Parameters, Result};
use tokio::sync::{mpsc, watch};
use tracing::info;

use crate::alerts;
use crate::collision_status;
use crate::command_stream;
use crate::gestures;
use crate::gripper_states;
use crate::identity;
use crate::joint_states;
use crate::limits;
use crate::motor_health;
use crate::owner;
use crate::pose;
use crate::robot;
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

/// Everything this node refuses to run on, or stops for: a parameter or a panel
/// address, which `setup` refuses, and the robot its initializer names or a panel
/// that stops, which end the node with a [`Fault`] after `setup` has returned.
///
/// Named rather than stringly typed so each refusal keeps its source, and so
/// this list is the one place to read what a launch can be rejected for. It
/// exists because returning a refusal, rather than panicking it, is what runs
/// the runtime's shutdown hooks: a panic unwinds straight past them.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("parameter command_rate_hz")]
    CommandRate(#[source] RateOutOfRange),

    #[error("parameter http_host must be an IP address, not {value:?}")]
    HttpHost {
        value: String,
        #[source]
        source: AddrParseError,
    },

    #[error("parameter http_port must name a port to serve on, not 0")]
    HttpPort,

    #[error("parameter max_stream_ee_velocity_m_s")]
    EeVelocity(#[source] NotPositiveFinite),

    #[error("parameter joint_jog_acceleration_rad_s2")]
    JogAcceleration(#[source] NotPositiveFinite),

    #[error("parameter joint_jog_velocity_rad_s")]
    JogVelocity(#[source] NotPositiveFinite),

    #[error("parameter max_gripper_rate_frac_s")]
    GripperRate(#[source] NotPositiveFinite),

    #[error("the model the robot's initializer serves")]
    Model(#[source] robot_description::UndrivenModel),

    #[error(transparent)]
    Limits(#[from] limits::LimitsAlreadySet),

    #[error("the operator panel stopped")]
    PanelStopped(#[source] ui::UiError),

    #[error(transparent)]
    Panel(#[from] ui::UiError),

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

/// The node's entry point, in the shape `NodeBuilder::run` (and the test harness) take
/// once `fault` is bound: parse every parameter and bind the panel, then, on a task of
/// its own, ask the robot's initializer for the model and run the panel for it. The
/// initializer may start after this node, so `setup` returns without its answer, and a
/// model the panel does not drive, or a panel that stops, raises `fault`.
pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>, fault: Fault) -> Result<()> {
    assemble(params, node_runner, fault)
        .await
        .map_err(Into::into)
}

/// What the launch parameters resolve to, each parsed in `setup`, before the robot's
/// model is known.
struct Launch {
    command_period: Duration,
    governor: state::GovernorSettings,
    jog: pose::JogCaps,
}

impl Launch {
    /// Parses every parameter the panel runs on, refusing the first it cannot.
    fn parse(params: &Parameters) -> NodeResult<Self> {
        // The operator streams the governor controls live; their launch defaults are
        // node parameters, kept in step with the backbone's, so the real arm starts
        // conservative (tight band, slow cap) and the sim launchers start fast.
        let max_ee_velocity = PositiveFinite::parse(params.max_stream_ee_velocity_m_s)
            .map_err(NodeError::EeVelocity)?;
        let jog_acceleration = PositiveFinite::parse(params.joint_jog_acceleration_rad_s2)
            .map_err(NodeError::JogAcceleration)?;
        let jog_velocity = PositiveFinite::parse(params.joint_jog_velocity_rad_s)
            .map_err(NodeError::JogVelocity)?;
        // The panel streams this rate to the backbone, which refuses the same
        // values; refusing here names the launcher parameter instead of surfacing
        // as a rejected frame after the node is already up.
        let gripper_rate = PositiveFinite::parse(params.max_gripper_rate_frac_s)
            .map_err(NodeError::GripperRate)?;
        // The panel and the backbone's governor must reject the same bands, or the
        // operator can stream one the backbone will not take.
        if !state::valid_governor_band(params.d_stop, params.d_safe) {
            return Err(NodeError::GovernorBand {
                d_stop: params.d_stop,
                d_safe: params.d_safe,
            });
        }
        Ok(Self {
            command_period: period_from_hz(params.command_rate_hz, MAX_RATE_HZ)
                .map_err(NodeError::CommandRate)?,
            governor: state::GovernorSettings {
                collision_enabled: params.collision_governor_enabled,
                d_stop: params.d_stop,
                d_safe: params.d_safe,
                max_stream_ee_velocity_m_s: max_ee_velocity.get(),
                max_gripper_rate_frac_s: gripper_rate.get(),
            },
            jog: pose::JogCaps {
                joint_accel_rad_s2: jog_acceleration.get(),
                joint_velocity_rad_s: jog_velocity.get(),
            },
        })
    }
}

async fn assemble(params: Parameters, node_runner: Arc<NodeRunner>, fault: Fault) -> NodeResult {
    // Pairing timestamps read this instance's bound clock, so the backbone
    // ages setpoints on one timeline.
    peppygen::clock::init(&node_runner).await?;
    let launch = Launch::parse(&params)?;

    // The panel's socket, owned before anything is spawned: a launch this
    // node cannot serve is refused here, while the daemon is still waiting for
    // setup. The launcher's port is preferred; a copy whose port another
    // process holds serves on one the operating system picks.
    let panel = ui::Panel::bind(panel_address(&params)?).await?;
    // The daemon renders the URLs an operator opens from this announcement,
    // one per address of the machine, so the fallback port reaches the
    // operator too.
    node_runner.announce_endpoint(
        "panel",
        EndpointBinding {
            scheme: "http".to_string(),
            address: panel.address(),
            path: String::new(),
        },
    )?;
    info!("operator panel bound at {}", panel.address());
    let token = node_runner.cancellation_token().clone();
    let run = run_once_identified(node_runner, launch, panel, fault.clone());
    tokio::spawn(
        fault.raise_on_panic("the start for the robot's model", run, move || {
            token.cancel()
        }),
    );
    Ok(())
}

/// Asks the robot's initializer for the model until it answers, then runs the panel
/// for it. A model the panel does not drive raises `fault`.
async fn run_once_identified(
    node_runner: Arc<NodeRunner>,
    launch: Launch,
    panel: ui::Panel,
    fault: Fault,
) {
    let Some(model) = identity::model(&node_runner).await else {
        return;
    };
    let token = node_runner.cancellation_token().clone();
    if let Err(refusal) = run(node_runner, launch, panel, &model, fault.clone()) {
        fault.raise(refusal, || token.cancel());
    }
}

/// Bakes the arm models and the gesture registry of `model`, then wires the state
/// owner, feedback streams, command publisher, and the panel's server together. A
/// panel that stops raises `fault`.
fn run(
    node_runner: Arc<NodeRunner>,
    launch: Launch,
    panel: ui::Panel,
    model: &str,
    fault: Fault,
) -> NodeResult {
    let token = node_runner.cancellation_token().clone();
    // The model's description gives the panel its arms' joint ranges (URDF
    // limits) and postures; the gripper axis is the unitless opening fraction
    // and everything else in the commander is model-blind.
    let driven = robot::LIMBS.model(model).map_err(NodeError::Model)?;
    info!("driving {model}");
    limits::init_limits(driven.joint_limits())?;
    // Arm models for the panel's Cartesian pose fields (pose <-> joints), built
    // from the same description the ranges came from so FK/IK match the backbone's chain.
    let models = pose::ArmModels::from_driven(&driven);
    // Bake the gesture roster against those models and the ranges the panel
    // clamps into: every gesture is resolved to a feasible joint trajectory
    // here or the node aborts bringup.
    let registry = gestures::Registry::bake(&models, limits::arm_ranges());
    let state = state::UiState::new(launch.governor, launch.jog);
    let command_period = launch.command_period;

    // The state owner is the one task that touches UiState; everything else holds a
    // channel end. Commands flow in from the WS, feedback in from the state streams
    // and the goal tasks, and the owner publishes the browser snapshot and the
    // per-tick command frame the publishers stream.
    let (command_tx, command_rx) = mpsc::channel::<owner::UiMsg>(COMMAND_CAP);
    let (feedback_tx, feedback_rx) = mpsc::channel::<owner::Feedback>(FEEDBACK_CAP);
    let (frame_tx, frame_rx) = watch::channel(command_stream::CommandFrame::from_state(&state));
    let (snapshot_tx, snapshot_rx) = watch::channel(String::new());

    // Feed the owner live arm, gripper, proximity, motor health and alert state off
    // the always-on streams.
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

    // The always-on publisher: streams each enabled arm's governed setpoint from
    // the owner's command frame at command_rate_hz. A disabled arm has None in the
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

    // Panel::serve is the long-lived HTTP + WebSocket server, on a task of its own.
    // A commander without its panel is not degraded, it is dead: nothing can
    // reach the owner. Raise the fault, which cancels the node so the daemon
    // restarts it, instead of standing ready with nothing listening.
    tokio::spawn(async move {
        if let Err(e) = panel.serve(command_tx, snapshot_rx, token.clone()).await {
            fault.raise(NodeError::PanelStopped(e), || token.cancel());
        }
    });
    Ok(())
}

/// The panel's preferred listen address. Port 0 is refused: the launcher names
/// the port operators are sent to, and [`ui::Panel::bind`] takes one from the
/// operating system only when that port is already held.
fn panel_address(params: &Parameters) -> NodeResult<SocketAddr> {
    let host: IpAddr = params
        .http_host
        .parse()
        .map_err(|source| NodeError::HttpHost {
            value: params.http_host.clone(),
            source,
        })?;
    (params.http_port != ui::ANY_PORT)
        .then(|| SocketAddr::new(host, params.http_port))
        .ok_or(NodeError::HttpPort)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::driven;

    /// A launch the node accepts, for the cases that spoil one field of it.
    fn params() -> Parameters {
        Parameters {
            collision_governor_enabled: true,
            command_rate_hz: 50,
            d_safe: 0.02,
            d_stop: 0.005,
            http_host: "0.0.0.0".to_string(),
            http_port: 8765,
            joint_jog_acceleration_rad_s2: 10.0,
            joint_jog_velocity_rad_s: 1.0,
            max_stream_ee_velocity_m_s: 0.5,
            max_gripper_rate_frac_s: 6.0,
        }
    }

    #[test]
    fn the_panel_prefers_the_launcher_s_address() {
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
        // Port 0 asks for an operating-system port on every launch. The node
        // takes one only when the launcher's port is already held, and logs
        // where it landed.
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
        // The panel hands out one description's ranges for the process; a second
        // call would mean a second description arriving after the first were read.
        // Sibling tests resolve the same process-wide ranges, so this call may
        // be the one that wins or one that already lost; either way the ranges
        // are resolved afterwards and the next call must be refused.
        limits::init_limits(driven("openarm_v2").joint_limits()).ok();
        assert!(
            limits::init_limits(driven("openarm_v1").joint_limits()).is_err(),
            "a second init_limits must be refused, not accepted silently"
        );
    }

    #[test]
    fn both_openarm_models_drive_this_node_and_another_is_refused_by_name() {
        for model in ["openarm_v1", "openarm_v2"] {
            robot::LIMBS
                .model(model)
                .unwrap_or_else(|error| panic!("{model}: {error}"));
        }
        let refused = robot::LIMBS
            .model("so101")
            .map_err(NodeError::Model)
            .expect_err("the SO-101 is not two arms");
        assert_eq!(
            refused.to_string(),
            "the model the robot's initializer serves"
        );
        let reason = std::error::Error::source(&refused).expect("the reason");
        assert!(
            reason.to_string().starts_with("model so101 describes"),
            "{reason}"
        );
    }
}
