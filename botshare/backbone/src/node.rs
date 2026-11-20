//! Node composition: parse every parameter up front, start the robot's tree and serve
//! describe_base; once the robot's initializer names the robot's model on robot_identity
//! and serves its description, and reports the robot ready on robot_ready, spawn the
//! control loop, the stream listeners, the action handlers and the robot stop for that
//! model's base. All motion lives in the sibling modules; this is only the assembly.

use std::sync::Arc;
use std::time::SystemTime;

use control_core::fault::Fault;
use control_core::robot_tree::RobotTree;
use frame_tree::FrameName;
use peppygen::{NodeRunner, Parameters, Result};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tracing::{error, info};

use crate::actions;
use crate::clock;
use crate::config::{BaseConfig, BaseParameters, ConfigError};
use crate::control::{self, Base, Publishers};
use crate::follow::follow_tree;
use crate::identity;
use crate::model::ModelError;
use crate::move_slot::MoveSlot;
use crate::placement::Placer;
use crate::services;
use crate::startup;
use crate::status::BaseStatus;
use crate::streams;
use crate::transforms::{ODOMETRY_FRAME, ROBOT_FRAME};

/// Everything this node refuses to run on: a parameter, by name, or the
/// runtime's own failure, passed through unchanged. A refusal `setup`
/// returns runs the runtime's shutdown hooks.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error(transparent)]
    Config(#[from] ConfigError),

    #[error("the model the robot's initializer serves")]
    Model(#[from] ModelError),

    #[error(
        "the robot's initializer serves the description {served} of {model}, and this \
         backbone was built with {built}: build the backbone and the initializer from one \
         public-peppy-libs commit"
    )]
    OtherDescription {
        model: &'static str,
        served: String,
        built: String,
    },

    #[error(transparent)]
    Runtime(#[from] peppygen::Error),
}

impl From<NodeError> for peppygen::Error {
    /// A runtime error passes back unchanged; a refusal of this node travels
    /// as `Error::Node`, with the parameter error reachable through
    /// `Error::source`.
    fn from(e: NodeError) -> Self {
        match e {
            NodeError::Runtime(e) => e,
            other => peppygen::Error::Node(Box::new(other)),
        }
    }
}

/// Spawn a task meant to run for the life of the node into the supervised
/// set, so its exit takes the node down like any other task's.
fn spawn_supervised<F>(set: &mut JoinSet<Result<()>>, task: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    set.spawn(async move {
        task.await;
        Ok(())
    });
}

/// The node's whole runtime, in the shape `NodeBuilder::run` and the test harness take
/// once `fault` is bound: parse the parameters, start the robot's tree and serve
/// describe_base's design answer, and, on a task of its own, ask the robot's initializer
/// for the model and run the base for it. The initializer may start after this node, so
/// `setup` returns without its answer, and a robot this node cannot run raises `fault`,
/// as does the end of a simulation clock's ticks.
pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>, fault: Fault) -> Result<()> {
    assemble(params, node_runner, fault)
        .await
        .map_err(Into::into)
}

async fn assemble(
    params: Parameters,
    runner: Arc<NodeRunner>,
    fault: Fault,
) -> std::result::Result<(), NodeError> {
    // Every time the node reads, the setpoints' receipt and the steps of the
    // limiter, is this instance's bound clock.
    peppygen::clock::init(&runner).await?;
    let started = SystemTime::now();
    let parameters = BaseParameters::parse(&params)?;
    // The robot's tree, on the same clock, filled from the start, so a goal's
    // placement finds every transform published since.
    let tree_clock = clock::node_clock(&runner, fault.clone()).await?;
    let tree = Arc::new(
        RobotTree::new(parameters.cache_window, tree_clock)
            .expect("duration_from_secs gives a window above 0"),
    );
    follow_tree(&runner, tree.clone()).await?;
    let publishers = Publishers::declare(&runner, tree.clone()).await?;
    let token = runner.cancellation_token().clone();
    let run = run_once_identified(runner, parameters, tree, publishers, fault.clone(), started);
    tokio::spawn(
        fault.raise_on_panic("the start for the robot's model", run, move || {
            token.cancel()
        }),
    );
    Ok(())
}

/// Asks the robot's initializer for the model until it answers, then runs the base for
/// it. A model whose URDF gives no differential base, or another description of it,
/// raises `fault`. The node `started` on the wall clock.
async fn run_once_identified(
    runner: Arc<NodeRunner>,
    parameters: BaseParameters,
    tree: Arc<RobotTree>,
    publishers: Publishers,
    fault: Fault,
    started: SystemTime,
) {
    let token = runner.cancellation_token().clone();
    let Some(identified) = identity::identify(&runner, &parameters).await else {
        return;
    };
    let config = match identified {
        Ok(config) => Arc::new(config),
        Err(refusal) => return fault.raise(refusal, || token.cancel()),
    };
    info!(
        "differential base: {:?}, steps every {} ns",
        config.wheels, config.step_ns
    );
    let placer = Arc::new(Placer::new(
        tree.clone(),
        FrameName::parse(ROBOT_FRAME).expect("base_footprint is a frame name"),
        FrameName::parse(ODOMETRY_FRAME).expect("odom is a frame name"),
        parameters.lookup_timeout,
    ));
    // What the base is, answered for the life of the node from its parameters and its
    // model alone, so ahead of the readiness gate.
    tokio::spawn(services::serve_describe_base(
        runner.clone(),
        config.clone(),
    ));
    startup::wait_until_ready(&runner, &token).await;
    if token.is_cancelled() {
        return;
    }
    run(runner, config, placer, publishers, started).await;
}

/// The control loop, the stream listeners, the action handlers and the robot stop of the
/// base `config` gives, each meant to run for the life of the node: the first to end
/// ends the node.
async fn run(
    runner: Arc<NodeRunner>,
    config: Arc<BaseConfig>,
    placer: Arc<Placer>,
    publishers: Publishers,
    started: SystemTime,
) {
    let token = runner.cancellation_token().clone();
    let status = BaseStatus::default();
    let slot = MoveSlot::default();
    // One event waits for the control loop at a time: each source waits for
    // the loop to take its last event before it hands it the next.
    let (events_tx, events_rx) = mpsc::channel(1);
    let base = Base::new(
        config.clone(),
        status.clone(),
        slot.clone(),
        publishers,
        started,
    );

    let mut set = JoinSet::new();
    {
        let token = token.clone();
        spawn_supervised(&mut set, async move {
            control::run(base, events_rx, token.cancelled()).await;
        });
    }
    spawn_supervised(
        &mut set,
        streams::run_wheel_listener(runner.clone(), events_tx.clone()),
    );
    spawn_supervised(
        &mut set,
        streams::run_setpoint_listener(runner.clone(), placer.clone(), events_tx.clone()),
    );
    set.spawn(actions::run_drive_base(
        runner.clone(),
        config.clone(),
        placer.clone(),
        status.clone(),
        slot.clone(),
        events_tx.clone(),
    ));
    set.spawn(actions::run_move_base(
        runner.clone(),
        config.clone(),
        placer,
        status,
        slot,
        events_tx.clone(),
    ));
    spawn_supervised(
        &mut set,
        services::run_robot_stop(
            runner.clone(),
            token.clone(),
            events_tx,
            config.stop_answer_timeout,
        ),
    );

    // The first task to finish ends the node: it cancels the node, every
    // other task stops and the process exits, the log naming the task.
    if let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok(())) => error!("base task exited; shutting node down"),
            Ok(Err(e)) => error!(error = %e, "base task failed; shutting node down"),
            Err(e) => error!(error = %e, "base task ended abnormally; shutting node down"),
        }
    }
    token.cancel();
    set.shutdown().await;
}
