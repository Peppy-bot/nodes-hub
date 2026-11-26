//! Node composition: parse every parameter, pair the cameras of the two camera
//! slots by name, start the robot's tree on the node's clock, then spawn the
//! control loop, the drive_states listener, describe_base's reader, the
//! cameras' descriptions and each camera's frame intake, the depth listener,
//! the four action handlers and clear_grid. Every behavior lives in the sibling
//! modules; this is only the assembly. The setup waits for no base and no
//! camera: until they answer, a goal waits up to the source timeout and is
//! refused with `resource_unavailable`, naming what it waits for.

use std::sync::{Arc, Mutex};

use control_core::fault::Fault;
use control_core::robot_tree::{Placer, RobotTree};
use peppygen::consumed_services::cameras::depth_stream_info;
use peppygen::consumed_services::geometry::get_depth_intrinsics;
use peppygen::paired_topics::base::drive_setpoints;
use peppygen::{NodeRunner, Parameters, Result};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tracing::{error, info, warn};

use crate::actions;
use crate::admission::{Admission, Shared};
use crate::base::{self, BaseReads};
use crate::cameras::{self, Sensors};
use crate::clock;
use crate::config::{ConfigError, NavigatorConfig};
use crate::control::{self, Navigator};
use crate::follow::{follow_tree, robot_frame};
use crate::grid::GridSlot;
use crate::intake::{self, Intake};
use crate::link::{self, Link};
use crate::planning::{Planner, Planning};
use crate::services;

/// Everything this node refuses to run on, by name, or the runtime's own
/// failure, passed through unchanged. A refusal `setup` returns runs the
/// runtime's shutdown hooks.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error(transparent)]
    Config(#[from] ConfigError),

    #[error("the camera slots: {0}")]
    Cameras(String),

    #[error(transparent)]
    Runtime(#[from] peppygen::Error),
}

impl From<NodeError> for peppygen::Error {
    /// A runtime error passes back unchanged; a refusal of this node travels
    /// as `Error::Node`, with its cause reachable through `Error::source`.
    fn from(e: NodeError) -> Self {
        match e {
            NodeError::Runtime(e) => e,
            other => peppygen::Error::Node(Box::new(other)),
        }
    }
}

/// Spawn a task meant to run for the life of the node into the supervised
/// set, so its end takes the node down like any other task's.
fn spawn_supervised<F>(set: &mut JoinSet<Result<()>>, task: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    set.spawn(async move {
        task.await;
        Ok(())
    });
}

/// The node's whole runtime, in the shape `NodeBuilder::run` and the test
/// harness take once `fault` is bound: [`setup_with`] a planner that runs
/// each plan at once.
pub async fn setup(params: Parameters, runner: Arc<NodeRunner>, fault: Fault) -> Result<()> {
    setup_with(params, runner, fault, Planner::default()).await
}

/// The node's whole runtime with `planner` running its plans: parse the
/// parameters, start the robot's tree, and spawn every task. The end of a
/// simulation clock's ticks raises `fault`.
pub async fn setup_with(
    params: Parameters,
    runner: Arc<NodeRunner>,
    fault: Fault,
    planner: Planner,
) -> Result<()> {
    assemble(params, runner, fault, planner)
        .await
        .map_err(Into::into)
}

async fn assemble(
    params: Parameters,
    runner: Arc<NodeRunner>,
    fault: Fault,
    planner: Planner,
) -> std::result::Result<(), NodeError> {
    peppygen::clock::init(&runner).await?;
    let config = Arc::new(NavigatorConfig::parse(&params)?);
    if let Some(warning) = config.approach_warning() {
        warn!("{warning}");
    }
    let slots = cameras::pair_slots(
        depth_stream_info::bound_members(&runner).as_slice(),
        get_depth_intrinsics::bound_members(&runner).as_slice(),
    )
    .map_err(NodeError::Cameras)?;
    let sensors = Arc::new(Sensors::new(slots));
    info!(
        "base navigator: its grid reads the cameras {:?}, on a grid of {} m and {} m voxels",
        sensors.names(),
        config.grid.side_m.get(),
        config.grid.voxel_m.get()
    );

    // Every time the node reads is this instance's bound clock, and the
    // robot's tree, filled from the start, waits on it.
    let clock = clock::node_clock(&runner, fault).await?;
    let tree = Arc::new(
        RobotTree::new(config.cache_window, clock.clone())
            .expect("duration_parameter gives a window above 0"),
    );
    follow_tree(&runner, tree.clone()).await?;
    let placer = Arc::new(Placer::new(
        tree.clone(),
        robot_frame(),
        config.lookup_timeout,
    ));
    let link = Arc::new(Mutex::new(Link::default()));
    let grid = Arc::new(Mutex::new(GridSlot::default()));
    let (starts_tx, starts_rx) = watch::channel(None);
    let (reads_tx, reads_rx) = watch::channel(None);
    // One event waits for the control loop at a time: each source waits for
    // the loop to take its last event before it hands it the next.
    let (events_tx, events_rx) = mpsc::channel(1);
    let shared = Shared {
        config: config.clone(),
        clock: clock.clone(),
        link: link.clone(),
        grid: grid.clone(),
        sensors: sensors.clone(),
    };
    let navigator = Navigator::new(shared.clone(), Planning::new(planner, config.planner));
    let setpoints = drive_setpoints::declare_publisher(&runner).await?;
    let admission = Admission {
        shared,
        base_reads: BaseReads {
            starts: starts_rx.clone(),
            reads: reads_rx,
            clock: clock.clone(),
        },
        placer,
    };

    let token = runner.cancellation_token().clone();
    let mut set = JoinSet::new();
    {
        let token = token.clone();
        spawn_supervised(&mut set, async move {
            control::run(navigator, setpoints, events_rx, token.cancelled()).await;
        });
    }
    spawn_supervised(
        &mut set,
        link::run_listener(runner.clone(), clock.clone(), link, grid.clone(), starts_tx),
    );
    spawn_supervised(
        &mut set,
        base::run_reader(
            runner.clone(),
            config.clone(),
            sensors.count(),
            starts_rx,
            reads_tx,
            grid.clone(),
        ),
    );
    // The descriptions end once every camera is described or refused, so
    // they run outside the supervised set; they end with the node too.
    tokio::spawn(cameras::run_descriptions(runner.clone(), sensors.clone()));
    let mut frame_senders = Vec::new();
    for source in 0..sensors.count().get() {
        let (frames_tx, frames_rx) = watch::channel(None);
        frame_senders.push(frames_tx);
        let intake = Intake {
            source,
            sensors: sensors.clone(),
            tree: tree.clone(),
            grid: grid.clone(),
            lookup_timeout: config.lookup_timeout,
        };
        spawn_supervised(&mut set, intake.run(frames_rx));
    }
    spawn_supervised(
        &mut set,
        intake::run_listener(runner.clone(), clock, sensors, frame_senders),
    );
    set.spawn(actions::run_navigate_to_pose(
        runner.clone(),
        admission.clone(),
        events_tx.clone(),
    ));
    set.spawn(actions::run_spin(
        runner.clone(),
        admission.clone(),
        events_tx.clone(),
    ));
    set.spawn(actions::run_drive_on_heading(
        runner.clone(),
        admission.clone(),
        events_tx.clone(),
    ));
    set.spawn(actions::run_back_up(
        runner.clone(),
        admission,
        events_tx.clone(),
    ));
    spawn_supervised(&mut set, services::serve_clear_grid(runner, events_tx));

    // The first task to end ends the node, and the daemon starts a clean
    // process with every task.
    tokio::spawn(async move {
        if let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(())) => error!("navigator task exited; shutting node down"),
                Ok(Err(e)) => error!(error = %e, "navigator task failed; shutting node down"),
                Err(e) => error!(error = %e, "navigator task ended abnormally; shutting node down"),
            }
        }
        token.cancel();
        set.shutdown().await;
    });
    Ok(())
}
