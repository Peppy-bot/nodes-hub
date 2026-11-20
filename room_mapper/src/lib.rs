//! room_mapper: maps the rooms a mobile base drives in with its 2-D lidar, saves and loads
//! the maps, and keeps the base localized in the loaded one: it implements room_map:v1 on
//! mapping_core. See README.md.

#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex};

use control_core::fault::Fault;
use mapping_core::MapStore;
use peppygen::{NodeRunner, Parameters, Result};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;

mod clock;
mod config;
mod geometry;
mod grids;
mod inputs;
mod lidar;
mod listeners;
mod messages;
mod odometry;
mod rooms;
mod serve;

use config::Settings;
use inputs::Inputs;
use rooms::members::Members;
use rooms::state::RoomState;
use rooms::worker;

/// Starts the node: its settings, the room over the saved maps, the services of room_map,
/// and its tasks, the listeners, the worker and the grid's publisher, any of which ending
/// stops the node.
pub async fn setup(params: Parameters, runner: Arc<NodeRunner>, fault: Fault) -> Result<()> {
    let settings = Settings::parse(&params).map_err(|e| peppygen::Error::Node(e.into()))?;
    let store =
        MapStore::open(&settings.maps_directory).map_err(|e| peppygen::Error::Node(e.into()))?;
    let clock = clock::node_clock(&runner, fault.clone()).await?;
    let inputs = Arc::new(Inputs::new(clock));
    let room = Arc::new(Mutex::new(RoomState::new(settings.clone(), store)));
    let (map_changes, changes) = watch::channel(());
    serve::serve(
        &runner,
        Members::new(
            room.clone(),
            inputs.clone(),
            settings.source_timeout,
            map_changes,
        ),
    );
    let (scans, arrived) = mpsc::unbounded_channel();
    let mut tasks = JoinSet::new();
    tasks.spawn(listeners::follow_base(
        runner.clone(),
        inputs.clone(),
        settings.footprint_padding,
        settings.source_timeout,
    ));
    tasks.spawn(listeners::follow_lidar(
        runner.clone(),
        inputs.clone(),
        scans,
        fault.clone(),
    ));
    tasks.spawn(worker::run(
        runner.clone(),
        room.clone(),
        inputs.clone(),
        arrived,
        settings.source_timeout,
    ));
    tasks.spawn(grids::run(
        runner.clone(),
        room,
        inputs,
        changes,
        settings.map_update_interval,
    ));
    tokio::spawn(supervise(tasks, runner, fault));
    Ok(())
}

/// Stops the node when the first of `tasks` ends, raising `fault` with why.
async fn supervise(mut tasks: JoinSet<Result<()>>, runner: Arc<NodeRunner>, fault: Fault) {
    let stopping = runner.cancellation_token().clone();
    let why = tokio::select! {
        () = stopping.cancelled() => return,
        ended = tasks.join_next() => match ended {
            Some(Ok(Ok(()))) => "a task of the node ended: its input closed".to_string(),
            Some(Ok(Err(error))) => format!("a task of the node failed: {error}"),
            Some(Err(error)) => format!("a task of the node panicked: {error}"),
            None => return,
        },
    };
    fault.raise(why, || stopping.cancel());
}
