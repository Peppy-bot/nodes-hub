//! lerobot_recorder: a robot-agnostic dataset recorder. It discovers the
//! producers bound to its cardinality slots, samples them onto a fixed fps
//! grid, and writes a LeRobot v3 dataset per session (optionally mirrored to
//! S3/R2). Binding order defines the dataset layout, so nothing here is
//! robot-specific.
//!
//! Task graph: drain tasks feed a latest-value snapshot cache (never any I/O
//! on a drain path: a blocked subscription callback freezes every subscription
//! in the node); the episode manager samples the cache at fps and feeds the
//! LeRobot writer on its own blocking thread; a storage task mirrors finalized
//! files. First supervised task to exit cancels the rest and the process.

mod config;
mod control;
mod episode;
mod lerobot_sink;
mod listeners;
mod plan;
mod provenance;
mod snapshot;
mod status;
mod storage;
mod types;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use peppygen::{NodeBuilder, Parameters, Result};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tracing::{error, info};

use crate::config::Config;
use crate::episode::Manager;
use crate::listeners::CameraRoute;
use crate::plan::RecordingPlan;
use crate::provenance::{ProducerLog, Session};
use crate::status::Status;
use crate::types::{FrameBuf, ProducerKey};

fn session_dir_name(now: SystemTime) -> String {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .expect("system clock after 1970")
        .as_secs();
    chrono::DateTime::from_timestamp(secs as i64, 0)
        .expect("unix seconds are in range")
        .format("%Y-%m-%d_%H-%M-%S")
        .to_string()
}

/// Moves the camera slots named by `index` out of `pool` into a per-producer
/// route map. Each slot belongs to exactly one route.
fn take_route(
    pool: &mut HashMap<usize, snapshot::Slot<Arc<FrameBuf>>>,
    index: &HashMap<ProducerKey, usize>,
    forced_depth: bool,
) -> CameraRoute {
    let by_producer = index
        .iter()
        .filter_map(|(key, slot)| pool.remove(slot).map(|s| (key.clone(), s)))
        .collect();
    CameraRoute {
        by_producer,
        forced_depth,
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    NodeBuilder::new().run(|params: Parameters, runner| async move {
        // The synchronized clock every staleness gate compares producer stamps
        // against (sim time under a simulated clock).
        peppygen::clock::init(&runner).await?;

        let config = Config::parse(&params).expect("invalid launch parameters");
        let session_dir = config.output_root.join(session_dir_name(SystemTime::now()));
        std::fs::create_dir_all(&session_dir).expect("output_root must be writable");
        let dataset_dir = session_dir.join(&config.dataset_name);
        info!("recording session at {}", session_dir.display());

        let token = runner.cancellation_token().clone();
        let plan = Arc::new(RecordingPlan::discover(&runner, &config));
        info!(
            "discovered {} state, {} action sources, {} camera features",
            plan.state_keys.len(),
            plan.action_keys.len(),
            plan.camera_count()
        );

        let producer_log = ProducerLog::new();
        let (cache_writer, cache_reader) = snapshot::cache(&plan);
        let (sink, sink_rx) = lerobot_sink::channel(&config);
        let (storage_tx, storage_rx) = storage::channel();
        let (status_tx, status_rx) = watch::channel(Status::initial());
        let (command_tx, command_rx) = mpsc::channel(4);

        let sink_thread = {
            let config = config.clone();
            let storage_tx = storage_tx.clone();
            tokio::task::spawn_blocking(move || {
                lerobot_sink::run(config, dataset_dir, sink_rx, storage_tx)
            })
        };

        let snapshot::CacheWriter {
            states,
            actions,
            cameras,
        } = cache_writer;
        let mut camera_pool: HashMap<usize, _> = cameras.into_iter().enumerate().collect();
        let color_route = take_route(&mut camera_pool, &plan.color_index, false);
        let rgbd_color_route = take_route(&mut camera_pool, &plan.rgbd_color_index, false);
        let rgbd_depth_route = take_route(&mut camera_pool, &plan.rgbd_depth_index, true);
        let depth_route = take_route(&mut camera_pool, &plan.depth_index, true);
        debug_assert!(
            camera_pool.is_empty(),
            "every camera slot is claimed by exactly one route"
        );

        let mut tasks: JoinSet<&'static str> = JoinSet::new();
        macro_rules! supervised {
            ($name:literal, $future:expr) => {{
                let future = $future;
                tasks.spawn(async move {
                    future.await;
                    $name
                });
            }};
        }
        supervised!(
            "state_sources",
            listeners::state_sources(
                runner.clone(),
                plan.state_index.clone(),
                states,
                producer_log.clone(),
                token.clone()
            )
        );
        supervised!(
            "action_sources",
            listeners::action_sources(
                runner.clone(),
                plan.action_index.clone(),
                actions,
                producer_log.clone(),
                token.clone()
            )
        );
        supervised!(
            "color_cameras",
            listeners::color_cameras(
                runner.clone(),
                color_route,
                producer_log.clone(),
                token.clone()
            )
        );
        supervised!(
            "rgbd_color",
            listeners::rgbd_color(
                runner.clone(),
                rgbd_color_route,
                producer_log.clone(),
                token.clone()
            )
        );
        supervised!(
            "rgbd_depth",
            listeners::rgbd_depth(
                runner.clone(),
                rgbd_depth_route,
                producer_log.clone(),
                token.clone()
            )
        );
        supervised!(
            "depth_cameras",
            listeners::depth_cameras(
                runner.clone(),
                depth_route,
                producer_log.clone(),
                token.clone()
            )
        );
        supervised!(
            "start_episode",
            control::start_service(runner.clone(), command_tx.clone(), token.clone())
        );
        supervised!(
            "stop_episode",
            control::stop_service(runner.clone(), command_tx.clone(), token.clone())
        );
        supervised!(
            "recorder_status",
            status::run(runner.clone(), config.clone(), status_rx, token.clone())
        );
        supervised!(
            "session_json",
            Session::new(&session_dir, params).run(producer_log.clone(), token.clone())
        );
        supervised!(
            "storage",
            storage::run(
                config.storage.clone(),
                session_dir.clone(),
                config.dataset_name.clone(),
                storage_rx,
                token.clone()
            )
        );
        supervised!(
            "episode_manager",
            Manager::new(config, plan.clone(), cache_reader, sink, status_tx)
                .run(command_rx, token.clone())
        );

        tokio::spawn(async move {
            let outcome = tasks.join_next().await;
            // A task that finishes while the token is already cancelled is
            // stopping in response to an external shutdown, not failing.
            let clean = token.is_cancelled() && matches!(outcome, Some(Ok(_)));
            match &outcome {
                Some(Ok(name)) if clean => info!("task {name} stopped on shutdown"),
                Some(Ok(name)) => error!("task {name} exited unexpectedly; shutting down"),
                Some(Err(e)) => error!("supervised task panicked: {e}; shutting down"),
                None => error!("no supervised tasks to join"),
            }
            token.cancel();
            let graceful = tokio::time::timeout(std::time::Duration::from_secs(120), async {
                while tasks.join_next().await.is_some() {}
            })
            .await;
            if graceful.is_err() {
                error!("tasks did not stop within 120 s; aborting them");
                tasks.shutdown().await;
            }
            drop(storage_tx);
            drop(command_tx);
            if let Err(e) = sink_thread.await {
                error!("lerobot writer thread panicked: {e}");
                std::process::exit(1);
            }
            if clean {
                info!("shutdown complete");
                std::process::exit(0);
            }
            std::process::exit(1);
        });
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_dir_names_are_utc_and_sortable() {
        let name = session_dir_name(UNIX_EPOCH + std::time::Duration::from_secs(1_751_500_800));
        assert_eq!(name, "2025-07-03_00-00-00");
        let later = session_dir_name(UNIX_EPOCH + std::time::Duration::from_secs(1_751_500_861));
        assert!(later > name);
    }
}
