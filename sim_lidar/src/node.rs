//! The relay loop and its assembly: the simulation's scans forward to the
//! contract surface, each in the lidar's frame that sensor_mounts gives, and
//! `setup` reads the lidar, spawns the loop and flags its death.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use camera_core::sensor_mounts::{
    DESCRIBE_SENSORS_TIMEOUT, DescribedSensors, LIDAR_2D, MountedSensor, check_relayed_frame,
    read_sensor,
};
use control_core::latched_log::LatchedLog;
use control_core::receiving::receive;
use peppygen::consumed_services::sensor_mounts::describe_sensors;
use peppygen::emitted_topics::lidar::scan_stream as lidar_scans;
use peppygen::paired_topics::simulation::scan_stream as simulation_scans;
use peppygen::{NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;
use tracing::{error, info};

/// Set when the relay leg ends on its own. `setup` returns as soon as the
/// leg is spawned, so the binary reads this flag after the node stops and
/// exits with a failure, which the stack shows as a relay that died.
static LEG_DIED: AtomicBool = AtomicBool::new(false);

/// Whether the relay leg ended on its own.
pub fn leg_died() -> bool {
    LEG_DIED.load(Ordering::SeqCst)
}

/// Forward the simulation's scans to the contract's scan_stream, each in the
/// frame the simulation names, which is `lidar`'s frame; a scan in another
/// frame is dropped, and the first of a run of them logged.
async fn relay_scans(runner: Arc<NodeRunner>, lidar: MountedSensor, token: CancellationToken) {
    let mut sub = match simulation_scans::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("simulation scan_stream subscribe: {e}"),
    };
    let publisher = match lidar_scans::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => return error!("declare scan_stream publisher: {e}"),
    };
    let publish_failures = LatchedLog::warn("scan_stream publish");
    let foreign_frames = LatchedLog::warn("relaying scan_stream");
    let mut first = true;
    receive(
        "simulation scan_stream",
        token.cancelled(),
        async move || sub.next().await,
        async move |(_, msg)| {
            if let Err(mismatch) = check_relayed_frame(&msg.header.frame, &lidar) {
                foreign_frames.failure(mismatch);
                return ControlFlow::Continue(());
            }
            foreign_frames.success();
            let header = lidar_scans::MessageHeader {
                timestamp: msg.header.timestamp,
                frame: msg.header.frame,
                capture_index: msg.header.capture_index,
            };
            let result = match lidar_scans::build_message(
                header,
                msg.min_angle_rad,
                msg.angle_increment_rad,
                msg.time_increment_s,
                msg.min_range_m,
                msg.max_range_m,
                msg.ranges,
                msg.intensities,
            ) {
                Ok(payload) => publisher.publish(payload).await.map_err(|e| e.to_string()),
                Err(e) => Err(e.to_string()),
            };
            match result {
                Ok(()) => {
                    publish_failures.success();
                    if first {
                        first = false;
                        info!("first scan relayed from the simulation");
                    }
                }
                Err(e) => publish_failures.failure(e),
            }
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// The node's entry point, which the binary and the test harness boot. It
/// reads the lidar it relays first, and relays nothing before: a relay whose
/// instance id names no lidar in its copy, or whose name sensor_mounts gives
/// no lidar's frame, fails its start.
pub async fn setup(_params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    let token = node_runner.cancellation_token().clone();
    let asker = node_runner.clone();
    let mounts = describe_sensors::bound_producer(&node_runner).clone();
    let Some(lidar) = read_sensor(
        LIDAR_2D,
        node_runner.copy(),
        node_runner.processor().bound_instance_id(),
        Some(async move || {
            describe_sensors::poll(&asker, &mounts, DESCRIBE_SENSORS_TIMEOUT)
                .await
                .map(|answer| DescribedSensors {
                    sensor_names: answer.data.sensor_names,
                    sensor_kinds: answer.data.sensor_kinds,
                    sensor_frames: answer.data.sensor_frames,
                    carried_by: answer.data.carried_by,
                })
        }),
        node_runner.cancellation_token().cancelled(),
    )
    .await
    .map_err(|refusal| peppygen::Error::Node(refusal.into()))?
    else {
        info!("stopped before sensor_mounts answered");
        return Ok(());
    };
    info!(
        "this relay is lidar '{}', whose scans name frame '{}'",
        lidar.name, lidar.frame
    );
    let scans = tokio::spawn(relay_scans(node_runner.clone(), lidar, token.clone()));
    // A relay leg that ends on its own is flagged and the node cancelled, so
    // the process exits with a failure the stack shows.
    tokio::spawn(async move {
        let _ = scans.await;
        if !token.is_cancelled() {
            LEG_DIED.store(true, Ordering::SeqCst);
            token.cancel();
        }
    });
    Ok(())
}
