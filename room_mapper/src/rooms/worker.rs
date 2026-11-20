//! The worker: each scan of the lidar, in arrival order, measured once the odometry
//! encloses its sweep and handed to the room, and map_poses and the transform map to odom
//! of each answer published. A scan waits for base_states that enclose its sweep and for
//! the read of describe_base at a new odometry start; it is dropped while the lidar's place
//! on the base is unknown or describe_base gave no footprint, and once it lies further than
//! the odometry's kept window behind the newest scan. A search that no scan reaches within
//! the source timeout ends.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use control_core::latched_log::LatchedLog;
use control_core::sync::lock_ignoring_poison;
use peppygen::NodeRunner;
use peppygen::emitted_topics::room_map::map_poses;
use peppygen::emitted_topics::transforms::transforms;
use tokio::sync::mpsc;
use tracing::warn;

use crate::geometry::on_floor;
use crate::inputs::{FactsRead, Inputs, Mount};
use crate::lidar::{ArrivedScan, Measurement};
use crate::messages::{duration_ns, orientation_of, position_of, time_of};
use crate::odometry::KEPT_NS;
use crate::rooms::members::MAP_FRAME;
use crate::rooms::state::{Answer, RoomState};

/// The frames of the transform the node publishes.
const ODOMETRY_FRAME: &str = "odom";

/// Runs the worker until the lidar's scans end, publishing on map_poses and transforms.
pub async fn run(
    runner: Arc<NodeRunner>,
    room: Arc<Mutex<RoomState>>,
    inputs: Arc<Inputs>,
    mut scans: mpsc::UnboundedReceiver<ArrivedScan>,
    source_timeout: Duration,
) -> peppygen::Result<()> {
    let map_poses = map_poses::declare_publisher(&runner).await?;
    let transforms = transforms::declare_publisher(&runner).await?;
    // Publishes map_poses and the transform map to odom of an answer, stamped at its scan.
    let publish = async |answer: Answer| {
        let stamp = time_of(answer.map_pose.stamp_ns);
        let pose = answer.map_pose;
        let pose_message = map_poses::build_message(
            MAP_FRAME.to_string(),
            stamp,
            answer.map_name,
            position_of(pose.pose),
            orientation_of(pose.pose),
            pose.deviations.position_std_m,
            pose.deviations.yaw_std_rad,
            pose.localized,
        );
        let transform_message = transforms::build_message(
            stamp,
            vec![transforms::MessageTransformsItem {
                frame: MAP_FRAME.to_string(),
                child_frame: ODOMETRY_FRAME.to_string(),
                position: position_of(answer.map_from_odometry),
                orientation: orientation_of(answer.map_from_odometry),
            }],
            time_of(answer.start_unix_ns),
        );
        for (topic, message, publisher) in [
            ("map_poses", pose_message, &map_poses),
            ("transforms", transform_message, &transforms),
        ] {
            let sent = match message {
                Ok(payload) => publisher.publish(payload).await,
                Err(error) => Err(error),
            };
            if let Err(error) = sent {
                warn!("{topic} is not published: {error}");
            }
        }
    };
    let mut waiting: VecDeque<ArrivedScan> = VecDeque::new();
    let dropped = LatchedLog::warn("scans of the lidar are dropped");
    let timeout_ns = duration_ns(source_timeout);
    let mut scanned_ns = inputs.clock.now_ns();
    loop {
        // A scan, a change of the inputs a waiting scan reads, or the lidar's silence past
        // the source timeout since its newest scan.
        // Scans first, so a scan that arrived is received before the silence it ends.
        tokio::select! {
            biased;
            arrived = scans.recv() => match arrived {
                Some(scan) => {
                    scanned_ns = inputs.arrival_of(scan.stamp_ns);
                    waiting.push_back(scan);
                    drop_stale(&mut waiting, &dropped);
                }
                None => return Ok(()),
            },
            () = inputs.changed.notified() => {}
            () = inputs.clock.until(scanned_ns + timeout_ns) => {
                lock_ignoring_poison(&room).search_went_silent();
                scanned_ns = inputs.clock.now_ns();
            }
        }
        while let Some(scan) = waiting.front() {
            let Some(answered) = take(&room, &inputs, scan, &dropped) else {
                break;
            };
            waiting.pop_front();
            let Some(answer) = answered else {
                continue;
            };
            publish(answer).await;
        }
    }
}

/// Drops the waiting scans stamped further than the odometry's kept window before the
/// newest: no base_states kept encloses their sweep once newer ones arrive.
fn drop_stale(waiting: &mut VecDeque<ArrivedScan>, dropped: &LatchedLog) {
    let Some(newest_ns) = waiting.back().map(|scan| scan.stamp_ns) else {
        return;
    };
    let stale = |scan: &ArrivedScan| newest_ns.saturating_sub(scan.stamp_ns) > KEPT_NS;
    while waiting.front().is_some_and(stale) {
        waiting.pop_front();
        dropped.failure(
            "the base's odometry does not enclose their sweep within the odometry's kept \
             window: check that the base's node publishes base_states",
        );
    }
}

/// The room's answer to `scan`: none while the scan waits for base_states that enclose it or
/// for the read of describe_base; `Some(None)` for a scan taken or dropped with no answer.
/// The room is held while the scan is taken, and the odometry only while it is measured.
fn take(
    room: &Mutex<RoomState>,
    inputs: &Inputs,
    scan: &ArrivedScan,
    dropped: &LatchedLog,
) -> Option<Option<Answer>> {
    let mount = match &*lock_ignoring_poison(&inputs.mount) {
        Mount::Placed {
            frame,
            base_from_lidar,
        } if *frame == scan.frame => *base_from_lidar,
        // No scan is taken without the lidar's place on the base.
        _ => return Some(None),
    };
    let footprint = match &*inputs.facts.borrow() {
        FactsRead::Read(facts) => facts.footprint.clone(),
        FactsRead::Reading(_) => return None,
        FactsRead::NoOdometry | FactsRead::Unanswered(_) => return Some(None),
    };
    let laser = match scan.laser_scan(&mount, &footprint) {
        Ok(laser) => laser,
        Err(error) => {
            warn!("a scan of the lidar is refused: {error:?}");
            return Some(None);
        }
    };
    // The room's lock comes before the odometry's, as the members take them.
    let mut room = lock_ignoring_poison(room);
    let (measured, kept) = {
        let odometry = lock_ignoring_poison(&inputs.odometry);
        let measured = match scan.measure(laser, &mount, &odometry) {
            Measurement::Waiting => return None,
            Measurement::Never => {
                dropped.failure(
                    "the base's odometry kept does not hold their time: the worker falls behind \
                     the lidar, or the odometry started over during a sweep",
                );
                return Some(None);
            }
            Measurement::Measured(measured) => {
                dropped.success();
                measured
            }
        };
        let kept = odometry.kept_of(measured.start);
        (measured, kept)
    };
    let odometry_at = |at_ns: i64| kept.as_ref()?.at(at_ns).and_then(|pose| on_floor(&pose));
    Some(tokio::task::block_in_place(|| {
        room.take_scan(&measured, odometry_at)
    }))
}
