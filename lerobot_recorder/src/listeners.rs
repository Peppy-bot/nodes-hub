//! Drain tasks: one per slot member. Each holds a single merged subscription
//! that fans in every producer bound to its cardinality slot, and routes each
//! message by producer identity into the right snapshot slot. No sink or disk
//! I/O ever runs here: the zenoh reception callback blocks when a subscription
//! buffer fills, so a stalled drain would freeze every subscription.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use peppygen::NodeRunner;
use peppygen::consumed_topics::{
    action_sources_joint_commands, color_cameras_video_stream, depth_cameras_video_stream,
    rgbd_cameras_color_stream, rgbd_cameras_depth_stream, state_sources_joint_states,
};
use peppylib::runtime::CancellationToken;
use tracing::warn;

use crate::provenance::ProducerLog;
use crate::snapshot::{CommandSample, JointSample, Slot, Stamped};
use crate::types::{CameraEncoding, FrameBuf, ProducerKey};

const RECEIVE_ERROR_BACKOFF: Duration = Duration::from_millis(100);

/// Loop shape shared by every drain: subscribe once, then route each
/// `(producer, message)` with `$route`. Each message binds `$key`, the
/// producer's [`ProducerKey`], which the route reuses (so it is built once).
/// A producer is recorded into the provenance log only the first time this
/// task sees it, keeping the steady state off the lock and the allocator.
macro_rules! drain {
    ($name:literal, $module:ident, $runner:expr, $token:expr, $log:expr, $producer:ident, $key:ident, $msg:ident, $route:block) => {{
        let mut subscription = match $module::subscribe(&$runner).await {
            Ok(s) => s,
            Err(e) => {
                warn!(concat!("subscribe ", $name, ": {}"), e);
                return;
            }
        };
        let mut seen: HashSet<ProducerKey> = HashSet::new();
        loop {
            let received = tokio::select! {
                _ = $token.cancelled() => return,
                received = subscription.next() => received,
            };
            match received {
                Ok(Some(($producer, $msg))) => {
                    let $key = ProducerKey::from_ref(&$producer);
                    if !seen.contains(&$key) {
                        seen.insert($key.clone());
                        $log.observe($name, &$producer.core_node, &$producer.instance_id);
                    }
                    $route
                }
                Ok(None) => {
                    warn!(concat!($name, " subscription closed"));
                    return;
                }
                Err(e) => {
                    warn!(concat!("receive ", $name, ": {}"), e);
                    tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                }
            }
        }
    }};
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// The producer stamp as nanoseconds since the Unix epoch, or `None` for a
/// pre-epoch stamp (garbage a consumer must not anchor staleness on).
fn stamp_ns(stamp: std::time::SystemTime) -> Option<u64> {
    stamp
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos() as u64)
}

pub async fn state_sources(
    runner: Arc<NodeRunner>,
    index: HashMap<ProducerKey, usize>,
    slots: Vec<Slot<JointSample>>,
    log: ProducerLog,
    token: CancellationToken,
) {
    drain!(
        "joint_states",
        state_sources_joint_states,
        runner,
        token,
        log,
        producer,
        key,
        m,
        {
            let Some(&slot) = index.get(&key) else {
                continue;
            };
            if !all_finite(&m.positions) || !all_finite(&m.velocities) {
                warn!(
                    "joint_states from {}: non-finite, dropping",
                    producer.instance_id
                );
                continue;
            }
            let Some(stamp) = stamp_ns(m.stamp) else {
                warn!(
                    "joint_states from {}: pre-epoch stamp, dropping",
                    producer.instance_id
                );
                continue;
            };
            slots[slot].send_replace(Some(Stamped::new(
                JointSample {
                    positions: m.positions,
                    velocities: m.velocities,
                },
                stamp,
            )));
        }
    )
}

pub async fn action_sources(
    runner: Arc<NodeRunner>,
    index: HashMap<ProducerKey, usize>,
    slots: Vec<Slot<CommandSample>>,
    log: ProducerLog,
    token: CancellationToken,
) {
    drain!(
        "joint_commands",
        action_sources_joint_commands,
        runner,
        token,
        log,
        producer,
        key,
        m,
        {
            let Some(&slot) = index.get(&key) else {
                continue;
            };
            if !all_finite(&m.positions) {
                warn!(
                    "joint_commands from {}: non-finite, dropping",
                    producer.instance_id
                );
                continue;
            }
            let Some(stamp) = stamp_ns(m.stamp) else {
                warn!(
                    "joint_commands from {}: pre-epoch stamp, dropping",
                    producer.instance_id
                );
                continue;
            };
            slots[slot].send_replace(Some(Stamped::new(
                CommandSample {
                    positions: m.positions,
                },
                stamp,
            )));
        }
    )
}

/// The camera slots one drain task owns, keyed by producer. Each dataset
/// camera feature is fed by exactly one stream, so a slot belongs to exactly
/// one route.
pub struct CameraRoute {
    pub by_producer: HashMap<ProducerKey, Slot<Arc<FrameBuf>>>,
    /// Depth streams carry `z16`; force it in case a producer mislabels.
    pub forced_depth: bool,
}

#[allow(clippy::too_many_arguments)]
fn route_frame(
    route: &CameraRoute,
    producer_key: ProducerKey,
    stamp: std::time::SystemTime,
    encoding: &str,
    width: u32,
    height: u32,
    bytes: Vec<u8>,
    label: &str,
) {
    let Some(slot) = route.by_producer.get(&producer_key) else {
        return;
    };
    let Some(stamp) = stamp_ns(stamp) else {
        warn!("{label}: pre-epoch stamp, dropping");
        return;
    };
    let enc = if route.forced_depth {
        CameraEncoding::Z16
    } else {
        match CameraEncoding::parse(encoding) {
            Some(e) => e,
            None => {
                warn!("{label}: unsupported encoding {encoding:?}, dropping");
                return;
            }
        }
    };
    slot.send_replace(Some(Stamped::new(
        Arc::new(FrameBuf {
            encoding: enc,
            width,
            height,
            bytes,
        }),
        stamp,
    )));
}

/// Every camera drain has the same body: subscribe to one stream module and
/// route each frame by producer. They differ only in the function name, the
/// stream label, and the generated topic module.
macro_rules! camera_drain {
    ($fn:ident, $label:literal, $module:ident) => {
        pub async fn $fn(
            runner: Arc<NodeRunner>,
            route: CameraRoute,
            log: ProducerLog,
            token: CancellationToken,
        ) {
            drain!($label, $module, runner, token, log, producer, key, m, {
                route_frame(
                    &route,
                    key,
                    m.header.stamp,
                    &m.encoding,
                    m.width,
                    m.height,
                    m.frame,
                    $label,
                );
            })
        }
    };
}

camera_drain!(color_cameras, "video_stream", color_cameras_video_stream);
camera_drain!(rgbd_color, "color_stream", rgbd_cameras_color_stream);
camera_drain!(rgbd_depth, "depth_stream", rgbd_cameras_depth_stream);
camera_drain!(
    depth_cameras,
    "depth_video_stream",
    depth_cameras_video_stream
);
