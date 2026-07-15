//! Drain tasks: one per slot member. Each holds a single merged subscription
//! that fans in every producer bound to its cardinality slot, and routes each
//! message by producer identity into the right snapshot slot. No sink or disk
//! I/O ever runs here: the zenoh reception callback blocks when a subscription
//! buffer fills, so a stalled drain would freeze every subscription.

use std::collections::HashMap;
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
/// `(producer, message)` with `$route`.
macro_rules! drain {
    ($name:literal, $module:ident, $runner:expr, $token:expr, $log:expr, $producer:ident, $msg:ident, $route:block) => {{
        let mut subscription = match $module::subscribe(&$runner).await {
            Ok(s) => s,
            Err(e) => {
                warn!(concat!("subscribe ", $name, ": {}"), e);
                return;
            }
        };
        loop {
            let received = tokio::select! {
                _ = $token.cancelled() => return,
                received = subscription.next() => received,
            };
            match received {
                Ok(Some(($producer, $msg))) => {
                    $log.observe($name, &$producer.core_node, &$producer.instance_id);
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
        m,
        {
            let Some(&slot) = index.get(&ProducerKey::from_ref(&producer)) else {
                continue;
            };
            if !all_finite(&m.positions) || !all_finite(&m.velocities) {
                warn!(
                    "joint_states from {}: non-finite, dropping",
                    producer.instance_id
                );
                continue;
            }
            slots[slot].send_replace(Some(Stamped::now(JointSample {
                positions: m.positions,
                velocities: m.velocities,
            })));
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
        m,
        {
            let Some(&slot) = index.get(&ProducerKey::from_ref(&producer)) else {
                continue;
            };
            if !all_finite(&m.positions) {
                warn!(
                    "joint_commands from {}: non-finite, dropping",
                    producer.instance_id
                );
                continue;
            }
            slots[slot].send_replace(Some(Stamped::now(CommandSample {
                positions: m.positions,
            })));
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

fn route_frame(
    route: &CameraRoute,
    producer_key: ProducerKey,
    encoding: &str,
    width: u32,
    height: u32,
    bytes: Vec<u8>,
    label: &str,
) {
    let Some(slot) = route.by_producer.get(&producer_key) else {
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
    slot.send_replace(Some(Stamped::now(Arc::new(FrameBuf {
        encoding: enc,
        width,
        height,
        bytes,
    }))));
}

pub async fn color_cameras(
    runner: Arc<NodeRunner>,
    route: CameraRoute,
    log: ProducerLog,
    token: CancellationToken,
) {
    drain!(
        "video_stream",
        color_cameras_video_stream,
        runner,
        token,
        log,
        producer,
        m,
        {
            route_frame(
                &route,
                ProducerKey::from_ref(&producer),
                &m.encoding,
                m.width,
                m.height,
                m.frame,
                "video_stream",
            );
        }
    )
}

pub async fn rgbd_color(
    runner: Arc<NodeRunner>,
    route: CameraRoute,
    log: ProducerLog,
    token: CancellationToken,
) {
    drain!(
        "color_stream",
        rgbd_cameras_color_stream,
        runner,
        token,
        log,
        producer,
        m,
        {
            route_frame(
                &route,
                ProducerKey::from_ref(&producer),
                &m.encoding,
                m.width,
                m.height,
                m.frame,
                "color_stream",
            );
        }
    )
}

pub async fn rgbd_depth(
    runner: Arc<NodeRunner>,
    route: CameraRoute,
    log: ProducerLog,
    token: CancellationToken,
) {
    drain!(
        "depth_stream",
        rgbd_cameras_depth_stream,
        runner,
        token,
        log,
        producer,
        m,
        {
            route_frame(
                &route,
                ProducerKey::from_ref(&producer),
                &m.encoding,
                m.width,
                m.height,
                m.frame,
                "depth_stream",
            );
        }
    )
}

pub async fn depth_cameras(
    runner: Arc<NodeRunner>,
    route: CameraRoute,
    log: ProducerLog,
    token: CancellationToken,
) {
    drain!(
        "depth_video_stream",
        depth_cameras_video_stream,
        runner,
        token,
        log,
        producer,
        m,
        {
            route_frame(
                &route,
                ProducerKey::from_ref(&producer),
                &m.encoding,
                m.width,
                m.height,
                m.frame,
                "depth_video_stream",
            );
        }
    )
}
