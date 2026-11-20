//! The followers of the transforms link: every message of transforms and of
//! static_transforms that a publisher bound to it sends, taken into the robot's tree under
//! the publisher that sent it, by control_core's followers.

use std::sync::Arc;

use control_core::robot_tree::{RobotTree, follow_moving, follow_static};
use frame_tree::{MovingMessage, Publisher, StaticMessage, TransformRecord};
use peppygen::consumed_topics::transforms::{static_transforms, transforms};
use peppygen::{NodeRunner, ProducerRef};

/// Subscribes to both topics of the transforms link, then takes each message into `tree`
/// on a task of its own until the node stops. A subscription that fails is the node's
/// startup error.
pub(crate) async fn follow_tree(
    runner: &Arc<NodeRunner>,
    tree: Arc<RobotTree>,
) -> peppygen::Result<()> {
    let mut moving = transforms::subscribe(runner).await?;
    let mut fixed = static_transforms::subscribe(runner).await?;
    let moving_stopped = runner.cancellation_token().clone();
    let fixed_stopped = moving_stopped.clone();
    tokio::spawn(follow_moving(
        tree.clone(),
        async move { moving_stopped.cancelled().await },
        async move || {
            let received = moving.next().await?;
            Ok::<_, peppygen::Error>(
                received
                    .map(|(producer, message)| (publisher_of(producer), moving_message(message))),
            )
        },
    ));
    tokio::spawn(follow_static(
        tree,
        async move { fixed_stopped.cancelled().await },
        async move || {
            let received = fixed.next().await?;
            Ok::<_, peppygen::Error>(
                received
                    .map(|(producer, message)| (publisher_of(producer), static_message(message))),
            )
        },
    ));
    Ok(())
}

/// The publisher of a message: its producer's core node and instance id.
fn publisher_of(producer: ProducerRef) -> Publisher {
    Publisher {
        core_node: producer.core_node,
        instance_id: producer.instance_id,
    }
}

/// The record of the tree for a transform record of the wire, field for field.
macro_rules! record_of {
    ($item:expr) => {{
        let item = $item;
        TransformRecord {
            frame: item.frame,
            child_frame: item.child_frame,
            position: item.position,
            orientation: item.orientation,
        }
    }};
}

/// A message of transforms as the tree takes it, field for field.
fn moving_message(message: transforms::Message) -> MovingMessage {
    MovingMessage {
        timestamp: message.timestamp,
        transforms: message
            .transforms
            .into_iter()
            .map(|item| record_of!(item))
            .collect(),
        start_timestamp: message.start_timestamp,
    }
}

/// A message of static_transforms as the tree takes it: its transforms, field for field,
/// each holding at every time.
fn static_message(message: static_transforms::Message) -> StaticMessage {
    StaticMessage {
        transforms: message
            .transforms
            .into_iter()
            .map(|item| record_of!(item))
            .collect(),
    }
}
