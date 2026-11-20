//! The followers of the transforms link: every message of transforms and of
//! static_transforms that a publisher bound to it sends, taken into the robot's tree under
//! the publisher that sent it.

use std::sync::Arc;

use control_core::robot_tree::{RobotTree, follow_moving, follow_static};
use frame_tree::{MovingMessage, Publisher, StaticMessage, TransformRecord};
use peppygen::NodeRunner;
use peppygen::consumed_topics::transforms::{static_transforms, transforms};

use crate::wire::{sendable, stopped};

/// Subscribes to both topics of the transforms link, then takes each message into `tree`
/// on a task of its own until the node stops.
pub async fn follow_tree(runner: &Arc<NodeRunner>, tree: Arc<RobotTree>) -> peppygen::Result<()> {
    let mut moving = sendable(transforms::subscribe(runner)).await?;
    let mut fixed = sendable(static_transforms::subscribe(runner)).await?;
    tokio::spawn(follow_moving(
        tree.clone(),
        stopped(runner),
        async move || {
            moving.next().await.map(|received| {
                received
                    .map(|(producer, message)| (publisher_of(producer), moving_message(message)))
            })
        },
    ));
    tokio::spawn(follow_static(tree, stopped(runner), async move || {
        fixed.next().await.map(|received| {
            received.map(|(producer, message)| (publisher_of(producer), static_message(message)))
        })
    }));
    Ok(())
}

/// The publisher of a message: its producer's core node and instance id.
fn publisher_of(producer: peppygen::ProducerRef) -> Publisher {
    Publisher {
        core_node: producer.core_node,
        instance_id: producer.instance_id,
    }
}

/// The transform records of a message's `transforms`, field for field: each topic's
/// generated item is a type of its own.
macro_rules! transform_records {
    ($items:expr) => {
        $items
            .into_iter()
            .map(|item| TransformRecord {
                frame: item.frame,
                child_frame: item.child_frame,
                position: item.position,
                orientation: item.orientation,
            })
            .collect()
    };
}

/// A message of transforms as the tree takes it, field for field.
fn moving_message(message: transforms::Message) -> MovingMessage {
    MovingMessage {
        timestamp: message.timestamp,
        transforms: transform_records!(message.transforms),
        start_timestamp: message.start_timestamp,
    }
}

/// A message of static_transforms as the tree takes it: its transforms, each holding at
/// every time.
fn static_message(message: static_transforms::Message) -> StaticMessage {
    StaticMessage {
        transforms: transform_records!(message.transforms),
    }
}
