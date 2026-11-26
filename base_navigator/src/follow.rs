//! The robot's tree as the navigator holds it: the followers of the transforms link,
//! which take every message of transforms and of static_transforms that a publisher bound
//! to it sends into the tree under the publisher that sent it, by control_core's
//! followers; and the two frames of base_state the navigator places by, the robot frame
//! and the odometry frame.

use std::sync::Arc;

use control_core::robot_tree::{RobotTree, follow_moving, follow_static};
use frame_tree::{FrameName, MovingMessage, Publisher, StaticMessage, TransformRecord};
use peppygen::consumed_topics::transforms::{static_transforms, transforms};
use peppygen::{NodeRunner, ProducerRef};

/// base_state's robot frame, the root link of the robot's URDF.
pub const ROBOT_FRAME: &str = "base_footprint";

/// base_state's odometry frame, in which the grid of the floor is kept and every goal
/// runs.
pub const ODOMETRY_FRAME: &str = "odom";

pub fn robot_frame() -> FrameName {
    FrameName::parse(ROBOT_FRAME).expect("a frame name tf takes")
}

pub fn odometry_frame() -> FrameName {
    FrameName::parse(ODOMETRY_FRAME).expect("a frame name tf takes")
}

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

/// The records of a message's transforms, field for field: each topic of the
/// transforms link generates an item type of its own.
macro_rules! records {
    ($transforms:expr) => {
        $transforms
            .into_iter()
            .map(|t| TransformRecord {
                frame: t.frame,
                child_frame: t.child_frame,
                position: t.position,
                orientation: t.orientation,
            })
            .collect()
    };
}

/// A message of transforms as the tree takes it, field for field.
fn moving_message(message: transforms::Message) -> MovingMessage {
    MovingMessage {
        timestamp: message.timestamp,
        transforms: records!(message.transforms),
        start_timestamp: message.start_timestamp,
    }
}

/// A message of static_transforms as the tree takes it, field for field.
fn static_message(message: static_transforms::Message) -> StaticMessage {
    StaticMessage {
        transforms: records!(message.transforms),
    }
}
