//! The motion planner's voxel map: occupancy_map's messages, parsed and drawn as they
//! come.

use std::ops::ControlFlow;
use std::sync::Arc;

use control_core::latched_log::LatchedLog;
use control_core::receiving;
use peppygen::NodeRunner;
use peppygen::consumed_topics::motion_planning::occupancy_map;
use tracing::error;

use crate::canvas::Canvas;
use crate::draw::planner_map::draw_planner_map;
use crate::model::planner_map::PlannerMap;
use crate::wire::{Shared, sendable, stopped};

/// Draws each message of occupancy_map in the robot frame as it comes, until the node
/// stops; a message in another frame is dropped, the first of a run logged.
pub async fn follow_planner_map(
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    shared: Arc<Shared>,
) {
    let cancelled = stopped(&runner);
    let mut subscription = match sendable(occupancy_map::subscribe(&runner)).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!("occupancy_map: no subscription, the planner's map is not drawn: {e}");
            return;
        }
    };
    let follower = MapFollower {
        canvas,
        unreadable: LatchedLog::warn("reading occupancy_map"),
    };
    let other_frames = LatchedLog::warn("drawing occupancy_map");
    receiving::receive(
        "occupancy_map",
        cancelled,
        async move || subscription.next().await,
        async move |(_producer, message)| {
            let held = shared.hold_to_robot_frame(&runner, "occupancy_map", &message.frame);
            match held {
                Ok(()) => {
                    other_frames.success();
                    follower.on_message(&message);
                }
                Err(reason) => other_frames.failure(reason),
            }
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// The map follower's state.
struct MapFollower {
    canvas: Arc<dyn Canvas>,
    unreadable: LatchedLog,
}

impl MapFollower {
    fn on_message(&self, message: &occupancy_map::Message) {
        match PlannerMap::parse(
            message.timestamp,
            message.voxel_size_m,
            &message
                .centers
                .iter()
                .map(|center| center.position)
                .collect::<Vec<_>>(),
            &message.waiting,
        ) {
            Ok(map) => {
                self.unreadable.success();
                draw_planner_map(self.canvas.as_ref(), &map);
            }
            Err(e) => self.unreadable.failure(e),
        }
    }
}
