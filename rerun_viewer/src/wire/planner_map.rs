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
use crate::wire::{sendable, stopped};

/// Draws each message of occupancy_map as it comes, until the node stops.
pub async fn follow_planner_map(runner: Arc<NodeRunner>, canvas: Arc<dyn Canvas>) {
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
    receiving::receive(
        "occupancy_map",
        cancelled,
        async move || subscription.next().await,
        async move |(_producer, message)| {
            follower.on_message(&message);
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
            &message.centers,
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
