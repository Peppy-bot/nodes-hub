//! The node's start: the notes, then one follower for each part of the stack that the
//! node is linked to.

use std::sync::Arc;

use peppygen::NodeRunner;
use peppygen::consumed_services::motion_planning::get_planning_scene;
use peppygen::consumed_topics::camera::video_stream;
use tracing::info;

use crate::canvas::Canvas;
use crate::config::Config;
use crate::draw::draw_notes;
use crate::wire::Shared;
use crate::wire::arms::follow_arms;
use crate::wire::camera::{DepthSettings, follow_color, follow_depth};
use crate::wire::planner_map::follow_planner_map;
use crate::wire::scene::follow_scene;

/// Starts the followers that draw on `canvas`: the arms always, the camera when the
/// camera link is bound, and the planning scene and the planner's voxel map when the
/// motion_planning link is bound.
pub async fn setup(
    config: Config,
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
) -> peppygen::Result<()> {
    peppygen::clock::init(&runner).await?;
    draw_notes(canvas.as_ref());
    let shared = Arc::new(Shared::default());
    tokio::spawn(follow_arms(
        runner.clone(),
        canvas.clone(),
        config.trail_length,
        shared.clone(),
    ));
    if video_stream::bound_producer(&runner).is_some() {
        tokio::spawn(follow_color(
            runner.clone(),
            canvas.clone(),
            config.image_interval,
        ));
        let settings = DepthSettings {
            image_interval: config.image_interval,
            cloud_interval: config.cloud_interval,
            cloud_stride: config.cloud_stride,
            camera_name: config.camera_name.clone(),
        };
        tokio::spawn(follow_depth(runner.clone(), canvas.clone(), settings));
    } else {
        info!("the camera link is vacant: no images and no points drawn");
    }
    if get_planning_scene::bound_producer(&runner).is_some() {
        tokio::spawn(follow_planner_map(runner.clone(), canvas.clone()));
        tokio::spawn(follow_scene(
            runner,
            canvas,
            config.scene_poll_interval,
            shared,
        ));
    } else {
        info!("the motion_planning link is vacant: no planning scene and no voxel map drawn");
    }
    Ok(())
}
