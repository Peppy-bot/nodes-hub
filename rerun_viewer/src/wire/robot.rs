//! The robot's body: the description that robot_identity gives, read once into the
//! collision shapes of each link and drawn. The arms follower places the links.

use std::sync::Arc;
use std::time::Duration;

use control_core::asking;
use peppygen::NodeRunner;
use peppygen::consumed_services::identity::get_description;
use tracing::{error, info};

use crate::canvas::Canvas;
use crate::draw::robot::draw_body;
use crate::model::robot::{BodyError, RobotBody};
use crate::wire::{Shared, sendable, stopped};

/// How long the viewer waits for the robot's description, which carries its meshes.
const DESCRIPTION_TIMEOUT: Duration = Duration::from_secs(10);

/// The body that `served` describes.
fn read_body(served: &get_description::ResponseData) -> Result<RobotBody, BodyError> {
    let meshes: Vec<(&str, &[u8])> = served
        .meshes
        .iter()
        .map(|mesh| (mesh.name.as_str(), mesh.data.as_slice()))
        .collect();
    RobotBody::parse(&served.urdf, &served.srdf, &meshes)
}

/// Asks the robot's description until it answers or the node stops, then shares the
/// body it describes with the arms follower and draws its shapes. The description
/// holds for the robot's whole session, so the viewer asks it once.
pub async fn follow_robot(runner: Arc<NodeRunner>, canvas: Arc<dyn Canvas>, shared: Arc<Shared>) {
    let asker = runner.clone();
    let asked = asking::ask("get_description", stopped(&runner), async move || {
        ask_description(&asker).await
    })
    .await;
    let Some(served) = asked else {
        return;
    };
    match read_body(&served) {
        Ok(body) => {
            info!(
                "the robot's body: {} links with collision shapes",
                body.links().len()
            );
            // Shared first: once a shape is drawn, each snapshot places the links.
            let body = Arc::new(body);
            shared.body.send_replace(Some(body.clone()));
            draw_body(canvas.as_ref(), &body);
        }
        Err(e) => error!("the robot's description: {e}; the robot's links are not drawn"),
    }
}

async fn ask_description(runner: &NodeRunner) -> Result<get_description::ResponseData, String> {
    let identity = get_description::bound_producer(runner).ok_or("the identity link is vacant")?;
    let answer = sendable(get_description::poll(runner, identity, DESCRIPTION_TIMEOUT))
        .await
        .map_err(|e| format!("get_description: {e}"))?;
    Ok(answer.data)
}
