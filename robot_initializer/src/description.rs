//! The description of this robot's model: its URDF, its SRDF and the
//! collision meshes the URDF names, as public-peppy-libs' robot_description
//! holds them, with their digest. The node answers the same files for as long
//! as it runs.

use std::sync::Arc;

use control_core::serving;
use peppygen::NodeRunner;
use peppygen::exposed_services::robot_identity::get_description;
use peppylib::runtime::CancellationToken;
use robot_description::Model;

/// The answer of `get_description` for `model`.
fn response(model: &Model) -> get_description::Response {
    get_description::Response::new(
        model.urdf().to_owned(),
        model.srdf().to_owned(),
        model
            .meshes()
            .iter()
            .map(|(name, data)| get_description::ResponseMeshesItem {
                name: (*name).to_owned(),
                data: data.to_vec(),
            })
            .collect(),
        model.digest().to_owned(),
    )
}

/// Starts serving `get_description` of `model`, for as long as the node runs.
pub fn serve(model: &'static Model, runner: Arc<NodeRunner>) {
    let token = runner.cancellation_token().clone();
    tokio::spawn(run(model, runner, token));
}

async fn run(model: &'static Model, runner: Arc<NodeRunner>, token: CancellationToken) {
    let response = response(model);
    tracing::info!(
        "get_description service started for {}, digest {}",
        model.id(),
        model.digest()
    );
    serving::serve("get_description", token.cancelled(), || {
        get_description::handle_requests(&runner, |_req| Ok(response.clone()))
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_holds_the_models_files_and_digest() {
        for model in robot_description::models() {
            let answer = response(model);
            assert_eq!(answer.urdf, model.urdf(), "{}", model.id());
            assert_eq!(answer.srdf, model.srdf(), "{}", model.id());
            assert_eq!(answer.digest, model.digest(), "{}", model.id());
            let meshes: Vec<(&str, &[u8])> = answer
                .meshes
                .iter()
                .map(|mesh| (mesh.name.as_str(), mesh.data.as_slice()))
                .collect();
            assert_eq!(meshes, model.meshes(), "{}", model.id());
        }
    }
}
