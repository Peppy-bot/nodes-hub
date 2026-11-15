//! The robot the backbone drives, as its initializer names it: the initializer answers
//! the robot's model on robot_identity's `get_identity` and serves the description of
//! that model on `get_description`. The backbone drives the bundled model the
//! initializer names, when its arms are the ones this node drives, and holds the
//! description's digest to the digest of the bundled description it was built with, so
//! the arms are always modeled from the URDF the rest of the robot reads. The backbone
//! asks after its setup has returned, since its initializer may start after it, and
//! asks until the initializer answers.

use std::sync::Arc;
use std::time::Duration;

use control_core::asking;
use peppygen::NodeRunner;
use peppygen::consumed_services::robot_identity::{get_description, get_identity};
use robot_description::Driven;
use srs_model::ARM_DOF;

use crate::node::NodeError;
use crate::robot::{ARMS, LIMBS};

/// How long the initializer may take to answer each question: the description carries
/// the collision meshes.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(10);

/// What the initializer answers: the robot's model id and the digest of the
/// description it serves.
#[derive(Debug)]
struct Answer {
    model: String,
    digest: String,
}

/// The bundled model the robot's initializer names, or `None` once the node is
/// cancelled before the initializer answers: refused when this node does not drive
/// the model, naming the models it drives, or when the initializer serves another
/// description of the model.
pub async fn identify(
    runner: &Arc<NodeRunner>,
) -> Option<Result<Driven<ARM_DOF, ARMS>, NodeError>> {
    let asker = runner.clone();
    let answer = asking::ask(
        "robot_identity",
        runner.cancellation_token().cancelled(),
        async move || ask(&asker).await,
    )
    .await?;
    Some(driven_by(answer))
}

/// One ask of the robot's model and of the digest of its description.
async fn ask(runner: &NodeRunner) -> peppygen::Result<Answer> {
    let model = get_identity::poll(runner, get_identity::bound_producer(runner), ANSWER_TIMEOUT)
        .await?
        .data
        .model;
    let digest = get_description::poll(
        runner,
        get_description::bound_producer(runner),
        ANSWER_TIMEOUT,
    )
    .await?
    .data
    .digest;
    Ok(Answer { model, digest })
}

/// The bundled model `answer` names, held to this node's arms and to the description
/// `answer` describes.
fn driven_by(answer: Answer) -> Result<Driven<ARM_DOF, ARMS>, NodeError> {
    let driven = LIMBS
        .model(&answer.model)
        .map_err(|undriven| NodeError::Model(undriven.into()))?;
    let built = driven.model().digest();
    if answer.digest != built {
        return Err(NodeError::OtherDescription {
            model: driven.model().id(),
            served: answer.digest,
            built: built.to_owned(),
        });
    }
    Ok(driven)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The answer of the initializer of `model`, built with this node's libraries.
    fn answer_of(model: &str) -> Answer {
        Answer {
            model: model.to_owned(),
            digest: robot_description::model(model)
                .expect("a bundled model")
                .digest()
                .to_owned(),
        }
    }

    // Both OpenArm models drive the backbone, each from the description its initializer
    // serves.
    #[test]
    fn an_openarm_the_initializer_names_drives_the_backbone() {
        for model in ["openarm_v1", "openarm_v2"] {
            let driven = driven_by(answer_of(model)).expect("an OpenArm drives the backbone");
            assert_eq!(driven.model().id(), model);
        }
    }

    // A model of other arms is refused naming the served model and the models of the
    // backbone's arms.
    #[test]
    fn a_model_the_backbone_does_not_drive_is_refused_naming_the_driven_ones() {
        let refused =
            driven_by(answer_of("so101")).expect_err("an SO-101 does not drive the backbone");
        assert_eq!(
            refused.to_string(),
            "the model the robot's initializer serves"
        );
        let model = std::error::Error::source(&refused)
            .expect("the model error")
            .to_string();
        assert!(
            model.starts_with("model so101 describes")
                && model
                    .ends_with("the models of that shape are \"openarm_v1\" and \"openarm_v2\""),
            "{model}"
        );
    }

    // An initializer that serves another build of the model's description is refused,
    // naming both digests and what to rebuild.
    #[test]
    fn another_description_of_the_model_is_refused_naming_both_digests() {
        let other = "0".repeat(64);
        let built = answer_of("openarm_v2").digest;
        let refused = driven_by(Answer {
            model: "openarm_v2".to_owned(),
            digest: other.clone(),
        })
        .expect_err("another description is refused")
        .to_string();
        assert_eq!(
            refused,
            format!(
                "the robot's initializer serves the description {other} of openarm_v2, and \
                 this backbone was built with {built}: build the backbone and the initializer \
                 from one public-peppy-libs commit"
            )
        );
    }
}
