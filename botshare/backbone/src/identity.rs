//! The robot the base drives, as its initializer names it: the initializer answers the
//! robot's model on robot_identity's `get_identity` and serves the description of that
//! model on `get_description`. The base drives the bundled model the initializer names,
//! when its URDF gives a differential base, and holds the description's digest to the
//! digest of the bundled description it was built with, so the base is always modeled
//! from the URDF the rest of the robot reads. The base asks after its setup has
//! returned, since its initializer may start after it, and asks until the initializer
//! answers.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use control_core::asking;
use peppygen::NodeRunner;
use peppygen::consumed_services::robot_identity::{get_description, get_identity};

use crate::config::{BaseConfig, BaseParameters};
use crate::node::NodeError;

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

/// The base of the bundled model the robot's initializer names, configured by
/// `parameters` for a node started at `started` on the wall clock, or `None` once the
/// node is cancelled before the initializer answers: refused when the model's URDF gives
/// no differential base, or when the initializer serves another description of the model.
pub async fn identify(
    runner: &Arc<NodeRunner>,
    parameters: &BaseParameters,
    started: SystemTime,
) -> Option<Result<BaseConfig, NodeError>> {
    let asker = runner.clone();
    let answer = asking::ask(
        "robot_identity",
        runner.cancellation_token().cancelled(),
        async move || ask(&asker).await,
    )
    .await?;
    Some(driven_by(answer, parameters, started))
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

/// The base of the bundled model `answer` names, held to the description `answer`
/// describes, for a node started at `started`.
fn driven_by(
    answer: Answer,
    parameters: &BaseParameters,
    started: SystemTime,
) -> Result<BaseConfig, NodeError> {
    let config = parameters.configure(&answer.model, started)?;
    config.model.hold_digest(&answer.digest)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_rig::params;

    fn parameters() -> BaseParameters {
        BaseParameters::parse(&params()).expect("the rig's parameters")
    }

    /// The base of the bundled model `answer` names, for a node started at
    /// the epoch.
    fn base_of(answer: Answer) -> Result<BaseConfig, NodeError> {
        driven_by(answer, &parameters(), SystemTime::UNIX_EPOCH)
    }

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

    #[test]
    fn the_botshare_base_the_initializer_names_drives_the_node() {
        let config = base_of(answer_of("botshare_base")).expect("the base");
        assert_eq!(config.model.id(), "botshare_base");
    }

    // A model whose URDF gives no differential base is refused naming why.
    #[test]
    fn a_model_of_no_base_is_refused() {
        let refused = base_of(answer_of("so101")).expect_err("an SO-101 arm");
        assert_eq!(
            refused.to_string(),
            "the model the robot's initializer serves"
        );
        let model = std::error::Error::source(&refused).expect("the model's refusal");
        assert!(
            model
                .to_string()
                .starts_with("model so101 is no differential base this node drives: "),
            "{model}"
        );
    }

    // Another build of the model's description is refused, naming both digests and what
    // to rebuild.
    #[test]
    fn another_description_of_the_model_is_refused_naming_both_digests() {
        let other = "0".repeat(64);
        let answer = Answer {
            model: "botshare_base".to_owned(),
            digest: other.clone(),
        };
        let refused = base_of(answer).expect_err("another description");
        let built = robot_description::model("botshare_base")
            .expect("the bundled base")
            .digest();
        assert_eq!(
            refused.to_string(),
            format!(
                "the robot's initializer serves the description {other} of botshare_base, and \
                 this backbone was built with {built}: build the backbone and the initializer \
                 from one public-peppy-libs commit"
            )
        );
    }
}
