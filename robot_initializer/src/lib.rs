//! Bringing one robot into being, and saying when it is ready. The robot is
//! of whatever model the launcher names, an OpenArm and an SO-101 alike.
//!
//! Every robot serves the readiness the backbone gates on. On a real robot
//! the drivers of its limbs answer for it; on a simulated robot the
//! simulation standing it answers for every limb at once.
//!
//! A robot that runs in a simulation also joins its scene, which is the one
//! thing a simulated robot does that a real one does not, and its setup ends
//! only once the simulation stands it.
//!
//! Every robot says who it is as well: the name it stands under, the model
//! it is and the core node hosting it, which is the name it joins a scene
//! under, and the description of that model. And it gives its guide: the
//! park pose of each arm that its description gives one.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Duration;

use peppygen::{NodeRunner, Parameters, Result};
use robot_description::Model;

mod description;
mod guide;
mod identity;
mod join;
mod readiness;

use guide::Guide;
use identity::Identity;
use join::Joined;
pub use join::{HostClock, Moment, StandClock};

/// The robot's own error: a fact about this robot, which is what a launcher
/// reads back when the node refuses to start.
pub(crate) fn refused(message: impl Into<String>) -> peppygen::Error {
    peppygen::Error::Node(std::io::Error::other(message.into()).into())
}

/// The model this robot is, as the launcher wrote it: a model of
/// robot_description, whose description the node serves and joins a scene
/// with.
fn model_of(model: &str) -> Result<&'static Model> {
    robot_description::model(model).map_err(|unknown| refused(format!("model: {unknown}")))
}

pub async fn setup(params: Parameters, runner: Arc<NodeRunner>) -> Result<()> {
    setup_with_clock(params, runner, HostClock::starting_now).await
}

/// [`setup`], with the wait for the robot to stand on the clock that `clock`
/// starts on the stand budget. The node runs on [`HostClock`]; a test drives
/// the wait with a clock of its own.
pub async fn setup_with_clock<C: StandClock>(
    params: Parameters,
    runner: Arc<NodeRunner>,
    clock: impl FnOnce(Duration) -> C + Send,
) -> Result<()> {
    let model = model_of(&params.model)?;
    let description = model.description();
    let identity = Identity::of(&runner, model.id());
    identity::serve(identity.clone(), runner.clone());
    description::serve(model, runner.clone());
    guide::serve(Guide::of(&description), runner.clone());
    match join::scene(&identity, &description, &params.placement, &runner, clock).await? {
        Joined::OwnHardware | Joined::Standing => readiness::serve(runner, model.id()),
        // A node that is stopping serves no readiness.
        Joined::Stopped => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_is_taken_as_the_launcher_wrote_it() {
        for bundled in robot_description::models() {
            assert_eq!(model_of(bundled.id()).unwrap().id(), bundled.id());
        }
    }

    #[test]
    fn a_model_of_no_description_is_refused_naming_the_models() {
        let models: Vec<String> = robot_description::models()
            .map(|bundled| format!("{:?}", bundled.id()))
            .collect();
        for model in ["", " ", "openarm", "OpenArm_v2"] {
            let refusal = model_of(model).unwrap_err().to_string();
            let expected = format!(
                "model: no robot model is named {model:?}; the models are {}",
                models.join(", ")
            );
            assert!(refusal.contains(&expected), "{refusal}");
        }
    }
}
