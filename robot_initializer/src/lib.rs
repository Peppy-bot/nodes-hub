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
//! under.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Duration;

use peppygen::{NodeRunner, Parameters, Result};

mod identity;
mod join;
mod readiness;

use identity::Identity;
use join::Joined;
pub use join::{HostClock, Moment, StandClock};

/// The robot's own error: a fact about this robot, which is what a launcher
/// reads back when the node refuses to start.
pub(crate) fn refused(message: impl Into<String>) -> peppygen::Error {
    peppygen::Error::Node(std::io::Error::other(message.into()).into())
}

/// The model this robot is, as the launcher wrote it. Which models exist is
/// a simulation's to say: it refuses one it does not stand, naming the ones
/// it does, so the only model refused here is one that names nothing.
fn model_of(model: &str) -> Result<&str> {
    if model.trim().is_empty() {
        return Err(refused(format!(
            "model names the robot's model as a simulation's catalogue does, for example openarm_v2 or so101, and this robot's is {model:?}"
        )));
    }
    Ok(model)
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
    let identity = Identity::of(&runner, model_of(&params.model)?);
    identity::serve(identity.clone(), runner.clone());
    match join::scene(&identity, &params.placement, &runner, clock).await? {
        Joined::OwnHardware | Joined::Standing => readiness::serve(runner),
        // A node that is stopping serves no readiness.
        Joined::Stopped => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_is_taken_as_the_launcher_wrote_it() {
        for model in ["openarm_v2", "so101"] {
            assert_eq!(model_of(model).unwrap(), model);
        }
    }

    #[test]
    fn a_model_that_names_nothing_is_refused() {
        for model in ["", " ", "\t\n"] {
            let refusal = model_of(model).unwrap_err().to_string();
            assert!(
                refusal.contains("model names the robot's model"),
                "{refusal}"
            );
        }
    }
}
