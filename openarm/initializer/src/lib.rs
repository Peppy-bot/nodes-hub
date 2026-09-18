//! Bringing one OpenArm into being, and saying when it is ready.
//!
//! Every robot serves the readiness the backbone gates on. On a real robot
//! the drivers of its limbs answer for it, one each; on a simulated robot
//! the simulation standing it answers for every limb at once.
//!
//! A robot that runs in a simulation also joins its scene, which is the one
//! thing a simulated robot does that a real one does not.

#![forbid(unsafe_code)]

use std::sync::Arc;

use peppygen::{NodeRunner, Parameters, Result};

mod join;
mod readiness;

/// The robot's own error: a fact about this robot, which is what a launcher
/// reads back when the node refuses to start.
pub(crate) fn refused(message: impl Into<String>) -> peppygen::Error {
    peppygen::Error::Node(std::io::Error::other(message.into()).into())
}

/// The OpenArm generation this robot is, as the launcher wrote it.
fn generation(hardware_version: &str) -> Result<&str> {
    match hardware_version {
        "v1" | "v2" => Ok(hardware_version),
        other => Err(refused(format!(
            "hardware_version names an OpenArm generation, v1 or v2, and this robot's is {other:?}"
        ))),
    }
}

pub async fn setup(params: Parameters, runner: Arc<NodeRunner>) -> Result<()> {
    let generation = generation(&params.hardware_version)?;
    join::scene(generation, &params.placement, &runner).await?;
    readiness::serve(runner)
}
