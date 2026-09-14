//! Bringing one OpenArm into being, and saying when it is ready.
//!
//! A robot that joins a simulation takes its seat there first: it attaches as
//! its own model, its limbs' setpoints go to the engine as one command, and
//! the state that comes back is published on the limb it measures. The engine
//! tells its robots apart by the seat that commands them, so everything
//! robot-facing here is the surface a simulation offered when it hosted one.
//!
//! Every robot then serves the readiness its four limbs aggregate to, which
//! the backbone gates on.

#![forbid(unsafe_code)]

use std::sync::Arc;

use peppygen::{NodeRunner, Parameters, Result};

mod limbs;
mod readiness;
mod seat;

/// The robot's own error: something about this robot, rather than a fault in
/// the machinery that carries it.
pub(crate) fn refused(message: impl Into<String>) -> peppygen::Error {
    peppygen::Error::Node(std::io::Error::other(message.into()).into())
}

pub async fn setup(params: Parameters, runner: Arc<NodeRunner>) -> Result<()> {
    let model = limbs::model_of(&params.hardware_version).map_err(refused)?;
    seat::take(model, &params, &runner).await?;
    readiness::serve(runner);
    Ok(())
}
