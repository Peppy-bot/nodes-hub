//! Who this robot is: the name it stands under, the model it is and the core
//! node hosting it. The node works it out once, when it starts, and both the
//! scene it joins and whoever asks `get_identity` read that one answer, so a
//! robot is never known under two names.

use std::sync::Arc;

use peppygen::NodeRunner;
use peppygen::exposed_services::robot_identity::get_identity;
use peppylib::runtime::CancellationToken;

/// One robot's identity, fixed for as long as the node runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// The name this robot stands under everywhere: the copy the launch put
    /// it in, or its own instance id for a robot launched outside one. It is
    /// the name the engine reads on every one of this robot's limb pairs,
    /// which is how it tells one robot's limbs from another's.
    pub robot: String,
    /// The model of this robot's generation, as a simulation's catalogue
    /// names it.
    pub model: String,
    /// The core node hosting this robot.
    pub core_node: String,
}

impl Identity {
    /// The identity of the robot `runner` runs, of the OpenArm `generation`.
    pub fn of(runner: &NodeRunner, generation: &str) -> Self {
        let processor = runner.processor();
        Self::named(
            runner.copy(),
            processor.bound_instance_id(),
            processor.bound_core_node(),
            generation,
        )
    }

    /// A robot runs under its copy's name, and under its own instance id
    /// when the launch put it in no copy.
    fn named(copy: Option<&str>, instance_id: &str, core_node: &str, generation: &str) -> Self {
        Self {
            robot: copy.unwrap_or(instance_id).to_owned(),
            model: format!("openarm_{generation}"),
            core_node: core_node.to_owned(),
        }
    }
}

/// Starts serving `get_identity`, for as long as the node runs.
pub fn serve(identity: Identity, runner: Arc<NodeRunner>) {
    let token = runner.cancellation_token().clone();
    tokio::spawn(run(identity, runner, token));
}

async fn run(identity: Identity, runner: Arc<NodeRunner>, token: CancellationToken) {
    tracing::info!(
        "get_identity service started for '{}', a {} on {}",
        identity.robot,
        identity.model,
        identity.core_node
    );
    loop {
        tokio::select! {
            _ = token.cancelled() => {
                tracing::info!("get_identity service shutting down");
                break;
            }
            result = get_identity::handle_next_request(&runner, |_req| {
                Ok(get_identity::Response::new(
                    identity.robot.clone(),
                    identity.model.clone(),
                    identity.core_node.clone(),
                ))
            }) => {
                if let Err(e) = result {
                    tracing::warn!("get_identity handler error: {e}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_robot_in_a_copy_runs_under_the_copy_name() {
        assert_eq!(
            Identity::named(Some("bravo"), "bravo_init_inst", "jetson-1", "v2"),
            Identity {
                robot: "bravo".to_owned(),
                model: "openarm_v2".to_owned(),
                core_node: "jetson-1".to_owned(),
            }
        );
    }

    #[test]
    fn a_robot_outside_a_copy_runs_under_its_instance_id() {
        assert_eq!(
            Identity::named(None, "init_inst", "jetson-1", "v1"),
            Identity {
                robot: "init_inst".to_owned(),
                model: "openarm_v1".to_owned(),
                core_node: "jetson-1".to_owned(),
            }
        );
    }
}
