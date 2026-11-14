//! The guide to this robot: the park pose of each arm whose group has a
//! `park` state in the SRDF of the robot's description. The node reads them
//! once, when it starts.

use std::sync::Arc;

use control_core::serving;
use peppygen::NodeRunner;
use peppygen::exposed_services::robot_guide::get_guide;
use peppylib::runtime::CancellationToken;
use robot_description::{RobotDescription, states};

/// The joints of one arm at its park pose.
#[derive(Debug, Clone, PartialEq)]
pub struct ParkPose {
    pub arm: String,
    pub joint_positions: Vec<f64>,
}

/// The guide to one robot, fixed for as long as the node runs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Guide {
    pub park_poses: Vec<ParkPose>,
}

impl Guide {
    /// The park pose of each arm of `description` that has one, in the
    /// description's arm order.
    pub fn of(description: &RobotDescription) -> Self {
        let park_poses = description
            .arms()
            .iter()
            .filter_map(|arm| {
                description
                    .state(&arm.name, states::PARK)
                    .map(|joints| ParkPose {
                        arm: arm.name.clone(),
                        joint_positions: joints.to_vec(),
                    })
            })
            .collect();
        Self { park_poses }
    }

    /// The answer of `get_guide`: the arrays of the robot_guide contract.
    fn response(&self) -> get_guide::Response {
        get_guide::Response::new(
            self.park_poses
                .iter()
                .map(|pose| pose.arm.clone())
                .collect(),
            self.park_poses
                .iter()
                .map(|pose| u32::try_from(pose.joint_positions.len()).unwrap_or(u32::MAX))
                .collect(),
            self.park_poses
                .iter()
                .flat_map(|pose| pose.joint_positions.iter().copied())
                .collect(),
        )
    }

    /// The arms that have a park pose, for the log line of the start.
    fn arms(&self) -> String {
        let arms: Vec<&str> = self
            .park_poses
            .iter()
            .map(|pose| pose.arm.as_str())
            .collect();
        arms.join(", ")
    }
}

/// Starts serving `get_guide`, for as long as the node runs.
pub fn serve(guide: Guide, runner: Arc<NodeRunner>) {
    let token = runner.cancellation_token().clone();
    tokio::spawn(run(guide, runner, token));
}

async fn run(guide: Guide, runner: Arc<NodeRunner>, token: CancellationToken) {
    tracing::info!(
        "get_guide service started: park poses for [{}]",
        guide.arms()
    );
    serving::serve("get_guide", token.cancelled(), || {
        get_guide::handle_requests(&runner, |_req| Ok(guide.response()))
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_concatenates_the_joints_as_the_contract_slices_them() {
        let pose = |arm: &str, joint_positions: &[f64]| ParkPose {
            arm: arm.to_owned(),
            joint_positions: joint_positions.to_vec(),
        };
        let answer = Guide {
            park_poses: vec![
                pose("left_arm", &[1.0, 2.0, 3.0]),
                pose("right_arm", &[4.0, 5.0]),
            ],
        }
        .response();
        assert_eq!(answer.park_arm_names, ["left_arm", "right_arm"]);
        assert_eq!(answer.park_joints_per_arm, [3, 2]);
        assert_eq!(answer.park_joint_positions, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
