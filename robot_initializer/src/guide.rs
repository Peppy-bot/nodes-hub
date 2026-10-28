//! The guide to this robot: the park pose of each arm that the robot's
//! launcher wrote one for. The node reads the poses once, when it starts, and
//! refuses to start on a value it cannot read, so `get_guide` answers only
//! what the launcher meant.
//!
//! A launcher writes the poses as one string, because a node parameter is a
//! primitive or a group: an entry `<arm>:<joint>,<joint>,...` for each arm
//! that has a park pose, and `;` between entries.

use std::sync::Arc;

use control_core::serving;
use peppygen::exposed_services::robot_guide::get_guide;
use peppygen::{NodeRunner, Result};
use peppylib::runtime::CancellationToken;

use crate::refused;

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
    /// The guide the launcher wrote in the `park_poses` parameter, or the
    /// refusal that names the value it cannot read. An empty value has no
    /// entry. An entry with no arm or no joint, a joint that is not a finite
    /// number and an arm named twice are refused.
    pub fn from_park_poses(park_poses: &str) -> Result<Self> {
        if park_poses.trim().is_empty() {
            return Ok(Self::default());
        }
        let mut poses: Vec<ParkPose> = Vec::new();
        for entry in park_poses.split(';') {
            let pose = park_pose(entry)?;
            if poses.iter().any(|seen| seen.arm == pose.arm) {
                return Err(refused(format!(
                    "park_poses names the arm '{}' twice: {park_poses:?}",
                    pose.arm
                )));
            }
            poses.push(pose);
        }
        Ok(Self { park_poses: poses })
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

/// One entry `<arm>:<joint>,<joint>,...` of `park_poses`.
fn park_pose(entry: &str) -> Result<ParkPose> {
    let shape = || {
        refused(format!(
            "park_poses holds `<arm>:<joint>,<joint>,...` for each arm, with `;` between arms, and one entry is {entry:?}"
        ))
    };
    let (arm, joints) = entry.split_once(':').ok_or_else(shape)?;
    let arm = arm.trim();
    if arm.is_empty() || joints.trim().is_empty() {
        return Err(shape());
    }
    let joint_positions = joints
        .split(',')
        .map(finite_number)
        .collect::<Option<Vec<f64>>>()
        .ok_or_else(|| {
            refused(format!(
                "park_poses gives '{arm}' a joint that is not a finite number: {entry:?}"
            ))
        })?;
    Ok(ParkPose {
        arm: arm.to_owned(),
        joint_positions,
    })
}

fn finite_number(text: &str) -> Option<f64> {
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
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
        get_guide::handle_next_request(&runner, |_req| Ok(guide.response()))
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEFT: &str = "left_arm:0.0584,-0.4923,-1.0367,1.7119,-0.4582,-0.1219,-1.1645";

    #[test]
    fn an_empty_value_gives_an_empty_guide() {
        for empty in ["", " ", "\t\n"] {
            assert_eq!(Guide::from_park_poses(empty).unwrap(), Guide::default());
        }
    }

    #[test]
    fn each_arm_gets_its_park_pose_in_the_order_written() {
        let guide =
            Guide::from_park_poses(&format!(" {LEFT} ; right_arm: -0.4474, 0.7605 ")).unwrap();
        assert_eq!(
            guide.park_poses,
            [
                ParkPose {
                    arm: "left_arm".to_owned(),
                    joint_positions: vec![
                        0.0584, -0.4923, -1.0367, 1.7119, -0.4582, -0.1219, -1.1645
                    ],
                },
                ParkPose {
                    arm: "right_arm".to_owned(),
                    joint_positions: vec![-0.4474, 0.7605],
                },
            ]
        );
    }

    #[test]
    fn the_answer_concatenates_the_joints_as_the_contract_slices_them() {
        let answer = Guide::from_park_poses("left_arm:1,2,3;right_arm:4,5")
            .unwrap()
            .response();
        assert_eq!(answer.park_arm_names, ["left_arm", "right_arm"]);
        assert_eq!(answer.park_joints_per_arm, [3, 2]);
        assert_eq!(answer.park_joint_positions, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// The refusal of `park_poses`, as the node starts.
    fn refusal(park_poses: &str) -> String {
        Guide::from_park_poses(park_poses).unwrap_err().to_string()
    }

    #[test]
    fn an_entry_without_an_arm_or_joints_is_refused_with_the_shape_of_an_entry() {
        for park_poses in [
            "0.1,0.2",
            ":0.1,0.2",
            "left_arm:",
            "left_arm:0.1;",
            "left_arm 0.1",
        ] {
            let refusal = refusal(park_poses);
            assert!(
                refusal.contains("park_poses holds `<arm>:<joint>,<joint>,...` for each arm"),
                "{park_poses:?}: {refusal}"
            );
        }
    }

    #[test]
    fn a_joint_that_is_not_a_finite_number_is_refused() {
        for park_poses in [
            "left_arm:0.1,x",
            "left_arm:0.1,,0.2",
            "left_arm:inf",
            "left_arm:NaN",
        ] {
            let refusal = refusal(park_poses);
            assert!(
                refusal.contains("park_poses gives 'left_arm' a joint that is not a finite number"),
                "{park_poses:?}: {refusal}"
            );
        }
    }

    #[test]
    fn an_arm_named_twice_is_refused() {
        let refusal = refusal("left_arm:0.1;left_arm:0.2");
        assert!(
            refusal.contains("park_poses names the arm 'left_arm' twice"),
            "{refusal}"
        );
    }
}
