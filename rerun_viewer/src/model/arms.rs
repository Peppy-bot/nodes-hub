//! The arms and the grippers of the robot: one snapshot of their measured state, the
//! trail of each grasp point, and the names of each arm's joints and grasp link.

use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroUsize;
use std::time::SystemTime;

use crate::model::placement::Placement;

/// One arm in a snapshot: its joint positions, in its joints' order, and its grasp
/// point in the robot frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmState {
    pub name: String,
    pub joint_positions: Vec<f64>,
    pub grasp: Placement,
}

/// One gripper in a snapshot: its opening, 0 closed and 1 fully open.
#[derive(Debug, Clone, PartialEq)]
pub struct GripperState {
    pub name: String,
    pub opening: f64,
}

/// The measured state of every arm and gripper at one instant.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmsSnapshot {
    pub at: SystemTime,
    pub arms: Vec<ArmState>,
    pub grippers: Vec<GripperState>,
}

/// The past grasp-point positions of each arm, the newest last, at most `length` each.
#[derive(Debug)]
pub struct Trails {
    length: NonZeroUsize,
    by_arm: BTreeMap<String, VecDeque<[f64; 3]>>,
}

impl Trails {
    pub fn new(length: NonZeroUsize) -> Self {
        Self {
            length,
            by_arm: BTreeMap::new(),
        }
    }

    /// Adds each arm's grasp point of `snapshot` to its trail, dropping the oldest point
    /// of a full trail.
    pub fn push(&mut self, snapshot: &ArmsSnapshot) {
        for arm in &snapshot.arms {
            let trail = self.by_arm.entry(arm.name.clone()).or_default();
            if trail.len() == self.length.get() {
                trail.pop_front();
            }
            trail.push_back(arm.grasp.position);
        }
    }

    /// The trail of the arm `name`, the oldest point first; empty for an arm not seen.
    pub fn of(&self, name: &str) -> Vec<[f64; 3]> {
        self.by_arm
            .get(name)
            .map(|trail| trail.iter().copied().collect())
            .unwrap_or_default()
    }
}

/// One arm as the motion planner names it: its joints, in limb_state's order, and the
/// link at its grasp point.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmDescription {
    pub name: String,
    pub joint_names: Vec<String>,
    pub grasp_link: String,
}

/// The arms as the motion planner describes them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ArmsDescription {
    pub arms: Vec<ArmDescription>,
}

impl ArmsDescription {
    /// The name of joint `index` of the arm `arm`: the planner's name when it gives one,
    /// else `joint_<index>`.
    pub fn joint_label(&self, arm: &str, index: usize) -> String {
        self.arms
            .iter()
            .find(|described| described.name == arm)
            .and_then(|described| described.joint_names.get(index))
            .cloned()
            .unwrap_or_else(|| format!("joint_{index}"))
    }

    /// The arm whose grasp link is `link`, if any.
    pub fn arm_with_grasp_link(&self, link: &str) -> Option<&str> {
        self.arms
            .iter()
            .find(|described| described.grasp_link == link)
            .map(|described| described.name.as_str())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn snapshot(positions: &[(&str, [f64; 3])]) -> ArmsSnapshot {
        ArmsSnapshot {
            at: SystemTime::UNIX_EPOCH,
            arms: positions
                .iter()
                .map(|(name, position)| ArmState {
                    name: name.to_string(),
                    joint_positions: vec![],
                    grasp: Placement::parse(*position, [0.0, 0.0, 0.0, 1.0]).unwrap(),
                })
                .collect(),
            grippers: vec![],
        }
    }

    #[test]
    fn a_trail_keeps_the_newest_points_up_to_its_length() {
        let mut trails = Trails::new(NonZeroUsize::new(2).unwrap());
        trails.push(&snapshot(&[("left_arm", [1.0, 0.0, 0.0])]));
        trails.push(&snapshot(&[("left_arm", [2.0, 0.0, 0.0])]));
        trails.push(&snapshot(&[("left_arm", [3.0, 0.0, 0.0])]));
        assert_eq!(
            trails.of("left_arm"),
            vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn each_arm_has_its_own_trail() {
        let mut trails = Trails::new(NonZeroUsize::new(5).unwrap());
        trails.push(&snapshot(&[
            ("left_arm", [1.0, 0.0, 0.0]),
            ("right_arm", [0.0, 1.0, 0.0]),
        ]));
        assert_eq!(trails.of("left_arm"), vec![[1.0, 0.0, 0.0]]);
        assert_eq!(trails.of("right_arm"), vec![[0.0, 1.0, 0.0]]);
        assert!(trails.of("third_arm").is_empty());
    }

    /// The planner's description of the right arm of an OpenArm: one joint named.
    pub(crate) fn openarm() -> ArmsDescription {
        ArmsDescription {
            arms: vec![ArmDescription {
                name: "right_arm".to_string(),
                joint_names: vec!["openarm_right_joint1".to_string()],
                grasp_link: "openarm_right_tcp".to_string(),
            }],
        }
    }

    #[test]
    fn a_joint_takes_the_planners_name_or_its_index() {
        let description = openarm();
        assert_eq!(
            description.joint_label("right_arm", 0),
            "openarm_right_joint1"
        );
        assert_eq!(description.joint_label("right_arm", 1), "joint_1");
        assert_eq!(description.joint_label("left_arm", 0), "joint_0");
        assert_eq!(
            ArmsDescription::default().joint_label("right_arm", 3),
            "joint_3"
        );
    }

    #[test]
    fn a_grasp_link_names_its_arm() {
        let description = openarm();
        assert_eq!(
            description.arm_with_grasp_link("openarm_right_tcp"),
            Some("right_arm")
        );
        assert_eq!(description.arm_with_grasp_link("openarm_right_link7"), None);
    }
}
