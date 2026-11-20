//! The arms: limb_state's snapshots, parsed and drawn as they come, and the links of
//! the robot's body placed where each snapshot's joints put them.

use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant};

use control_core::latched_log::LatchedLog;
use control_core::receiving;
use control_core::throttle::Throttle;
use peppygen::NodeRunner;
use peppygen::consumed_services::motion_planning::describe_arms;
use peppygen::consumed_topics::limb_state::limb_states;
use tracing::error;

use crate::canvas::Canvas;
use crate::draw::arms::draw_arms;
use crate::draw::robot::draw_link_poses;
use crate::model::arms::{ArmState, ArmsDescription, ArmsSnapshot, GripperState, Trails};
use crate::model::placement::Placement;
use crate::wire::{Shared, sendable, stopped};

/// A limb_states message whose arrays do not fit its names.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SnapshotError {
    #[error("joints_per_arm has {got} entries for {arms} arms")]
    JointCounts { arms: usize, got: usize },
    #[error("joint_positions has {got} values, joints_per_arm adds up to {expected}")]
    JointPositions { expected: usize, got: usize },
    #[error("grasp_points has {got} poses for {arms} arms")]
    GraspPoints { arms: usize, got: usize },
    #[error("gripper_openings has {got} values for {grippers} grippers")]
    Grippers { grippers: usize, got: usize },
    #[error("the grasp point of {arm}: {reason}")]
    Grasp { arm: String, reason: String },
}

/// The snapshot `message` holds.
pub fn parse_limb_states(message: &limb_states::Message) -> Result<ArmsSnapshot, SnapshotError> {
    let arms = message.arm_names.len();
    if message.joints_per_arm.len() != arms {
        return Err(SnapshotError::JointCounts {
            arms,
            got: message.joints_per_arm.len(),
        });
    }
    let expected = message.joints_per_arm.iter().map(|&n| n as usize).sum();
    if message.joint_positions.len() != expected {
        return Err(SnapshotError::JointPositions {
            expected,
            got: message.joint_positions.len(),
        });
    }
    if message.grasp_points.len() != arms {
        return Err(SnapshotError::GraspPoints {
            arms,
            got: message.grasp_points.len(),
        });
    }
    let grippers = message.gripper_names.len();
    if message.gripper_openings.len() != grippers {
        return Err(SnapshotError::Grippers {
            grippers,
            got: message.gripper_openings.len(),
        });
    }
    let mut joints = message.joint_positions.as_slice();
    let mut arm_states = Vec::with_capacity(arms);
    for (index, name) in message.arm_names.iter().enumerate() {
        let (own, rest) = joints.split_at(message.joints_per_arm[index] as usize);
        joints = rest;
        let point = &message.grasp_points[index];
        let grasp = Placement::parse(point.position, point.orientation).map_err(|e| {
            SnapshotError::Grasp {
                arm: name.clone(),
                reason: e.to_string(),
            }
        })?;
        arm_states.push(ArmState {
            name: name.clone(),
            joint_positions: own.to_vec(),
            grasp,
        });
    }
    let gripper_states = message
        .gripper_names
        .iter()
        .zip(&message.gripper_openings)
        .map(|(name, opening)| GripperState {
            name: name.clone(),
            opening: *opening,
        })
        .collect();
    Ok(ArmsSnapshot {
        at: message.timestamp,
        arms: arm_states,
        grippers: gripper_states,
    })
}

/// Draws the snapshots of limb_states, at most one each `draw_interval`, until the node
/// stops; each snapshot in the robot frame adds its point to the trails, drawn or not, and
/// one in another frame is dropped, the first of a run logged. With the planner linked,
/// the joints are plotted once the planner has named them.
pub async fn follow_arms(
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    trail_length: NonZeroUsize,
    draw_interval: Duration,
    shared: Arc<Shared>,
) {
    let planner_linked = describe_arms::bound_producer(&runner).is_some();
    let cancelled = stopped(&runner);
    let mut subscription = match sendable(limb_states::subscribe(&runner)).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!("limb_states: no subscription, no arms drawn: {e}");
            return;
        }
    };
    let other_frames = LatchedLog::warn("drawing limb_states");
    let mut follower = ArmsFollower {
        canvas,
        shared: shared.clone(),
        planner_linked,
        trails: Trails::new(trail_length),
        throttle: Throttle::new(draw_interval),
        unreadable: LatchedLog::warn("reading limb_states"),
        unplaced: LatchedLog::warn("placing the robot's links"),
    };
    receiving::receive(
        "limb_states",
        cancelled,
        async move || subscription.next().await,
        async move |(_producer, message)| {
            let held = shared.hold_to_robot_frame(&runner, "limb_states", &message.frame);
            match held {
                Ok(()) => {
                    other_frames.success();
                    follower.on_snapshot(&message, Instant::now());
                }
                Err(reason) => other_frames.failure(reason),
            }
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// The arms follower's state: the trails so far.
struct ArmsFollower {
    canvas: Arc<dyn Canvas>,
    shared: Arc<Shared>,
    /// Whether the motion_planning link, which names the joints, is bound.
    planner_linked: bool,
    trails: Trails,
    throttle: Throttle,
    unreadable: LatchedLog,
    unplaced: LatchedLog,
}

impl ArmsFollower {
    /// Takes the snapshot that came at `now`: its point goes to the trails, and it is
    /// drawn when the draw interval has passed since the snapshot drawn last.
    fn on_snapshot(&mut self, message: &limb_states::Message, now: Instant) {
        match parse_limb_states(message) {
            Ok(snapshot) => {
                self.unreadable.success();
                self.trails.push(&snapshot);
                if !self.throttle.admit_at(now) {
                    return;
                }
                // Without the planner, each joint is named by its index.
                let joint_names = if self.planner_linked {
                    self.shared.arms.borrow().clone()
                } else {
                    Some(ArmsDescription::default())
                };
                draw_arms(
                    self.canvas.as_ref(),
                    &snapshot,
                    &self.trails,
                    joint_names.as_ref(),
                );
                self.place_links(&snapshot);
            }
            Err(e) => self.unreadable.failure(e),
        }
    }

    /// Places the links of the robot's body for `snapshot`, once the description is
    /// read.
    fn place_links(&self, snapshot: &ArmsSnapshot) {
        let Some(body) = self.shared.body.borrow().clone() else {
            return;
        };
        match body.link_poses(snapshot) {
            Ok(poses) => {
                self.unplaced.success();
                draw_link_poses(self.canvas.as_ref(), snapshot.at, &poses);
            }
            Err(e) => self.unplaced.failure(e),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    /// A snapshot of two arms of 2 and 1 joints and one gripper.
    pub(crate) fn two_arms() -> limb_states::Message {
        limb_states::Message {
            timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(3),
            frame: "base_link".to_string(),
            arm_names: vec!["left_arm".to_string(), "right_arm".to_string()],
            joints_per_arm: vec![2, 1],
            joint_positions: vec![0.1, 0.2, 0.3],
            grasp_points: vec![
                limb_states::MessageGraspPointsItem {
                    position: [0.3, 0.2, 0.4],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                },
                limb_states::MessageGraspPointsItem {
                    position: [0.3, -0.2, 0.4],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                },
            ],
            gripper_names: vec!["right_gripper".to_string()],
            gripper_openings: vec![0.5],
        }
    }

    /// Two arms as `two_arms`, the right one's grasp point at `x`.
    fn right_arm_at(x: f64) -> limb_states::Message {
        let mut message = two_arms();
        message.grasp_points[1].position[0] = x;
        message
    }

    #[test]
    fn a_snapshot_within_the_draw_interval_is_not_drawn_and_still_adds_to_the_trail() {
        let canvas = Arc::new(crate::testing::FakeCanvas::default());
        let mut follower = ArmsFollower {
            canvas: canvas.clone(),
            shared: Arc::new(Shared::default()),
            planner_linked: false,
            trails: Trails::new(NonZeroUsize::new(10).unwrap()),
            throttle: Throttle::new(Duration::from_millis(100)),
            unreadable: LatchedLog::warn("reading limb_states"),
            unplaced: LatchedLog::warn("placing the robot's links"),
        };
        let start = Instant::now();
        follower.on_snapshot(&right_arm_at(0.30), start);
        follower.on_snapshot(&right_arm_at(0.31), start + Duration::from_millis(50));
        follower.on_snapshot(&right_arm_at(0.32), start + Duration::from_millis(100));
        let trails: Vec<Vec<[f64; 3]>> = canvas
            .at("world/arms/right_arm/trail")
            .into_iter()
            .map(|drawn| match drawn {
                crate::drawn::Drawn::LineStrip { points, .. } => points,
                other => panic!("a trail, not {other:?}"),
            })
            .collect();
        assert_eq!(
            trails,
            [
                vec![[0.30, -0.2, 0.4]],
                vec![[0.30, -0.2, 0.4], [0.31, -0.2, 0.4], [0.32, -0.2, 0.4]]
            ],
            "the first and the third snapshot are drawn; the third trail holds all three"
        );
    }

    #[test]
    fn a_snapshot_splits_the_joints_and_poses_of_each_arm() {
        let snapshot = parse_limb_states(&two_arms()).expect("a valid message");
        assert_eq!(snapshot.at, SystemTime::UNIX_EPOCH + Duration::from_secs(3));
        assert_eq!(snapshot.arms[0].name, "left_arm");
        assert_eq!(snapshot.arms[0].joint_positions, [0.1, 0.2]);
        assert_eq!(snapshot.arms[0].grasp.position, [0.3, 0.2, 0.4]);
        assert_eq!(snapshot.arms[1].name, "right_arm");
        assert_eq!(snapshot.arms[1].joint_positions, [0.3]);
        assert_eq!(snapshot.arms[1].grasp.position, [0.3, -0.2, 0.4]);
        assert_eq!(
            snapshot.grippers,
            [GripperState {
                name: "right_gripper".to_string(),
                opening: 0.5
            }]
        );
    }

    #[test]
    fn each_array_that_does_not_fit_the_names_is_refused() {
        let mut message = two_arms();
        message.joints_per_arm = vec![2];
        assert_eq!(
            parse_limb_states(&message),
            Err(SnapshotError::JointCounts { arms: 2, got: 1 })
        );
        let mut message = two_arms();
        message.joint_positions.pop();
        assert_eq!(
            parse_limb_states(&message),
            Err(SnapshotError::JointPositions {
                expected: 3,
                got: 2
            })
        );
        let mut message = two_arms();
        message.grasp_points.pop();
        assert_eq!(
            parse_limb_states(&message),
            Err(SnapshotError::GraspPoints { arms: 2, got: 1 })
        );
        let mut message = two_arms();
        message.gripper_openings.clear();
        assert_eq!(
            parse_limb_states(&message),
            Err(SnapshotError::Grippers {
                grippers: 1,
                got: 0
            })
        );
        let mut message = two_arms();
        message.grasp_points[0].orientation[3] = 2.0;
        assert!(matches!(
            parse_limb_states(&message),
            Err(SnapshotError::Grasp { arm, .. }) if arm == "left_arm"
        ));
    }
}
