//! The arms: each grasp point with its axes and its trail, and the joints and the
//! grippers as plots.

use crate::canvas::{Canvas, Rgba};
use crate::model::arms::{ArmsDescription, ArmsSnapshot, Trails};

/// The length of each axis drawn at a grasp point (m).
const GRASP_AXIS_LENGTH: f32 = 0.08;

const TRAIL_COLOR: Rgba = [255, 190, 0, 255];

/// The entity of the grasp point of the arm `arm`.
pub fn grasp_path(arm: &str) -> String {
    format!("world/arms/{arm}/grasp")
}

pub fn trail_path(arm: &str) -> String {
    format!("world/arms/{arm}/trail")
}

pub fn joint_path(arm: &str, joint: &str) -> String {
    format!("plots/joints/{arm}/{joint}")
}

pub fn gripper_path(gripper: &str) -> String {
    format!("plots/grippers/{gripper}")
}

/// Draws `snapshot` at its time: each arm's grasp point, its trail in `trails`, and,
/// when `joint_names` is known, its joints by those names, then each gripper's opening.
pub fn draw_arms(
    canvas: &dyn Canvas,
    snapshot: &ArmsSnapshot,
    trails: &Trails,
    joint_names: Option<&ArmsDescription>,
) {
    canvas.set_time(snapshot.at);
    for arm in &snapshot.arms {
        canvas.frame(&grasp_path(&arm.name), &arm.grasp.pose, GRASP_AXIS_LENGTH);
        canvas.line_strip(&trail_path(&arm.name), &trails.of(&arm.name), TRAIL_COLOR);
        let Some(description) = joint_names else {
            continue;
        };
        for (index, position) in arm.joint_positions.iter().enumerate() {
            let joint = description.joint_label(&arm.name, index);
            canvas.scalar(&joint_path(&arm.name, &joint), *position);
        }
    }
    for gripper in &snapshot.grippers {
        canvas.scalar(&gripper_path(&gripper.name), gripper.opening);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::arms::tests::openarm;
    use crate::model::arms::{ArmState, GripperState};
    use crate::model::placement::Placement;
    use crate::testing::{Drawn, FakeCanvas};
    use std::num::NonZeroUsize;
    use std::time::{Duration, SystemTime};

    #[test]
    fn a_snapshot_draws_each_arm_and_each_gripper_at_its_time() {
        let at = SystemTime::UNIX_EPOCH + Duration::from_secs(7);
        let snapshot = ArmsSnapshot {
            at,
            arms: vec![ArmState {
                name: "right_arm".to_string(),
                joint_positions: vec![0.5, -0.25],
                grasp: Placement::parse([0.3, -0.2, 0.4], [0.0, 0.0, 0.0, 1.0]).unwrap(),
            }],
            grippers: vec![GripperState {
                name: "right_gripper".to_string(),
                opening: 0.75,
            }],
        };
        let mut trails = Trails::new(NonZeroUsize::new(10).unwrap());
        trails.push(&snapshot);
        let canvas = FakeCanvas::default();
        draw_arms(&canvas, &snapshot, &trails, Some(&openarm()));
        let drawn = canvas.drawn();
        assert!(matches!(drawn[0], Drawn::Time(time) if time == at));
        let Drawn::Frame { path, pose, .. } = &drawn[1] else {
            panic!("the grasp point first, got {:?}", drawn[1])
        };
        assert_eq!(path, "world/arms/right_arm/grasp");
        assert_eq!(pose.origin(), [0.3, -0.2, 0.4]);
        assert!(matches!(
            &drawn[2],
            Drawn::LineStrip { path, points, .. }
                if path == "world/arms/right_arm/trail" && points == &[[0.3, -0.2, 0.4]]
        ));
        let scalars: Vec<(String, f64)> = drawn
            .iter()
            .filter_map(|drawn| match drawn {
                Drawn::Scalar { path, value } => Some((path.clone(), *value)),
                _ => None,
            })
            .collect();
        assert_eq!(
            scalars,
            [
                (
                    "plots/joints/right_arm/openarm_right_joint1".to_string(),
                    0.5
                ),
                ("plots/joints/right_arm/joint_1".to_string(), -0.25),
                ("plots/grippers/right_gripper".to_string(), 0.75),
            ]
        );
    }

    #[test]
    fn the_joints_wait_for_their_names() {
        let snapshot = ArmsSnapshot {
            at: SystemTime::UNIX_EPOCH,
            arms: vec![ArmState {
                name: "right_arm".to_string(),
                joint_positions: vec![0.5],
                grasp: Placement::parse([0.3, -0.2, 0.4], [0.0, 0.0, 0.0, 1.0]).unwrap(),
            }],
            grippers: vec![],
        };
        let trails = Trails::new(NonZeroUsize::new(10).unwrap());
        let canvas = FakeCanvas::default();
        draw_arms(&canvas, &snapshot, &trails, None);
        assert!(
            !canvas
                .drawn()
                .iter()
                .any(|drawn| matches!(drawn, Drawn::Scalar { .. }))
        );
        assert_eq!(canvas.at("world/arms/right_arm/grasp").len(), 1);
    }
}
