//! The arms: each grasp point with its axes and its trail, and the joints and the
//! grippers as plots, one plot for each arm with a line for each joint, and one plot
//! with a line for each gripper.

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

/// The root of the plots of the arms' joints.
pub const JOINT_PLOTS_PATH: &str = "plots/joints";

/// The plot of the joints of the arm `arm`: one line for each joint.
pub fn joints_path(arm: &str) -> String {
    format!("{JOINT_PLOTS_PATH}/{arm}")
}

/// The plot of the grippers' openings: one line for each gripper.
pub const GRIPPERS_PATH: &str = "plots/grippers";

/// Draws `snapshot` at its time: each arm's grasp point, its trail in `trails`, and,
/// when `joint_names` is known, its joints as the lines of its plot, by those names,
/// then the grippers' openings as the lines of theirs.
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
        let names: Vec<String> = (0..arm.joint_positions.len())
            .map(|index| description.joint_label(&arm.name, index))
            .collect();
        let plot = joints_path(&arm.name);
        canvas.series_names(&plot, &names);
        canvas.scalars(&plot, &arm.joint_positions);
    }
    if snapshot.grippers.is_empty() {
        return;
    }
    let (names, openings): (Vec<String>, Vec<f64>) = snapshot
        .grippers
        .iter()
        .map(|gripper| (gripper.name.clone(), gripper.opening))
        .unzip();
    canvas.series_names(GRIPPERS_PATH, &names);
    canvas.scalars(GRIPPERS_PATH, &openings);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drawn::Drawn;
    use crate::model::arms::tests::openarm;
    use crate::model::arms::{ArmState, GripperState};
    use crate::model::placement::Placement;
    use crate::testing::FakeCanvas;
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
        let names = |names: &[&str]| {
            names
                .iter()
                .map(|name| name.to_string())
                .collect::<Vec<_>>()
        };
        assert!(matches!(
            &drawn[3..],
            [
                Drawn::SeriesNames { path: joints, names: joint_names },
                Drawn::Scalars { path: joint_values_at, values: joint_values },
                Drawn::SeriesNames { path: grippers, names: gripper_names },
                Drawn::Scalars { path: gripper_values_at, values: gripper_values },
            ] if joints == "plots/joints/right_arm"
                && joint_names == &names(&["openarm_right_joint1", "joint_1"])
                && joint_values_at == "plots/joints/right_arm"
                && joint_values == &[0.5, -0.25]
                && grippers == "plots/grippers"
                && gripper_names == &names(&["right_gripper"])
                && gripper_values_at == "plots/grippers"
                && gripper_values == &[0.75]
        ));
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
                .any(|drawn| matches!(drawn, Drawn::Scalars { .. }))
        );
        assert_eq!(canvas.at("world/arms/right_arm/grasp").len(), 1);
    }
}
