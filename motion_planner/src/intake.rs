//! The robot and the planning scene's objects as the map cuts them out of a depth frame:
//! both arms where they stood when the camera took it, carrying their fingers at the
//! grippers' openings then and the objects attached to their grasp links, the body fixed
//! around them, and the objects no link carries where they stand, each piece and each
//! shape grown by the padding.

use collision_geometry::chain::{BodyGeometry, Padding, Rider};
use motion_planning_core::{PlanningScene, PosedChain, RobotMask};
use openarm_description::{ARM_DOF, ArmPair, Side};

use crate::history::ArmJoints;
use crate::robot::ArmModel;

/// The robot of `models` and `body`, its arms at `arms` carrying `carried`, each arm's
/// fingers and held objects ([`with_held_objects`](crate::fingers::with_held_objects)),
/// with the objects of `scene` that no link carries, grown by `padding`.
pub fn robot_mask<'a>(
    models: &'a ArmPair<ArmModel>,
    body: &'a BodyGeometry,
    arms: &ArmJoints,
    carried: &'a ArmPair<Vec<Rider<'a>>>,
    scene: &'a PlanningScene,
    padding: Padding,
) -> RobotMask<'a, ARM_DOF> {
    let chains: Vec<PosedChain<'a, ARM_DOF>> = Side::BOTH
        .into_iter()
        .map(|side| PosedChain {
            geometry: &models[side].geometry,
            joints: arms[side],
            riders: &carried[side],
        })
        .collect();
    RobotMask::new(&chains, body, padding).with_objects(scene.standing_shapes())
}
