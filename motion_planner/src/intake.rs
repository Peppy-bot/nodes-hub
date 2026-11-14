//! The robot and the planning scene's objects as the map cuts them out of a depth frame:
//! each arm where it stood when the camera took it, carrying its fingers at the gripper's
//! opening then and the objects attached to its grasp link, the body fixed around them, and
//! the objects no link carries where they stand, each piece and each shape grown by the
//! padding.

use collision_geometry::chain::{BodyGeometry, Padding, Rider};
use motion_planning_core::{PlanningScene, PosedChain, RobotMask};
use robot_description::PerArm;

use crate::history::ArmJoints;
use crate::robot::ArmModel;

/// The robot of `models` and `body`, its arms at `arms` carrying `carried`, each arm's
/// fingers and held objects ([`with_held_objects`](crate::fingers::with_held_objects)),
/// with the objects of `scene` that no link carries, grown by `padding`.
pub fn robot_mask<'a, const N: usize, const K: usize>(
    models: &'a PerArm<ArmModel<N>, K>,
    body: &'a BodyGeometry,
    arms: &ArmJoints<N, K>,
    carried: &'a PerArm<Vec<Rider<'a>>, K>,
    scene: &'a PlanningScene,
    padding: Padding,
) -> RobotMask<'a, N> {
    let chains: Vec<PosedChain<'a, N>> = PerArm::<(), K>::ids()
        .map(|arm| PosedChain {
            geometry: &models[arm].geometry,
            joints: arms[arm],
            riders: &carried[arm],
        })
        .collect();
    RobotMask::new(&chains, body, padding).with_objects(scene.standing_shapes())
}
