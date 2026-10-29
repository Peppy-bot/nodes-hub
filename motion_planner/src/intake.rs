//! The robot as the map cuts it out of a depth frame: both arms where they stood when the
//! camera took it, their fingers at the grippers' openings then, and the body fixed around
//! them, each piece grown by the padding.

use collision_geometry::chain::{BodyGeometry, Padding, Rider};
use motion_planning_core::{PosedChain, RobotMask};
use openarm_description::{ARM_DOF, ArmPair, Side};

use crate::history::Arms;
use crate::robot::ArmModel;

/// The robot of `models` and `body`, its arms at `arms` carrying `fingers`, grown by
/// `padding`.
pub fn robot_mask<'a>(
    models: &'a ArmPair<ArmModel>,
    body: &'a BodyGeometry,
    arms: &Arms,
    fingers: &'a ArmPair<Vec<Rider<'a>>>,
    padding: Padding,
) -> RobotMask<'a, ARM_DOF> {
    let chains: Vec<PosedChain<'a, ARM_DOF>> = Side::BOTH
        .into_iter()
        .map(|side| PosedChain {
            geometry: &models[side].geometry,
            joints: arms[side],
            riders: &fingers[side],
        })
        .collect();
    RobotMask::new(&chains, body, padding)
}
