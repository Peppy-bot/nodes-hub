//! Shared fixture loading for the integration tests. Builds the FK chain and the
//! SRS model directly from the bundled fixture URDF and the `base` link, the same
//! robot-agnostic entry point (`from_urdf`) production uses. `side` is
//! `"left"`/`"right"`; left vs right is just a different chain in the same URDF,
//! selected by the base link.

use srs_model::fk::ForwardKinematics;
use srs_model::model::ArmModel;

const FIXTURE_URDF: &str = include_str!("../fixtures/openarm_v10.urdf");

fn base(side: &str) -> String {
    format!("openarm_{side}_link0")
}

pub fn fk(side: &str) -> ForwardKinematics {
    ForwardKinematics::from_urdf(FIXTURE_URDF, &base(side)).expect("load fixture fk")
}

pub fn model(side: &str) -> ArmModel {
    ArmModel::from_urdf(FIXTURE_URDF, &base(side)).expect("load fixture model")
}
