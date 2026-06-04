//! Shared fixture loading for the integration tests. Builds the FK chain and the
//! SRS model directly from the bundled fixture URDF and the `(base, tip)` link
//! names, the same robot-agnostic entry points (`from_urdf`) production uses.
//! `side` is `"left"`/`"right"`; left vs right is just a different chain in the
//! same URDF, selected by the link names.

use srs_model::fk::ForwardKinematics;
use srs_model::model::ArmModel;

const FIXTURE_URDF: &str = include_str!("../fixtures/openarm_v10.urdf");

fn links(side: &str) -> (String, String) {
    (
        format!("openarm_{side}_link0"),
        format!("openarm_{side}_link7"),
    )
}

pub fn fk(side: &str) -> ForwardKinematics {
    let (base, tip) = links(side);
    ForwardKinematics::from_urdf(FIXTURE_URDF, &base, &tip).expect("load fixture fk")
}

pub fn model(side: &str) -> ArmModel {
    let (base, tip) = links(side);
    ArmModel::from_urdf(FIXTURE_URDF, &base, &tip).expect("load fixture model")
}
