//! What both of this node's tests boot a robot with. Each test binary
//! compiles its own copy, so a builder only one of them calls is unused in
//! the other.
#![allow(dead_code)]

use std::time::Duration;

use peppygen::Parameters;
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::simulation::attach;
use peppygen::parameters::placement::Placement;

/// Joints of one OpenArm arm, as the engine reports them.
const ARM_DOF: u32 = 7;

/// How long a test waits for the node's `setup` to return.
const SETUP_BUDGET: Duration = Duration::from_secs(30);

/// How often a test checks whether the node's `setup` has returned.
const SETUP_POLL: Duration = Duration::from_millis(50);

/// The four limbs of an OpenArm, each answering for itself.
pub const LIMBS: usize = 4;

/// A robot of the OpenArm generation `hardware_version`, standing wherever
/// a simulation parks it.
pub fn parameters(hardware_version: &str) -> Parameters {
    Parameters {
        hardware_version: hardware_version.into(),
        placement: Placement {
            auto: true,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            yaw: 0.0,
        },
    }
}

/// The same robot, standing at the placement it is given.
pub fn standing_at(hardware_version: &str, x: f64, y: f64, z: f64, yaw: f64) -> Parameters {
    Parameters {
        placement: Placement {
            auto: false,
            x,
            y,
            z,
            yaw,
        },
        ..parameters(hardware_version)
    }
}

/// A robot that joins a simulation: the `simulation` slot is bound and the
/// robot has no drivers of its own, so the node attaches before it does
/// anything else and the simulation answers for its readiness.
pub fn joining_a_simulation(hardware_version: &str) -> Config {
    Config {
        parameters: Some(parameters(hardware_version)),
        ..Config::default()
    }
}

/// A robot with no simulation bound: the `simulation` slot is vacant, so the
/// node joins no scene and serves readiness alone, over four limb instances.
pub fn on_its_own_hardware() -> Config {
    Config {
        parameters: Some(parameters("v2")),
        simulation_vacant: true,
        limbs_instances: LIMBS,
        ..Config::default()
    }
}

/// The engine's answer to a robot it stood: the model's limbs, listed right
/// before left, which is not the order of this robot's own slots.
pub fn stood() -> attach::GoalResponseData {
    attach::GoalResponseData {
        arm_names: vec!["right".into(), "left".into()],
        arm_joints: vec![ARM_DOF, ARM_DOF],
        gripper_names: vec!["right".into(), "left".into()],
    }
}

/// The engine's mock, which a robot joining a simulation always has bound.
pub fn simulation_of(
    mocks: &mut peppygen::fixtures::harness::Mocks,
) -> peppygen::mock::deps::simulation::Mock {
    mocks
        .deps
        .simulation
        .take()
        .expect("a robot joining a simulation has one bound")
}

/// Waits for the node's `setup` to return, failing the test after
/// `SETUP_BUDGET`.
pub async fn await_setup_return(harness: &Harness) {
    let deadline = tokio::time::Instant::now() + SETUP_BUDGET;
    while !harness.setup_finished() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "setup must return within {SETUP_BUDGET:?}"
        );
        tokio::time::sleep(SETUP_POLL).await;
    }
}

/// Tears the harness down once the node's `setup` has returned, so a setup
/// error reaches the caller. Teardown aborts a setup still running after the
/// shutdown grace and reports it as a clean stop.
pub async fn shutdown_once_setup_returns(harness: Harness) -> peppygen::Result<()> {
    await_setup_return(&harness).await;
    harness.shutdown().await
}
