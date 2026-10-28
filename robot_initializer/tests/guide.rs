//! The robot's guide, over the wire: `get_guide` answers the park poses the
//! launcher wrote, sliced as the robot_guide contract says, and empty arrays
//! for a robot whose launcher wrote none. A value the node cannot read stops
//! it before it serves anything; the guide module's own tests cover each
//! refusal.

use std::time::Duration;

use peppygen::fixtures::exposed_services::robot_guide::get_guide;
use peppygen::fixtures::harness::{Config, Harness};

mod common;
use common::{await_setup_return, on_its_own_hardware, parameters};

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);

/// How long a service that nothing serves is waited on before concluding it
/// is not being served.
const UNSERVED: Duration = Duration::from_millis(500);

/// An OpenArm on its own hardware whose launcher wrote `park_poses`.
fn guided(park_poses: &str) -> Config {
    let mut config = on_its_own_hardware();
    config.parameters = Some(peppygen::Parameters {
        park_poses: park_poses.into(),
        ..parameters("openarm_v2")
    });
    config
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_gives_the_park_poses_its_launcher_wrote() -> peppygen::Result<()> {
    let config = guided(
        "left_arm:0.0584,-0.4923,-1.0367,1.7119,-0.4582,-0.1219,-1.1645;right_arm:-0.4474,0.7605,0.8815,1.6469,0.8363,-0.2129,0.9232",
    );
    let (harness, _mocks) = Harness::start_with(config, robot_initializer::setup).await?;

    let guide = get_guide::poll(&harness, WIRE).await?;
    assert_eq!(guide.park_arm_names, ["left_arm", "right_arm"]);
    assert_eq!(guide.park_joints_per_arm, [7, 7]);
    assert_eq!(
        guide.park_joint_positions,
        [
            0.0584, -0.4923, -1.0367, 1.7119, -0.4582, -0.1219, -1.1645, -0.4474, 0.7605, 0.8815,
            1.6469, 0.8363, -0.2129, 0.9232
        ]
    );

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_whose_launcher_wrote_no_guide_gives_an_empty_one() -> peppygen::Result<()> {
    let (harness, _mocks) =
        Harness::start_with(on_its_own_hardware(), robot_initializer::setup).await?;

    let guide = get_guide::poll(&harness, WIRE).await?;
    assert!(guide.park_arm_names.is_empty() && guide.park_joints_per_arm.is_empty());
    assert!(guide.park_joint_positions.is_empty());

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_whose_park_poses_the_node_cannot_read_does_not_start() -> peppygen::Result<()> {
    let (harness, _mocks) =
        Harness::start_with(guided("left_arm:0.1,x"), robot_initializer::setup).await?;

    // Setup has returned, so a guide it never served never answers.
    await_setup_return(&harness).await;
    assert!(
        get_guide::poll(&harness, UNSERVED).await.is_err(),
        "a robot that refused its park poses serves no guide"
    );
    let failure = harness
        .shutdown()
        .await
        .expect_err("a robot whose park poses the node cannot read fails to start")
        .to_string();
    assert!(
        failure.contains("park_poses gives 'left_arm' a joint that is not a finite number"),
        "{failure}"
    );
    Ok(())
}
