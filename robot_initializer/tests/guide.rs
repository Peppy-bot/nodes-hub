//! The robot's guide, over the wire: `get_guide` answers the park state of
//! each arm of the SRDF of the robot's description, sliced as the
//! robot_guide contract says, and empty arrays for a robot whose description
//! gives no arm a park state.

use std::time::Duration;

use peppygen::fixtures::exposed_services::robot_guide::get_guide;
use peppygen::fixtures::harness::Harness;
use robot_description::states;

mod common;
use common::{an_so101_on_its_own_hardware, on_its_own_hardware};

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_gives_the_park_states_of_its_description() -> peppygen::Result<()> {
    let (harness, _mocks) =
        Harness::start_with(on_its_own_hardware(), robot_initializer::setup).await?;
    let description = robot_description::model("openarm_v2")
        .expect("a bundled model")
        .description();
    let park = |arm: &str| {
        description
            .state(arm, states::PARK)
            .unwrap_or_else(|| panic!("openarm_v2's SRDF gives {arm} a park state"))
            .to_vec()
    };

    let guide = get_guide::poll(&harness, WIRE).await?;
    assert_eq!(guide.park_arm_names, ["left_arm", "right_arm"]);
    assert_eq!(guide.park_joints_per_arm, [7, 7]);
    assert_eq!(
        guide.park_joint_positions,
        [park("left_arm"), park("right_arm")].concat()
    );

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_whose_description_gives_no_park_state_gives_an_empty_guide() -> peppygen::Result<()>
{
    let description = robot_description::model("so101")
        .expect("a bundled model")
        .description();
    assert!(
        description
            .arms()
            .iter()
            .all(|arm| description.state(&arm.name, states::PARK).is_none()),
        "this test needs a model whose SRDF gives no arm a park state, and so101's gives one"
    );
    let (harness, _mocks) =
        Harness::start_with(an_so101_on_its_own_hardware(), robot_initializer::setup).await?;

    let guide = get_guide::poll(&harness, WIRE).await?;
    assert!(guide.park_arm_names.is_empty() && guide.park_joints_per_arm.is_empty());
    assert!(guide.park_joint_positions.is_empty());

    harness.shutdown().await
}
