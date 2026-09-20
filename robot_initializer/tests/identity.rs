//! Who the robot says it is, over the wire: `get_identity` answers the name
//! the node runs under, the model the launcher wrote and the core node
//! hosting it, which is the name and the model it joins a scene under. The
//! generated harness runs a node outside any copy, so the name it answers
//! here is its instance id; the copy's name taking over inside a copy is
//! covered by the identity module's own tests.

use std::time::Duration;

use peppygen::fixtures::exposed_services::robot_identity::get_identity;
use peppygen::fixtures::harness::Harness;

mod common;
use common::{
    an_so101_on_its_own_hardware, joining_a_simulation, on_its_own_hardware, simulation_of,
};

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);

/// The core node the harness runs the node on.
fn core_node_of(harness: &Harness) -> &str {
    harness.node_runner().processor().bound_core_node()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_on_its_own_hardware_says_who_it_is() -> peppygen::Result<()> {
    let (harness, _mocks) =
        Harness::start_with(on_its_own_hardware(), robot_initializer::setup).await?;

    let identity = get_identity::poll(&harness, WIRE).await?;
    assert_eq!(identity.robot, harness.instance_id());
    assert_eq!(identity.model, "openarm_v2");
    assert_eq!(identity.core_node, core_node_of(&harness));

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_of_any_model_says_the_model_it_was_launched_as() -> peppygen::Result<()> {
    let (harness, _mocks) =
        Harness::start_with(an_so101_on_its_own_hardware(), robot_initializer::setup).await?;

    let identity = get_identity::poll(&harness, WIRE).await?;
    assert_eq!(identity.robot, harness.instance_id());
    assert_eq!(identity.model, "so101");
    assert_eq!(identity.core_node, core_node_of(&harness));

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_joins_the_scene_under_the_name_it_answers() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v1"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let goal = simulation.attach.next_goal(WIRE).await?;

    // The engine has admitted nothing yet: who the robot is does not wait on
    // the scene, and it is the robot the attach goal names.
    let identity = get_identity::poll(&harness, WIRE).await?;
    assert_eq!(identity.robot, goal.request.robot);
    assert_eq!(identity.model, goal.request.model);
    assert_eq!(identity.robot, harness.instance_id());
    assert_eq!(identity.model, "openarm_v1");
    assert_eq!(identity.core_node, core_node_of(&harness));

    goal.reject(Some("that is enough of this robot"), None)
        .await?;
    let _ = harness.shutdown().await;
    Ok(())
}
