//! The transforms of the robot's fixed joints, over the wire: the node
//! publishes one static_transforms message once it holds the model, before
//! a simulation stands the robot, holding the transform record of every fixed
//! joint of the model's URDF and of no other joint, with no timestamp since
//! each holds at every time. The topic retains it, so a node that subscribes after
//! the publication receives it too.

use std::time::Duration;

use peppygen::emitted_topics::transforms::static_transforms::{self, MessageTransformsItem};
use peppygen::fixtures::emitted_topics::transforms::static_transforms::Message;
use peppygen::fixtures::harness::Harness;
use peppygen::mock::deps::simulation::attach;
use robot_description::chain_kinematics::nalgebra::UnitQuaternion;
use robot_description::urdf_rs::JointType;

mod common;
use common::{
    an_so101_on_its_own_hardware, joining_a_simulation, on_its_own_hardware, simulation_of,
};

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);

/// How far a published value may stand from the URDF's: the rounding of a
/// pose built from a URDF origin.
const TOLERANCE: f64 = 1e-12;

/// The message the node published first.
async fn first_message(harness: &mut Harness) -> Message {
    tokio::time::timeout(WIRE, harness.emitted.transforms_static_transforms.next())
        .await
        .expect("static_transforms is published within the wire's time")
        .expect("the subscription reads")
        .expect("the subscription stays open")
}

/// Asserts that `message` holds the transform record of every fixed joint of
/// the URDF of `model`, in the URDF's order, at the joint's origin, and no
/// other.
fn assert_holds_the_fixed_joints_of(model: &str, message: &Message) {
    let description = robot_description::model(model)
        .expect("a bundled model")
        .description();
    let fixed: Vec<_> = description
        .robot()
        .joints
        .iter()
        .filter(|joint| joint.joint_type == JointType::Fixed)
        .collect();
    assert!(!fixed.is_empty(), "{model} has fixed joints");
    assert_eq!(message.transforms.len(), fixed.len(), "{model}");
    for (record, joint) in message.transforms.iter().zip(&fixed) {
        assert_eq!(
            (record.frame.as_str(), record.child_frame.as_str()),
            (joint.parent.link.as_str(), joint.child.link.as_str()),
            "{model}"
        );
        let position = record.position;
        for (published, urdf) in position.iter().zip(joint.origin.xyz.0) {
            assert!(
                (published - urdf).abs() <= TOLERANCE,
                "{model} {}: {position:?} vs {:?}",
                joint.name,
                joint.origin.xyz.0
            );
        }
        let published = UnitQuaternion::from_quaternion(record.orientation.into());
        let [roll, pitch, yaw] = joint.origin.rpy.0;
        let urdf = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        assert!(
            published.angle_to(&urdf) <= TOLERANCE,
            "{model} {}: {published:?} vs {urdf:?}",
            joint.name
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_publishes_the_transform_of_every_fixed_joint_of_its_model() -> peppygen::Result<()>
{
    for (model, config) in [
        ("openarm_v2", on_its_own_hardware()),
        ("so101", an_so101_on_its_own_hardware()),
    ] {
        let (mut harness, _mocks) = Harness::start_with(config, robot_initializer::setup).await?;
        let message = first_message(&mut harness).await;
        assert_holds_the_fixed_joints_of(model, &message);
        harness.shutdown().await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_publishes_its_fixed_joints_before_a_simulation_stands_it() -> peppygen::Result<()>
{
    let (mut harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v1"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let goal = simulation.attach.next_goal(WIRE).await?;

    // The engine has admitted nothing yet.
    let message = first_message(&mut harness).await;
    assert_holds_the_fixed_joints_of("openarm_v1", &message);

    goal.refuse(
        attach::FailureKind::Rejected,
        "that is enough of this robot",
    )
    .await?;
    let _ = harness.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_node_that_subscribes_after_the_publication_receives_it() -> peppygen::Result<()> {
    let (mut harness, _mocks) =
        Harness::start_with(on_its_own_hardware(), robot_initializer::setup).await?;
    let published = first_message(&mut harness).await;

    // The harness subscribed before the node started; this subscriber comes
    // after the publication, as a node launched later does.
    let mut late = peppylib::TopicMessenger::subscribe(
        harness.session(),
        "fixture-core",
        "late-subscriber",
        peppylib::messaging::SenderTarget::contract("transforms", "v1")?,
        "static_transforms",
        &harness.node_producer_ref(),
        peppylib::config::QoSProfile::Reliable,
        peppylib::config::TopicRetention::latest(1).expect("a depth in range"),
        peppylib::messaging::TopicBuffer::KeepLatest,
    )
    .await?;
    let received = tokio::time::timeout(WIRE, late.on_next_message())
        .await
        .expect("the retained message reaches a later subscriber")
        .expect("the subscription stays open");
    let expected = static_transforms::build_message(
        published
            .transforms
            .into_iter()
            .map(|record| MessageTransformsItem {
                frame: record.frame,
                child_frame: record.child_frame,
                position: record.position,
                orientation: record.orientation,
            })
            .collect(),
    )?;
    assert_eq!(received.payload(), expected);

    harness.shutdown().await
}
