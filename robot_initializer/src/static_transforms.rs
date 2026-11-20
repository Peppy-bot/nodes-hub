//! The transforms of this robot's fixed joints, which the node publishes on
//! transforms' static_transforms once it holds the model: one message that
//! holds every fixed joint of the model's URDF, each a transform record from
//! the joint's parent link (frame) to its child link (child_frame) at the
//! joint's origin, with no timestamp since each holds at every time. The topic
//! retains the message, so a node that subscribes later receives it too, for
//! as long as this node runs. The node publishes nothing on transforms: the
//! robot's backbone publishes the joints it measures.

use frame_tree::{StaticMessage, TransformRecord};
use peppygen::emitted_topics::transforms::static_transforms::{self, MessageTransformsItem};
use peppygen::{NodeRunner, Result};
use robot_description::RobotDescription;

use crate::refused;

/// The static_transforms message of `description`: the transform of every
/// fixed joint of its URDF, in the URDF's order of joints. A joint whose
/// transform the transforms contract does not take refuses the node, naming
/// the joint's links and why.
fn message_of(description: &RobotDescription) -> Result<StaticMessage> {
    let transforms = description
        .fixed_transforms()
        .iter()
        .map(|fixed| {
            fixed
                .frame_transform()
                .map_err(|refusal| refused(format!("static_transforms: {refusal}")))
        })
        .collect::<Result<Vec<_>>>()?;
    StaticMessage::from_transforms(&transforms)
        .map_err(|refusal| refused(format!("static_transforms: {refusal}")))
}

/// The record of the wire for a transform record of the tree, field for
/// field.
fn item_of(record: TransformRecord) -> MessageTransformsItem {
    let TransformRecord {
        frame,
        child_frame,
        position,
        orientation,
    } = record;
    MessageTransformsItem {
        frame,
        child_frame,
        position,
        orientation,
    }
}

/// Publishes the transforms of the fixed joints of `description`, once,
/// with no timestamp: each holds at every time.
pub async fn publish(description: &RobotDescription, runner: &NodeRunner) -> Result<()> {
    let message = message_of(description)?;
    let joints = message.transforms.len();
    let publisher = static_transforms::declare_publisher(runner).await?;
    publisher
        .publish(static_transforms::build_message(
            message.transforms.into_iter().map(item_of).collect(),
        )?)
        .await?;
    tracing::info!("static_transforms published: the transforms of {joints} fixed joints");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_bundled_model_gives_one_transform_per_fixed_joint() {
        for model in robot_description::models() {
            let description = model.description();
            let message = message_of(&description).expect("a bundled model's fixed joints");
            let count = description.fixed_transforms().len();
            assert_eq!(message.transforms.len(), count, "{}", model.id());
        }
    }

    #[test]
    fn a_fixed_joint_at_no_finite_origin_refuses_the_node_naming_its_links() {
        let model = robot_description::model("openarm_v2").expect("a bundled model");
        let chest = r#"xyz="0.0792 0.0315 0.7941""#;
        assert!(
            model.urdf().contains(chest),
            "openarm_v2 places its chest camera there"
        );
        let urdf = model.urdf().replace(chest, r#"xyz="inf 0.0315 0.7941""#);
        let description =
            RobotDescription::parse(&urdf, model.srdf()).expect("the URDF reader takes it");
        let refusal = message_of(&description)
            .expect_err("the contract takes finite positions alone")
            .to_string();
        assert!(
            refusal.contains(
                "static_transforms: the joint from openarm_body_link0 to chest_optical_frame"
            ),
            "{refusal}"
        );
    }
}
