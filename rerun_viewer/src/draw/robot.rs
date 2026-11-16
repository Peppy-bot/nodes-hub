//! The robot's body: the collision shapes of each link, drawn once for the whole
//! recording in the color of the link's part, and each link placed where the joints of
//! each snapshot put it.

use std::time::SystemTime;

use camera_core::pose::Pose;

use crate::canvas::{Canvas, Look, Rgba};
use crate::draw::part_path;
use crate::model::robot::{BodyLink, LinkPart, RobotBody};

/// Where the links are drawn: the robot frame.
pub const ROBOT_PATH: &str = "world/robot";

/// The body and the base of each arm: mid grey.
const BODY_LOOK: Look = solid([150, 154, 162, 255]);
/// The odd segments of an arm, from the first: dark grey.
const ODD_SEGMENT_LOOK: Look = solid([62, 66, 76, 255]);
/// The even segments of an arm: light grey, so that each segment stands apart from the
/// next.
const EVEN_SEGMENT_LOOK: Look = solid([214, 217, 224, 255]);
/// The fingers: orange.
const FINGER_LOOK: Look = solid([235, 140, 52, 255]);

const fn solid(color: Rgba) -> Look {
    Look {
        color,
        wireframe: false,
    }
}

/// How the shapes of a link of `part` are drawn.
fn part_look(part: LinkPart) -> Look {
    match part {
        LinkPart::Body => BODY_LOOK,
        LinkPart::Arm { segment } if segment % 2 == 1 => ODD_SEGMENT_LOOK,
        LinkPart::Arm { .. } => EVEN_SEGMENT_LOOK,
        LinkPart::Finger => FINGER_LOOK,
    }
}

/// The entity of the link `link`, which its shapes are drawn under.
pub fn link_path(link: &str) -> String {
    format!("{ROBOT_PATH}/{link}")
}

/// Draws the collision shapes of each link of `body`, under the link's entity, in the
/// look of the link's part, for the whole recording: they move with the link and never
/// change.
pub fn draw_body(canvas: &dyn Canvas, body: &RobotBody) {
    for link in body.links() {
        for (index, shape) in link.shapes.iter().enumerate() {
            let path = part_path(ROBOT_PATH, &link.name, "collision", index, shape);
            canvas.static_shape(&path, shape, part_look(link.part));
        }
    }
}

/// Places each link of `poses` at its pose, at `at`.
pub fn draw_link_poses(canvas: &dyn Canvas, at: SystemTime, poses: &[(&BodyLink, Pose)]) {
    canvas.set_time(at);
    for (link, pose) in poses {
        canvas.place(&link_path(&link.name), pose);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::scene::ShapeKind;
    use crate::testing::{Drawn, FakeCanvas, mini};

    #[test]
    fn each_shape_is_drawn_once_for_the_whole_recording_under_its_link() {
        let canvas = FakeCanvas::default();
        draw_body(&canvas, &mini());
        let drawn: Vec<(String, Look)> = canvas
            .drawn()
            .iter()
            .map(|drawn| match drawn {
                Drawn::StaticShape { path, look, .. } => (path.clone(), *look),
                other => panic!("a static shape, not {other:?}"),
            })
            .collect();
        let expected = [
            ("world/robot/base/collision_0_box", BODY_LOOK),
            ("world/robot/base/collision_1_cylinder", BODY_LOOK),
            ("world/robot/base/collision_2_sphere", BODY_LOOK),
            ("world/robot/base/collision_3_sphere", BODY_LOOK),
            ("world/robot/upper/collision_0_cylinder", ODD_SEGMENT_LOOK),
            ("world/robot/hand/collision_0_mesh", EVEN_SEGMENT_LOOK),
            ("world/robot/finger/collision_0_sphere", FINGER_LOOK),
        ];
        assert_eq!(drawn, expected.map(|(path, look)| (path.to_string(), look)));
        let Drawn::StaticShape { shape, .. } =
            &canvas.at("world/robot/upper/collision_0_cylinder")[0]
        else {
            panic!("a shape")
        };
        assert_eq!(
            shape.kind,
            ShapeKind::Cylinder {
                height: 0.2,
                radius: 0.03
            }
        );
        assert_eq!(shape.pose.origin(), [0.0, 0.0, 0.1]);
    }

    #[test]
    fn each_link_is_placed_at_the_time_of_its_snapshot() {
        let canvas = FakeCanvas::default();
        let body = mini();
        let at = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(7);
        let upper = crate::model::placement::translation([0.0, 0.0, 0.1]);
        draw_link_poses(&canvas, at, &[(&body.links()[1], upper)]);
        let drawn = canvas.drawn();
        assert!(matches!(drawn[0], Drawn::Time(time) if time == at));
        assert!(matches!(
            &drawn[1],
            Drawn::Place { path, pose } if path == "world/robot/upper" && pose.origin() == [0.0, 0.0, 0.1]
        ));
    }
}
