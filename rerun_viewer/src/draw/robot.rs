//! The robot's body: the collision shapes of each link, drawn once for the whole
//! recording, and each link placed where the joints of each snapshot put it.

use std::time::SystemTime;

use camera_core::pose::Pose;

use crate::canvas::{Canvas, Look};
use crate::draw::part_path;
use crate::model::robot::{BodyLink, RobotBody};

/// Where the links are drawn: the robot frame.
pub const ROBOT_PATH: &str = "world/robot";

/// Light grey, a little see-through, so that the points and the cubes behind a link
/// show where the viewer draws see-through meshes.
const BODY_LOOK: Look = Look {
    color: [200, 205, 215, 200],
    wireframe: false,
};

/// The entity of the link `link`, which its shapes are drawn under.
pub fn link_path(link: &str) -> String {
    format!("{ROBOT_PATH}/{link}")
}

/// Draws the collision shapes of each link of `body`, under the link's entity, for the
/// whole recording: they move with the link and never change.
pub fn draw_body(canvas: &dyn Canvas, body: &RobotBody) {
    for link in body.links() {
        for (index, shape) in link.shapes.iter().enumerate() {
            let path = part_path(ROBOT_PATH, &link.name, "collision", index, shape);
            canvas.static_shape(&path, shape, BODY_LOOK);
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
        let paths: Vec<String> = canvas
            .drawn()
            .iter()
            .map(|drawn| {
                assert!(matches!(drawn, Drawn::StaticShape { look, .. } if *look == BODY_LOOK));
                drawn.path().expect("a shape's entity").to_string()
            })
            .collect();
        assert_eq!(
            paths,
            [
                "world/robot/base/collision_0_box",
                "world/robot/base/collision_1_cylinder",
                "world/robot/base/collision_2_sphere",
                "world/robot/base/collision_3_sphere",
                "world/robot/upper/collision_0_cylinder",
                "world/robot/hand/collision_0_mesh",
                "world/robot/finger/collision_0_sphere",
            ]
        );
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
