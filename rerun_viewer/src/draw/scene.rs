//! The planning scene: each object as its shapes with its id as a label, each carried
//! object at its arm's grasp point, a shell of the clearance around each shape, the
//! allowed pairs, and the log of the changes.

use std::collections::BTreeSet;
use std::time::SystemTime;

use crate::canvas::{Canvas, Look};
use crate::draw::arms::grasp_path;
use crate::draw::part_path;
use crate::model::arms::ArmsDescription;
use crate::model::changes::SceneChange;
use crate::model::scene::{PlacedShape, PlanningScene, attached_shapes, object_shapes, shell};

pub const CHANGES_PATH: &str = "scene/changes";
pub const ALLOWED_PAIRS_PATH: &str = "scene/allowed_pairs";

/// An object that stands in the scene: green.
const OBJECT_LOOK: Look = Look {
    color: [60, 180, 75, 255],
    wireframe: false,
};
/// An object that an arm carries: magenta. It stands between the fingers of the arm, so
/// its color is far from the color of each part of the robot (`draw::robot::part_look`).
const ATTACHED_LOOK: Look = Look {
    color: [240, 50, 230, 255],
    wireframe: false,
};
/// The clearance around a shape: a red wireframe.
const SHELL_LOOK: Look = Look {
    color: [230, 25, 75, 255],
    wireframe: true,
};

/// Where the shapes and the shells of the free objects are drawn.
pub const OBJECTS_PATH: &str = "world/scene/objects";
pub const SHELLS_PATH: &str = "world/scene/shells";

/// Where the shapes and the shells of the objects that the arm `arm` carries are drawn:
/// under its grasp point, which they move with.
pub fn attached_path(arm: &str) -> String {
    format!("{}/attached", grasp_path(arm))
}

/// Draws `scene` at `at`, which `changes` took it to, and clears each entity of
/// `previous`, the entities of the scene drawn before, that this scene does not draw.
/// Each carried object is drawn under the grasp point of the arm whose grasp link
/// carries it, as `description` names it; one carried by another link is not drawn.
/// The log gets each change and, with changes, each carried object not drawn: a scene
/// drawn again unchanged logs nothing. Gives the entities it drew.
pub fn draw_scene(
    canvas: &dyn Canvas,
    at: SystemTime,
    scene: &PlanningScene,
    changes: &[SceneChange],
    description: &ArmsDescription,
    previous: &BTreeSet<String>,
) -> BTreeSet<String> {
    canvas.set_time(at);
    let mut drawing = SceneDrawing {
        canvas,
        clearance_m: scene.clearance_m,
        drawn: BTreeSet::new(),
    };
    for object in &scene.objects {
        drawing.shapes(
            OBJECTS_PATH,
            SHELLS_PATH,
            &object.id,
            &object_shapes(object),
            OBJECT_LOOK,
        );
        drawing.label(
            OBJECTS_PATH,
            &object.id,
            object.placement.position,
            &object.id,
            OBJECT_LOOK,
        );
    }
    let mut not_drawn = Vec::new();
    for attached in &scene.attached {
        let Some(arm) = description.arm_with_grasp_link(&attached.link) else {
            not_drawn.push(format!(
                "{} is attached to {}, which is no arm's grasp link: not drawn",
                attached.id, attached.link
            ));
            continue;
        };
        let parent = attached_path(arm);
        drawing.shapes(
            &parent,
            &parent,
            &attached.id,
            &attached_shapes(attached),
            ATTACHED_LOOK,
        );
        // The planner holds the object as carried, whatever became of the real one.
        drawing.label(
            &parent,
            &attached.id,
            attached.placement.position,
            &format!("{} (carried)", attached.id),
            ATTACHED_LOOK,
        );
    }
    let drawn = drawing.drawn;
    for gone in previous.difference(&drawn) {
        canvas.clear(gone);
    }
    canvas.text(ALLOWED_PAIRS_PATH, &allowed_pairs_text(scene));
    if changes.is_empty() {
        return drawn;
    }
    for change in changes {
        canvas.log_line(CHANGES_PATH, &change.to_string());
    }
    for line in not_drawn {
        canvas.log_line(CHANGES_PATH, &line);
    }
    drawn
}

/// One drawing of a scene: the entities drawn so far.
struct SceneDrawing<'a> {
    canvas: &'a dyn Canvas,
    clearance_m: f64,
    drawn: BTreeSet<String>,
}

impl SceneDrawing<'_> {
    /// Draws the shapes of the object `id` under `shape_parent`, and, with a clearance
    /// above 0, the shell of each under `shell_parent`.
    fn shapes(
        &mut self,
        shape_parent: &str,
        shell_parent: &str,
        id: &str,
        shapes: &[PlacedShape],
        look: Look,
    ) {
        for (index, shape) in shapes.iter().enumerate() {
            self.draw(
                part_path(shape_parent, id, "shape", index, shape),
                shape,
                look,
            );
            if self.clearance_m > 0.0 {
                let grown = shell(shape, self.clearance_m);
                self.draw(
                    part_path(shell_parent, id, "shell", index, &grown),
                    &grown,
                    SHELL_LOOK,
                );
            }
        }
    }

    fn draw(&mut self, path: String, shape: &PlacedShape, look: Look) {
        self.canvas.shape(&path, shape, look);
        self.drawn.insert(path);
    }

    /// Draws `text` at `position` of the frame that the object `id` under `parent` is
    /// placed in, in the color of `look`.
    fn label(&mut self, parent: &str, id: &str, position: [f64; 3], text: &str, look: Look) {
        let path = format!("{parent}/{id}/label");
        self.canvas.label(&path, position, text, look.color);
        self.drawn.insert(path);
    }
}

/// The allowed pairs of `scene` as a markdown list.
fn allowed_pairs_text(scene: &PlanningScene) -> String {
    if scene.allowed_pairs.is_empty() {
        return "No pair is allowed to touch.".to_string();
    }
    scene
        .allowed_pairs
        .iter()
        .map(|(a, b)| format!("- {a} may touch {b}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::Rgba;
    use crate::draw::robot::part_look;
    use crate::drawn::Drawn;
    use crate::model::arms::tests::openarm;
    use crate::model::changes::scene_changes;
    use crate::model::robot::LinkPart;
    use crate::model::scene::tests::{at, boxed, held};
    use crate::model::scene::{AttachedObject, SceneObject, Shape, ShapeKind};
    use crate::testing::FakeCanvas;

    fn scene_with(objects: Vec<SceneObject>, attached: Vec<AttachedObject>) -> PlanningScene {
        PlanningScene {
            objects,
            attached,
            allowed_pairs: vec![],
            clearance_m: 0.04,
        }
    }

    fn cleared_paths(canvas: &FakeCanvas) -> Vec<String> {
        canvas
            .drawn()
            .iter()
            .filter_map(|drawn| match drawn {
                Drawn::Clear { path } => Some(path.clone()),
                _ => None,
            })
            .collect()
    }

    /// Draws `scene` over the entities `previous`, and gives the canvas and the entities
    /// it drew.
    fn draw(scene: &PlanningScene, previous: &BTreeSet<String>) -> (FakeCanvas, BTreeSet<String>) {
        let canvas = FakeCanvas::default();
        let drawn = draw_scene(
            &canvas,
            SystemTime::UNIX_EPOCH,
            scene,
            &[],
            &openarm(),
            previous,
        );
        (canvas, drawn)
    }

    fn shape_paths(canvas: &FakeCanvas) -> Vec<String> {
        canvas
            .drawn()
            .iter()
            .filter_map(|drawn| match drawn {
                Drawn::Shape { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect()
    }

    /// The least difference, in one channel of the two colors, that sets a shape of the
    /// scene apart from a link of the robot that it touches.
    const APART: u8 = 64;

    #[test]
    fn an_object_of_the_scene_has_a_color_far_from_the_color_of_each_part_of_the_robot() {
        let parts = [
            LinkPart::Body,
            LinkPart::Arm { segment: 1 },
            LinkPart::Arm { segment: 2 },
            LinkPart::Finger,
        ];
        let looks = [
            ("an object", OBJECT_LOOK),
            ("a carried object", ATTACHED_LOOK),
        ];
        for (name, look) in looks {
            for part in parts {
                let apart = widest_difference(look.color, part_look(part).color);
                assert!(apart >= APART, "{name} against {part:?}: {apart}");
            }
        }
    }

    /// The largest difference between two colors in one channel.
    fn widest_difference(one: Rgba, other: Rgba) -> u8 {
        let channels = one.into_iter().zip(other);
        let differences = channels.map(|(a, b)| a.abs_diff(b));
        differences.max().expect("a color has channels")
    }

    #[test]
    fn each_object_is_drawn_with_its_shell_and_the_changes_are_logged() {
        let scene = scene_with(vec![boxed("brain/lime_1", [0.4, -0.2, 0.1])], vec![]);
        let empty = PlanningScene::empty_like(&scene);
        let canvas = FakeCanvas::default();
        let drawn = draw_scene(
            &canvas,
            SystemTime::UNIX_EPOCH,
            &scene,
            &scene_changes(&empty, &scene),
            &openarm(),
            &BTreeSet::new(),
        );
        assert_eq!(
            shape_paths(&canvas),
            [
                "world/scene/objects/brain/lime_1/shape_0_box",
                "world/scene/shells/brain/lime_1/shell_0_box"
            ]
        );
        assert_eq!(drawn.len(), 3, "the shape, its shell and the label");
        assert!(matches!(
            &canvas.at("world/scene/objects/brain/lime_1/label")[..],
            [Drawn::Label { position, text, .. }]
                if *position == [0.4, -0.2, 0.1] && text == "brain/lime_1"
        ));
        let shell = canvas.at("world/scene/shells/brain/lime_1/shell_0_box");
        let Drawn::Shape { shape, look, .. } = &shell[0] else {
            panic!("a shape")
        };
        assert!(look.wireframe);
        assert_eq!(
            shape.kind,
            ShapeKind::Box {
                size: [0.1 + 0.08, 0.2 + 0.08, 0.3 + 0.08]
            }
        );
        assert_eq!(shape.pose.position(), [0.4, -0.2, 0.1]);
        assert_eq!(canvas.log_lines(CHANGES_PATH), ["added brain/lime_1"]);
    }

    #[test]
    fn a_carried_object_is_drawn_under_its_arms_grasp_point() {
        let scene = scene_with(vec![], vec![held("brain/lime_1", "openarm_right_tcp")]);
        let canvas = FakeCanvas::default();
        draw_scene(
            &canvas,
            SystemTime::UNIX_EPOCH,
            &scene,
            &[],
            &openarm(),
            &BTreeSet::new(),
        );
        assert_eq!(
            shape_paths(&canvas),
            [
                "world/arms/right_arm/grasp/attached/brain/lime_1/shape_0_box",
                "world/arms/right_arm/grasp/attached/brain/lime_1/shell_0_box"
            ]
        );
        let shape = canvas.at("world/arms/right_arm/grasp/attached/brain/lime_1/shape_0_box");
        let Drawn::Shape { shape, .. } = &shape[0] else {
            panic!("a shape")
        };
        assert!(matches!(
            &canvas.at("world/arms/right_arm/grasp/attached/brain/lime_1/label")[..],
            [Drawn::Label { position, text, .. }]
                if *position == [0.0, 0.0, 0.05] && text == "brain/lime_1 (carried)"
        ));
        assert_eq!(
            shape.pose.position(),
            [0.0, 0.0, 0.05],
            "in the grasp link's frame"
        );
    }

    #[test]
    fn an_object_carried_by_another_link_is_logged_with_the_changes_and_not_drawn() {
        let scene = scene_with(vec![], vec![held("brain/lime_1", "openarm_right_link7")]);
        let changes = scene_changes(&PlanningScene::empty_like(&scene), &scene);
        let canvas = FakeCanvas::default();
        let drawn = draw_scene(
            &canvas,
            SystemTime::UNIX_EPOCH,
            &scene,
            &changes,
            &openarm(),
            &BTreeSet::new(),
        );
        assert!(drawn.is_empty());
        let not_drawn = "brain/lime_1 is attached to openarm_right_link7, which is no arm's grasp link: not drawn";
        assert_eq!(
            canvas.log_lines(CHANGES_PATH).last().map(String::as_str),
            Some(not_drawn)
        );
        let lines = canvas.log_lines(CHANGES_PATH).len();
        draw_scene(
            &canvas,
            SystemTime::UNIX_EPOCH,
            &scene,
            &[],
            &openarm(),
            &drawn,
        );
        assert_eq!(
            canvas.log_lines(CHANGES_PATH).len(),
            lines,
            "the same scene drawn again logs nothing"
        );
    }

    #[test]
    fn what_the_scene_no_longer_holds_is_cleared() {
        let before = scene_with(vec![boxed("brain/lime_1", [0.4, -0.2, 0.1])], vec![]);
        let (_, drawn_before) = draw(&before, &BTreeSet::new());
        let after = scene_with(vec![boxed("table", [0.5, 0.0, 0.0])], vec![]);
        let (canvas, _) = draw(&after, &drawn_before);
        assert_eq!(
            cleared_paths(&canvas),
            [
                "world/scene/objects/brain/lime_1/label",
                "world/scene/objects/brain/lime_1/shape_0_box",
                "world/scene/shells/brain/lime_1/shell_0_box"
            ]
        );
    }

    #[test]
    fn a_shape_that_changes_kind_clears_its_old_entity() {
        let before = scene_with(vec![boxed("brain/lime_1", [0.4, -0.2, 0.1])], vec![]);
        let (_, drawn_before) = draw(&before, &BTreeSet::new());
        let mut after = before.clone();
        after.objects[0].shapes[0].kind = ShapeKind::Sphere { radius: 0.03 };
        let (canvas, _) = draw(&after, &drawn_before);
        assert_eq!(
            shape_paths(&canvas),
            [
                "world/scene/objects/brain/lime_1/shape_0_sphere",
                "world/scene/shells/brain/lime_1/shell_0_sphere"
            ]
        );
        assert_eq!(
            cleared_paths(&canvas),
            [
                "world/scene/objects/brain/lime_1/shape_0_box",
                "world/scene/shells/brain/lime_1/shell_0_box"
            ]
        );
    }

    #[test]
    fn a_shape_that_goes_clears_its_entity() {
        let mut before = scene_with(vec![boxed("table", [0.5, 0.0, 0.0])], vec![]);
        before.objects[0].shapes.push(Shape {
            kind: ShapeKind::Box {
                size: [0.1, 0.1, 0.1],
            },
            placement: at([0.0, 0.0, 0.2]),
        });
        let (_, drawn_before) = draw(&before, &BTreeSet::new());
        let after = scene_with(vec![boxed("table", [0.5, 0.0, 0.0])], vec![]);
        let (canvas, _) = draw(&after, &drawn_before);
        assert_eq!(
            cleared_paths(&canvas),
            [
                "world/scene/objects/table/shape_1_box",
                "world/scene/shells/table/shell_1_box"
            ]
        );
    }

    #[test]
    fn the_shells_go_when_the_clearance_goes_to_zero() {
        let before = scene_with(vec![boxed("table", [0.5, 0.0, 0.0])], vec![]);
        let (_, drawn_before) = draw(&before, &BTreeSet::new());
        let mut after = before.clone();
        after.clearance_m = 0.0;
        let (canvas, _) = draw(&after, &drawn_before);
        assert_eq!(
            cleared_paths(&canvas),
            ["world/scene/shells/table/shell_0_box"]
        );
    }

    #[test]
    fn no_shell_is_drawn_without_a_clearance() {
        let mut scene = scene_with(vec![boxed("table", [0.5, 0.0, 0.0])], vec![]);
        scene.clearance_m = 0.0;
        let canvas = FakeCanvas::default();
        draw_scene(
            &canvas,
            SystemTime::UNIX_EPOCH,
            &scene,
            &[],
            &openarm(),
            &BTreeSet::new(),
        );
        assert_eq!(
            shape_paths(&canvas),
            ["world/scene/objects/table/shape_0_box"]
        );
    }

    #[test]
    fn the_allowed_pairs_are_listed() {
        let mut scene = scene_with(vec![], vec![]);
        assert_eq!(allowed_pairs_text(&scene), "No pair is allowed to touch.");
        scene.allowed_pairs = vec![
            (
                "brain/lime_1".to_string(),
                "openarm_right_finger1".to_string(),
            ),
            (
                "brain/lime_1".to_string(),
                "openarm_right_finger2".to_string(),
            ),
        ];
        assert_eq!(
            allowed_pairs_text(&scene),
            "- brain/lime_1 may touch openarm_right_finger1\n- brain/lime_1 may touch openarm_right_finger2"
        );
    }
}
