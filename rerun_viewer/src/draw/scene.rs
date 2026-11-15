//! The planning scene: each object as its shapes, each carried object at its arm's grasp
//! point, a shell of the clearance around each shape, the allowed pairs, and the log of
//! the changes.

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

const OBJECT_LOOK: Look = Look {
    color: [60, 180, 75, 255],
    wireframe: false,
};
const ATTACHED_LOOK: Look = Look {
    color: [245, 130, 48, 255],
    wireframe: false,
};
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
/// carries it, as `description` names it; one carried by another link is only logged.
/// Gives the entities it drew.
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
    }
    let drawn = drawing.drawn;
    for gone in previous.difference(&drawn) {
        canvas.clear(gone);
    }
    canvas.text(ALLOWED_PAIRS_PATH, &allowed_pairs_text(scene));
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
    use crate::model::arms::tests::openarm;
    use crate::model::changes::scene_changes;
    use crate::model::scene::tests::{at, boxed, held};
    use crate::model::scene::{AttachedObject, SceneObject, Shape, ShapeKind};
    use crate::testing::{Drawn, FakeCanvas};

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
        assert_eq!(drawn.len(), 2);
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
        assert_eq!(shape.pose.origin(), [0.4, -0.2, 0.1]);
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
        assert_eq!(
            shape.pose.origin(),
            [0.0, 0.0, 0.05],
            "in the grasp link's frame"
        );
    }

    #[test]
    fn an_object_carried_by_another_link_is_logged_and_not_drawn() {
        let scene = scene_with(vec![], vec![held("brain/lime_1", "openarm_right_link7")]);
        let canvas = FakeCanvas::default();
        let drawn = draw_scene(
            &canvas,
            SystemTime::UNIX_EPOCH,
            &scene,
            &[],
            &openarm(),
            &BTreeSet::new(),
        );
        assert!(drawn.is_empty());
        assert_eq!(
            canvas.log_lines(CHANGES_PATH),
            [
                "brain/lime_1 is attached to openarm_right_link7, which is no arm's grasp link: not drawn"
            ]
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
