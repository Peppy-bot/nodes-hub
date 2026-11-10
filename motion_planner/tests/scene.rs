//! The planning scene over the generated harness, as a caller of motion_planning's scene
//! members sees it: a box between the start and the goal is avoided until it is
//! removed; an item attached to the grasp point touches the hand until the gripper's
//! links are its touch links, is carried around the box, and keeps clear of the other
//! arm and the body; an item the right arm holds stands where it holds it for the left
//! arm; pairs let a held item rest on a table and the fingers close on an item; a closed
//! mesh around the arm is a collision and an open mesh is refused; get_planning_scene
//! gives exactly what was set, a replacing add, a move, an attach and a detach keep the
//! pairs and a remove drops those that name the object; fields an operation does not read
//! are not checked; every refusal comes with rejected, or with precondition_failed before
//! the arms are measured, and changes nothing; an object clears the map where it is added
//! or moved to, and the depth points inside an object or an attached item mark nothing;
//! and the depth points within padding_offset_m of an object mark nothing, those beyond
//! it mark the map.

mod common;

use std::time::Duration;

use chain_kinematics::nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};
use collision_geometry::chain::RiderBody;
use motion_planning_core::{pose_from_parts, pose_parts};
use openarm_description::{ARM_DOF, Side};
use peppygen::exposed_services::motion_planning::apply_collision_objects::{
    ApplyCollisionObjectsObjectsItem as Object, ApplyCollisionObjectsObjectsItemShapesItem as Shape,
};
use peppygen::fixtures::exposed_actions::motion_planning::plan_joint_path::FailureKind;
use peppygen::fixtures::exposed_services::motion_planning::{
    apply_attached_collision_object, apply_collision_objects, check_joint_path, get_planning_scene,
    set_allowed_collision,
};
use peppygen::fixtures::harness::{Config, Harness, Mocks};

use common::{
    CLEAR_OR_ALLOW_WAY_OUT, Ended, OPENARM_V2, OVER_THE_TABLE, OWN_BODY_WAY_OUT, RIGHT_READY,
    Solid, TIMEOUT, UNDER_THE_TABLE, arms, at_ms, check, left_path, looking_at, parameters, plan,
    plan_to, planned_waypoints, publish_arms, render, start, start_with_camera,
    start_without_camera, table_across, take_frame, wait_for_the_arms,
};

/// The left arm's grasp link.
const GRASP_LINK: &str = "openarm_left_tcp";

/// The pose of the left arm's grasp link with the arm at `q`.
fn grasp_pose(q: &[f64; ARM_DOF]) -> Isometry3<f64> {
    OPENARM_V2.arms[Side::Left]
        .geometry
        .chain()
        .world_ee_pose(q)
}

/// The left gripper's links, which a held item may touch.
fn gripper_links() -> Vec<String> {
    OPENARM_V2.arms[Side::Left].gripper_links.clone()
}

/// A primitive of `kind` and `dimensions` at `pose` in its object's frame.
fn primitive(kind: &str, dimensions: &[f64], pose: &Isometry3<f64>) -> Shape {
    let (position, orientation) = pose_parts(pose);
    Shape {
        kind: kind.to_string(),
        dimensions: dimensions.to_vec(),
        mesh_vertices: Vec::new(),
        mesh_triangles: Vec::new(),
        position,
        orientation,
    }
}

/// A mesh of `vertices` and `triangles` at its object's origin.
fn mesh(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> Shape {
    Shape {
        kind: "mesh".to_string(),
        dimensions: Vec::new(),
        mesh_vertices: vertices.iter().flatten().copied().collect(),
        mesh_triangles: triangles.iter().flatten().copied().collect(),
        position: [0.0; 3],
        orientation: [0.0, 0.0, 0.0, 1.0],
    }
}

/// The closed cube of side `side` about its origin as a mesh: its eight corners, and two
/// triangles a face, counterclockwise seen from outside.
fn cube_mesh(side: f64) -> Shape {
    let h = side / 2.0;
    let corners: Vec<[f64; 3]> = (0..8)
        .map(|k| [k & 1, k >> 1 & 1, k >> 2 & 1].map(|bit| if bit == 1 { h } else { -h }))
        .collect();
    let triangles = [
        [0, 2, 1],
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 6],
        [0, 1, 4],
        [1, 5, 4],
        [2, 6, 3],
        [3, 6, 7],
        [0, 4, 2],
        [2, 4, 6],
        [1, 3, 5],
        [3, 7, 5],
    ];
    mesh(&corners, &triangles)
}

/// An entry of apply_collision_objects: `operation` on the object `id` at `pose`, of
/// `shapes`.
fn object(operation: &str, id: &str, pose: &Isometry3<f64>, shapes: Vec<Shape>) -> Object {
    let (position, orientation) = pose_parts(pose);
    Object {
        id: id.to_string(),
        operation: operation.to_string(),
        position,
        orientation,
        shapes,
    }
}

/// The add of the object `id` at `pose`, of `shapes`.
fn add(id: &str, pose: &Isometry3<f64>, shapes: Vec<Shape>) -> Object {
    object("add", id, pose, shapes)
}

/// The add of the box `solid` as the object `id`.
fn add_solid(id: &str, (low, high): Solid) -> Object {
    let center: [f64; 3] = std::array::from_fn(|i| (low[i] + high[i]) / 2.0);
    let sides: [f64; 3] = std::array::from_fn(|i| high[i] - low[i]);
    add(
        id,
        &Isometry3::translation(center[0], center[1], center[2]),
        vec![primitive("box", &sides, &Isometry3::identity())],
    )
}

/// The move of the object `id` to `pose`.
fn move_to(id: &str, pose: &Isometry3<f64>) -> Object {
    object("move", id, pose, Vec::new())
}

/// The removal of the object `id`.
fn remove(id: &str) -> Object {
    object("remove", id, &Isometry3::identity(), Vec::new())
}

/// A poll's answer: taken, or the failure the node replied with.
fn answered<F>(
    polled: peppygen::Result<()>,
    failure_of: impl Fn(&peppygen::Error) -> Option<F>,
) -> Result<(), F> {
    polled.map_err(|e| failure_of(&e).unwrap_or_else(|| panic!("{e}")))
}

async fn apply(
    harness: &Harness,
    objects: Vec<Object>,
) -> Result<(), apply_collision_objects::Failure> {
    answered(
        apply_collision_objects::poll(
            harness,
            &apply_collision_objects::RequestData { objects },
            TIMEOUT,
        )
        .await,
        apply_collision_objects::Failure::of,
    )
}

/// Attaches the object `id` to `link`, touching `touch_links`, or detaches it.
async fn attachment(
    harness: &Harness,
    operation: &str,
    id: &str,
    link: &str,
    touch_links: &[String],
) -> Result<(), apply_attached_collision_object::Failure> {
    answered(
        apply_attached_collision_object::poll(
            harness,
            &apply_attached_collision_object::RequestData {
                object_id: id.to_string(),
                link_name: link.to_string(),
                touch_links: touch_links.to_vec(),
                operation: operation.to_string(),
            },
            TIMEOUT,
        )
        .await,
        apply_attached_collision_object::Failure::of,
    )
}

/// Attaches the object `id` to the left grasp link, touching `touch_links`.
async fn attach(
    harness: &Harness,
    id: &str,
    touch_links: &[String],
) -> Result<(), apply_attached_collision_object::Failure> {
    attachment(harness, "attach", id, GRASP_LINK, touch_links).await
}

/// Detaches the object `id`.
async fn detach(
    harness: &Harness,
    id: &str,
) -> Result<(), apply_attached_collision_object::Failure> {
    attachment(harness, "detach", id, "", &[]).await
}

/// Lets the object `name` touch `other_names`, or stops it.
async fn allow(
    harness: &Harness,
    name: &str,
    other_names: &[String],
    allowed: bool,
) -> Result<(), set_allowed_collision::Failure> {
    answered(
        set_allowed_collision::poll(
            harness,
            &set_allowed_collision::RequestData {
                name: name.to_string(),
                other_names: other_names.to_vec(),
                allowed,
            },
            TIMEOUT,
        )
        .await,
        set_allowed_collision::Failure::of,
    )
}

/// `table_across` the motion from under the table to over it, raised by the clearance so the
/// arm under it starts outside the clearance of it: a start inside it is refused.
fn box_across() -> Solid {
    let ((low, high), _) = table_across(&UNDER_THE_TABLE, &OVER_THE_TABLE);
    let rise = parameters().min_clearance_m;
    let raised = |corner: [f64; 3]| [corner[0], corner[1], corner[2] + rise];
    (raised(low), raised(high))
}

async fn scene(harness: &Harness) -> get_planning_scene::Response {
    get_planning_scene::poll(harness, TIMEOUT)
        .await
        .expect("the scene")
}

/// The check of the left arm standing where it was measured.
async fn standing(harness: &Harness, q: [f64; ARM_DOF]) -> check_joint_path::Response {
    check(harness, &left_path(&[], &[q]))
        .await
        .expect("an answer")
}

// With a box across the straight motion from under the table to over it, the straight
// path fails on the box, and a plan goes around it, its path passing its check. Once the
// box is removed, the straight path passes, and a plan goes through the box's place: its
// path fails once the box is back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_box_between_start_and_goal_is_avoided_until_it_is_removed() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    apply(&harness, vec![add_solid("box", box_across())])
        .await
        .expect("the box is added");
    let straight = left_path(&[], &[OVER_THE_TABLE]);
    let through = check(&harness, &straight).await.expect("an answer");
    assert!(
        !through.clear
            && through.clear_waypoints == 0
            && through.reason.starts_with("on the run from the start, ")
            && through.reason.contains(" from the map and the objects, "),
        "the straight path through the box, with no camera: {through:?}"
    );
    let around = plan(&harness, &plan_to(&OVER_THE_TABLE))
        .await
        .expect("a path around the box");
    let waypoints = planned_waypoints(&around);
    let path = left_path(&around.arms[0].start_joint_positions, &waypoints);
    let checked = check(&harness, &path).await.expect("an answer");
    assert!(
        checked.clear && checked.clear_waypoints as usize == waypoints.len(),
        "the planned path around the box: {checked:?}"
    );

    apply(&harness, vec![remove("box")])
        .await
        .expect("the box is removed");
    assert!(
        check(&harness, &straight).await.expect("an answer").clear,
        "the straight path once the box is removed"
    );
    let free = plan(&harness, &plan_to(&OVER_THE_TABLE))
        .await
        .expect("a path");
    let free_path = left_path(
        &free.arms[0].start_joint_positions,
        &planned_waypoints(&free),
    );
    apply(&harness, vec![add_solid("box", box_across())])
        .await
        .expect("the box is back");
    assert!(
        !check(&harness, &free_path).await.expect("an answer").clear,
        "the plan without the box goes through its place"
    );
    harness.shutdown().await
}

/// How far (m) the plan's clearance from the map and the objects lies above
/// min_clearance_m (`motion_planning_core`'s README, A valid state).
const WORLD_MARGIN_M: f64 = 0.001;

/// The plan's clearance from the map and the objects (m), at the launcher's parameters.
fn world_clearance_m() -> f64 {
    parameters().min_clearance_m + WORLD_MARGIN_M
}

/// How far (m) the fingers reach into an item between them, along the approach axis.
const GRIP_M: f64 = 0.01;
/// How far (m) a table top reaches into the end of an item set down on it.
const SET_DOWN_M: f64 = 0.005;
/// How far (m) past the plan's clearance a held item's end lies from the gripper, so a
/// table top that reaches `SET_DOWN_M` into it keeps the plan's clearance from the gripper
/// with this to spare.
const SPARE_M: f64 = 0.004;

/// The left gripper where the arm stands at `q`, in the grasp link's frame, the approach
/// axis z and the jaws' closing axis y: the midpoint of the fingers' centers, how far
/// apart their centers lie along y, how far their tips reach along z, how far the
/// gripper's links and fingers reach along z (its front), and how far the arm's links
/// behind the gripper reach along z (the wrist).
struct Gripper {
    midpoint: Vector3<f64>,
    width: f64,
    tips: f64,
    front: f64,
    wrist: f64,
}

/// The farthest along z, in the grasp link's frame, of `points`.
fn farthest_z(points: impl IntoIterator<Item = Point3<f64>>) -> f64 {
    points
        .into_iter()
        .map(|p| p.z)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The left gripper where the arm stands at `q`.
fn left_gripper(q: &[f64; ARM_DOF]) -> Gripper {
    let arm = &OPENARM_V2.arms[Side::Left];
    let posed = arm.geometry.chain().at(q);
    let to_grasp = grasp_pose(q).inverse();
    let fingers: Vec<Vec<Point3<f64>>> = OPENARM_V2
        .own
        .fingers(arm.id, &arm.geometry)
        .expect("the left arm's fingers")
        .into_iter()
        .map(|finger| {
            let pose = to_grasp * posed.link_pose_world(finger.segment) * finger.offset;
            let RiderBody::Pieces(pieces) = finger.body else {
                panic!("a finger is a part of the robot");
            };
            pieces
                .iter()
                .flat_map(|piece| piece.vertices().iter().map(move |v| pose * v))
                .collect()
        })
        .collect();
    let center = |points: &[Point3<f64>]| {
        points
            .iter()
            .fold(Vector3::zeros(), |sum, p| sum + p.coords)
            / points.len() as f64
    };
    let [a, b] = fingers.as_slice() else {
        panic!("the left gripper has {} fingers", fingers.len());
    };
    let (a, b) = (center(a), center(b));
    let width = (a.y - b.y).abs();
    assert!(width > 0.02, "the fingers' centers stand {width} m apart");
    let links: Vec<(&str, Point3<f64>)> = arm
        .geometry
        .place(q)
        .pieces()
        .flat_map(|(link, piece)| {
            piece
                .vertices()
                .map(|v| (link, to_grasp * v))
                .collect::<Vec<_>>()
        })
        .collect();
    let in_gripper = |link: &str| arm.gripper_links.iter().any(|g| g == link);
    let tips = farthest_z(fingers.iter().flatten().copied());
    Gripper {
        midpoint: (a + b) / 2.0,
        width,
        tips,
        front: farthest_z(
            links
                .iter()
                .filter(|(link, _)| in_gripper(link))
                .map(|(_, p)| *p),
        )
        .max(tips),
        wrist: farthest_z(
            links
                .iter()
                .filter(|(link, _)| !in_gripper(link))
                .map(|(_, p)| *p),
        ),
    }
}

impl Gripper {
    /// Where a held item ends along z: `SET_DOWN_M` and `SPARE_M` past the plan's
    /// clearance from the gripper's front.
    fn item_end(&self) -> f64 {
        self.front + world_clearance_m() + SET_DOWN_M + SPARE_M
    }

    /// Asserts that an item that begins at `from` along z keeps the plan's clearance from
    /// the arm's links behind the gripper.
    fn assert_clear_of_the_wrist(&self, from: f64) {
        assert!(
            from - self.wrist >= world_clearance_m(),
            "an item from {from} m lies {} m from the wrist",
            from - self.wrist
        );
    }
}

/// An item between the left gripper's fingers where the arm stands at `q`, from `GRIP_M`
/// before the fingers' tips to the item's end ([`box_between_the_fingers`]).
fn item_between_the_fingers(q: &[f64; ARM_DOF]) -> (Isometry3<f64>, Shape, Isometry3<f64>) {
    let gripper = left_gripper(q);
    let from = gripper.tips - GRIP_M;
    gripper.assert_clear_of_the_wrist(from);
    box_between_the_fingers(q, &gripper, from, gripper.item_end())
}

/// A box between the fingers of `gripper`, the left gripper where the arm stands at `q`:
/// in the grasp link's frame, from `from` to `end` along the approach axis, as wide along
/// the closing axis as the fingers' centers lie apart, so each finger reaches into it,
/// and 1 cm across. Its pose and its shape, and the pose of its end, in the grasp link's
/// orientation.
fn box_between_the_fingers(
    q: &[f64; ARM_DOF],
    gripper: &Gripper,
    from: f64,
    end: f64,
) -> (Isometry3<f64>, Shape, Isometry3<f64>) {
    let length = end - from;
    let (x, y) = (gripper.midpoint.x, gripper.midpoint.y);
    let grasp = grasp_pose(q);
    (
        grasp * Translation3::new(x, y, from + length / 2.0),
        primitive(
            "box",
            &[0.01, gripper.width, length],
            &Isometry3::identity(),
        ),
        grasp * Translation3::new(x, y, end),
    )
}

/// The add of a table top, 10 cm square and 2 cm thick across the approach axis of `end`,
/// the pose of a held item's end, past that end and reaching `SET_DOWN_M` into the item.
fn table_under(end: &Isometry3<f64>) -> Object {
    let thickness = 0.02;
    add(
        "table",
        &(end * Translation3::new(0.0, 0.0, thickness / 2.0 - SET_DOWN_M)),
        vec![primitive(
            "box",
            &[0.1, 0.1, thickness],
            &Isometry3::identity(),
        )],
    )
}

/// An item the left gripper holds at its grasp point where the arm stands at `q`: a box in
/// the grasp link's frame, 2 cm square across the approach axis, from 1.5 cm inside the
/// gripper's front to the item's end along it. Its pose and its shape.
fn item_at_the_grasp_point(q: &[f64; ARM_DOF]) -> (Isometry3<f64>, Shape) {
    let gripper = left_gripper(q);
    let from = gripper.front - 0.015;
    gripper.assert_clear_of_the_wrist(from);
    let length = gripper.item_end() - from;
    (
        grasp_pose(q) * Translation3::new(0.0, 0.0, from + length / 2.0),
        primitive("box", &[0.02, 0.02, length], &Isometry3::identity()),
    )
}

// An item at the grasp point that reaches into the hand leaves the arm standing clear
// while pairs let it touch the gripper's links. Attached with no touch links and no pairs,
// it touches the hand: a check from where the arm stands fails at its start, and a plan
// fails with start_in_collision. With the gripper's links as its touch links, a plan
// carries the item around a box added across the arm's motion, and the path passes its
// check.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_attached_item_is_carried_around_the_box() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    let (pose, shape) = item_at_the_grasp_point(&UNDER_THE_TABLE);
    apply(&harness, vec![add("item", &pose, vec![shape])])
        .await
        .expect("the item is added");
    allow(&harness, "item", &gripper_links(), true)
        .await
        .expect("the item may touch the gripper");
    let allowed = standing(&harness, UNDER_THE_TABLE).await;
    assert!(
        allowed.clear,
        "the item in the hand, which it may touch: {allowed:?}"
    );
    allow(&harness, "item", &gripper_links(), false)
        .await
        .expect("the item may not touch the gripper");
    attach(&harness, "item", &[])
        .await
        .expect("the item is attached");
    let touching = standing(&harness, UNDER_THE_TABLE).await;
    assert!(
        !touching.clear
            && touching.clear_waypoints == 0
            && touching
                .reason
                .starts_with("the start is in contact: item touches openarm_left_"),
        "an attached item in the hand with no touch links: {touching:?}"
    );
    let Err(Ended::Failed(collided)) = plan(&harness, &plan_to(&OVER_THE_TABLE)).await else {
        panic!("a plan from a start in contact fails once taken");
    };
    assert_eq!(collided.kind, FailureKind::StartInCollision, "{collided}");

    detach(&harness, "item")
        .await
        .expect("the item is detached");
    attach(&harness, "item", &gripper_links())
        .await
        .expect("the item is attached, touching the gripper");
    apply(&harness, vec![add_solid("box", box_across())])
        .await
        .expect("the box is added");
    let carried = plan(&harness, &plan_to(&OVER_THE_TABLE))
        .await
        .expect("a path carrying the item around the box");
    let waypoints = planned_waypoints(&carried);
    let checked = check(
        &harness,
        &left_path(&carried.arms[0].start_joint_positions, &waypoints),
    )
    .await
    .expect("an answer");
    assert!(
        checked.clear && checked.clear_waypoints as usize == waypoints.len(),
        "the path carrying the item: {checked:?}"
    );
    harness.shutdown().await
}

// An item the left gripper holds keeps clear of what the plan does not move: an item
// reaching into the right hand, and one reaching into the torso, put the start in
// contact with them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_held_item_keeps_clear_of_the_other_arm_and_the_body() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    let grasp = grasp_pose(&UNDER_THE_TABLE);
    let mut model = OPENARM_V2.own.model.clone();
    let bodies = model
        .world_pieces(&[UNDER_THE_TABLE, RIGHT_READY])
        .expect("the bodies where the arms stand");
    let center_of = |body: &str| {
        let (_, pieces) = bodies
            .iter()
            .find(|(name, _)| *name == body)
            .unwrap_or_else(|| panic!("the self model has no body {body}"));
        pieces[0].bounding_sphere().0
    };
    for body in ["openarm_right_ee_base_link", "openarm_body_link0"] {
        let reaching = Isometry3::from_parts(
            Translation3::from((grasp.inverse() * center_of(body)).coords),
            UnitQuaternion::identity(),
        );
        apply(
            &harness,
            vec![add(
                "item",
                &grasp,
                vec![primitive("sphere", &[0.02], &reaching)],
            )],
        )
        .await
        .expect("the item is added");
        attach(&harness, "item", &gripper_links())
            .await
            .expect("the item is attached");
        let touching = standing(&harness, UNDER_THE_TABLE).await;
        assert!(
            !touching.clear
                && touching.reason
                    == format!("the start is in contact: item touches {body}{OWN_BODY_WAY_OUT}"),
            "an item reaching into {body}: {touching:?}"
        );
        detach(&harness, "item")
            .await
            .expect("the item is detached");
        apply(&harness, vec![remove("item")])
            .await
            .expect("the item is removed");
    }
    harness.shutdown().await
}

// An item at the left grasp point that reaches into the left hand, attached to the right
// arm's grasp link: a check of the left arm standing puts its start in contact with the
// item where the right arm holds it, until the item's touch links are the left gripper's
// links. A detach leaves the item where the right arm held it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_item_the_right_arm_holds_stands_where_it_holds_it() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    let right_grasp_link = OPENARM_V2.arms[Side::Right].grasp_link;
    let (at_the_left_grasp, item) = item_at_the_grasp_point(&UNDER_THE_TABLE);
    for (touch_links, clear) in [(Vec::new(), false), (gripper_links(), true)] {
        apply(
            &harness,
            vec![add("item", &at_the_left_grasp, vec![item.clone()])],
        )
        .await
        .expect("the item is added");
        attachment(&harness, "attach", "item", right_grasp_link, &touch_links)
            .await
            .expect("the right arm holds the item");
        let standing_still = standing(&harness, UNDER_THE_TABLE).await;
        assert_eq!(
            standing_still.clear, clear,
            "touch links {touch_links:?}: {standing_still:?}"
        );
        if !clear {
            assert!(
                standing_still
                    .reason
                    .starts_with("the start is in contact: openarm_left_")
                    && standing_still
                        .reason
                        .ends_with(&format!(" touches item{CLEAR_OR_ALLOW_WAY_OUT}")),
                "{standing_still:?}"
            );
        }
        detach(&harness, "item")
            .await
            .expect("the item is detached");
        let left_there = scene(&harness).await;
        let [item] = left_there.objects.as_slice() else {
            panic!("one object: {left_there:?}")
        };
        assert_pose(item.position, item.orientation, &at_the_left_grasp);
        apply(&harness, vec![remove("item")])
            .await
            .expect("the item is removed");
    }
    harness.shutdown().await
}

// A held item resting on a table touches it until a pair lets them touch. An item between
// the fingers, attached to nothing, touches them until a pair of the item and the
// gripper's links lets them close on it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pairs_let_the_item_rest_on_the_table_and_the_fingers_close_on_it() -> peppygen::Result<()>
{
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    let (pose, shape, end) = item_between_the_fingers(&UNDER_THE_TABLE);
    apply(&harness, vec![add("item", &pose, vec![shape])])
        .await
        .expect("the item is added");
    let between = standing(&harness, UNDER_THE_TABLE).await;
    assert!(
        !between.clear && between.reason.contains("touches item"),
        "the fingers on an item no pair names: {between:?}"
    );
    allow(&harness, "item", &gripper_links(), true)
        .await
        .expect("the item may touch the gripper");
    let closed = standing(&harness, UNDER_THE_TABLE).await;
    assert!(closed.clear, "the fingers close on the item: {closed:?}");

    attach(&harness, "item", &gripper_links())
        .await
        .expect("the item is attached");
    apply(&harness, vec![table_under(&end)])
        .await
        .expect("the table is added");
    let resting = standing(&harness, UNDER_THE_TABLE).await;
    assert!(
        !resting.clear && resting.reason.contains("item touches table"),
        "the held item on the table: {resting:?}"
    );
    allow(&harness, "item", &["table".to_string()], true)
        .await
        .expect("the item may touch the table");
    let set_down = standing(&harness, UNDER_THE_TABLE).await;
    assert!(
        set_down.clear,
        "the held item rests on the table: {set_down:?}"
    );
    harness.shutdown().await
}

/// How far (m) from the wrist an item gripped deep begins: nearer than the plan's
/// clearance, out of contact.
const DEEP_M: f64 = 0.01;
/// How far (m) into the wrist an item gripped too deep reaches.
const INTO_M: f64 = 0.005;
/// How far (m) a short item reaches past the gripper's front.
const SHORT_M: f64 = 0.02;

// A held item keeps out of contact with the robot's links outside its touch links, with
// no clearance from them: an item gripped deep, `DEEP_M` from the wrist, nearer than the
// plan's clearance, leaves the arm standing clear, and one that reaches `INTO_M` into the
// wrist puts the start in contact with it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_held_item_keeps_out_of_contact_with_the_links_it_may_not_touch() -> peppygen::Result<()>
{
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    let gripper = left_gripper(&UNDER_THE_TABLE);
    assert!(DEEP_M < world_clearance_m());
    for (from, touching) in [
        (gripper.wrist + DEEP_M, false),
        (gripper.wrist - INTO_M, true),
    ] {
        let (pose, shape, _) =
            box_between_the_fingers(&UNDER_THE_TABLE, &gripper, from, gripper.item_end());
        apply(&harness, vec![add("item", &pose, vec![shape])])
            .await
            .expect("the item is added");
        attach(&harness, "item", &gripper_links())
            .await
            .expect("the item is attached");
        let held = standing(&harness, UNDER_THE_TABLE).await;
        if touching {
            assert!(
                !held.clear
                    && held.reason.starts_with("the start is in contact: ")
                    && held.reason.contains("item")
                    && held.reason.contains("openarm_left_link6"),
                "an item reaching into the wrist: {held:?}"
            );
        } else {
            assert!(held.clear, "an item gripped deep: {held:?}");
        }
        detach(&harness, "item").await.expect("detached");
        apply(&harness, vec![remove("item")])
            .await
            .expect("removed");
    }
    harness.shutdown().await
}

// A held item that ends `SHORT_M` past the gripper's front, set down on a table, brings
// the table nearer the gripper than the clearance: with the item and the table paired
// alone, a check and a plan from there are refused, the reason naming the gripper, the
// table and the way out, and once the table is paired with the gripper's links too it
// stands clear.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn setting_a_short_item_down_needs_the_table_paired_with_the_gripper() -> peppygen::Result<()>
{
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    let gripper = left_gripper(&UNDER_THE_TABLE);
    let from = gripper.tips - GRIP_M;
    gripper.assert_clear_of_the_wrist(from);
    let end = gripper.front + SHORT_M;
    assert!(end - SET_DOWN_M - gripper.front < world_clearance_m());
    let (pose, shape, end) = box_between_the_fingers(&UNDER_THE_TABLE, &gripper, from, end);
    apply(&harness, vec![add("item", &pose, vec![shape])])
        .await
        .expect("the item is added");
    attach(&harness, "item", &gripper_links())
        .await
        .expect("the item is attached");
    apply(&harness, vec![table_under(&end)])
        .await
        .expect("the table is added");
    allow(&harness, "item", &["table".to_string()], true)
        .await
        .expect("the item may touch the table");
    let near = standing(&harness, UNDER_THE_TABLE).await;
    assert!(
        !near.clear
            && near.reason.starts_with("the start keeps openarm_left_")
            && near.reason.contains(" from table, inside the clearance ")
            && near.reason.ends_with(CLEAR_OR_ALLOW_WAY_OUT),
        "the gripper near the table: {near:?}"
    );
    let Err(Ended::Failed(refused)) = plan(&harness, &plan_to(&OVER_THE_TABLE)).await else {
        panic!("a plan from a start inside the clearance fails once taken");
    };
    assert_eq!(refused.kind, FailureKind::StartInCollision, "{refused}");
    assert_eq!(refused.message, near.reason);
    allow(&harness, "table", &gripper_links(), true)
        .await
        .expect("the table may touch the gripper");
    let set_down = standing(&harness, UNDER_THE_TABLE).await;
    assert!(set_down.clear, "the short item set down: {set_down:?}");
    harness.shutdown().await
}

// A closed cube mesh that holds the whole left arm, no link reaching its faces, is a
// collision: the start is in contact with it. The same cube with one triangle missing is
// open, and refused with rejected.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_closed_mesh_about_the_arm_collides_and_an_open_one_is_refused() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    let corners: Vec<Point3<f64>> = OPENARM_V2.arms[Side::Left]
        .geometry
        .place(&UNDER_THE_TABLE)
        .pieces()
        .flat_map(|(_, piece)| piece.vertices().collect::<Vec<_>>())
        .collect();
    let low = corners
        .iter()
        .fold(Vector3::repeat(f64::INFINITY), |m, p| m.inf(&p.coords));
    let high = corners
        .iter()
        .fold(Vector3::repeat(f64::NEG_INFINITY), |m, p| m.sup(&p.coords));
    // 10 cm beyond the arm on every side.
    let side = (high - low).max() + 0.2;
    let center = Isometry3::from_parts(
        Translation3::from((low + high) / 2.0),
        UnitQuaternion::identity(),
    );
    apply(&harness, vec![add("cube", &center, vec![cube_mesh(side)])])
        .await
        .expect("the cube is added");
    let inside = standing(&harness, UNDER_THE_TABLE).await;
    assert!(
        !inside.clear && inside.clear_waypoints == 0 && inside.reason.contains("touches cube"),
        "the arm inside the cube: {inside:?}"
    );
    let mut open = cube_mesh(side);
    open.mesh_triangles.truncate(open.mesh_triangles.len() - 3);
    let refused = apply(&harness, vec![add("open", &center, vec![open])])
        .await
        .expect_err("an open mesh is refused");
    assert_eq!(
        refused.kind,
        apply_collision_objects::FailureKind::Rejected,
        "{refused}"
    );
    harness.shutdown().await
}

// get_planning_scene gives exactly what was set: the objects no link carries, in the
// order of their ids, their shapes posed in their frames; the attached item under
// attached_objects alone, posed in the grasp link's frame; the allowed pairs once each;
// and the clearance and the voxel size the launcher sets. An add that replaces the table,
// a move of it and a detach of the item keep the pairs, the item left where the hand held
// it; a remove drops every pair that names the object.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_scene_gives_exactly_what_was_set() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    let turned = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
    let pose = Isometry3::from_parts(Translation3::new(0.5, -0.3, 0.2), turned);
    let shapes = vec![
        primitive(
            "box",
            &[0.1, 0.2, 0.3],
            &Isometry3::translation(0.0, 0.0, 0.1),
        ),
        primitive(
            "cylinder",
            &[0.2, 0.05],
            &Isometry3::rotation(Vector3::x() * 0.5),
        ),
    ];
    let sphere = vec![primitive("sphere", &[0.04], &Isometry3::identity())];
    let held = grasp_pose(&UNDER_THE_TABLE) * Isometry3::translation(0.0, 0.0, 0.05);
    apply(
        &harness,
        vec![
            add("table", &pose, shapes.clone()),
            add("item", &held, sphere.clone()),
            add(
                "cube",
                &Isometry3::translation(0.6, 0.4, 0.3),
                vec![cube_mesh(0.1)],
            ),
        ],
    )
    .await
    .expect("added");
    attach(&harness, "item", &gripper_links())
        .await
        .expect("attached");
    allow(
        &harness,
        "item",
        &["table".to_string(), "occupancy_map".to_string()],
        true,
    )
    .await
    .expect("allowed");
    allow(&harness, "table", &["item".to_string()], true)
        .await
        .expect("allowed again from the other name");

    let answer = scene(&harness).await;
    assert_eq!(answer.clearance_m, parameters().min_clearance_m);
    assert_eq!(answer.voxel_size_m, parameters().voxel_size_m);
    let ids: Vec<&str> = answer.objects.iter().map(|o| o.id.as_str()).collect();
    assert_eq!(ids, ["cube", "table"]);
    let table = &answer.objects[1];
    assert_pose(table.position, table.orientation, &pose);
    let kinds: Vec<&str> = table.shapes.iter().map(|s| s.kind.as_str()).collect();
    assert_eq!(kinds, ["box", "cylinder"]);
    for (got, sent) in table.shapes.iter().zip(&shapes) {
        assert_eq!(got.dimensions, sent.dimensions);
        assert_pose(
            got.position,
            got.orientation,
            &pose_from_parts(sent.position, sent.orientation).expect("a pose"),
        );
    }
    let cube = &answer.objects[0];
    let sent = cube_mesh(0.1);
    assert_eq!(
        (
            &cube.shapes[0].mesh_vertices,
            &cube.shapes[0].mesh_triangles
        ),
        (&sent.mesh_vertices, &sent.mesh_triangles)
    );
    let [item] = answer.attached_objects.as_slice() else {
        panic!("one attached object: {:?}", answer.attached_objects);
    };
    assert_eq!(
        (item.id.as_str(), item.link_name.as_str()),
        ("item", GRASP_LINK)
    );
    let mut touch_links = item.touch_links.clone();
    touch_links.sort_unstable();
    let mut gripper = gripper_links();
    gripper.sort_unstable();
    assert_eq!(touch_links, gripper);
    assert_pose(
        item.position,
        item.orientation,
        &Isometry3::translation(0.0, 0.0, 0.05),
    );
    assert_eq!(item.shapes[0].dimensions, [0.04]);
    assert_eq!(
        pair_names(&answer),
        [("item", "occupancy_map"), ("item", "table")]
    );

    let moved = Isometry3::translation(0.4, -0.4, 0.3);
    apply(
        &harness,
        vec![
            add("table", &pose, sphere.clone()),
            move_to("table", &moved),
        ],
    )
    .await
    .expect("replaced and moved");
    detach(&harness, "item").await.expect("detached");
    let kept = scene(&harness).await;
    assert_eq!(
        pair_names(&kept),
        [("item", "occupancy_map"), ("item", "table")]
    );
    assert!(kept.attached_objects.is_empty(), "{kept:?}");
    let ids: Vec<&str> = kept.objects.iter().map(|o| o.id.as_str()).collect();
    assert_eq!(ids, ["cube", "item", "table"]);
    let (item, table) = (&kept.objects[1], &kept.objects[2]);
    assert_pose(item.position, item.orientation, &held);
    assert_pose(table.position, table.orientation, &moved);
    let kinds: Vec<&str> = table.shapes.iter().map(|s| s.kind.as_str()).collect();
    assert_eq!(kinds, ["sphere"]);

    apply(&harness, vec![remove("table")])
        .await
        .expect("removed");
    assert_eq!(
        pair_names(&scene(&harness).await),
        [("item", "occupancy_map")]
    );
    harness.shutdown().await
}

/// The allowed pairs of `answer`, each by its two names.
fn pair_names(answer: &get_planning_scene::Response) -> Vec<(&str, &str)> {
    answer
        .allowed_pairs
        .iter()
        .map(|p| (p.name_a.as_str(), p.name_b.as_str()))
        .collect()
}

/// Asserts that `position` and `orientation` give `expected`, to within a nanometer and a
/// nanoradian.
fn assert_pose(position: [f64; 3], orientation: [f64; 4], expected: &Isometry3<f64>) {
    let got = pose_from_parts(position, orientation).expect("a pose");
    assert!(
        (got.translation.vector - expected.translation.vector).norm() < 1e-9
            && got.rotation.angle_to(&expected.rotation) < 1e-9,
        "{got} against {expected}"
    );
}

/// Waits until the node over `harness` serves its planning scene, which it does once its
/// robot's identity answers.
async fn served(harness: &Harness) {
    for _ in 0..TIMEOUT.as_secs() {
        if get_planning_scene::poll(harness, Duration::from_secs(1))
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    panic!("the node never served its planning scene");
}

// Each refusal of the scene members comes with its kind and changes nothing. Before the
// arms are measured, an attach is refused with precondition_failed. Every other refusal
// is rejected: a batch whose second entry moves an unknown object, which applies
// neither entry; an empty id, and an id that names a link of an arm or of the body, or
// the voxel map; an add of no shape, of a shape of the wrong dimensions, or at an
// orientation that is no unit quaternion; an operation neither member has; a remove of
// an unknown object; an add, a move or a remove of an attached object; an attach to a
// link that is no grasp link, with a touch link the robot does not have, or of an
// attached object; a detach of an object no link carries, or of an unknown one; and a
// pair whose name is no object, whose other names hold a name of nothing, or the object
// itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_refusal_comes_with_its_kind_and_changes_nothing() -> peppygen::Result<()> {
    let config = Config {
        parameters: Some(parameters()),
        camera_vacant: true,
        geometry_vacant: true,
        camera_mounts_vacant: true,
        workspace_vacant: true,
        ..Config::default()
    };
    let (harness, mocks) = start(config).await?;
    served(&harness).await;
    let here = Isometry3::translation(0.6, 0.0, 0.4);
    let block = || vec![primitive("box", &[0.05; 3], &Isometry3::identity())];
    apply(
        &harness,
        vec![add("item", &here, block()), add("loose", &here, block())],
    )
    .await
    .expect("added");
    let unmeasured = attach(&harness, "item", &[])
        .await
        .expect_err("an attach before the arms are measured");
    assert_eq!(
        unmeasured.kind,
        apply_attached_collision_object::FailureKind::PreconditionFailed,
        "{unmeasured}"
    );
    measure_arms(&harness, &mocks).await?;
    attach(&harness, "item", &[])
        .await
        .expect("the item is attached once the arms are measured");

    let mut unit_less = add("block", &here, block());
    unit_less.orientation = [0.0, 0.0, 0.0, 2.0];
    let attached = "\"item\" is attached to openarm_left_tcp; detach it first";
    let batches = [
        (
            vec![add("block", &here, block()), move_to("nothing", &here)],
            "objects[1] (\"nothing\"): no object of the scene is named \"nothing\"",
        ),
        (
            vec![add("openarm_left_link4", &here, block())],
            "is a link of the robot",
        ),
        (
            vec![add("openarm_body_link0", &here, block())],
            "is a link of the robot",
        ),
        (
            vec![add("occupancy_map", &here, block())],
            "names the voxel map",
        ),
        (
            vec![add("block", &here, Vec::new())],
            "an object has at least one shape",
        ),
        (
            vec![add(
                "block",
                &here,
                vec![primitive("sphere", &[0.02, 0.03], &Isometry3::identity())],
            )],
            "shapes[0]: a sphere has 1 dimensions",
        ),
        (vec![unit_less], "an orientation is a unit quaternion"),
        (
            vec![object("rotate", "block", &here, block())],
            "operation is \"rotate\"",
        ),
        (vec![remove("nothing")], "no object of the scene is named"),
        (vec![add("", &here, block())], "an object's id is empty"),
        (vec![add("item", &here, block())], attached),
        (vec![move_to("item", &here)], attached),
        (vec![remove("item")], attached),
    ];
    for (batch, says) in batches {
        let refused = apply(&harness, batch).await.expect_err("a refused batch");
        assert!(
            refused.kind == apply_collision_objects::FailureKind::Rejected
                && refused.message.contains(says),
            "{says}: {refused}"
        );
    }
    let attachments = [
        (
            attachment(&harness, "attach", "loose", "openarm_left_link4", &[]).await,
            "openarm_left_link4 is no arm's grasp_link",
        ),
        (
            attachment(
                &harness,
                "attach",
                "loose",
                GRASP_LINK,
                &["nowhere".to_string()],
            )
            .await,
            "\"nowhere\" is no link of the robot",
        ),
        (attach(&harness, "item", &[]).await, attached),
        (
            detach(&harness, "loose").await,
            "\"loose\" is attached to no link",
        ),
        (
            detach(&harness, "nothing").await,
            "no object of the scene is named \"nothing\"",
        ),
        (
            attachment(&harness, "grab", "loose", GRASP_LINK, &[]).await,
            "operation is \"grab\"",
        ),
    ];
    for (answer, says) in attachments {
        let refused = answer.expect_err("a refused attachment");
        assert!(
            refused.kind == apply_attached_collision_object::FailureKind::Rejected
                && refused.message.contains(says),
            "{says}: {refused}"
        );
    }
    let pairs = [
        (
            allow(&harness, "openarm_left_link4", &["item".to_string()], true).await,
            "no object of the scene is named \"openarm_left_link4\"",
        ),
        (
            allow(&harness, "item", &["nowhere".to_string()], true).await,
            "\"nowhere\" is no link of the robot, no object of the scene",
        ),
        (
            allow(&harness, "item", &["item".to_string()], true).await,
            "\"item\" is paired with itself",
        ),
    ];
    for (answer, says) in pairs {
        let refused = answer.expect_err("a refused pair");
        assert!(
            refused.kind == set_allowed_collision::FailureKind::Rejected
                && refused.message.contains(says),
            "{says}: {refused}"
        );
    }

    let answer = scene(&harness).await;
    let ids: Vec<&str> = answer.objects.iter().map(|o| o.id.as_str()).collect();
    let attached_ids: Vec<&str> = answer
        .attached_objects
        .iter()
        .map(|o| o.id.as_str())
        .collect();
    assert_eq!(ids, ["loose"], "{answer:?}");
    assert_eq!(attached_ids, ["item"], "{answer:?}");
    assert!(answer.allowed_pairs.is_empty(), "{answer:?}");
    harness.shutdown().await
}

/// Publishes the arms at Ready but the left one under the table of `table_across`, and
/// waits until a check of the left arm standing still stops being refused.
async fn measure_arms(harness: &Harness, mocks: &Mocks) -> peppygen::Result<()> {
    publish_arms(mocks, at_ms(0), &arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    wait_for_the_arms(harness, UNDER_THE_TABLE).await;
    Ok(())
}

// Fields an operation does not read are not checked: a move whose shapes cannot be built,
// a detach that names no grasp link and a touch link the robot does not have, and a remove
// at an orientation that is no unit quaternion each apply.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fields_an_operation_does_not_read_are_not_checked() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    let here = Isometry3::translation(0.6, 0.0, 0.4);
    let block = vec![primitive("box", &[0.05; 3], &Isometry3::identity())];
    apply(&harness, vec![add("item", &here, block)])
        .await
        .expect("added");
    let mut unbuilt = move_to("item", &here);
    unbuilt.shapes = vec![primitive("sphere", &[], &Isometry3::identity())];
    apply(&harness, vec![unbuilt])
        .await
        .expect("a move reads no shapes");
    attach(&harness, "item", &[]).await.expect("attached");
    attachment(
        &harness,
        "detach",
        "item",
        "openarm_left_link4",
        &["nowhere".to_string()],
    )
    .await
    .expect("a detach reads no link and no touch links");
    let mut unit_less = remove("item");
    unit_less.orientation = [0.0; 4];
    apply(&harness, vec![unit_less])
        .await
        .expect("a remove reads no pose");
    let answer = scene(&harness).await;
    assert!(
        answer.objects.is_empty() && answer.attached_objects.is_empty(),
        "{answer:?}"
    );
    harness.shutdown().await
}

// With a camera that sees a table across the left arm's motion, the straight path fails
// on the map. An object added aside leaves it so, and moved onto the table clears its
// voxels: once a later call removes it the path passes. Frames taken while an object
// stands over the table mark nothing inside it, where the same frames without it mark
// the table again; an object added there clears those voxels too. A frame that shows an
// item the left gripper holds marks nothing where the item is.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn objects_clear_the_map_and_their_points_mark_nothing() -> peppygen::Result<()> {
    let (table, _) = table_across(&UNDER_THE_TABLE, &OVER_THE_TABLE);
    let ([x0, y0, _], [x1, y1, top]) = table;
    let (x, y) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let camera = looking_at(Vector3::new(x + 0.5, y, top + 0.7), Vector3::new(x, y, top));
    let (harness, mocks) = start_with_camera(&camera).await?;
    let start = arms(UNDER_THE_TABLE, RIGHT_READY);
    let seen = render(&camera, None, &[table]);
    let straight = left_path(&[], &[OVER_THE_TABLE]);
    let blocked = |answer: check_joint_path::Response| {
        assert!(
            !answer.clear && answer.reason.contains(" from the map and the objects, "),
            "the straight path through the table's voxels: {answer:?}"
        );
    };
    let passes = |answer: check_joint_path::Response| {
        assert!(answer.clear, "the straight path: {answer:?}");
    };

    // Three frames mark a voxel the scene cleared: OctoMap's log-odds climb from their
    // floor by a hit each.
    let take_three = async |from_ms: u64| -> peppygen::Result<()> {
        for at in [from_ms, from_ms + 200, from_ms + 400].map(at_ms) {
            publish_arms(&mocks, at, &start).await?;
            take_frame(&harness, &mocks, at, &seen, UNDER_THE_TABLE).await?;
        }
        Ok(())
    };

    take_three(0).await?;
    blocked(check(&harness, &straight).await.expect("an answer"));
    let aside = Isometry3::translation(x, y, top + 1.0);
    apply(
        &harness,
        vec![add(
            "slab",
            &aside,
            vec![primitive("box", &[0.2, 0.2, 0.05], &Isometry3::identity())],
        )],
    )
    .await
    .expect("added aside");
    blocked(check(&harness, &straight).await.expect("an answer"));
    apply(
        &harness,
        vec![move_to("slab", &Isometry3::translation(x, y, top - 0.025))],
    )
    .await
    .expect("moved onto the table");
    apply(&harness, vec![remove("slab")])
        .await
        .expect("removed");
    passes(check(&harness, &straight).await.expect("an answer"));

    apply(&harness, vec![add_solid("table", table)])
        .await
        .expect("added");
    take_three(1_000).await?;
    apply(&harness, vec![remove("table")])
        .await
        .expect("removed");
    passes(check(&harness, &straight).await.expect("an answer"));

    take_three(2_000).await?;
    blocked(check(&harness, &straight).await.expect("an answer"));
    apply(&harness, vec![add_solid("table", table)])
        .await
        .expect("added");
    apply(&harness, vec![remove("table")])
        .await
        .expect("removed");
    passes(check(&harness, &straight).await.expect("an answer"));

    // A cube 4 cm on a side, along the robot's axes, held about the grasp point.
    let held = grasp_pose(&UNDER_THE_TABLE).translation.vector;
    let item: Solid = (
        (held - Vector3::repeat(0.02)).into(),
        (held + Vector3::repeat(0.02)).into(),
    );
    apply(&harness, vec![add_solid("item", item)])
        .await
        .expect("added");
    attach(&harness, "item", &gripper_links())
        .await
        .expect("attached");
    publish_arms(&mocks, at_ms(3_000), &start).await?;
    let holding = render(&camera, Some(&start), &[item]);
    let (low, high) = item;
    let on_the_item = holding
        .iter()
        .flatten()
        .map(|p| camera * Point3::from(*p))
        .filter(|p| (0..3).all(|i| low[i] <= p[i] && p[i] <= high[i]))
        .count();
    assert!(on_the_item > 0, "the camera sees the item");
    take_frame(&harness, &mocks, at_ms(3_000), &holding, UNDER_THE_TABLE).await?;
    passes(standing(&harness, UNDER_THE_TABLE).await);
    harness.shutdown().await
}

// A camera above the table across the left arm's motion sees its top alone. Frames
// taken while a lid stands 1 cm less than the launcher's padding_offset_m above the top
// mark nothing: once the lid is removed the straight path passes. Frames taken while it
// stands 1 cm more than padding_offset_m above it mark the table: the straight path fails
// on the map. The renderer's samples lie at most 4 mm inside the top, within either
// centimeter.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn depth_points_within_the_padding_of_an_object_mark_nothing() -> peppygen::Result<()> {
    let offset = parameters().padding_offset_m;
    let (table, _) = table_across(&UNDER_THE_TABLE, &OVER_THE_TABLE);
    let ([x0, y0, _], [x1, y1, top]) = table;
    let (x, y) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let camera = looking_at(Vector3::new(x, y, top + 0.7), Vector3::new(x, y, top));
    let (harness, mocks) = start_with_camera(&camera).await?;
    let start = arms(UNDER_THE_TABLE, RIGHT_READY);
    let seen = render(&camera, None, &[table]);
    let straight = left_path(&[], &[OVER_THE_TABLE]);
    for (gap, from_ms, marked) in [(offset - 0.01, 0, false), (offset + 0.01, 1_000, true)] {
        apply(
            &harness,
            vec![add(
                "lid",
                &Isometry3::translation(x, y, top + gap + 0.005),
                vec![primitive("box", &[0.3, 0.3, 0.01], &Isometry3::identity())],
            )],
        )
        .await
        .expect("the lid is added");
        // Three frames mark a voxel from the floor of its log-odds.
        for at in [from_ms, from_ms + 200, from_ms + 400].map(at_ms) {
            publish_arms(&mocks, at, &start).await?;
            take_frame(&harness, &mocks, at, &seen, UNDER_THE_TABLE).await?;
        }
        apply(&harness, vec![remove("lid")])
            .await
            .expect("the lid is removed");
        let answer = check(&harness, &straight).await.expect("an answer");
        assert_eq!(
            !answer.clear && answer.reason.contains(" from the map and the objects, "),
            marked,
            "frames under a lid {gap} m above the table: {answer:?}"
        );
    }
    harness.shutdown().await
}
