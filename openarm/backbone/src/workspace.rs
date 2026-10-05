//! The OpenArm's part of the workspace contract's answer, from the robot's
//! design alone: whether an arm reaches a point, and where the perception
//! camera stands, in the robot frame. Reach is the arms' closed-form inverse
//! kinematics, the grasp point at the target, to within the reach tolerance,
//! in each grasp orientation `workspace_core` lists; the view is
//! `workspace_core`'s field-of-view and depth test of the perception camera,
//! placed where the design fixes it and seen through the intrinsics and the
//! depth stream its camera gives. The parsing of a request, the reach of the
//! surfaces measured last and the composition of each answer are
//! `workspace_core::design`'s, so this answer reads as every robot's and as a
//! simulation's; the refusals of the perception camera's geometry are this
//! module's own.

use std::str::FromStr;
use std::sync::{Mutex, MutexGuard};

use srs_model::nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
use srs_model::{Arm, ArmAnglePolicy};
use workspace_core::design::{
    self, Positions, PositionsAnswer, PositionsReach, ReachMemo, SurfaceAnswer, SurfaceHeight,
    SurfaceReach, ViewCheck,
};
use workspace_core::{
    Camera, CameraFacts, Depth, DepthModel, GraspOrientation, Intrinsics, PerceptionCameraError,
    REACH_TOLERANCE, Reach, UnknownDepthModel,
};

use crate::arm_pair::ArmPair;
use crate::camera_mounts::{CameraMounts, DesignCamera};
use crate::types::Side;

/// Why the perception camera's view cannot be checked now, from what the
/// camera linked as perception_geometry answered.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum CameraGeometryError {
    #[error(
        "cannot read the colour intrinsics of the {camera} camera, the perception camera: {reason}"
    )]
    Colour {
        camera: &'static str,
        reason: String,
    },

    #[error(
        "cannot read the depth intrinsics of the {camera} camera, the perception camera: {reason}"
    )]
    Depth {
        camera: &'static str,
        reason: String,
    },

    #[error(
        "the camera linked as the {camera} camera, the perception camera, gives no depth: {reason}"
    )]
    NoDepth {
        camera: &'static str,
        reason: String,
    },
}

/// The pinhole model of a colour stream as camera_geometry:v1 gives it,
/// parsed: an image with pixels, positive focal lengths and a finite
/// principal point.
pub fn intrinsics_from_wire(
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
) -> Result<Intrinsics, &'static str> {
    if width == 0 || height == 0 {
        return Err("its image has no pixels");
    }
    if !(fx.is_finite() && fx > 0.0 && fy.is_finite() && fy > 0.0) {
        return Err("its focal lengths are not positive numbers");
    }
    if !(cx.is_finite() && cy.is_finite()) {
        return Err("its principal point is not a finite point");
    }
    Ok(Intrinsics {
        width,
        height,
        fx,
        fy,
        cx,
        cy,
    })
}

/// Why a depth stream's answer is not a depth stream camera_geometry:v1
/// describes.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum DepthFromWireError {
    #[error(transparent)]
    Model(#[from] UnknownDepthModel),

    #[error(
        "its depth range is not two finite depths, the nearest at least 0 and under the farthest"
    )]
    Range,
}

/// What a depth stream measures as camera_geometry:v1 gives it, parsed: a
/// depth model the contract names (`depth_model`), and the samples it reads,
/// finite, the nearest at least 0 and under the farthest.
pub fn depth_from_wire(
    depth_model: &str,
    nearest: f64,
    farthest: f64,
) -> Result<Depth, DepthFromWireError> {
    let model = DepthModel::from_str(depth_model)?;
    if !(nearest.is_finite() && farthest.is_finite() && 0.0 <= nearest && nearest < farthest) {
        return Err(DepthFromWireError::Range);
    }
    Ok(Depth {
        model,
        range: [nearest, farthest],
    })
}

/// The robot's perception camera as its design fixes it: the one camera that
/// gives depth and that no arm carries, and where its optical frame stands in
/// the robot frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerceptionCamera {
    pub name: &'static str,
    position: [f64; 3],
    /// Row-major: the columns are the optical frame's axes.
    rotation: [f64; 9],
}

impl PerceptionCamera {
    /// The perception camera of `cameras`, if the design has one.
    fn of(cameras: &[DesignCamera]) -> Result<Option<Self>, PerceptionCameraError> {
        let facts: Vec<CameraFacts<'static>> = cameras
            .iter()
            .map(|camera| CameraFacts {
                name: camera.name,
                gives_depth: camera.depth_range.is_some(),
                carried_by_arm: camera.fixed_pose.is_none(),
            })
            .collect();
        let Some(name) = workspace_core::perception_camera(&facts)? else {
            return Ok(None);
        };
        let pose = cameras
            .iter()
            .find(|camera| camera.name == name)
            .and_then(|camera| camera.fixed_pose)
            .expect("the perception camera is a camera of the design that no arm carries");
        let translation = pose.translation.vector;
        let rotation = pose.rotation.to_rotation_matrix();
        Ok(Some(Self {
            name,
            position: [translation.x, translation.y, translation.z],
            rotation: std::array::from_fn(|i| rotation[(i / 3, i % 3)]),
        }))
    }

    /// The camera at the design's pose, seen through the `intrinsics` of its
    /// colour stream and measuring `depth`.
    pub fn camera(&self, intrinsics: Intrinsics, depth: Depth) -> Camera {
        Camera {
            position: self.position,
            rotation: self.rotation,
            intrinsics,
            depth: Some(depth),
        }
    }
}

/// The pose the grasp point takes to reach `target` in `orientation`, in the
/// robot frame. The tool frame `arm_model` mounts is the grasp frame of
/// `workspace_core`: +Z the approach out of the gripper, +Y the jaw closing
/// axis.
pub fn grasp_pose(orientation: GraspOrientation, target: [f64; 3]) -> Isometry3<f64> {
    let rotation =
        Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&orientation.rotation()));
    Isometry3::from_parts(
        Translation3::new(target[0], target[1], target[2]),
        UnitQuaternion::from_rotation_matrix(&rotation),
    )
}

/// The grasps at `target`, in the robot frame, in the 16 grasp orientations.
pub fn grasp_poses(target: [f64; 3]) -> impl Iterator<Item = Isometry3<f64>> {
    GraspOrientation::all().map(move |orientation| grasp_pose(orientation, target))
}

/// The robot's workspace from its design: its two arms, its perception
/// camera if it has one, and the reach of the surfaces measured last.
pub struct Workspace {
    arms: ArmPair<Arm>,
    perception_camera: Option<PerceptionCamera>,
    reach_memo: Mutex<ReachMemo>,
}

impl Workspace {
    /// The workspace of the robot whose arms are `arms`, built by
    /// `arm_model`, and whose cameras are `mounts`. A design with more than
    /// one camera that could be its perception camera is refused.
    pub fn new(arms: ArmPair<Arm>, mounts: &CameraMounts) -> Result<Self, PerceptionCameraError> {
        Ok(Self {
            arms,
            perception_camera: PerceptionCamera::of(&mounts.design_cameras())?,
            reach_memo: Mutex::new(ReachMemo::new()),
        })
    }

    pub fn perception_camera(&self) -> Option<&PerceptionCamera> {
        self.perception_camera.as_ref()
    }

    /// How an answer checks the view when no linked camera gives the
    /// perception camera's geometry: not at all. The answer then says why:
    /// the robot has no perception camera, or no camera geometry is linked
    /// for the perception camera it names.
    pub fn unlinked_view_check(&self) -> ViewCheck<'static> {
        match self.perception_camera {
            None => ViewCheck::NoPerceptionCamera,
            Some(perception) => ViewCheck::NoCameraGeometry {
                camera: perception.name,
            },
        }
    }

    /// Whether an arm brings its grasp point to `target`, in the robot frame,
    /// to within [`REACH_TOLERANCE`], in one of the grasp orientations: the
    /// arm nearest the target is tried first, so it names the arm when both
    /// reach. When none does, how far the closest stops its grasp point
    /// short of the target in any orientation.
    pub fn reach(&self, target: [f64; 3]) -> Reach {
        let grasps: Vec<Isometry3<f64>> = grasp_poses(target).collect();
        match self
            .sides_nearest_first(target)
            .into_iter()
            .find(|&side| self.reaches(side, &grasps))
        {
            Some(side) => Reach::Reached {
                arm: Side::ARM_NAMES[side.index()].to_owned(),
            },
            None => Reach::Short {
                by: self.shortfall(target),
            },
        }
    }

    /// Where on a flat, level surface at `height` the robot can work, the
    /// view checked as `view_check` says.
    pub fn describe(&self, height: SurfaceHeight, view_check: &ViewCheck<'_>) -> SurfaceAnswer {
        design::describe_surface(&self.surface_reach(height), view_check)
    }

    /// Whether the robot can work each of `positions`, the view checked as
    /// `view_check` says.
    pub fn check(&self, positions: &Positions, view_check: &ViewCheck<'_>) -> PositionsAnswer {
        let reach = PositionsReach::measure(positions, |point| self.reach(point));
        design::check_positions(&reach, view_check)
    }

    /// The reach of the surface at `height`: from the memo when the height
    /// was measured last, else measured now and stored.
    fn surface_reach(&self, height: SurfaceHeight) -> SurfaceReach {
        if let Some(stored) = self.reach_memo().get(height) {
            return stored;
        }
        let measured = SurfaceReach::measure(height, |target| self.reach(target));
        self.reach_memo().insert(measured)
    }

    fn reach_memo(&self) -> MutexGuard<'_, ReachMemo> {
        self.reach_memo
            .lock()
            .expect("no panic while the reach memo is held")
    }

    /// Both sides, the one whose arm base stands nearer `target` first; left
    /// first when they stand as near.
    fn sides_nearest_first(&self, target: [f64; 3]) -> [Side; 2] {
        let target = Vector3::new(target[0], target[1], target[2]);
        let distance = |side: Side| {
            let base = self.arms.get(side).world_pose(&Isometry3::identity());
            (base.translation.vector - target).norm()
        };
        let mut sides = [Side::Left, Side::Right];
        sides.sort_by(|a, b| distance(*a).total_cmp(&distance(*b)));
        sides
    }

    /// Whether the arm of `side` reaches one of `grasps` to within
    /// [`REACH_TOLERANCE`]: the exact solver, seeded at the Ready posture,
    /// finds an in-limit solution at some arm angle for the grasp or, when the
    /// grasp lies within the tolerance of an edge of the arm's reach, outside
    /// or inside it, for the grasp moved into that reach by less than the
    /// tolerance, in the same orientation ([`Arm::solve_ik_within`]).
    fn reaches(&self, side: Side, grasps: &[Isometry3<f64>]) -> bool {
        let arm = self.arms.get(side);
        let seed = openarm_description::ready(side.model());
        grasps.iter().any(|grasp| {
            arm.solve_ik_within(
                &arm.base_pose(grasp),
                REACH_TOLERANCE,
                ArmAnglePolicy::FromSeed,
                &seed,
            )
            .is_some()
        })
    }

    /// How far short of `target`, in the robot frame, the closest arm stops
    /// its grasp point in any orientation, its joint limits aside
    /// ([`Arm::position_shortfall`]): the least over the arms. 0 when the
    /// target lies within an arm's reach in some orientation.
    fn shortfall(&self, target: [f64; 3]) -> f64 {
        let point = Isometry3::translation(target[0], target[1], target[2]);
        [Side::Left, Side::Right]
            .into_iter()
            .map(|side| {
                let arm = self.arms.get(side);
                arm.position_shortfall(&arm.base_pose(&point).translation.vector)
            })
            .fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use openarm_description::HardwareVersion;
    use srs_model::Solution;
    use workspace_core::design::MAX_COORDINATE;
    use workspace_core::messages::robot_frame_placement;
    use workspace_core::{GraspDirection, View};

    fn arms(version: HardwareVersion) -> ArmPair<Arm> {
        let arm = |side: Side| crate::arm_model(version, side.model()).expect("build the arm");
        ArmPair::new(arm(Side::Left), arm(Side::Right))
    }

    /// Poses at `target`, in the robot frame, in orientations spread over
    /// the whole sphere: 400 approach directions along a Fibonacci spiral,
    /// which never points straight down, each at 4 rolls a quarter turn
    /// apart.
    fn poses_all_around(target: [f64; 3]) -> Vec<Isometry3<f64>> {
        const DIRECTIONS: usize = 400;
        let golden_angle = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        (0..DIRECTIONS)
            .flat_map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / DIRECTIONS as f64;
                let radius = (1.0 - z * z).sqrt();
                let (sin, cos) = (golden_angle * i as f64).sin_cos();
                let approach = Vector3::new(radius * cos, radius * sin, z);
                let pointing = UnitQuaternion::rotation_between(&Vector3::z(), &approach)
                    .expect("the spiral never points straight down");
                (0..4).map(move |roll| {
                    let rolled = UnitQuaternion::from_euler_angles(
                        0.0,
                        0.0,
                        roll as f64 * std::f64::consts::FRAC_PI_2,
                    );
                    Isometry3::from_parts(Translation3::from(target), pointing * rolled)
                })
            })
            .collect()
    }

    /// How far the closest arm stops short of the nearest of `poses`, in the
    /// robot frame, its joint limits aside: the least
    /// [`Arm::reach_shortfall`] over the poses and both arms.
    fn least_shortfall(
        arms: &ArmPair<Arm>,
        poses: impl IntoIterator<Item = Isometry3<f64>>,
    ) -> f64 {
        poses
            .into_iter()
            .flat_map(|pose| {
                [Side::Left, Side::Right].map(|side| {
                    let arm = arms.get(side);
                    arm.reach_shortfall(&arm.base_pose(&pose))
                })
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// [`least_shortfall`] of the grasps at `target` in the 16 grasp
    /// orientations.
    fn least_grasp_shortfall(arms: &ArmPair<Arm>, target: [f64; 3]) -> f64 {
        least_shortfall(arms, grasp_poses(target))
    }

    /// The exact solver's in-limit solution for the arm of `side` at `pose`,
    /// in the robot frame, seeded at the Ready posture as [`Workspace::reaches`]
    /// seeds it, without the reach tolerance.
    fn solve_from_ready(
        arms: &ArmPair<Arm>,
        side: Side,
        pose: &Isometry3<f64>,
    ) -> Option<Solution> {
        let arm = arms.get(side);
        arm.solve_ik(
            &arm.base_pose(pose),
            ArmAnglePolicy::FromSeed,
            &openarm_description::ready(side.model()),
        )
    }

    /// Whether an arm's exact solver takes one of `poses`, from the Ready
    /// posture ([`solve_from_ready`]).
    fn an_arm_solves_exactly(
        arms: &ArmPair<Arm>,
        poses: impl IntoIterator<Item = Isometry3<f64>>,
    ) -> bool {
        poses.into_iter().any(|pose| {
            [Side::Left, Side::Right]
                .into_iter()
                .any(|side| solve_from_ready(arms, side, &pose).is_some())
        })
    }

    fn workspace(version: HardwareVersion) -> Workspace {
        let mounts = CameraMounts::resolve(version).expect("the mounts resolve");
        Workspace::new(arms(version), &mounts).expect("one perception camera at most")
    }

    /// The simulated chest camera: 1280 by 720 pixels over a 52° vertical
    /// field of view, measuring optical depths (z) from 0.1 to 10 m.
    fn chest(workspace: &Workspace) -> Camera {
        workspace
            .perception_camera()
            .expect("v2 has a perception camera")
            .camera(
                Intrinsics::from_vertical_fov(52f64.to_radians(), 1280, 720),
                Depth {
                    model: DepthModel::Z,
                    range: [0.1, 10.0],
                },
            )
    }

    /// The view checked through the simulated chest camera ([`chest`]).
    fn chest_view(workspace: &Workspace) -> ViewCheck<'static> {
        let perception = workspace
            .perception_camera()
            .expect("v2 has a perception camera");
        ViewCheck::Checked {
            camera: perception.name,
            geometry: chest(workspace),
        }
    }

    fn height(metres: f64) -> SurfaceHeight {
        SurfaceHeight::from_wire(metres).expect("a finite height")
    }

    fn positions(values: &[f64]) -> Positions {
        Positions::from_wire(values).expect("valid positions")
    }

    #[test]
    fn a_grasp_orientation_is_the_tool_pose_the_solved_gripper_takes() {
        let target = [0.30, 0.0, 0.45];
        let arms = arms(HardwareVersion::V2);
        for side in [Side::Left, Side::Right] {
            let arm = arms.get(side);
            let mut solved_directions = Vec::new();
            for orientation in GraspOrientation::all() {
                let Some(solution) =
                    solve_from_ready(&arms, side, &grasp_pose(orientation, target))
                else {
                    continue;
                };
                let reached = arm.world_pose(&arm.at(&solution.q).ee_pose());
                let position = reached.translation.vector;
                assert!(
                    (position - Vector3::from(target)).norm() < 1e-6,
                    "{side:?} {orientation:?}: the grasp point stands at {position}"
                );
                let rotation = orientation.rotation();
                let asked = |column: usize| {
                    Vector3::new(rotation[column], rotation[3 + column], rotation[6 + column])
                };
                let approach = reached.rotation * Vector3::z();
                assert!(
                    (approach - Vector3::from(orientation.direction.approach())).norm() < 1e-6,
                    "{side:?} {orientation:?}: the gripper points along {approach}"
                );
                assert!((approach - asked(2)).norm() < 1e-6);
                let jaws = reached.rotation * Vector3::y();
                assert!(
                    (jaws - asked(1)).norm() < 1e-6,
                    "{side:?} {orientation:?}: the jaws close along {jaws}"
                );
                solved_directions.push(orientation.direction);
            }
            for direction in GraspDirection::ALL {
                assert!(
                    solved_directions.contains(&direction),
                    "{side:?} reaches the target pointing {}",
                    direction.name()
                );
            }
        }
    }

    #[test]
    fn the_perception_camera_of_v2_is_the_chest_camera_and_v1_has_none() {
        let v2 = workspace(HardwareVersion::V2);
        let perception = v2.perception_camera().expect("v2 has one");
        assert_eq!(perception.name, "chest");
        let camera = chest(&v2);
        assert_eq!(camera.position, [0.0792, 0.0315, 0.7941]);
        // The optical axis, the third column, looks forward and down.
        let (view_x, view_z) = (camera.rotation[2], camera.rotation[8]);
        assert!(view_x > 0.4 && view_z < -0.8, "({view_x}, {view_z})");
        assert_eq!(workspace(HardwareVersion::V1).perception_camera(), None);
    }

    #[test]
    fn a_point_in_front_at_table_height_is_reached_and_seen() {
        let v2 = workspace(HardwareVersion::V2);
        let answer = v2.check(&positions(&[0.30, 0.0, 0.45]), &chest_view(&v2));
        let [point] = answer.points.as_slice() else {
            panic!("one point answered: {answer:?}");
        };
        assert_eq!(point.position, [0.30, 0.0, 0.45]);
        assert!(point.reach.reached(), "{point:?}");
        assert_eq!(point.view, View::Seen);
        assert!(point.workable() && answer.all_workable());
        assert_eq!(
            point.message,
            format!(
                "Workable: {} reaches it and the chest camera sees it.",
                point.reach.arm()
            )
        );
        assert_eq!(answer.message, "The point is workable.");
    }

    #[test]
    fn a_point_a_metre_in_front_is_out_of_reach_by_how_far_the_closest_arm_stops() {
        let v2 = workspace(HardwareVersion::V2);
        let target = [1.0, 0.0, 0.45];
        let answer = v2.check(&positions(&target), &chest_view(&v2));
        let point = &answer.points[0];
        let Reach::Short { by } = point.reach else {
            panic!("a metre ahead is out of reach: {point:?}");
        };
        assert!(by > REACH_TOLERANCE, "{by}");
        // The closest arm stops `by` short in its best orientation, whichever
        // it is: no orientation of the sphere stops it shorter, and the best
        // of them comes within a few millimetres of `by`.
        let arms = arms(HardwareVersion::V2);
        let least_sampled = least_shortfall(&arms, poses_all_around(target));
        assert!(by <= least_sampled + 1e-9, "{by} > {least_sampled}");
        assert!(least_sampled - by < 0.003, "{by} vs {least_sampled}");
        let least_grasp = least_grasp_shortfall(&arms, target);
        assert!(
            by < least_grasp,
            "the grasp orientations stop {least_grasp} short"
        );
        assert!(!answer.all_workable());
        assert!(
            point
                .message
                .starts_with(&format!("Not workable: it is out of reach by {by:.2} m")),
            "{}",
            point.message
        );
        assert_eq!(answer.message, "The point is not workable.");
    }

    #[test]
    fn a_point_the_grasp_point_gets_to_tilted_falls_short_by_nothing_in_a_grasp_orientation() {
        let v2 = workspace(HardwareVersion::V2);
        let target = [0.40, 0.0, 0.30];
        let answer = v2.check(&positions(&target), &chest_view(&v2));
        let point = &answer.points[0];
        assert_eq!(point.reach, Reach::Short { by: 0.0 });
        assert_eq!(
            point.message,
            "Not workable: no arm reaches it with its gripper pointing down or forward; \
             the chest camera sees it."
        );
        let arms = arms(HardwareVersion::V2);
        assert!(
            an_arm_solves_exactly(&arms, poses_all_around(target)),
            "an arm gets its grasp point there, tilted"
        );
    }

    #[test]
    fn a_point_beyond_an_arms_reach_by_less_than_the_reach_tolerance_is_reached() {
        let version = HardwareVersion::V2;
        let mounts = CameraMounts::resolve(version).expect("the mounts resolve");
        let arms = arms(version);
        // How far the arms' wrist centres stand outside their reach for the
        // best grasp orientation at `x` ahead, 0.2 m to the left and 0.2 m
        // up.
        let beyond = |x: f64| least_grasp_shortfall(&arms, [x, 0.2, 0.2]);
        let half_the_tolerance = REACH_TOLERANCE / 2.0;
        let (mut inside, mut outside) = (0.2, 1.0);
        for _ in 0..60 {
            let middle = (inside + outside) / 2.0;
            if beyond(middle) < half_the_tolerance {
                inside = middle;
            } else {
                outside = middle;
            }
        }
        let target = [outside, 0.2, 0.2];
        assert!((beyond(outside) - half_the_tolerance).abs() < 1e-9);
        assert!(
            !an_arm_solves_exactly(&arms, grasp_poses(target)),
            "no arm takes a grasp exactly"
        );
        let workspace = Workspace::new(arms, &mounts).expect("one perception camera at most");
        assert_eq!(
            workspace.reach(target),
            Reach::Reached {
                arm: "left_arm".to_owned()
            }
        );
    }

    #[test]
    fn the_arm_on_the_side_of_a_point_names_its_reach_and_the_left_one_a_point_between() {
        let v2 = workspace(HardwareVersion::V2);
        let reaches =
            [[0.30, 0.2, 0.45], [0.30, -0.2, 0.45], [0.30, 0.0, 0.45]].map(|point| v2.reach(point));
        let arms = reaches.each_ref().map(Reach::arm);
        assert_eq!(arms, ["left_arm", "right_arm", "left_arm"]);
    }

    #[test]
    fn a_surface_at_table_height_is_workable_inside_both_the_reach_and_the_view() {
        let v2 = workspace(HardwareVersion::V2);
        let answer = v2.describe(height(0.45), &chest_view(&v2));
        assert!(answer.workable, "{}", answer.message);
        let [x_min, x_max, y_min, y_max] = answer.rectangle.expect("a rectangle");
        for bounds in [answer.reach, answer.view] {
            let [bx_min, bx_max, by_min, by_max] = bounds.expect("bounds");
            assert!(bx_min <= x_min && x_max <= bx_max && by_min <= y_min && y_max <= by_max);
        }
        assert!(answer.area >= workspace_core::MIN_WORKABLE_AREA);
        let rectangle = workspace_core::Rectangle {
            x: [x_min, x_max],
            y: [y_min, y_max],
        };
        assert_eq!(
            answer.message,
            format!(
                "Workable: {:.3} m². {}",
                answer.area,
                robot_frame_placement(&rectangle)
            )
        );
    }

    #[test]
    fn a_surface_above_the_chest_camera_is_not_visible_and_says_how_far_above_it_is() {
        let v2 = workspace(HardwareVersion::V2);
        let answer = v2.describe(height(0.96), &chest_view(&v2));
        assert!(!answer.workable);
        assert!(answer.reach.is_some(), "the arms reach part of it");
        assert_eq!((answer.view, answer.rectangle), (None, None));
        assert_eq!(answer.area, 0.0);
        assert_eq!(
            answer.message,
            "Not visible: the surface is 0.17 m above the chest camera, which cannot see it."
        );
    }

    #[test]
    fn a_surface_just_below_the_chest_camera_says_how_far_ahead_the_arms_reach_it() {
        let v2 = workspace(HardwareVersion::V2);
        let answer = v2.describe(height(0.75), &chest_view(&v2));
        assert!(!answer.workable);
        assert_eq!((answer.view, answer.rectangle), (None, None));
        let [x_min, x_max, ..] = answer.reach.expect("the arms reach part of it");
        assert_eq!(
            answer.message,
            format!(
                "Not visible: the arms reach it from {x_min:.2} to {x_max:.2} m ahead of the robot, \
                 but the chest camera sees none of it."
            )
        );
    }

    #[test]
    fn a_robot_without_a_perception_camera_is_judged_on_reach_alone_and_says_so() {
        let v1 = workspace(HardwareVersion::V1);
        let view_check = v1.unlinked_view_check();
        assert_eq!(view_check, ViewCheck::NoPerceptionCamera);
        let unchecked = "The view is not checked: the robot has no perception camera.";
        let checked = v1.check(&positions(&[0.30, -0.2, 0.45]), &view_check);
        let [point] = checked.points.as_slice() else {
            panic!("one point answered: {checked:?}");
        };
        assert_eq!(
            point.reach,
            Reach::Reached {
                arm: "right_arm".to_owned()
            }
        );
        assert_eq!(point.view, View::NoCamera);
        assert_eq!(
            checked.message,
            format!("The point is workable. {unchecked}")
        );
        let surface = v1.describe(height(0.45), &view_check);
        assert!(surface.workable, "{}", surface.message);
        assert_eq!(surface.view, None);
        assert!(surface.message.ends_with(unchecked), "{}", surface.message);
    }

    #[test]
    fn a_perception_camera_with_no_linked_geometry_leaves_the_view_unchecked_and_names_it() {
        let v2 = workspace(HardwareVersion::V2);
        let view_check = v2.unlinked_view_check();
        assert_eq!(view_check, ViewCheck::NoCameraGeometry { camera: "chest" });
        let unchecked =
            "The view is not checked: no camera geometry is linked for the chest camera.";
        let checked = v2.check(&positions(&[0.30, 0.0, 0.45]), &view_check);
        let [point] = checked.points.as_slice() else {
            panic!("one point answered: {checked:?}");
        };
        assert!(point.reach.reached(), "{point:?}");
        assert_eq!(point.view, View::NoCamera);
        assert_eq!(
            checked.message,
            format!("The point is workable. {unchecked}")
        );
        let surface = v2.describe(height(0.45), &view_check);
        assert!(surface.workable, "{}", surface.message);
        assert_eq!(surface.view, None);
        assert!(surface.message.ends_with(unchecked), "{}", surface.message);
    }

    #[test]
    fn a_measured_surface_is_stored_and_its_height_answers_from_the_stored_reach() {
        let v2 = workspace(HardwareVersion::V2);
        let view = chest_view(&v2);
        let table = height(0.45);
        assert!(v2.describe(table, &view).workable);
        assert_eq!(v2.reach_memo().len(), 1);
        let table_reach = SurfaceReach::measure(table, |target| v2.reach(target));
        assert_eq!(
            v2.reach_memo().get(table),
            Some(table_reach),
            "the reach measured for the table is stored"
        );
        // The arms reach part of a shelf 0.30 m up, but the reach stored for
        // its height says that no arm reaches any of it: the answer is the
        // stored one.
        let shelf = height(0.30);
        let shelf_reach = SurfaceReach::measure(shelf, |target| v2.reach(target));
        assert!(shelf_reach.reaches().iter().any(Reach::reached));
        v2.reach_memo()
            .insert(SurfaceReach::measure(shelf, |_| Reach::Short { by: 0.5 }));
        let answer = v2.describe(shelf, &view);
        assert!(!answer.workable);
        assert_eq!(answer.reach, None, "{}", answer.message);
        assert_eq!(v2.reach_memo().len(), 2);
    }

    #[test]
    fn a_point_at_the_coordinate_bound_falls_short_by_a_finite_distance() {
        let v2 = workspace(HardwareVersion::V2);
        let Reach::Short { by } = v2.reach([MAX_COORDINATE, 0.0, 0.0]) else {
            panic!("a kilometre ahead is out of reach");
        };
        assert!(by.is_finite() && by > 990.0, "{by}");
    }

    #[test]
    fn a_camera_geometry_that_is_no_pinhole_is_refused() {
        let model = intrinsics_from_wire(1280, 720, 738.1, 738.1, 639.5, 359.5).expect("valid");
        assert_eq!((model.width, model.fx, model.cy), (1280, 738.1, 359.5));
        for (width, height, fx, fy, cx, cy) in [
            (0, 720, 738.1, 738.1, 639.5, 359.5),
            (1280, 0, 738.1, 738.1, 639.5, 359.5),
            (1280, 720, 0.0, 738.1, 639.5, 359.5),
            (1280, 720, 738.1, -1.0, 639.5, 359.5),
            (1280, 720, f64::NAN, 738.1, 639.5, 359.5),
            (1280, 720, 738.1, 738.1, f64::INFINITY, 359.5),
            (1280, 720, 738.1, 738.1, 639.5, f64::NAN),
        ] {
            assert!(
                intrinsics_from_wire(width, height, fx, fy, cx, cy).is_err(),
                "{width}x{height} fx {fx} fy {fy} cx {cx} cy {cy}"
            );
        }
    }

    #[test]
    fn a_depth_stream_is_parsed_with_its_model_and_one_of_an_unknown_model_or_range_is_refused() {
        assert_eq!(
            depth_from_wire("z", 0.1, 10.0),
            Ok(Depth {
                model: DepthModel::Z,
                range: [0.1, 10.0]
            })
        );
        assert_eq!(
            depth_from_wire("range", 0.0, 10.0),
            Ok(Depth {
                model: DepthModel::Range,
                range: [0.0, 10.0]
            })
        );
        for model in ["", "disparity", "Z"] {
            let refused = depth_from_wire(model, 0.1, 10.0);
            assert_eq!(
                refused,
                Err(DepthFromWireError::Model(UnknownDepthModel(model.into())))
            );
            let reason = refused.unwrap_err().to_string();
            assert!(reason.contains(&format!("'{model}'")), "{reason}");
        }
        for (nearest, farthest) in [(-0.1, 10.0), (2.0, 2.0), (3.0, 1.0), (0.1, f64::INFINITY)] {
            assert_eq!(
                depth_from_wire("z", nearest, farthest),
                Err(DepthFromWireError::Range),
                "{nearest}..{farthest}"
            );
        }
    }

    #[test]
    fn a_range_camera_does_not_see_a_point_whose_distance_passes_its_farthest_depth() {
        let v2 = workspace(HardwareVersion::V2);
        let perception = v2.perception_camera().expect("v2 has one");
        let intrinsics = Intrinsics::from_vertical_fov(52f64.to_radians(), 1280, 720);
        let camera = |model: &str| {
            perception.camera(
                intrinsics,
                depth_from_wire(model, 0.1, 0.6).expect("a depth stream"),
            )
        };
        // Straight ahead along the optical axis, then 0.25 m to the right of
        // the image: optical z 0.55, inside the range, but 0.60 m away.
        let (position, rotation) = (perception.position, perception.rotation);
        let point: [f64; 3] = std::array::from_fn(|i| {
            position[i] + 0.55 * rotation[3 * i + 2] + 0.25 * rotation[3 * i]
        });
        assert_eq!(camera("z").view_of(point), View::Seen);
        assert_eq!(camera("range").view_of(point), View::OutOfDepth);
    }

    #[test]
    fn a_camera_geometry_refusal_names_the_perception_camera_and_what_the_camera_answered() {
        let reason = || "the camera gives colour alone".to_owned();
        let cases = [
            (
                CameraGeometryError::Colour {
                    camera: "chest",
                    reason: reason(),
                },
                "cannot read the colour intrinsics of the chest camera, the perception camera: the camera gives colour alone",
            ),
            (
                CameraGeometryError::Depth {
                    camera: "chest",
                    reason: reason(),
                },
                "cannot read the depth intrinsics of the chest camera, the perception camera: the camera gives colour alone",
            ),
            (
                CameraGeometryError::NoDepth {
                    camera: "chest",
                    reason: reason(),
                },
                "the camera linked as the chest camera, the perception camera, gives no depth: the camera gives colour alone",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }
}
