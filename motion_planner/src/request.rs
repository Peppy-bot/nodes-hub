//! The requests of plan_joint_path, plan_cartesian_path and check_joint_path, parsed
//! against the robot's arms: the arms, where each one's path starts, and each arm's goal,
//! its poses or its waypoints, in the terms of motion_planning_core. Each record of a
//! request names its frame, the robot frame when it names none, and its timestamp: a
//! constraint in a link of an arm the plan moves, or in a link fixed to one, is held against
//! that link at each state, as MoveIt holds a constraint whose frame is a link of the robot;
//! every other record is placed in the robot frame by the planner's tree at its timestamp. A
//! request that does not parse is refused with `rejected`, and the message says what to send
//! instead; a record the tree cannot place is refused with the failure of its placement.

use std::borrow::Borrow;
use std::time::{Duration, SystemTime};

use chain_kinematics::ChainFrame;
use chain_kinematics::nalgebra::{Isometry3, UnitQuaternion, Vector3};
use collision_geometry::chain::Rider;
use control_core::geometry::{GeometryError, Point, Pose, Stamped};
use control_core::positive_finite::PositiveFinite;
use control_core::requested_arms::{common_waypoint_count, place_of_arm, places_in_order};
use control_core::robot_tree::PlaceError;
use frame_tree::FrameName;
use motion_planning_core::{
    Axis, FollowedFrame, Frame, Goal, GoalError, JointTarget, OrientationTarget, PositionTarget,
    Reference,
};
use peppygen::exposed_actions::motion_planning::plan_cartesian_path;
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    GoalGoalJointConstraintsItem, GoalRequestData,
};
use peppygen::exposed_services::motion_planning::check_joint_path;
use robot_description::{ArmId, PerArm};

use crate::robot::ArmModel;

/// The longest a plan may plan, in seconds.
pub const MAX_PLANNING_TIME_S: f64 = 60.0;

/// A choice a plan_joint_path goal names by one of its field's names.
pub trait Named: Copy + PartialEq + 'static {
    /// The goal's field that names it.
    const FIELD: &'static str;
    /// Every choice, in the contract's order.
    const ALL: &'static [Self];

    /// The name a goal gives it.
    fn name(self) -> &'static str;

    /// The choice named `name`, refused with the names a goal can send.
    fn named(name: &str) -> Result<Self, String> {
        Self::ALL
            .iter()
            .copied()
            .find(|choice| choice.name() == name)
            .ok_or_else(|| {
                let names: Vec<String> = Self::ALL
                    .iter()
                    .map(|choice| format!("{:?}", choice.name()))
                    .collect();
                format!("{} is {}, not {name:?}", Self::FIELD, listed(&names, "or"))
            })
    }
}

/// The planner a plan_joint_path goal names, as MoveIt's MotionPlanRequest names its
/// pipeline_id and planner_id.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Planner {
    /// RRTConnect, MoveIt's pipeline "ompl" with planner_id "RRTConnectkConfigDefault":
    /// the first valid path it finds, simplified in the time left.
    RrtConnect,
    /// RRT*, MoveIt's pipeline "ompl" with planner_id "RRTstarkConfigDefault": a path it
    /// shortens until the plan's time ends, then simplified once.
    RrtStar,
}

impl Named for Planner {
    const FIELD: &'static str = "planner";
    const ALL: &'static [Self] = &[Self::RrtConnect, Self::RrtStar];

    fn name(self) -> &'static str {
        match self {
            Self::RrtConnect => "rrt_connect",
            Self::RrtStar => "rrt_star",
        }
    }
}

/// What refines the planner's path of a plan_joint_path goal, as a MoveIt pipeline chains
/// a refining planner after its planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Refiner {
    /// The planner's path, as it answers it.
    None,
    /// The planner's path refined by MoveIt's CHOMP, every iteration of it
    /// (`motion_planning_core::Refiner::Chomp`): the refined path when it passes the plan's
    /// checks and costs less in CHOMP's cost, and the planner's path otherwise.
    Chomp,
}

impl Named for Refiner {
    const FIELD: &'static str = "refiner";
    const ALL: &'static [Self] = &[Self::None, Self::Chomp];

    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Chomp => "chomp",
        }
    }
}

impl From<Planner> for ompl_bridge::Planner {
    fn from(planner: Planner) -> Self {
        match planner {
            Planner::RrtConnect => Self::RrtConnect,
            Planner::RrtStar => Self::RrtStar,
        }
    }
}

impl From<Refiner> for Option<motion_planning_core::Refiner> {
    fn from(refiner: Refiner) -> Self {
        match refiner {
            Refiner::None => None,
            Refiner::Chomp => Some(motion_planning_core::Refiner::Chomp),
        }
    }
}

/// `names` as a list in a sentence, the last joined by `conjunction`: "a", "a and b",
/// "a, b or c".
pub fn listed<S: Borrow<str>>(names: &[S], conjunction: &str) -> String {
    match names {
        [] => String::new(),
        [one] => one.borrow().to_string(),
        [init @ .., last] => format!("{} {conjunction} {}", init.join(", "), last.borrow()),
    }
}

/// Why a request is refused: it does not parse, or the planner's tree cannot place a record of
/// it at its timestamp.
#[derive(Debug)]
pub enum RequestError {
    /// Refused with `rejected`, the message saying what to send instead.
    Rejected(String),
    /// Refused with the failure kind of the placement of a record (`PlaceError::failure_kind`).
    Unplaced(PlaceError),
}

impl From<String> for RequestError {
    fn from(message: String) -> Self {
        Self::Rejected(message)
    }
}

impl From<PlaceError> for RequestError {
    fn from(error: PlaceError) -> Self {
        Self::Unplaced(error)
    }
}

/// What places a value `T` of a request, in the frame it names at its timestamp, in the robot
/// frame: the planner's tree.
pub type ValuePlacer<'p, T> = dyn Fn(&Stamped<T>) -> Result<T, PlaceError> + 'p;

/// The frame a record names: `frame`, or the robot frame `robot` for a record that names none.
pub fn frame_or_robot<'f>(frame: Option<&'f str>, robot: &'f FrameName) -> &'f str {
    frame.unwrap_or(robot.as_str())
}

/// A plan_joint_path goal that parses.
#[derive(Debug)]
pub struct PlanQuery<const N: usize, const K: usize> {
    /// The planner that plans the path.
    pub planner: Planner,
    /// What refines the planner's path.
    pub refiner: Refiner,
    /// The goal's arms, in its order: the order of the answer.
    pub order: Vec<ArmId>,
    /// Where each arm's path starts.
    pub starts: ArmStarts<N, K>,
    /// The arms the plan moves, each to its goal; every other arm of `order` stays where
    /// it starts.
    pub arm_goals: ArmGoals<K>,
    /// The orientations each arm's path holds, each in a frame of an arm the plan moves at
    /// that arm's place in the plan's arms, or in the robot frame.
    pub path_orientations: PerArm<Vec<HeldOrientation>, K>,
    /// How long the plan may take.
    pub time_limit: Duration,
}

/// An orientation a path holds of a link of an arm: the link as sent, and the target.
#[derive(Debug)]
pub struct HeldOrientation {
    pub link_name: String,
    pub target: OrientationTarget,
}

/// The arms a plan moves, each with its goal: one, or every arm of a robot of more than one.
#[derive(Debug)]
pub enum ArmGoals<const K: usize> {
    One {
        arm: ArmId,
        goal: Box<Goal>,
    },
    /// Every arm, which a state of the plan holds in the robot's order.
    Every(Box<PerArm<Goal, K>>),
}

/// A check_joint_path request that parses.
#[derive(Debug)]
pub struct CheckQuery<const N: usize, const K: usize> {
    /// Where each arm's path starts.
    pub starts: ArmStarts<N, K>,
    /// The waypoints after the start of the arms the request names, at least one.
    pub waypoints: Checked<N, K>,
}

/// A plan_cartesian_path goal that parses.
#[derive(Debug)]
pub struct CartesianQuery<const N: usize, const K: usize> {
    /// The arm the path moves.
    pub arm: ArmId,
    /// Where its path starts.
    pub starts: ArmStarts<N, K>,
    /// What follows the lines.
    pub frame: FollowedFrame,
    /// The poses it passes through.
    pub poses: CartesianPoses,
    pub avoid_collisions: bool,
}

/// The poses a Cartesian path passes through: placed in the robot frame by the planner's
/// tree, or in a link of the path's arm, placed where that link stands at the path's start,
/// as MoveIt's Cartesian path places a target in its link's frame.
#[derive(Debug)]
pub enum CartesianPoses {
    Robot(Vec<Isometry3<f64>>),
    Link {
        frame: Frame,
        poses: Vec<Isometry3<f64>>,
    },
}

impl CartesianPoses {
    /// The poses in the robot frame, the arm of `chain` at `start`.
    pub fn at_start<const N: usize>(
        &self,
        chain: &chain_kinematics::Chain<N>,
        start: &[f64; N],
    ) -> Vec<Isometry3<f64>> {
        match self {
            Self::Robot(poses) => poses.clone(),
            Self::Link { frame, poses } => {
                let link = ChainFrame::from(*frame).pose_at(&chain.at(start));
                poses.iter().map(|pose| link * pose).collect()
            }
        }
    }
}

/// The waypoints of a check: one arm's, or every arm's of a robot of more than one, as many
/// for each.
#[derive(Debug, PartialEq)]
pub enum Checked<const N: usize, const K: usize> {
    One {
        arm: ArmId,
        waypoints: Vec<[f64; N]>,
    },
    Every(Vec<PerArm<[f64; N], K>>),
}

impl<const N: usize, const K: usize> Checked<N, K> {
    /// The waypoints of `arm`; none for an arm the check does not name.
    pub fn of(&self, arm: ArmId) -> Option<Vec<[f64; N]>> {
        match self {
            Self::One {
                arm: named,
                waypoints,
            } => (*named == arm).then(|| waypoints.clone()),
            Self::Every(waypoints) => Some(waypoints.iter().map(|all| all[arm]).collect()),
        }
    }
}

/// A position or an orientation constraint of a request: the arm whose link it holds, the
/// link as sent and its frame of the arm, the pose its record gives in the frame it names
/// at its time, and what it holds of the link.
struct Constraint<'r, H> {
    arm: ArmId,
    name: &'r str,
    link: Frame,
    stamped: Stamped<Pose>,
    held: H,
}

/// What an orientation constraint holds of its link.
#[derive(Clone, Copy)]
struct Turn {
    tolerance_rad: PositiveFinite,
    free_axis: Option<Axis>,
}

/// Where a constraint's pose is: in a frame of an arm the plan moves, held against that
/// frame at each state, or placed in the robot frame by the planner's tree.
#[derive(Clone, Copy)]
enum Located {
    Mobile {
        arm: ArmId,
        frame: Frame,
        pose: Pose,
    },
    Robot(Pose),
}

impl Located {
    /// The reference of the pose, an arm at its place in the plan's arms `group`, and the
    /// pose in it.
    fn in_group(self, group: &[ArmId]) -> (Reference, Pose) {
        match self {
            Self::Robot(pose) => (Reference::Robot, pose),
            Self::Mobile { arm, frame, pose } => {
                let arm = group
                    .iter()
                    .position(|named| *named == arm)
                    .expect("a frame of an arm the plan moves is a frame of its arms");
                (Reference::Arm { arm, frame }, pose)
            }
        }
    }
}

/// Parses a plan_joint_path goal against `arms`, whose fingers hang as `fingers` hold
/// them, each constraint with no frame in the robot frame `robot`, and each in a frame of no arm
/// the plan moves placed by `place`.
pub fn parse_plan<const N: usize, const K: usize>(
    raw: &GoalRequestData,
    arms: &PerArm<ArmModel<N>, K>,
    fingers: &PerArm<Vec<Rider>, K>,
    robot: &FrameName,
    place: &ValuePlacer<Pose>,
) -> Result<PlanQuery<N, K>, RequestError> {
    let (order, starts) = named_arms(
        arms,
        raw.arms
            .iter()
            .map(|a| (a.arm_name.as_str(), a.start_joint_positions.as_slice())),
    )?;
    named_once(
        "goal_joint_constraints",
        raw.goal_joint_constraints
            .iter()
            .map(|c| c.joint_name.as_str()),
    )?;
    named_once(
        "goal_position_constraints",
        raw.goal_position_constraints
            .iter()
            .map(|c| c.link_name.as_str()),
    )?;
    named_once(
        "goal_orientation_constraints",
        raw.goal_orientation_constraints
            .iter()
            .map(|c| c.link_name.as_str()),
    )?;
    named_once(
        "path_orientation_constraints",
        raw.path_orientation_constraints
            .iter()
            .map(|c| c.link_name.as_str()),
    )?;
    let named: Vec<&ArmModel<N>> = order.iter().map(|arm| &arms[*arm]).collect();
    let joints = raw
        .goal_joint_constraints
        .iter()
        .map(|c| joint_target(&named, c))
        .collect::<Result<Vec<_>, String>>()?;
    let positions = (raw.goal_position_constraints.iter())
        .map(|c| {
            let within = tolerance("tolerance_m", &c.link_name, c.tolerance_m)?;
            let point = Point::parse(c.position)
                .map_err(|e| of_record("goal_position_constraints", &c.link_name, &e))?;
            let record = Record {
                link_name: &c.link_name,
                frame: frame_or_robot(c.frame.as_deref(), robot),
                timestamp: c.timestamp,
                pose: Pose::new(point, control_core::geometry::UnitQuaternion::IDENTITY),
            };
            constraint(&named, fingers, "goal_position_constraints", record, within)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let orientations = (raw.goal_orientation_constraints.iter())
        .map(|c| {
            let record = OrientationRecord {
                link_name: &c.link_name,
                frame: frame_or_robot(c.frame.as_deref(), robot),
                timestamp: c.timestamp,
                orientation: c.orientation,
                tolerance_rad: c.tolerance_rad,
                free_axis: &c.free_axis,
            };
            orientation_constraint(&named, fingers, "goal_orientation_constraints", record)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let paths = (raw.path_orientation_constraints.iter())
        .map(|c| {
            let record = OrientationRecord {
                link_name: &c.link_name,
                frame: frame_or_robot(c.frame.as_deref(), robot),
                timestamp: c.timestamp,
                orientation: c.orientation,
                tolerance_rad: c.tolerance_rad,
                free_axis: &c.free_axis,
            };
            orientation_constraint(&named, fingers, "path_orientation_constraints", record)
        })
        .collect::<Result<Vec<_>, String>>()?;
    goals_of(
        arms,
        &joints,
        &unplaced(&positions),
        &unplaced(&orientations),
        &[],
    )?;
    let planner = Planner::named(&raw.planner)?;
    let refiner = Refiner::named(&raw.refiner)?;
    let time = raw.allowed_planning_time_s;
    if !(time > 0.0 && time <= MAX_PLANNING_TIME_S) {
        return Err(RequestError::Rejected(format!(
            "allowed_planning_time_s is {time}; send the seconds the plan may take, more than 0 \
             and at most {MAX_PLANNING_TIME_S}"
        )));
    }
    let moving: Vec<ArmId> = PerArm::<(), K>::ids()
        .filter(|arm| {
            joints.iter().any(|c| c.arm == *arm)
                || positions.iter().any(|c| c.arm == *arm)
                || orientations.iter().any(|c| c.arm == *arm)
        })
        .collect();
    let moving_models: Vec<&ArmModel<N>> = moving.iter().map(|arm| &arms[*arm]).collect();
    let carrier = |stamped: &Stamped<Pose>| carrier_of(stamped, &moving_models, fingers);
    let locator = |stamped: &Stamped<Pose>| -> Result<Located, PlaceError> {
        Ok(match carrier(stamped) {
            Some((arm, frame)) => Located::Mobile {
                arm,
                frame,
                pose: *stamped.value(),
            },
            None => Located::Robot(place(stamped)?),
        })
    };
    let positions = located(&positions, &locator)?;
    let orientations = located(&orientations, &locator)?;
    let paths = located(&paths, &locator)?;
    let goals = goals_of(arms, &joints, &positions, &orientations, &moving)?;
    let arm_goals = arm_goals(goals, &moving).ok_or_else(|| GoalError::Empty.to_string())?;
    Ok(PlanQuery {
        planner,
        refiner,
        order,
        starts,
        arm_goals,
        path_orientations: PerArm::from_fn(|arm| {
            (paths.iter())
                .filter(|(c, _)| c.arm == arm)
                .map(|(c, located)| HeldOrientation {
                    link_name: c.name.to_string(),
                    target: orientation_target(c, *located, &moving),
                })
                .collect()
        }),
        time_limit: Duration::from_secs_f64(time),
    })
}

/// Each arm's goal of `joints`, `positions` and `orientations`, each pose where its location
/// puts it, an arm at its place in the plan's arms `group`; none for an arm no constraint
/// names. A goal that mixes joints with poses, or names more than one link of its arm, is
/// refused, saying what to send.
fn goals_of<const N: usize, const K: usize>(
    arms: &PerArm<ArmModel<N>, K>,
    joints: &[JointConstraint],
    positions: &[(&Constraint<'_, PositiveFinite>, Located)],
    orientations: &[(&Constraint<'_, Turn>, Located)],
    group: &[ArmId],
) -> Result<PerArm<Option<Goal>, K>, String> {
    PerArm::from_fn(|arm| {
        let joints: Vec<JointTarget> = (joints.iter())
            .filter(|c| c.arm == arm)
            .map(|c| c.target)
            .collect();
        let positions: Vec<_> = (positions.iter())
            .filter(|(c, _)| c.arm == arm)
            .copied()
            .collect();
        let orientations: Vec<_> = (orientations.iter())
            .filter(|(c, _)| c.arm == arm)
            .copied()
            .collect();
        let constrained = !(joints.is_empty() && positions.is_empty() && orientations.is_empty());
        constrained
            .then(|| {
                let links: (Vec<&str>, Vec<&str>) = (
                    positions.iter().map(|(c, _)| c.name).collect(),
                    orientations.iter().map(|(c, _)| c.name).collect(),
                );
                Goal::new(
                    N,
                    joints,
                    (positions.into_iter())
                        .map(|(c, located)| position_target(c, located, group))
                        .collect(),
                    (orientations.into_iter())
                        .map(|(c, located)| orientation_target(c, located, group))
                        .collect(),
                )
                .map_err(|e| goal_refusal(&arms[arm], e, links))
            })
            .transpose()
    })
    .transpose()
}

/// Each of `constraints` where its record gives its pose, unplaced, as a refusal of the
/// goal's shape reads it.
fn unplaced<'c, 'r, H>(
    constraints: &'c [Constraint<'r, H>],
) -> Vec<(&'c Constraint<'r, H>, Located)> {
    (constraints.iter())
        .map(|c| (c, Located::Robot(*c.stamped.value())))
        .collect()
}

/// Each of `constraints` where `locate` puts its pose: in a frame of an arm the plan moves,
/// or placed in the robot frame; the failure of the first it cannot place.
fn located<'c, 'r, H>(
    constraints: &'c [Constraint<'r, H>],
    locate: &dyn Fn(&Stamped<Pose>) -> Result<Located, PlaceError>,
) -> Result<Vec<(&'c Constraint<'r, H>, Located)>, PlaceError> {
    (constraints.iter())
        .map(|c| Ok((c, locate(&c.stamped)?)))
        .collect()
}

/// A record of a constraint as a request gives it: the link it holds, as sent, the frame
/// and the timestamp it names, none for the latest time, and its pose in that frame.
struct Record<'r> {
    link_name: &'r str,
    frame: &'r str,
    timestamp: Option<SystemTime>,
    pose: Pose,
}

/// A record of an orientation constraint as a request gives it.
struct OrientationRecord<'r> {
    link_name: &'r str,
    frame: &'r str,
    timestamp: Option<SystemTime>,
    orientation: [f64; 4],
    tolerance_rad: f64,
    free_axis: &'r str,
}

/// The constraint of `record` of `list`, a link of an arm of `named`, whose fingers hang as
/// `fingers` hold them, holding `held`.
fn constraint<'r, H, const N: usize, const K: usize>(
    named: &[&ArmModel<N>],
    fingers: &PerArm<Vec<Rider>, K>,
    list: &str,
    record: Record<'r>,
    held: H,
) -> Result<Constraint<'r, H>, String> {
    let (arm, link) = frame(named, fingers, record.link_name)?;
    let stamped = Stamped::parse(record.frame, record.timestamp, record.pose)
        .map_err(|e| of_record(list, record.link_name, &e))?;
    Ok(Constraint {
        arm,
        name: record.link_name,
        link,
        stamped,
        held,
    })
}

/// The orientation constraint of `record` of `list`, a link of an arm of `named`, whose
/// fingers hang as `fingers` hold them.
fn orientation_constraint<'r, const N: usize, const K: usize>(
    named: &[&ArmModel<N>],
    fingers: &PerArm<Vec<Rider>, K>,
    list: &str,
    record: OrientationRecord<'r>,
) -> Result<Constraint<'r, Turn>, String> {
    let turn = Turn {
        tolerance_rad: tolerance("tolerance_rad", record.link_name, record.tolerance_rad)?,
        free_axis: Axis::free_named(record.free_axis)?,
    };
    let pose = Pose::parse([0.0; 3], record.orientation)
        .map_err(|e| of_record(list, record.link_name, &e))?;
    let record = Record {
        link_name: record.link_name,
        frame: record.frame,
        timestamp: record.timestamp,
        pose,
    };
    constraint(named, fingers, list, record, turn)
}

/// The arm of `moving`, whose fingers hang as `fingers` hold them, whose frame `stamped`
/// names, a link of it or a link fixed to one, and that frame; none for a frame of no arm of
/// `moving`.
fn carrier_of<const N: usize, const K: usize>(
    stamped: &Stamped<Pose>,
    moving: &[&ArmModel<N>],
    fingers: &PerArm<Vec<Rider>, K>,
) -> Option<(ArmId, Frame)> {
    let name = stamped.frame().as_str();
    moving
        .iter()
        .find_map(|arm| frame_of(arm, &fingers[arm.id], name).map(|frame| (arm.id, frame)))
}

/// The refusal of the record of `link_name` in `list` that holds `error`.
fn of_record(list: &str, link_name: &str, error: &GeometryError) -> String {
    format!("{list} of {link_name}: {error}")
}

/// The position target of `c`, its pose located at `located`, an arm of it at its place in
/// the plan's arms `group`.
fn position_target(
    c: &Constraint<'_, PositiveFinite>,
    located: Located,
    group: &[ArmId],
) -> PositionTarget {
    let (reference, pose) = located.in_group(group);
    PositionTarget {
        frame: c.link,
        reference,
        position: Vector3::from(pose.position()),
        tolerance_m: c.held,
    }
}

/// The orientation target of `c`, its pose located at `located`, an arm of it at its place
/// in the plan's arms `group`.
fn orientation_target(
    c: &Constraint<'_, Turn>,
    located: Located,
    group: &[ArmId],
) -> OrientationTarget {
    let (reference, pose) = located.in_group(group);
    OrientationTarget {
        frame: c.link,
        reference,
        orientation: UnitQuaternion::from(pose.orientation()),
        tolerance_rad: c.held.tolerance_rad,
        free_axis: c.held.free_axis,
    }
}

/// The goal of the arms that goals constrain, the plan's arms `group` in the order a state
/// holds their joints: one, or every arm, or none when no arm has a goal. An arm of a robot
/// of more than one that a goal leaves out stays where it starts.
fn arm_goals<const K: usize>(
    goals: PerArm<Option<Goal>, K>,
    group: &[ArmId],
) -> Option<ArmGoals<K>> {
    match group {
        [] => None,
        [arm] => {
            let goal = goals.into_iter().flatten().next()?;
            Some(ArmGoals::One {
                arm: *arm,
                goal: Box::new(goal),
            })
        }
        _ => Some(ArmGoals::Every(Box::new(goals.transpose()?))),
    }
}

/// The robot's arm names.
fn robot_arms<const N: usize, const K: usize>(arms: &PerArm<ArmModel<N>, K>) -> Vec<&str> {
    arms.iter().map(|arm| arm.name.as_str()).collect()
}

/// Parses a plan_cartesian_path goal against `arms`, whose fingers hang as `fingers` hold
/// them, its waypoints in the robot frame `robot` for a goal that names no frame, and in a frame
/// of no link of its arm placed by `place`.
pub fn parse_cartesian<const N: usize, const K: usize>(
    raw: &plan_cartesian_path::GoalRequestData,
    arms: &PerArm<ArmModel<N>, K>,
    fingers: &PerArm<Vec<Rider>, K>,
    robot: &FrameName,
    place: &ValuePlacer<Vec<Pose>>,
) -> Result<CartesianQuery<N, K>, RequestError> {
    let id = place_of_arm("arm_name", &raw.arm_name, &robot_arms(arms)).map(ArmId::new)?;
    let arm = &arms[id];
    let mut starts = PerArm::from_fn(|_| None);
    starts[id] = start_of(arm, &raw.start_joint_positions)?;
    let frame = frame_of(arm, &fingers[id], &raw.link_name)
        .ok_or_else(|| none_has(&[arm], "link", &raw.link_name, |arm| &arm.link_names))?;
    let frame = FollowedFrame::of(arm.geometry.chain(), frame).ok_or_else(|| {
        format!(
            "{} is not fixed to {}, so a path cannot keep it on a line; send {}, or a link fixed to it",
            raw.link_name, arm.grasp_link, arm.grasp_link
        )
    })?;
    if raw.waypoints.is_empty() {
        return Err(RequestError::Rejected(
            "waypoints holds no pose: send one or more".to_string(),
        ));
    }
    let waypoints = (raw.waypoints.iter().enumerate())
        .map(|(k, w)| {
            Pose::parse(w.position, w.orientation).map_err(|e| format!("waypoint {}: {e}", k + 1))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let named = frame_or_robot(raw.frame.as_deref(), robot);
    let stamped = Stamped::parse(named, raw.timestamp, waypoints).map_err(|e| e.to_string())?;
    let isometries = |poses: Vec<Pose>| poses.into_iter().map(Isometry3::from).collect();
    let poses = match frame_of(arm, &fingers[id], named) {
        Some(link) => CartesianPoses::Link {
            frame: link,
            poses: isometries(stamped.value().clone()),
        },
        None => CartesianPoses::Robot(isometries(place(&stamped)?)),
    };
    Ok(CartesianQuery {
        arm: id,
        starts,
        frame,
        poses,
        avoid_collisions: raw.avoid_collisions,
    })
}

/// Parses a check_joint_path request against `arms`.
pub fn parse_check<const N: usize, const K: usize>(
    raw: &check_joint_path::RequestData,
    arms: &PerArm<ArmModel<N>, K>,
) -> Result<CheckQuery<N, K>, String> {
    let (order, starts) = named_arms(
        arms,
        raw.arms
            .iter()
            .map(|a| (a.arm_name.as_str(), a.start_joint_positions.as_slice())),
    )?;
    waypoint_count(
        arms,
        &order,
        &raw.arms
            .iter()
            .map(|a| a.waypoints.len())
            .collect::<Vec<_>>(),
    )?;
    let parsed = order
        .iter()
        .zip(&raw.arms)
        .map(|(arm, path)| {
            path.waypoints
                .iter()
                .enumerate()
                .map(|(k, w)| {
                    joint_positions(
                        &arms[*arm],
                        &format!("waypoint {}", k + 1),
                        &w.joint_positions,
                    )
                })
                .collect::<Result<Vec<_>, String>>()
                .map(|waypoints| (*arm, waypoints))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let waypoints = match parsed.as_slice() {
        [(arm, waypoints)] => Checked::One {
            arm: *arm,
            waypoints: waypoints.clone(),
        },
        _ => {
            let of = |id: ArmId| {
                parsed
                    .iter()
                    .find_map(|(named, waypoints)| (*named == id).then_some(waypoints))
                    .expect("every arm is named")
            };
            let count = parsed[0].1.len();
            Checked::Every(
                (0..count)
                    .map(|k| PerArm::from_fn(|id| of(id)[k]))
                    .collect(),
            )
        }
    };
    Ok(CheckQuery { starts, waypoints })
}

/// A joint target of a goal, and the arm it constrains.
struct JointConstraint {
    arm: ArmId,
    target: JointTarget,
}

/// The refusal of `error`, `arm`'s goal, whose position and orientation constraints name
/// `links`: a mix of joints and poses, or a pose of more than one of its links, says what
/// to send.
fn goal_refusal<const N: usize>(
    arm: &ArmModel<N>,
    error: GoalError,
    links: (Vec<&str>, Vec<&str>),
) -> String {
    match error {
        GoalError::Mixed => format!(
            "the goal names joints and links of {}; an arm's goal holds its joints, or the \
             position and the orientation of one of its links, not both",
            arm.name
        ),
        GoalError::Frames(..) => one_link(arm, links),
        e => e.to_string(),
    }
}

/// The refusal of a goal whose position and orientation constraints name more than one
/// link of `arm`, `positions` and `orientations`: it names the lists and the links sent,
/// and what to send.
fn one_link<const N: usize>(
    arm: &ArmModel<N>,
    (positions, orientations): (Vec<&str>, Vec<&str>),
) -> String {
    let named = positions
        .iter()
        .chain(&orientations)
        .fold(Vec::new(), |mut named, link| {
            if !named.contains(link) {
                named.push(*link);
            }
            named
        });
    let lists: Vec<&str> = [
        ("goal_position_constraints", positions.is_empty()),
        ("goal_orientation_constraints", orientations.is_empty()),
    ]
    .into_iter()
    .filter(|(_, empty)| !empty)
    .map(|(list, _)| list)
    .collect();
    let name = if lists.len() == 1 { "names" } else { "name" };
    format!(
        "{} {name} {} of {}; an arm's goal holds the position and the orientation of one of \
         its links: send both for one link, such as {}",
        lists.join(" and "),
        named.join(" and "),
        arm.name,
        arm.grasp_link
    )
}

/// Where each arm's path starts: `None` where the arm stands, for an empty
/// start_joint_positions, and for an arm the request does not name.
pub type ArmStarts<const N: usize, const K: usize> = PerArm<Option<[f64; N]>, K>;

/// The arms a request names, by each named arm's name and its start_joint_positions in
/// the request's order, each once, and where the path of each starts.
fn named_arms<'v, const N: usize, const K: usize>(
    arms: &PerArm<ArmModel<N>, K>,
    named: impl Iterator<Item = (&'v str, &'v [f64])>,
) -> Result<(Vec<ArmId>, ArmStarts<N, K>), String> {
    let (names, values): (Vec<&str>, Vec<&[f64]>) = named.unzip();
    let order: Vec<ArmId> = places_in_order(&names, &robot_arms(arms))?
        .into_iter()
        .map(ArmId::new)
        .collect();
    let starts = order.iter().zip(values).try_fold(
        PerArm::from_fn(|_| None),
        |mut starts, (arm, values)| {
            starts[*arm] = start_of(&arms[*arm], values)?;
            Ok::<_, String>(starts)
        },
    )?;
    Ok((order, starts))
}

/// Refuses `counts`, each arm's waypoints in `order`, unless the first holds one or more and
/// every other as many.
fn waypoint_count<const N: usize, const K: usize>(
    arms: &PerArm<ArmModel<N>, K>,
    order: &[ArmId],
    counts: &[usize],
) -> Result<(), String> {
    let named: Vec<(&str, usize)> = order
        .iter()
        .map(|arm| arms[*arm].name.as_str())
        .zip(counts.iter().copied())
        .collect();
    common_waypoint_count(&named).map(|_| ())
}

/// Where `arm`'s path starts: `None` for an empty list, where the arm stands.
fn start_of<const N: usize>(arm: &ArmModel<N>, values: &[f64]) -> Result<Option<[f64; N]>, String> {
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() != N {
        return Err(format!(
            "{}, or none to start where the arm stands",
            miscounted(arm, "start_joint_positions", values.len())
        ));
    }
    joint_positions(arm, "start_joint_positions", values).map(Some)
}

/// The refusal of `count` values in `what` of `arm`, which holds one for each joint.
fn miscounted<const N: usize>(arm: &ArmModel<N>, what: &str, count: usize) -> String {
    format!(
        "{what} of {} has {count} values; send one for each of its {N} joints",
        arm.name
    )
}

/// One position for each joint of `arm`, each finite.
fn joint_positions<const N: usize>(
    arm: &ArmModel<N>,
    what: &str,
    values: &[f64],
) -> Result<[f64; N], String> {
    let q: [f64; N] = values
        .try_into()
        .map_err(|_| miscounted(arm, what, values.len()))?;
    match q.iter().position(|v| !v.is_finite()) {
        None => Ok(q),
        Some(i) => Err(format!(
            "{what} of {} holds {} for {}; a position is a finite number",
            arm.name, q[i], arm.joint_names[i]
        )),
    }
}

/// Refuses a name given twice in `list`.
fn named_once<'a>(list: &str, names: impl Iterator<Item = &'a str>) -> Result<(), String> {
    let names: Vec<&str> = names.collect();
    names
        .iter()
        .enumerate()
        .find(|(k, name)| names[..*k].contains(name))
        .map_or(Ok(()), |(_, name)| {
            Err(format!("{name} is named twice in {list}"))
        })
}

/// The refusal of `name`, a `what` that no arm of `named` has: the arm's own list of its
/// `what`s for one arm, and each arm's for several.
fn none_has<const N: usize>(
    named: &[&ArmModel<N>],
    what: &str,
    name: &str,
    listed: impl Fn(&ArmModel<N>) -> &[String],
) -> String {
    match named {
        [arm] => format!(
            "{} has no {what} {name:?}; its {what}s are {}",
            arm.name,
            listed(arm).join(", ")
        ),
        _ => format!(
            "no arm of arms has a {what} {name:?}; {}",
            named
                .iter()
                .map(|arm| format!("{}'s {what}s are {}", arm.name, listed(arm).join(", ")))
                .collect::<Vec<_>>()
                .join("; ")
        ),
    }
}

/// The tolerance `field` of the constraint on `named`, refused unless a finite number above
/// 0, the message naming the field and the joint or link.
fn tolerance(field: &str, named: &str, value: f64) -> Result<PositiveFinite, String> {
    PositiveFinite::parse(value)
        .map_err(|_| format!("{field} of {named} is {value}; send a finite number above 0"))
}

/// The joint constraint `c` of a joint of an arm of `named`.
fn joint_target<const N: usize>(
    named: &[&ArmModel<N>],
    c: &GoalGoalJointConstraintsItem,
) -> Result<JointConstraint, String> {
    let (arm, joint) = named
        .iter()
        .find_map(|arm| {
            let joint = arm.joint_names.iter().position(|j| *j == c.joint_name)?;
            Some((arm, joint))
        })
        .ok_or_else(|| none_has(named, "joint", &c.joint_name, |arm| &arm.joint_names))?;
    let limit = arm.geometry.chain().limits()[joint];
    let Some(position) = limit.take_target(c.joint_position) else {
        return Err(format!(
            "the target {} of {} lies outside its limits {} to {}",
            c.joint_position, c.joint_name, limit.lo, limit.hi
        ));
    };
    Ok(JointConstraint {
        arm: arm.id,
        target: JointTarget {
            joint,
            position,
            above: tolerance("tolerance_above", &c.joint_name, c.tolerance_above)?,
            below: tolerance("tolerance_below", &c.joint_name, c.tolerance_below)?,
        },
    })
}

/// The arm of `named` that has the link `link`, and the frame of the link: the grasp
/// point; a finger of the arm's `fingers`, where it hangs off the hand; or the link of a
/// joint of the arm, or a link fixed below it.
fn frame<const N: usize, const K: usize>(
    named: &[&ArmModel<N>],
    fingers: &PerArm<Vec<Rider>, K>,
    link: &str,
) -> Result<(ArmId, Frame), String> {
    named
        .iter()
        .find_map(|arm| frame_of(arm, &fingers[arm.id], link).map(|frame| (arm.id, frame)))
        .ok_or_else(|| none_has(named, "link", link, |arm| &arm.link_names))
}

/// The frame of the link `link` of `arm`, whose fingers are `fingers`; none for a link
/// the arm does not have.
fn frame_of<const N: usize>(arm: &ArmModel<N>, fingers: &[Rider], link: &str) -> Option<Frame> {
    if link == arm.grasp_link {
        return Some(Frame::GraspPoint);
    }
    if let Some(finger) = fingers.iter().find(|finger| finger.name == link) {
        return Some(Frame::from(finger));
    }
    let chain = arm.geometry.chain();
    let tree = chain.tree();
    let posed = chain.at(&[0.0; N]);
    (0..N).find_map(|segment| {
        let carrier = posed.link_name(segment);
        let offset = if carrier == link {
            Isometry3::identity()
        } else {
            tree.fixed_path_to(tree.link_index(carrier)?, link)?
        };
        Some(Frame::Link { segment, offset })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::UNIX_EPOCH;

    use control_core::geometry::Placed;
    use frame_tree::{FrameTree, LookupError};

    use peppygen::exposed_actions::motion_planning::plan_joint_path::{
        GoalArmsItem, GoalGoalOrientationConstraintsItem, GoalGoalPositionConstraintsItem,
        GoalPathOrientationConstraintsItem,
    };
    use peppygen::exposed_services::motion_planning::check_joint_path::{
        CheckJointPathArmsItem, CheckJointPathArmsItemWaypointsItem,
    };

    use crate::robot::tests::{ARM_DOF, ARMS, LEFT, RIGHT};

    use crate::fingers::riders_of_the_arms;
    use crate::robot::tests::OPENARM_V2;

    fn arms() -> &'static PerArm<ArmModel<ARM_DOF>, ARMS> {
        &OPENARM_V2.arms
    }

    fn arm(name: &str) -> GoalArmsItem {
        GoalArmsItem {
            arm_name: name.to_string(),
            start_joint_positions: Vec::new(),
        }
    }

    /// A goal of the left arm for its grasp point, from where it stands.
    fn goal() -> GoalRequestData {
        GoalRequestData {
            arms: vec![arm("left_arm")],
            goal_joint_constraints: Vec::new(),
            goal_position_constraints: vec![GoalGoalPositionConstraintsItem {
                link_name: "openarm_left_tcp".to_string(),
                frame: Some(robot_frame()),
                timestamp: None,
                position: [0.4, 0.2, 0.5],
                tolerance_m: 0.01,
            }],
            path_orientation_constraints: Vec::new(),
            goal_orientation_constraints: vec![GoalGoalOrientationConstraintsItem {
                link_name: "openarm_left_tcp".to_string(),
                frame: Some(robot_frame()),
                timestamp: None,
                orientation: [0.0, 0.0, 0.0, 1.0005],
                tolerance_rad: 0.1,
                free_axis: "z".to_string(),
            }],
            planner: "rrt_connect".to_string(),
            refiner: "none".to_string(),
            allowed_planning_time_s: 1.0,
        }
    }

    fn joint_constraint(name: &str, position: f64) -> GoalGoalJointConstraintsItem {
        GoalGoalJointConstraintsItem {
            joint_name: name.to_string(),
            joint_position: position,
            tolerance_above: 0.01,
            tolerance_below: 0.01,
        }
    }

    fn joint_goal(name: &str, position: f64) -> GoalRequestData {
        GoalRequestData {
            goal_joint_constraints: vec![joint_constraint(name, position)],
            goal_position_constraints: Vec::new(),
            path_orientation_constraints: Vec::new(),
            goal_orientation_constraints: Vec::new(),
            ..goal()
        }
    }

    /// The robot frame of OpenArm v2, as a request names it.
    fn robot_frame() -> String {
        OPENARM_V2.frame.to_string()
    }

    /// Places a pose as a tree that holds no transform does: in the robot frame where it
    /// is, and in any other frame refused as unknown.
    fn in_an_empty_tree<T: Placed>(stamped: &Stamped<T>) -> Result<T, PlaceError> {
        let tree = FrameTree::new(Duration::from_secs(10)).expect("a window");
        let found = tree.lookup(&OPENARM_V2.frame, stamped.frame(), stamped.time())?;
        Ok(stamped.value().placed_by(&found.placement)?)
    }

    /// The message of a refusal with rejected.
    fn rejected(error: RequestError) -> String {
        match error {
            RequestError::Rejected(message) => message,
            RequestError::Unplaced(error) => panic!("a refusal with rejected, not {error}"),
        }
    }

    fn parse(goal: &GoalRequestData) -> Result<PlanQuery<ARM_DOF, ARMS>, String> {
        let fingers = riders_of_the_arms(&OPENARM_V2.own, arms()).expect("the arms' fingers");
        parse_plan(goal, arms(), &fingers, &OPENARM_V2.frame, &in_an_empty_tree).map_err(rejected)
    }

    /// A Cartesian path of the left arm's grasp point through one pose, from where it stands.
    fn cartesian_goal() -> plan_cartesian_path::GoalRequestData {
        plan_cartesian_path::GoalRequestData {
            arm_name: "left_arm".to_string(),
            start_joint_positions: Vec::new(),
            link_name: "openarm_left_tcp".to_string(),
            frame: Some(robot_frame()),
            timestamp: None,
            waypoints: vec![plan_cartesian_path::GoalWaypointsItem {
                position: [0.4, 0.2, 0.5],
                orientation: [0.0, 0.0, 0.0, 1.0005],
            }],
            avoid_collisions: true,
        }
    }

    fn parse_cartesian_goal(
        goal: &plan_cartesian_path::GoalRequestData,
    ) -> Result<CartesianQuery<ARM_DOF, ARMS>, String> {
        let fingers = riders_of_the_arms(&OPENARM_V2.own, arms()).expect("the arms' fingers");
        parse_cartesian(goal, arms(), &fingers, &OPENARM_V2.frame, &in_an_empty_tree)
            .map_err(rejected)
    }

    // A Cartesian path names its arm, where it starts, the frame it follows, and each pose
    // with its orientation normalized.
    #[test]
    fn a_cartesian_goal_parses_into_its_arm_frame_and_poses() {
        let query = parse_cartesian_goal(&cartesian_goal()).expect("a query");
        assert_eq!(query.arm, LEFT);
        assert_eq!(query.starts[LEFT], None);
        assert!(query.avoid_collisions);
        assert_eq!(
            query.frame,
            FollowedFrame::of(arms()[LEFT].geometry.chain(), Frame::GraspPoint)
                .expect("the grasp point")
        );
        let CartesianPoses::Robot(poses) = query.poses else {
            panic!("waypoints in the robot frame are placed by the tree");
        };
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].rotation.w, 1.0);
    }

    // What a Cartesian path cannot take is refused, each refusal saying what to send: an
    // arm the robot does not have, a start of the wrong length, a link of the other arm, an
    // inner link, a finger, no pose, and a pose that is no pose.
    #[test]
    fn a_cartesian_goal_it_cannot_take_is_refused_with_what_to_send() {
        let refusal = |change: &dyn Fn(&mut plan_cartesian_path::GoalRequestData)| {
            let mut goal = cartesian_goal();
            change(&mut goal);
            parse_cartesian_goal(&goal).expect_err("a refusal")
        };
        assert_eq!(
            refusal(&|goal| goal.arm_name = "third_arm".to_string()),
            "arm_name names \"third_arm\", no arm of this robot: its arms are \"left_arm\" and \
             \"right_arm\""
        );
        assert_eq!(
            refusal(&|goal| goal.start_joint_positions = vec![0.0; 6]),
            "start_joint_positions of left_arm has 6 values; send one for each of its 7 joints, \
             or none to start where the arm stands"
        );
        let other_arm = refusal(&|goal| goal.link_name = "openarm_right_tcp".to_string());
        assert!(
            other_arm.starts_with("left_arm has no link \"openarm_right_tcp\""),
            "{other_arm}"
        );
        for link in ["openarm_left_link3", "openarm_left_ee_link1"] {
            assert_eq!(
                refusal(&|goal| goal.link_name = link.to_string()),
                format!(
                    "{link} is not fixed to openarm_left_tcp, so a path cannot keep it on a line; \
                     send openarm_left_tcp, or a link fixed to it"
                )
            );
        }
        assert_eq!(
            refusal(&|goal| goal.waypoints.clear()),
            "waypoints holds no pose: send one or more"
        );
        assert!(
            refusal(&|goal| goal.waypoints[0].orientation = [0.0, 0.0, 0.0, 2.0]).starts_with(
                "waypoint 1: the orientation [0, 0, 0, 2] has a squared norm of 4, 0.01 or more \
                 from 1"
            ),
        );
        let not_finite = refusal(&|goal| goal.waypoints[0].position[0] = f64::NAN);
        assert!(
            not_finite.starts_with("waypoint 1: ") && not_finite.contains("NaN"),
            "{not_finite}"
        );
        assert_eq!(
            refusal(&|goal| goal.frame = Some(String::new())),
            "the frame is empty: name a frame of the robot's tree, such as the robot frame that \
             sensor_mounts' describe_sensors names"
        );
    }

    fn refusal(goal: GoalRequestData) -> String {
        parse(&goal).expect_err("a refusal")
    }

    // A goal for the grasp link parses into its planner and refiner, the arm it moves, its
    // start where it stands, and its time; a near-unit orientation is normalized.
    #[test]
    fn a_goal_of_the_grasp_link_parses() {
        let query = parse(&goal()).expect("a query");
        assert_eq!(query.planner, Planner::RrtConnect);
        assert_eq!(query.refiner, Refiner::None);
        assert!(matches!(
            query.arm_goals,
            ArmGoals::One { arm, .. } if arm == LEFT
        ));
        assert_eq!(query.order, [LEFT]);
        assert_eq!(query.starts, PerArm::new([None, None]));
        assert_eq!(query.time_limit, Duration::from_secs(1));
        let longer = GoalRequestData {
            allowed_planning_time_s: 2.5,
            ..goal()
        };
        let query = parse(&longer).expect("a query");
        assert_eq!(query.time_limit, Duration::from_millis(2500));
        let mut started = goal();
        started.arms[0].start_joint_positions = vec![0.1; ARM_DOF];
        assert_eq!(
            parse(&started).expect("a query").starts,
            PerArm::new([Some([0.1; ARM_DOF]), None])
        );
    }

    // Each planner and each refiner parses from the name a goal gives it, which no
    // other choice of its field shares.
    #[test]
    fn each_choice_parses_from_its_name() {
        fn each_parses<T: Named + std::fmt::Debug>() {
            for &choice in T::ALL {
                assert_eq!(T::named(choice.name()), Ok(choice));
            }
        }
        each_parses::<Planner>();
        each_parses::<Refiner>();
    }

    // Each planner names the libraries' planner of the same name, and each refiner what
    // refines the planner's path.
    #[test]
    fn each_choice_names_the_planner_and_refiner_of_the_libraries() {
        assert_eq!(
            ompl_bridge::Planner::from(Planner::RrtConnect),
            ompl_bridge::Planner::RrtConnect
        );
        assert_eq!(
            ompl_bridge::Planner::from(Planner::RrtStar),
            ompl_bridge::Planner::RrtStar
        );
        assert_eq!(
            Option::<motion_planning_core::Refiner>::from(Refiner::None),
            None
        );
        assert_eq!(
            Option::<motion_planning_core::Refiner>::from(Refiner::Chomp),
            Some(motion_planning_core::Refiner::Chomp)
        );
    }

    #[test]
    fn a_list_joins_its_last_name_by_the_conjunction() {
        assert_eq!(listed::<&str>(&[], "or"), "");
        assert_eq!(listed(&["a"], "or"), "a");
        assert_eq!(listed(&["a", "b"], "and"), "a and b");
        assert_eq!(listed(&["a", "b", "c"], "or"), "a, b or c");
    }

    /// A path orientation of `link`, within 0.1 rad of the identity, free about `free_axis`.
    fn held(link: &str, free_axis: &str) -> GoalPathOrientationConstraintsItem {
        GoalPathOrientationConstraintsItem {
            link_name: link.to_string(),
            frame: Some(robot_frame()),
            timestamp: None,
            orientation: [0.0, 0.0, 0.0, 1.0],
            tolerance_rad: 0.1,
            free_axis: free_axis.to_string(),
        }
    }

    // Each path orientation goes to the arm whose link it names, as sent; an arm that holds one
    // and has no goal stays where it starts.
    #[test]
    fn a_path_orientation_goes_to_the_arm_of_its_link() {
        let mut both = goal();
        both.arms.push(arm("right_arm"));
        both.path_orientation_constraints = vec![
            held("openarm_right_tcp", "none"),
            held("openarm_left_tcp", "z"),
        ];
        let query = parse(&both).expect("a query");
        assert!(matches!(
            query.arm_goals,
            ArmGoals::One { arm, .. } if arm == LEFT
        ));
        let [left] = query.path_orientations[LEFT].as_slice() else {
            panic!("one orientation of the left arm");
        };
        assert_eq!(left.link_name, "openarm_left_tcp");
        assert_eq!(left.target.free_axis, Some(Axis::Z));
        let [right] = query.path_orientations[RIGHT].as_slice() else {
            panic!("one orientation of the right arm");
        };
        assert_eq!(right.link_name, "openarm_right_tcp");
        assert_eq!(right.target.free_axis, None);
    }

    // A path orientation of a link outside the named arms, a link named twice, a tolerance of 0
    // and an unknown free axis are refused, and the message says what to send instead.
    #[test]
    fn a_path_orientation_it_cannot_take_is_refused_with_what_to_send() {
        let refused = |constraints: Vec<GoalPathOrientationConstraintsItem>| {
            refusal(GoalRequestData {
                path_orientation_constraints: constraints,
                ..goal()
            })
        };
        assert!(
            refused(vec![held("openarm_right_tcp", "none")]).contains("openarm_right_tcp"),
            "a link of an arm the goal does not name"
        );
        assert_eq!(
            refused(vec![
                held("openarm_left_tcp", "none"),
                held("openarm_left_tcp", "z")
            ]),
            "openarm_left_tcp is named twice in path_orientation_constraints"
        );
        assert!(
            refused(vec![GoalPathOrientationConstraintsItem {
                tolerance_rad: 0.0,
                ..held("openarm_left_tcp", "none")
            }])
            .contains("tolerance_rad")
        );
        assert_eq!(
            refused(vec![held("openarm_left_tcp", "w")]),
            "free_axis is \"none\", \"x\", \"y\" or \"z\", not \"w\""
        );
    }

    // An arm the goal names with no constraint stays where it starts: the plan moves the
    // arm the constraints name, in the goal's order of arms.
    #[test]
    fn an_arm_with_no_constraint_is_named_and_held() {
        let mut both = goal();
        both.arms = vec![arm("right_arm"), arm("left_arm")];
        both.arms[0].start_joint_positions = vec![0.2; ARM_DOF];
        let query = parse(&both).expect("a query");
        assert_eq!(query.order, [RIGHT, LEFT]);
        assert!(matches!(
            query.arm_goals,
            ArmGoals::One { arm, .. } if arm == LEFT
        ));
        assert_eq!(query.starts, PerArm::new([None, Some([0.2; ARM_DOF])]));
    }

    // Each constraint goes to the named arm whose joint or link it names, and arms may
    // take different kinds: joints for one, a pose for the other.
    #[test]
    fn each_arm_takes_the_constraints_of_its_own_joints_and_links() {
        let mut both = goal();
        both.arms.push(arm("right_arm"));
        both.goal_joint_constraints = vec![joint_constraint("openarm_right_joint4", 1.0)];
        let query = parse(&both).expect("a query");
        let ArmGoals::Every(goals) = query.arm_goals else {
            panic!("both arms move");
        };
        assert_eq!(goals[LEFT], goal_of_the_left_grasp_point());
        assert_eq!(
            goals[RIGHT],
            Goal::new(
                ARM_DOF,
                vec![JointTarget {
                    joint: 3,
                    position: 1.0,
                    above: PositiveFinite::parse(0.01).expect("a tolerance"),
                    below: PositiveFinite::parse(0.01).expect("a tolerance"),
                }],
                Vec::new(),
                Vec::new(),
            )
            .expect("a goal")
        );
    }

    // A joint target a hair under its lower limit, as a measured position sent back can lie,
    // is taken at the limit; one 1.1 mrad under it is refused.
    #[test]
    fn a_joint_target_a_hair_outside_its_limit_is_taken_at_the_limit() {
        let lower = arms()[LEFT].geometry.chain().limits()[3].lo;
        let query = parse(&joint_goal("openarm_left_joint4", lower - 0.0005)).expect("a query");
        let ArmGoals::One { goal, .. } = query.arm_goals else {
            panic!("one arm moves");
        };
        assert_eq!(
            *goal,
            Goal::new(
                ARM_DOF,
                vec![JointTarget {
                    joint: 3,
                    position: lower,
                    above: PositiveFinite::parse(0.01).expect("a tolerance"),
                    below: PositiveFinite::parse(0.01).expect("a tolerance"),
                }],
                Vec::new(),
                Vec::new(),
            )
            .expect("a goal")
        );
        let beyond = lower - 0.0011;
        assert!(
            refusal(joint_goal("openarm_left_joint4", beyond)).starts_with(&format!(
                "the target {beyond} of openarm_left_joint4 lies outside its limits"
            ))
        );
    }

    /// The left arm's goal of `goal()`.
    fn goal_of_the_left_grasp_point() -> Goal {
        let ArmGoals::One { goal, .. } = parse(&goal()).expect("a query").arm_goals else {
            panic!("one arm moves");
        };
        *goal
    }

    // A goal whose position names one link and whose orientation names another, or whose
    // two positions name two links, of one arm, is refused, naming the lists and the links
    // sent and what to send.
    #[test]
    fn a_goal_of_two_links_of_an_arm_is_refused_with_what_to_send() {
        let mut two = goal();
        two.goal_orientation_constraints[0].link_name = "openarm_left_link4".to_string();
        assert_eq!(
            refusal(two),
            "goal_position_constraints and goal_orientation_constraints name \
             openarm_left_tcp and openarm_left_link4 of left_arm; an arm's goal holds the \
             position and the orientation of one of its links: send both for one link, such \
             as openarm_left_tcp"
        );
        let mut positions = goal();
        positions.goal_orientation_constraints.clear();
        let mut elbow = positions.goal_position_constraints[0].clone();
        elbow.link_name = "openarm_left_link4".to_string();
        positions.goal_position_constraints.push(elbow);
        assert_eq!(
            refusal(positions),
            "goal_position_constraints names openarm_left_tcp and openarm_left_link4 of \
             left_arm; an arm's goal holds the position and the orientation of one of its \
             links: send both for one link, such as openarm_left_tcp"
        );
        let mut right_two = goal();
        right_two.arms.push(arm("right_arm"));
        let mut right_tcp = right_two.goal_position_constraints[0].clone();
        right_tcp.link_name = "openarm_right_tcp".to_string();
        let mut right_elbow = right_tcp.clone();
        right_elbow.link_name = "openarm_right_link4".to_string();
        right_two
            .goal_position_constraints
            .extend([right_tcp, right_elbow]);
        assert_eq!(
            refusal(right_two),
            "goal_position_constraints names openarm_right_tcp and openarm_right_link4 of \
             right_arm; an arm's goal holds the position and the orientation of one of its \
             links: send both for one link, such as openarm_right_tcp"
        );
    }

    // An arm whose constraints mix joints with a position or an orientation is refused,
    // naming the arm, also beside another arm that takes a pose.
    #[test]
    fn an_arm_whose_goal_mixes_joints_and_poses_is_refused() {
        let mut mixed = goal();
        mixed.goal_joint_constraints = vec![joint_constraint("openarm_left_joint4", 1.0)];
        let expected = "the goal names joints and links of left_arm; an arm's goal holds its \
                        joints, or the position and the orientation of one of its links, not both";
        assert_eq!(refusal(mixed), expected);
        let mut beside = goal();
        beside.arms.push(arm("right_arm"));
        beside.goal_joint_constraints = vec![joint_constraint("openarm_left_joint4", 1.0)];
        assert_eq!(refusal(beside), expected);
    }

    // A goal that names what the robot does not have, or names it twice, or a number
    // outside its range, is refused, and the message says what to send instead.
    #[test]
    fn a_goal_outside_the_robot_is_refused_with_what_to_send() {
        let mut third = goal();
        third.arms[0].arm_name = "third_arm".to_string();
        assert_eq!(
            refusal(third),
            "arms names \"third_arm\", no arm of this robot: its arms are \"left_arm\" and \
             \"right_arm\""
        );
        let mut twice = goal();
        twice.arms.push(arm("left_arm"));
        assert_eq!(
            refusal(twice),
            "arms names \"left_arm\" twice: name each arm once"
        );
        let none = GoalRequestData {
            arms: Vec::new(),
            ..goal()
        };
        assert_eq!(
            refusal(none),
            "arms holds no arm: name one or more of \"left_arm\" and \"right_arm\""
        );
        let mut short = goal();
        short.arms[0].start_joint_positions = vec![0.0; 6];
        assert_eq!(
            refusal(short),
            "start_joint_positions of left_arm has 6 values; send one for each of its 7 joints, \
             or none to start where the arm stands"
        );
        assert!(
            refusal(joint_goal("openarm_right_joint4", 1.0)).starts_with(
                "left_arm has no joint \"openarm_right_joint4\"; its joints are openarm_left_joint1"
            )
        );
        let mut both_named = joint_goal("openarm_torso_joint", 1.0);
        both_named.arms.push(arm("right_arm"));
        let reason = refusal(both_named);
        assert!(
            reason.starts_with(
                "no arm of arms has a joint \"openarm_torso_joint\"; left_arm's joints are \
                 openarm_left_joint1"
            ) && reason.contains("; right_arm's joints are openarm_right_joint1"),
            "{reason}"
        );
        assert!(
            refusal(joint_goal("openarm_left_joint4", -1.0))
                .starts_with("the target -1 of openarm_left_joint4 lies outside its limits")
        );
        let mut named_twice = joint_goal("openarm_left_joint4", 1.0);
        named_twice
            .goal_joint_constraints
            .push(named_twice.goal_joint_constraints[0].clone());
        assert_eq!(
            refusal(named_twice),
            "openarm_left_joint4 is named twice in goal_joint_constraints"
        );
        let mut loose = joint_goal("openarm_left_joint4", 1.0);
        loose.goal_joint_constraints[0].tolerance_below = 0.0;
        assert_eq!(
            refusal(loose),
            "tolerance_below of openarm_left_joint4 is 0; send a finite number above 0"
        );
        let mut other = goal();
        other.goal_orientation_constraints[0].link_name = "openarm_right_tcp".to_string();
        assert!(refusal(other).starts_with(
            "left_arm has no link \"openarm_right_tcp\"; its links are openarm_left_link1"
        ));
        let mut skewed = goal();
        skewed.goal_orientation_constraints[0].orientation = [0.0, 0.0, 0.0, 1.01];
        assert!(refusal(skewed).starts_with(
            "goal_orientation_constraints of openarm_left_tcp: the orientation [0, 0, 0, 1.01] \
             has a squared norm of"
        ));
        let mut unframed = goal();
        unframed.goal_position_constraints[0].frame = Some(String::new());
        assert_eq!(
            refusal(unframed),
            "goal_position_constraints of openarm_left_tcp: the frame is empty: name a frame of \
             the robot's tree, such as the robot frame that sensor_mounts' describe_sensors names"
        );
        let mut axis = goal();
        axis.goal_orientation_constraints[0].free_axis = "w".to_string();
        assert_eq!(
            refusal(axis),
            "free_axis is \"none\", \"x\", \"y\" or \"z\", not \"w\""
        );
        let empty = GoalRequestData {
            goal_position_constraints: Vec::new(),
            path_orientation_constraints: Vec::new(),
            goal_orientation_constraints: Vec::new(),
            ..goal()
        };
        assert_eq!(
            refusal(empty),
            "a goal names at least one joint, position or orientation"
        );
        let slow = GoalRequestData {
            allowed_planning_time_s: 61.0,
            ..goal()
        };
        assert_eq!(
            refusal(slow),
            "allowed_planning_time_s is 61; send the seconds the plan may take, more than 0 and \
             at most 60"
        );
        for (time, shown) in [(0.0, "0"), (-1.0, "-1"), (f64::NAN, "NaN")] {
            let refused = GoalRequestData {
                allowed_planning_time_s: time,
                ..goal()
            };
            assert_eq!(
                refusal(refused),
                format!(
                    "allowed_planning_time_s is {shown}; send the seconds the plan may take, more \
                     than 0 and at most 60"
                )
            );
        }
        let shortest = GoalRequestData {
            planner: "shortest".to_string(),
            ..goal()
        };
        assert_eq!(
            refusal(shortest),
            "planner is \"rrt_connect\" or \"rrt_star\", not \"shortest\""
        );
        let smooth = GoalRequestData {
            refiner: "smooth".to_string(),
            ..goal()
        };
        assert_eq!(
            refusal(smooth),
            "refiner is \"none\" or \"chomp\", not \"smooth\""
        );
    }

    fn path_of(name: &str, waypoints: Vec<Vec<f64>>) -> CheckJointPathArmsItem {
        CheckJointPathArmsItem {
            arm_name: name.to_string(),
            start_joint_positions: Vec::new(),
            waypoints: waypoints
                .into_iter()
                .map(|joint_positions| CheckJointPathArmsItemWaypointsItem { joint_positions })
                .collect(),
        }
    }

    fn path(waypoints: Vec<Vec<f64>>) -> check_joint_path::RequestData {
        check_joint_path::RequestData {
            arms: vec![path_of("right_arm", waypoints)],
        }
    }

    // A path of the arm parses into its waypoints, a waypoint outside a joint limit
    // among them; a path with no waypoint, or a waypoint that is not a position for
    // each joint, is refused.
    #[test]
    fn a_path_parses_into_its_waypoints() {
        let query = parse_check(&path(vec![vec![0.2; ARM_DOF], vec![9.0; ARM_DOF]]), arms())
            .expect("a query");
        assert_eq!(
            query.waypoints,
            Checked::One {
                arm: RIGHT,
                waypoints: vec![[0.2; ARM_DOF], [9.0; ARM_DOF]],
            }
        );
        assert_eq!(
            parse_check(&path(Vec::new()), arms()).expect_err("no waypoint"),
            "right_arm holds no waypoint: a joint path passes through one or more"
        );
        let mut nan = vec![0.0; ARM_DOF];
        nan[2] = f64::NAN;
        assert_eq!(
            parse_check(&path(vec![vec![0.0; ARM_DOF], nan]), arms()).expect_err("a NaN"),
            "waypoint 2 of right_arm holds NaN for openarm_right_joint3; a position is a finite number"
        );
    }

    // A path of both arms parses into each arm's waypoints, in the request's order; arms
    // that hold different numbers of waypoints are refused, naming them.
    #[test]
    fn a_path_of_both_arms_holds_as_many_waypoints_for_each() {
        let both = check_joint_path::RequestData {
            arms: vec![
                path_of("right_arm", vec![vec![0.2; ARM_DOF]]),
                path_of("left_arm", vec![vec![0.1; ARM_DOF]]),
            ],
        };
        let query = parse_check(&both, arms()).expect("a query");
        assert_eq!(
            query.waypoints,
            Checked::Every(vec![PerArm::new([[0.1; ARM_DOF], [0.2; ARM_DOF]])])
        );
        let uneven = check_joint_path::RequestData {
            arms: vec![
                path_of("left_arm", vec![vec![0.1; ARM_DOF]]),
                path_of("right_arm", vec![vec![0.2; ARM_DOF], vec![0.3; ARM_DOF]]),
            ],
        };
        assert_eq!(
            parse_check(&uneven, arms()).expect_err("uneven"),
            "right_arm holds 2 waypoints, where left_arm holds 1: every arm passes through as \
             many"
        );
    }

    // A constraint holds any link of the arm: the grasp point, the link of a joint, and
    // a finger where it hangs off the hand.
    #[test]
    fn a_constraint_holds_any_link_of_the_arm() {
        let frame_of = |link: &str| {
            let mut goal = goal();
            goal.goal_position_constraints[0].link_name = link.to_string();
            goal.goal_orientation_constraints.clear();
            match parse(&goal)
                .unwrap_or_else(|e| panic!("{link}: {e}"))
                .arm_goals
            {
                ArmGoals::One { goal, .. } => *goal,
                ArmGoals::Every(_) => panic!("{link}: one arm moves"),
            }
        };
        let held = |frame: Frame| {
            Goal::new(
                ARM_DOF,
                Vec::new(),
                vec![PositionTarget {
                    frame,
                    reference: Reference::Robot,
                    position: Vector3::new(0.4, 0.2, 0.5),
                    tolerance_m: PositiveFinite::parse(0.01).expect("a tolerance"),
                }],
                Vec::new(),
            )
            .expect("a goal")
        };
        assert_eq!(frame_of("openarm_left_tcp"), held(Frame::GraspPoint));
        assert_eq!(
            frame_of("openarm_left_link4"),
            held(Frame::Link {
                segment: 3,
                offset: Isometry3::identity(),
            })
        );
        let left = &arms()[LEFT];
        let fingers = OPENARM_V2
            .own
            .fingers(left.id, &left.geometry)
            .expect("the left arm's fingers");
        let finger = fingers.first().expect("a left finger");
        assert_eq!(
            frame_of(finger.name),
            held(Frame::Part {
                segment: finger.segment,
                offset: finger.offset,
            })
        );
        for link in &arms()[LEFT].link_names {
            frame_of(link);
        }
    }

    /// The frame a test's tree places 1 m along the robot frame's x, turned half a turn about
    /// z, which the arithmetic holds exactly.
    const TABLE: &str = "table";

    /// Places a pose as a tree that holds `TABLE` alone does, and refuses any frame it does
    /// not hold as unknown.
    fn in_a_tree_of_the_table<T: Placed>(stamped: &Stamped<T>) -> Result<T, PlaceError> {
        let mut tree = FrameTree::new(Duration::from_secs(10)).expect("a window");
        tree.take_static(
            &frame_tree::Publisher {
                core_node: "core".into(),
                instance_id: "init".into(),
            },
            &frame_tree::StaticMessage {
                transforms: vec![frame_tree::TransformRecord {
                    frame: robot_frame(),
                    child_frame: TABLE.into(),
                    position: [1.0, 0.0, 0.0],
                    orientation: [0.0, 0.0, 1.0, 0.0],
                }],
            },
        )
        .expect("taken");
        let found = tree.lookup(&OPENARM_V2.frame, stamped.frame(), stamped.time())?;
        Ok(stamped.value().placed_by(&found.placement)?)
    }

    /// The query of `goal` with the tree of `in_a_tree_of_the_table`, or its refusal.
    fn parse_in_the_tree(goal: &GoalRequestData) -> Result<PlanQuery<ARM_DOF, ARMS>, RequestError> {
        let fingers = riders_of_the_arms(&OPENARM_V2.own, arms()).expect("the arms' fingers");
        parse_plan(
            goal,
            arms(),
            &fingers,
            &OPENARM_V2.frame,
            &in_a_tree_of_the_table,
        )
    }

    // A constraint in a frame of no arm the plan moves is placed in the robot frame by the
    // tree: a position given in the table's frame lands where the table puts it, and an
    // orientation turns by the table's turn. A frame the tree does not hold is refused with
    // the lookup's failure.
    #[test]
    fn a_constraint_in_a_frame_of_the_tree_is_placed_in_the_robot_frame() {
        let mut on_the_table = goal();
        on_the_table.goal_position_constraints[0].frame = Some(TABLE.to_string());
        on_the_table.goal_position_constraints[0].position = [0.0, 0.5, 0.2];
        on_the_table.goal_orientation_constraints[0].frame = Some(TABLE.to_string());
        on_the_table.goal_orientation_constraints[0].orientation = [0.0, 0.0, 0.0, 1.0];
        let query = parse_in_the_tree(&on_the_table).expect("a query");
        let ArmGoals::One { goal: placed, .. } = query.arm_goals else {
            panic!("one arm moves");
        };
        let half_turn = UnitQuaternion::from(
            control_core::geometry::UnitQuaternion::parse([0.0, 0.0, 1.0, 0.0])
                .expect("a half turn about z"),
        );
        let expected = Goal::new(
            ARM_DOF,
            Vec::new(),
            vec![PositionTarget {
                frame: Frame::GraspPoint,
                reference: Reference::Robot,
                position: Vector3::new(1.0, -0.5, 0.2),
                tolerance_m: PositiveFinite::parse(0.01).expect("a tolerance"),
            }],
            vec![OrientationTarget {
                frame: Frame::GraspPoint,
                reference: Reference::Robot,
                orientation: half_turn,
                tolerance_rad: PositiveFinite::parse(0.1).expect("a tolerance"),
                free_axis: Some(Axis::Z),
            }],
        )
        .expect("a goal");
        assert_eq!(*placed, expected);

        let mut nowhere = goal();
        nowhere.goal_position_constraints[0].frame = Some("nowhere".to_string());
        assert!(matches!(
            parse_in_the_tree(&nowhere),
            Err(RequestError::Unplaced(PlaceError::Lookup(
                LookupError::UnknownFrame { .. }
            )))
        ));
    }

    // A constraint and a Cartesian path that name no frame lie in the robot frame: in a tree
    // that holds the table's frame too, each places where the same record in the robot frame
    // places.
    #[test]
    fn a_record_that_names_no_frame_lies_in_the_robot_frame() {
        let goal_of =
            |goal: &GoalRequestData| match parse_in_the_tree(goal).expect("a query").arm_goals {
                ArmGoals::One { goal, .. } => goal,
                ArmGoals::Every(_) => panic!("one arm moves"),
            };
        let mut unnamed = goal();
        unnamed.goal_position_constraints[0].frame = None;
        unnamed.goal_orientation_constraints[0].frame = None;
        assert_eq!(goal_of(&unnamed), goal_of(&goal()));

        let fingers = riders_of_the_arms(&OPENARM_V2.own, arms()).expect("the arms' fingers");
        let poses_of = |goal: &plan_cartesian_path::GoalRequestData| {
            let query = parse_cartesian(
                goal,
                arms(),
                &fingers,
                &OPENARM_V2.frame,
                &in_a_tree_of_the_table,
            )
            .expect("a query");
            match query.poses {
                CartesianPoses::Robot(poses) => poses,
                CartesianPoses::Link { .. } => panic!("the waypoints lie in no link"),
            }
        };
        let unnamed_path = plan_cartesian_path::GoalRequestData {
            frame: None,
            ..cartesian_goal()
        };
        assert_eq!(poses_of(&unnamed_path), poses_of(&cartesian_goal()));
    }

    /// A goal of both arms: the right arm's joint `joint` to `position`, and the left
    /// arm's grasp point at `position_in` in `frame`, from where they stand.
    fn left_in(frame: &str, position_in: [f64; 3]) -> GoalRequestData {
        let mut both = goal();
        both.arms.push(arm("right_arm"));
        both.goal_orientation_constraints.clear();
        both.goal_position_constraints[0].frame = Some(frame.to_string());
        both.goal_position_constraints[0].position = position_in;
        both.goal_joint_constraints = vec![joint_constraint("openarm_right_joint4", 1.0)];
        both
    }

    // A constraint in a link of an arm the plan moves stays in that link, held against it at
    // each state, the arm at its place in the plan's arms; a constraint in a link of its own
    // arm stays in that link too. Its timestamp names no time, and the tree places nothing.
    #[test]
    fn a_constraint_in_a_link_of_a_moving_arm_is_held_against_that_link() {
        let query =
            parse_in_the_tree(&left_in("openarm_right_tcp", [0.0, 0.1, 0.0])).expect("a query");
        let ArmGoals::Every(goals) = query.arm_goals else {
            panic!("both arms move");
        };
        let expected = Goal::new(
            ARM_DOF,
            Vec::new(),
            vec![PositionTarget {
                frame: Frame::GraspPoint,
                reference: Reference::Arm {
                    arm: RIGHT.place(),
                    frame: Frame::GraspPoint,
                },
                position: Vector3::new(0.0, 0.1, 0.0),
                tolerance_m: PositiveFinite::parse(0.01).expect("a tolerance"),
            }],
            Vec::new(),
        )
        .expect("a goal");
        assert_eq!(goals[LEFT], expected);

        let mut own = goal();
        own.goal_orientation_constraints.clear();
        own.goal_position_constraints[0].frame = Some("openarm_left_link4".to_string());
        own.goal_position_constraints[0].timestamp = Some(UNIX_EPOCH + Duration::from_secs(5));
        let query = parse_in_the_tree(&own).expect("a query");
        let ArmGoals::One {
            goal: in_its_link, ..
        } = query.arm_goals
        else {
            panic!("one arm moves");
        };
        let expected = Goal::new(
            ARM_DOF,
            Vec::new(),
            vec![PositionTarget {
                frame: Frame::GraspPoint,
                reference: Reference::Arm {
                    arm: 0,
                    frame: Frame::Link {
                        segment: 3,
                        offset: Isometry3::identity(),
                    },
                },
                position: Vector3::new(0.4, 0.2, 0.5),
                tolerance_m: PositiveFinite::parse(0.01).expect("a tolerance"),
            }],
            Vec::new(),
        )
        .expect("a goal");
        assert_eq!(*in_its_link, expected);
    }

    // A link of an arm the plan leaves where it starts is placed by the tree, and a path
    // orientation of such an arm in a frame of an arm the plan moves is held in that frame, at
    // the moving arm's place in the plan.
    #[test]
    fn a_link_of_an_arm_that_stays_is_placed_by_the_tree_or_held_in_a_moving_frame() {
        let mut beside = goal();
        beside.arms.push(arm("right_arm"));
        beside.goal_orientation_constraints.clear();
        beside.goal_position_constraints[0].frame = Some("openarm_right_tcp".to_string());
        assert!(matches!(
            parse_in_the_tree(&beside),
            Err(RequestError::Unplaced(PlaceError::Lookup(
                LookupError::UnknownFrame { .. }
            )))
        ));
        let mut standing_held = goal();
        standing_held.arms.push(arm("right_arm"));
        standing_held.path_orientation_constraints = vec![GoalPathOrientationConstraintsItem {
            frame: Some("openarm_left_tcp".to_string()),
            ..held("openarm_right_tcp", "none")
        }];
        let query = parse_in_the_tree(&standing_held).expect("a query");
        assert!(matches!(query.arm_goals, ArmGoals::One { arm, .. } if arm == LEFT));
        let [held_in_the_hand] = query.path_orientations[RIGHT].as_slice() else {
            panic!("the right arm holds one orientation");
        };
        assert_eq!(
            held_in_the_hand.target.reference,
            Reference::Arm {
                arm: 0,
                frame: Frame::GraspPoint
            },
            "the standing arm's orientation is held in the moving hand's frame"
        );
    }

    // Waypoints in a link of the path's arm are placed where that link stands at the path's
    // start; in any other frame the tree places them.
    #[test]
    fn waypoints_in_a_link_of_the_arm_are_placed_where_it_stands_at_the_start() {
        let mut in_the_hand = cartesian_goal();
        in_the_hand.frame = Some("openarm_left_tcp".to_string());
        in_the_hand.waypoints[0].position = [0.0, 0.0, 0.05];
        in_the_hand.waypoints[0].orientation = [0.0, 0.0, 0.0, 1.0];
        let query = parse_cartesian_goal(&in_the_hand).expect("a query");
        let chain = arms()[LEFT].geometry.chain();
        let start = [0.2; ARM_DOF];
        let hand = chain.root_ee_pose(&start);
        let [placed] = query.poses.at_start(chain, &start)[..] else {
            panic!("one pose");
        };
        let expected = hand * Isometry3::translation(0.0, 0.0, 0.05);
        assert!(
            (placed.translation.vector - expected.translation.vector).norm() < 1e-12
                && placed.rotation.angle_to(&expected.rotation) < 1e-12,
            "{placed} against {expected}"
        );
        let mut on_the_table = cartesian_goal();
        on_the_table.frame = Some(TABLE.to_string());
        let fingers = riders_of_the_arms(&OPENARM_V2.own, arms()).expect("the arms' fingers");
        let query = parse_cartesian(
            &on_the_table,
            arms(),
            &fingers,
            &OPENARM_V2.frame,
            &in_a_tree_of_the_table,
        )
        .expect("a query");
        assert!(matches!(query.poses, CartesianPoses::Robot(_)));
        let mut nowhere = cartesian_goal();
        nowhere.frame = Some("nowhere".to_string());
        assert!(matches!(
            parse_cartesian(
                &nowhere,
                arms(),
                &fingers,
                &OPENARM_V2.frame,
                &in_a_tree_of_the_table
            ),
            Err(RequestError::Unplaced(PlaceError::Lookup(
                LookupError::UnknownFrame { .. }
            )))
        ));
    }
}
