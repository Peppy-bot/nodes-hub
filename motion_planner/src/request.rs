//! The requests of plan_joint_path, plan_cartesian_path and check_joint_path, parsed
//! against the robot's arms: the arms, where each one's path starts, and each arm's goal,
//! its poses or its waypoints, in the terms of motion_planning_core. A request that does
//! not parse is refused with `rejected`, and the message says what to send instead.

use std::borrow::Borrow;
use std::time::Duration;

use chain_kinematics::nalgebra::{Isometry3, Vector3};
use collision_geometry::chain::Rider;
use control_core::positive_finite::PositiveFinite;
use motion_planning_core::{
    Axis, FollowedFrame, Frame, Goal, GoalError, JointTarget, OrientationTarget, PositionTarget,
    orientation_from_parts, pose_from_parts, poses_in_range,
};
use openarm_description::{common_waypoint_count, no_such_arm, places_in_order};
use peppygen::exposed_actions::motion_planning::plan_cartesian_path;
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    GoalGoalJointConstraintsItem, GoalGoalPositionConstraintsItem, GoalRequestData,
};
use peppygen::exposed_services::motion_planning::check_joint_path;

use crate::arm_set::{ArmId, PerArm};
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

/// A plan_joint_path goal that parses.
#[derive(Debug)]
pub struct PlanQuery<const N: usize> {
    /// The planner that plans the path.
    pub planner: Planner,
    /// What refines the planner's path.
    pub refiner: Refiner,
    /// The goal's arms, in its order: the order of the answer.
    pub order: Vec<ArmId>,
    /// Where each arm's path starts.
    pub starts: ArmStarts<N>,
    /// The arms the plan moves, each to its goal; every other arm of `order` stays where
    /// it starts.
    pub arm_goals: ArmGoals,
    /// The orientations each arm's path holds.
    pub path_orientations: PerArm<Vec<HeldOrientation>>,
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
pub enum ArmGoals {
    One { arm: ArmId, goal: Box<Goal> },
    Every(Box<PerArm<Goal>>),
}

/// A check_joint_path request that parses.
#[derive(Debug)]
pub struct CheckQuery<const N: usize> {
    /// Where each arm's path starts.
    pub starts: ArmStarts<N>,
    /// The waypoints after the start of the arms the request names, at least one.
    pub waypoints: Checked<N>,
}

/// A plan_cartesian_path goal that parses.
#[derive(Debug)]
pub struct CartesianQuery<const N: usize> {
    /// The arm the path moves.
    pub arm: ArmId,
    /// Where its path starts.
    pub starts: ArmStarts<N>,
    /// What follows the lines.
    pub frame: FollowedFrame,
    /// The poses it passes through.
    pub poses: Vec<Isometry3<f64>>,
    pub avoid_collisions: bool,
}

/// The waypoints of a check: one arm's, or every arm's of a robot of more than one, as many
/// for each.
#[derive(Debug, PartialEq)]
pub enum Checked<const N: usize> {
    One {
        arm: ArmId,
        waypoints: Vec<[f64; N]>,
    },
    Every(Vec<PerArm<[f64; N]>>),
}

impl<const N: usize> Checked<N> {
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

/// Parses a plan_joint_path goal against `arms`, whose fingers hang as `fingers` hold
/// them.
pub fn parse_plan<const N: usize>(
    raw: &GoalRequestData,
    arms: &PerArm<ArmModel<N>>,
    fingers: &PerArm<Vec<Rider>>,
) -> Result<PlanQuery<N>, String> {
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
    let positions = raw
        .goal_position_constraints
        .iter()
        .map(|c| position_target(&named, fingers, c))
        .collect::<Result<Vec<_>, String>>()?;
    let target_of = |link_name, orientation, tolerance_rad, free_axis| {
        orientation_target(
            &named,
            fingers,
            link_name,
            orientation,
            tolerance_rad,
            free_axis,
        )
    };
    let orientations = (raw.goal_orientation_constraints.iter())
        .map(|c| target_of(&c.link_name, c.orientation, c.tolerance_rad, &c.free_axis))
        .collect::<Result<Vec<_>, String>>()?;
    let path_orientations = (raw.path_orientation_constraints.iter())
        .map(|c| target_of(&c.link_name, c.orientation, c.tolerance_rad, &c.free_axis))
        .collect::<Result<Vec<_>, String>>()?;
    let goals = arms
        .each(|arm, model| {
            let (joints, positions, orientations) = (
                of_the_arm(&joints, arm),
                of_the_arm(&positions, arm),
                of_the_arm(&orientations, arm),
            );
            let constrained =
                !(joints.is_empty() && positions.is_empty() && orientations.is_empty());
            constrained
                .then(|| {
                    let links = (names(&positions), names(&orientations));
                    Goal::new(
                        N,
                        targets(joints),
                        targets(positions),
                        targets(orientations),
                    )
                    .map_err(|e| goal_refusal(model, e, links))
                })
                .transpose()
        })
        .transpose()?;
    let arm_goals = arm_goals(goals).ok_or_else(|| GoalError::Empty.to_string())?;
    let planner = Planner::named(&raw.planner)?;
    let refiner = Refiner::named(&raw.refiner)?;
    let time = raw.allowed_planning_time_s;
    if !(time > 0.0 && time <= MAX_PLANNING_TIME_S) {
        return Err(format!(
            "allowed_planning_time_s is {time}; send the seconds the plan may take, more than 0 \
             and at most {MAX_PLANNING_TIME_S}"
        ));
    }
    Ok(PlanQuery {
        planner,
        refiner,
        order,
        starts,
        arm_goals,
        path_orientations: arms.each(|arm, _| {
            of_the_arm(&path_orientations, arm)
                .into_iter()
                .map(|routed| HeldOrientation {
                    link_name: routed.name.to_string(),
                    target: routed.target,
                })
                .collect()
        }),
        time_limit: Duration::from_secs_f64(time),
    })
}

/// The goal of the arms that goals constrain: one, or every arm, or none when no arm has a
/// goal. An arm of a robot of more than one that a goal leaves out stays where it starts.
fn arm_goals(goals: PerArm<Option<Goal>>) -> Option<ArmGoals> {
    let named: Vec<ArmId> = goals.ids().filter(|arm| goals[*arm].is_some()).collect();
    match named[..] {
        [] => None,
        [arm] => {
            let goal = goals.into_iter().flatten().next()?;
            Some(ArmGoals::One {
                arm,
                goal: Box::new(goal),
            })
        }
        _ => Some(ArmGoals::Every(Box::new(goals.transpose()?))),
    }
}

/// The robot's arm names.
fn robot_arms<const N: usize>(arms: &PerArm<ArmModel<N>>) -> Vec<&'static str> {
    arms.iter().map(|arm| arm.name).collect()
}

/// Parses a plan_cartesian_path goal against `arms`, whose fingers hang as `fingers` hold
/// them.
pub fn parse_cartesian<const N: usize>(
    raw: &plan_cartesian_path::GoalRequestData,
    arms: &PerArm<ArmModel<N>>,
    fingers: &PerArm<Vec<Rider>>,
) -> Result<CartesianQuery<N>, String> {
    let id = arms
        .ids()
        .find(|arm| arms[*arm].name == raw.arm_name)
        .ok_or_else(|| no_such_arm("arm_name", &raw.arm_name, &robot_arms(arms)))?;
    let arm = &arms[id];
    let mut starts = arms.each(|_, _| None);
    starts[id] = start_of(arm, &raw.start_joint_positions)?;
    let frame = frame_of(arm, &fingers[id], &raw.link_name)
        .ok_or_else(|| none_has(&[arm], "link", &raw.link_name, |arm| &arm.link_names))?;
    let frame = FollowedFrame::of(arm.geometry.chain(), frame).ok_or_else(|| {
        format!(
            "{} is not fixed to {}, so a path cannot keep it on a line; send {}, or a link fixed to it",
            raw.link_name, arm.grasp_link, arm.grasp_link
        )
    })?;
    let poses = raw
        .waypoints
        .iter()
        .enumerate()
        .map(|(k, w)| {
            pose_from_parts(w.position, w.orientation)
                .map_err(|e| format!("waypoint {}: {e}", k + 1))
        })
        .collect::<Result<Vec<_>, String>>()?;
    poses_in_range(&poses)?;
    Ok(CartesianQuery {
        arm: id,
        starts,
        frame,
        poses,
        avoid_collisions: raw.avoid_collisions,
    })
}

/// Parses a check_joint_path request against `arms`.
pub fn parse_check<const N: usize>(
    raw: &check_joint_path::RequestData,
    arms: &PerArm<ArmModel<N>>,
) -> Result<CheckQuery<N>, String> {
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
            Checked::Every((0..count).map(|k| arms.each(|id, _| of(id)[k])).collect())
        }
    };
    Ok(CheckQuery { starts, waypoints })
}

/// A target of a goal, the joint or link it names as sent, and the arm it constrains.
#[derive(Clone, Copy)]
struct Routed<'r, T> {
    arm: ArmId,
    name: &'r str,
    target: T,
}

/// The targets of `routed` that constrain `arm`.
fn of_the_arm<'r, T: Copy>(routed: &[Routed<'r, T>], arm: ArmId) -> Vec<Routed<'r, T>> {
    routed.iter().filter(|r| r.arm == arm).copied().collect()
}

/// The names of `routed`, as sent.
fn names<'r, T>(routed: &[Routed<'r, T>]) -> Vec<&'r str> {
    routed.iter().map(|r| r.name).collect()
}

/// The targets of `routed`.
fn targets<T>(routed: Vec<Routed<'_, T>>) -> Vec<T> {
    routed.into_iter().map(|r| r.target).collect()
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
pub type ArmStarts<const N: usize> = PerArm<Option<[f64; N]>>;

/// The arms a request names, by each named arm's name and its start_joint_positions in
/// the request's order, each once, and where the path of each starts.
fn named_arms<'v, const N: usize>(
    arms: &PerArm<ArmModel<N>>,
    named: impl Iterator<Item = (&'v str, &'v [f64])>,
) -> Result<(Vec<ArmId>, ArmStarts<N>), String> {
    let (names, values): (Vec<&str>, Vec<&[f64]>) = named.unzip();
    let order: Vec<ArmId> = places_in_order(&names, &robot_arms(arms))?
        .into_iter()
        .map(ArmId::new)
        .collect();
    let starts = order.iter().zip(values).try_fold(
        arms.each(|_, _| None),
        |mut starts, (arm, values)| {
            starts[*arm] = start_of(&arms[*arm], values)?;
            Ok::<_, String>(starts)
        },
    )?;
    Ok((order, starts))
}

/// Refuses `counts`, each arm's waypoints in `order`, unless the first holds one or more and
/// every other as many.
fn waypoint_count<const N: usize>(
    arms: &PerArm<ArmModel<N>>,
    order: &[ArmId],
    counts: &[usize],
) -> Result<(), String> {
    let named: Vec<(&str, usize)> = order
        .iter()
        .map(|arm| arms[*arm].name)
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

fn joint_target<'r, const N: usize>(
    named: &[&ArmModel<N>],
    c: &'r GoalGoalJointConstraintsItem,
) -> Result<Routed<'r, JointTarget>, String> {
    let (arm, joint) = named
        .iter()
        .find_map(|arm| {
            let joint = arm.joint_names.iter().position(|j| *j == c.joint_name)?;
            Some((arm, joint))
        })
        .ok_or_else(|| none_has(named, "joint", &c.joint_name, |arm| &arm.joint_names))?;
    let limit = arm.geometry.chain().limits()[joint];
    let Some(position) = limit.take_target(c.position) else {
        return Err(format!(
            "the target {} of {} lies outside its limits {} to {}",
            c.position, c.joint_name, limit.lo, limit.hi
        ));
    };
    Ok(Routed {
        arm: arm.id,
        name: &c.joint_name,
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
fn frame<const N: usize>(
    named: &[&ArmModel<N>],
    fingers: &PerArm<Vec<Rider>>,
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

fn position_target<'r, const N: usize>(
    named: &[&ArmModel<N>],
    fingers: &PerArm<Vec<Rider>>,
    c: &'r GoalGoalPositionConstraintsItem,
) -> Result<Routed<'r, PositionTarget>, String> {
    let (arm, frame) = frame(named, fingers, &c.link_name)?;
    if !c.position.iter().all(|v| v.is_finite()) {
        return Err(format!(
            "the position {:?} of {} is not finite; send three finite numbers (m)",
            c.position, c.link_name
        ));
    }
    Ok(Routed {
        arm,
        name: &c.link_name,
        target: PositionTarget {
            frame,
            position: Vector3::from(c.position),
            tolerance_m: tolerance("tolerance_m", &c.link_name, c.tolerance_m)?,
        },
    })
}

fn orientation_target<'r, const N: usize>(
    named: &[&ArmModel<N>],
    fingers: &PerArm<Vec<Rider>>,
    link_name: &'r str,
    orientation: [f64; 4],
    tolerance_rad: f64,
    free_axis: &str,
) -> Result<Routed<'r, OrientationTarget>, String> {
    let (arm, frame) = frame(named, fingers, link_name)?;
    let orientation = orientation_from_parts(orientation)?;
    let free_axis = Axis::free_named(free_axis)?;
    Ok(Routed {
        arm,
        name: link_name,
        target: OrientationTarget {
            frame,
            orientation,
            tolerance_rad: tolerance("tolerance_rad", link_name, tolerance_rad)?,
            free_axis,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use peppygen::exposed_actions::motion_planning::plan_joint_path::{
        GoalArmsItem, GoalGoalOrientationConstraintsItem, GoalPathOrientationConstraintsItem,
    };
    use peppygen::exposed_services::motion_planning::check_joint_path::{
        CheckJointPathArmsItem, CheckJointPathArmsItemWaypointsItem,
    };

    use openarm_description::{ARM_DOF, Side};

    use crate::fingers::riders_of_the_arms;
    use crate::robot::tests::OPENARM_V2;

    fn arms() -> &'static PerArm<ArmModel<ARM_DOF>> {
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
                position: [0.4, 0.2, 0.5],
                tolerance_m: 0.01,
            }],
            path_orientation_constraints: Vec::new(),
            goal_orientation_constraints: vec![GoalGoalOrientationConstraintsItem {
                link_name: "openarm_left_tcp".to_string(),
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
            position,
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

    fn parse(goal: &GoalRequestData) -> Result<PlanQuery<ARM_DOF>, String> {
        let fingers = riders_of_the_arms(&OPENARM_V2.own, arms()).expect("the arms' fingers");
        parse_plan(goal, arms(), &fingers)
    }

    /// A Cartesian path of the left arm's grasp point through one pose, from where it stands.
    fn cartesian_goal() -> plan_cartesian_path::GoalRequestData {
        plan_cartesian_path::GoalRequestData {
            arm_name: "left_arm".to_string(),
            start_joint_positions: Vec::new(),
            link_name: "openarm_left_tcp".to_string(),
            waypoints: vec![plan_cartesian_path::GoalWaypointsItem {
                position: [0.4, 0.2, 0.5],
                orientation: [0.0, 0.0, 0.0, 1.0005],
            }],
            avoid_collisions: true,
        }
    }

    fn parse_cartesian_goal(
        goal: &plan_cartesian_path::GoalRequestData,
    ) -> Result<CartesianQuery<ARM_DOF>, String> {
        let fingers = riders_of_the_arms(&OPENARM_V2.own, arms()).expect("the arms' fingers");
        parse_cartesian(goal, arms(), &fingers)
    }

    // A Cartesian path names its arm, where it starts, the frame it follows, and each pose
    // with its orientation normalized.
    #[test]
    fn a_cartesian_goal_parses_into_its_arm_frame_and_poses() {
        let query = parse_cartesian_goal(&cartesian_goal()).expect("a query");
        assert_eq!(query.arm, Side::Left.into());
        assert_eq!(query.starts[Side::Left], None);
        assert!(query.avoid_collisions);
        assert_eq!(
            query.frame,
            FollowedFrame::of(arms()[Side::Left].geometry.chain(), Frame::GraspPoint)
                .expect("the grasp point")
        );
        assert_eq!(query.poses.len(), 1);
        assert_eq!(query.poses[0].rotation.w, 1.0);
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
            refusal(&|goal| goal.waypoints[0].orientation = [0.0, 0.0, 0.0, 2.0])
                .starts_with("waypoint 1: an orientation is a unit quaternion"),
        );
        assert!(
            refusal(&|goal| goal.waypoints[0].position[0] = f64::NAN)
                .starts_with("waypoint 1: a position is finite"),
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
            ArmGoals::One { arm, .. } if arm == Side::Left.into()
        ));
        assert_eq!(query.order, [ArmId::from(Side::Left)]);
        assert_eq!(query.starts, PerArm::both(None, None));
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
            PerArm::both(Some([0.1; ARM_DOF]), None)
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
            ArmGoals::One { arm, .. } if arm == Side::Left.into()
        ));
        let [left] = query.path_orientations[Side::Left].as_slice() else {
            panic!("one orientation of the left arm");
        };
        assert_eq!(left.link_name, "openarm_left_tcp");
        assert_eq!(left.target.free_axis, Some(Axis::Z));
        let [right] = query.path_orientations[Side::Right].as_slice() else {
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
        assert_eq!(query.order, [ArmId::from(Side::Right), Side::Left.into()]);
        assert!(matches!(
            query.arm_goals,
            ArmGoals::One { arm, .. } if arm == Side::Left.into()
        ));
        assert_eq!(query.starts, PerArm::both(None, Some([0.2; ARM_DOF])));
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
        assert_eq!(goals[Side::Left], goal_of_the_left_grasp_point());
        assert_eq!(
            goals[Side::Right],
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

    // A joint target a hair under its floor, as a measured position sent back can lie, is
    // taken at its floor; one 1.1 mrad under it is refused.
    #[test]
    fn a_joint_target_a_hair_outside_its_limit_is_taken_at_the_limit() {
        let floor = arms()[Side::Left].geometry.chain().limits()[3].lo;
        assert_eq!(floor, 0.05, "openarm_left_joint4's floor");
        let query = parse(&joint_goal("openarm_left_joint4", 0.0495)).expect("a query");
        let ArmGoals::One { goal, .. } = query.arm_goals else {
            panic!("one arm moves");
        };
        assert_eq!(
            *goal,
            Goal::new(
                ARM_DOF,
                vec![JointTarget {
                    joint: 3,
                    position: floor,
                    above: PositiveFinite::parse(0.01).expect("a tolerance"),
                    below: PositiveFinite::parse(0.01).expect("a tolerance"),
                }],
                Vec::new(),
                Vec::new(),
            )
            .expect("a goal")
        );
        assert!(
            refusal(joint_goal("openarm_left_joint4", 0.0489))
                .starts_with("the target 0.0489 of openarm_left_joint4 lies outside its limits")
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
        assert!(refusal(skewed).starts_with("an orientation is a unit quaternion"));
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
                arm: Side::Right.into(),
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
            Checked::Every(vec![PerArm::both([0.1; ARM_DOF], [0.2; ARM_DOF])])
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
        let left = &arms()[Side::Left];
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
        for link in &arms()[Side::Left].link_names {
            frame_of(link);
        }
    }
}
