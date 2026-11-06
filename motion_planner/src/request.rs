//! The requests of plan_joint_path and check_joint_path, parsed against the robot's arms:
//! the arms, where each one's path starts, and each arm's goal or its waypoints, in the
//! terms of motion_planning_core. A request that does not parse is refused with
//! `rejected`, and the message says what to send instead.

use std::time::Duration;

use chain_kinematics::nalgebra::{Isometry3, Vector3};
use collision_geometry::chain::Rider;
use control_core::positive_finite::PositiveFinite;
use motion_planning_core::{
    Axis, Frame, Goal, GoalError, JointTarget, OrientationTarget, PositionTarget,
    orientation_from_parts,
};
use openarm_description::{ARM_DOF, ArmOrder, ArmPair, Side};
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    GoalGoalJointConstraintsItem, GoalGoalOrientationConstraintsItem,
    GoalGoalPositionConstraintsItem, GoalRequestData,
};
use peppygen::exposed_services::motion_planning::check_joint_path;

use crate::robot::ArmModel;

/// The longest a plan may plan, in seconds.
pub const MAX_PLANNING_TIME_S: f64 = 60.0;

/// The one objective the planner serves: the first valid path RRTConnect finds,
/// simplified in the time left.
pub const OBJECTIVE: &str = "any";

/// A plan_joint_path goal that parses.
#[derive(Debug)]
pub struct PlanQuery {
    /// The goal's arms, in its order: the order of the answer.
    pub order: ArmOrder,
    /// Where each arm's path starts.
    pub starts: ArmStarts,
    /// The arms the plan moves, each to its goal; every other arm of `order` stays where
    /// it starts.
    pub arm_goals: ArmGoals,
    /// How long the plan may take.
    pub time_limit: Duration,
}

/// The arms a plan moves, each with its goal.
#[derive(Debug)]
pub enum ArmGoals {
    One { side: Side, goal: Box<Goal> },
    Both(Box<ArmPair<Goal>>),
}

/// A check_joint_path request that parses.
#[derive(Debug)]
pub struct CheckQuery {
    /// Where each arm's path starts.
    pub starts: ArmStarts,
    /// The waypoints after the start of the arms the request names, at least one.
    pub waypoints: Checked,
}

/// The waypoints of a check: one arm's, or both arms' together, as many for each.
#[derive(Debug, PartialEq)]
pub enum Checked {
    One {
        side: Side,
        waypoints: Vec<[f64; ARM_DOF]>,
    },
    Both(Vec<ArmPair<[f64; ARM_DOF]>>),
}

impl Checked {
    /// The waypoints of the arm on `side`; none for an arm the check does not name.
    pub fn of(&self, side: Side) -> Option<Vec<[f64; ARM_DOF]>> {
        match self {
            Self::One {
                side: named,
                waypoints,
            } => (*named == side).then(|| waypoints.clone()),
            Self::Both(waypoints) => Some(waypoints.iter().map(|pair| pair[side]).collect()),
        }
    }
}

/// Parses a plan_joint_path goal against `arms`, whose fingers hang as `fingers` hold
/// them.
pub fn parse_plan(
    raw: &GoalRequestData,
    arms: &ArmPair<ArmModel>,
    fingers: &ArmPair<Vec<Rider>>,
) -> Result<PlanQuery, String> {
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
    let named: Vec<&ArmModel> = order.sides().iter().map(|side| &arms[*side]).collect();
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
    let orientations = raw
        .goal_orientation_constraints
        .iter()
        .map(|c| orientation_target(&named, fingers, c))
        .collect::<Result<Vec<_>, String>>()?;
    let goals = ArmPair::from_fn(|side| {
        let (joints, positions, orientations) = (
            of_the_arm(&joints, side),
            of_the_arm(&positions, side),
            of_the_arm(&orientations, side),
        );
        let constrained = !(joints.is_empty() && positions.is_empty() && orientations.is_empty());
        constrained.then(|| {
            let links = (names(&positions), names(&orientations));
            Goal::new(
                ARM_DOF,
                targets(joints),
                targets(positions),
                targets(orientations),
            )
            .map_err(|e| goal_refusal(&arms[side], e, links))
        })
    });
    let arm_goals = match (goals.left.transpose()?, goals.right.transpose()?) {
        (None, None) => return Err(GoalError::Empty.to_string()),
        (Some(goal), None) => ArmGoals::One {
            side: Side::Left,
            goal: Box::new(goal),
        },
        (None, Some(goal)) => ArmGoals::One {
            side: Side::Right,
            goal: Box::new(goal),
        },
        (Some(left), Some(right)) => ArmGoals::Both(Box::new(ArmPair::new(left, right))),
    };
    if raw.objective != OBJECTIVE {
        return Err(format!(
            "objective is {OBJECTIVE:?}, not {:?}",
            raw.objective
        ));
    }
    let time = raw.allowed_planning_time_s;
    if !(time > 0.0 && time <= MAX_PLANNING_TIME_S) {
        return Err(format!(
            "allowed_planning_time_s is {time}; send the seconds the plan may take, more than 0 \
             and at most {MAX_PLANNING_TIME_S}"
        ));
    }
    Ok(PlanQuery {
        order,
        starts,
        arm_goals,
        time_limit: Duration::from_secs_f64(time),
    })
}

/// Parses a check_joint_path request against `arms`.
pub fn parse_check(
    raw: &check_joint_path::RequestData,
    arms: &ArmPair<ArmModel>,
) -> Result<CheckQuery, String> {
    let (order, starts) = named_arms(
        arms,
        raw.arms
            .iter()
            .map(|a| (a.arm_name.as_str(), a.start_joint_positions.as_slice())),
    )?;
    order.waypoint_count(
        &raw.arms
            .iter()
            .map(|a| a.waypoints.len())
            .collect::<Vec<_>>(),
    )?;
    let parsed = order
        .sides()
        .iter()
        .zip(&raw.arms)
        .map(|(side, path)| {
            path.waypoints
                .iter()
                .enumerate()
                .map(|(k, w)| {
                    joint_positions(
                        &arms[*side],
                        &format!("waypoint {}", k + 1),
                        &w.joint_positions,
                    )
                })
                .collect::<Result<Vec<_>, String>>()
                .map(|waypoints| (*side, waypoints))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let waypoints = match parsed.as_slice() {
        [(side, waypoints)] => Checked::One {
            side: *side,
            waypoints: waypoints.clone(),
        },
        _ => {
            let of = |side: Side| {
                parsed
                    .iter()
                    .find_map(|(named, waypoints)| (*named == side).then_some(waypoints))
                    .expect("both arms are named")
            };
            Checked::Both(
                of(Side::Left)
                    .iter()
                    .zip(of(Side::Right))
                    .map(|(left, right)| ArmPair::new(*left, *right))
                    .collect(),
            )
        }
    };
    Ok(CheckQuery { starts, waypoints })
}

/// A target of a goal, the joint or link it names as sent, and the arm it constrains.
#[derive(Clone, Copy)]
struct Routed<'r, T> {
    side: Side,
    name: &'r str,
    target: T,
}

/// The targets of `routed` that constrain the arm on `side`.
fn of_the_arm<'r, T: Copy>(routed: &[Routed<'r, T>], side: Side) -> Vec<Routed<'r, T>> {
    routed.iter().filter(|r| r.side == side).copied().collect()
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
fn goal_refusal(arm: &ArmModel, error: GoalError, links: (Vec<&str>, Vec<&str>)) -> String {
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
fn one_link(arm: &ArmModel, (positions, orientations): (Vec<&str>, Vec<&str>)) -> String {
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
pub type ArmStarts = ArmPair<Option<[f64; ARM_DOF]>>;

/// The arms a request names, by each named arm's name and its start_joint_positions in
/// the request's order, and where the path of each starts.
fn named_arms<'v>(
    arms: &ArmPair<ArmModel>,
    named: impl Iterator<Item = (&'v str, &'v [f64])>,
) -> Result<(ArmOrder, ArmStarts), String> {
    let (names, values): (Vec<&str>, Vec<&[f64]>) = named.unzip();
    let order = ArmOrder::parse(&names)?;
    let starts = order.sides().iter().zip(values).try_fold(
        ArmPair::new(None, None),
        |mut starts, (side, values)| {
            starts[*side] = start_of(&arms[*side], values)?;
            Ok::<_, String>(starts)
        },
    )?;
    Ok((order, starts))
}

/// Where `arm`'s path starts: `None` for an empty list, where the arm stands.
fn start_of(arm: &ArmModel, values: &[f64]) -> Result<Option<[f64; ARM_DOF]>, String> {
    match values.len() {
        0 => Ok(None),
        ARM_DOF => joint_positions(arm, "start_joint_positions", values).map(Some),
        count => Err(format!(
            "{}, or none to start where the arm stands",
            miscounted(arm, "start_joint_positions", count)
        )),
    }
}

/// The refusal of `count` values in `what` of `arm`, which holds one for each joint.
fn miscounted(arm: &ArmModel, what: &str, count: usize) -> String {
    format!(
        "{what} of {} has {count} values; send one for each of its {ARM_DOF} joints",
        arm.name
    )
}

/// One position for each joint of `arm`, each finite.
fn joint_positions(arm: &ArmModel, what: &str, values: &[f64]) -> Result<[f64; ARM_DOF], String> {
    let q: [f64; ARM_DOF] = values
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
fn none_has(
    named: &[&ArmModel],
    what: &str,
    name: &str,
    listed: impl Fn(&ArmModel) -> &[String],
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

fn joint_target<'r>(
    named: &[&ArmModel],
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
    if !(limit.lo..=limit.hi).contains(&c.position) {
        return Err(format!(
            "the target {} of {} lies outside its limits {} to {}",
            c.position, c.joint_name, limit.lo, limit.hi
        ));
    }
    Ok(Routed {
        side: arm.side,
        name: &c.joint_name,
        target: JointTarget {
            joint,
            position: c.position,
            above: tolerance("tolerance_above", &c.joint_name, c.tolerance_above)?,
            below: tolerance("tolerance_below", &c.joint_name, c.tolerance_below)?,
        },
    })
}

/// The arm of `named` that has the link `link`, and the frame of the link: the grasp
/// point; a finger of the arm's `fingers`, where it hangs off the hand; or the link of a
/// joint of the arm, or a link fixed below it.
fn frame(
    named: &[&ArmModel],
    fingers: &ArmPair<Vec<Rider>>,
    link: &str,
) -> Result<(Side, Frame), String> {
    named
        .iter()
        .find_map(|arm| frame_of(arm, &fingers[arm.side], link).map(|frame| (arm.side, frame)))
        .ok_or_else(|| none_has(named, "link", link, |arm| &arm.link_names))
}

/// The frame of the link `link` of `arm`, whose fingers are `fingers`; none for a link
/// the arm does not have.
fn frame_of(arm: &ArmModel, fingers: &[Rider], link: &str) -> Option<Frame> {
    if link == arm.grasp_link {
        return Some(Frame::GraspPoint);
    }
    if let Some(finger) = fingers.iter().find(|finger| finger.name == link) {
        return Some(Frame::from(finger));
    }
    let chain = arm.geometry.chain();
    let tree = chain.tree();
    let posed = chain.at(&[0.0; ARM_DOF]);
    (0..ARM_DOF).find_map(|segment| {
        let carrier = posed.link_name(segment);
        let offset = if carrier == link {
            Isometry3::identity()
        } else {
            tree.fixed_path_to(tree.link_index(carrier)?, link)?
        };
        Some(Frame::Link { segment, offset })
    })
}

fn position_target<'r>(
    named: &[&ArmModel],
    fingers: &ArmPair<Vec<Rider>>,
    c: &'r GoalGoalPositionConstraintsItem,
) -> Result<Routed<'r, PositionTarget>, String> {
    let (side, frame) = frame(named, fingers, &c.link_name)?;
    if !c.position.iter().all(|v| v.is_finite()) {
        return Err(format!(
            "the position {:?} of {} is not finite; send three finite numbers (m)",
            c.position, c.link_name
        ));
    }
    Ok(Routed {
        side,
        name: &c.link_name,
        target: PositionTarget {
            frame,
            position: Vector3::from(c.position),
            tolerance_m: tolerance("tolerance_m", &c.link_name, c.tolerance_m)?,
        },
    })
}

fn orientation_target<'r>(
    named: &[&ArmModel],
    fingers: &ArmPair<Vec<Rider>>,
    c: &'r GoalGoalOrientationConstraintsItem,
) -> Result<Routed<'r, OrientationTarget>, String> {
    let (side, frame) = frame(named, fingers, &c.link_name)?;
    let orientation = orientation_from_parts(c.orientation)?;
    let free_axis = match c.free_axis.as_str() {
        "none" => None,
        "x" => Some(Axis::X),
        "y" => Some(Axis::Y),
        "z" => Some(Axis::Z),
        other => {
            return Err(format!(
                "free_axis is \"none\", \"x\", \"y\" or \"z\", not {other:?}"
            ));
        }
    };
    Ok(Routed {
        side,
        name: &c.link_name,
        target: OrientationTarget {
            frame,
            orientation,
            tolerance_rad: tolerance("tolerance_rad", &c.link_name, c.tolerance_rad)?,
            free_axis,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use peppygen::exposed_actions::motion_planning::plan_joint_path::GoalArmsItem;
    use peppygen::exposed_services::motion_planning::check_joint_path::{
        CheckJointPathArmsItem, CheckJointPathArmsItemWaypointsItem,
    };

    use crate::fingers::{riders, riders_of_both_arms};
    use crate::robot::tests::OPENARM_V2;

    fn arms() -> &'static ArmPair<ArmModel> {
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
            goal_orientation_constraints: vec![GoalGoalOrientationConstraintsItem {
                link_name: "openarm_left_tcp".to_string(),
                orientation: [0.0, 0.0, 0.0, 1.0005],
                tolerance_rad: 0.1,
                free_axis: "z".to_string(),
            }],
            objective: "any".to_string(),
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
            goal_orientation_constraints: Vec::new(),
            ..goal()
        }
    }

    fn parse(goal: &GoalRequestData) -> Result<PlanQuery, String> {
        let fingers =
            riders_of_both_arms(&OPENARM_V2.self_model, arms()).expect("the arms' fingers");
        parse_plan(goal, arms(), &fingers)
    }

    fn refusal(goal: GoalRequestData) -> String {
        parse(&goal).expect_err("a refusal")
    }

    // A goal for the grasp link parses into the arm it moves, its start where it stands,
    // and its time; a near-unit orientation is normalized.
    #[test]
    fn a_goal_of_the_grasp_link_parses() {
        let query = parse(&goal()).expect("a query");
        assert!(matches!(
            query.arm_goals,
            ArmGoals::One {
                side: Side::Left,
                ..
            }
        ));
        assert_eq!(query.order.sides(), [Side::Left]);
        assert_eq!(query.starts, ArmPair::new(None, None));
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
            ArmPair::new(Some([0.1; ARM_DOF]), None)
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
        assert_eq!(query.order.sides(), [Side::Right, Side::Left]);
        assert!(matches!(
            query.arm_goals,
            ArmGoals::One {
                side: Side::Left,
                ..
            }
        ));
        assert_eq!(query.starts, ArmPair::new(None, Some([0.2; ARM_DOF])));
    }

    // Each constraint goes to the named arm whose joint or link it names, and arms may
    // take different kinds: joints for one, a pose for the other.
    #[test]
    fn each_arm_takes_the_constraints_of_its_own_joints_and_links() {
        let mut both = goal();
        both.arms.push(arm("right_arm"));
        both.goal_joint_constraints = vec![joint_constraint("openarm_right_joint4", 1.0)];
        let query = parse(&both).expect("a query");
        let ArmGoals::Both(goals) = query.arm_goals else {
            panic!("both arms move");
        };
        assert_eq!(goals.left, goal_of_the_left_grasp_point());
        assert_eq!(
            goals.right,
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
            objective: "shortest".to_string(),
            ..goal()
        };
        assert_eq!(refusal(shortest), "objective is \"any\", not \"shortest\"");
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
                side: Side::Right,
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
            Checked::Both(vec![ArmPair::new([0.1; ARM_DOF], [0.2; ARM_DOF])])
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
                ArmGoals::Both(_) => panic!("{link}: one arm moves"),
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
        let fingers = riders(&OPENARM_V2.self_model, &left.geometry, Side::Left)
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
