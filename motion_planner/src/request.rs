//! The requests of plan_joint_path and check_joint_path, parsed against the robot's arms:
//! the arm, where its path starts, and the goal or the waypoints, in the terms of
//! motion_planning_core. A request that does not parse is refused with `rejected`, and
//! the message says what to send instead.

use std::num::NonZeroU64;
use std::time::Duration;

use chain_kinematics::nalgebra::{Isometry3, Quaternion, UnitQuaternion, Vector3};
use collision_geometry::chain::Rider;
use control_core::positive_finite::PositiveFinite;
use motion_planning_core::{
    Axis, Frame, Goal, GoalError, JointTarget, OrientationTarget, PositionTarget,
};
use ompl_bridge::Planner;
use openarm_description::{ARM_DOF, ArmPair, Side};
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    GoalGoalJointConstraintsItem, GoalGoalOrientationConstraintsItem,
    GoalGoalPositionConstraintsItem, GoalRequestData,
};
use peppygen::exposed_services::motion_planning::check_joint_path;

use crate::robot::ArmModel;

/// How far a quaternion's norm may lie from 1 and still name an orientation, which is
/// then normalized.
const UNIT_NORM_SLACK: f64 = 1e-3;
/// The longest a plan may plan, in seconds.
pub const MAX_PLANNING_TIME_S: f64 = 60.0;

/// What a plan optimizes.
#[derive(Debug, Clone, Copy)]
pub enum Objective {
    /// The first valid path found.
    Any,
    /// The path's joint-space length, shortened until the plan's deadline.
    Shortest,
}

impl Objective {
    /// The planner that plans for it: RRTConnect answers the first path it finds, and
    /// RRT* keeps shortening its path until the plan's deadline.
    pub fn planner(self) -> Planner {
        match self {
            Self::Any => Planner::RrtConnect,
            Self::Shortest => Planner::RrtStar,
        }
    }

    /// The iterations each attempt of a plan may take: `bound` for `Any`, and no bound for
    /// `Shortest`, which the plan's deadline, the stop flag or a cancel ends.
    pub fn iterations(self, bound: NonZeroU64) -> NonZeroU64 {
        match self {
            Self::Any => bound,
            Self::Shortest => NonZeroU64::MAX,
        }
    }
}

/// A plan_joint_path goal that parses.
#[derive(Debug)]
pub struct PlanQuery {
    pub side: Side,
    /// Where the path starts; `None`: where the arm stands.
    pub start: Option<[f64; ARM_DOF]>,
    pub goal: Goal,
    pub objective: Objective,
    /// How long the plan may take.
    pub time_limit: Duration,
}

/// A check_joint_path request that parses.
#[derive(Debug)]
pub struct CheckQuery {
    pub side: Side,
    /// Where the path starts; `None`: where the arm stands.
    pub start: Option<[f64; ARM_DOF]>,
    /// The waypoints after the start, at least one.
    pub waypoints: Vec<[f64; ARM_DOF]>,
}

/// Parses a plan_joint_path goal against `arms`, whose fingers hang as `fingers` hold
/// them.
pub fn parse_plan(
    raw: &GoalRequestData,
    arms: &ArmPair<ArmModel>,
    fingers: &ArmPair<Vec<Rider>>,
) -> Result<PlanQuery, String> {
    let names: Vec<&str> = raw.arms.iter().map(|a| a.arm_name.as_str()).collect();
    let arm = the_arm(arms, &names)?;
    let start = start_of(arm, &raw.arms[0].start_joint_positions)?;
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
    let joints = raw
        .goal_joint_constraints
        .iter()
        .map(|c| joint_target(arm, c))
        .collect::<Result<Vec<_>, String>>()?;
    let positions = raw
        .goal_position_constraints
        .iter()
        .map(|c| position_target(arm, &fingers[arm.side], c))
        .collect::<Result<Vec<_>, String>>()?;
    let orientations = raw
        .goal_orientation_constraints
        .iter()
        .map(|c| orientation_target(arm, &fingers[arm.side], c))
        .collect::<Result<Vec<_>, String>>()?;
    let goal = Goal::new(ARM_DOF, joints, positions, orientations).map_err(|e| match e {
        GoalError::Frames(..) => one_link(arm, raw),
        e => e.to_string(),
    })?;
    let objective = match raw.objective.as_str() {
        "any" => Objective::Any,
        "shortest" => Objective::Shortest,
        other => {
            return Err(format!(
                "objective is \"any\" or \"shortest\", not {other:?}"
            ));
        }
    };
    let time = raw.allowed_planning_time_s;
    if !(time > 0.0 && time <= MAX_PLANNING_TIME_S) {
        return Err(format!(
            "allowed_planning_time_s is {time}; send the seconds the plan may take, more than 0 \
             and at most {MAX_PLANNING_TIME_S}"
        ));
    }
    Ok(PlanQuery {
        side: arm.side,
        start,
        goal,
        objective,
        time_limit: Duration::from_secs_f64(time),
    })
}

/// Parses a check_joint_path request against `arms`.
pub fn parse_check(
    raw: &check_joint_path::RequestData,
    arms: &ArmPair<ArmModel>,
) -> Result<CheckQuery, String> {
    let names: Vec<&str> = raw.arms.iter().map(|a| a.arm_name.as_str()).collect();
    let arm = the_arm(arms, &names)?;
    let path = &raw.arms[0];
    let start = start_of(arm, &path.start_joint_positions)?;
    if path.waypoints.is_empty() {
        return Err(format!(
            "the path of {} has no waypoint; send at least one after its start",
            arm.name
        ));
    }
    let waypoints = path
        .waypoints
        .iter()
        .enumerate()
        .map(|(k, w)| joint_positions(arm, &format!("waypoint {}", k + 1), &w.joint_positions))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(CheckQuery {
        side: arm.side,
        start,
        waypoints,
    })
}

/// The refusal of a goal whose position and orientation constraints name more than one
/// link of `arm`: it names the lists and the links sent, and what to send.
fn one_link(arm: &ArmModel, raw: &GoalRequestData) -> String {
    let named: Vec<&str> = raw
        .goal_position_constraints
        .iter()
        .map(|c| c.link_name.as_str())
        .chain(
            raw.goal_orientation_constraints
                .iter()
                .map(|c| c.link_name.as_str()),
        )
        .fold(Vec::new(), |mut named, link| {
            if !named.contains(&link) {
                named.push(link);
            }
            named
        });
    let lists: Vec<&str> = [
        (
            "goal_position_constraints",
            raw.goal_position_constraints.is_empty(),
        ),
        (
            "goal_orientation_constraints",
            raw.goal_orientation_constraints.is_empty(),
        ),
    ]
    .into_iter()
    .filter(|(_, empty)| !empty)
    .map(|(list, _)| list)
    .collect();
    let name = if lists.len() == 1 { "names" } else { "name" };
    format!(
        "{} {name} {}; a goal holds the position and the orientation of one link: send both for one link, such as {}",
        lists.join(" and "),
        named.join(" and "),
        arm.grasp_link
    )
}

/// The arm `names` names, refused unless it names one arm of `arms`.
fn the_arm<'a>(arms: &'a ArmPair<ArmModel>, names: &[&str]) -> Result<&'a ArmModel, String> {
    let [name] = names else {
        return Err(format!(
            "arms holds {} arms; a request names one arm",
            names.len()
        ));
    };
    arms.iter().find(|arm| arm.name == *name).ok_or_else(|| {
        format!(
            "the robot has no arm named {name:?}; its arms are {}",
            arms.iter()
                .map(|arm| arm.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
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

/// The tolerance `field` of the constraint on `named`, refused unless a finite number above
/// 0, the message naming the field and the joint or link.
fn tolerance(field: &str, named: &str, value: f64) -> Result<PositiveFinite, String> {
    PositiveFinite::parse(value)
        .map_err(|_| format!("{field} of {named} is {value}; send a finite number above 0"))
}

fn joint_target(arm: &ArmModel, c: &GoalGoalJointConstraintsItem) -> Result<JointTarget, String> {
    let joint = arm
        .joint_names
        .iter()
        .position(|j| *j == c.joint_name)
        .ok_or_else(|| {
            format!(
                "{} has no joint {:?}; its joints are {}",
                arm.name,
                c.joint_name,
                arm.joint_names.join(", ")
            )
        })?;
    let limit = arm.geometry.chain().limits()[joint];
    if !(limit.lo..=limit.hi).contains(&c.position) {
        return Err(format!(
            "the target {} of {} lies outside its limits {} to {}",
            c.position, c.joint_name, limit.lo, limit.hi
        ));
    }
    Ok(JointTarget {
        joint,
        position: c.position,
        above: tolerance("tolerance_above", &c.joint_name, c.tolerance_above)?,
        below: tolerance("tolerance_below", &c.joint_name, c.tolerance_below)?,
    })
}

/// The frame of the link of `arm` that a position or an orientation constraint names:
/// the grasp point; a finger of the arm's `fingers`, where it hangs off the hand; or the
/// link of a joint of the arm, or a link fixed below it.
fn frame(arm: &ArmModel, fingers: &[Rider], link: &str) -> Result<Frame, String> {
    if link == arm.grasp_link {
        return Ok(Frame::GraspPoint);
    }
    if let Some(finger) = fingers.iter().find(|finger| finger.link == link) {
        return Ok(Frame::Link {
            segment: finger.segment,
            offset: finger.offset,
        });
    }
    let chain = arm.geometry.chain();
    let tree = chain.tree();
    let posed = chain.at(&[0.0; ARM_DOF]);
    (0..ARM_DOF)
        .find_map(|segment| {
            let carrier = posed.link_name(segment);
            let offset = if carrier == link {
                Isometry3::identity()
            } else {
                tree.fixed_path_to(tree.link_index(carrier)?, link)?
            };
            Some(Frame::Link { segment, offset })
        })
        .ok_or_else(|| {
            format!(
                "{} has no link {link:?}; its links are {}",
                arm.name,
                arm.link_names.join(", ")
            )
        })
}

fn position_target(
    arm: &ArmModel,
    fingers: &[Rider],
    c: &GoalGoalPositionConstraintsItem,
) -> Result<PositionTarget, String> {
    let frame = frame(arm, fingers, &c.link_name)?;
    if !c.position.iter().all(|v| v.is_finite()) {
        return Err(format!(
            "the position {:?} of {} is not finite; send three finite numbers (m)",
            c.position, c.link_name
        ));
    }
    Ok(PositionTarget {
        frame,
        position: Vector3::from(c.position),
        tolerance_m: tolerance("tolerance_m", &c.link_name, c.tolerance_m)?,
    })
}

fn orientation_target(
    arm: &ArmModel,
    fingers: &[Rider],
    c: &GoalGoalOrientationConstraintsItem,
) -> Result<OrientationTarget, String> {
    let frame = frame(arm, fingers, &c.link_name)?;
    let [x, y, z, w] = c.orientation;
    let quaternion = Quaternion::new(w, x, y, z);
    if !(1.0 - UNIT_NORM_SLACK..=1.0 + UNIT_NORM_SLACK).contains(&quaternion.norm()) {
        return Err(format!(
            "an orientation is a unit quaternion [x, y, z, w]; [{x}, {y}, {z}, {w}] has norm {}",
            quaternion.norm()
        ));
    }
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
    Ok(OrientationTarget {
        frame,
        orientation: UnitQuaternion::from_quaternion(quaternion),
        tolerance_rad: tolerance("tolerance_rad", &c.link_name, c.tolerance_rad)?,
        free_axis,
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

    /// A goal of the left arm for its grasp point, from where it stands.
    fn goal() -> GoalRequestData {
        GoalRequestData {
            arms: vec![GoalArmsItem {
                arm_name: "left_arm".to_string(),
                start_joint_positions: Vec::new(),
            }],
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

    fn joint_goal(name: &str, position: f64) -> GoalRequestData {
        GoalRequestData {
            goal_joint_constraints: vec![GoalGoalJointConstraintsItem {
                joint_name: name.to_string(),
                position,
                tolerance_above: 0.01,
                tolerance_below: 0.01,
            }],
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

    // A goal for the grasp link parses into the arm's side, its start where it stands,
    // the default time, and the objective's planner; a near-unit orientation is
    // normalized.
    #[test]
    fn a_goal_of_the_grasp_link_parses() {
        let query = parse(&goal()).expect("a query");
        assert_eq!(query.side, Side::Left);
        assert_eq!(query.start, None);
        assert_eq!(query.time_limit, Duration::from_secs(1));
        assert_eq!(query.objective.planner(), Planner::RrtConnect);
        let shortest = GoalRequestData {
            objective: "shortest".to_string(),
            allowed_planning_time_s: 2.5,
            ..goal()
        };
        let query = parse(&shortest).expect("a query");
        assert_eq!(query.objective.planner(), Planner::RrtStar);
        let bound = NonZeroU64::new(20_000).expect("nonzero");
        assert_eq!(query.objective.iterations(bound), NonZeroU64::MAX);
        assert_eq!(Objective::Any.iterations(bound), bound);
        assert_eq!(query.time_limit, Duration::from_millis(2500));
        let mut started = goal();
        started.arms[0].start_joint_positions = vec![0.1; ARM_DOF];
        assert_eq!(
            parse(&started).expect("a query").start,
            Some([0.1; ARM_DOF])
        );
    }

    // A goal whose position names one link and whose orientation names another, or whose
    // two positions name two links, is refused, naming the lists and the links sent and
    // what to send.
    #[test]
    fn a_goal_of_two_links_is_refused_with_what_to_send() {
        let mut two = goal();
        two.goal_orientation_constraints[0].link_name = "openarm_left_link4".to_string();
        assert_eq!(
            refusal(two),
            "goal_position_constraints and goal_orientation_constraints name \
             openarm_left_tcp and openarm_left_link4; a goal holds the position and the \
             orientation of one link: send both for one link, such as openarm_left_tcp"
        );
        let mut positions = goal();
        positions.goal_orientation_constraints.clear();
        let mut elbow = positions.goal_position_constraints[0].clone();
        elbow.link_name = "openarm_left_link4".to_string();
        positions.goal_position_constraints.push(elbow);
        assert_eq!(
            refusal(positions),
            "goal_position_constraints names openarm_left_tcp and openarm_left_link4; a goal \
             holds the position and the orientation of one link: send both for one link, \
             such as openarm_left_tcp"
        );
    }

    // A goal that names what the robot does not have, or names it twice, or a number
    // outside its range, is refused, and the message says what to send instead.
    #[test]
    fn a_goal_outside_the_robot_is_refused_with_what_to_send() {
        let mut two = goal();
        two.arms.push(GoalArmsItem {
            arm_name: "right_arm".to_string(),
            start_joint_positions: Vec::new(),
        });
        assert_eq!(refusal(two), "arms holds 2 arms; a request names one arm");
        let mut third = goal();
        third.arms[0].arm_name = "third_arm".to_string();
        assert_eq!(
            refusal(third),
            "the robot has no arm named \"third_arm\"; its arms are left_arm, right_arm"
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
        assert!(
            refusal(joint_goal("openarm_left_joint4", -1.0))
                .starts_with("the target -1 of openarm_left_joint4 lies outside its limits")
        );
        let mut twice = joint_goal("openarm_left_joint4", 1.0);
        twice
            .goal_joint_constraints
            .push(twice.goal_joint_constraints[0].clone());
        assert_eq!(
            refusal(twice),
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
        let mut mixed = goal();
        mixed.goal_joint_constraints =
            joint_goal("openarm_left_joint4", 1.0).goal_joint_constraints;
        assert_eq!(
            refusal(mixed),
            "a goal names joints, or positions and orientations, not both"
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
        let smoothest = GoalRequestData {
            objective: "smoothest".to_string(),
            ..goal()
        };
        assert_eq!(
            refusal(smoothest),
            "objective is \"any\" or \"shortest\", not \"smoothest\""
        );
    }

    fn path(waypoints: Vec<Vec<f64>>) -> check_joint_path::RequestData {
        check_joint_path::RequestData {
            arms: vec![CheckJointPathArmsItem {
                arm_name: "right_arm".to_string(),
                start_joint_positions: Vec::new(),
                waypoints: waypoints
                    .into_iter()
                    .map(|joint_positions| CheckJointPathArmsItemWaypointsItem { joint_positions })
                    .collect(),
            }],
        }
    }

    // A path of the arm parses into its waypoints, a waypoint outside a joint limit
    // among them; a path with no waypoint, or a waypoint that is not a position for
    // each joint, is refused.
    #[test]
    fn a_path_parses_into_its_waypoints() {
        let query = parse_check(&path(vec![vec![0.2; ARM_DOF], vec![9.0; ARM_DOF]]), arms())
            .expect("a query");
        assert_eq!(query.side, Side::Right);
        assert_eq!(query.waypoints, vec![[0.2; ARM_DOF], [9.0; ARM_DOF]]);
        assert_eq!(
            parse_check(&path(Vec::new()), arms()).expect_err("no waypoint"),
            "the path of right_arm has no waypoint; send at least one after its start"
        );
        let mut nan = vec![0.0; ARM_DOF];
        nan[2] = f64::NAN;
        assert_eq!(
            parse_check(&path(vec![vec![0.0; ARM_DOF], nan]), arms()).expect_err("a NaN"),
            "waypoint 2 of right_arm holds NaN for openarm_right_joint3; a position is a finite number"
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
            parse(&goal).unwrap_or_else(|e| panic!("{link}: {e}")).goal
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
            frame_of(finger.link),
            held(Frame::Link {
                segment: finger.segment,
                offset: finger.offset,
            })
        );
        for link in &arms()[Side::Left].link_names {
            frame_of(link);
        }
    }
}
