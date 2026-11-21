//! The base_motion goals, checked in the order base_motion states: each goal's fields
//! parsed off the wire, every value the base takes or the refusal (`rejected`) naming
//! the value it does not; then, once the robot's tree has placed the goal (see
//! `actions`), its placed value held to the floor plane and to the base's limits, each
//! refused with `rejected` naming the value as placed and what to send.

use control_core::geometry::{GeometryError, Pose, Stamped, Twist};
use control_core::move_seconds::MoveSeconds;
use control_core::planar_pose::{PlanarPose, normalize_angle};
use control_core::positive_finite::PositiveFinite;
use control_core::robot_tree::Placer;
use control_core::time::nanoseconds;
use control_core::twist::PlanarTwist;
use peppygen::exposed_actions::base_motion::{drive_base, move_base};

use crate::config::{AxisSpec, BaseAxes};
use crate::move_control::{MovePlan, Tolerances, plan_move};

/// The longest a goal may be asked to take or may plan, and to settle after
/// its planned end (s): the bound base_motion states.
const MAX_MOVE_DURATION_S: f64 = 60.0;

/// What to send in place of a target off the floor plane, as base_motion says.
const FLOOR_TARGET: &str = "send a target in odom or the robot frame with a position [x, y, 0] \
                            and an orientation [0, 0, sin(yaw/2), cos(yaw/2)]";

/// What to send in place of a velocity off the floor plane, as base_motion says.
const FLOOR_VELOCITY: &str = "send a velocity in the robot frame of [vx, vy, 0] and [0, 0, wz]";

/// A drive_base goal's fields: the velocity in its frame at its time, how long
/// (ns) to hold it once the command reaches it, and how long (ns) the base may
/// take to rest once the command reaches 0.
#[derive(Clone, Debug, PartialEq)]
pub struct DriveFields {
    pub velocity: Stamped<Twist>,
    hold_duration_ns: i64,
    settle_timeout_ns: i64,
}

/// A drive_base goal: the velocity to hold in the robot frame, for how long
/// (ns) once the command reaches it, and how long (ns) the base may take to
/// rest once the command reaches 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveRequest {
    pub velocity: PlanarTwist,
    pub hold_duration_ns: i64,
    pub settle_timeout_ns: i64,
}

/// A move_base goal's fields: the target in its frame at its time, the
/// duration asked for (s), the tolerances, and how long (ns) the base may
/// stand outside them after the planned end.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveFields {
    pub target: Stamped<Pose>,
    duration_s: f64,
    tolerances: Tolerances,
    settle_timeout_ns: i64,
}

/// A move_base goal: the target in odom, its motion planned at the base's
/// limits over the duration asked for, from where the robot frame stands at
/// acceptance, the tolerances, and how long (ns) the base may stand outside
/// them after the planned end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveRequest {
    pub target: PlanarPose,
    pub plan: MovePlan,
    pub tolerances: Tolerances,
    pub settle_timeout_ns: i64,
    /// The node's `rotate_to_heading_min_angle_rad` (rad).
    pub rotate_to_heading_min_angle: f64,
}

/// The seconds of `field` off the wire, from 0 to base_motion's bound.
fn move_seconds(field: &str, value: f64) -> Result<f64, String> {
    MoveSeconds::from_wire(field, value, MAX_MOVE_DURATION_S).map(MoveSeconds::get)
}

fn tolerance(field: &str, value: f64) -> Result<f64, String> {
    PositiveFinite::parse_field(field, value).map(PositiveFinite::get)
}

/// The velocity `value` of `field`, placed in the robot frame, within `range`
/// of describe_base's `limits`.
fn within_range(field: &str, value: f64, limits: &str, range: &AxisSpec) -> Result<f64, String> {
    if (range.min_velocity..=range.max_velocity).contains(&value) {
        return Ok(value);
    }
    Err(format!(
        "{field}, placed in the robot frame, is {value}, outside the base's {limits} of {} to {}: \
         send a velocity within them",
        range.min_velocity, range.max_velocity
    ))
}

/// `values` as the refusals of control_core print a vector.
fn listed(values: &[f64]) -> String {
    let printed: Vec<String> = values.iter().map(f64::to_string).collect();
    format!("[{}]", printed.join(", "))
}

/// The refusal of a target that, placed in odom, lies off the floor plane.
fn target_off_the_floor(refusal: &GeometryError) -> String {
    match refusal {
        GeometryError::OffFloor { z } => format!(
            "the target's position, placed in odom, lies {z} m above the floor plane: {FLOOR_TARGET}"
        ),
        GeometryError::Tilted {
            orientation,
            tilt_rad,
        } => format!(
            "the target's orientation, placed in odom, {} tilts {tilt_rad} rad off the vertical: \
             {FLOOR_TARGET}",
            listed(orientation)
        ),
        other => format!("the target, placed in odom: {other}"),
    }
}

/// The refusal of a velocity that, placed in the robot frame, moves off the
/// floor plane.
pub fn velocity_off_the_floor(refusal: &GeometryError) -> String {
    match refusal {
        GeometryError::Vertical { z } => format!(
            "the velocity's linear_velocity, placed in the robot frame, moves {z} m/s along z: \
             {FLOOR_VELOCITY}"
        ),
        GeometryError::TiltingTurn { x, y } => format!(
            "the velocity's angular_velocity, placed in the robot frame, turns {x} rad/s about x \
             and {y} rad/s about y: {FLOOR_VELOCITY}"
        ),
        other => format!("the velocity, placed in the robot frame: {other}"),
    }
}

/// `pose` on the floor plane as the control loop steers by it, its heading
/// wrapped.
fn floor_pose(pose: PlanarPose) -> PlanarPose {
    PlanarPose::new(pose.x(), pose.y(), normalize_angle(pose.yaw()))
        .expect("a finite heading wraps to a finite heading")
}

/// A drive_base goal's fields off the wire, its velocity in the frame
/// `placer` reads the goal's frame as.
pub fn drive_fields(
    data: &drive_base::GoalRequestData,
    placer: &Placer,
) -> Result<DriveFields, String> {
    let hold_duration_s = MoveSeconds::positive_from_wire(
        "hold_duration_s",
        data.hold_duration_s,
        MAX_MOVE_DURATION_S,
    )?
    .get();
    let settle_timeout_s = move_seconds("settle_timeout_s", data.settle_timeout_s)?;
    let velocity = &data.velocity;
    let velocity = Twist::parse(velocity.linear_velocity, velocity.angular_velocity)
        .and_then(|twist| {
            Stamped::parse(
                placer.frame_of(velocity.frame.as_deref()),
                velocity.timestamp,
                twist,
            )
        })
        .map_err(|e| format!("the velocity: {e}"))?;
    Ok(DriveFields {
        velocity,
        hold_duration_ns: nanoseconds(hold_duration_s),
        settle_timeout_ns: nanoseconds(settle_timeout_s),
    })
}

/// The drive_base goal of `fields`, its velocity `placed` in the robot frame:
/// on the floor plane, each axis within describe_base's limits along `axes`.
pub fn drive_request(
    fields: DriveFields,
    placed: Twist,
    axes: &BaseAxes,
) -> Result<DriveRequest, String> {
    let planar = placed
        .planar()
        .map_err(|refusal| velocity_off_the_floor(&refusal))?;
    let x = within_range(
        "the velocity's linear_velocity x",
        planar.linear_x(),
        "linear_limits along x",
        &axes.x,
    )?;
    let y = within_range(
        "the velocity's linear_velocity y",
        planar.linear_y(),
        "linear_limits along y",
        &AxisSpec::STILL,
    )?;
    let turn = within_range(
        "the velocity's angular_velocity z",
        planar.angular_z(),
        "angular_limits",
        &axes.yaw,
    )?;
    Ok(DriveRequest {
        velocity: PlanarTwist::new(x, y, turn).expect("placed axes are finite"),
        hold_duration_ns: fields.hold_duration_ns,
        settle_timeout_ns: fields.settle_timeout_ns,
    })
}

/// A move_base goal's fields off the wire, its target in the frame `placer`
/// reads the goal's frame as.
pub fn move_fields(
    data: &move_base::GoalRequestData,
    placer: &Placer,
) -> Result<MoveFields, String> {
    let duration_s = move_seconds("duration_s", data.duration_s)?;
    let tolerances = Tolerances {
        position: tolerance(
            "settle_position_tolerance_m",
            data.settle_position_tolerance_m,
        )?,
        yaw: tolerance(
            "settle_orientation_tolerance_rad",
            data.settle_orientation_tolerance_rad,
        )?,
    };
    let settle_timeout_s = move_seconds("settle_timeout_s", data.settle_timeout_s)?;
    let target = &data.target;
    let target = Pose::parse(target.position, target.orientation)
        .and_then(|pose| {
            Stamped::parse(
                placer.frame_of(target.frame.as_deref()),
                target.timestamp,
                pose,
            )
        })
        .map_err(|e| format!("the target: {e}"))?;
    Ok(MoveFields {
        target,
        duration_s,
        tolerances,
        settle_timeout_ns: nanoseconds(settle_timeout_s),
    })
}

/// The move_base goal of `fields`, its target placed `in_odometry` and, where
/// the robot frame stands now, `from_robot`: on the floor plane, its motion
/// planned along `axes` from the robot frame, and refused when its plan takes
/// longer than 60 s, which the fastest move to a far target does on a slow
/// base.
pub fn move_request(
    fields: MoveFields,
    in_odometry: Pose,
    from_robot: Pose,
    axes: &BaseAxes,
    rotate_to_heading_min_angle: f64,
) -> Result<MoveRequest, String> {
    let on_the_floor = |pose: Pose| {
        pose.planar()
            .map(floor_pose)
            .map_err(|refusal| target_off_the_floor(&refusal))
    };
    let target = on_the_floor(in_odometry)?;
    let plan = plan_move(
        on_the_floor(from_robot)?,
        fields.tolerances.position,
        axes.x.profile(),
        axes.yaw.profile(),
        fields.duration_s,
    );
    if plan.duration_s > MAX_MOVE_DURATION_S {
        return Err(format!(
            "the move to the target takes {:.1} s at the base's limits, more than {MAX_MOVE_DURATION_S:.0} s",
            plan.duration_s
        ));
    }
    Ok(MoveRequest {
        target,
        plan,
        tolerances: fields.tolerances,
        settle_timeout_ns: fields.settle_timeout_ns,
        rotate_to_heading_min_angle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::PI;

    use crate::test_rig::{AXES, ROTATE_TO_HEADING_MIN_ANGLE_RAD, placer};

    const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

    fn drive_goal(
        linear_velocity: [f64; 3],
        angular_velocity: [f64; 3],
        hold_duration_s: f64,
        settle_timeout_s: f64,
    ) -> drive_base::GoalRequestData {
        drive_base::GoalRequestData {
            velocity: drive_base::GoalVelocity {
                frame: None,
                timestamp: None,
                linear_velocity,
                angular_velocity,
            },
            hold_duration_s,
            settle_timeout_s,
        }
    }

    /// The drive_base goal of these fields in the robot frame, where the
    /// velocity places as it is given.
    fn drive(
        linear_velocity: [f64; 3],
        angular_velocity: [f64; 3],
        hold_duration_s: f64,
        settle_timeout_s: f64,
    ) -> Result<DriveRequest, String> {
        let fields = drive_fields(
            &drive_goal(
                linear_velocity,
                angular_velocity,
                hold_duration_s,
                settle_timeout_s,
            ),
            &placer(),
        )?;
        let placed = *fields.velocity.value();
        drive_request(fields, placed, &AXES)
    }

    fn move_goal(
        position: [f64; 3],
        orientation: [f64; 4],
        duration_s: f64,
        settle_timeout_s: f64,
    ) -> move_base::GoalRequestData {
        move_base::GoalRequestData {
            target: move_base::GoalTarget {
                frame: None,
                timestamp: None,
                position,
                orientation,
            },
            duration_s,
            settle_position_tolerance_m: 0.02,
            settle_orientation_tolerance_rad: 0.03,
            settle_timeout_s,
        }
    }

    /// The move_base goal of `data` with the robot frame at odom's origin,
    /// where the target places in odom and in the robot frame as it is given.
    fn move_with(
        axes: &BaseAxes,
        data: &move_base::GoalRequestData,
    ) -> Result<MoveRequest, String> {
        let fields = move_fields(data, &placer())?;
        let placed = *fields.target.value();
        move_request(
            fields,
            placed,
            placed,
            axes,
            ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        )
    }

    fn movement(
        position: [f64; 3],
        duration_s: f64,
        settle_timeout_s: f64,
    ) -> Result<MoveRequest, String> {
        move_with(
            &AXES,
            &move_goal(position, IDENTITY, duration_s, settle_timeout_s),
        )
    }

    #[test]
    fn a_drive_within_the_axes_ranges_is_taken() {
        let request =
            drive([-0.5, 0.0, 0.0], [0.0, 0.0, 1.5], 60.0, 2.0).expect("within the ranges");
        assert_eq!(request.velocity, PlanarTwist::new(-0.5, 0.0, 1.5).unwrap());
        assert_eq!(request.hold_duration_ns, 60_000_000_000);
        assert_eq!(request.settle_timeout_ns, 2_000_000_000);
    }

    #[test]
    fn a_drive_outside_an_axis_range_is_refused_naming_the_value_and_the_range() {
        assert_eq!(
            drive([0.6, 0.0, 0.0], [0.0; 3], 1.0, 1.0),
            Err(
                "the velocity's linear_velocity x, placed in the robot frame, is 0.6, outside the \
                 base's linear_limits along x of -0.5 to 0.5: send a velocity within them"
                    .to_string()
            )
        );
        assert_eq!(
            drive([0.0; 3], [0.0, 0.0, -1.6], 1.0, 1.0),
            Err(
                "the velocity's angular_velocity z, placed in the robot frame, is -1.6, outside \
                 the base's angular_limits of -1.5 to 1.5: send a velocity within them"
                    .to_string()
            )
        );
        assert_eq!(
            drive([0.0, 0.1, 0.0], [0.0; 3], 1.0, 1.0),
            Err(
                "the velocity's linear_velocity y, placed in the robot frame, is 0.1, outside the \
                 base's linear_limits along y of 0 to 0: send a velocity within them"
                    .to_string()
            )
        );
        let not_finite = drive([f64::NAN, 0.0, 0.0], [0.0; 3], 1.0, 1.0).unwrap_err();
        assert!(
            not_finite
                .starts_with("the velocity: a twist must be finite: got linear_velocity x NaN"),
            "{not_finite}"
        );
    }

    #[test]
    fn a_drive_or_a_move_off_the_floor_plane_is_refused_saying_what_to_send() {
        assert_eq!(
            drive([0.1, 0.0, 0.2], [0.0; 3], 1.0, 1.0),
            Err("the velocity's linear_velocity, placed in the robot frame, moves 0.2 m/s along z: \
                 send a velocity in the robot frame of [vx, vy, 0] and [0, 0, wz]"
                .to_string())
        );
        assert_eq!(
            drive([0.1, 0.0, 0.0], [0.3, 0.0, 0.0], 1.0, 1.0),
            Err("the velocity's angular_velocity, placed in the robot frame, turns 0.3 rad/s about \
                 x and 0 rad/s about y: send a velocity in the robot frame of [vx, vy, 0] and \
                 [0, 0, wz]"
                .to_string())
        );
        assert_eq!(
            movement([1.0, 0.0, -0.1], 0.0, 0.0).err(),
            Some(
                "the target's position, placed in odom, lies -0.1 m above the floor plane: send a \
                 target in odom or the robot frame with a position [x, y, 0] and an orientation \
                 [0, 0, sin(yaw/2), cos(yaw/2)]"
                    .to_string()
            )
        );
        assert_eq!(
            move_with(
                &AXES,
                &move_goal([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0, 0.0)
            )
            .err(),
            Some(
                "the target's orientation, placed in odom, [1, 0, 0, 0] tilts 3.141592653589793 \
                 rad off the vertical: send a target in odom or the robot frame with a position \
                 [x, y, 0] and an orientation [0, 0, sin(yaw/2), cos(yaw/2)]"
                    .to_string()
            )
        );
    }

    #[test]
    fn a_drive_refuses_a_hold_of_0_and_times_past_sixty_seconds() {
        assert_eq!(
            drive([0.1, 0.0, 0.0], [0.0; 3], 0.0, 1.0),
            Err(
                "hold_duration_s must be a number of seconds above 0 and at most 60, got 0"
                    .to_string()
            )
        );
        assert!(drive([0.1, 0.0, 0.0], [0.0; 3], 60.5, 1.0).is_err());
        assert_eq!(
            drive([0.1, 0.0, 0.0], [0.0; 3], 1.0, 61.0),
            Err("settle_timeout_s must be a number of seconds from 0 to 60, got 61".to_string())
        );
        assert!(drive([0.1, 0.0, 0.0], [0.0; 3], 1.0, 0.0).is_ok());
    }

    #[test]
    fn a_move_takes_any_target_its_plan_runs_within_sixty_seconds_and_wraps_the_heading() {
        assert!(movement([3.0, 4.01, 0.0], 0.0, 0.0).is_ok());
        let half_turns = 3.0 * PI / 2.0;
        let wrapped = move_with(
            &AXES,
            &move_goal(
                [0.0; 3],
                [0.0, 0.0, half_turns.sin(), half_turns.cos()],
                0.0,
                1.0,
            ),
        )
        .expect("any heading");
        assert!(
            (wrapped.target.yaw().abs() - PI).abs() < 1e-12,
            "{wrapped:?}"
        );
    }

    #[test]
    fn a_move_refuses_times_past_sixty_seconds_tolerances_not_above_zero_and_no_unit_quaternion() {
        assert!(movement([1.0, 0.0, 0.0], 61.0, 0.0).is_err());
        assert!(movement([1.0, 0.0, 0.0], 0.0, -1.0).is_err());
        let not_finite = movement([f64::INFINITY, 0.0, 0.0], 0.0, 0.0).unwrap_err();
        assert!(not_finite.starts_with("the target: "), "{not_finite}");
        let no_tolerance = move_with(
            &AXES,
            &move_base::GoalRequestData {
                settle_position_tolerance_m: 0.0,
                ..move_goal([1.0, 0.0, 0.0], IDENTITY, 0.0, 0.0)
            },
        );
        assert_eq!(
            no_tolerance,
            Err("settle_position_tolerance_m must be a positive number, got 0".to_string())
        );
        let no_unit = move_with(
            &AXES,
            &move_goal([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5], 0.0, 0.0),
        );
        assert_eq!(
            no_unit,
            Err(
                "the target: the orientation [0, 0, 0, 0.5] has a squared norm of 0.25, 0.01 or \
                 more from 1: send a unit quaternion [x, y, z, w], such as [0, 0, 0, 1]"
                    .to_string()
            )
        );
    }

    // The target the loop steers to is the one in odom, and the plan runs from
    // where the robot frame stands: 1 m ahead of a robot frame 1 m along odom's
    // x plans the 1 m move.
    #[test]
    fn a_move_steers_to_its_target_in_odom_and_plans_from_the_robot_frame() {
        let data = move_goal([2.0, 0.0, 0.0], IDENTITY, 0.0, 0.0);
        let fields = move_fields(&data, &placer()).expect("the fields");
        let in_odometry = Pose::parse([2.0, 0.0, 0.0], IDENTITY).unwrap();
        let from_robot = Pose::parse([1.0, 0.0, 0.0], IDENTITY).unwrap();
        let request = move_request(
            fields,
            in_odometry,
            from_robot,
            &AXES,
            ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        )
        .expect("a move");
        assert_eq!(request.target, PlanarPose::new(2.0, 0.0, 0.0).unwrap());
        let one_meter = movement([1.0, 0.0, 0.0], 0.0, 0.0).expect("a move");
        assert_eq!(request.plan, one_meter.plan);
    }

    #[test]
    fn a_move_whose_plan_takes_longer_than_sixty_seconds_is_refused_naming_its_time() {
        // At 0.05 m/s the 5 m drive alone takes 100 s past its ramps.
        let slow = BaseAxes {
            x: AxisSpec {
                min_velocity: -0.05,
                max_velocity: 0.05,
                deadband: 0.001,
                ..AXES.x
            },
            ..AXES
        };
        let at = |position: [f64; 3], duration_s: f64| {
            move_with(&slow, &move_goal(position, IDENTITY, duration_s, 0.0))
        };
        let refused = at([5.0, 0.0, 0.0], 0.0).unwrap_err();
        assert!(
            refused.starts_with("the move to the target takes 100.")
                && refused.ends_with("s at the base's limits, more than 60 s"),
            "{refused}"
        );
        // 2.5 m takes about 50 s: taken, its plan at least that long.
        let taken = at([2.5, 0.0, 0.0], 0.0).expect("within 60 s");
        assert!(taken.plan.duration_s > 50.0 && taken.plan.duration_s <= 60.0);
        // Asked for 60 s, the same move plans 60 s.
        let asked = at([2.5, 0.0, 0.0], 60.0).expect("within 60 s");
        assert_eq!(asked.plan.duration_s, 60.0);
    }
}
