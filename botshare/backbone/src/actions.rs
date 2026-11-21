//! The base_motion actions: each goal is parsed off the wire and refused with
//! `rejected` for a value the base does not take, placed by the robot's tree
//! and refused with the lookup's failure kind where the tree cannot place it,
//! held to the floor plane and the base's limits, then admitted against the
//! base's state ([`BaseStatus::admit`]), and handed with its claim on the
//! base to the control loop, which runs it and ends it.

use std::sync::Arc;

use control_core::move_slot::MoveSlot;
use control_core::robot_tree::{Placer, Unplaced};
use control_core::sync::lock_ignoring_poison;
use peppygen::exposed_actions::base_motion::{drive_base, move_base};
use peppygen::{NodeRunner, Result};
use tokio::sync::mpsc;
use tracing::error;

use crate::clock;
use crate::config::BaseConfig;
use crate::control::{AcceptedGoal, Event};
use crate::failure::{Failure, FailureKind, TargetFailure};
use crate::goal::{
    DriveRequest, MoveRequest, drive_fields, drive_request, move_fields, move_request,
};
use crate::status::{Admitted, BaseStatus};
use crate::transforms::odometry_frame;

fn rejected(reason: String) -> TargetFailure {
    Failure::new(FailureKind::Rejected, reason).into()
}

/// A drive_base goal off the wire: its fields, its velocity placed in the
/// robot frame by `placer`, then held to the base's limits.
fn drive(
    data: &drive_base::GoalRequestData,
    config: &BaseConfig,
    placer: &Placer,
) -> std::result::Result<DriveRequest, TargetFailure> {
    let fields = drive_fields(data, placer).map_err(rejected)?;
    let placed = placer
        .place_velocity_now(&fields.velocity)
        .map_err(|e| Unplaced::of("the velocity", &e))?;
    drive_request(fields, placed, &config.axes).map_err(rejected)
}

/// A move_base goal off the wire: its fields, its target placed by `placer` in
/// odom and in the robot frame, then held to the floor plane and planned.
fn movement(
    data: &move_base::GoalRequestData,
    config: &BaseConfig,
    placer: &Placer,
) -> std::result::Result<MoveRequest, TargetFailure> {
    let fields = move_fields(data, placer).map_err(rejected)?;
    let unplaced = |e| Unplaced::of("the target", &e);
    let in_odometry = placer
        .place_in_now(&fields.target, &odometry_frame())
        .map_err(unplaced)?
        .value;
    let from_robot = placer.place_now(&fields.target).map_err(unplaced)?.value;
    move_request(
        fields,
        in_odometry,
        from_robot,
        &config.axes,
        config.rotate_to_heading_min_angle_rad,
    )
    .map_err(rejected)
}

/// The goal's request parsed and placed by `place`, then the base's move slot
/// claimed for it, or the refusal. The odometry the robot's tree places by is
/// read before the placement, and the goal is admitted on it: the control
/// loop fails a goal whose odometry started over since
/// ([`BaseStatus::odometry_start`]).
fn admit<R>(
    place: impl FnOnce() -> std::result::Result<R, TargetFailure>,
    status: &BaseStatus,
    slot: &MoveSlot,
) -> std::result::Result<(R, Admitted), TargetFailure> {
    let placed_by = status.odometry_start();
    let request = place()?;
    let admitted = status.admit(clock::now_ns(), placed_by, slot)?;
    Ok((request, admitted))
}

/// Expose the action `$action` as `$fn_name`: parse and place each goal with
/// `$parse`, admit it, and hand it to the control loop as
/// `AcceptedGoal::$variant`.
/// Written once here because the two generated action modules carry
/// distinct types with an identical surface.
macro_rules! goal_runner {
    ($fn_name:ident, $action:ident, $parse:expr, $variant:ident) => {
        pub async fn $fn_name(
            runner: Arc<NodeRunner>,
            config: Arc<BaseConfig>,
            placer: Arc<Placer>,
            status: BaseStatus,
            slot: MoveSlot,
            events: mpsc::Sender<Event>,
        ) -> Result<()> {
            let parse: fn(
                &$action::GoalRequestData,
                &BaseConfig,
                &Placer,
            ) -> std::result::Result<_, TargetFailure> = $parse;
            let mut handle = $action::ActionHandle::expose(&runner).await?;
            loop {
                // The admitted request and claim pass out of the decider,
                // which the generated handle calls once per goal.
                let admitted = std::sync::Mutex::new(None);
                let accepted = handle
                    .handle_goal_next_request(|request| {
                        Ok(
                            match admit(|| parse(&request.data, &config, &placer), &status, &slot) {
                                Ok(goal) => {
                                    *lock_ignoring_poison(&admitted) = Some(goal);
                                    $action::GoalDecision::accept()
                                }
                                Err(refusal) => {
                                    let (kind, message) =
                                        refusal.for_member($action::FailureKind::parse);
                                    $action::GoalDecision::refuse(kind, message)
                                }
                            },
                        )
                    })
                    .await?;
                let Some(ctx) = accepted else { return Ok(()) };
                let (request, claim) = lock_ignoring_poison(&admitted)
                    .take()
                    .expect("the decider keeps the goal it accepts");
                let goal = AcceptedGoal::$variant {
                    request,
                    ctx: Box::new(ctx),
                    claim,
                };
                if events.send(Event::Goal(goal)).await.is_err() {
                    error!("{}: the control loop is not running", stringify!($action));
                    return Ok(());
                }
            }
        }
    };
}

goal_runner!(run_drive_base, drive_base, drive, Drive);
goal_runner!(run_move_base, move_base, movement, Move);

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::FRAC_PI_2;
    use std::time::{Duration, SystemTime};

    use control_core::node_clock::NodeClock;
    use control_core::robot_tree::RobotTree;
    use frame_tree::arithmetic::quaternion_of_yaw;
    use frame_tree::{Publisher, StaticMessage, TransformRecord};

    use crate::config::BaseParameters;
    use crate::test_rig::params;
    use crate::transforms::robot_frame;

    // A drive_base velocity in a frame turned a quarter turn about +z from the
    // robot frame is placed in the robot frame before the base's limits judge
    // it: 0.1 m/s along that frame's -y is 0.1 m/s ahead, and its turn about z
    // keeps its rate. The placement waits on the runtime as the generated
    // decider's does, on a multi-thread runtime.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_drive_in_a_turned_frame_is_placed_in_the_robot_frame() {
        let tree =
            RobotTree::new(Duration::from_secs(10), NodeClock::wall()).expect("a window above 0");
        let initializer = Publisher {
            core_node: "core".to_owned(),
            instance_id: "initializer".to_owned(),
        };
        let turned = StaticMessage {
            transforms: vec![TransformRecord::new(
                "base_footprint",
                "turned",
                [0.2, 0.0, 0.3],
                quaternion_of_yaw(FRAC_PI_2),
            )],
        };
        tree.take_static(&initializer, &turned)
            .expect("the turned frame");
        let placer = Placer::new(Arc::new(tree), robot_frame(), Duration::ZERO);
        let config = BaseParameters::parse(&params())
            .expect("the rig's parameters")
            .configure("botshare_base", SystemTime::UNIX_EPOCH)
            .expect("the bundled base");
        let data = drive_base::GoalRequestData {
            velocity: drive_base::GoalVelocity {
                frame: Some("turned".to_owned()),
                timestamp: None,
                linear_velocity: [0.0, -0.1, 0.0],
                angular_velocity: [0.0, 0.0, 0.2],
            },
            hold_duration_s: 1.0,
            settle_timeout_s: 1.0,
        };
        let request = drive(&data, &config, &placer).expect("a drive in the turned frame");
        let velocity = request.velocity;
        assert!(
            (velocity.linear_x() - 0.1).abs() < 1e-12
                && velocity.linear_y() == 0.0
                && (velocity.angular_z() - 0.2).abs() < 1e-12,
            "{velocity:?}"
        );
    }
}
