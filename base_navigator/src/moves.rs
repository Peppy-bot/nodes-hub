//! The moves of base_navigation that hold no pose: a spin turns the base in
//! place by its angle, a drive_on_heading drives it ahead and a back_up back
//! by their distance, each from where the robot frame stands when it is
//! accepted, in odom, as the [`Turn`] and the [`StraightDrive`] of the
//! recoveries step and sweep. A move fails with `blocked` when its sweep meets
//! an occupied cell, and with `timeout` when its timeout_s, above 0, passes
//! before the odometry measures the whole turn or distance. It completes at
//! the step the odometry measures it, as Nav2's Spin and DriveOnHeading
//! succeed there (nav2_behaviors plugins/spin.cpp:119-121,
//! include/nav2_behaviors/plugins/drive_on_heading.hpp:131-133).

use control_core::planar_pose::PlanarPose;
use control_core::twist::PlanarTwist;
use navigation_core::PaddedFootprint;
use navigation_core::grid::CostGrid;

use crate::base::Base;
use crate::config::NavigatorConfig;
use crate::failure::{Ended, Failure, FailureKind};
use crate::recovery::{Direction, MotionStep, SpinRequest, StraightDrive, StraightRequest, Turn};

/// What a move runs.
#[derive(Clone, Copy, Debug)]
enum Motion {
    Turn(Turn),
    Straight(StraightDrive),
}

/// What a move reports each step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MoveFeedback {
    /// A spin's angle turned (rad), by the odometry.
    AngleTurned(f64),
    /// A drive's distance (m) from where it started, by the odometry.
    DistanceTraveled(f64),
}

/// What the base reports at a step, as a move reads it.
#[derive(Clone, Copy, Debug)]
pub struct MoveInputs<'a> {
    pub now_ns: i64,
    pub pose: PlanarPose,
    pub grid: &'a CostGrid,
    pub footprint: &'a PaddedFootprint,
}

/// What one step of a move gives.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveStep {
    /// The command of the step, before the stop; none once the move ends.
    pub command: Option<PlanarTwist>,
    pub feedback: MoveFeedback,
    pub ended: Option<Ended<FailureKind>>,
}

/// A move in flight.
#[derive(Clone, Copy, Debug)]
pub struct Move {
    motion: Motion,
}

impl Move {
    /// The spin of `request` for `base`, from `start` at `now_ns`.
    pub fn spin(
        request: SpinRequest,
        base: &Base,
        config: &NavigatorConfig,
        start: PlanarPose,
        now_ns: i64,
    ) -> Self {
        let turn = Turn::new(request, base, config, start.yaw(), now_ns);
        Self {
            motion: Motion::Turn(turn),
        }
    }

    /// The drive_on_heading (`Forward`) or back_up (`Backward`) of `request`
    /// for `base`, from `start` at `now_ns`.
    pub fn straight(
        direction: Direction,
        request: StraightRequest,
        base: &Base,
        config: &NavigatorConfig,
        start: PlanarPose,
        now_ns: i64,
    ) -> Self {
        let drive = StraightDrive::new(direction, request, base, config, start, now_ns);
        Self {
            motion: Motion::Straight(drive),
        }
    }

    fn feedback(&self, pose: PlanarPose) -> MoveFeedback {
        match &self.motion {
            Motion::Turn(turn) => MoveFeedback::AngleTurned(turn.turned_rad()),
            Motion::Straight(drive) => MoveFeedback::DistanceTraveled(drive.traveled_m(pose)),
        }
    }

    /// The step of `inputs`.
    pub fn step(&mut self, inputs: &MoveInputs) -> MoveStep {
        if let Motion::Turn(turn) = &mut self.motion {
            turn.measure(inputs.pose.yaw());
        }
        let feedback = self.feedback(inputs.pose);
        let motion_step = match &mut self.motion {
            Motion::Turn(turn) => {
                turn.step(inputs.pose, inputs.now_ns, inputs.grid, inputs.footprint)
            }
            Motion::Straight(drive) => {
                drive.step(inputs.pose, inputs.now_ns, inputs.grid, inputs.footprint)
            }
        };
        let (command, ended) = match motion_step {
            MotionStep::Command(command) => (Some(command), None),
            MotionStep::Done => (None, Some(Ended::Completed)),
            MotionStep::Blocked(why) => (
                None,
                Some(Ended::Failed(Failure::new(FailureKind::Blocked, why))),
            ),
            MotionStep::TimedOut(why) => (None, Some(Ended::TimedOut(why))),
        };
        MoveStep {
            command,
            feedback,
            ended,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use control_core::positive_finite::PositiveFinite;

    use crate::base::tests::base;
    use crate::config::tests::config;
    use crate::recovery::tests::grid;

    fn inputs<'a>(
        now_ns: i64,
        pose: PlanarPose,
        grid: &'a CostGrid,
        footprint: &'a PaddedFootprint,
    ) -> MoveInputs<'a> {
        MoveInputs {
            now_ns,
            pose,
            grid,
            footprint,
        }
    }

    // A drive_on_heading drives ahead until the odometry measures its
    // distance, and completes at that step with no command, as Nav2's
    // DriveOnHeading succeeds once it has traveled its distance.
    #[test]
    fn a_drive_on_heading_completes_at_the_step_the_odometry_measures_its_distance() {
        let (base, config, free) = (base(1.0), config(), grid(&[]));
        let footprint = &base.padded_footprint;
        let request = StraightRequest {
            distance_m: PositiveFinite::parse(0.5).unwrap(),
            speed_m_s: PositiveFinite::parse(0.3).unwrap(),
            timeout_ns: None,
        };
        let mut drive = Move::straight(
            Direction::Forward,
            request,
            &base,
            &config,
            PlanarPose::IDENTITY,
            0,
        );
        let ahead = drive.step(&inputs(0, PlanarPose::IDENTITY, &free, footprint));
        assert!(ahead.command.is_some_and(|c| c.linear_x() > 0.0) && ahead.ended.is_none());
        let short = PlanarPose::new(0.49, 0.0, 0.0).unwrap();
        let driving = drive.step(&inputs(1, short, &free, footprint));
        assert!(driving.command.is_some() && driving.ended.is_none());
        let there = PlanarPose::new(0.5, 0.0, 0.0).unwrap();
        let reached = drive.step(&inputs(2, there, &free, footprint));
        assert_eq!(
            (reached.command, reached.feedback, reached.ended),
            (
                None,
                MoveFeedback::DistanceTraveled(0.5),
                Some(Ended::Completed)
            )
        );
    }

    // A spin completes at the step the odometry measures its angle; one
    // whose timeout passes before its turn fails with timeout.
    #[test]
    fn a_spin_completes_at_its_angle_and_times_out_before_it() {
        let (base, config, free) = (base(1.0), config(), grid(&[]));
        let footprint = &base.padded_footprint;
        let short = SpinRequest {
            angle_rad: 1.0,
            timeout_ns: Some(1_000),
        };
        let mut late = Move::spin(short, &base, &config, PlanarPose::IDENTITY, 0);
        let ended = late
            .step(&inputs(1_001, PlanarPose::IDENTITY, &free, footprint))
            .ended;
        assert!(matches!(&ended, Some(Ended::TimedOut(_))), "{ended:?}");
        let unbounded = SpinRequest {
            angle_rad: 1.0,
            timeout_ns: None,
        };
        let mut spin = Move::spin(unbounded, &base, &config, PlanarPose::IDENTITY, 0);
        let turned = PlanarPose::new(0.0, 0.0, 1.0).unwrap();
        let step = spin.step(&inputs(0, turned, &free, footprint));
        assert_eq!((step.command, step.ended), (None, Some(Ended::Completed)));
    }
}
