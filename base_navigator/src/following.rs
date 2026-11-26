//! One follow of a planned path with navigation_core's ports of Nav2's
//! controller server, in its order (nav2_controller controller_server.cpp
//! `computeControl`, `computeAndPublishVelocity`): the path handler prunes
//! the plan to the poses ahead of the base and places them in odom, where the
//! grid of the floor is, the plan being in the frame its navigation runs in,
//! the progress checker judges the base's progress, regulated pure pursuit
//! gives the command, and the goal checker judges whether the base reached the
//! goal.
//!
//! A follow fails when the base makes no progress, and when the controller
//! gives no command that keeps the padded footprint off every occupied cell
//! along its arc within the arc look-ahead time
//! (`max_allowed_time_to_collision_up_to_carrot_s`) for longer than the
//! failure tolerance since the follow started or the newest command that did,
//! as the controller server's `failure_tolerance` bounds it; within that
//! tolerance the step has no command, and the base gets zero velocity. A
//! base or a plan off the grid, which the controller cannot follow, ends the
//! follow for another cause. Each plan the follow takes holds its own goal
//! checker, as the controller server resets its goal checker at each new
//! plan.

use control_core::non_negative_finite::NonNegativeFinite;
use control_core::planar_pose::PlanarPose;
use control_core::time::seconds;
use control_core::twist::PlanarTwist;
use navigation_core::controller::{ControlError, Lookahead, PurePursuit, PurePursuitParams};
use navigation_core::goal_checker::GoalCheckerParams;
use navigation_core::grid::CostGrid;
use navigation_core::inflation::Inflation;
use navigation_core::path_handler::{GlobalPlan, PathHandlerParams};
use navigation_core::progress_checker::ProgressChecker;

use crate::config::FollowSettings;
use crate::drive::NavigationRequest;
use crate::planning::Path;

/// What a step of a follow gives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Followed {
    /// The goal checker finds the goal reached.
    Reached,
    /// The command for the base, before the stop.
    Command(PlanarTwist),
    /// No command within the failure tolerance: the base gets zero velocity.
    NoCommand,
}

/// Why a follow ends.
#[derive(Clone, Debug, PartialEq)]
pub enum FollowEnd {
    /// The follow failed: no progress, or no command past the failure
    /// tolerance.
    Failed(String),
    /// The base or its plan lies off the grid: the navigation cannot go on.
    CannotGoOn(String),
}

/// A follow's controller, plan with its goal checker, and progress checker.
pub struct Follower {
    controller: PurePursuit,
    plan: GlobalPlan,
    progress: ProgressChecker,
    path_handler: PathHandlerParams,
    goal_checker: GoalCheckerParams,
    failure_tolerance_ns: i64,
    /// The start of the follow, or the step of its newest command.
    valid_since_ns: i64,
}

/// The plan of `path` with a goal checker that judges the position afresh.
fn plan_of(
    path: Path,
    path_handler: PathHandlerParams,
    goal_checker: GoalCheckerParams,
) -> GlobalPlan {
    GlobalPlan::new(path.into_poses(), path_handler, goal_checker).expect("a path holds a pose")
}

impl Follower {
    /// The follow of `path` for `request` from `now_ns`, the controller
    /// driving at the request's speed on grids that `inflation` inflated.
    pub fn new(
        path: Path,
        request: &NavigationRequest,
        settings: &FollowSettings,
        inflation: &Inflation,
        now_ns: i64,
    ) -> Self {
        let c = &settings.controller;
        let params = PurePursuitParams {
            max_linear_vel_m_s: request.speed_m_s,
            lookahead: Lookahead::VelocityScaled(c.lookahead),
            rotate_to_heading_angular_vel_rad_s: c.rotate_to_heading_angular_vel_rad_s,
            rotate_to_heading_min_angle_rad: c.rotate_to_heading_min_angle_rad,
            max_angular_accel_rad_s2: c.max_angular_accel_rad_s2,
            min_approach_linear_velocity_m_s: c.min_approach_linear_velocity_m_s,
            approach_velocity_scaling_dist_m: c.approach_velocity_scaling_dist_m,
            regulated_linear_scaling_min_radius_m: c.regulated_linear_scaling_min_radius_m,
            regulated_linear_scaling_min_speed_m_s: c.regulated_linear_scaling_min_speed_m_s,
            cost_scaling_dist_m: c.cost_scaling_dist_m,
            cost_scaling_gain: c.cost_scaling_gain,
            max_allowed_time_to_collision_up_to_carrot_s: c
                .max_allowed_time_to_collision_up_to_carrot_s,
            min_distance_to_obstacle_m: None,
            controller_frequency_hz: c.controller_frequency_hz,
        };
        let tolerance = |value: f64| {
            NonNegativeFinite::parse(value).expect("a goal's tolerance is a positive number")
        };
        let goal_checker = GoalCheckerParams {
            xy_goal_tolerance_m: tolerance(request.position_tolerance_m.get()),
            xy_goal_tolerance_buffer_m: tolerance(0.0),
            yaw_goal_tolerance_rad: tolerance(request.orientation_tolerance_rad.get()),
            path_length_tolerance_m: settings.path_length_tolerance_m,
        };
        Self {
            controller: PurePursuit::new(params, inflation),
            plan: plan_of(path, settings.path_handler, goal_checker),
            progress: ProgressChecker::new(settings.progress),
            path_handler: settings.path_handler,
            goal_checker,
            failure_tolerance_ns: settings.failure_tolerance_ns,
            valid_since_ns: now_ns,
        }
    }

    /// Follow `path`, a new plan's, from now on: the controller
    /// server's `setPlannerPath`.
    pub fn take_path(&mut self, path: Path) {
        self.plan = plan_of(path, self.path_handler, self.goal_checker);
    }

    /// The step at `now_ns` of the base at `pose`, moving at `velocity`, on
    /// `grid`, all in odom, the plan's frame standing in odom at
    /// `odom_from_plan`.
    pub fn step(
        &mut self,
        pose: PlanarPose,
        velocity: PlanarTwist,
        grid: &CostGrid,
        now_ns: i64,
        odom_from_plan: PlanarPose,
    ) -> Result<Followed, FollowEnd> {
        let pruned = self
            .plan
            .prune(pose, odom_from_plan, &grid.geometry())
            .map_err(|e| FollowEnd::CannotGoOn(e.to_string()))?;
        self.progress
            .check(pose, now_ns)
            .map_err(|e| FollowEnd::Failed(e.to_string()))?;
        let commanded =
            match self
                .controller
                .compute(pose, velocity, self.plan.goal_checker(), &pruned, grid)
            {
                Ok(command) => {
                    self.valid_since_ns = now_ns;
                    Followed::Command(command)
                }
                Err(e @ ControlError::NoValidControl) => {
                    let without_ns = now_ns - self.valid_since_ns;
                    if without_ns > self.failure_tolerance_ns {
                        return Err(FollowEnd::Failed(format!(
                            "{e}, for {} s, longer than failure_tolerance_s {} s",
                            seconds(without_ns),
                            seconds(self.failure_tolerance_ns)
                        )));
                    }
                    Followed::NoCommand
                }
                Err(e @ (ControlError::BaseOffGrid | ControlError::PlanNotInBaseFrame)) => {
                    return Err(FollowEnd::CannotGoOn(e.to_string()));
                }
            };
        if self.plan.goal_checker().is_goal_reached(pose, &pruned) {
            return Ok(Followed::Reached);
        }
        Ok(commanded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use control_core::positive_finite::PositiveFinite;

    use std::f64::consts::FRAC_PI_2;

    use crate::base::tests::base;
    use crate::config::tests::config;
    use crate::recovery::tests::grid;

    fn request() -> NavigationRequest {
        NavigationRequest {
            target: PlanarPose::new(1.0, 0.0, 0.0).unwrap(),
            position_tolerance_m: PositiveFinite::parse(0.1).unwrap(),
            orientation_tolerance_rad: PositiveFinite::parse(0.2).unwrap(),
            max_speed_m_s: None,
            speed_m_s: PositiveFinite::parse(0.5).unwrap(),
        }
    }

    /// A straight path along x from the origin to `to`, every 0.05 m.
    fn path(to: f64) -> Path {
        let poses = (0..=(to / 0.05).round() as usize)
            .map(|k| PlanarPose::new(0.05 * k as f64, 0.0, 0.0).unwrap())
            .collect();
        Path::new(poses).expect("a pose")
    }

    fn follower(failure_tolerance_ns: i64) -> Follower {
        let config = config();
        let settings = FollowSettings {
            failure_tolerance_ns,
            ..config.follow
        };
        Follower::new(path(1.0), &request(), &settings, &base(1.0).inflation, 0)
    }

    // A free path gets a command ahead, and a base within the tolerances at
    // the path's end has reached its goal.
    #[test]
    fn a_free_path_is_followed_ahead_and_reached_at_its_end() {
        let free = grid(&[]);
        let mut follower = follower(0);
        let first = follower
            .step(
                PlanarPose::IDENTITY,
                PlanarTwist::ZERO,
                &free,
                0,
                PlanarPose::IDENTITY,
            )
            .expect("a step");
        assert!(
            matches!(first, Followed::Command(command) if command.linear_x() > 0.0),
            "{first:?}"
        );
        let at_end = PlanarPose::new(0.97, 0.0, 0.05).unwrap();
        assert_eq!(
            follower.step(
                at_end,
                PlanarTwist::ZERO,
                &free,
                50_000_000,
                PlanarPose::IDENTITY
            ),
            Ok(Followed::Reached)
        );
    }

    // A box on the path ahead gives no valid command: within the failure
    // tolerance the step has none, and past it the follow fails.
    #[test]
    fn no_valid_command_past_the_failure_tolerance_fails_the_follow() {
        let walled = grid(&[[0.3, 0.0], [0.3, 0.05], [0.3, -0.05]]);
        let mut follower = follower(100_000_000);
        assert_eq!(
            follower.step(
                PlanarPose::IDENTITY,
                PlanarTwist::ZERO,
                &walled,
                50_000_000,
                PlanarPose::IDENTITY
            ),
            Ok(Followed::NoCommand)
        );
        assert!(matches!(
            follower.step(
                PlanarPose::IDENTITY,
                PlanarTwist::ZERO,
                &walled,
                150_000_000,
                PlanarPose::IDENTITY
            ),
            Err(FollowEnd::Failed(_))
        ));
    }

    // A path in a frame that stands in odom turned a quarter turn is followed
    // where it lies in odom: ahead along odom's y from where it starts, and
    // reached at its end placed in odom.
    #[test]
    fn a_path_in_another_frame_is_followed_where_it_lies_in_odom() {
        let free = grid(&[]);
        let mut follower = follower(0);
        let odom_from_plan = PlanarPose::new(0.0, 0.0, FRAC_PI_2).unwrap();
        let facing_y = PlanarPose::new(0.0, 0.0, FRAC_PI_2).unwrap();
        let first = follower
            .step(facing_y, PlanarTwist::ZERO, &free, 0, odom_from_plan)
            .expect("a step");
        assert!(
            matches!(first, Followed::Command(command)
                if command.linear_x() > 0.0 && command.angular_z().abs() < 0.1),
            "{first:?}"
        );
        let at_end = PlanarPose::new(0.0, 0.97, FRAC_PI_2 + 0.05).unwrap();
        assert_eq!(
            follower.step(at_end, PlanarTwist::ZERO, &free, 50_000_000, odom_from_plan),
            Ok(Followed::Reached)
        );
    }
}
