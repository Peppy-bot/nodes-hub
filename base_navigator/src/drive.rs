//! One navigation of the base to a pose, as base_navigation states it and as
//! Nav2's NavigateToPose runs its default behavior tree: tries, each a plan and
//! a follow, and the recoveries between them. A navigation reads time and the
//! world as arguments and touches no wire: the control loop hands each step its
//! inputs, runs the plans it asks for on the planner's task, empties the grid
//! when it asks, and sends what the step gives.
//!
//! A navigation runs in odom, or in map on room_map's loaded map ([`RunFrame`]).
//! Its goal, its path and its feedback are in the frame it runs in; it places
//! the path in odom where it reads the grid of the floor, which is kept there,
//! as Nav2's controller places its plan in its cost map's frame. Its turns and
//! back-ups, and the distance it drove, read the odometry.
//!
//! A try plans a path, or keeps the one it has while that path, from its pose
//! nearest the robot frame's origin to the goal, is shorter than the
//! near-goal distance and has no pose in collision or off the grid that
//! judges it, and follows it, planning again at the planning rate under the
//! same rule. That grid is the grid of the floor, with the newest readings,
//! where a pose lies on it; for a navigation in map, a pose off the grid of
//! the floor is judged on the grid in map the plan of its path ran on, which
//! each plan that finds a path replaces. So the newest readings count near
//! the base and the map of that plan beyond. Nav2's ValidatePath judges the
//! path left on the planner's global costmap, whose obstacle layer goes on
//! marking and whose static layer takes each new map at its next update
//! (nav2_costmap_2d plugins/static_layer.cpp:301-316, :380-382); the
//! navigator judges on a newer grid in map once a plan that finds a path has
//! run on it. A plan the planner refuses leaves the navigation no path to
//! keep, as Nav2's ComputePathToPose writes an empty path when it aborts
//! (nav2_behavior_tree plugins/action/compute_path_to_pose_action.cpp:71-77)
//! and IsGoalNearby fails on an empty path
//! (plugins/condition/is_goal_nearby_condition.cpp:45-47). The try that the
//! clearing recovery starts reads that rule on the grid the clearing empties,
//! where no pose is in collision, as Nav2's tree checks the path on the
//! cleared costmap (nav2_bt_navigator behavior_trees
//! navigate_to_pose_w_replanning_and_recovery.xml:19-26, :51-54), and follows
//! the path it keeps from the step after the clearing, on the emptied grid,
//! as Nav2's FollowPath (:36) runs once the ClearingActions end. A plan that
//! finds no path empties the grid and plans once more on the readings that
//! arrive after the clearing; a follow that fails empties the grid the same
//! way and follows the navigation's path once more. Its replanning runs on
//! meanwhile, as Nav2's tree ticks ComputePathToPose under its RateController
//! (:16-34) while FollowPath's RecoveryNode (:35-41) retries, both in a
//! PipelineSequence (:10-42) that ticks each child at each tick
//! (nav2_behavior_tree plugins/control/pipeline_sequence.cpp:36-60): a path a
//! plan finds becomes the retry's, and a refusal leaves the retry none, so
//! that it fails, as Nav2's controller server refuses an empty path to follow
//! (nav2_controller controller_server.cpp:483-486). A try fails when its plan
//! fails after its retry, when a plan finds the base or the goal in
//! collision, and when its following fails after its retry. When a try fails,
//! the next recovery runs when the newest follow had failed or the newest
//! plan found no path; otherwise the navigation fails: `start_in_collision`,
//! `unreachable`, or `attempt_failed`.
//! The recoveries run in [`RECOVERIES`]' order, each once: after the clearing,
//! the turn or the wait a new try starts, a turn that fails passes to the
//! wait, and once the back-up ends the navigation fails, `blocked` when the
//! newest follow had failed and `no_path` otherwise.
//!
//! The navigation completes at the step the goal checker finds its goal
//! reached, as Nav2's controller ends FollowPath there (nav2_controller
//! plugins/simple_goal_checker.cpp:118-130).

use std::sync::Arc;

use control_core::planar_pose::{PlanarPose, PlanarPoseError};
use control_core::positive_finite::PositiveFinite;
use control_core::twist::PlanarTwist;
use frame_tree::arithmetic::normalize_angle;
use navigation_core::PaddedFootprint;
use navigation_core::grid::{CostGrid, INSCRIBED, LETHAL};
use navigation_core::planner::PlanError;

use crate::base::Base;
use crate::config::NavigatorConfig;
use crate::failure::{Ended, FailureKind, NavigationFailure, NavigationKind};
use crate::following::{FollowEnd, Followed, Follower};
use crate::planning::{Path, PlanDone, PlanRequest, Planned};
use crate::recovery::{
    Direction, MotionStep, RECOVERIES, Recovery, SpinRequest, StraightDrive, StraightRequest, Turn,
};
use crate::room_map::MapRun;

/// The measured speed (m/s) at or below which the time left is 0.
const SLOWEST_ESTIMATED_SPEED_M_S: f64 = 0.01;

/// The distance left (m) at or below which the time left is 0.
const NEAREST_ESTIMATED_DISTANCE_M: f64 = 0.1;

/// The frame a navigation runs in: odom, for a target in any frame but map,
/// or map, on room_map's loaded map.
#[derive(Clone, Debug, PartialEq)]
pub enum RunFrame {
    Odometry,
    Map(MapRun),
}

/// A navigation the checks admit: its request, and the frame it runs in,
/// which the control loop places it by.
#[derive(Clone, Debug, PartialEq)]
pub struct NavigationGoal {
    pub request: NavigationRequest,
    pub frame: RunFrame,
}

/// A navigation's goal, as the navigation takes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationRequest {
    /// The target in the frame the navigation runs in, its yaw in [-pi, pi].
    pub target: PlanarPose,
    pub position_tolerance_m: PositiveFinite,
    pub orientation_tolerance_rad: PositiveFinite,
    /// The goal's max_speed_m_s, none for 0: the fastest the base may move,
    /// as the norm of its linear_velocity.
    pub max_speed_m_s: Option<PositiveFinite>,
    /// The controller's speed: the base's forward limit, at most
    /// `max_speed_m_s`.
    pub speed_m_s: PositiveFinite,
}

/// Where the robot frame stands at a step: in odom, and in the frame its
/// navigation runs in, which stands in odom at `odom_from_run`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasePose {
    in_odom: PlanarPose,
    in_run: PlanarPose,
    odom_from_run: PlanarPose,
}

impl BasePose {
    /// The robot frame at `in_odom`, for a navigation that runs in odom.
    pub fn in_odometry(in_odom: PlanarPose) -> Self {
        Self {
            in_odom,
            in_run: in_odom,
            odom_from_run: PlanarPose::IDENTITY,
        }
    }

    /// The robot frame at `in_odom`, for a navigation that runs in a frame
    /// that stands in odom at `odom_from_run`; refused for a place in that
    /// frame that is not finite.
    pub fn in_frame(
        in_odom: PlanarPose,
        odom_from_run: PlanarPose,
    ) -> Result<Self, PlanarPoseError> {
        Ok(Self {
            in_odom,
            in_run: in_odom.seen_from(odom_from_run)?,
            odom_from_run,
        })
    }

    /// `pose`, a pose in the frame the navigation runs in, in odom; none for
    /// a place in odom that is not finite.
    fn to_odom(self, pose: PlanarPose) -> Option<PlanarPose> {
        self.odom_from_run.compose(pose).ok()
    }
}

/// What the base and the grid report at a step, as a navigation reads them.
#[derive(Clone, Debug)]
pub struct NavigationInputs<'a> {
    pub now_ns: i64,
    /// Where the newest drive_states places the robot frame.
    pub base: BasePose,
    pub velocity: PlanarTwist,
    /// The grid of the floor's occupancy, in odom.
    pub grid: &'a CostGrid,
    pub footprint: &'a PaddedFootprint,
    /// Whether every sensor's reading went into the grid since its newest
    /// clearing.
    pub read_since_clearing: bool,
    /// The plan that finished since the step before, if one did.
    pub plan: Option<PlanDone>,
}

/// What a navigation reports each step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavigationFeedback {
    pub distance_remaining_m: f64,
    pub angle_remaining_rad: f64,
    pub estimated_time_remaining_s: f64,
    pub recoveries: u32,
}

/// What one step of a navigation gives.
#[derive(Clone, Debug, PartialEq)]
pub struct NavigationStep {
    /// The command of the step, before the stop; none while it plans before
    /// a follow, at the step the clearing recovery starts, and once it ends.
    pub command: Option<PlanarTwist>,
    /// Empty the grid before the next step.
    pub clear_grid: bool,
    /// Start this plan on the planner's task.
    pub plan: Option<PlanRequest>,
    pub feedback: NavigationFeedback,
    pub ended: Option<Ended<NavigationKind>>,
}

/// What the newest plan of a navigation found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NewestPlan {
    None,
    Path,
    NoPath,
    StartInCollision,
    GoalInCollision,
    OffGrid,
}

/// What a step asks of the loop: its command, before the stop, a clearing,
/// a plan, and the navigation's ending.
#[derive(Default)]
struct StepAsks {
    command: Option<PlanarTwist>,
    clear_grid: bool,
    plan: Option<PlanRequest>,
    ended: Option<Ended<NavigationKind>>,
}

/// What a navigation does now.
enum Phase {
    /// A try's plan: it waits for every sensor's reading after a clearing
    /// while `after_clearing`, then asks, then waits for the plan it asked;
    /// `retry` once the plan's one retry runs.
    Planning {
        retry: bool,
        after_clearing: bool,
        asked: Option<u64>,
    },
    /// A try's follow: the follower, whether the follow's one retry ran, and
    /// the replanning while it follows.
    Following {
        follower: Box<Follower>,
        retried: bool,
        replan: Replan,
    },
    /// A failed follow's retry: its replanning runs on, and once every
    /// sensor's reading after the clearing is in, it follows the
    /// navigation's path once more.
    RetryingFollow {
        replan: Replan,
    },
    /// The try of the clearing recovery that keeps its path: it takes no
    /// follow step at the step that clears, and follows the path from the
    /// next step on, on the grid the clearing emptied.
    FollowAfterClearing(Path),
    Turning(Turn),
    Waiting {
        until_ns: i64,
    },
    BackingUp(StraightDrive),
    /// The navigation ended: its steps ask for nothing.
    Ended,
}

/// The replanning of a follow.
enum Replan {
    /// The next plan is due at `due_ns`.
    Due { due_ns: i64 },
    /// A plan runs; `retry` for the plan's one retry.
    Asked { id: u64, retry: bool },
    /// A plan found no path: the grid is emptied, and the retry waits for
    /// every sensor's reading after the clearing.
    AwaitingReadings,
}

/// Whether `pose`, in `grid`'s frame, is in collision on `grid`: an occupied
/// cell lies within the padded footprint's inscribed radius of it, which the
/// inflated grid marks at least inscribed; a pose off the grid counts in
/// collision, as Nav2's ValidatePath counts a pose off its costmap lethal
/// (nav2_planner is_path_valid_service.hpp:262-266), and so does one whose
/// place in that frame is not finite.
fn in_collision(grid: &CostGrid, pose: Option<PlanarPose>) -> bool {
    pose.and_then(|pose| grid.geometry().cell_of([pose.x(), pose.y()]))
        .is_none_or(|cell| matches!(grid.cost(cell), INSCRIBED | LETHAL))
}

/// Whether `pose`, in `grid`'s frame, lies off `grid`, as one whose place in
/// that frame is not finite does.
fn off_grid(grid: &CostGrid, pose: Option<PlanarPose>) -> bool {
    pose.and_then(|pose| grid.geometry().cell_of([pose.x(), pose.y()]))
        .is_none()
}

/// Whether every sensor's reading went into the grid since its newest
/// clearing, at `inputs` and after the clearing `step` asks for: the loop
/// empties the grid after the step, so a step that clears has no reading
/// since that clearing.
fn read_since_clearing(inputs: &NavigationInputs, step: &StepAsks) -> bool {
    inputs.read_since_clearing && !step.clear_grid
}

/// The path a navigation has, and, for a navigation in map, the grid in map
/// its plan ran on, which judges the poses of the path off the grid of the
/// floor when the navigation decides whether it keeps the path.
#[derive(Clone, Debug)]
struct HeldPath {
    path: Path,
    map_grid: Option<Arc<CostGrid>>,
}

/// A navigation in flight.
pub struct Navigation {
    request: NavigationRequest,
    base: Arc<Base>,
    config: Arc<NavigatorConfig>,
    phase: Phase,
    /// The path the navigation has, none before its first plan and after a
    /// plan the planner refuses.
    path: Option<HeldPath>,
    /// How many of [`RECOVERIES`] have run.
    recoveries_run: usize,
    /// The clearings, turns, waits and back-ups started, the clearing of each
    /// retry included.
    recoveries: u32,
    newest_plan: NewestPlan,
    newest_follow_failed: bool,
    next_plan_id: u64,
    driven_m: f64,
    last_position: PlanarPose,
}

impl Navigation {
    /// The navigation of `request` for `base`, the robot frame at `start` in
    /// odom: its first try plans at its first step.
    pub fn new(
        request: NavigationRequest,
        base: Arc<Base>,
        config: Arc<NavigatorConfig>,
        start: PlanarPose,
    ) -> Self {
        Self {
            request,
            base,
            config,
            phase: Phase::Planning {
                retry: false,
                after_clearing: false,
                asked: None,
            },
            path: None,
            recoveries_run: 0,
            recoveries: 0,
            newest_plan: NewestPlan::None,
            newest_follow_failed: false,
            next_plan_id: 0,
            driven_m: 0.0,
            last_position: start,
        }
    }

    /// How far (m) the base drove since the navigation was accepted, by its
    /// odometry.
    pub fn driven_m(&self) -> f64 {
        self.driven_m
    }

    fn feedback(&self, pose: PlanarPose, speed_m_s: f64) -> NavigationFeedback {
        let target = self.request.target;
        let distance_remaining_m = self.path.as_ref().map_or_else(
            || pose.distance_to(target),
            |held| held.path.remaining_from(pose),
        );
        let estimated_time_remaining_s = if speed_m_s <= SLOWEST_ESTIMATED_SPEED_M_S
            || distance_remaining_m <= NEAREST_ESTIMATED_DISTANCE_M
        {
            0.0
        } else {
            distance_remaining_m / speed_m_s
        };
        NavigationFeedback {
            distance_remaining_m,
            angle_remaining_rad: normalize_angle(target.yaw() - pose.yaw()).abs(),
            estimated_time_remaining_s,
            recoveries: self.recoveries,
        }
    }

    /// The path the navigation keeps with the base at `base`, if it keeps the
    /// one it has: shorter than the near-goal distance from its pose nearest
    /// the robot frame's origin, with no pose from there on in collision where
    /// [`Navigation::kept_path_judged`] judges it, as Nav2's ValidatePath
    /// checks the path left on the planner's global costmap
    /// (nav2_bt_navigator navigate_to_pose_w_replanning_and_recovery.xml:23-25).
    fn kept_path(&self, base: BasePose, floor: &CostGrid) -> Option<&Path> {
        self.kept_path_judged(base, floor, in_collision)
    }

    /// The path the navigation keeps with the base at `base` once a clearing
    /// empties the grid of the floor, where no pose is in collision, if it
    /// keeps the one it has: shorter than the near-goal distance from its pose
    /// nearest the robot frame's origin, with every pose from there on on the
    /// grid that judges it.
    fn kept_path_after_clearing(&self, base: BasePose, floor: &CostGrid) -> Option<&Path> {
        self.kept_path_judged(base, floor, off_grid)
    }

    /// The path the navigation keeps with the base at `base` while no pose of
    /// it from its pose nearest the robot frame's origin on is `blocked` on
    /// the grid that judges the pose. That is `floor`, the grid of the floor in
    /// odom, with the newest readings, where the pose lies on it. For a
    /// navigation in map, a pose off `floor` is judged on the grid in map its
    /// plan ran on, which holds the map beyond the grid of the floor.
    fn kept_path_judged(
        &self,
        base: BasePose,
        floor: &CostGrid,
        blocked: fn(&CostGrid, Option<PlanarPose>) -> bool,
    ) -> Option<&Path> {
        let map = self.path.as_ref()?.map_grid.as_deref();
        self.kept_path_unless(base, |on_path| {
            let in_odom = base.to_odom(on_path);
            match map {
                Some(map) if off_grid(floor, in_odom) => blocked(map, Some(on_path)),
                _ => blocked(floor, in_odom),
            }
        })
    }

    /// The path the navigation has, if it is shorter than the near-goal
    /// distance from its pose nearest the robot frame's origin at `base`, with
    /// no pose from there on that `blocks`.
    fn kept_path_unless(
        &self,
        base: BasePose,
        blocks: impl Fn(PlanarPose) -> bool,
    ) -> Option<&Path> {
        let path = &self.path.as_ref()?.path;
        let nearest = path.nearest_index(base.in_run);
        let kept = path.length_from(nearest) < self.config.near_goal_distance_m.get()
            && !path.poses()[nearest..]
                .iter()
                .any(|&on_path| blocks(on_path));
        kept.then_some(path)
    }

    /// A plan from the robot frame at `base` to the goal, in the frame the
    /// navigation runs in.
    fn ask(&mut self, base: BasePose) -> PlanRequest {
        let id = self.next_plan_id;
        self.next_plan_id += 1;
        PlanRequest {
            id,
            start: base.in_run,
            goal: self.request.target,
        }
    }

    /// The replan due a planning period after `now_ns`.
    fn replan_due(&self, now_ns: i64) -> Replan {
        Replan::Due {
            due_ns: now_ns + self.config.replan_period_ns,
        }
    }

    /// A try's follow of `path` from `inputs`, its first replan due a
    /// planning period on.
    fn follow(&mut self, path: Path, inputs: &NavigationInputs) -> Phase {
        let replan = self.replan_due(inputs.now_ns);
        self.follow_with(path, inputs, false, replan)
    }

    /// The follow of `path` from `inputs` with the replanning `replan`,
    /// `retried` for the follow's retry.
    fn follow_with(
        &mut self,
        path: Path,
        inputs: &NavigationInputs,
        retried: bool,
        replan: Replan,
    ) -> Phase {
        self.newest_follow_failed = false;
        Phase::Following {
            follower: Box::new(Follower::new(
                path,
                &self.request,
                &self.config.follow,
                &self.base.inflation,
                inputs.now_ns,
            )),
            retried,
            replan,
        }
    }

    /// The start of a try at `inputs`: the path it keeps, followed at once,
    /// or a plan.
    fn start_try(&mut self, inputs: &NavigationInputs) -> Phase {
        if let Some(path) = self.kept_path(inputs.base, inputs.grid).cloned() {
            return self.follow(path, inputs);
        }
        Phase::Planning {
            retry: false,
            after_clearing: false,
            asked: None,
        }
    }

    /// The start of the clearing recovery's try at `inputs`, the step that
    /// clears: the path it keeps, followed from the next step on, or a plan
    /// on the readings after the clearing.
    fn start_try_after_clearing(&self, inputs: &NavigationInputs) -> Phase {
        if let Some(path) = self.kept_path_after_clearing(inputs.base, inputs.grid) {
            return Phase::FollowAfterClearing(path.clone());
        }
        Phase::Planning {
            retry: false,
            after_clearing: true,
            asked: None,
        }
    }

    /// The failure of the navigation with `kind` and `message`.
    fn failed(
        kind: impl Into<NavigationKind>,
        message: impl Into<String>,
    ) -> Ended<NavigationKind> {
        Ended::Failed(NavigationFailure::new(kind.into(), message))
    }

    /// What a try that failed for `why` runs: the next recovery, as its
    /// phase, when the newest follow had failed or the newest plan found no
    /// path, or the navigation's failure.
    fn try_failed(
        &mut self,
        why: String,
        inputs: &NavigationInputs,
        step: &mut StepAsks,
    ) -> Option<Phase> {
        if !(self.newest_follow_failed || self.newest_plan == NewestPlan::NoPath) {
            let kind = match self.newest_plan {
                NewestPlan::StartInCollision => NavigationKind::StartInCollision,
                NewestPlan::GoalInCollision => NavigationKind::Unreachable,
                _ => FailureKind::AttemptFailed.into(),
            };
            step.ended = Some(Self::failed(kind, why));
            return None;
        }
        self.recover(why, inputs, step)
    }

    /// The next recovery's phase, the reason being why the try failed, or the
    /// navigation's failure once the back-up has run.
    fn recover(
        &mut self,
        why: String,
        inputs: &NavigationInputs,
        step: &mut StepAsks,
    ) -> Option<Phase> {
        let Some(&recovery) = RECOVERIES.get(self.recoveries_run) else {
            step.ended = Some(self.after_back_up(why));
            return None;
        };
        self.recoveries_run += 1;
        self.recoveries += 1;
        let recoveries = self.config.recoveries;
        match recovery {
            Recovery::Clear => {
                step.clear_grid = true;
                Some(self.start_try_after_clearing(inputs))
            }
            Recovery::Turn => {
                let turn = SpinRequest {
                    angle_rad: recoveries.turn_angle_rad.get(),
                    timeout_ns: Some(recoveries.turn_time_allowance_ns),
                };
                Some(Phase::Turning(Turn::new(
                    turn,
                    &self.base,
                    &self.config,
                    inputs.base.in_odom.yaw(),
                    inputs.now_ns,
                )))
            }
            Recovery::Wait => Some(Phase::Waiting {
                until_ns: inputs.now_ns + recoveries.wait_ns,
            }),
            Recovery::BackUp => {
                if Direction::Backward.fastest_m_s(&self.base) == 0.0 {
                    step.ended = Some(self.after_back_up(format!(
                        "{why}; the base drives no velocity backward, so the back-up fails at once"
                    )));
                    return None;
                }
                let back_up = StraightRequest {
                    distance_m: recoveries.back_up_distance_m,
                    speed_m_s: recoveries.back_up_speed_m_s,
                    timeout_ns: Some(recoveries.back_up_time_allowance_ns),
                };
                let drive = StraightDrive::new(
                    Direction::Backward,
                    back_up,
                    &self.base,
                    &self.config,
                    inputs.base.in_odom,
                    inputs.now_ns,
                );
                Some(Phase::BackingUp(
                    self.request
                        .max_speed_m_s
                        .map_or(drive, |cap| drive.at_most(cap.get())),
                ))
            }
        }
    }

    /// The navigation's failure once its back-up ended: `blocked` when the
    /// newest follow had failed, and `no_path` otherwise.
    fn after_back_up(&self, why: String) -> Ended<NavigationKind> {
        let message = format!("after the clearing, the turn, the wait and the back-up: {why}");
        if self.newest_follow_failed {
            return Self::failed(FailureKind::Blocked, message);
        }
        Self::failed(NavigationKind::NoPath, message)
    }

    /// Take `done`, a plan of this navigation, as its newest plan: the path it
    /// found, which becomes the navigation's with the grid in map it ran on, or
    /// why it gives none. A
    /// refusal leaves the navigation no path, as Nav2's ComputePathToPose
    /// writes an empty path when it aborts.
    fn take_plan(&mut self, done: PlanDone) -> Result<Path, PlanTaken> {
        let (newest, why) = match done.planned {
            Planned::Path(path) => {
                self.newest_plan = NewestPlan::Path;
                self.path = Some(HeldPath {
                    path: path.clone(),
                    map_grid: done.map_grid,
                });
                return Ok(path);
            }
            Planned::Lost => {
                return Err(PlanTaken::Unplanned(
                    "the planner's task ended with no plan".to_string(),
                ));
            }
            Planned::GridRefused(why) => return Err(PlanTaken::Unplanned(why)),
            Planned::Refused(e @ PlanError::NoPath) => (NewestPlan::NoPath, e),
            Planned::Refused(e @ PlanError::StartOccupied(_)) => (NewestPlan::StartInCollision, e),
            Planned::Refused(e @ PlanError::GoalOccupied(_)) => (NewestPlan::GoalInCollision, e),
            Planned::Refused(e @ (PlanError::StartOffGrid(_) | PlanError::GoalOffGrid(_))) => {
                (NewestPlan::OffGrid, e)
            }
        };
        self.newest_plan = newest;
        self.path = None;
        Err(PlanTaken::Refused(why.to_string()))
    }

    /// The step of `inputs`.
    pub fn step(&mut self, inputs: NavigationInputs) -> NavigationStep {
        let in_odom = inputs.base.in_odom;
        self.driven_m += self.last_position.distance_to(in_odom);
        self.last_position = in_odom;
        let speed_m_s = inputs.velocity.linear_x().hypot(inputs.velocity.linear_y());
        let mut asks = StepAsks::default();
        let phase = std::mem::replace(&mut self.phase, Phase::Ended);
        self.phase = self
            .advance(phase, &inputs, &mut asks)
            .unwrap_or(Phase::Ended);
        let ended = asks.ended.is_some();
        NavigationStep {
            command: asks.command.filter(|_| !ended),
            clear_grid: asks.clear_grid,
            plan: asks.plan,
            // The step may have taken a new path, from which the feedback
            // reads the length left.
            feedback: self.feedback(inputs.base.in_run, speed_m_s),
            ended: asks.ended,
        }
    }

    /// The phase after `phase` at `inputs`, the step's command and asks
    /// written into `step`; none once the navigation ends. A phase that ends
    /// within the step hands over to the next in the same step.
    fn advance(
        &mut self,
        mut phase: Phase,
        inputs: &NavigationInputs,
        step: &mut StepAsks,
    ) -> Option<Phase> {
        loop {
            phase = match phase {
                Phase::Planning {
                    retry,
                    after_clearing,
                    asked,
                } => {
                    if after_clearing && !read_since_clearing(inputs, step) {
                        return Some(Phase::Planning {
                            retry,
                            after_clearing,
                            asked,
                        });
                    }
                    let Some(id) = asked else {
                        let request = self.ask(inputs.base);
                        step.plan = Some(request);
                        return Some(Phase::Planning {
                            retry,
                            after_clearing: false,
                            asked: Some(request.id),
                        });
                    };
                    let Some(done) = inputs.plan.clone().filter(|done| done.id == id) else {
                        return Some(Phase::Planning {
                            retry,
                            after_clearing,
                            asked,
                        });
                    };
                    match self.take_plan(done) {
                        Ok(path) => self.follow(path, inputs),
                        Err(PlanTaken::Unplanned(why)) => {
                            step.ended = Some(Self::failed(FailureKind::AttemptFailed, why));
                            return None;
                        }
                        Err(PlanTaken::Refused(_))
                            if self.newest_plan == NewestPlan::NoPath && !retry =>
                        {
                            step.clear_grid = true;
                            self.recoveries += 1;
                            Phase::Planning {
                                retry: true,
                                after_clearing: true,
                                asked: None,
                            }
                        }
                        Err(PlanTaken::Refused(why)) => self.try_failed(why, inputs, step)?,
                    }
                }
                Phase::Following {
                    follower,
                    retried,
                    replan,
                } => return self.following(follower, retried, replan, inputs, step),
                Phase::RetryingFollow { replan } => {
                    return self.retrying_follow(replan, inputs, step);
                }
                Phase::FollowAfterClearing(path) => {
                    if step.clear_grid {
                        return Some(Phase::FollowAfterClearing(path));
                    }
                    self.follow(path, inputs)
                }
                Phase::Turning(turn) => {
                    let mut turn = turn;
                    let in_odom = inputs.base.in_odom;
                    turn.measure(in_odom.yaw());
                    match turn.step(in_odom, inputs.now_ns, inputs.grid, inputs.footprint) {
                        MotionStep::Command(command) => {
                            step.command = Some(command);
                            return Some(Phase::Turning(turn));
                        }
                        MotionStep::Done => self.start_try(inputs),
                        // A turn that fails passes straight to the wait, the
                        // recovery after it.
                        MotionStep::Blocked(why) | MotionStep::TimedOut(why) => {
                            self.recover(why, inputs, step)?
                        }
                    }
                }
                Phase::Waiting { until_ns } => {
                    if inputs.now_ns < until_ns {
                        step.command = Some(PlanarTwist::ZERO);
                        return Some(Phase::Waiting { until_ns });
                    }
                    self.start_try(inputs)
                }
                Phase::BackingUp(mut drive) => {
                    let in_odom = inputs.base.in_odom;
                    match drive.step(in_odom, inputs.now_ns, inputs.grid, inputs.footprint) {
                        MotionStep::Command(command) => {
                            step.command = Some(command);
                            return Some(Phase::BackingUp(drive));
                        }
                        MotionStep::Done => {
                            step.ended = Some(
                                self.after_back_up("the back-up covered its distance".to_string()),
                            );
                            return None;
                        }
                        MotionStep::Blocked(why) | MotionStep::TimedOut(why) => {
                            step.ended = Some(self.after_back_up(format!("the back-up: {why}")));
                            return None;
                        }
                    }
                }
                Phase::Ended => return Some(Phase::Ended),
            };
        }
    }

    /// The step of a follow at `inputs`: its replanning first, then the
    /// follower's command.
    fn following(
        &mut self,
        mut follower: Box<Follower>,
        retried: bool,
        replan: Replan,
        inputs: &NavigationInputs,
        step: &mut StepAsks,
    ) -> Option<Phase> {
        let replan = match self.replanned(replan, inputs, step) {
            Ok((replan, found)) => {
                if let Some(path) = found {
                    follower.take_path(path);
                }
                replan
            }
            Err(why) => {
                let phase = self.try_failed(why, inputs, step)?;
                return self.advance(phase, inputs, step);
            }
        };
        let base = inputs.base;
        match follower.step(
            base.in_odom,
            inputs.velocity,
            inputs.grid,
            inputs.now_ns,
            base.odom_from_run,
        ) {
            Ok(Followed::Command(command)) => step.command = Some(command),
            Ok(Followed::NoCommand) => step.command = Some(PlanarTwist::ZERO),
            Ok(Followed::Reached) => {
                step.ended = Some(Ended::Completed);
                return None;
            }
            Err(FollowEnd::CannotGoOn(why)) => {
                step.ended = Some(Self::failed(FailureKind::AttemptFailed, why));
                return None;
            }
            Err(FollowEnd::Failed(why)) => {
                self.newest_follow_failed = true;
                if retried {
                    let phase = self.try_failed(why, inputs, step)?;
                    return self.advance(phase, inputs, step);
                }
                step.clear_grid = true;
                self.recoveries += 1;
                step.command = Some(PlanarTwist::ZERO);
                return Some(Phase::RetryingFollow { replan });
            }
        }
        Some(Phase::Following {
            follower,
            retried,
            replan,
        })
    }

    /// The step of a failed follow's retry at `inputs`: its replanning
    /// first, whose plans change the navigation's path, then, once every
    /// sensor's reading after the clearing is in, the follow of that path
    /// once more; with no path, the try fails.
    fn retrying_follow(
        &mut self,
        replan: Replan,
        inputs: &NavigationInputs,
        step: &mut StepAsks,
    ) -> Option<Phase> {
        let replan = match self.replanned(replan, inputs, step) {
            Ok((replan, _)) => replan,
            Err(why) => {
                let phase = self.try_failed(why, inputs, step)?;
                return self.advance(phase, inputs, step);
            }
        };
        if !read_since_clearing(inputs, step) {
            step.command = Some(PlanarTwist::ZERO);
            return Some(Phase::RetryingFollow { replan });
        }
        let phase = match self.path.clone() {
            Some(held) => self.follow_with(held.path, inputs, true, replan),
            None => self.try_failed(
                "the follow's retry has no path: its replan found none".to_string(),
                inputs,
                step,
            )?,
        };
        self.advance(phase, inputs, step)
    }

    /// The replanning of a follow at `inputs`, and the path a plan found at
    /// it: a plan asked when one is due and the path is not kept, a plan
    /// taken into the navigation's path, a no-path's clearing and retry, or
    /// why the try fails.
    fn replanned(
        &mut self,
        replan: Replan,
        inputs: &NavigationInputs,
        step: &mut StepAsks,
    ) -> Result<(Replan, Option<Path>), String> {
        let due = self.replan_due(inputs.now_ns);
        match replan {
            Replan::Due { due_ns } if inputs.now_ns < due_ns => Ok((Replan::Due { due_ns }, None)),
            Replan::Due { .. } if self.kept_path(inputs.base, inputs.grid).is_some() => {
                Ok((due, None))
            }
            Replan::Due { .. } => {
                let request = self.ask(inputs.base);
                step.plan = Some(request);
                let asked = Replan::Asked {
                    id: request.id,
                    retry: false,
                };
                Ok((asked, None))
            }
            Replan::AwaitingReadings if !read_since_clearing(inputs, step) => {
                Ok((Replan::AwaitingReadings, None))
            }
            Replan::AwaitingReadings => {
                let request = self.ask(inputs.base);
                step.plan = Some(request);
                let asked = Replan::Asked {
                    id: request.id,
                    retry: true,
                };
                Ok((asked, None))
            }
            Replan::Asked { id, retry } => {
                let Some(done) = inputs.plan.clone().filter(|done| done.id == id) else {
                    return Ok((Replan::Asked { id, retry }, None));
                };
                match self.take_plan(done) {
                    Ok(path) => Ok((due, Some(path))),
                    Err(PlanTaken::Unplanned(why)) => Err(why),
                    Err(PlanTaken::Refused(_))
                        if self.newest_plan == NewestPlan::NoPath && !retry =>
                    {
                        step.clear_grid = true;
                        self.recoveries += 1;
                        Ok((Replan::AwaitingReadings, None))
                    }
                    Err(PlanTaken::Refused(why)) => Err(why),
                }
            }
        }
    }
}

/// Why a plan gives no path: the planner refused it, or no plan ran.
enum PlanTaken {
    Refused(String),
    Unplanned(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    use navigation_core::grid::{FREE, GridGeometry};
    use navigation_core::planner::plan;

    use crate::base::tests::base;
    use crate::config::tests::config;

    const STEP_NS: i64 = 50_000_000;

    /// The goal of every navigation here, 1.5 m ahead of the start.
    fn goal() -> PlanarPose {
        PlanarPose::new(1.5, 0.0, 0.0).unwrap()
    }

    fn navigation() -> Navigation {
        let request = NavigationRequest {
            target: goal(),
            position_tolerance_m: PositiveFinite::parse(0.1).unwrap(),
            orientation_tolerance_rad: PositiveFinite::parse(0.2).unwrap(),
            max_speed_m_s: None,
            speed_m_s: PositiveFinite::parse(0.5).unwrap(),
        };
        Navigation::new(
            request,
            Arc::new(base(1.0)),
            Arc::new(config()),
            PlanarPose::IDENTITY,
        )
    }

    /// A grid of 0.05 m cells 4 m on a side around the origin, its `lethal`
    /// cells occupied and inflated for the tests' base.
    fn grid(lethal: impl Fn([f64; 2]) -> bool) -> CostGrid {
        grid_from([-2.0, -2.0], lethal)
    }

    /// A grid of 0.05 m cells 4 m on a side from `corner`, its lowest x and
    /// y, its `lethal` cells occupied and inflated for the tests' base.
    fn grid_from(corner: [f64; 2], lethal: impl Fn([f64; 2]) -> bool) -> CostGrid {
        let geometry =
            GridGeometry::new(80, 80, PositiveFinite::parse(0.05).unwrap(), corner).unwrap();
        let costs = geometry
            .cells()
            .map(|cell| {
                if lethal(geometry.center_of(cell)) {
                    LETHAL
                } else {
                    FREE
                }
            })
            .collect();
        let mut grid = CostGrid::from_costs(geometry, costs).unwrap();
        base(1.0).inflation.inflate(&mut grid).unwrap();
        grid
    }

    /// A straight path from the origin to the goal, every 0.05 m.
    fn straight_to_the_goal() -> Path {
        let poses = (0..=30)
            .map(|k| PlanarPose::new(0.05 * f64::from(k), 0.0, 0.0).unwrap())
            .collect();
        Path::new(poses).expect("31 poses")
    }

    /// The base 0.45 m from the goal along [`straight_to_the_goal`], nearer
    /// than the near-goal distance.
    fn near() -> PlanarPose {
        PlanarPose::new(1.05, 0.0, 0.0).unwrap()
    }

    /// Whether `[x, y]` lies in a wall across [`straight_to_the_goal`]
    /// between [`near`] and the goal.
    fn wall_ahead([x, y]: [f64; 2]) -> bool {
        (1.3..1.4).contains(&x) && y.abs() < 0.5
    }

    /// A grid like [`grid`]'s whose edge, at x 1.4 m, falls short of the
    /// goal.
    fn cut_short(lethal: impl Fn([f64; 2]) -> bool) -> CostGrid {
        grid_from([-2.6, -2.0], lethal)
    }

    /// A grid whose ring of occupied cells 0.6 m to 0.7 m from the goal
    /// shuts the goal in.
    fn penned() -> CostGrid {
        grid(|[x, y]| (0.6..0.7).contains(&(x - goal().x()).hypot(y - goal().y())))
    }

    /// The step at `now_ns` of the base resting at the origin on `grid`,
    /// every sensor's reading in since the newest clearing when
    /// `read_since_clearing`, with the plan `plan` finished.
    fn step(
        navigation: &mut Navigation,
        now_ns: i64,
        grid: &CostGrid,
        plan: Option<PlanDone>,
        read_since_clearing: bool,
    ) -> NavigationStep {
        step_at(
            navigation,
            now_ns,
            PlanarPose::IDENTITY,
            grid,
            plan,
            read_since_clearing,
        )
    }

    /// The step at `now_ns` of the base resting at `pose` on `grid`, as
    /// [`step`] gives it.
    fn step_at(
        navigation: &mut Navigation,
        now_ns: i64,
        pose: PlanarPose,
        grid: &CostGrid,
        plan: Option<PlanDone>,
        read_since_clearing: bool,
    ) -> NavigationStep {
        let footprint = base(1.0).padded_footprint;
        navigation.step(NavigationInputs {
            now_ns,
            base: BasePose::in_odometry(pose),
            velocity: PlanarTwist::ZERO,
            grid,
            footprint: &footprint,
            read_since_clearing,
            plan,
        })
    }

    /// The planner's refusal of `request`: it finds no path.
    fn no_path(request: PlanRequest) -> Option<PlanDone> {
        Some(PlanDone {
            id: request.id,
            planned: Planned::Refused(PlanError::NoPath),
            map_grid: None,
        })
    }

    /// The steps from `now_ns`, the base resting at `pose` on `grid`, of a
    /// plan, `request`, and its retry, which each find no path: the refusal
    /// of `request` empties the grid, and the retry is asked on the readings
    /// after the clearing. The step of the retry's refusal is returned, with
    /// the time of the step after it.
    fn both_refused(
        navigation: &mut Navigation,
        request: PlanRequest,
        now_ns: i64,
        pose: PlanarPose,
        grid: &CostGrid,
    ) -> (NavigationStep, i64) {
        let cleared = step_at(navigation, now_ns, pose, grid, no_path(request), true);
        assert!(cleared.clear_grid, "{cleared:?}");
        let retry = step_at(navigation, now_ns + STEP_NS, pose, grid, None, true)
            .plan
            .expect("the retry, on the readings after the clearing");
        let refused = step_at(
            navigation,
            now_ns + 2 * STEP_NS,
            pose,
            grid,
            no_path(retry),
            true,
        );
        (refused, now_ns + 3 * STEP_NS)
    }

    /// The plan `request` asks for on `grid`, run here.
    fn planned(request: PlanRequest, grid: &CostGrid) -> PlanDone {
        let planned = Planned::from(plan(
            grid,
            &base(1.0).padded_footprint,
            request.start,
            request.goal,
            &config().planner,
        ));
        PlanDone {
            id: request.id,
            planned,
            map_grid: None,
        }
    }

    // A navigation whose goal is in collision fails with unreachable at
    // its first plan, and commands nothing, then or after.
    #[test]
    fn a_goal_in_collision_fails_unreachable_before_the_base_moves() {
        let mut navigation = navigation();
        let on_the_goal = grid(|[x, y]| (x - goal().x()).hypot(y - goal().y()) < 0.1);
        let asked = step(&mut navigation, 0, &on_the_goal, None, true);
        assert_eq!(asked.command, None);
        let request = asked.plan.expect("the first step asks for a plan");
        let ended = step(
            &mut navigation,
            STEP_NS,
            &on_the_goal,
            Some(planned(request, &on_the_goal)),
            true,
        );
        assert!(
            matches!(&ended.ended, Some(Ended::Failed(f)) if f.kind == NavigationKind::Unreachable),
            "{:?}",
            ended.ended
        );
        assert_eq!(ended.command, None);
        let after = step(&mut navigation, 2 * STEP_NS, &on_the_goal, None, true);
        assert_eq!(
            (after.command, after.clear_grid, after.plan, after.ended),
            (None, false, None, None),
            "an ended navigation asks for nothing"
        );
    }

    // While a plan runs, every step gives its feedback and no step asks
    // for a second plan.
    #[test]
    fn a_plan_still_running_asks_for_no_second() {
        let mut navigation = navigation();
        let free = grid(|_| false);
        assert!(step(&mut navigation, 0, &free, None, true).plan.is_some());
        for k in 1..40 {
            let held = step(&mut navigation, k * STEP_NS, &free, None, true);
            assert_eq!(held.plan, None);
            assert_eq!(held.ended, None);
            assert_eq!(held.feedback.distance_remaining_m, 1.5);
        }
    }

    // A pen around the goal runs the plan's retry with its clearing, then
    // the recoveries in order, the clearing, the turn, the wait, the
    // back-up, each once, with no command backward before the back-up, and
    // fails with no_path once the back-up ends.
    #[test]
    fn the_recoveries_run_in_order_then_the_navigation_fails_no_path() {
        let mut navigation = navigation();
        let pen = penned();
        let mut pending = None;
        let mut clearings = 0;
        let mut backward_from = None;
        let mut now_ns = 0;
        let mut read_since_clearing = true;
        let ended = loop {
            let taken = step(
                &mut navigation,
                now_ns,
                &pen,
                pending.take(),
                read_since_clearing,
            );
            read_since_clearing = !taken.clear_grid;
            clearings += u32::from(taken.clear_grid);
            pending = taken.plan.map(|request| planned(request, &pen));
            if taken.command.is_some_and(|c| c.linear_x() < 0.0) {
                backward_from.get_or_insert(taken.feedback.recoveries);
            }
            if let Some(ended) = taken.ended {
                break (ended, taken.feedback.recoveries);
            }
            now_ns += STEP_NS;
            assert!(now_ns < 60_000_000_000, "the navigation ends");
        };
        assert!(
            matches!(&ended.0, Ended::Failed(f) if f.kind == NavigationKind::NoPath),
            "{ended:?}"
        );
        assert_eq!(backward_from, Some(ended.1), "backward only in the back-up");
        // Each plan's retry clears once, and the clearing recovery once.
        assert!(clearings >= 2, "{clearings}");
    }

    // A plan that finds no path empties the grid and plans once more on the
    // readings that arrive after the clearing, and so does the new try of
    // the clearing recovery, which has no path to keep: no plan is asked in
    // the step that clears, nor while a sensor's reading after the clearing
    // has not arrived.
    #[test]
    fn a_plan_after_a_clearing_waits_for_the_readings_after_it() {
        let mut navigation = navigation();
        let pen = penned();
        let first = step(&mut navigation, 0, &pen, None, true)
            .plan
            .expect("the first step asks for a plan");
        let mut now_ns = STEP_NS;
        let mut pending = Some(planned(first, &pen));
        for recoveries in [1, 2] {
            let cleared = step(&mut navigation, now_ns, &pen, pending.take(), true);
            assert!(cleared.clear_grid);
            assert_eq!(cleared.feedback.recoveries, recoveries);
            assert_eq!(cleared.plan, None, "a plan in the clearing's step");
            now_ns += STEP_NS;
            let lagging = step(&mut navigation, now_ns, &pen, None, false);
            assert_eq!(
                lagging.plan, None,
                "a plan before the readings after the clearing"
            );
            now_ns += STEP_NS;
            let read = step(&mut navigation, now_ns, &pen, None, true);
            let request = read.plan.expect("a plan once the readings arrive");
            pending = Some(planned(request, &pen));
            now_ns += STEP_NS;
        }
    }

    /// The step at which the clearing recovery of `navigation`, the base
    /// resting at `pose` on `walled`, starts its try, and the time of the
    /// step after it. The first plan gives the straight path to the goal,
    /// which `walled` blocks ahead of the base, so the follow finds no valid
    /// command past the failure tolerance, empties the grid and follows the
    /// path once more on the readings after the clearing, and that retry
    /// fails the same way. No step before the clearing recovery's asks for a
    /// plan or commands more than zero velocity.
    fn the_clearings_step(
        navigation: &mut Navigation,
        pose: PlanarPose,
        walled: &CostGrid,
    ) -> (NavigationStep, i64) {
        let request = step_at(navigation, 0, pose, walled, None, true)
            .plan
            .expect("the first step asks for a plan");
        let mut plan = Some(PlanDone {
            id: request.id,
            planned: Planned::Path(straight_to_the_goal()),
            map_grid: None,
        });
        let mut now_ns = STEP_NS;
        let mut read_since_clearing = true;
        loop {
            let taken = step_at(
                navigation,
                now_ns,
                pose,
                walled,
                plan.take(),
                read_since_clearing,
            );
            assert_eq!(taken.ended, None);
            assert!(
                taken
                    .command
                    .is_none_or(|command| command == PlanarTwist::ZERO),
                "at {now_ns} ns"
            );
            now_ns += STEP_NS;
            if taken.clear_grid && taken.feedback.recoveries == 2 {
                return (taken, now_ns);
            }
            assert_eq!(taken.plan, None, "a plan before the clearing recovery");
            read_since_clearing = !taken.clear_grid;
            assert!(
                now_ns < 2_000_000_000,
                "the follows fail within their tolerance"
            );
        }
    }

    // A follow that finds no valid command past the failure tolerance
    // empties the grid and follows the path once more, with zero velocity
    // meanwhile; when the retry fails too, the try fails and the first
    // recovery, the clearing, runs. Its try keeps no path longer than the
    // near-goal distance: it plans on the readings that arrive after the
    // clearing.
    #[test]
    fn a_failed_follow_retries_once_then_the_clearing_runs() {
        let mut navigation = navigation();
        let walled = grid(|[x, y]| (0.4..0.5).contains(&x) && y.abs() < 0.5);
        let (cleared, now_ns) = the_clearings_step(&mut navigation, PlanarPose::IDENTITY, &walled);
        assert_eq!((cleared.command, cleared.plan), (None, None));
        let lagging = step(&mut navigation, now_ns, &walled, None, false);
        assert_eq!(
            lagging.plan, None,
            "a plan before the readings after the clearing"
        );
        let read = step(&mut navigation, now_ns + STEP_NS, &walled, None, true);
        let request = read.plan.expect("a plan once the readings arrive");
        assert_eq!((request.goal, read.feedback.recoveries), (goal(), 2));
    }

    // base_navigation's near-goal distance after the clearing recovery: the
    // try it starts keeps a path shorter than the near-goal distance from
    // its pose nearest the robot frame's origin, every pose of it on the
    // grid, which the clearing empties of every pose in collision. The step
    // that clears commands nothing, and the follow runs from the next step
    // on, on the grid the clearing emptied, before the readings after the
    // clearing. With a pose of that path off the grid, the try waits for
    // those readings and plans.
    #[test]
    fn the_clearings_try_keeps_a_near_path_on_the_grid_and_follows_it_on_the_emptied_grid() {
        let walled = grid(wall_ahead);
        let mut kept = navigation();
        let (cleared, now_ns) = the_clearings_step(&mut kept, near(), &walled);
        assert_eq!((cleared.command, cleared.plan), (None, None));
        let emptied = grid(|_| false);
        let followed = step_at(&mut kept, now_ns, near(), &emptied, None, false);
        assert!(
            followed
                .command
                .is_some_and(|command| command.linear_x() > 0.0)
                && followed.plan.is_none(),
            "{followed:?}"
        );

        let cut = cut_short(wall_ahead);
        let mut planned = navigation();
        let (cleared, now_ns) = the_clearings_step(&mut planned, near(), &cut);
        assert_eq!((cleared.command, cleared.plan), (None, None));
        let lagging = step_at(&mut planned, now_ns, near(), &cut, None, false);
        assert_eq!((lagging.command, lagging.plan), (None, None));
        let read = step_at(&mut planned, now_ns + STEP_NS, near(), &cut, None, true);
        assert!(read.plan.is_some(), "a plan once the readings arrive");
    }

    /// The replan of `navigation`'s follow, the base resting at [`near`],
    /// and the time of the step after it. The first plan gives
    /// [`straight_to_the_goal`], which the follow keeps on a free grid until
    /// the replan is due; on `blocking` then, the follow asks for a plan.
    fn the_replan(navigation: &mut Navigation, blocking: &CostGrid) -> (PlanRequest, i64) {
        let free = grid(|_| false);
        let request = step_at(navigation, 0, near(), &free, None, true)
            .plan
            .expect("the first step asks for a plan");
        let path = PlanDone {
            id: request.id,
            planned: Planned::Path(straight_to_the_goal()),
            map_grid: None,
        };
        step_at(navigation, STEP_NS, near(), &free, Some(path), true);
        let replan_due_ns = STEP_NS + config().replan_period_ns;
        for now_ns in (2..)
            .map(|k| k * STEP_NS)
            .take_while(|&t| t < replan_due_ns)
        {
            step_at(navigation, now_ns, near(), &free, None, true);
        }
        let replan = step_at(navigation, replan_due_ns, near(), blocking, None, true)
            .plan
            .expect("a plan at the planning rate once the path is blocked");
        (replan, replan_due_ns + STEP_NS)
    }

    /// The step at which the clearing recovery of `navigation` starts its
    /// try after [`the_replan`] on `blocking` and the replan's retry find no
    /// path, and the time of the step after it.
    fn the_clearings_step_after_a_refused_replan(
        navigation: &mut Navigation,
        blocking: &CostGrid,
    ) -> (NavigationStep, i64) {
        let (replan, now_ns) = the_replan(navigation, blocking);
        let (cleared, now_ns) = both_refused(navigation, replan, now_ns, near(), blocking);
        assert!(
            cleared.clear_grid && cleared.feedback.recoveries == 2,
            "{cleared:?}"
        );
        (cleared, now_ns)
    }

    // A plan that the planner refuses leaves the navigation no path to
    // keep, as Nav2's ComputePathToPose writes an empty path when it
    // aborts: once the replan of a near path and the replan's retry find no
    // path, the clearing recovery's try, with no path to keep on the
    // emptied grid, commands nothing and plans on the readings after the
    // clearing.
    #[test]
    fn after_a_refused_replan_the_clearings_try_plans_on_the_readings_after_it() {
        let mut navigation = navigation();
        let walled = grid(wall_ahead);
        let (cleared, now_ns) = the_clearings_step_after_a_refused_replan(&mut navigation, &walled);
        assert_eq!((cleared.command, cleared.plan), (None, None));
        let emptied = grid(|_| false);
        let lagging = step_at(&mut navigation, now_ns, near(), &emptied, None, false);
        assert_eq!(
            (lagging.command, lagging.plan),
            (None, None),
            "before the readings after the clearing"
        );
        let read = step_at(
            &mut navigation,
            now_ns + STEP_NS,
            near(),
            &walled,
            None,
            true,
        );
        assert!(read.command.is_none() && read.plan.is_some(), "{read:?}");
    }

    // The same for the tries after the turn and the wait: once the
    // clearing recovery's try, on a grid with a pose of the near path off
    // it, plans and finds no path twice, the turn runs; the try after it,
    // with no path to keep on a free grid, plans at once, and once that plan
    // finds no path twice, the wait runs, and the try after it plans at
    // once too.
    #[test]
    fn after_refused_plans_the_tries_after_the_turn_and_the_wait_plan() {
        let mut navigation = navigation();
        let cut = cut_short(|_| false);
        let (_, now_ns) = the_clearings_step_after_a_refused_replan(&mut navigation, &cut);
        let free = grid(|_| false);
        let plan = step_at(&mut navigation, now_ns, near(), &free, None, true)
            .plan
            .expect("the clearing's try plans on the readings after it");
        let (turning, now_ns) =
            both_refused(&mut navigation, plan, now_ns + STEP_NS, near(), &free);
        assert!(
            turning
                .command
                .is_some_and(|command| command.angular_z() > 0.0)
                && turning.feedback.recoveries == 4,
            "{turning:?}"
        );

        let turn_angle_rad = config().recoveries.turn_angle_rad.get();
        let turned = PlanarPose::new(near().x(), near().y(), turn_angle_rad + 0.1).unwrap();
        let after_turn = step_at(&mut navigation, now_ns, turned, &free, None, true);
        assert_eq!(after_turn.command, None);
        let plan = after_turn.plan.expect("the turn's try plans at once");

        let (waiting, now_ns) =
            both_refused(&mut navigation, plan, now_ns + STEP_NS, turned, &free);
        assert_eq!(
            (waiting.command, waiting.feedback.recoveries),
            (Some(PlanarTwist::ZERO), 6)
        );
        let waited_ns = now_ns - STEP_NS + config().recoveries.wait_ns;
        let after_wait = step_at(&mut navigation, waited_ns, turned, &free, None, true);
        assert!(
            after_wait.command.is_none() && after_wait.plan.is_some(),
            "{after_wait:?}"
        );
    }

    /// The step from `now_ns` at which the follow of `navigation`, the base
    /// resting at [`near`] on `walled`, fails with no plan finishing
    /// meanwhile, and the time of the step after it.
    fn the_follows_failure(
        navigation: &mut Navigation,
        mut now_ns: i64,
        walled: &CostGrid,
    ) -> (NavigationStep, i64) {
        loop {
            let taken = step_at(navigation, now_ns, near(), walled, None, true);
            now_ns += STEP_NS;
            if taken.clear_grid {
                return (taken, now_ns);
            }
            assert!(
                now_ns < 3_000_000_000,
                "the follow fails within its tolerance"
            );
        }
    }

    /// A path from [`near`] 0.5 m to its left, then to the goal.
    fn by_the_left() -> Path {
        let poses = (0..=10)
            .map(|k| PlanarPose::new(near().x(), 0.05 * f64::from(k), 0.0).unwrap())
            .chain([goal()])
            .collect();
        Path::new(poses).expect("12 poses")
    }

    // A replan that finds no path leaves the navigation no path, so when
    // the follow then fails and no plan gives it one while its retry waits
    // for the readings after its clearing, the retry has none to follow: the
    // try fails at the retry, as Nav2's controller server refuses the empty
    // path that ComputePathToPose leaves, and the clearing recovery's try
    // plans on the readings after the clearing.
    #[test]
    fn a_follows_retry_with_no_path_fails_and_the_clearings_try_plans() {
        let mut navigation = navigation();
        let walled = grid(wall_ahead);
        let (replan, now_ns) = the_replan(&mut navigation, &walled);
        let refused = step_at(
            &mut navigation,
            now_ns,
            near(),
            &walled,
            no_path(replan),
            true,
        );
        assert!(refused.clear_grid, "{refused:?}");
        let (failed, now_ns) = the_follows_failure(&mut navigation, now_ns + STEP_NS, &walled);
        assert_eq!(
            (failed.command, failed.plan, failed.feedback.recoveries),
            (Some(PlanarTwist::ZERO), None, 2)
        );
        let retried = step_at(&mut navigation, now_ns, near(), &walled, None, true);
        assert_eq!(
            (retried.command, retried.plan, retried.feedback.recoveries),
            (None, None, 3),
            "the clearing recovery, at the retry"
        );
        assert!(retried.clear_grid);
        let read = step_at(
            &mut navigation,
            now_ns + STEP_NS,
            near(),
            &walled,
            None,
            true,
        );
        assert!(read.command.is_none() && read.plan.is_some(), "{read:?}");
    }

    // While a failed follow's retry waits for the readings after its
    // clearing, the replan running goes on, as Nav2's ComputePathToPose
    // runs on while FollowPath's RecoveryNode retries: the path a plan finds
    // then replaces the path, and the retry follows it.
    #[test]
    fn a_path_found_while_the_follows_retry_waits_is_retried() {
        let mut navigation = navigation();
        let walled = grid(wall_ahead);
        let (replan, now_ns) = the_replan(&mut navigation, &walled);
        let (_, now_ns) = the_follows_failure(&mut navigation, now_ns, &walled);
        let found = PlanDone {
            id: replan.id,
            planned: Planned::Path(by_the_left()),
            map_grid: None,
        };
        let waiting = step_at(&mut navigation, now_ns, near(), &walled, Some(found), false);
        assert_eq!(waiting.command, Some(PlanarTwist::ZERO));
        let emptied = grid(|_| false);
        let retried = step_at(
            &mut navigation,
            now_ns + STEP_NS,
            near(),
            &emptied,
            None,
            true,
        );
        assert!(
            retried
                .command
                .is_some_and(|command| command.angular_z() > 0.0),
            "a turn to the left, onto the path found: {retried:?}"
        );
    }

    // The same when a replan that found no path left the navigation none:
    // the path the replan's retry finds while the follow's retry waits gives
    // the retry a path to follow.
    #[test]
    fn a_path_found_while_the_follows_retry_waits_gives_it_one() {
        let mut navigation = navigation();
        let walled = grid(wall_ahead);
        let (replan, now_ns) = the_replan(&mut navigation, &walled);
        step_at(
            &mut navigation,
            now_ns,
            near(),
            &walled,
            no_path(replan),
            true,
        );
        let retry = step_at(
            &mut navigation,
            now_ns + STEP_NS,
            near(),
            &walled,
            None,
            true,
        )
        .plan
        .expect("the replan's retry, on the readings after its clearing");
        let (_, now_ns) = the_follows_failure(&mut navigation, now_ns + 2 * STEP_NS, &walled);
        let found = PlanDone {
            id: retry.id,
            planned: Planned::Path(straight_to_the_goal()),
            map_grid: None,
        };
        let waiting = step_at(&mut navigation, now_ns, near(), &walled, Some(found), false);
        assert_eq!(waiting.command, Some(PlanarTwist::ZERO));
        let emptied = grid(|_| false);
        let retried = step_at(
            &mut navigation,
            now_ns + STEP_NS,
            near(),
            &emptied,
            None,
            true,
        );
        assert!(
            retried
                .command
                .is_some_and(|command| command.linear_x() > 0.0)
                && retried.plan.is_none(),
            "{retried:?}"
        );
    }

    // A replan that finds no path while the follow's retry waits empties
    // the navigation's path, as a refused plan does at any step: the
    // replan's own retry clears the grid, and the follow's retry, with no
    // path, fails the try at the readings after the clearing.
    #[test]
    fn a_path_refused_while_the_follows_retry_waits_leaves_it_none() {
        let mut navigation = navigation();
        let walled = grid(wall_ahead);
        let (replan, now_ns) = the_replan(&mut navigation, &walled);
        let (_, now_ns) = the_follows_failure(&mut navigation, now_ns, &walled);
        let refused = step_at(
            &mut navigation,
            now_ns,
            near(),
            &walled,
            no_path(replan),
            false,
        );
        assert_eq!(
            (
                refused.command,
                refused.clear_grid,
                refused.feedback.recoveries
            ),
            (Some(PlanarTwist::ZERO), true, 2)
        );
        let retried = step_at(
            &mut navigation,
            now_ns + STEP_NS,
            near(),
            &walled,
            None,
            true,
        );
        assert_eq!(
            (
                retried.command,
                retried.clear_grid,
                retried.feedback.recoveries
            ),
            (None, true, 3),
            "the clearing recovery, at the retry"
        );
    }

    // A replan asked before the clearing recovery, refused in the step after
    // it, leaves the path the clearing's try keeps: the follow runs on it.
    #[test]
    fn a_plan_refused_after_the_clearing_leaves_the_kept_path_followed() {
        let mut navigation = navigation();
        let walled = grid(wall_ahead);
        let (replan, mut now_ns) = the_replan(&mut navigation, &walled);
        let mut read_since_clearing = true;
        loop {
            let taken = step_at(
                &mut navigation,
                now_ns,
                near(),
                &walled,
                None,
                read_since_clearing,
            );
            now_ns += STEP_NS;
            if taken.clear_grid && taken.feedback.recoveries == 2 {
                break;
            }
            read_since_clearing = !taken.clear_grid;
            assert!(
                now_ns < 4_000_000_000,
                "the follow and its retry fail within their tolerance"
            );
        }
        let emptied = grid(|_| false);
        let followed = step_at(
            &mut navigation,
            now_ns,
            near(),
            &emptied,
            no_path(replan),
            false,
        );
        assert!(
            followed
                .command
                .is_some_and(|command| command.linear_x() > 0.0)
                && followed.plan.is_none(),
            "{followed:?}"
        );
    }

    // A navigation completes at the step its goal checker finds the goal
    // reached, with no command, whatever the base's velocity then.
    #[test]
    fn a_navigation_completes_at_the_step_its_goal_is_reached() {
        let free = grid(|_| false);
        let footprint = base(1.0).padded_footprint;
        let mut navigation = navigation();
        let request = step(&mut navigation, 0, &free, None, true)
            .plan
            .expect("the first step asks for a plan");
        let path = PlanDone {
            id: request.id,
            planned: Planned::Path(straight_to_the_goal()),
            map_grid: None,
        };
        let at_goal = navigation.step(NavigationInputs {
            now_ns: STEP_NS,
            base: BasePose::in_odometry(goal()),
            velocity: PlanarTwist::new(0.1, 0.0, 0.0).unwrap(),
            grid: &free,
            footprint: &footprint,
            read_since_clearing: true,
            plan: Some(path),
        });
        assert_eq!(
            (at_goal.command, at_goal.ended),
            (None, Some(Ended::Completed))
        );
    }

    // base_navigation's near-goal distance: at the planning rate a follow
    // keeps a path shorter than the near-goal distance from its pose nearest
    // the robot frame's origin, with no pose in collision, and asks for no
    // plan; from farther along the path it asks for one.
    #[test]
    fn a_path_within_the_near_goal_distance_is_kept_at_the_planning_rate() {
        let free = grid(|_| false);
        let footprint = base(1.0).padded_footprint;
        let replan_due_ns = STEP_NS + config().replan_period_ns;
        let plan_at_the_replan = |pose: PlanarPose| {
            let mut navigation = navigation();
            let request = step(&mut navigation, 0, &free, None, true)
                .plan
                .expect("the first step asks for a plan");
            let path = PlanDone {
                id: request.id,
                planned: Planned::Path(straight_to_the_goal()),
                map_grid: None,
            };
            step(&mut navigation, STEP_NS, &free, Some(path), true);
            navigation
                .step(NavigationInputs {
                    now_ns: replan_due_ns,
                    base: BasePose::in_odometry(pose),
                    velocity: PlanarTwist::ZERO,
                    grid: &free,
                    footprint: &footprint,
                    read_since_clearing: true,
                    plan: None,
                })
                .plan
        };
        let near_x = goal().x() - config().near_goal_distance_m.get() / 2.0;
        let near = PlanarPose::new(near_x, 0.0, 0.0).unwrap();
        assert_eq!(plan_at_the_replan(near), None, "the near path is kept");
        assert!(
            plan_at_the_replan(PlanarPose::IDENTITY).is_some(),
            "a plan from 1.5 m out"
        );
    }

    // A navigation in map keeps its path by the path's poses placed in odom,
    // where the grid of the floor is: a wall across the placed path makes it
    // plan again at the planning rate, and one across odom's own y of 0, where
    // the path's poses lie in the frame they are planned in, does not.
    #[test]
    fn a_navigation_in_map_reads_its_path_placed_in_odom() {
        let footprint = base(1.0).padded_footprint;
        let odom_from_run = PlanarPose::new(0.0, 1.0, 0.0).unwrap();
        let base_at = |in_run: PlanarPose| {
            let in_odom = odom_from_run.compose(in_run).unwrap();
            BasePose::in_frame(in_odom, odom_from_run).unwrap()
        };
        let replan_due_ns = STEP_NS + config().replan_period_ns;
        let plan_at_the_replan = |grid: &CostGrid| {
            let mut navigation = navigation();
            let inputs = |now_ns, in_run, plan| NavigationInputs {
                now_ns,
                base: base_at(in_run),
                velocity: PlanarTwist::ZERO,
                grid,
                footprint: &footprint,
                read_since_clearing: true,
                plan,
            };
            let request = navigation
                .step(inputs(0, PlanarPose::IDENTITY, None))
                .plan
                .expect("the first step asks for a plan");
            let path = PlanDone {
                id: request.id,
                planned: Planned::Path(straight_to_the_goal()),
                map_grid: None,
            };
            navigation.step(inputs(STEP_NS, PlanarPose::IDENTITY, Some(path)));
            navigation.step(inputs(replan_due_ns, near(), None)).plan
        };
        let across_the_placed_path =
            grid(|[x, y]| (1.3..1.4).contains(&x) && (y - 1.0).abs() < 0.5);
        let across_the_unplaced_path = grid(wall_ahead);
        assert!(plan_at_the_replan(&across_the_placed_path).is_some());
        assert_eq!(plan_at_the_replan(&across_the_unplaced_path), None);
    }
}
