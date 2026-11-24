//! The planner on a task of its own: a navigation asks for a plan from where
//! the base stands to its goal, the plan runs on a blocking thread on the
//! grid's occupancy as it stood at the ask, and the control loop takes the
//! outcome at the first step after it finished. A plan still running delays
//! no step of the controller and the stop, and a step asks for no second plan
//! while one runs.
//!
//! The [`Planner`] runs each plan at once, or, [`Planner::gated`], once its
//! [`PlanGate`] lets it start, so a test holds a plan running across as many
//! steps as it likes.

use std::sync::Arc;

use control_core::planar_pose::PlanarPose;
use navigation_core::PaddedFootprint;
use navigation_core::grid::CostGrid;
use navigation_core::planner::{PlanError, PlannerParams, plan};
use tokio::sync::oneshot::error::TryRecvError;
use tokio::sync::{Semaphore, oneshot};

/// A path the planner found: its poses in odom, from the base's cell to the
/// goal, at least one.
#[derive(Clone, Debug, PartialEq)]
pub struct Path(Vec<PlanarPose>);

impl Path {
    /// The path of `poses`; none when they hold no pose.
    pub fn new(poses: Vec<PlanarPose>) -> Option<Self> {
        (!poses.is_empty()).then_some(Self(poses))
    }

    pub fn poses(&self) -> &[PlanarPose] {
        &self.0
    }

    pub fn into_poses(self) -> Vec<PlanarPose> {
        self.0
    }

    /// The index of the pose nearest the robot frame's origin at `pose`.
    pub fn nearest_index(&self, pose: PlanarPose) -> usize {
        self.0
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance_to(pose).total_cmp(&b.1.distance_to(pose)))
            .map(|(index, _)| index)
            .expect("a path holds a pose")
    }

    /// The length (m) of the path from its pose at `index` to its end.
    pub fn length_from(&self, index: usize) -> f64 {
        self.0[index..]
            .windows(2)
            .map(|leg| leg[0].distance_to(leg[1]))
            .sum()
    }

    /// The length (m) of the path from the robot frame's origin at `pose`:
    /// the distance to the path's pose nearest it, then along the path from
    /// there.
    pub fn remaining_from(&self, pose: PlanarPose) -> f64 {
        let nearest = self.nearest_index(pose);
        self.0[nearest].distance_to(pose) + self.length_from(nearest)
    }
}

/// What a plan found.
#[derive(Clone, Debug, PartialEq)]
pub enum Planned {
    Path(Path),
    /// Why the planner found no path.
    Refused(PlanError),
    /// The planner's task ended with no outcome.
    Lost,
}

impl From<Result<Vec<PlanarPose>, PlanError>> for Planned {
    /// What the planner's outcome found: its path runs from the start's cell
    /// to the goal, so it holds a pose.
    fn from(outcome: Result<Vec<PlanarPose>, PlanError>) -> Self {
        match outcome {
            Ok(poses) => Self::Path(Path::new(poses).expect("the planner's path holds a pose")),
            Err(refusal) => Self::Refused(refusal),
        }
    }
}

/// A navigation's ask for a plan: its number, and the start and the goal in
/// odom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanRequest {
    pub id: u64,
    pub start: PlanarPose,
    pub goal: PlanarPose,
}

/// A plan that finished: the number of its ask, and what it found.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanDone {
    pub id: u64,
    pub planned: Planned,
}

/// The gate a gated planner waits at before each plan: each permit lets one
/// plan start.
#[derive(Clone, Debug)]
pub struct PlanGate(Arc<Semaphore>);

impl PlanGate {
    /// Let `plans` more plans start.
    pub fn release(&self, plans: usize) {
        self.0.add_permits(plans);
    }
}

/// How the node runs its plans: at once, or each once its gate lets it.
#[derive(Clone, Debug, Default)]
pub struct Planner {
    gate: Option<PlanGate>,
}

impl Planner {
    /// A planner whose plans wait at the gate it gives, which lets none start
    /// until it is released.
    pub fn gated() -> (Self, PlanGate) {
        let gate = PlanGate(Arc::new(Semaphore::new(0)));
        (
            Self {
                gate: Some(gate.clone()),
            },
            gate,
        )
    }
}

/// The plan of the navigation in flight while one runs.
struct Pending {
    id: u64,
    outcome: oneshot::Receiver<Result<Vec<PlanarPose>, PlanError>>,
}

/// The control loop's planning: the planner, its settings, and the plan
/// running.
pub struct Planning {
    planner: Planner,
    params: PlannerParams,
    pending: Option<Pending>,
}

impl Planning {
    /// The planning of `planner` with `params`.
    pub fn new(planner: Planner, params: PlannerParams) -> Self {
        Self {
            planner,
            params,
            pending: None,
        }
    }

    /// Start the plan of `request` on `grid` for a base of `footprint`, in
    /// place of a plan that still runs, whose outcome no step takes.
    pub fn start(&mut self, request: PlanRequest, grid: Arc<CostGrid>, footprint: PaddedFootprint) {
        let (sender, outcome) = oneshot::channel();
        let gate = self.planner.gate.clone();
        let params = self.params;
        tokio::spawn(async move {
            if let Some(PlanGate(gate)) = gate {
                let Ok(permit) = gate.acquire().await else {
                    return;
                };
                permit.forget();
            }
            let planned = tokio::task::spawn_blocking(move || {
                plan(&grid, &footprint, request.start, request.goal, &params)
            })
            .await;
            if let Ok(outcome) = planned {
                let _ = sender.send(outcome);
            }
        });
        self.pending = Some(Pending {
            id: request.id,
            outcome,
        });
    }

    /// The plan that finished since the step before, if one did.
    pub fn finished(&mut self) -> Option<PlanDone> {
        let pending = self.pending.as_mut()?;
        let planned = match pending.outcome.try_recv() {
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Closed) => Planned::Lost,
            Ok(outcome) => Planned::from(outcome),
        };
        let id = pending.id;
        self.pending = None;
        Some(PlanDone { id, planned })
    }

    /// Forget the plan running: no step takes its outcome.
    pub fn forget(&mut self) {
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::base::tests::base;
    use crate::config::tests::config;
    use crate::grid::tests::grid;

    // A path holds at least one pose.
    #[test]
    fn a_path_holds_a_pose() {
        assert_eq!(Path::new(Vec::new()), None);
        let one = vec![PlanarPose::IDENTITY];
        assert_eq!(Path::new(one.clone()).map(Path::into_poses), Some(one));
    }

    // The path left from the robot frame's origin is never shorter than the
    // straight line to the path's end, and is the whole path from its start.
    #[test]
    fn what_remains_along_a_path_is_at_least_the_straight_distance_to_its_end() {
        let poses = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
            .iter()
            .map(|&[x, y]| PlanarPose::new(x, y, 0.0).unwrap())
            .collect();
        let path = Path::new(poses).expect("three poses");
        assert_eq!(path.remaining_from(path.poses()[0]), 2.0);
        let beside = PlanarPose::new(0.5, -0.5, 0.0).unwrap();
        let left = path.remaining_from(beside);
        assert!(left >= beside.distance_to(path.poses()[2]), "{left}");
    }

    fn request(id: u64) -> PlanRequest {
        PlanRequest {
            id,
            start: PlanarPose::IDENTITY,
            goal: PlanarPose::new(1.0, 0.0, 0.0).expect("a goal"),
        }
    }

    /// The outcome of the plan running, once its task sends it.
    async fn outcome(planning: &mut Planning) -> Result<Vec<PlanarPose>, PlanError> {
        let pending = planning.pending.as_mut().expect("a plan runs");
        (&mut pending.outcome)
            .await
            .expect("the plan's task sends its outcome")
    }

    // A gated plan stays unfinished at every step until its gate lets
    // it start, and then finishes with the path of its ask.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_gated_plan_finishes_only_once_released() {
        let (planner, gate) = Planner::gated();
        let mut planning = Planning::new(planner, config().planner);
        let occupancy = grid().occupancy().expect("an occupancy");
        planning.start(request(7), occupancy, base(1.0).padded_footprint);
        for _ in 0..10 {
            tokio::task::yield_now().await;
            assert_eq!(planning.finished(), None, "the gate holds the plan");
        }
        gate.release(1);
        let path = outcome(&mut planning)
            .await
            .expect("a path on an empty grid");
        assert!(path.last().is_some_and(|end| (end.x() - 1.0).abs() < 0.05));
    }

    // A plan started in place of another leaves the other's outcome untaken.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_new_plan_replaces_the_one_running() {
        let (planner, gate) = Planner::gated();
        let mut planning = Planning::new(planner, config().planner);
        let occupancy = grid().occupancy().expect("an occupancy");
        planning.start(request(1), occupancy.clone(), base(1.0).padded_footprint);
        planning.start(request(2), occupancy, base(1.0).padded_footprint);
        assert_eq!(planning.pending.as_ref().map(|pending| pending.id), Some(2));
        gate.release(2);
        outcome(&mut planning).await.expect("a path");
    }
}
