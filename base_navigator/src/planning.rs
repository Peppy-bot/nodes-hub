//! The planner on a task of its own: a navigation asks for a plan from where
//! the base stands to its goal, in the frame it runs in, the plan runs on a
//! blocking thread on the grid as it stood at the ask, and the control loop
//! takes the outcome at the first step after it finished. A navigation in odom
//! plans on the grid of the floor's occupancy. A navigation in map plans as
//! Nav2's global cost map composes its layers: the loaded map's newest grid as
//! the static layer, padded with unknown cells out to the extent the goal was
//! admitted against, the grid of the floor's obstacles projected onto it, and
//! the inflation, composed on the blocking thread. A plan still running delays
//! no step of the controller and the stop, and a step asks for no second plan
//! while one runs.
//!
//! The [`Planner`] runs each plan at once, or, [`Planner::gated`], once its
//! [`PlanGate`] lets it start, so a test holds a plan running across as many
//! steps as it likes.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use control_core::planar_pose::PlanarPose;
use navigation_core::PaddedFootprint;
use navigation_core::costmap::compose;
use navigation_core::grid::CostGrid;
use navigation_core::inflation::Inflation;
use navigation_core::obstacles::ObstacleLayer;
use navigation_core::planner::{PlanError, PlannerParams, plan};
use tokio::sync::oneshot::error::TryRecvError;
use tokio::sync::{Semaphore, oneshot};

use crate::room_map::{MapGrid, PlanLattice};

/// A path the planner found: its poses in the frame its navigation runs in,
/// from the base's cell to the goal, at least one.
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
    /// Why the plan's grid was not built: the planner did not run.
    GridRefused(String),
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
/// the frame the navigation runs in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanRequest {
    pub id: u64,
    pub start: PlanarPose,
    pub goal: PlanarPose,
}

/// The layers of a plan of a navigation in map: the lattice that holds the
/// extent its goal was admitted against, the loaded map's newest grid as the
/// static layer, none while it has no cell, the grid of the floor's obstacles
/// projected onto the lattice, and the inflation.
pub struct MapLayers {
    pub lattice: PlanLattice,
    pub map: Option<Arc<MapGrid>>,
    pub obstacles: ObstacleLayer,
    pub inflation: Inflation,
}

impl MapLayers {
    /// The cost grid of the layers, composed as Nav2's global cost map
    /// composes a static, an obstacle and an inflation layer.
    fn composed(&self) -> Result<CostGrid, String> {
        let saved = self
            .map
            .as_deref()
            .map(|map| self.lattice.static_layer(map))
            .transpose()
            .map_err(|e| format!("the loaded map's grid on the plan's grid: {e}"))?;
        compose(saved.as_ref(), &self.obstacles, Some(&self.inflation))
            .map_err(|e| format!("the plan's grid: {e}"))
    }
}

/// The grid a plan runs on.
pub enum PlanGrid {
    /// The grid of the floor's occupancy in odom, for a navigation in odom.
    Floor(Arc<CostGrid>),
    /// The layers of a navigation in map, composed on the planner's thread.
    Map(Box<MapLayers>),
}

impl PlanGrid {
    /// The cost grid the planner reads, or why it gives none.
    fn costs(self) -> Result<Arc<CostGrid>, String> {
        match self {
            Self::Floor(grid) => Ok(grid),
            Self::Map(layers) => layers.composed().map(Arc::new),
        }
    }
}

/// A plan that finished: the number of its ask, what it found, and, for a
/// plan of a navigation in map, the grid in map it ran on, composed of its
/// layers, which judges the poses of the path it found off the grid of the
/// floor.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanDone {
    pub id: u64,
    pub planned: Planned,
    pub map_grid: Option<Arc<CostGrid>>,
}

/// What a plan's task hands the control loop: what the plan found, and the
/// grid in map it ran on for a navigation in map.
struct Outcome {
    planned: Planned,
    map_grid: Option<Arc<CostGrid>>,
}

/// The gate a gated planner waits at before each plan: each permit lets one
/// plan start. It counts the plans asked of the planner.
#[derive(Clone, Debug)]
pub struct PlanGate {
    permits: Arc<Semaphore>,
    asked: Arc<AtomicUsize>,
}

impl PlanGate {
    /// Let `plans` more plans start.
    pub fn release(&self, plans: usize) {
        self.permits.add_permits(plans);
    }

    /// How many plans were asked of the planner, each counted when the
    /// control loop asks it.
    pub fn asked(&self) -> usize {
        self.asked.load(Ordering::SeqCst)
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
        let gate = PlanGate {
            permits: Arc::new(Semaphore::new(0)),
            asked: Arc::default(),
        };
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
    outcome: oneshot::Receiver<Outcome>,
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
    pub fn start(&mut self, request: PlanRequest, grid: PlanGrid, footprint: PaddedFootprint) {
        self.count_asked();
        let (sender, outcome) = oneshot::channel();
        let gate = self.planner.gate.clone();
        let params = self.params;
        let in_map = matches!(grid, PlanGrid::Map(_));
        tokio::spawn(async move {
            if let Some(gate) = gate {
                let Ok(permit) = gate.permits.acquire().await else {
                    return;
                };
                permit.forget();
            }
            let outcome = tokio::task::spawn_blocking(move || match grid.costs() {
                Ok(costs) => Outcome {
                    planned: Planned::from(plan(
                        &costs,
                        &footprint,
                        request.start,
                        request.goal,
                        &params,
                    )),
                    map_grid: in_map.then_some(costs),
                },
                Err(why) => Outcome {
                    planned: Planned::GridRefused(why),
                    map_grid: None,
                },
            })
            .await;
            if let Ok(outcome) = outcome {
                let _ = sender.send(outcome);
            }
        });
        self.pending = Some(Pending {
            id: request.id,
            outcome,
        });
    }

    /// Finish the plan of `request` at once, its grid not built for `why`, in
    /// place of a plan that still runs, whose outcome no step takes.
    pub fn refuse(&mut self, request: PlanRequest, why: String) {
        self.count_asked();
        let (sender, outcome) = oneshot::channel();
        let _ = sender.send(Outcome {
            planned: Planned::GridRefused(why),
            map_grid: None,
        });
        self.pending = Some(Pending {
            id: request.id,
            outcome,
        });
    }

    /// The plan that finished since the step before, if one did.
    pub fn finished(&mut self) -> Option<PlanDone> {
        let pending = self.pending.as_mut()?;
        let Outcome { planned, map_grid } = match pending.outcome.try_recv() {
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Closed) => Outcome {
                planned: Planned::Lost,
                map_grid: None,
            },
            Ok(outcome) => outcome,
        };
        let id = pending.id;
        self.pending = None;
        Some(PlanDone {
            id,
            planned,
            map_grid,
        })
    }

    /// Counts a plan asked on the gate of a gated planner.
    fn count_asked(&self) {
        if let Some(gate) = &self.planner.gate {
            gate.asked.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Forget the plan running: no step takes its outcome.
    pub fn forget(&mut self) {
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::SystemTime;

    use control_core::positive_finite::PositiveFinite;
    use navigation_core::saved_grid::MapExtent;
    use peppygen::consumed_topics::room_map::map_grids;

    use crate::base::tests::base;
    use crate::config::tests::config;
    use crate::follow::MAP_FRAME;
    use crate::grid::tests::grid;
    use crate::room_map::{AdmittedMap, LoadedGrid, MapRun};

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
    async fn outcome(planning: &mut Planning) -> Outcome {
        let pending = planning.pending.as_mut().expect("a plan runs");
        (&mut pending.outcome)
            .await
            .expect("the plan's task sends its outcome")
    }

    // A gated plan stays unfinished at every step until its gate lets
    // it start, and then finishes with the path of its ask; the gate counts
    // it asked, and a plan on the grid of the floor hands back no grid in map.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_gated_plan_finishes_only_once_released() {
        let (planner, gate) = Planner::gated();
        let mut planning = Planning::new(planner, config().planner);
        let occupancy = grid().occupancy(None).expect("an occupancy");
        planning.start(
            request(7),
            PlanGrid::Floor(occupancy),
            base(1.0).padded_footprint,
        );
        assert_eq!(gate.asked(), 1);
        for _ in 0..10 {
            tokio::task::yield_now().await;
            assert_eq!(planning.finished(), None, "the gate holds the plan");
        }
        gate.release(1);
        let Outcome { planned, map_grid } = outcome(&mut planning).await;
        assert!(map_grid.is_none());
        let Planned::Path(path) = planned else {
            panic!("a path on an empty grid");
        };
        assert!(
            path.poses()
                .last()
                .is_some_and(|end| (end.x() - 1.0).abs() < 0.05)
        );
    }

    // A plan started in place of another leaves the other's outcome untaken.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_new_plan_replaces_the_one_running() {
        let (planner, gate) = Planner::gated();
        let mut planning = Planning::new(planner, config().planner);
        let occupancy = grid().occupancy(None).expect("an occupancy");
        planning.start(
            request(1),
            PlanGrid::Floor(occupancy.clone()),
            base(1.0).padded_footprint,
        );
        planning.start(
            request(2),
            PlanGrid::Floor(occupancy),
            base(1.0).padded_footprint,
        );
        assert_eq!(planning.pending.as_ref().map(|pending| pending.id), Some(2));
        assert_eq!(gate.asked(), 2);
        gate.release(2);
        assert!(matches!(
            outcome(&mut planning).await.planned,
            Planned::Path(_)
        ));
    }

    /// A map 4 m by 3 m of 0.05 m cells from map's origin, a wall across it from
    /// x 1.9 m to 2.1 m up to y 1.8 m, and a doorway above.
    fn walled_map() -> Arc<MapGrid> {
        let cells = (0..60)
            .flat_map(|y| (0..80).map(move |x| (x, y)))
            .map(|(x, y)| {
                if (38..42).contains(&x) && y < 36 {
                    100
                } else {
                    0
                }
            })
            .collect();
        let message = map_grids::Message {
            frame: MAP_FRAME.to_string(),
            timestamp: SystemTime::UNIX_EPOCH,
            map_name: "lab".to_string(),
            map_load_timestamp: SystemTime::UNIX_EPOCH,
            cell_size_m: 0.05,
            extent: Some([0.0, 4.0, 0.0, 3.0]),
            columns: 80,
            rows: 60,
            cells,
        };
        LoadedGrid::parse(message, 1)
            .expect("a map's grid")
            .grid
            .expect("a grid of cells")
    }

    // A plan of a navigation in map runs on the loaded map's grid, padded out to
    // the extent the goal was admitted against, over the grid of the floor's
    // obstacles: its path goes around the map's wall, through the doorway, and
    // it hands back that grid, on the plan's lattice.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_plan_in_map_goes_around_a_wall_the_map_holds() {
        let map = walled_map();
        let run = MapRun {
            map: AdmittedMap {
                name: "lab".to_string(),
                extent: MapExtent::parse([0.0, 4.0, 0.0, 3.0]).unwrap(),
                cell_size_m: PositiveFinite::parse(0.05).unwrap(),
            },
            loaded_at: SystemTime::UNIX_EPOCH,
        };
        let lattice = run.plan_lattice(Some(&map)).expect("a lattice");
        let geometry = lattice.geometry;
        let obstacles = grid()
            .obstacles_on(PlanarPose::IDENTITY, geometry)
            .expect("the obstacles");
        let layers = MapLayers {
            lattice,
            map: Some(map),
            obstacles,
            inflation: base(1.0).inflation,
        };
        let mut planning = Planning::new(Planner::default(), config().planner);
        let across = PlanRequest {
            id: 4,
            start: PlanarPose::new(0.5, 0.5, 0.0).unwrap(),
            goal: PlanarPose::new(3.5, 0.5, 0.0).unwrap(),
        };
        planning.start(
            across,
            PlanGrid::Map(Box::new(layers)),
            base(1.0).padded_footprint,
        );
        let Outcome { planned, map_grid } = outcome(&mut planning).await;
        assert_eq!(
            map_grid.map(|grid| grid.geometry()),
            Some(geometry),
            "the plan hands back the grid in map it ran on"
        );
        let Planned::Path(path) = planned else {
            panic!("a path through the doorway");
        };
        let at_the_wall: Vec<&PlanarPose> = path
            .poses()
            .iter()
            .filter(|pose| (1.9..=2.1).contains(&pose.x()))
            .collect();
        assert!(
            !at_the_wall.is_empty() && at_the_wall.iter().all(|pose| pose.y() > 1.8),
            "{at_the_wall:?}"
        );
    }

    // A plan whose grid the control loop could not build finishes at once,
    // refused with the reason.
    #[test]
    fn a_plan_whose_grid_was_not_built_finishes_refused() {
        let mut planning = Planning::new(Planner::default(), config().planner);
        planning.refuse(request(3), "the extent spans too many cells".to_string());
        assert_eq!(
            planning.finished(),
            Some(PlanDone {
                id: 3,
                planned: Planned::GridRefused("the extent spans too many cells".to_string()),
                map_grid: None,
            })
        );
    }
}
