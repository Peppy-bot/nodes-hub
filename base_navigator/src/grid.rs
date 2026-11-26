//! The grid of the floor: navigation_core's `LiveObstacles` kept in odom, a
//! square of `grid.side_m` around its center, from `grid.min_z_m` up to
//! `max_obstacle_height_m`, with the stop that reads each sensor's newest
//! reading beside it. Each reading goes in with the sensor's place on the base
//! and the base's place in odom at its time: a depth frame's at its timestamp,
//! a scan's at its first and at its last ray, each ray placed where the sensor
//! stood at the ray's time. The grid drops what lies within the base's outline,
//! marks only the returns within `obstacle_max_range_m` of the sensor along the
//! floor (of the sensor at a scan's first ray), and clears the padded footprint
//! where the base stood at the reading's time, a scan's first ray.
//!
//! The grid answers one occupancy query, [`LiveGrid::occupancy`]: the cost
//! grid of what the sensors have seen, inflated for the padded footprint,
//! built once per change. For a navigation in map it holds the loaded map's
//! occupied cells too ([`MapOverlay`]), placed on the grid where map stands in
//! odom: the map's grid composed as Nav2's static layer under the obstacles,
//! so a cell the map holds occupied stays occupied under the footprint's
//! clearing and after clear_grid. The planner of a navigation in odom, the
//! collision test, the sweeps of the turns and the straight drives, and the
//! slowing all read it; the slowing ([`LiveGrid::slowed`]) checks the padded
//! footprint against each sensor's newest points and against the cost grid
//! along its outline. A navigation in map plans on the obstacles projected
//! onto its map's lattice ([`LiveGrid::obstacles_on`]).
//!
//! [`GridSlot`] holds the grid between the tasks: none from each start of the
//! odometry until describe_base's answer gives the footprint, then centered on
//! the odometry's origin, where the base stands at the start. While a goal
//! runs, and while the release burst after it streams, the grid rolls with
//! the base, as Nav2's local costmap rolls (nav2_bringup
//! params/nav2_params.yaml:246 `rolling_window: true`; nav2_costmap_2d
//! src/layered_costmap.cpp:144-148): it centers on the base when the goal is
//! accepted ([`GridSlot::roll_from`]) and with each reading, on where the
//! base stood at the reading's time, a scan's first ray, keeping what the
//! square still holds.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use camera_core::laser_scan::{LaserScan, Sweep};
use control_core::planar_pose::PlanarPose;
use control_core::positive_finite::PositiveFinite;
use control_core::twist::PlanarTwist;
use frame_tree::Transform;
use navigation_core::costmap::{ComposeError, compose};
use navigation_core::grid::{CostGrid, GridGeometry, PlacementNotFinite};
use navigation_core::inflation::Inflation;
use navigation_core::obstacles::{GridPoses, LiveObstacles, LiveObstaclesError, ObstacleLayer};
use navigation_core::saved_grid::SavedGrid;
use navigation_core::stop::{Stop, StopCause, StopDecision, StopError, StopParams, UnknownSource};
use occupancy_map::Bounds;

use crate::base::Base;
use crate::config::{GridSettings, NavigatorConfig};
use crate::follow::odometry_frame;

/// Why a describe_base answer gives no grid.
#[derive(Debug, thiserror::Error)]
pub enum GridError {
    #[error(transparent)]
    Live(#[from] LiveObstaclesError),
    #[error("the stop's simulation_time_step_s against its horizon")]
    Stop(#[from] StopError),
}

/// Why the grid gives no occupancy.
#[derive(Debug, thiserror::Error)]
pub enum OccupancyError {
    #[error(transparent)]
    Placement(#[from] PlacementNotFinite),
    #[error(transparent)]
    Compose(#[from] ComposeError),
}

/// The square `grid.side_m` on a side around `center`, from `grid.min_z_m` up
/// to the obstacle band's top.
pub fn bounds_around(grid: &GridSettings, center: [f64; 2]) -> Bounds {
    let half = grid.half_side_m();
    Bounds {
        min: [center[0] - half, center[1] - half, grid.min_z_m],
        max: [
            center[0] + half,
            center[1] + half,
            grid.heights.max_obstacle_height_m(),
        ],
    }
}

/// The loaded map's occupied cells over the grid of the floor, for a
/// navigation in map: the map's newest grid, its number among the grids the
/// navigator took, and where map stands in odom.
#[derive(Clone, Copy, Debug)]
pub struct MapOverlay<'a> {
    pub map: &'a SavedGrid,
    pub grid_seq: u64,
    pub odom_from_map: PlanarPose,
}

/// The occupancy as built, and the overlay it holds, by its grid's number and
/// placement.
struct Occupancy {
    overlay: Option<(u64, PlanarPose)>,
    grid: Arc<CostGrid>,
}

/// The grid around one center, the stop beside it, and the occupancy of the
/// grid as it stands, once built.
pub struct LiveGrid {
    live: LiveObstacles,
    stop: Stop,
    inflation: Inflation,
    settings: GridSettings,
    center: [f64; 2],
    /// For each sensor, whether a reading of it went in since the newest
    /// clearing, or since the grid was made.
    read_since_clearing: Vec<bool>,
    occupancy: Option<Occupancy>,
}

impl LiveGrid {
    /// The empty grid of `config` around `center` for `base`, whose `sensors`
    /// sensors give its readings, with the stop for `base`'s horizon.
    pub fn new(
        config: &NavigatorConfig,
        base: &Base,
        sensors: NonZeroUsize,
        center: [f64; 2],
    ) -> Result<Self, GridError> {
        let settings = config.grid;
        let stop = Stop::parse(StopParams {
            time_before_collision_s: PositiveFinite::parse(base.stop_horizon_s(&config.stop))
                .expect("the larger of a positive time and a sum of positive times"),
            simulation_time_step_s: config.stop.simulation_time_step_s,
            source_timeout_s: config.stop.source_timeout_s,
        })?;
        Ok(Self {
            live: LiveObstacles::new(
                odometry_frame(),
                bounds_around(&settings, center),
                settings.heights,
                settings.obstacle_max_range_m,
                base.padded_footprint.clone(),
                sensors,
            )?,
            stop,
            inflation: base.inflation.clone(),
            settings,
            center,
            read_since_clearing: vec![false; sensors.get()],
            occupancy: None,
        })
    }

    /// The grid centered on `at`, keeping what the new square holds.
    fn center_on(&mut self, at: [f64; 2]) -> Result<(), LiveObstaclesError> {
        if self.center == at {
            return Ok(());
        }
        self.live = self.live.recentered(bounds_around(&self.settings, at))?;
        self.center = at;
        self.occupancy = None;
        Ok(())
    }

    /// Insert the points of a depth frame of `source`, in the sensor's frame,
    /// the base at `base` in odom and the sensor at `sensor_on_base` in the
    /// robot frame at the frame's time, a point farther than `range` clearing
    /// to it; the frame arrived at `arrived_ns` on the node's clock, the
    /// source's newest reading for the stop.
    pub fn insert_depth_frame(
        &mut self,
        source: usize,
        points: &[[f32; 3]],
        base: PlanarPose,
        sensor_on_base: Transform,
        range: PositiveFinite,
        arrived_ns: i64,
    ) -> Result<(), UnknownSource> {
        self.live
            .insert_depth_frame(source, points, base, sensor_on_base, range)?;
        self.reading_went_in(source, arrived_ns)
    }

    /// Insert the hits of a scan of `source`, the base at `base`'s poses in
    /// odom at the scan's first and last rays and the sensor at
    /// `sensor_on_base` in the robot frame, each hit placed at its ray's time
    /// and `range`, the scan's farthest range, bounding its rays; the scan
    /// arrived at `arrived_ns` on the node's clock, the source's newest reading
    /// for the stop.
    pub fn insert_laser_scan(
        &mut self,
        source: usize,
        scan: &LaserScan,
        base: Sweep<PlanarPose>,
        sensor_on_base: Transform,
        range: PositiveFinite,
        arrived_ns: i64,
    ) -> Result<(), UnknownSource> {
        self.live
            .insert_laser_scan(source, scan, base, sensor_on_base, range)?;
        self.reading_went_in(source, arrived_ns)
    }

    /// A reading of `source` that arrived at `arrived_ns` went in: the stop's
    /// newest of the source, read since the newest clearing, and a change of
    /// the occupancy.
    fn reading_went_in(&mut self, source: usize, arrived_ns: i64) -> Result<(), UnknownSource> {
        self.stop.reading_arrived(&self.live, source, arrived_ns)?;
        self.read_since_clearing[source] = true;
        self.occupancy = None;
        Ok(())
    }

    /// Empty the grid, as clear_grid empties it: every cell counts as never
    /// seen. Each sensor's newest reading stays the stop's until its next.
    pub fn clear(&mut self) {
        self.live.clear();
        self.read_since_clearing.fill(false);
        self.occupancy = None;
    }

    /// Whether every sensor's reading went in since the newest clearing.
    pub fn read_since_clearing(&self) -> bool {
        self.read_since_clearing.iter().all(|read| *read)
    }

    /// The occupancy of the grid as it stands: what the sensors have seen,
    /// projected onto the grid of the voxel map's columns in odom, over the
    /// occupied cells of `overlay`'s map placed on that grid when a navigation
    /// in map reads it, and inflated for the padded footprint; built once per
    /// change of either.
    pub fn occupancy(
        &mut self,
        overlay: Option<MapOverlay>,
    ) -> Result<Arc<CostGrid>, OccupancyError> {
        let key = overlay.map(|overlay| (overlay.grid_seq, overlay.odom_from_map));
        let cached = self
            .occupancy
            .as_ref()
            .filter(|occupancy| occupancy.overlay == key);
        if let Some(occupancy) = cached {
            return Ok(occupancy.grid.clone());
        }
        let columns = self.live.columns();
        let layer = self.live.project(PlanarPose::IDENTITY, columns)?;
        let placed = overlay
            .map(|overlay| overlay.map.placed_on(overlay.odom_from_map, columns))
            .transpose()?;
        let grid = Arc::new(compose(placed.as_ref(), &layer, Some(&self.inflation))?);
        self.occupancy = Some(Occupancy {
            overlay: key,
            grid: grid.clone(),
        });
        Ok(grid)
    }

    /// The obstacles the sensors have seen on a grid of `geometry` in map, the
    /// voxel map, in odom, placed in map at `map_from_odom`: the obstacle layer
    /// of a plan in map.
    pub fn obstacles_on(
        &self,
        map_from_odom: PlanarPose,
        geometry: GridGeometry,
    ) -> Result<ObstacleLayer, PlacementNotFinite> {
        self.live.project(map_from_odom, geometry)
    }

    /// What the stop does with `command` at `now_ns`, the base at `base` in
    /// odom: it reads the occupancy, with `overlay`'s map for a navigation in
    /// map, and each sensor's newest reading.
    pub fn slowed(
        &mut self,
        command: PlanarTwist,
        base: PlanarPose,
        now_ns: i64,
        overlay: Option<MapOverlay>,
    ) -> Result<StopDecision, OccupancyError> {
        let occupancy = self.occupancy(overlay)?;
        let occupied = self.live.occupied_seen_from(
            &occupancy,
            GridPoses {
                grid_from_map: PlanarPose::IDENTITY,
                base,
            },
        );
        Ok(self.stop.decide(&self.live, command, &occupied, now_ns))
    }

    /// The index of each sensor silent at `now_ns`, in the sensors' order.
    pub fn silent_sensors(&self, now_ns: i64) -> Vec<usize> {
        self.stop
            .silent_sources(&self.live, now_ns)
            .into_iter()
            .map(|cause| match cause {
                StopCause::NoReading { source } | StopCause::StaleSource { source, .. } => source,
            })
            .collect()
    }
}

/// Whether the grid centers on the base with each reading: while a goal runs
/// or the release burst after it streams. The control loop stops it apart
/// from the grid's lock, so the step that ends a burst waits for no reading's
/// insertion. Clones read one flag.
#[derive(Clone, Debug, Default)]
pub struct Rolling(Arc<AtomicBool>);

impl Rolling {
    /// No goal runs and no release burst streams: the grid stays where it
    /// stands.
    pub fn stop(&self) {
        self.0.store(false, Ordering::Relaxed);
    }

    fn start(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    fn rolls(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// The grid between the tasks: none while the grid waits for describe_base.
#[derive(Default)]
pub struct GridSlot {
    grid: Option<LiveGrid>,
    rolling: Rolling,
}

impl GridSlot {
    /// Whether the grid rolls, which [`GridSlot::roll_from`] starts.
    pub fn rolling(&self) -> Rolling {
        self.rolling.clone()
    }

    pub fn grid_mut(&mut self) -> Option<&mut LiveGrid> {
        self.grid.as_mut()
    }

    pub fn set(&mut self, grid: LiveGrid) {
        self.grid = Some(grid);
    }

    /// Hold no grid: the odometry started over, and describe_base is read
    /// again.
    pub fn empty(&mut self) {
        self.grid = None;
    }

    /// Empty the grid, as clear_grid empties it.
    pub fn clear(&mut self) {
        if let Some(grid) = self.grid.as_mut() {
            grid.clear();
        }
    }

    /// A goal starts with the base at `base` in odom: center the grid on it,
    /// keeping what the square still holds, and center it on the base with
    /// each reading from now on.
    pub fn roll_from(&mut self, base: [f64; 2]) -> Result<(), LiveObstaclesError> {
        self.rolling.start();
        self.grid
            .as_mut()
            .map_or(Ok(()), |grid| grid.center_on(base))
    }

    /// The grid a reading taken with the base at `base` in odom goes into:
    /// centered on `base` while the grid rolls; none while the grid waits for
    /// describe_base.
    pub fn grid_for_reading(
        &mut self,
        base: PlanarPose,
    ) -> Result<Option<&mut LiveGrid>, LiveObstaclesError> {
        let rolls = self.rolling.rolls();
        let Some(grid) = self.grid.as_mut() else {
            return Ok(None);
        };
        if rolls {
            grid.center_on([base.x(), base.y()])?;
        }
        Ok(Some(grid))
    }

    /// The index of each sensor silent at `now_ns`; every sensor of
    /// `sensors` while no grid is held.
    pub fn silent_sensors(&self, now_ns: i64, sensors: usize) -> Vec<usize> {
        match &self.grid {
            Some(grid) => grid.silent_sensors(now_ns),
            None => (0..sensors).collect(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use std::f64::consts::FRAC_PI_2;

    use navigation_core::grid::{INSCRIBED, LETHAL};

    use crate::base::tests::base;
    use crate::config::tests::config;

    pub(crate) fn grid() -> LiveGrid {
        LiveGrid::new(&config(), &base(1.0), NonZeroUsize::MIN, [0.0, 0.0]).expect("a grid")
    }

    /// The points of a wall of returns 1 m ahead of a sensor on the base's
    /// origin looking along +x, from 0.15 m to 0.35 m above the floor and
    /// 0.3 m to each side, in the sensor's frame.
    pub(crate) fn wall_ahead() -> Vec<[f32; 3]> {
        (0..=12)
            .flat_map(|j| {
                (0..=4).map(move |k| [1.0, -0.3 + 0.05 * j as f32, 0.15 + 0.05 * k as f32])
            })
            .collect()
    }

    fn sensor_on_base() -> Transform {
        Transform::IDENTITY
    }

    fn range() -> PositiveFinite {
        PositiveFinite::parse(4.0).expect("a range")
    }

    // A reading of a wall marks its cells, the clearing forgets them, and a
    // reading after the clearing marks them again.
    #[test]
    fn a_reading_marks_the_grid_and_a_clearing_forgets_it() {
        let mut grid = grid();
        let wall = |grid: &mut LiveGrid| {
            let occupancy = grid.occupancy(None).expect("an occupancy");
            let cell = occupancy.geometry().cell_of([1.0, 0.0]).expect("a cell");
            occupancy.cost(cell) == LETHAL
        };
        assert!(!grid.read_since_clearing());
        grid.insert_depth_frame(
            0,
            &wall_ahead(),
            PlanarPose::IDENTITY,
            sensor_on_base(),
            range(),
            0,
        )
        .expect("source 0");
        assert!(wall(&mut grid) && grid.read_since_clearing());
        grid.clear();
        assert!(!wall(&mut grid) && !grid.read_since_clearing());
        grid.insert_depth_frame(
            0,
            &wall_ahead(),
            PlanarPose::IDENTITY,
            sensor_on_base(),
            range(),
            1,
        )
        .expect("source 0");
        assert!(wall(&mut grid));
    }

    // A sensor is silent before its first reading and once its newest is
    // older than the source timeout.
    #[test]
    fn a_sensor_is_silent_without_a_reading_within_the_source_timeout() {
        let mut grid = grid();
        assert_eq!(grid.silent_sensors(0), [0]);
        grid.insert_depth_frame(0, &[], PlanarPose::IDENTITY, sensor_on_base(), range(), 0)
            .expect("source 0");
        assert!(grid.silent_sensors(500_000_000).is_empty());
        assert_eq!(grid.silent_sensors(500_000_001), [0]);
    }

    /// Whether the occupancy of `grid` holds a cell at `point` in odom.
    pub(crate) fn covers(grid: &mut LiveGrid, point: [f64; 2]) -> bool {
        let occupancy = grid.occupancy(None).expect("an occupancy");
        occupancy.geometry().cell_of(point).is_some()
    }

    /// Insert `points`, a reading of the sensor on the base's origin, into
    /// the grid of `slot` for a reading with the base at `x` along odom's x,
    /// heading along it.
    pub(crate) fn read_at(slot: &mut GridSlot, x: f64, points: &[[f32; 3]]) {
        let at = PlanarPose::new(x, 0.0, 0.0).expect("a pose");
        slot.grid_for_reading(at)
            .expect("the grid recenters")
            .expect("a grid")
            .insert_depth_frame(0, points, at, sensor_on_base(), range(), 0)
            .expect("source 0");
    }

    // While a goal runs the grid centers on the base at its acceptance and
    // with each reading, keeping what the square still holds: a wall seen
    // from the start stays marked once the base has moved 2 m along x; once
    // the grid stops rolling, a reading leaves it where it stands.
    #[test]
    fn the_grid_rolls_with_the_base_while_a_goal_runs() {
        let mut slot = GridSlot::default();
        slot.set(grid());
        read_at(&mut slot, 0.0, &wall_ahead());
        assert!(!covers(slot.grid_mut().expect("a grid"), [4.0, 0.0]));
        slot.roll_from([0.5, 0.0]).expect("the grid recenters");
        assert!(covers(slot.grid_mut().expect("a grid"), [2.9, 0.0]));
        read_at(&mut slot, 2.0, &[]);
        let grid = slot.grid_mut().expect("a grid");
        assert!(covers(grid, [4.4, 0.0]) && !covers(grid, [-0.6, 0.0]));
        let occupancy = grid.occupancy(None).expect("an occupancy");
        let wall = occupancy.geometry().cell_of([1.0, 0.0]).expect("a cell");
        assert_eq!(occupancy.cost(wall), LETHAL, "the wall stays marked");
        slot.rolling().stop();
        read_at(&mut slot, 3.0, &[]);
        assert!(covers(slot.grid_mut().expect("a grid"), [-0.4, 0.0]));
    }

    /// A map's grid of 0.05 m cells, `columns` by `rows` from `origin` in map,
    /// every cell occupied where `occupied` holds of its index and free
    /// elsewhere.
    fn map_grid(
        columns: u32,
        rows: u32,
        origin: [f64; 2],
        occupied: impl Fn(u32, u32) -> bool,
    ) -> SavedGrid {
        let cells: Vec<u8> = (0..rows)
            .flat_map(|y| (0..columns).map(move |x| (x, y)))
            .map(|(x, y)| if occupied(x, y) { 100 } else { 0 })
            .collect();
        let cell = PositiveFinite::parse(0.05).expect("a cell");
        SavedGrid::from_cells(columns, rows, cell, origin, &cells).expect("a map's grid")
    }

    /// The cost of the cell of `grid`'s occupancy, with `overlay`, that holds
    /// `point` in odom.
    fn cost_at(grid: &mut LiveGrid, overlay: Option<MapOverlay>, point: [f64; 2]) -> u8 {
        let occupancy = grid.occupancy(overlay).expect("an occupancy");
        occupancy.cost(occupancy.geometry().cell_of(point).expect("a cell"))
    }

    // base_navigation's map-frame navigation: the occupancy counts occupied a
    // cell the loaded map holds occupied where the sensors' rays hold it free,
    // the map placed where map stands in odom, and keeps it after a clearing.
    #[test]
    fn the_occupancy_of_a_map_navigation_holds_the_maps_occupied_cells() {
        let mut grid = grid();
        grid.insert_depth_frame(
            0,
            &wall_ahead(),
            PlanarPose::IDENTITY,
            sensor_on_base(),
            range(),
            0,
        )
        .expect("source 0");
        // map turned a quarter turn in odom and moved: its cell (2, 1), from
        // (1.0, 0.35) to (1.05, 0.4) in map, covers (0.6, 0.0) to (0.65, 0.05)
        // in odom, where the rays to the wall 1 m ahead pass.
        let map = map_grid(4, 4, [0.9, 0.3], |x, y| (x, y) == (2, 1));
        let overlay = MapOverlay {
            map: &map,
            grid_seq: 1,
            odom_from_map: PlanarPose::new(1.0, -1.0, FRAC_PI_2).expect("a pose"),
        };
        let between = [0.62, 0.02];
        assert!(cost_at(&mut grid, None, between) < INSCRIBED);
        assert_eq!(cost_at(&mut grid, Some(overlay), between), LETHAL);
        assert_eq!(cost_at(&mut grid, None, [1.02, 0.02]), LETHAL, "the wall");
        let moved = MapOverlay {
            odom_from_map: PlanarPose::new(1.0, -0.5, FRAC_PI_2).expect("a pose"),
            ..overlay
        };
        assert!(
            cost_at(&mut grid, Some(moved), between) < INSCRIBED,
            "the occupancy is built again for a new placement"
        );
        grid.clear();
        assert_eq!(cost_at(&mut grid, Some(overlay), between), LETHAL);
    }

    // base_navigation's stop: for a map-frame navigation the slowing reads the
    // loaded map's occupied cells, so a wall the map alone holds, ahead where
    // the sensors' rays hold the grid free, slows the base before it.
    #[test]
    fn the_stop_of_a_map_navigation_slows_before_a_wall_the_map_alone_holds() {
        let mut grid = grid();
        let now_ns = 1_000_000_000;
        // A wall 2 m ahead, past the stop's reach, whose rays clear the floor
        // up to it.
        let far_wall: Vec<[f32; 3]> = wall_ahead()
            .into_iter()
            .map(|[_, y, z]| [2.0, y, z])
            .collect();
        grid.insert_depth_frame(
            0,
            &far_wall,
            PlanarPose::IDENTITY,
            sensor_on_base(),
            range(),
            now_ns,
        )
        .expect("source 0");
        // A wall from 0.5 m to 0.6 m ahead, 1 m wide, in a map placed on odom.
        let map = map_grid(2, 20, [0.5, -0.5], |_, _| true);
        let overlay = MapOverlay {
            map: &map,
            grid_seq: 1,
            odom_from_map: PlanarPose::IDENTITY,
        };
        let ahead = PlanarTwist::new(0.5, 0.0, 0.0).expect("a twist");
        let without = grid
            .slowed(ahead, PlanarPose::IDENTITY, now_ns, None)
            .expect("a decision");
        assert_eq!(without.velocity(), ahead, "{without:?}");
        let with = grid
            .slowed(ahead, PlanarPose::IDENTITY, now_ns, Some(overlay))
            .expect("a decision");
        assert!(
            with.velocity().linear_x() < ahead.linear_x(),
            "the map's wall slows the base: {with:?}"
        );
    }
}
