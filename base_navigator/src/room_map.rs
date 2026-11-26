//! room_map's loaded map as the navigator reads it, for a navigation whose target is in
//! the map frame. Two listeners follow the link and ask nothing:
//!
//! - map_grids: each grid in map is of a load of the loaded map ([`MapLoad`]), its map's
//!   name and its map_load_timestamp; the newest is kept ([`LoadedGrid`]). A grid whose
//!   extent holds `columns` by `rows` cells of its cell's side from its least corner
//!   becomes navigation_core's saved grid ([`MapGrid`]); a new map's grid before its
//!   mapping read a scan, with no extent and no cell, builds none. A grid in another
//!   frame, with a cell off 0, 100 and 255, whose extent does not hold its cells, or with
//!   cells and no extent, is dropped and logged, its reason kept until a grid reads, and
//!   the grid before it stays.
//!   The first grid of each load is logged.
//! - map_poses: each map_poses in map is placed as it arrives ([`Estimate`]): the tree
//!   gives the robot frame in odom at its timestamp, and `map_from_odom = map_pose ∘
//!   odom_pose⁻¹` is the transform room_map publishes beside it. Each estimate records
//!   the start of the base's odometry the newest drive_states reported when the tree
//!   placed it, and its arrival on the node's clock. A map_poses the tree cannot place is
//!   dropped and logged with the lookup's failure kind, and its reason stays until a
//!   map_poses is placed.
//!
//! A map-frame goal asks get_map_state, the generated poll raced against the source
//! timeout on the node's clock ([`RoomLink::state`]), and waits, up to the source timeout
//! from the answer, for a map_poses that arrives after the ask and for a grid of the
//! answer's map ([`RoomMap::admitted_load`]), as Nav2's planner waits for its static
//! layer's first map; the navigation runs on the load of that map's newest grid. At each
//! step of the navigation, [`RoomMap::place`] gives where map stands in odom, by the newest estimate
//! of the navigation's odometry, and the map's newest grid; or why the navigation ends
//! with not_localized, in this order: the loaded map changed (the newest grid is of
//! another load, or the estimate names another map), the base is not localized, or no
//! map_poses of the odometry arrived within the source timeout. A plan of the navigation
//! runs on [`PlanLattice`]: the map's newest grid, padded with unknown cells out to the
//! extent the goal was admitted against. The follow, the stop and the turns read the map's
//! occupied cells on the grid of the floor, as base_navigation asks, a departure from
//! Nav2, whose local costmap holds no static layer (nav2_bringup
//! params/nav2_params.yaml:251) where its global costmap holds one (:305).

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use control_core::geometry::Pose;
use control_core::latched_log::LatchedLog;
use control_core::node_clock::NodeClock;
use control_core::planar_pose::PlanarPose;
use control_core::positive_finite::PositiveFinite;
use control_core::receiving::receive;
use control_core::robot_tree::RobotTree;
use control_core::sync::lock_ignoring_poison;
use control_core::time::seconds;
use frame_tree::Stamp;
use navigation_core::grid::GridGeometry;
use navigation_core::saved_grid::{MapExtent, SavedGrid, SavedGridError};
use peppygen::consumed_services::room_map::get_map_state;
use peppygen::consumed_topics::room_map::{map_grids, map_poses};
use peppygen::{NodeRunner, ProducerRef};
use tokio::sync::watch;
use tracing::{error, info};

use crate::base::ANSWER_TIMEOUT;
use crate::clock;
use crate::follow::{MAP_FRAME, robot_in_odom};
use crate::link::Link;

/// An unknown cell's value in map_grids: the occupancy grid's -1 read unsigned.
const UNKNOWN_VALUE: u8 = 255;

/// How near a whole number of cells an edge of an extent lies, in cells, to count as on
/// that cell's side: room_mapper writes each edge as an origin plus a count of cells
/// times the cell, so an edge lies a rounding error off the side.
const ON_A_CELL_SIDE_CELLS: f64 = 1e-6;

/// `cells`, a distance in cells, on the nearest cell side when it lies within
/// [`ON_A_CELL_SIDE_CELLS`] of one.
fn snapped(cells: f64) -> f64 {
    let side = cells.round();
    if (cells - side).abs() <= ON_A_CELL_SIDE_CELLS {
        side
    } else {
        cells
    }
}

/// A load of room_map's loaded map, as map_grids gives it: the map's name, and when
/// room_map loaded it, on the wall clock of the mapper's machine. Each load_map and each
/// start_mapping is a new load; finish_mapping keeps its mapping's.
#[derive(Debug, PartialEq)]
pub struct MapLoad {
    pub map_name: String,
    pub loaded_at: SystemTime,
}

/// The newest map_grids the navigator took: the load it is of, and its grid of cells,
/// none for a new map's grid before its mapping read a scan.
#[derive(Debug)]
pub struct LoadedGrid {
    pub load: MapLoad,
    pub grid: Option<Arc<MapGrid>>,
}

impl LoadedGrid {
    /// The grid of `message`, the `seq`th the navigator took, or why it is dropped.
    pub fn parse(message: map_grids::Message, seq: u64) -> Result<Self, String> {
        if message.frame != MAP_FRAME {
            return Err(format!(
                "the grid is in the frame {:?}, where room_map gives the loaded map's grid in \
                 {MAP_FRAME:?}",
                message.frame
            ));
        }
        let load = MapLoad {
            map_name: message.map_name.clone(),
            loaded_at: message.map_load_timestamp,
        };
        let grid = match message.extent {
            Some(extent) => Some(Arc::new(MapGrid::parse(extent, message, seq)?)),
            None => {
                refuse_cells_without_extent(&message)?;
                None
            }
        };
        Ok(Self { load, grid })
    }
}

/// Refuses a grid with no extent that gives a column, a row or a cell: room_map gives
/// a new map's grid before its first scan with none.
fn refuse_cells_without_extent(message: &map_grids::Message) -> Result<(), String> {
    if message.columns == 0 && message.rows == 0 && message.cells.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the grid has no extent and gives {} columns, {} rows and {} cells: room_map gives a \
         grid with no extent only before a new map's first scan, with no column, row or cell",
        message.columns,
        message.rows,
        message.cells.len()
    ))
}

/// The loaded map's grid of one map_grids that has an extent: its cells as map_grids
/// gives them, the saved grid they make in map, and its number among the grids the
/// navigator took.
#[derive(Debug)]
pub struct MapGrid {
    cells: Vec<u8>,
    pub saved: SavedGrid,
    pub seq: u64,
}

impl MapGrid {
    /// The grid of `message`, whose extent is `extent`, the `seq`th the navigator took, or
    /// why it is dropped.
    fn parse(extent: [f64; 4], message: map_grids::Message, seq: u64) -> Result<Self, String> {
        let extent = MapExtent::parse(extent).map_err(|e| e.to_string())?;
        let cell = PositiveFinite::parse_field("cell_size_m", message.cell_size_m)?;
        for (index, axis, cells, given) in [
            (0, "x", "columns", message.columns),
            (1, "y", "rows", message.rows),
        ] {
            let spanned = ((extent.max()[index] - extent.min()[index]) / cell.get()).round();
            if spanned != f64::from(given) {
                return Err(format!(
                    "the extent {:?} spans {spanned} cells of {} m along {axis}, where the grid \
                     gives {given} {cells}",
                    extent,
                    cell.get()
                ));
            }
        }
        let saved = SavedGrid::from_cells(
            message.columns,
            message.rows,
            cell,
            extent.min(),
            &message.cells,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            cells: message.cells,
            saved,
            seq,
        })
    }

    /// The value map_grids gave the cell `x` along and `y` up, or the unknown value off
    /// the grid.
    fn value_at(&self, x: i64, y: i64) -> u8 {
        let geometry = self.saved.grid().geometry();
        let (Ok(x), Ok(y)) = (u32::try_from(x), u32::try_from(y)) else {
            return UNKNOWN_VALUE;
        };
        if x >= geometry.size_x() || y >= geometry.size_y() {
            return UNKNOWN_VALUE;
        }
        self.cells[y as usize * geometry.size_x() as usize + x as usize]
    }
}

/// The grid a navigation in map plans on: the cells of the extent it was admitted
/// against and of its map's newest grid, on that grid's lattice, and where that grid's
/// first cell lies on it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanLattice {
    pub geometry: GridGeometry,
    grid_at: [i64; 2],
}

impl PlanLattice {
    /// The static layer of `grid`, the lattice's map grid: its cells where it holds them,
    /// and unknown every other cell, as Nav2's static layer leaves a cell its map does not
    /// cover.
    pub fn static_layer(&self, grid: &MapGrid) -> Result<SavedGrid, SavedGridError> {
        let geometry = self.geometry;
        let [at_x, at_y] = self.grid_at;
        let cells: Vec<u8> = (0..i64::from(geometry.size_y()))
            .flat_map(|y| {
                (0..i64::from(geometry.size_x())).map(move |x| grid.value_at(x - at_x, y - at_y))
            })
            .collect();
        let cell = PositiveFinite::parse(geometry.resolution_m())
            .expect("a grid's resolution is a positive finite number");
        SavedGrid::from_cells(
            geometry.size_x(),
            geometry.size_y(),
            cell,
            geometry.origin(),
            &cells,
        )
    }
}

/// The loaded map a map-frame goal is admitted on: its name, the extent get_map_state
/// gave it, and its cell's side.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedMap {
    pub name: String,
    pub extent: MapExtent,
    pub cell_size_m: PositiveFinite,
}

/// What a navigation in map runs on: its map, and when room_map loaded it, the
/// map_load_timestamp of the map's newest grid at the navigation's admission.
#[derive(Clone, Debug, PartialEq)]
pub struct MapRun {
    pub map: AdmittedMap,
    pub loaded_at: SystemTime,
}

impl MapRun {
    /// The lattice a plan of this run reads with `grid`, the map's newest grid, none
    /// while it has no cell: from the grid's lattice, or from the extent's least corner at
    /// get_map_state's cell, the fewest cells that hold the grid and the closed extent, a
    /// point on the extent's far edge in the last cell.
    pub fn plan_lattice(&self, grid: Option<&MapGrid>) -> Result<PlanLattice, String> {
        let extent = self.map.extent;
        let (origin, cell, held) = match grid {
            Some(grid) => {
                let geometry = grid.saved.grid().geometry();
                (
                    geometry.origin(),
                    geometry.resolution_m(),
                    [geometry.size_x(), geometry.size_y()].map(i64::from),
                )
            }
            None => (extent.min(), self.map.cell_size_m.get(), [0, 0]),
        };
        let cell_of =
            |axis: usize, point: f64| snapped((point - origin[axis]) / cell).floor() as i64;
        let first = [0, 1].map(|axis| cell_of(axis, extent.min()[axis]).min(0));
        let last = [0, 1].map(|axis| cell_of(axis, extent.max()[axis]).max(held[axis] - 1));
        let size = [0, 1].map(|axis| u32::try_from(last[axis] - first[axis] + 1));
        let [Ok(size_x), Ok(size_y)] = size else {
            return Err(format!(
                "the extent {extent:?} and the map's grid span more cells of {cell} m than a grid \
                 holds"
            ));
        };
        let corner = [0, 1].map(|axis| origin[axis] + first[axis] as f64 * cell);
        let geometry = GridGeometry::new(
            size_x,
            size_y,
            PositiveFinite::parse(cell).expect("a grid's resolution is a positive finite number"),
            corner,
        )
        .map_err(|e| format!("the grid that holds the extent {extent:?}: {e}"))?;
        Ok(PlanLattice {
            geometry,
            grid_at: first.map(|cell| -cell),
        })
    }
}

/// get_map_state's loaded map, parsed: its name, its cell's side, and its extent, none
/// while a new map's mapping has read no scan.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedMap {
    pub name: String,
    pub cell_size_m: PositiveFinite,
    pub extent: Option<MapExtent>,
}

/// get_map_state's answer as a map-frame goal reads it: the loaded map, none while no map
/// is loaded; whether a set_initial_pose waits for its scan; and whether the base is
/// localized.
#[derive(Clone, Debug, PartialEq)]
pub struct MapState {
    pub map: Option<LoadedMap>,
    pub waiting: bool,
    pub localized: bool,
}

impl MapState {
    /// The state `answer` gives, or why it gives none: a cell's side that is not a
    /// positive number, or an extent whose edges make no rectangle.
    pub fn parse(answer: get_map_state::ResponseData) -> Result<Self, String> {
        let map = answer
            .map
            .map(|map| {
                let cell_size_m = PositiveFinite::parse_field("cell_size_m", map.cell_size_m)?;
                let extent = map
                    .extent
                    .map(MapExtent::parse)
                    .transpose()
                    .map_err(|e| e.to_string())?;
                Ok::<_, String>(LoadedMap {
                    name: map.map_name,
                    cell_size_m,
                    extent,
                })
            })
            .transpose()?;
        Ok(Self {
            map,
            waiting: answer.waiting,
            localized: answer.localized,
        })
    }
}

/// One map_poses, placed: its map, where map stands in odom and odom in map, whether it
/// reports the base localized, the start of the base's odometry it was placed in, and its
/// arrival, on the node's clock and in the order of the map_poses that arrived.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    pub map_name: String,
    pub map_from_odom: PlanarPose,
    pub odom_from_map: PlanarPose,
    pub localized: bool,
    pub odometry_start: SystemTime,
    pub arrived_ns: i64,
    pub arrival: u64,
}

impl Estimate {
    /// The estimate of `message`, the robot frame at `odom_pose` in odom at its timestamp,
    /// in the odometry started at `odometry_start`, the `arrival`th map_poses, at
    /// `arrived_ns`: or why it is dropped.
    pub fn placed(
        message: &map_poses::Message,
        odom_pose: PlanarPose,
        odometry_start: SystemTime,
        arrived_ns: i64,
        arrival: u64,
    ) -> Result<Self, String> {
        if message.frame != MAP_FRAME {
            return Err(format!(
                "the pose is in the frame {:?}, where room_map gives the robot frame's place in \
                 the loaded map in {MAP_FRAME:?}",
                message.frame
            ));
        }
        let map_pose = Pose::parse(message.position, message.orientation)
            .and_then(Pose::planar)
            .map_err(|e| format!("the pose in map: {e}"))?;
        let not_finite = |e| format!("the transform map to odom: {e}");
        let base_from_odom = PlanarPose::IDENTITY
            .seen_from(odom_pose)
            .map_err(not_finite)?;
        let map_from_odom = map_pose.compose(base_from_odom).map_err(not_finite)?;
        let odom_from_map = PlanarPose::IDENTITY
            .seen_from(map_from_odom)
            .map_err(not_finite)?;
        Ok(Self {
            map_name: message.map_name.clone(),
            map_from_odom,
            odom_from_map,
            localized: message.localized,
            odometry_start,
            arrived_ns,
            arrival,
        })
    }
}

/// Where map stands in odom at a step of a navigation in map, and the map's newest grid,
/// none while it has no cell.
#[derive(Clone, Debug)]
pub struct MapPlacement {
    pub map_from_odom: PlanarPose,
    pub odom_from_map: PlanarPose,
    pub grid: Option<Arc<MapGrid>>,
}

/// What the listeners hold.
#[derive(Debug, Default)]
struct Held {
    /// The newest grid that read, none before the first.
    grid: Option<LoadedGrid>,
    /// How many grids the listener took.
    grids_taken: u64,
    /// Why the newest map_grids was dropped, while none read after it.
    grid_dropped: Option<String>,
    /// How many map_poses arrived.
    arrivals: u64,
    newest: Option<Estimate>,
    /// The newest estimate of the odometry before the newest estimate's.
    of_the_odometry_before: Option<Estimate>,
    /// Why the newest map_poses was dropped, while none was placed after it.
    dropped: Option<String>,
}

impl Held {
    fn take_estimate(&mut self, estimate: Estimate) {
        let odometry_start = estimate.odometry_start;
        let replaced = self.newest.replace(estimate);
        if let Some(older) = replaced.filter(|older| older.odometry_start != odometry_start) {
            self.of_the_odometry_before = Some(older);
        }
        self.dropped = None;
    }

    /// The newest estimate placed in the odometry started at `odometry_start`.
    fn of_odometry(&self, odometry_start: SystemTime) -> Option<&Estimate> {
        [&self.newest, &self.of_the_odometry_before]
            .into_iter()
            .flatten()
            .find(|estimate| estimate.odometry_start == odometry_start)
    }

    /// The load of the newest grid when it is of `map_name`, with an estimate of that map
    /// placed in the odometry started at `odometry_start`, arrived after the first `after`
    /// map_poses and reporting the base localized; or what a map-frame goal still waits
    /// for, in words.
    fn admitted_load(
        &self,
        map_name: &str,
        odometry_start: SystemTime,
        after: u64,
    ) -> Result<SystemTime, String> {
        let localized = self.newest.as_ref().is_some_and(|estimate| {
            estimate.map_name == map_name
                && estimate.odometry_start == odometry_start
                && estimate.arrival > after
                && estimate.localized
        });
        let loaded_at = self
            .grid
            .as_ref()
            .filter(|grid| grid.load.map_name == map_name)
            .map(|grid| grid.load.loaded_at);
        if let (true, Some(loaded_at)) = (localized, loaded_at) {
            return Ok(loaded_at);
        }
        let dropped = |what: &str, reason: &Option<String>| {
            reason
                .as_ref()
                .map(|reason| format!(", the newest {what} was dropped: {reason}"))
                .unwrap_or_default()
        };
        let waits: Vec<String> = [
            (!localized).then(|| {
                format!(
                    "no map_poses of map {map_name} reporting the base localized and placed in \
                     the base's odometry arrived after get_map_state was asked{}",
                    dropped("map_poses", &self.dropped)
                )
            }),
            loaded_at.is_none().then(|| {
                format!(
                    "no map_grids of map {map_name} arrived{}",
                    dropped("map_grids", &self.grid_dropped)
                )
            }),
        ]
        .into_iter()
        .flatten()
        .collect();
        Err(waits.join("; "))
    }
}

/// The loaded map as the listeners hold it, shared with the goals' checks and the control
/// loop; each map_poses' arrival, placed or dropped, and each grid that reads changes
/// [`RoomMap::changes`].
#[derive(Debug)]
pub struct RoomMap {
    held: Mutex<Held>,
    changes: watch::Sender<u64>,
}

impl Default for RoomMap {
    fn default() -> Self {
        Self {
            held: Mutex::default(),
            changes: watch::Sender::new(0),
        }
    }
}

impl RoomMap {
    /// Take the grid of `message`, or why it is dropped, the grid before staying: for
    /// the first grid of its load, the map's name and its grid's size.
    pub fn take_grid(&self, message: map_grids::Message) -> Result<Option<String>, String> {
        let first = {
            let mut held = lock_ignoring_poison(&self.held);
            held.grids_taken += 1;
            let taken = LoadedGrid::parse(message, held.grids_taken)
                .inspect_err(|reason| held.grid_dropped = Some(reason.clone()))?;
            let first = (held.grid.as_ref().map(|grid| &grid.load) != Some(&taken.load))
                .then(|| first_of_its_load(&taken));
            held.grid = Some(taken);
            held.grid_dropped = None;
            first
        };
        self.changes.send_modify(|changes| *changes += 1);
        Ok(first)
    }

    /// The number of a map_poses that arrived, in the order of their arrivals.
    fn arrived(&self) -> u64 {
        let mut held = lock_ignoring_poison(&self.held);
        held.arrivals += 1;
        held.arrivals
    }

    /// Take a map_poses' estimate, or why it is dropped.
    pub fn take_estimate(&self, placed: Result<Estimate, String>) {
        {
            let mut held = lock_ignoring_poison(&self.held);
            match placed {
                Ok(estimate) => held.take_estimate(estimate),
                Err(reason) => held.dropped = Some(reason),
            }
        }
        self.changes.send_modify(|changes| *changes += 1);
    }

    /// How many map_poses arrived so far.
    pub fn arrivals(&self) -> u64 {
        lock_ignoring_poison(&self.held).arrivals
    }

    /// The load a map-frame goal on `map_name` is admitted on: once a grid of `map_name`
    /// is the newest grid and an estimate of `map_name` arrived after the first `after`
    /// map_poses, placed in the odometry the newest drive_states of `link` reports and
    /// reporting the base localized, the map_load_timestamp of that grid. Waits on `clock`
    /// until it reads `deadline_ns`; then what the goal still waits for, in words.
    pub async fn admitted_load(
        &self,
        map_name: &str,
        after: u64,
        link: &Mutex<Link>,
        clock: &NodeClock,
        deadline_ns: i64,
    ) -> Result<SystemTime, String> {
        let mut changes = self.changes.subscribe();
        loop {
            changes.borrow_and_update();
            let odometry_start = lock_ignoring_poison(link)
                .newest()
                .expect("a goal is checked once drive_states arrived")
                .state
                .odometry_start;
            let admitted =
                lock_ignoring_poison(&self.held).admitted_load(map_name, odometry_start, after);
            let Err(waits) = admitted else {
                return admitted;
            };
            if clock::now_ns(clock) >= deadline_ns {
                return Err(waits);
            }
            let deadline = u64::try_from(deadline_ns).unwrap_or(u64::MAX);
            tokio::select! {
                () = clock.until(deadline) => {}
                // The sender lives with `self`, so the wait never sees it close.
                _ = changes.changed() => {}
            }
        }
    }

    /// Where map stands in odom at `now_ns` for a step of the navigation `run`, the base's
    /// odometry started at `odometry_start`, by the newest estimate placed in that
    /// odometry, and the map's newest grid; or why the navigation ends with not_localized,
    /// the first that holds of: the loaded map changed, the base is not localized, and no
    /// map_poses of the odometry arrived within `source_timeout_ns`.
    pub fn place(
        &self,
        run: &MapRun,
        odometry_start: SystemTime,
        now_ns: i64,
        source_timeout_ns: i64,
    ) -> Result<MapPlacement, String> {
        let held = lock_ignoring_poison(&self.held);
        let map_name = &run.map.name;
        if let Some(changed) = held
            .grid
            .as_ref()
            .and_then(|grid| load_changed(run, &grid.load))
        {
            return Err(changed);
        }
        let estimate = held.of_odometry(odometry_start);
        if let Some(changed) = estimate.and_then(|estimate| map_named_otherwise(run, estimate)) {
            return Err(changed);
        }
        if estimate.is_some_and(|estimate| !estimate.localized) {
            return Err(format!(
                "map_poses reports the base not localized in map {map_name}: its scans stop \
                 matching the map, or a search runs"
            ));
        }
        let fresh = estimate.filter(|estimate| now_ns - estimate.arrived_ns <= source_timeout_ns);
        let Some(estimate) = fresh else {
            let since = estimate.map_or_else(
                || "since the base's odometry started".to_string(),
                |estimate| format!("for {} s", seconds(now_ns - estimate.arrived_ns)),
            );
            let dropped = held
                .dropped
                .as_ref()
                .map(|reason| format!("; the newest map_poses was dropped: {reason}"))
                .unwrap_or_default();
            return Err(format!(
                "no map_poses of map {map_name} placed in the base's odometry arrived {since}, \
                 longer than source_timeout_s {} s: room_map holds no estimate, as after a \
                 remove_map or a load_map, or the lidar or the mapper is silent{dropped}",
                seconds(source_timeout_ns)
            ));
        };
        Ok(MapPlacement {
            map_from_odom: estimate.map_from_odom,
            odom_from_map: estimate.odom_from_map,
            grid: held.grid.as_ref().and_then(|grid| grid.grid.clone()),
        })
    }
}

/// What the log says of `taken`, the first grid of its load: its map, and its grid's size
/// or that it has no cell yet.
fn first_of_its_load(taken: &LoadedGrid) -> String {
    let map_name = &taken.load.map_name;
    let Some(grid) = &taken.grid else {
        return format!("map {map_name}, with no cell before its mapping reads a scan");
    };
    let geometry = grid.saved.grid().geometry();
    format!(
        "map {map_name}, {} by {} cells of {} m",
        geometry.size_x(),
        geometry.size_y(),
        geometry.resolution_m()
    )
}

/// Why `load`, the newest grid's, says the loaded map of the navigation `run` changed:
/// it is of another map, or of another load of the run's map, as a load_map or a
/// start_mapping gives; none while it is the run's load.
fn load_changed(run: &MapRun, load: &MapLoad) -> Option<String> {
    let map_name = &run.map.name;
    if load.map_name != *map_name {
        return Some(format!(
            "the loaded map changed: map_grids gives map {}, where the navigation runs on map \
             {map_name}: a load_map or a start_mapping changed it",
            load.map_name
        ));
    }
    (load.loaded_at != run.loaded_at).then(|| {
        format!(
            "the loaded map changed: map_grids gives map {map_name} loaded again, its \
             map_load_timestamp other than the navigation's: a load_map loaded it again"
        )
    })
}

/// Why `estimate` says the loaded map of the navigation `run` changed: it names another
/// map; none while it names the run's.
fn map_named_otherwise(run: &MapRun, estimate: &Estimate) -> Option<String> {
    let map_name = &run.map.name;
    (estimate.map_name != *map_name).then(|| {
        format!(
            "the loaded map changed: map_poses names map {}, where the navigation runs on map \
             {map_name}: a load_map or a start_mapping changed it",
            estimate.map_name
        )
    })
}

/// The navigator's room_map link: the loaded map the listeners hold, and the producer the
/// goals' checks ask get_map_state of.
pub struct RoomLink {
    pub map: Arc<RoomMap>,
    runner: Arc<NodeRunner>,
    producer: ProducerRef,
}

impl RoomLink {
    pub fn new(runner: Arc<NodeRunner>, producer: ProducerRef) -> Self {
        Self {
            map: Arc::default(),
            runner,
            producer,
        }
    }

    /// get_map_state's answer, raced against `timeout_ns` on `clock`, with the number of
    /// map_poses that had arrived when it was asked; or why it gives none.
    pub async fn state(
        &self,
        clock: &NodeClock,
        timeout_ns: i64,
    ) -> Result<(MapState, u64), String> {
        let deadline =
            u64::try_from(clock::now_ns(clock).saturating_add(timeout_ns)).unwrap_or(u64::MAX);
        let arrivals = self.map.arrivals();
        let asked = get_map_state::poll(&self.runner, &self.producer, ANSWER_TIMEOUT);
        let answer = tokio::select! {
            answer = asked => answer,
            () = clock.until(deadline) => {
                return Err(format!(
                    "room_map's get_map_state gave no answer within source_timeout_s {} s: wait \
                     for the room mapper",
                    seconds(timeout_ns)
                ));
            }
        };
        let answer = answer
            .map_err(|e| format!("room_map's get_map_state: {e}: wait for the room mapper"))?;
        let state = MapState::parse(answer.data)
            .map_err(|e| format!("room_map's get_map_state answers {e}"))?;
        Ok((state, arrivals))
    }
}

/// Receive map_grids until the node cancels, each into `room`, logging the first grid of
/// each load and each dropped grid.
pub async fn run_grid_listener(runner: Arc<NodeRunner>, room: Arc<RoomMap>) {
    let mut subscription = match map_grids::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return error!("map_grids subscribe: {e}"),
    };
    let dropped = LatchedLog::warn("map_grids");
    receive(
        "map_grids",
        runner.cancellation_token().cancelled(),
        async move || subscription.next().await,
        async move |(_producer, message): (ProducerRef, map_grids::Message)| {
            match tokio::task::block_in_place(|| room.take_grid(message)) {
                Ok(first) => {
                    dropped.success();
                    if let Some(map) = first {
                        info!("map_grids: the first grid of {map}");
                    }
                }
                Err(reason) => dropped.failure(format!("dropping a grid: {reason}")),
            }
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// What the map_poses listener places each map_poses by: the robot's tree, each lookup
/// waiting up to `lookup_timeout`, drive_states for the odometry it is placed in, and the
/// node's clock for its arrival.
pub struct PoseFollower {
    pub room: Arc<RoomMap>,
    pub tree: Arc<RobotTree>,
    pub link: Arc<Mutex<Link>>,
    pub clock: NodeClock,
    pub lookup_timeout: Duration,
}

impl PoseFollower {
    /// The estimate of `message`, the `arrival`th map_poses, at `arrived_ns`, or why it is
    /// dropped.
    async fn place(
        &self,
        message: &map_poses::Message,
        arrived_ns: i64,
        arrival: u64,
    ) -> Result<Estimate, String> {
        let stamp = Stamp::from_time(message.timestamp)
            .map_err(|e| format!("the pose's timestamp: {e}"))?;
        let odom_pose = robot_in_odom(
            &self.tree,
            stamp,
            self.lookup_timeout,
            "the pose's timestamp",
        )
        .await?;
        let odometry_start = lock_ignoring_poison(&self.link)
            .newest()
            .ok_or("no drive_states arrived to give the base's odometry")?
            .state
            .odometry_start;
        Estimate::placed(message, odom_pose, odometry_start, arrived_ns, arrival)
    }

    /// Receive map_poses until the node cancels, placing each as it arrives into the room,
    /// logging each dropped map_poses.
    pub async fn run(self, runner: Arc<NodeRunner>) {
        let mut subscription = match map_poses::subscribe(&runner).await {
            Ok(subscription) => subscription,
            Err(e) => return error!("map_poses subscribe: {e}"),
        };
        let dropped = LatchedLog::warn("map_poses");
        receive(
            "map_poses",
            runner.cancellation_token().cancelled(),
            async move || subscription.next().await,
            async move |(_producer, message): (ProducerRef, map_poses::Message)| {
                let arrived_ns = clock::now_ns(&self.clock);
                let arrival = self.room.arrived();
                let placed = self.place(&message, arrived_ns, arrival).await;
                match &placed {
                    Ok(_) => dropped.success(),
                    Err(reason) => dropped.failure(format!("dropping a map_poses: {reason}")),
                }
                self.room.take_estimate(placed);
                ControlFlow::Continue(())
            },
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use frame_tree::arithmetic::quaternion_of_yaw;
    use frame_tree::{MovingMessage, Publisher, TransformRecord};
    use hegel::TestCase;
    use hegel::generators as gs;
    use navigation_core::grid::{FREE, LETHAL, UNKNOWN};

    use control_core::drive_link::FollowState;

    use crate::follow::{ODOMETRY_FRAME, ROBOT_FRAME};
    use crate::link::tests::state;

    const NOW_NS: i64 = 10_000_000_000;

    const TIMEOUT_NS: i64 = 500_000_000;

    fn at(seconds: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)
    }

    fn planar(x: f64, y: f64, yaw: f64) -> PlanarPose {
        PlanarPose::new(x, y, yaw).expect("a finite pose")
    }

    /// map in odom's place for the tests: odom stands 2 m along map's x, 1 m along its y,
    /// turned 0.4 rad.
    fn map_from_odom() -> PlanarPose {
        planar(2.0, 1.0, 0.4)
    }

    /// A map_poses of `map_name` placing the robot frame at `pose` in map.
    fn map_pose(map_name: &str, pose: PlanarPose, localized: bool) -> map_poses::Message {
        map_poses::Message {
            frame: MAP_FRAME.to_string(),
            timestamp: at(10),
            map_name: map_name.to_string(),
            position: [pose.x(), pose.y(), 0.0],
            orientation: quaternion_of_yaw(pose.yaw()),
            position_std_m: 0.05,
            orientation_std_rad: 0.02,
            localized,
        }
    }

    /// A grid of map `map_name`, loaded at 3 s, 4 columns by 3 rows of 0.5 m from
    /// (-1, -1), its cell (1, 2) occupied and (3, 0) unknown.
    fn grid_message(map_name: &str) -> map_grids::Message {
        let mut cells = vec![0; 12];
        cells[2 * 4 + 1] = 100;
        cells[3] = 255;
        map_grids::Message {
            frame: MAP_FRAME.to_string(),
            timestamp: at(10),
            map_name: map_name.to_string(),
            map_load_timestamp: at(3),
            cell_size_m: 0.5,
            extent: Some([-1.0, 1.0, -1.0, 0.5]),
            columns: 4,
            rows: 3,
            cells,
        }
    }

    /// The grid of map `map_name` loaded at 3 s before its mapping read a scan.
    fn empty_grid_message(map_name: &str) -> map_grids::Message {
        map_grids::Message {
            extent: None,
            columns: 0,
            rows: 0,
            cells: Vec::new(),
            ..grid_message(map_name)
        }
    }

    /// The grid of `message`, which has an extent.
    fn map_grid(message: map_grids::Message) -> Arc<MapGrid> {
        LoadedGrid::parse(message, 1)
            .expect("a grid")
            .grid
            .expect("a grid of cells")
    }

    fn admitted(map_name: &str, extent: [f64; 4]) -> AdmittedMap {
        AdmittedMap {
            name: map_name.to_string(),
            extent: MapExtent::parse(extent).expect("an extent"),
            cell_size_m: PositiveFinite::parse(0.5).unwrap(),
        }
    }

    fn run(map_name: &str) -> MapRun {
        MapRun {
            map: admitted(map_name, [-1.0, 1.0, -1.0, 0.5]),
            loaded_at: at(3),
        }
    }

    /// An estimate of map `map_name`, map in odom at [`map_from_odom`], placed in the
    /// odometry started at `odometry_s` and arrived at `arrived_ns`.
    fn estimate(map_name: &str, odometry_s: u64, arrived_ns: i64) -> Estimate {
        Estimate {
            map_name: map_name.to_string(),
            map_from_odom: map_from_odom(),
            odom_from_map: PlanarPose::IDENTITY.seen_from(map_from_odom()).unwrap(),
            localized: true,
            odometry_start: at(odometry_s),
            arrived_ns,
            arrival: 1,
        }
    }

    // A map_poses placed by an odometry that stands off the map gives the transform
    // room_map publishes beside it: map to odom, followed by odom to the robot frame,
    // places the robot frame at the map_poses' pose.
    #[hegel::test]
    fn a_map_poses_places_map_in_odom(tc: TestCase) {
        let draw = |low: f64, high: f64| tc.draw(gs::floats().min_value(low).max_value(high));
        let odom_pose = planar(draw(-5.0, 5.0), draw(-5.0, 5.0), draw(-3.0, 3.0));
        let in_map = map_from_odom().compose(odom_pose).unwrap();
        let estimate = Estimate::placed(&map_pose("lab", in_map, true), odom_pose, at(1), 0, 1)
            .expect("an estimate");
        let placed = estimate.map_from_odom.compose(odom_pose).unwrap();
        assert!(
            placed.distance_to(in_map) < 1e-9 && placed.yaw_to(in_map).abs() < 1e-9,
            "{placed:?} against {in_map:?}"
        );
        let back = estimate.odom_from_map.compose(in_map).unwrap();
        assert!(back.distance_to(odom_pose) < 1e-9, "{back:?}");
    }

    // A map_poses in another frame, or off the floor plane, is dropped naming why.
    #[test]
    fn a_map_poses_in_another_frame_or_off_the_floor_is_dropped() {
        let odom = map_pose("lab", PlanarPose::IDENTITY, true);
        let in_odom = map_poses::Message {
            frame: ODOMETRY_FRAME.to_string(),
            ..odom.clone()
        };
        let refused = Estimate::placed(&in_odom, PlanarPose::IDENTITY, at(1), 0, 1)
            .expect_err("a pose in odom");
        assert!(
            refused.contains("\"odom\"") && refused.contains("\"map\""),
            "{refused}"
        );
        let lifted = map_poses::Message {
            position: [0.0, 0.0, 0.3],
            ..odom
        };
        let refused = Estimate::placed(&lifted, PlanarPose::IDENTITY, at(1), 0, 1)
            .expect_err("a pose off the floor");
        assert!(refused.contains("above the floor plane"), "{refused}");
    }

    // A map_poses whose timestamp the tree holds no odometry at is dropped with the
    // lookup's failure kind.
    #[tokio::test]
    async fn a_map_poses_the_tree_cannot_place_is_dropped_with_the_lookups_kind() {
        let tree = RobotTree::new(Duration::from_secs(10), NodeClock::wall()).expect("a window");
        let backbone = Publisher {
            core_node: "core".to_owned(),
            instance_id: "backbone".to_owned(),
        };
        tree.take_moving(
            &backbone,
            &MovingMessage {
                timestamp: at(20),
                transforms: vec![TransformRecord::new(
                    ODOMETRY_FRAME,
                    ROBOT_FRAME,
                    [1.0, 0.0, 0.0],
                    quaternion_of_yaw(0.0),
                )],
                start_timestamp: at(1),
            },
        )
        .expect("odom to the robot frame");
        let mut link = Link::default();
        link.take(state(1, 20, 1.0, FollowState::Following), 0);
        let follower = PoseFollower {
            room: Arc::default(),
            tree: Arc::new(tree),
            link: Arc::new(Mutex::new(link)),
            clock: NodeClock::wall(),
            lookup_timeout: Duration::ZERO,
        };
        let refused = follower
            .place(&map_pose("lab", PlanarPose::IDENTITY, true), 0, 1)
            .await
            .expect_err("no odometry at the map_poses' timestamp");
        assert!(refused.starts_with("extrapolation: "), "{refused}");
    }

    /// The causes a test sets, each true breaking one rule of a step's placement.
    #[derive(Clone, Copy, Debug)]
    struct Causes {
        loaded_again: bool,
        grid_of_another_map: bool,
        other_map: bool,
        not_localized: bool,
        silent: bool,
        none_of_the_odometry: bool,
    }

    /// The words of the first cause that holds, none while none does.
    fn first_cause(c: Causes) -> Option<&'static str> {
        if c.loaded_again || c.grid_of_another_map {
            return Some("the loaded map changed");
        }
        if c.none_of_the_odometry {
            return Some("no map_poses");
        }
        [
            (c.other_map, "the loaded map changed"),
            (c.not_localized, "not localized"),
            (c.silent, "no map_poses"),
        ]
        .into_iter()
        .find_map(|(holds, words)| holds.then_some(words))
    }

    // A step of a navigation in map ends it by the first of: the loaded map changed,
    // the base not localized, and no map_poses of the odometry within the source
    // timeout; with none, map stands in odom where the newest estimate places it.
    #[hegel::test]
    fn a_steps_placement_gives_the_first_cause_in_order(tc: TestCase) {
        let draw = || tc.draw(gs::booleans());
        let causes = Causes {
            loaded_again: draw(),
            grid_of_another_map: draw(),
            other_map: draw(),
            not_localized: draw(),
            silent: draw(),
            none_of_the_odometry: draw(),
        };
        let room = RoomMap::default();
        let arrived_ns = if causes.silent {
            NOW_NS - TIMEOUT_NS - 1
        } else {
            NOW_NS - TIMEOUT_NS
        };
        let newest = Estimate {
            map_name: if causes.other_map { "office" } else { "lab" }.to_string(),
            localized: !causes.not_localized,
            ..estimate("lab", 1, arrived_ns)
        };
        room.take_estimate(Ok(newest));
        let grid = map_grids::Message {
            map_load_timestamp: if causes.loaded_again { at(4) } else { at(3) },
            ..grid_message(if causes.grid_of_another_map {
                "office"
            } else {
                "lab"
            })
        };
        room.take_grid(grid).expect("a grid");
        let odometry_start = if causes.none_of_the_odometry {
            at(2)
        } else {
            at(1)
        };
        let placed = room.place(&run("lab"), odometry_start, NOW_NS, TIMEOUT_NS);
        match (first_cause(causes), placed) {
            (None, Ok(placement)) => {
                assert_eq!(placement.map_from_odom, map_from_odom());
                assert_eq!(placement.grid.map(|grid| grid.seq), Some(1));
            }
            (Some(words), Err(message)) => {
                assert!(message.contains(words), "{causes:?}: {message}")
            }
            (expected, placed) => panic!("{causes:?}: {expected:?} against {placed:?}"),
        }
    }

    // A step reads the newest estimate of its own odometry: one placed in a newer
    // odometry before drive_states report it leaves the step its own.
    #[test]
    fn a_step_reads_the_newest_estimate_of_its_odometry() {
        let room = RoomMap::default();
        room.take_estimate(Ok(estimate("lab", 1, NOW_NS)));
        room.take_estimate(Ok(Estimate {
            map_from_odom: PlanarPose::IDENTITY,
            ..estimate("lab", 2, NOW_NS)
        }));
        let placed = room
            .place(&run("lab"), at(1), NOW_NS, TIMEOUT_NS)
            .expect("the estimate of odometry 1");
        assert_eq!(placed.map_from_odom, map_from_odom());
        let newer = room
            .place(&run("lab"), at(2), NOW_NS, TIMEOUT_NS)
            .expect("the estimate of odometry 2");
        assert_eq!(newer.map_from_odom, PlanarPose::IDENTITY);
    }

    // The silent ending names the seconds since the newest map_poses and the newest
    // drop's reason.
    #[test]
    fn the_silent_ending_names_the_seconds_and_the_newest_drop() {
        let room = RoomMap::default();
        room.take_estimate(Ok(estimate("lab", 1, NOW_NS - 2 * TIMEOUT_NS)));
        room.take_estimate(Err("extrapolation: the robot frame".to_string()));
        let silent = room
            .place(&run("lab"), at(1), NOW_NS, TIMEOUT_NS)
            .expect_err("silent");
        for part in [
            "for 1 s",
            "source_timeout_s 0.5 s",
            "remove_map",
            "dropped: extrapolation: the robot frame",
        ] {
            assert!(silent.contains(part), "{part}: {silent}");
        }
    }

    // The first grid of a load names the map and its size, a later one nothing. A grid
    // in another frame, with a cell off 0, 100 and 255, whose extent does not hold its
    // cells, or with cells and no extent, is dropped, and the grid before it stays.
    #[test]
    fn a_grid_that_does_not_read_keeps_the_grid_before_it() {
        let room = RoomMap::default();
        assert_eq!(
            room.take_grid(grid_message("lab")),
            Ok(Some("map lab, 4 by 3 cells of 0.5 m".to_string()))
        );
        let refusals = [
            map_grids::Message {
                frame: ODOMETRY_FRAME.to_string(),
                ..grid_message("lab")
            },
            map_grids::Message {
                cells: [vec![50], vec![0; 11]].concat(),
                ..grid_message("lab")
            },
            map_grids::Message {
                extent: Some([-1.0, 1.5, -1.0, 0.5]),
                ..grid_message("lab")
            },
            map_grids::Message {
                extent: None,
                ..grid_message("lab")
            },
        ];
        let words = ["\"odom\"", "50", "5 cells", "no extent and gives 4 columns"];
        for (message, words) in refusals.into_iter().zip(words) {
            let refused = room.take_grid(message).expect_err("a refusal");
            assert!(refused.contains(words), "{words}: {refused}");
        }
        room.take_estimate(Ok(estimate("lab", 1, NOW_NS)));
        let placed = room
            .place(&run("lab"), at(1), NOW_NS, TIMEOUT_NS)
            .expect("a placement");
        assert_eq!(placed.grid.map(|grid| grid.seq), Some(1));
        assert_eq!(room.take_grid(grid_message("lab")), Ok(None));
    }

    // The first grid of each load is logged: a new map's grid before its first scan, with
    // no cell, places map without a grid; a grid of cells of that load names nothing new;
    // a load_map of the same map, a new load, is logged again.
    #[test]
    fn the_first_grid_of_each_load_is_logged_and_an_empty_grid_holds_no_cell() {
        let room = RoomMap::default();
        assert_eq!(
            room.take_grid(empty_grid_message("lab")),
            Ok(Some(
                "map lab, with no cell before its mapping reads a scan".to_string()
            ))
        );
        room.take_estimate(Ok(estimate("lab", 1, NOW_NS)));
        let placed = room
            .place(&run("lab"), at(1), NOW_NS, TIMEOUT_NS)
            .expect("a placement on the empty grid's load");
        assert!(placed.grid.is_none(), "{placed:?}");
        assert_eq!(room.take_grid(grid_message("lab")), Ok(None));
        let reloaded = map_grids::Message {
            map_load_timestamp: at(5),
            ..grid_message("lab")
        };
        assert_eq!(
            room.take_grid(reloaded),
            Ok(Some("map lab, 4 by 3 cells of 0.5 m".to_string()))
        );
    }

    // A map-frame goal is admitted once a grid of its map is the newest and a localized
    // map_poses of it arrived after the ask, on that grid's load; until then it names
    // each it waits for, and the newest drop's reason.
    #[test]
    fn a_map_goal_is_admitted_on_the_load_of_its_maps_newest_grid() {
        let room = RoomMap::default();
        let admitted = |after| lock_ignoring_poison(&room.held).admitted_load("lab", at(1), after);
        let none = admitted(0).expect_err("nothing arrived");
        for part in ["no map_poses of map lab", "no map_grids of map lab arrived"] {
            assert!(none.contains(part), "{part}: {none}");
        }
        room.take_grid(map_grids::Message {
            frame: ODOMETRY_FRAME.to_string(),
            ..grid_message("lab")
        })
        .expect_err("a grid in odom");
        room.take_estimate(Ok(estimate("lab", 1, NOW_NS)));
        let no_grid = admitted(0).expect_err("no grid read");
        assert!(
            !no_grid.contains("no map_poses")
                && no_grid.contains("map_grids was dropped: the grid is in the frame"),
            "{no_grid}"
        );
        room.take_grid(grid_message("office")).expect("a grid");
        assert!(admitted(0).is_err(), "the newest grid is of another map");
        room.take_grid(grid_message("lab")).expect("a grid");
        assert_eq!(admitted(0), Ok(at(3)));
        assert!(
            admitted(1)
                .expect_err("arrived by the ask")
                .contains("no map_poses"),
            "a map_poses that arrived by the ask counts for none"
        );
    }

    // A grid's cells read 0, 100 and 255 as free, lethal and unknown, row by row from
    // the extent's least corner.
    #[test]
    fn a_grids_cells_read_row_by_row_from_its_least_corner() {
        let grid = map_grid(grid_message("lab"));
        let saved = grid.saved.grid();
        let cost_at = |point| saved.cost(saved.geometry().cell_of(point).expect("a cell"));
        assert_eq!(cost_at([-0.25, 0.25]), LETHAL);
        assert_eq!(cost_at([0.75, -0.75]), UNKNOWN);
        assert_eq!(cost_at([-0.75, -0.75]), FREE);
    }

    // A plan's lattice holds the map's grid and the extent the goal was admitted
    // against, a point on the extent's far edge in its last cell, the cells off the
    // map's grid unknown; with no grid it holds the extent alone.
    #[test]
    fn a_plans_lattice_pads_the_grid_out_to_the_admitted_extent() {
        let grid = map_grid(grid_message("lab"));
        let wider = MapRun {
            map: admitted("lab", [-2.0, 1.0, -1.0, 0.5]),
            loaded_at: at(3),
        };
        let lattice = wider.plan_lattice(Some(&grid)).expect("a lattice");
        let geometry = lattice.geometry;
        assert_eq!(
            (geometry.size_x(), geometry.size_y(), geometry.origin()),
            (7, 4, [-2.0, -1.0])
        );
        for far_corner in [[1.0, 0.5], [-2.0, 0.5]] {
            assert!(geometry.cell_of(far_corner).is_some(), "{far_corner:?}");
        }
        let layer = lattice.static_layer(&grid).expect("a static layer");
        let costs = layer.grid();
        let cost_at = |point| costs.cost(geometry.cell_of(point).expect("a cell"));
        assert_eq!(cost_at([-0.25, 0.25]), LETHAL, "the map's occupied cell");
        assert_eq!(cost_at([-1.75, 0.25]), UNKNOWN, "a cell off the map's grid");
        assert_eq!(cost_at([-0.75, -0.75]), FREE, "a free cell of the map");
        let alone = wider.plan_lattice(None).expect("a lattice");
        assert_eq!(
            (alone.geometry.size_x(), alone.geometry.size_y()),
            (7, 4),
            "the extent alone, its far edges in the last cells"
        );
    }

    // get_map_state's answer is refused for a cell's side that is not positive and for
    // an extent whose edges make no rectangle, and read otherwise.
    #[test]
    fn get_map_states_answer_reads_its_map() {
        let answer = |cell_size_m, extent| {
            get_map_state::ResponseData::new(
                Some(get_map_state::ResponseDataMap {
                    map_name: "lab".to_string(),
                    cell_size_m,
                    extent,
                    scans: 3,
                    mapping_start_timestamp: at(1),
                }),
                false,
                false,
                true,
                false,
            )
        };
        let state = MapState::parse(answer(0.05, Some([-1.0, 1.0, -2.0, 2.0]))).expect("a state");
        assert_eq!(
            state,
            MapState {
                map: Some(LoadedMap {
                    name: "lab".to_string(),
                    cell_size_m: PositiveFinite::parse(0.05).unwrap(),
                    extent: Some(MapExtent::parse([-1.0, 1.0, -2.0, 2.0]).unwrap()),
                }),
                waiting: true,
                localized: false,
            }
        );
        assert!(MapState::parse(answer(0.0, None)).is_err());
        assert!(MapState::parse(answer(0.05, Some([1.0, -1.0, 0.0, 1.0]))).is_err());
    }
}
