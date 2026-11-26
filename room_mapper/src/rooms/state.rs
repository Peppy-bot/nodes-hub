//! The room: the loaded map and the stamp of its load, the estimate of the robot frame in
//! it, a waiting set_initial_pose, the node's one drand48 stream, and the start of the
//! transform map to odom, as room_map:v1 states each. The members change it through its
//! methods, and the worker hands it each scan.
//!
//! The estimate lives in the odometry frame it began in. When the base's odometry starts
//! over, the [`Carrier`] carries each later scan's odometry into that frame, the base
//! counted as still between the newest scan before the new start and the first scan
//! after it, as room_map states; the transform map to odom is published in the odometry
//! of the scan it places.

use std::time::SystemTime;

use camera_core::laser_scan::Sweep;
use control_core::planar_pose::PlanarPose;
use mapping_core::{
    Drand48, FinishError, GlobalLocalization, GridExtent, InitialPose, InitialPoseFailure,
    Localization, MapName, MapPose, MapStore, MapSummary, Mapping, OccupancyGrid, PlacedScan,
    RoomMap, SearchStep, StoreError,
};
use tokio::sync::oneshot;
use tracing::{info, warn};

use crate::config::Settings;
use crate::lidar::{Arrival, MeasuredScan};
use crate::messages::map_extent;
use crate::odometry::OdometryStart;

/// Why a member is refused or fails, by room_map:v1's failure kinds.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    Rejected(String),
    PreconditionFailed(String),
    ResourceUnavailable(String),
    AttemptFailed(String),
    NoMatch(String),
    Extrapolation(String),
}

/// The loaded map, with when it was loaded (ns, wall clock), map_grids' map_load_timestamp.
enum Loaded {
    None,
    Building {
        name: MapName,
        mapping: Box<Mapping>,
        mapping_start_unix_ns: i64,
        load_unix_ns: i64,
    },
    Saved {
        name: MapName,
        map: Box<RoomMap>,
        load_unix_ns: i64,
    },
}

/// The stamps of the loads of maps (ns, wall clock): each strictly after the one before, so
/// a wall clock that steps back repeats no stamp.
#[derive(Clone, Copy, Debug, Default)]
struct LoadStamps {
    newest_ns: Option<i64>,
}

impl LoadStamps {
    /// The stamp of a load while the wall clock reads `now_ns`: `now_ns`, or 1 ns past the
    /// newest stamp while the clock reads no later than it.
    fn next(&mut self, now_ns: i64) -> i64 {
        let stamp = self
            .newest_ns
            .map_or(now_ns, |newest| now_ns.max(newest.saturating_add(1)));
        self.newest_ns = Some(stamp);
        stamp
    }
}

/// The estimate of the robot frame in the loaded map.
enum Estimate {
    None,
    /// Mapping's, while mapping runs.
    Mapping,
    Matching(Box<Localization>),
    Searching(Box<GlobalLocalization>),
    /// A search that no scan reached within the source timeout ended: its newest update
    /// stays, carried by the odometry alone, not localized.
    SearchEnded,
}

/// The newest update of the estimate, with the base's pose in the estimate's odometry
/// frame at its scan, which carries it to each later scan.
#[derive(Clone, Copy, Debug)]
struct Update {
    answer: MapPose,
    base: PlanarPose,
}

/// What carries a scan's odometry into the frame the estimate began in.
#[derive(Clone, Copy, Debug)]
struct Carrier {
    /// The odometry start of the newest scan.
    start: OdometryStart,
    /// The estimate's frame from that odometry's frame.
    from_odometry: PlanarPose,
    /// The base in that odometry at the newest scan.
    base: PlanarPose,
}

/// A set_initial_pose that waits for its scan: the first scan that arrives after its call
/// in the odometry it was called in.
struct WaitingPose {
    initial: InitialPose,
    /// The target's time (ns); none for the call's time.
    target_ns: Option<i64>,
    /// The scans arrived at the call.
    called_after: Arrival,
    called_start: OdometryStart,
    /// The base in that odometry at the call.
    base_at_call: PlanarPose,
    reply: oneshot::Sender<Result<MapPose, Refusal>>,
}

/// The base's odometry where a member reads it: its start and the base's newest pose in
/// it, with the scans arrived at the call.
#[derive(Clone, Copy, Debug)]
pub struct OdometryNow {
    pub start: OdometryStart,
    pub base: PlanarPose,
    pub arrived: Arrival,
}

/// An answer of the estimate to a scan: the loaded map's name, the pose map_poses carries,
/// the transform map to odom of that scan's odometry, and the start of that transform (ns,
/// wall clock), each read under one hold of the room.
#[derive(Clone, Debug)]
pub struct Answer {
    pub map_name: String,
    pub map_pose: MapPose,
    pub map_from_odometry: PlanarPose,
    pub start_unix_ns: i64,
}

/// A map as room_map's map object gives it, with the extent its kind of map has.
#[derive(Clone, Debug, PartialEq)]
pub struct MapObject<Extent> {
    pub name: String,
    pub resolution_m: f64,
    pub extent: Extent,
    pub scans: u32,
    pub mapping_start_unix_ns: i64,
}

/// A saved map, which has a grid.
pub type SavedMap = MapObject<GridExtent>;

/// The loaded map: a saved map, or the map being built, which has no extent while its
/// mapping has read no scan.
pub type LoadedMap = MapObject<Option<GridExtent>>;

impl From<SavedMap> for LoadedMap {
    fn from(saved: SavedMap) -> Self {
        MapObject {
            name: saved.name,
            resolution_m: saved.resolution_m,
            extent: Some(saved.extent),
            scans: saved.scans,
            mapping_start_unix_ns: saved.mapping_start_unix_ns,
        }
    }
}

/// What a loaded map's grid is drawn from: a saved map, whose grid is fixed, or the scans in
/// the pose graph of the map being built, whose grid changes as their number does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridSource {
    Saved,
    Building { graph_scans: usize },
}

/// The loaded map's grid as map_grids carries it, with what it is drawn from.
pub struct MapGrid {
    pub source: GridSource,
    pub map_name: String,
    /// When the map was loaded (ns, wall clock).
    pub load_unix_ns: i64,
    pub cell_size_m: f64,
    /// None for a new map whose mapping has read no scan, whose grid has no cell.
    pub grid: Option<OccupancyGrid>,
}

/// get_map_state's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct MapState {
    pub map: Option<LoadedMap>,
    pub mapping: bool,
    pub searching: bool,
    pub waiting: bool,
    pub localized: bool,
}

/// The transform map to odom's start over: pending at a load_map, a start_mapping and a
/// remove_map, and due when a scan's odometry start differs from the one before.
#[derive(Clone, Copy, Debug, Default)]
struct TransformStart {
    pending: bool,
    current: Option<(OdometryStart, i64)>,
}

/// The room.
pub struct RoomState {
    settings: Settings,
    store: MapStore,
    loaded: Loaded,
    estimate: Estimate,
    /// The scans arrived at the call that started the estimate, which takes the scans that
    /// arrive after it alone.
    estimate_after: Arrival,
    newest: Option<Update>,
    waiting: Option<WaitingPose>,
    stream: Drand48,
    carrier: Option<Carrier>,
    transform_start: TransformStart,
    load_stamps: LoadStamps,
}

impl RoomState {
    pub fn new(settings: Settings, store: MapStore) -> Self {
        let stream = Drand48::seeded(settings.random_seed);
        Self {
            settings,
            store,
            loaded: Loaded::None,
            estimate: Estimate::None,
            estimate_after: Arrival::default(),
            newest: None,
            waiting: None,
            stream,
            carrier: None,
            transform_start: TransformStart::default(),
            load_stamps: LoadStamps::default(),
        }
    }

    pub fn state(&self) -> MapState {
        let mapping = matches!(self.loaded, Loaded::Building { .. });
        MapState {
            map: self.map_object(),
            mapping,
            searching: matches!(self.estimate, Estimate::Searching(_)),
            waiting: self.waiting.is_some(),
            // While mapping runs, the base is localized, as room_map states.
            localized: mapping || self.newest.is_some_and(|newest| newest.answer.localized),
        }
    }

    /// The loaded map's name; none while no map is loaded.
    pub fn loaded_name(&self) -> Option<&MapName> {
        match &self.loaded {
            Loaded::None => None,
            Loaded::Building { name, .. } | Loaded::Saved { name, .. } => Some(name),
        }
    }

    fn map_object(&self) -> Option<LoadedMap> {
        match &self.loaded {
            Loaded::None => None,
            Loaded::Building {
                name,
                mapping,
                mapping_start_unix_ns,
                ..
            } => Some(MapObject {
                name: name.to_string(),
                resolution_m: self.settings.mapping.resolution,
                extent: mapping
                    .extent()
                    .inspect_err(|e| warn!("mapping: its scans make no grid: {e}"))
                    .ok()
                    .flatten(),
                scans: mapping.scans(),
                mapping_start_unix_ns: *mapping_start_unix_ns,
            }),
            Loaded::Saved { name, map, .. } => Some(object_of(name, &map.summary()).into()),
        }
    }

    /// What the loaded map's grid is drawn from; none while no map is loaded.
    pub fn grid_source(&self) -> Option<GridSource> {
        match &self.loaded {
            Loaded::None => None,
            Loaded::Building { mapping, .. } => Some(GridSource::Building {
                graph_scans: mapping.graph_scans(),
            }),
            Loaded::Saved { .. } => Some(GridSource::Saved),
        }
    }

    /// The loaded map's grid: a saved map's, or the grid of the map being built as it
    /// stands, which has no cell before its mapping reads its first scan. None while no map
    /// is loaded, and none while the scans of the map being built make no grid, which is
    /// logged.
    pub fn map_grid(&self) -> Option<MapGrid> {
        let (name, load_unix_ns, grid) = match &self.loaded {
            Loaded::None => return None,
            Loaded::Building {
                name,
                mapping,
                load_unix_ns,
                ..
            } => (
                name,
                *load_unix_ns,
                mapping
                    .grid()
                    .inspect_err(|e| warn!("mapping: its scans make no grid: {e}"))
                    .ok()?,
            ),
            Loaded::Saved {
                name,
                map,
                load_unix_ns,
            } => (name, *load_unix_ns, Some(map.grid().clone())),
        };
        Some(MapGrid {
            source: self.grid_source()?,
            map_name: name.to_string(),
            load_unix_ns,
            // The map's cell size as get_map_state gives it: mapping's resolution before the
            // first scan.
            cell_size_m: grid
                .as_ref()
                .map_or(self.settings.mapping.resolution, |grid| {
                    grid.extent().resolution_m
                }),
            grid,
        })
    }

    /// Refuses with precondition_failed while mapping runs.
    pub fn refuse_mapping(&self) -> Result<(), Refusal> {
        match self.loaded {
            Loaded::Building { .. } => Err(Refusal::PreconditionFailed(
                "mapping runs: finish it with finish_mapping first".to_string(),
            )),
            _ => Ok(()),
        }
    }

    /// Refuses with precondition_failed while a set_initial_pose waits.
    pub fn refuse_waiting(&self) -> Result<(), Refusal> {
        match self.waiting {
            Some(_) => Err(Refusal::PreconditionFailed(
                "a set_initial_pose waits for its scan: wait for its answer".to_string(),
            )),
            None => Ok(()),
        }
    }

    /// Refuses with precondition_failed while no map is loaded.
    pub fn refuse_no_map(&self) -> Result<(), Refusal> {
        match self.loaded {
            Loaded::None => Err(Refusal::PreconditionFailed(
                "no map is loaded: load one with load_map, or map one with start_mapping"
                    .to_string(),
            )),
            _ => Ok(()),
        }
    }

    /// A map name, refused with rejected unless it is one.
    pub fn name_of(requested: &str) -> Result<MapName, Refusal> {
        MapName::parse(requested).map_err(|e| Refusal::Rejected(e.to_string()))
    }

    /// Refuses with rejected a name that a saved map has.
    pub fn refuse_saved_name(&self, name: &MapName) -> Result<(), Refusal> {
        if self.store.contains(name).map_err(store_failure)? {
            return Err(Refusal::Rejected(format!(
                "a saved map is named {name}: name the new map otherwise, or remove {name} \
                 with remove_map"
            )));
        }
        Ok(())
    }

    /// Refuses with rejected a name that no saved map has.
    fn refuse_unsaved_name(&self, name: &MapName) -> Result<(), Refusal> {
        if !self.store.contains(name).map_err(store_failure)? {
            return Err(Refusal::Rejected(format!(
                "no map named {name} is saved: list_maps gives the saved maps' names"
            )));
        }
        Ok(())
    }

    /// start_mapping, its checks done: a new map named `name` whose map frame is the
    /// odometry frame now, mapping the scans that arrive after the call.
    pub fn start_mapping(&mut self, name: MapName, now: OdometryNow) {
        let mapping =
            Mapping::start(self.settings.mapping).expect("the setup checked mapping's parameters");
        self.estimate = Estimate::Mapping;
        self.estimate_after = now.arrived;
        self.newest = None;
        self.carrier = Some(Carrier {
            start: now.start,
            from_odometry: PlanarPose::IDENTITY,
            base: now.base,
        });
        let now_ns = unix_now_ns();
        self.loaded = Loaded::Building {
            name: name.clone(),
            mapping: Box::new(mapping),
            mapping_start_unix_ns: now_ns,
            load_unix_ns: self.load_stamps.next(now_ns),
        };
        self.transform_start.pending = true;
        info!("start_mapping: map {name}");
    }

    /// finish_mapping: the saved map's object, loaded with the stamp of its mapping's start.
    /// A failure leaves mapping going on.
    pub fn finish_mapping(&mut self) -> Result<SavedMap, Refusal> {
        let Loaded::Building {
            name,
            mapping,
            mapping_start_unix_ns,
            load_unix_ns,
        } = &self.loaded
        else {
            return Err(Refusal::PreconditionFailed(
                "mapping does not run: start one with start_mapping".to_string(),
            ));
        };
        let finished = mapping
            .as_ref()
            .clone()
            .finish(*mapping_start_unix_ns, self.settings.localization)
            .map_err(|e| {
                Refusal::AttemptFailed(match e {
                    FinishError::Grid(error) => format!("{error}; mapping goes on"),
                    FinishError::Localization { error, .. } => {
                        format!("the map's localization does not start ({error}); mapping goes on")
                    }
                })
            })?;
        let Some((map, localization)) = finished else {
            return Err(Refusal::AttemptFailed(
                "the map holds no scan: drive the base so its lidar sees the room, then \
                 finish; mapping goes on"
                    .to_string(),
            ));
        };
        self.store.save(name, &map).map_err(|e| {
            Refusal::AttemptFailed(format!("the map was not saved ({e}); mapping goes on"))
        })?;
        let (name, load_unix_ns) = (name.clone(), *load_unix_ns);
        let object = object_of(&name, &map.summary());
        info!("finish_mapping: map {name} saved");
        self.loaded = Loaded::Saved {
            name,
            map: Box::new(map),
            load_unix_ns,
        };
        self.estimate = Estimate::Matching(Box::new(localization));
        // The matching holds the estimate from here, which its limits judge.
        let limits = self.settings.localization.matching_limits;
        if let Some(newest) = &mut self.newest {
            newest.answer.localized = newest.answer.deviations.within(&limits);
        }
        Ok(object)
    }

    /// list_maps: the saved maps whose description reads, sorted by name.
    pub fn list_maps(&self) -> Result<Vec<SavedMap>, Refusal> {
        Ok(self
            .store
            .list()
            .map_err(store_failure)?
            .iter()
            .map(|saved| object_of(&saved.name, &saved.summary))
            .collect())
    }

    /// remove_map, its name parsed. Removing the loaded map unloads it: no map is loaded,
    /// no estimate stays, and a search ends.
    pub fn remove_map(&mut self, name: &MapName) -> Result<(), Refusal> {
        self.refuse_unsaved_name(name)?;
        let loaded =
            self.loaded_name() == Some(name) && matches!(self.loaded, Loaded::Saved { .. });
        if loaded {
            self.refuse_waiting()?;
        }
        self.store.remove(name).map_err(store_failure)?;
        if loaded {
            self.loaded = Loaded::None;
            self.forget_estimate();
            self.transform_start.pending = true;
        }
        info!("remove_map: map {name} removed");
        Ok(())
    }

    /// load_map, its name parsed and its preconditions checked: the loaded map's object.
    /// A map that does not read leaves the loaded map and the estimate as they were.
    pub fn load_map(&mut self, name: &MapName) -> Result<SavedMap, Refusal> {
        self.refuse_unsaved_name(name)?;
        self.refuse_mapping()?;
        self.refuse_waiting()?;
        let map = self.store.load(name).map_err(store_failure)?;
        let object = object_of(name, &map.summary());
        info!("load_map: map {name} loaded");
        self.loaded = Loaded::Saved {
            name: name.clone(),
            map: Box::new(map),
            load_unix_ns: self.load_stamps.next(unix_now_ns()),
        };
        self.forget_estimate();
        self.transform_start.pending = true;
        Ok(object)
    }

    /// Refuses with rejected a `point` (m) of the map frame outside the loaded map, its
    /// extent's rectangle: any point on a new map whose mapping has read no scan, whose grid
    /// covers no point. Refuses nothing while no map is loaded.
    pub fn refuse_outside(&self, point: [f64; 2]) -> Result<(), Refusal> {
        let Some(object) = self.map_object() else {
            return Ok(());
        };
        let [x, y] = point;
        match object.extent {
            None => Err(Refusal::Rejected(format!(
                "({x}, {y}) lies outside the loaded map {}: its mapping has read no scan, so \
                 its grid covers no point; drive the base until get_map_state gives an extent",
                object.name
            ))),
            Some(extent) if !map_extent(&extent).contains(point) => {
                Err(Refusal::Rejected(format!(
                    "({x}, {y}) lies outside the loaded map {}: get_map_state gives its extent",
                    object.name
                )))
            }
            Some(_) => Ok(()),
        }
    }

    /// set_initial_pose, its checks done: the receiver of its answer, once its scan
    /// arrives. The member answers one call at a time.
    pub fn set_initial_pose(
        &mut self,
        initial: InitialPose,
        target_ns: Option<i64>,
        now: OdometryNow,
    ) -> oneshot::Receiver<Result<MapPose, Refusal>> {
        assert!(
            self.waiting.is_none(),
            "set_initial_pose's member answers one call at a time, each ending its wait"
        );
        let (reply, answer) = oneshot::channel();
        self.waiting = Some(WaitingPose {
            initial,
            target_ns,
            called_after: now.arrived,
            called_start: now.start,
            base_at_call: now.base,
            reply,
        });
        answer
    }

    /// A set_initial_pose that no scan reached by its deadline waits no more, and the
    /// estimate stays as it was: whether one still waited.
    pub fn give_up_waiting(&mut self) -> bool {
        self.waiting.take().is_some()
    }

    /// reinitialize_global_localization, its preconditions checked: a search of the
    /// loaded map on the node's stream, which takes the scans that arrive after the call.
    pub fn reinitialize(&mut self, now: OdometryNow) -> Result<(), Refusal> {
        let Loaded::Saved { map, .. } = &self.loaded else {
            unreachable!("its preconditions hold a saved map loaded");
        };
        let search = GlobalLocalization::start(map, self.settings.search, &mut self.stream)
            .map_err(|e| Refusal::AttemptFailed(e.to_string()))?;
        self.estimate = Estimate::Searching(Box::new(search));
        self.estimate_after = now.arrived;
        self.newest = None;
        self.carrier = Some(Carrier {
            start: now.start,
            from_odometry: PlanarPose::IDENTITY,
            base: now.base,
        });
        info!("reinitialize_global_localization: a search of the loaded map");
        Ok(())
    }

    /// A search that no scan reached within the source timeout ends; its newest update
    /// stays, not localized.
    pub fn search_went_silent(&mut self) {
        if matches!(self.estimate, Estimate::Searching(_)) {
            warn!("the search ends: no lidar scan arrived within the source timeout");
            self.estimate = Estimate::SearchEnded;
            if let Some(newest) = &mut self.newest {
                newest.answer.localized = false;
            }
        }
    }

    fn forget_estimate(&mut self) {
        self.estimate = Estimate::None;
        self.newest = None;
        self.carrier = None;
    }

    /// Takes `scan` into the waiting set_initial_pose and the estimate: map_poses' pose of
    /// the base at the scan with the transform map to odom, while an estimate exists. An
    /// update answers with its own pose; any other scan with the newest update carried by
    /// the odometry to it.
    pub fn take_scan(
        &mut self,
        scan: &MeasuredScan,
        odometry_at: impl Fn(i64) -> Option<PlanarPose>,
    ) -> Option<Answer> {
        let matched = self.answer_waiting(scan, &odometry_at);
        // A scan that arrived by the call that started the estimate takes no part in it,
        // the scan that matched a set_initial_pose aside, which the match placed.
        if !matched && scan.arrival <= self.estimate_after {
            return None;
        }
        let map_name = self.loaded_name()?.to_string();
        let carrier = self.carry(scan)?;
        let placed = placed_in(scan, carrier.from_odometry)?;
        let updated = if scan.arrival > self.estimate_after {
            self.estimate_takes(&placed)
        } else {
            None
        };
        if let Some(answer) = updated {
            self.newest = Some(Update {
                answer,
                base: placed.odometry(),
            });
        }
        let map_pose = updated.or_else(|| self.carried_to(&placed))?;
        let odometry_undone = PlanarPose::IDENTITY.seen_from(placed.odometry()).ok()?;
        let map_from_estimate_frame = map_pose.pose.compose(odometry_undone).ok()?;
        Some(Answer {
            map_name,
            map_pose,
            map_from_odometry: map_from_estimate_frame
                .compose(carrier.from_odometry)
                .ok()?,
            start_unix_ns: self.transform_started(scan.start),
        })
    }

    /// The carrier at `scan`: a new odometry start carries the frame on from the newest
    /// scan before it, the base counted as still between the two scans. None while no
    /// estimate exists, and for a carry past the finite numbers, which leaves the carrier.
    fn carry(&mut self, scan: &MeasuredScan) -> Option<Carrier> {
        let carrier = self.carrier.as_mut()?;
        if carrier.start != scan.start {
            let base_undone = PlanarPose::IDENTITY.seen_from(scan.base).ok()?;
            let from_odometry = carrier
                .from_odometry
                .compose(carrier.base)
                .ok()?
                .compose(base_undone)
                .ok()?;
            info!("the base's odometry started over: the estimate carries on from its last scan");
            carrier.from_odometry = from_odometry;
            carrier.start = scan.start;
        }
        carrier.base = scan.base;
        Some(*carrier)
    }

    /// The transform map to odom's start (ns, wall clock) at a scan of `start`.
    fn transform_started(&mut self, start: OdometryStart) -> i64 {
        let started = &mut self.transform_start;
        match started.current {
            Some((odometry_start, at)) if !started.pending && odometry_start == start => at,
            _ => {
                let at = unix_now_ns();
                started.current = Some((start, at));
                started.pending = false;
                at
            }
        }
    }

    /// The newest update carried by the odometry to `placed`.
    fn carried_to(&self, placed: &PlacedScan) -> Option<MapPose> {
        let newest = self.newest?;
        let base_undone = PlanarPose::IDENTITY.seen_from(newest.base).ok()?;
        let map_from_estimate_frame = newest.answer.pose.compose(base_undone).ok()?;
        Some(MapPose {
            stamp_ns: placed.stamp_ns(),
            pose: map_from_estimate_frame.compose(placed.odometry()).ok()?,
            ..newest.answer
        })
    }

    /// A waiting set_initial_pose takes the first scan that arrives after its call in the
    /// odometry it was called in, its target carried by that odometry from the target's
    /// time, or from the call, to the scan: whether the scan matched it.
    fn answer_waiting(
        &mut self,
        scan: &MeasuredScan,
        odometry_at: &impl Fn(i64) -> Option<PlanarPose>,
    ) -> bool {
        let Some(waiting) = self.waiting.take_if(|waiting| {
            scan.arrival > waiting.called_after && scan.start == waiting.called_start
        }) else {
            return false;
        };
        let answer = self.match_initial_pose(&waiting, scan, odometry_at);
        if let Ok(pose) = &answer {
            info!(
                "set_initial_pose: matched at ({}, {})",
                pose.pose.x(),
                pose.pose.y()
            );
        }
        let matched = answer.is_ok();
        let _ = waiting.reply.send(answer);
        matched
    }

    fn match_initial_pose(
        &mut self,
        waiting: &WaitingPose,
        scan: &MeasuredScan,
        odometry_at: &impl Fn(i64) -> Option<PlanarPose>,
    ) -> Result<MapPose, Refusal> {
        let Loaded::Saved { map, .. } = &self.loaded else {
            return Err(Refusal::PreconditionFailed(
                "the loaded map changed while set_initial_pose waited".to_string(),
            ));
        };
        let base_at_target = match waiting.target_ns {
            None => waiting.base_at_call,
            Some(at_ns) => odometry_at(at_ns).ok_or_else(|| {
                Refusal::Extrapolation(
                    "no two base_states of the scan's odometry enclose the target's timestamp: \
                     give a timestamp the base's odometry holds, or none for now"
                        .to_string(),
                )
            })?,
        };
        let placed = placed_in(scan, PlanarPose::IDENTITY).ok_or_else(|| {
            Refusal::AttemptFailed("the scan places no ray: a pose is not finite".to_string())
        })?;
        match Localization::from_initial_pose(
            map,
            self.settings.localization,
            &waiting.initial,
            base_at_target,
            &placed,
        ) {
            Ok((localization, pose)) => {
                self.estimate = Estimate::Matching(Box::new(localization));
                self.estimate_after = scan.arrival;
                self.newest = Some(Update {
                    answer: pose,
                    base: placed.odometry(),
                });
                self.carrier = Some(Carrier {
                    start: scan.start,
                    from_odometry: PlanarPose::IDENTITY,
                    base: scan.base,
                });
                Ok(pose)
            }
            Err(InitialPoseFailure::NoMatch { best }) => {
                Err(Refusal::NoMatch(self.no_match_cause(best)))
            }
            Err(InitialPoseFailure::Scan(error)) => Err(Refusal::AttemptFailed(format!(
                "the scan does not read as the map's scans do: {error}"
            ))),
            Err(other) => Err(Refusal::AttemptFailed(other.to_string())),
        }
    }

    /// Why set_initial_pose found no match, from `best`, the match within the search
    /// whose deviations lie past the matching's limits or whose pose lies past the search,
    /// none when the scan matched nowhere within the search.
    fn no_match_cause(&self, best: Option<MapPose>) -> String {
        let limits = self.settings.localization.matching_limits;
        let Some(best) = best else {
            return "no scan matched the map within the search: give a pose nearer the base, \
                    or a wider search"
                .to_string();
        };
        let (pose, deviations) = (best.pose, best.deviations);
        let at = format!(
            "the scan matched the map best at x {:.3} m, y {:.3} m, yaw {:.3} rad",
            pose.x(),
            pose.y(),
            pose.yaw()
        );
        if deviations.within(&limits) {
            return format!(
                "{at}, past the search: give a pose nearer the base, or a wider search"
            );
        }
        format!(
            "{at}, its deviations {:.3} m and {:.3} rad past the matching's limits, {} m and \
             {} rad: give a pose nearer the base, or launch with larger matching_limits",
            deviations.position_std_m,
            deviations.yaw_std_rad,
            limits.position_std_m.get(),
            limits.yaw_std_rad.get()
        )
    }

    /// The estimate's update at `placed`: the mapping's, the matching's or the search's.
    fn estimate_takes(&mut self, placed: &PlacedScan) -> Option<MapPose> {
        match std::mem::replace(&mut self.estimate, Estimate::None) {
            Estimate::Searching(search) => match search.add(placed, &mut self.stream) {
                SearchStep::Searching(search, answered) => {
                    self.estimate = Estimate::Searching(search.into());
                    answered
                }
                SearchStep::Found(localization, found) => {
                    info!(
                        "the search localized the base at ({}, {})",
                        found.pose.x(),
                        found.pose.y()
                    );
                    self.estimate = Estimate::Matching(Box::new(localization));
                    Some(found)
                }
            },
            estimate => {
                self.estimate = estimate;
                let taken = match (&mut self.estimate, &mut self.loaded) {
                    (Estimate::Mapping, Loaded::Building { mapping, .. }) => {
                        mapping.add(placed).map_err(|e| format!("mapping: {e}"))
                    }
                    (Estimate::Matching(localization), _) => localization
                        .add(placed)
                        .map_err(|e| format!("localization: {e}")),
                    _ => return None,
                };
                taken
                    .inspect_err(|reason| warn!("a scan is refused: {reason}"))
                    .ok()
                    .flatten()
            }
        }
    }
}

/// `scan` as mapping_core takes it, its odometry carried into the estimate's frame by
/// `from_odometry`; none for a scan whose rays a pose does not place.
fn placed_in(scan: &MeasuredScan, from_odometry: PlanarPose) -> Option<PlacedScan> {
    let carried = |pose| from_odometry.compose(pose).ok();
    PlacedScan::new(
        scan.stamp_ns,
        &scan.laser,
        &Sweep {
            first_ray: carried(scan.sensor.first_ray)?,
            last_ray: carried(scan.sensor.last_ray)?,
        },
        carried(scan.base)?,
    )
    .inspect_err(|e| warn!("a scan places no ray: {e}"))
    .ok()
}

fn object_of(name: &MapName, summary: &MapSummary) -> SavedMap {
    MapObject {
        name: name.to_string(),
        resolution_m: summary.extent.resolution_m,
        extent: summary.extent,
        scans: summary.scans,
        mapping_start_unix_ns: summary.mapping_start_unix_ns,
    }
}

fn store_failure(error: StoreError) -> Refusal {
    match error {
        StoreError::NotFound { .. } | StoreError::Exists { .. } => {
            Refusal::Rejected(error.to_string())
        }
        _ => Refusal::AttemptFailed(error.to_string()),
    }
}

/// Now on the wall clock (ns since the Unix epoch).
fn unix_now_ns() -> i64 {
    crate::messages::ns_of(SystemTime::now())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_load_stamp_lies_strictly_after_the_one_before() {
        let mut stamps = LoadStamps::default();
        assert_eq!(stamps.next(1_000), 1_000, "the first load");
        assert_eq!(stamps.next(5_000), 5_000, "a later clock");
        assert_eq!(stamps.next(5_000), 5_001, "the same reading");
        assert_eq!(stamps.next(2_000), 5_002, "a clock that stepped back");
        assert_eq!(stamps.next(9_000), 9_000, "the clock past the newest stamp");
    }
}
