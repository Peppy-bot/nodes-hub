//! The members of room_map:v1, as the node answers each: the read of describe_base at the
//! newest odometry start waited for first by the members that test the odometry, then
//! the checks in room_map's order, `rejected`, `precondition_failed`, then
//! `resource_unavailable`, then the change of the room.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use control_core::geometry::Pose;
use control_core::sync::{lock_ignoring_poison, wait_for};
use mapping_core::{InitialPose, MapPose};
use tokio::sync::mpsc;

use crate::geometry::on_floor;
use crate::inputs::{BaseFacts, FactsRead, Inputs, Mount};
use crate::messages::duration_ns;
use crate::rooms::state::{MapGrid, MapState, OdometryNow, Refusal, RoomState, SavedMap};

/// How long (ns) after the newest base_states arrived the base still counts as heard,
/// as room_map's members name it: 0.5 s.
const BASE_HEARD_NS: u64 = 500_000_000;

/// The frame of every pose of room_map.
pub const MAP_FRAME: &str = "map";

/// A set_initial_pose's target as its request gives it.
#[derive(Clone, Debug)]
pub struct TargetRequest {
    pub frame: String,
    /// The target's time (ns, the clock of transforms); none for the call's time.
    pub at_ns: Option<i64>,
    pub position: [f64; 3],
    pub orientation: [f64; 4],
}

/// The members, over the room and the node's inputs.
#[derive(Clone)]
pub struct Members {
    room: Arc<Mutex<RoomState>>,
    inputs: Arc<Inputs>,
    source_timeout: Duration,
    /// Takes the loaded map's grid at each load_map, start_mapping and finish_mapping,
    /// which map_grids publishes at once.
    grids: mpsc::UnboundedSender<MapGrid>,
}

impl Members {
    pub fn new(
        room: Arc<Mutex<RoomState>>,
        inputs: Arc<Inputs>,
        source_timeout: Duration,
        grids: mpsc::UnboundedSender<MapGrid>,
    ) -> Self {
        Self {
            room,
            inputs,
            source_timeout,
            grids,
        }
    }

    /// Hands map_grids the loaded map's grid, drawn while the member that changed the map
    /// still holds `room`, so grids of the loaded map go out in the order of its changes.
    fn hand_over_grid(&self, room: &RoomState) {
        let Some(grid) = room.map_grid() else {
            return;
        };
        // A send fails only once the grid's publisher has ended, which stops the node.
        let _ = self.grids.send(grid);
    }

    /// The room, waited for off the runtime's threads while the worker takes a scan.
    fn room(&self) -> MutexGuard<'_, RoomState> {
        tokio::task::block_in_place(|| lock_ignoring_poison(&self.room))
    }

    pub fn get_map_state(&self) -> MapState {
        self.room().state()
    }

    pub fn start_mapping(&self, map_name: &str) -> Result<(), Refusal> {
        let facts = self.base_facts()?;
        let name = RoomState::name_of(map_name)?;
        let mut room = self.room();
        room.refuse_saved_name(&name)?;
        room.refuse_mapping()?;
        room.refuse_waiting()?;
        refuse_commanded(&facts)?;
        let now = self.odometry_now()?;
        self.lidar_heard()?;
        room.start_mapping(name, now);
        self.hand_over_grid(&room);
        Ok(())
    }

    pub fn finish_mapping(&self) -> Result<SavedMap, Refusal> {
        let mut room = self.room();
        let finished = room.finish_mapping()?;
        self.hand_over_grid(&room);
        Ok(finished)
    }

    pub fn list_maps(&self) -> Result<Vec<SavedMap>, Refusal> {
        self.room().list_maps()
    }

    pub fn remove_map(&self, map_name: &str) -> Result<(), Refusal> {
        let name = RoomState::name_of(map_name)?;
        self.room().remove_map(&name)
    }

    pub fn load_map(&self, map_name: &str) -> Result<SavedMap, Refusal> {
        let name = RoomState::name_of(map_name)?;
        let mut room = self.room();
        let loaded = room.load_map(&name)?;
        self.hand_over_grid(&room);
        Ok(loaded)
    }

    /// set_initial_pose: the matched pose once the first scan after the call answers it.
    pub fn set_initial_pose(
        &self,
        target: &TargetRequest,
        search_radius_m: f64,
        search_angle_rad: f64,
    ) -> Result<MapPose, Refusal> {
        let facts = self.base_facts()?;
        let pose = parse_target(target)?;
        let initial = InitialPose::parse(pose, search_radius_m, search_angle_rad)
            .map_err(|e| Refusal::Rejected(e.to_string()))?;
        let answer = {
            let mut room = self.room();
            room.refuse_outside([pose.x(), pose.y()])?;
            room.refuse_no_map()?;
            room.refuse_mapping()?;
            refuse_commanded(&facts)?;
            let now = self.odometry_now()?;
            room.set_initial_pose(initial, target.at_ns, now)
        };
        let deadline = self.inputs.clock.now_ns() + self.source_timeout_ns();
        let mut answer = answer;
        let answered = wait_for(async {
            tokio::select! {
                answer = &mut answer => Some(answer),
                () = self.inputs.clock.until(deadline) => None,
            }
        });
        // A scan that took the call before the deadline answers it, however long its match
        // runs: the source timeout bounds the wait for a scan alone.
        let answered = match answered {
            None if !self.room().give_up_waiting() => Some(wait_for(answer)),
            answered => answered,
        };
        match answered {
            Some(Ok(answer)) => answer,
            Some(Err(_)) | None => Err(Refusal::ResourceUnavailable(format!(
                "no lidar scan in the base's odometry arrived within the source timeout ({} s): \
                 {}",
                self.source_timeout.as_secs_f64(),
                self.lidar_state()
            ))),
        }
    }

    pub fn reinitialize_global_localization(&self) -> Result<(), Refusal> {
        let facts = self.base_facts()?;
        let mut room = self.room();
        room.refuse_no_map()?;
        room.refuse_mapping()?;
        room.refuse_waiting()?;
        refuse_commanded(&facts)?;
        let now = self
            .newest_odometry()
            .expect("describe_base was read at a start of the base's odometry");
        room.reinitialize(now)
    }

    fn source_timeout_ns(&self) -> u64 {
        duration_ns(self.source_timeout)
    }

    /// The read of describe_base at the newest odometry start, waited for within the source
    /// timeout; resource_unavailable when it does not answer.
    fn base_facts(&self) -> Result<BaseFacts, Refusal> {
        let mut read = self.inputs.facts.subscribe();
        let deadline = self.inputs.clock.now_ns() + self.source_timeout_ns();
        let answered = wait_for(async {
            tokio::select! {
                answered = read.wait_for(|facts| {
                    matches!(facts, FactsRead::Read(_) | FactsRead::Unanswered(_))
                }) => answered.ok().map(|facts| facts.clone()),
                () = self.inputs.clock.until(deadline) => None,
            }
        });
        let timeout_s = self.source_timeout.as_secs_f64();
        match answered.unwrap_or_else(|| read.borrow().clone()) {
            FactsRead::Read(facts) => Ok(facts),
            FactsRead::Unanswered(why) => Err(Refusal::ResourceUnavailable(why)),
            FactsRead::NoOdometry => Err(Refusal::ResourceUnavailable(format!(
                "no base_states arrived within the source timeout ({timeout_s} s): run the \
                 base's node and bind it to base"
            ))),
            FactsRead::Reading(_) => Err(Refusal::ResourceUnavailable(format!(
                "describe_base did not answer within the source timeout ({timeout_s} s) after \
                 the base's odometry started: check the base's node"
            ))),
        }
    }

    /// The base's newest odometry, refused with resource_unavailable while the newest
    /// base_states arrived more than 0.5 s ago.
    fn odometry_now(&self) -> Result<OdometryNow, Refusal> {
        let heard = self
            .inputs
            .since_base()
            .is_some_and(|since| since <= BASE_HEARD_NS);
        match self.newest_odometry() {
            Some(now) if heard => Ok(now),
            _ => Err(Refusal::ResourceUnavailable(
                "more than 0.5 s have passed since the newest base_states arrived: check the \
                 base's node"
                    .to_string(),
            )),
        }
    }

    /// The base's newest odometry, which the read of describe_base found.
    fn newest_odometry(&self) -> Option<OdometryNow> {
        let (start, measurement) = lock_ignoring_poison(&self.inputs.odometry).newest()?;
        Some(OdometryNow {
            start,
            base: on_floor(&measurement.odom_from_base)?,
            arrived: self.inputs.scans_arrived(),
        })
    }

    /// Refuses with resource_unavailable while the lidar is silent or its place on the base
    /// is unknown.
    fn lidar_heard(&self) -> Result<(), Refusal> {
        let heard = self
            .inputs
            .since_scan()
            .is_some_and(|since| since <= self.source_timeout_ns());
        let placed = matches!(
            *lock_ignoring_poison(&self.inputs.mount),
            Mount::Placed { .. }
        );
        if heard && placed {
            return Ok(());
        }
        Err(Refusal::ResourceUnavailable(self.lidar_state()))
    }

    /// Why the lidar's scans do not reach the room, in words.
    fn lidar_state(&self) -> String {
        let silent = self
            .inputs
            .since_scan()
            .is_none_or(|since| since > self.source_timeout_ns());
        match &*lock_ignoring_poison(&self.inputs.mount) {
            _ if silent => format!(
                "the lidar is silent: no scan arrived within the source timeout ({} s); check \
                 the lidar's node",
                self.source_timeout.as_secs_f64()
            ),
            Mount::Unknown => "no scan of the lidar has named its frame yet".to_string(),
            Mount::Failing(why) => format!(
                "the lidar's place on the base is unknown: {why}; check that robot_initializer \
                 publishes the model's static_transforms"
            ),
            Mount::Placed { .. } => "the lidar's scans reach the room".to_string(),
        }
    }
}

/// Refuses with precondition_failed a base whose describe_base names commanded odometry.
fn refuse_commanded(facts: &BaseFacts) -> Result<(), Refusal> {
    if facts.commanded {
        return Err(Refusal::PreconditionFailed(
            "describe_base names commanded odometry: a map needs a base that measures its \
             motion"
                .to_string(),
        ));
    }
    Ok(())
}

/// A set_initial_pose target as a pose of the floor: in map alone, its position's z 0 and
/// its orientation a turn about +Z, each within control_core's planar tolerance.
fn parse_target(target: &TargetRequest) -> Result<control_core::planar_pose::PlanarPose, Refusal> {
    if target.frame != MAP_FRAME {
        return Err(Refusal::Rejected(format!(
            "the target is in {:?}: give it in {MAP_FRAME:?}, as AMCL takes an initial pose \
             in its global frame",
            target.frame
        )));
    }
    Pose::parse(target.position, target.orientation)
        .and_then(Pose::planar)
        .map_err(|e| Refusal::Rejected(format!("the target: {e}")))
}
