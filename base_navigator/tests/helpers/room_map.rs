//! A fake room_map on the generated mock of the navigator's room_map link: the
//! grids of maps of boxes on map_grids, each of a load of its map; a map_poses
//! after each report of the base, placing the robot frame where it truly stands
//! in the map, by a map frame the test sets turned and moved in the world; and a
//! task that answers each get_map_state from the state the test sets, so no
//! request of the node waits in the mock when the rig shuts down, counting the
//! asks it answered. The test switches map_poses off, skips the next few,
//! reports the base not localized in the next few, and holds the answers of
//! get_map_state, as a silent mapper holds them.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use control_core::sync::lock_ignoring_poison;
use frame_tree::arithmetic::quaternion_of_yaw;
use peppygen::consumed_services::room_map::get_map_state;
use peppygen::mock::deps::room_map::{self, map_grids, map_poses};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use super::mock_base::Planar;
use super::scene::Box3;

/// The map frame, as room_map names it.
pub const MAP_FRAME: &str = "map";

/// How long the answering task waits for the node's next get_map_state: a
/// bound no test reaches.
const NEXT_ASK: Duration = Duration::from_secs(3600);

/// When room_map loaded the tests' maps, unless a test loads one again.
const LOADED_AT: Duration = Duration::from_secs(7);

/// A map of boxes: its name, when room_map loaded it, its cells' side, the
/// corner of its first cell in map, its columns and rows, and the boxes it
/// holds occupied, in map, every other cell free.
#[derive(Clone, Debug, PartialEq)]
pub struct FakeMap {
    pub name: &'static str,
    pub loaded_at: SystemTime,
    pub cell_m: f64,
    pub origin: [f64; 2],
    pub columns: u32,
    pub rows: u32,
    pub occupied: Vec<Box3>,
}

impl FakeMap {
    /// The map `name`, loaded at [`LOADED_AT`], 10 m along map's x from -3 m
    /// and 6 m along its y from -3 m, in cells of 0.05 m, holding `occupied`.
    pub fn named(name: &'static str, occupied: &[Box3]) -> Self {
        Self {
            name,
            loaded_at: SystemTime::UNIX_EPOCH + LOADED_AT,
            cell_m: 0.05,
            origin: [-3.0, -3.0],
            columns: 200,
            rows: 120,
            occupied: occupied.to_vec(),
        }
    }

    /// `[x_min, x_max, y_min, y_max]` in map, as room_map gives a map's extent.
    pub fn extent(&self) -> [f64; 4] {
        let [x, y] = self.origin;
        [
            x,
            x + f64::from(self.columns) * self.cell_m,
            y,
            y + f64::from(self.rows) * self.cell_m,
        ]
    }

    /// The map loaded again `later` after its load.
    pub fn loaded_again(&self, later: Duration) -> Self {
        Self {
            loaded_at: self.loaded_at + later,
            ..self.clone()
        }
    }

    /// The map's grid stamped `timestamp`: 100 each cell whose center a box
    /// covers, 0 every other, row by row from the first.
    pub fn grid(&self, timestamp: SystemTime) -> map_grids::Message {
        let [x, y] = self.origin;
        let cells = (0..self.rows)
            .flat_map(|row| (0..self.columns).map(move |column| (column, row)))
            .map(|(column, row)| {
                let center = [
                    x + (f64::from(column) + 0.5) * self.cell_m,
                    y + (f64::from(row) + 0.5) * self.cell_m,
                ];
                if self.occupied.iter().any(|b| b.covers(center)) {
                    100
                } else {
                    0
                }
            })
            .collect();
        map_grids::Message {
            frame: MAP_FRAME.to_string(),
            timestamp,
            map_name: self.name.to_string(),
            map_load_timestamp: self.loaded_at,
            cell_size_m: self.cell_m,
            extent: Some(self.extent()),
            columns: self.columns,
            rows: self.rows,
            cells,
        }
    }

    /// get_map_state's map, as room_map describes the loaded map.
    pub fn described(&self) -> get_map_state::ResponseDataMap {
        get_map_state::ResponseDataMap {
            map_name: self.name.to_string(),
            cell_size_m: self.cell_m,
            extent: Some(self.extent()),
            scans: 120,
            mapping_start_timestamp: SystemTime::UNIX_EPOCH,
        }
    }
}

/// room_map's estimate of the robot frame in its loaded map: the map it is
/// in, where the world stands in that map, and whether the base is localized.
#[derive(Clone, Debug, PartialEq)]
pub struct FakeEstimate {
    pub map_name: &'static str,
    pub map_from_world: Planar,
    pub localized: bool,
}

/// What the fake gives, as the test sets it.
#[derive(Clone, Debug)]
pub struct RoomState {
    /// What get_map_state answers.
    pub answer: get_map_state::ResponseData,
    /// Whether get_map_state answers; while it does not, each ask waits.
    pub answers: bool,
    /// The estimate the map_poses carry, none while room_map holds none.
    pub estimate: Option<FakeEstimate>,
    /// Whether a map_poses follows each report of the base.
    pub poses_on: bool,
    /// How many of the next reports of the base get no map_poses.
    pub skip_poses: usize,
    /// How many of the next map_poses report the base not localized.
    pub unlocalized_poses: usize,
}

impl RoomState {
    /// `map` loaded and the base localized in it, the world standing in map
    /// at `map_from_world`.
    pub fn localized_in(map: &FakeMap, map_from_world: Planar) -> Self {
        Self {
            answer: get_map_state::ResponseData::new(
                Some(map.described()),
                false,
                false,
                false,
                true,
            ),
            answers: true,
            estimate: Some(FakeEstimate {
                map_name: map.name,
                map_from_world,
                localized: true,
            }),
            poses_on: true,
            skip_poses: 0,
            unlocalized_poses: 0,
        }
    }

    /// The map_poses of the report stamped `timestamp`, the robot frame truly
    /// at `truth` in the world, if one goes out: the next skipped or the
    /// estimate's, the next unlocalized reporting the base not localized.
    fn next_pose(&mut self, timestamp: SystemTime, truth: Planar) -> Option<map_poses::Message> {
        if !self.poses_on {
            return None;
        }
        if self.skip_poses > 0 {
            self.skip_poses -= 1;
            return None;
        }
        let estimate = self.estimate.as_ref()?;
        let unlocalized = self.unlocalized_poses > 0;
        self.unlocalized_poses = self.unlocalized_poses.saturating_sub(1);
        Some(map_pose(estimate, timestamp, truth, !unlocalized))
    }
}

/// `inner`, a pose in the frame `outer` places, in the frame `outer` is
/// placed in.
pub fn placed(outer: Planar, inner: Planar) -> Planar {
    let (sin, cos) = outer.yaw.sin_cos();
    Planar {
        x: outer.x + cos * inner.x - sin * inner.y,
        y: outer.y + sin * inner.x + cos * inner.y,
        yaw: outer.yaw + inner.yaw,
    }
}

/// The map_poses of `estimate` stamped `timestamp`, the robot frame truly at
/// `truth` in the world, reporting the base localized when `localized`.
fn map_pose(
    estimate: &FakeEstimate,
    timestamp: SystemTime,
    truth: Planar,
    localized: bool,
) -> map_poses::Message {
    let pose = placed(estimate.map_from_world, truth);
    map_poses::Message {
        frame: MAP_FRAME.to_string(),
        timestamp,
        map_name: estimate.map_name.to_string(),
        position: [pose.x, pose.y, 0.0],
        orientation: quaternion_of_yaw(pose.yaw),
        position_std_m: 0.05,
        orientation_std_rad: 0.02,
        localized: localized && estimate.localized,
    }
}

/// The fake room_map: its mock, the state the test sets, the task that
/// answers get_map_state, which ends with the fake, and the number of asks it
/// answered.
pub struct FakeRoomMap {
    mock: Arc<room_map::Mock>,
    pub state: Arc<Mutex<RoomState>>,
    answering: JoinHandle<()>,
    answered: watch::Receiver<u64>,
}

impl Drop for FakeRoomMap {
    fn drop(&mut self) {
        self.answering.abort();
    }
}

/// Answer each get_map_state of the node from `state` while it answers,
/// counting each answer in `answered` once it went out; while it does not,
/// hold each ask unanswered.
async fn answer_each(
    mock: Arc<room_map::Mock>,
    state: Arc<Mutex<RoomState>>,
    answered: watch::Sender<u64>,
) {
    let mut held = Vec::new();
    while let Ok(responder) = mock.get_map_state.next_request(NEXT_ASK).await {
        let (answers, answer) = {
            let state = lock_ignoring_poison(&state);
            (state.answers, state.answer.clone())
        };
        if !answers {
            held.push(responder);
            continue;
        }
        // An ask the node gave up on takes no answer, which nothing reads.
        let _ = responder.respond(answer).await;
        answered.send_modify(|count| *count += 1);
    }
}

impl FakeRoomMap {
    /// The fake on `mock`, giving `state`.
    pub fn new(mock: room_map::Mock, state: RoomState) -> Self {
        let mock = Arc::new(mock);
        let state = Arc::new(Mutex::new(state));
        let (counting, answered) = watch::channel(0);
        let answering = tokio::spawn(answer_each(mock.clone(), state.clone(), counting));
        Self {
            mock,
            state,
            answering,
            answered,
        }
    }

    /// The asks of get_map_state the fake answered, which a test reads changes of.
    pub fn answered(&self) -> watch::Receiver<u64> {
        self.answered.clone()
    }

    /// Whether the fake answers get_map_state.
    pub fn answers(&self) -> bool {
        lock_ignoring_poison(&self.state).answers
    }

    /// Change the state the fake gives.
    pub fn set(&self, change: impl FnOnce(&mut RoomState)) {
        change(&mut lock_ignoring_poison(&self.state));
    }

    /// Publish the grid of `map` stamped `timestamp`.
    pub async fn publish_grid(&self, map: &FakeMap, timestamp: SystemTime) {
        self.mock
            .map_grids
            .publish(&map.grid(timestamp))
            .await
            .expect("the fake publishes the map's grid");
    }

    /// Publish the map_poses that follows the report stamped `timestamp`, the
    /// robot frame truly at `truth` in the world, if one goes out.
    pub async fn follow_report(&self, timestamp: SystemTime, truth: Planar) {
        let pose = lock_ignoring_poison(&self.state).next_pose(timestamp, truth);
        if let Some(pose) = pose {
            self.mock
                .map_poses
                .publish(&pose)
                .await
                .expect("the fake publishes map_poses");
        }
    }
}
