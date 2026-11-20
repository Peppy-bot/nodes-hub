//! Recorded drives, on a developer's machine alone: a lidar recording of Waldo's from
//! ros_reference's recording cache (`${XDG_CACHE_HOME:-$HOME/.cache}/peppy-ros-reference/
//! recordings`), replayed through the node's topics on the rig of `common`, each record of
//! the base and each scan at its own stamp, the next scan sent once the mapping counted the
//! one before. Each test is ignored by default; `cargo test -- --ignored` runs it where the
//! cache holds its recording.

use std::path::PathBuf;

use peppygen::exposed_services::room_map::set_initial_pose::SetInitialPoseTarget;
use peppygen::fixtures::emitted_topics::room_map::map_poses;
use peppygen::fixtures::exposed_services::room_map::{
    finish_mapping, get_map_state, load_map, set_initial_pose, start_mapping,
};
use serde_json::Value;

mod common;
use common::{Fan, HANG, LIDAR_FRAME, NANOS_PER_SECOND, Rig, WIRE, at, first_after};

/// Two laps of Waldo's differential base through two rooms joined by a doorway, around three
/// crates, with odometry that drifts by its wheels' slip alone: 1873 scans of an 800-ray
/// lidar 0.4 m above base_footprint, the first lap scans 0 to 907.
const TWO_ROOMS: &str = "e2b7a32474887447148c975fcd33c465bde0159487b41bd5bc7d8a0a88b6b5d8";

/// A record of the base: its stamp (ns), and its pose by its odometry and its true pose in
/// the world, each `[x, y, yaw]`.
#[derive(Clone, Copy, Debug)]
struct BaseRecord {
    at_ns: i64,
    odometry: [f64; 3],
    truth: [f64; 3],
}

/// A lidar recording: the lidar's fan and its mount on the base `[x, y, z, yaw]`, each
/// scan's stamp (ns) and ranges, the base's records, and the scans of each lap, `[first,
/// end)`.
struct Recording {
    fan: Fan,
    mount: [f64; 4],
    stamps: Vec<i64>,
    scans: Vec<Vec<f32>>,
    base: Vec<BaseRecord>,
    laps: Vec<[usize; 2]>,
}

impl Recording {
    /// The recording `hash` of the cache.
    fn read(hash: &str) -> Self {
        let cache = std::env::var_os("XDG_CACHE_HOME").map_or_else(
            || {
                PathBuf::from(std::env::var_os("HOME").expect("HOME names a directory"))
                    .join(".cache")
            },
            PathBuf::from,
        );
        let directory = cache.join("peppy-ros-reference/recordings").join(hash);
        let read = |file: &str| {
            std::fs::read(directory.join(file)).unwrap_or_else(|e| {
                panic!(
                    "{}: {e}: the recording is not in the cache",
                    directory.join(file).display()
                )
            })
        };
        let manifest: Value =
            serde_json::from_slice(&read("recording.json")).expect("the manifest is JSON");
        let number = |value: &Value| value.as_f64().expect("a number");
        let lidar = &manifest["lidar"];
        let rays = lidar["rays"].as_u64().expect("a count of rays") as usize;
        let fan = Fan {
            min_angle_rad: number(&lidar["angle_min_rad"]) as f32,
            angle_increment_rad: number(&lidar["angle_increment_rad"]) as f32,
            time_increment_s: number(&lidar["time_increment_s"]) as f32,
            min_range_m: number(&lidar["range_min_m"]) as f32,
            max_range_m: number(&lidar["range_max_m"]) as f32,
        };
        let mount: [f64; 4] = serde_json::from_value(manifest["robot"]["mount"].clone())
            .expect("a mount of four numbers");
        let stamps = manifest["stamps"]
            .as_array()
            .expect("the scans' stamps")
            .iter()
            .map(|stamp| stamp[0].as_i64().expect("a stamp (ns)"))
            .collect();
        let scans = read("scans.f32")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>()
            .chunks_exact(rays)
            .map(<[f32]>::to_vec)
            .collect();
        let base = read("base.f64")
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| f64::from_le_bytes(*bytes))
            .collect::<Vec<_>>()
            .as_chunks::<7>()
            .0
            .iter()
            .map(|record| BaseRecord {
                at_ns: record[0] as i64,
                odometry: [record[1], record[2], record[3]],
                truth: [record[4], record[5], record[6]],
            })
            .collect();
        let laps =
            serde_json::from_value(manifest["drive"]["laps"].clone()).expect("the laps' scans");
        Recording {
            fan,
            mount,
            stamps,
            scans,
            base,
            laps,
        }
    }

    /// The time (ns) from a scan's first ray to its last.
    fn sweep_ns(&self, scan: usize) -> i64 {
        let rays = self.scans[scan].len() as f64;
        ((rays - 1.0) * f64::from(self.fan.time_increment_s) * NANOS_PER_SECOND).round() as i64
    }

    /// The base's record at `at_ns`.
    fn base_at(&self, at_ns: i64) -> BaseRecord {
        *self
            .base
            .iter()
            .find(|record| record.at_ns == at_ns)
            .expect("a record of the base at each scan's stamp")
    }
}

/// The recording replayed through the rig, its times moved past the rig's clock.
struct Replay<'a> {
    rig: &'a Rig,
    drive: &'a Recording,
    next_record: usize,
    offset_s: f64,
}

impl Replay<'_> {
    fn seconds(&self, at_ns: i64) -> f64 {
        self.offset_s + at_ns as f64 / NANOS_PER_SECOND
    }

    /// Scan `index`, after the records of the base up to one at or after its last ray.
    async fn scan(&mut self, index: usize) {
        let stamp = self.drive.stamps[index];
        let last_ray = stamp + self.drive.sweep_ns(index);
        loop {
            let record = self.drive.base[self.next_record];
            self.next_record += 1;
            let at_s = self.seconds(record.at_ns);
            self.rig.tick_to(at_s).await;
            self.rig.publish_base_state(at_s, record.odometry).await;
            if record.at_ns >= last_ray {
                break;
            }
        }
        self.rig
            .publish_scan(
                self.seconds(stamp),
                &self.drive.fan,
                self.drive.scans[index].clone(),
            )
            .await;
    }
}

/// Waits until the mapping has read `scans` scans.
async fn mapped(rig: &Rig, scans: u32) {
    let read = async {
        loop {
            let state = get_map_state::poll(&rig.harness, WIRE)
                .await
                .expect("a state");
            if state.map.is_some_and(|map| map.scans >= scans) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    };
    tokio::time::timeout(HANG, read)
        .await
        .unwrap_or_else(|_| panic!("the mapping read no {scans} scans before the hang guard"));
}

/// `a` composed with `b`, poses of the floor `[x, y, yaw]`.
fn compose(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let (sin, cos) = a[2].sin_cos();
    [
        a[0] + cos * b[0] - sin * b[1],
        a[1] + sin * b[0] + cos * b[1],
        a[2] + b[2],
    ]
}

fn inverse(a: [f64; 3]) -> [f64; 3] {
    let (sin, cos) = a[2].sin_cos();
    [-cos * a[0] - sin * a[1], sin * a[0] - cos * a[1], -a[2]]
}

/// The yaw of a turn about +Z, `[x, y, z, w]`.
fn yaw_of(orientation: &[f64]) -> f64 {
    2.0 * orientation[2].atan2(orientation[3])
}

/// The cell of the map (m): the mapping's resolution in `tests/parameters.json`.
fn cell_m() -> f64 {
    let parameters: Value = serde_json::from_str(include_str!("parameters.json"))
        .expect("tests/parameters.json is JSON");
    parameters["mapping"]["resolution"]
        .as_f64()
        .expect("a resolution")
}

/// How far (m) the farthest ray end of scan `index` lies from its true place: each ray's end
/// as the base at `estimate` places it, and as the base at `truth` does.
fn farthest_offset(drive: &Recording, index: usize, estimate: [f64; 3], truth: [f64; 3]) -> f64 {
    drive.scans[index]
        .iter()
        .enumerate()
        .map(|(ray, range)| {
            let angle = f64::from(drive.fan.min_angle_rad)
                + ray as f64 * f64::from(drive.fan.angle_increment_rad);
            let end = [
                f64::from(*range) * angle.cos(),
                f64::from(*range) * angle.sin(),
                0.0,
            ];
            let [ex, ey, _] = compose(estimate, end);
            let [tx, ty, _] = compose(truth, end);
            (ex - tx).hypot(ey - ty)
        })
        .fold(0.0, f64::max)
}

/// The lap `[first, end)` mapped as `two_rooms` and saved: the first scan it mapped, and the
/// world from its map frame. The lap's scans make the lidar heard and its place known, after
/// which mapping starts and takes each scan that arrives after the call.
async fn map_lap(rig: &Rig, replay: &mut Replay<'_>, [first, end]: [usize; 2]) -> MappedLap {
    let request = start_mapping::RequestData {
        map_name: "two_rooms".into(),
    };
    let mut mapped_from = first;
    loop {
        replay.scan(mapped_from).await;
        mapped_from += 1;
        let Err(refused) = start_mapping::poll(&rig.harness, &request, WIRE).await else {
            break;
        };
        let failure = start_mapping::Failure::of(&refused).expect("an error reply");
        assert!(
            failure.kind == start_mapping::FailureKind::ResourceUnavailable && mapped_from < end,
            "{failure:?}"
        );
    }
    for index in mapped_from..end {
        replay.scan(index).await;
        mapped(rig, (index - mapped_from + 1) as u32).await;
    }
    // The map frame is the odometry frame of the first scan mapped, as karto keeps that
    // scan's odometric pose: the world from the map is the truth at that scan less its
    // odometry.
    let first_mapped = replay.drive.base_at(replay.drive.stamps[mapped_from]);
    MappedLap {
        mapped_from,
        world_from_map: compose(first_mapped.truth, inverse(first_mapped.odometry)),
    }
}

/// A lap mapped: the first scan mapping took, and the world from the map frame.
struct MappedLap {
    mapped_from: usize,
    world_from_map: [f64; 3],
}

impl MappedLap {
    /// The true pose of the base at `record` in the map frame.
    fn truth_of(&self, record: &BaseRecord) -> [f64; 3] {
        compose(inverse(self.world_from_map), record.truth)
    }
}

/// The lidar's place on the base as the recording mounts it, the answer of the node's
/// lookup.
fn enqueue_mount(rig: &Rig, drive: &Recording) {
    let [x, y, z, yaw] = drive.mount;
    let (sin, cos) = (yaw / 2.0).sin_cos();
    rig.mocks
        .deps
        .transform_lookup
        .lookup_transform
        .enqueue_response(common::lookup_answer(
            LIDAR_FRAME,
            [x, y, z],
            [0.0, 0.0, sin, cos],
        ))
        .expect("a lookup answer");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "reads a recording of the developer's cache"]
async fn the_map_of_a_loop_through_two_rooms_closes_the_loop() {
    let drive = Recording::read(TWO_ROOMS);
    let mut rig = Rig::start("recorded_loop", "measured", "").await;
    enqueue_mount(&rig, &drive);
    let mut replay = Replay {
        rig: &rig,
        drive: &drive,
        next_record: 0,
        offset_s: rig.now_s.get(),
    };
    let [_, end] = drive.laps[0];
    let lap = map_lap(&rig, &mut replay, drive.laps[0]).await;
    let last = end - 1;
    let last_s = replay.seconds(drive.stamps[last]);
    let estimate = first_after(
        async || {
            rig.harness
                .emitted
                .room_map_map_poses
                .next()
                .await
                .ok()
                .flatten()
        },
        |pose: &map_poses::Message| pose.timestamp,
        at(last_s - 1e-6),
    )
    .await;
    finish_mapping::poll(&rig.harness, WIRE)
        .await
        .expect("the map is saved");
    let truth = lap.truth_of(&drive.base_at(drive.stamps[last]));
    let estimated = [
        estimate.position[0],
        estimate.position[1],
        yaw_of(&estimate.orientation),
    ];
    let farthest = farthest_offset(&drive, last, estimated, truth);
    let cell = cell_m();
    println!(
        "lap 1, scans {} to {last}: the farthest ray end lies {farthest} m off; estimate \
         {estimated:?}, truth {truth:?}",
        lap.mapped_from
    );
    assert!(
        farthest <= cell,
        "a wall the lap's last scan sees lies {farthest} m off its place, more than a cell \
         ({cell} m): estimate {estimated:?}, truth {truth:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "reads a recording of the developer's cache"]
async fn after_a_restart_set_initial_pose_matches_in_the_saved_map() {
    let drive = Recording::read(TWO_ROOMS);
    let rig = Rig::start("recorded_match", "measured", "").await;
    enqueue_mount(&rig, &drive);
    let mut replay = Replay {
        rig: &rig,
        drive: &drive,
        next_record: 0,
        offset_s: rig.now_s.get(),
    };
    let lap = map_lap(&rig, &mut replay, drive.laps[0]).await;
    finish_mapping::poll(&rig.harness, WIRE)
        .await
        .expect("the map is saved");
    // The node restarts; the base and its lidar go on from where they stood.
    let (next_record, offset_s) = (replay.next_record, replay.offset_s);
    let rig = rig.restart("measured", "", 8).await;
    enqueue_mount(&rig, &drive);
    let mut replay = Replay {
        rig: &rig,
        drive: &drive,
        next_record,
        offset_s,
    };
    load_map::poll(
        &rig.harness,
        &load_map::RequestData {
            map_name: "two_rooms".into(),
        },
        WIRE,
    )
    .await
    .expect("the saved map loads");
    let [second, end] = drive.laps[1];
    replay.scan(second).await;
    // The target lies 0.3 m and 0.3 rad off the base's true pose at the newest record of
    // the base, criterion 2's offsets, the 0.3 m along the diagonal so that it falls between
    // the cells of the matcher's correlation grid; searched within 0.5 m and 0.5 rad.
    let record = drive.base[replay.next_record - 1];
    let [tx, ty, tyaw] = lap.truth_of(&record);
    let (sin, cos) = ((tyaw + 0.3) / 2.0).sin_cos();
    let request = set_initial_pose::RequestData {
        target: SetInitialPoseTarget {
            frame: "map".into(),
            timestamp: Some(at(replay.seconds(record.at_ns))),
            position: [
                tx + 0.3 * std::f64::consts::FRAC_1_SQRT_2,
                ty + 0.3 * std::f64::consts::FRAC_1_SQRT_2,
                0.0,
            ],
            orientation: [0.0, 0.0, sin, cos],
        },
        search_radius_m: 0.5,
        search_angle_rad: 0.5,
    };
    let pending = set_initial_pose::poll(&rig.harness, &request, WIRE);
    tokio::pin!(pending);
    let mut next = second + 1;
    let answer = loop {
        tokio::select! {
            answer = &mut pending => break answer,
            () = replay.scan(next), if next < end => next += 1,
        }
    };
    let matched = answer
        .expect("the scan after the call matches")
        .current_pose;
    let matched_s = matched
        .timestamp
        .duration_since(std::time::UNIX_EPOCH)
        .expect("after the epoch")
        .as_secs_f64();
    let index = drive
        .stamps
        .iter()
        .position(|stamp| (replay.seconds(*stamp) - matched_s).abs() < 1e-6)
        .expect("the matched scan is a scan of the drive");
    let truth = lap.truth_of(&drive.base_at(drive.stamps[index]));
    let estimated = [
        matched.position[0],
        matched.position[1],
        yaw_of(&matched.orientation),
    ];
    let farthest = farthest_offset(&drive, index, estimated, truth);
    let cell = cell_m();
    println!(
        "set_initial_pose matched at scan {index}: the farthest ray end lies {farthest} m off; \
         match {estimated:?}, truth {truth:?}"
    );
    assert!(
        farthest <= cell,
        "the match places a wall {farthest} m off its place, more than a cell ({cell} m): \
         match {estimated:?}, truth {truth:?}"
    );
}
