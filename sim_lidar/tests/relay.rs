//! Integration tests over the generated harness: the node in-process, the
//! simulation peer played by the generated pairing mock over the real wire,
//! the robot's sensor_mounts played by the generated service mock, and the
//! contract surface observed the way a consumer sees it.

use std::time::{Duration, SystemTime};

use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppygen::mock::deps::sensor_mounts::describe_sensors;
use peppygen::mock::pairings::simulation::scan_stream as simulation_scans;
use peppygen::paired_topics::simulation::scan_stream::MessageHeader;

/// Bounds every wait for a relayed scan and for the node's ask. Generous:
/// the assertions are about what arrives, never about how fast.
const TIMEOUT: Duration = Duration::from_secs(10);

/// The id the relay runs under. The harness runs a node outside any copy, so
/// the id is the lidar's name as it is; a copy's prefix coming off is covered
/// by camera_core's own tests.
const LIDAR: &str = "lidar";

/// The frame sensor_mounts gives the lidar [`LIDAR`].
const LIDAR_FRAME: &str = "lidar_frame";

/// A frame that is not the lidar's, which a simulation that sweeps another
/// lidar, or another robot's, names.
const OTHER_FRAME: &str = "rear_lidar_frame";

/// A scan the simulation sweeps of the lidar [`LIDAR`], in its frame: four
/// rays a quarter turn apart, their ranges a surface, the mark of a ray that
/// met nothing, the mark of a ray the lidar could not measure and a surface
/// near the far limit.
fn scan(timestamp: SystemTime, capture_index: u32) -> simulation_scans::Message {
    simulation_scans::Message {
        header: MessageHeader {
            timestamp,
            frame: LIDAR_FRAME.to_string(),
            capture_index,
        },
        min_angle_rad: -std::f32::consts::PI,
        angle_increment_rad: std::f32::consts::FRAC_PI_2,
        time_increment_s: -0.025,
        min_range_m: 0.05,
        max_range_m: 12.0,
        ranges: vec![1.25, 10_000.0, 0.0, 11.5],
        intensities: vec![0.5, 0.0, 0.0, 0.25],
    }
}

/// The answer of sensor_mounts for a robot with a chest camera and the lidar
/// [`LIDAR`] on its base.
fn described_sensors() -> describe_sensors::ResponseData {
    let strings = |values: &[&str]| values.iter().map(|value| (*value).to_owned()).collect();
    describe_sensors::ResponseData {
        robot_frame: "base_footprint".to_owned(),
        sensor_names: strings(&["chest", LIDAR]),
        sensor_kinds: strings(&["rgbd_camera", "lidar_2d"]),
        sensor_frames: strings(&["chest_color_optical_frame", LIDAR_FRAME]),
        carried_by: strings(&["", ""]),
    }
}

/// A boot of the relay under `instance_id`, its ask of sensor_mounts answered
/// with [`described_sensors`].
async fn started(instance_id: &str) -> peppygen::Result<(Harness, Mocks)> {
    let config = Config {
        instance_id: Some(instance_id.to_string()),
        ..Config::default()
    };
    let (harness, mocks) = Harness::start_with(config, sim_lidar::setup).await?;
    mocks
        .deps
        .sensor_mounts
        .describe_sensors
        .next_request(TIMEOUT)
        .await?
        .respond(described_sensors())
        .await?;
    Ok((harness, mocks))
}

/// The error the start of a relay under `instance_id` fails with.
async fn refusal_of(instance_id: &str) -> String {
    let (harness, _mocks) = started(instance_id).await.expect("the relay boots");
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    while !harness.setup_finished() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the setup never returned"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    harness
        .shutdown()
        .await
        .expect_err("the start fails")
        .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relays_scans_field_for_field_at_any_timestamp() -> peppygen::Result<()> {
    let (mut harness, mocks) = started(LIDAR).await?;

    // A scan stamped at the epoch is a scan taken at that time, which the relay
    // passes on like any other. The publishes share one mock publisher, so their
    // order holds.
    let timestamp = SystemTime::UNIX_EPOCH + Duration::from_secs(1_780_000_000);
    let published = [scan(SystemTime::UNIX_EPOCH, 7), scan(timestamp, 42)];
    for sent in &published {
        mocks.pairings.simulation.scan_stream.publish(sent).await?;
    }

    // Field for field: each scan keeps its stamp, its frame, its counter and
    // its fan, which place each ray in time and angle, and both range marks.
    for sent in &published {
        let relayed = tokio::time::timeout(TIMEOUT, harness.emitted.lidar_scan_stream.next())
            .await
            .expect("no scan reached the contract surface")?
            .expect("scan_stream subscription should be open");
        assert_eq!(relayed.header.timestamp, sent.header.timestamp);
        assert_eq!(relayed.header.frame, LIDAR_FRAME);
        assert_eq!(relayed.header.capture_index, sent.header.capture_index);
        assert_eq!(
            (
                relayed.min_angle_rad,
                relayed.angle_increment_rad,
                relayed.time_increment_s,
                relayed.min_range_m,
                relayed.max_range_m,
            ),
            (
                sent.min_angle_rad,
                sent.angle_increment_rad,
                sent.time_increment_s,
                sent.min_range_m,
                sent.max_range_m,
            )
        );
        assert_eq!(relayed.ranges, sent.ranges);
        assert_eq!(relayed.intensities, sent.intensities);
    }

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_scan_in_another_frame_than_the_lidars_is_dropped() -> peppygen::Result<()> {
    let (mut harness, mocks) = started(LIDAR).await?;

    // The two publishes share one mock publisher, so their order holds: the
    // first scan to surface proves both the drop and the relay.
    let timestamp = SystemTime::UNIX_EPOCH + Duration::from_secs(1_780_000_000);
    let mut foreign = scan(timestamp, 7);
    foreign.header.frame = OTHER_FRAME.to_string();
    let simulation = &mocks.pairings.simulation;
    simulation.scan_stream.publish(&foreign).await?;
    simulation.scan_stream.publish(&scan(timestamp, 8)).await?;

    let relayed = tokio::time::timeout(TIMEOUT, harness.emitted.lidar_scan_stream.next())
        .await
        .expect("no scan reached the contract surface")?
        .expect("scan_stream subscription should be open");
    assert_eq!(relayed.header.frame, LIDAR_FRAME);
    assert_eq!(relayed.header.capture_index, 8);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lidar_sensor_mounts_does_not_list_fails_its_start() {
    let refusal = refusal_of("rear_lidar").await;
    assert!(
        refusal.contains(
            "sensor_mounts' describe_sensors names no sensor \"rear_lidar\" (its sensor_names: \
             chest, lidar): run the lidar under one of those names"
        ),
        "{refusal}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_relay_under_a_cameras_name_fails_its_start() {
    let refusal = refusal_of("chest").await;
    assert!(
        refusal.contains(
            "sensor_mounts' describe_sensors names \"chest\" a sensor of \"rgbd_camera\": run \
             the lidar under the name of a lidar, a sensor of lidar_2d"
        ),
        "{refusal}"
    );
}
