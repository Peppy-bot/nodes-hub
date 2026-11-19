//! transform_lookup's lookup_transform over the wire, from a tree whose publishers are the
//! robot's initializer, which sends the fixed transform of the camera on the arm, and the
//! backbone, which sends the measured transform of the arm in the base link.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use control_core::fault::Fault;
use peppygen::consumed_topics::transforms::{static_transforms, transforms};
use peppygen::fixtures::exposed_services::transform_lookup::lookup_transform::{
    self, Failure, FailureKind, RequestData, Response,
};
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppygen::{NodeRunner, Parameters};
use peppylib::testing::HarnessClock;

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);

/// The longest wait a request takes. The tests' simulation clock never passes it, so a
/// lookup that waits ends when a message lets it succeed.
const LONGEST_WAIT_S: f64 = 1.0;

/// The publishers of the tree, as the harness binds them.
const INITIALIZER: &str = "init_inst";
const BACKBONE: &str = "backbone";

const NANOS_PER_SECOND: u64 = 1_000_000_000;

fn at(seconds: f64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs_f64(seconds)
}

/// The node on the harness's simulation clock, which reads 0 until a test ticks it.
fn config() -> Config {
    Config {
        parameters: Some(Parameters {
            cache_window_s: 10.0,
        }),
        transforms_instance_ids: vec![INITIALIZER.into(), BACKBONE.into()],
        clock: HarnessClock::Consumer,
        ..Config::default()
    }
}

fn setup(
    params: Parameters,
    runner: Arc<NodeRunner>,
) -> impl Future<Output = peppygen::Result<()>> + Send + 'static {
    transform_server::setup(params, runner, Fault::default())
}

async fn started() -> peppygen::Result<(Harness, Mocks)> {
    Harness::start_with(config(), setup).await
}

/// The camera fixed 1 m above the arm link, as the initializer sends it.
fn camera_on_the_arm() -> static_transforms::Message {
    static_transforms::Message {
        child_frames: vec!["camera".into()],
        parent_frames: vec!["arm".into()],
        positions: vec![0.0, 0.0, 1.0],
        orientations: vec![0.0, 0.0, 0.0, 1.0],
    }
}

/// The arm link `x` m along the base link's x, measured at `seconds`.
fn arm_at(seconds: f64, x: f64) -> transforms::Message {
    transforms::Message {
        timestamp: at(seconds),
        child_frames: vec!["arm".into()],
        parent_frames: vec!["base_link".into()],
        positions: vec![x, 0.0, 0.0],
        orientations: vec![0.0, 0.0, 0.0, 1.0],
        start_timestamp: at(1.0),
    }
}

fn request(target: &str, source: &str, time: Option<f64>, timeout_s: f64) -> RequestData {
    RequestData {
        target_frame: target.into(),
        source_frame: source.into(),
        time: time.map(at),
        timeout_s,
    }
}

async fn lookup(harness: &Harness, request: &RequestData) -> Result<Response, Failure> {
    lookup_transform::poll(harness, request, WIRE)
        .await
        .map_err(|error| Failure::of(&error).expect("an error reply of the member"))
}

/// Publishes the camera on the arm and the arm at 20 s and 21 s, and waits until the tree
/// holds them all.
async fn publish_the_tree(harness: &Harness, mocks: &Mocks) -> peppygen::Result<()> {
    let [initializer, backbone] = &mocks.deps.transforms[..] else {
        panic!("the harness binds two publishers");
    };
    initializer
        .static_transforms
        .publish(&camera_on_the_arm())
        .await?;
    backbone.transforms.publish(&arm_at(20.0, 1.0)).await?;
    backbone.transforms.publish(&arm_at(21.0, 2.0)).await?;
    let newest = lookup(
        harness,
        &request("base_link", "camera", Some(21.0), LONGEST_WAIT_S),
    )
    .await;
    assert_eq!(newest.map(|answer| answer.position), Ok([2.0, 0.0, 1.0]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lookup_at_a_time_interpolates_and_one_at_the_latest_time_answers_its_time()
-> peppygen::Result<()> {
    let (harness, mocks) = started().await?;
    publish_the_tree(&harness, &mocks).await?;

    let between = lookup(&harness, &request("base_link", "camera", Some(20.5), 0.0))
        .await
        .expect("a lookup between two stamps");
    assert_eq!(between.position, [1.5, 0.0, 1.0]);
    assert_eq!(between.orientation, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(between.timestamp, Some(at(20.5)));

    let latest = lookup(&harness, &request("base_link", "camera", None, 0.0))
        .await
        .expect("a lookup at the latest time");
    assert_eq!(latest.position, [2.0, 0.0, 1.0]);
    assert_eq!(latest.timestamp, Some(at(21.0)));

    let fixed = lookup(&harness, &request("arm", "camera", None, 0.0))
        .await
        .expect("a chain of static transforms");
    assert_eq!(fixed.position, [0.0, 0.0, 1.0]);
    assert_eq!(fixed.timestamp, None, "a static chain holds at every time");

    let reverse = lookup(&harness, &request("camera", "base_link", Some(20.0), 0.0))
        .await
        .expect("the reverse lookup");
    assert_eq!(reverse.position, [-1.0, 0.0, -1.0]);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_failure_of_a_lookup_gives_its_kind_and_names_its_cause() -> peppygen::Result<()> {
    let (harness, mocks) = started().await?;
    publish_the_tree(&harness, &mocks).await?;
    let [initializer, _] = &mocks.deps.transforms[..] else {
        panic!("the harness binds two publishers");
    };
    initializer
        .static_transforms
        .publish(&static_transforms::Message {
            child_frames: vec!["camera".into(), "head".into()],
            parent_frames: vec!["arm".into(), "torso".into()],
            positions: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            orientations: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        })
        .await?;
    lookup(&harness, &request("torso", "head", None, LONGEST_WAIT_S))
        .await
        .expect("the tree takes the initializer's new set");

    let unknown = lookup(&harness, &request("base_link", "gripper", None, 0.0))
        .await
        .expect_err("no transform names the gripper");
    assert_eq!(unknown.kind, FailureKind::UnknownFrame);
    assert!(
        unknown.message.ends_with(
            "gripper: look up one of the frames it holds: arm, base_link, camera, head, torso"
        ),
        "{}",
        unknown.message
    );

    let apart = lookup(&harness, &request("torso", "camera", None, 0.0))
        .await
        .expect_err("the torso and the camera are in two trees");
    assert_eq!(apart.kind, FailureKind::NoChain);
    assert!(apart.message.contains("torso"), "{}", apart.message);

    let later = lookup(&harness, &request("base_link", "camera", Some(30.0), 0.0))
        .await
        .expect_err("the arm holds no stamp at 30 s");
    assert_eq!(later.kind, FailureKind::Extrapolation);
    assert!(
        later.message.contains("base_link -> arm"),
        "{}",
        later.message
    );

    let unnamed = lookup(&harness, &request("", "camera", None, 0.0))
        .await
        .expect_err("an empty target frame");
    assert_eq!(unnamed.kind, FailureKind::Rejected);
    assert_eq!(
        unnamed.message,
        "target_frame is empty: send the name of a link of the robot's URDF"
    );

    let negative = lookup(&harness, &request("base_link", "camera", None, -1.0))
        .await
        .expect_err("a negative wait");
    assert_eq!(negative.kind, FailureKind::Rejected);

    let too_long = lookup(&harness, &request("base_link", "camera", None, 1.5))
        .await
        .expect_err("a wait above 1 s");
    assert_eq!(too_long.kind, FailureKind::Rejected);
    assert!(
        too_long.message.ends_with("send a timeout_s from 0 to 1"),
        "{}",
        too_long.message
    );

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_transform_from_two_publishers_breaks_the_tree_and_names_both() -> peppygen::Result<()> {
    let (harness, mocks) = started().await?;
    publish_the_tree(&harness, &mocks).await?;
    let [initializer, _] = &mocks.deps.transforms[..] else {
        panic!("the harness binds two publishers");
    };
    // The initializer sends the arm too, and a marker on it that only it sends: once the
    // marker stands on the arm, the tree holds the arm from both publishers.
    initializer
        .transforms
        .publish(&transforms::Message {
            child_frames: vec!["arm".into(), "marker".into()],
            parent_frames: vec!["base_link".into(), "arm".into()],
            positions: vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            orientations: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ..arm_at(21.0, 2.0)
        })
        .await?;
    lookup(
        &harness,
        &request("arm", "marker", Some(21.0), LONGEST_WAIT_S),
    )
    .await
    .expect("the marker stands on the arm");

    let broken = lookup(&harness, &request("base_link", "camera", Some(21.0), 0.0))
        .await
        .expect_err("the arm has two publishers");
    assert_eq!(broken.kind, FailureKind::BrokenTree);
    assert!(
        broken.message.contains(INITIALIZER) && broken.message.contains(BACKBONE),
        "{}",
        broken.message
    );

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lookup_sent_with_the_message_that_lets_it_succeed_answers_it() -> peppygen::Result<()> {
    let (harness, mocks) = started().await?;
    harness.clock.tick(21 * NANOS_PER_SECOND).await?;
    let [initializer, backbone] = &mocks.deps.transforms[..] else {
        panic!("the harness binds two publishers");
    };
    initializer
        .static_transforms
        .publish(&camera_on_the_arm())
        .await?;
    // The simulation clock stands at 21 s, so the wait of 1 ms on it does not run out. The
    // request and the backbone's stamp of 21 s go out together, and whichever reaches the
    // node first, the lookup answers with the place the stamp gives.
    let waiting = request("base_link", "camera", Some(21.0), 0.001);
    let measured = arm_at(21.0, 2.0);
    let (answer, published) = tokio::join!(
        lookup(&harness, &waiting),
        backbone.transforms.publish(&measured),
    );
    published?;
    assert_eq!(answer.map(|answer| answer.position), Ok([2.0, 0.0, 1.0]));

    harness.shutdown().await
}
