//! The planner's voxel map over the wire: each message of occupancy_map becomes the
//! cubes of the map at the time the planner read it, and a map that holds no depth frame
//! clears them and says why.

mod common;

use std::time::{Duration, SystemTime};

use common::{TIMEOUT, bound, publish_once_subscribed, start};
use peppygen::consumed_topics::motion_planning::occupancy_map::Message;
use rerun_viewer::config::default_parameters;
use rerun_viewer::draw::planner_map::{MAP_PATH, MAP_STATE_PATH};
use rerun_viewer::drawn::Drawn;

/// A map read at second `second` of the planner's clock, of cubes of 2 cm.
fn map(second: u64, centers: Vec<f64>, waiting: &str) -> Message {
    Message {
        timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(second),
        voxel_size_m: 0.02,
        centers,
        waiting: waiting.to_string(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn each_map_is_drawn_and_a_map_without_a_frame_clears_the_cubes() -> Result<(), String> {
    let (harness, mocks, canvas) = start(bound(&["motion_planning"], default_parameters())).await?;
    let topic = &mocks
        .deps
        .motion_planning
        .as_ref()
        .expect("the planner is linked")
        .occupancy_map;
    let held = map(5, vec![0.31, 0.01, 0.51, 0.33, 0.01, 0.51], "");
    publish_once_subscribed(topic.wait_for_subscriber(TIMEOUT), topic.publish(&held)).await?;
    canvas
        .wait_until(TIMEOUT, "the map's cubes", |drawn| {
            drawn
                .iter()
                .any(|call| matches!(call, Drawn::Cubes { path, .. } if path == MAP_PATH))
        })
        .await?;
    let Drawn::Cubes { centers, side, .. } = &canvas.at(MAP_PATH)[0] else {
        panic!("cubes")
    };
    assert_eq!(centers, &[[0.31, 0.01, 0.51], [0.33, 0.01, 0.51]]);
    assert_eq!(*side, 0.02);
    let waiting = map(6, vec![], "no depth frame has reached the map yet");
    publish_once_subscribed(topic.wait_for_subscriber(TIMEOUT), topic.publish(&waiting)).await?;
    // The text is the last call of a map's drawing: once it says why the map waits, the
    // cubes are cleared.
    let said = "The planner's map holds 0 occupied cubes of 0.02 m. It holds no depth frame: \
                no depth frame has reached the map yet.";
    canvas
        .wait_until(TIMEOUT, "the waiting map's text", |drawn| {
            drawn.iter().any(|call| {
                matches!(call, Drawn::Text { path, markdown } if path == MAP_STATE_PATH && markdown == said)
            })
        })
        .await?;
    assert!(matches!(
        &canvas.at(MAP_PATH)[..],
        [Drawn::Cubes { .. }, Drawn::Clear { .. }]
    ));
    harness.shutdown().await.map_err(|e| e.to_string())
}
