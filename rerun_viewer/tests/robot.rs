//! The robot's body over the wire: robot_identity's description becomes the collision
//! shapes of each link, and limb_state's snapshots place the links.

mod common;

use std::f64::consts::FRAC_PI_2;

use common::{
    TIMEOUT, bound, drew_at, mini_description, mini_snapshot, publish_once_subscribed, start,
};
use rerun_viewer::config::default_parameters;
use rerun_viewer::drawn::Drawn;
use rerun_viewer::model::scene::ShapeKind;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_description_draws_the_links_and_each_snapshot_places_them() -> Result<(), String> {
    let (harness, mocks, canvas) = start(bound(&["identity"], default_parameters())).await?;
    mocks
        .deps
        .identity
        .as_ref()
        .expect("the identity link is bound")
        .get_description
        .next_request(TIMEOUT)
        .await
        .map_err(|e| e.to_string())?
        .respond(mini_description())
        .await
        .map_err(|e| e.to_string())?;
    canvas
        .wait_until(TIMEOUT, "the finger's shape", |drawn| {
            drew_at(drawn, "world/robot/finger/collision_0_sphere")
        })
        .await?;
    assert!(matches!(
        &canvas.at("world/robot/upper/collision_0_cylinder")[..],
        [Drawn::StaticShape { shape, .. }]
            if shape.kind == (ShapeKind::Cylinder { height: 0.2, radius: 0.03 })
    ));
    // lift and reach turn a quarter each; the gripper is half open.
    let limb_states = &mocks.deps.limb_state.limb_states;
    publish_once_subscribed(
        limb_states.wait_for_subscriber(TIMEOUT),
        limb_states.publish(&mini_snapshot(FRAC_PI_2, FRAC_PI_2, 0.5)),
    )
    .await?;
    canvas
        .wait_until(TIMEOUT, "the finger placed", |drawn| {
            drew_at(drawn, "world/robot/finger")
        })
        .await?;
    let Drawn::Place { pose, .. } = &canvas.at("world/robot/finger")[0] else {
        panic!("the finger is placed")
    };
    let expected = [-0.02, 0.0, 0.2];
    assert!(
        pose.origin()
            .iter()
            .zip(expected)
            .all(|(got, want)| (got - want).abs() < 1e-12),
        "the finger at {:?}, not {expected:?}",
        pose.origin()
    );
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_description_that_does_not_read_draws_no_links() -> Result<(), String> {
    let (harness, mocks, canvas) = start(bound(&["identity"], default_parameters())).await?;
    let mut broken = mini_description();
    broken.meshes.clear();
    mocks
        .deps
        .identity
        .as_ref()
        .expect("the identity link is bound")
        .get_description
        .next_request(TIMEOUT)
        .await
        .map_err(|e| e.to_string())?
        .respond(broken)
        .await
        .map_err(|e| e.to_string())?;
    // The arms still draw: the snapshot's grasp point shows the node went on.
    let limb_states = &mocks.deps.limb_state.limb_states;
    publish_once_subscribed(
        limb_states.wait_for_subscriber(TIMEOUT),
        limb_states.publish(&mini_snapshot(0.0, 0.0, 0.5)),
    )
    .await?;
    canvas
        .wait_until(TIMEOUT, "the grasp point", |drawn| {
            drew_at(drawn, "world/arms/arm/grasp")
        })
        .await?;
    assert!(
        !canvas.drawn().iter().any(|drawn| drawn
            .path()
            .is_some_and(|path| path.starts_with("world/robot"))),
        "no link drawn"
    );
    harness.shutdown().await.map_err(|e| e.to_string())
}
