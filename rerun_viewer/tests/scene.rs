//! The planning scene over the wire: each read that differs from the one before is
//! drawn, and its changes are logged.

mod common;

use common::{TIMEOUT, bound, drew_at, free_lime, held_lime, right_arm, scene, start};
use rerun_viewer::config::default_parameters;
use rerun_viewer::draw::scene::CHANGES_PATH;
use rerun_viewer::testing::Drawn;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lime_is_added_then_attached_to_the_right_arm() -> Result<(), String> {
    let (harness, mocks, canvas) = start(bound(&["motion_planning"], default_parameters())).await?;
    let planner = mocks
        .deps
        .motion_planning
        .as_ref()
        .expect("the planner is linked");
    planner
        .describe_arms
        .next_request(TIMEOUT)
        .await
        .map_err(|e| e.to_string())?
        .respond(right_arm())
        .await
        .map_err(|e| e.to_string())?;
    planner
        .get_planning_scene
        .next_request(TIMEOUT)
        .await
        .map_err(|e| e.to_string())?
        .respond(scene(vec![free_lime([0.4, -0.2, 0.1])], vec![]))
        .await
        .map_err(|e| e.to_string())?;
    canvas
        .wait_until(TIMEOUT, "the free lime", |drawn| {
            drew_at(drawn, "world/scene/shells/brain/lime_1/shell_0_sphere")
        })
        .await?;
    let Drawn::Shape { shape, look, .. } =
        &canvas.at("world/scene/shells/brain/lime_1/shell_0_sphere")[0]
    else {
        panic!("the shell is a shape")
    };
    assert!(look.wireframe);
    assert_eq!(shape.pose.origin(), [0.4, -0.2, 0.1]);
    planner
        .get_planning_scene
        .next_request(TIMEOUT)
        .await
        .map_err(|e| e.to_string())?
        .respond(scene(vec![], vec![held_lime()]))
        .await
        .map_err(|e| e.to_string())?;
    canvas
        .wait_until(TIMEOUT, "the attach", |drawn| {
            drawn.iter().any(|call| {
                matches!(call, Drawn::LogLine { line, .. }
                    if line == "attached brain/lime_1 to openarm_right_tcp")
            })
        })
        .await?;
    assert_eq!(
        canvas.log_lines(CHANGES_PATH),
        [
            "added brain/lime_1",
            "attached brain/lime_1 to openarm_right_tcp"
        ]
    );
    let Drawn::Shape { shape, .. } =
        &canvas.at("world/arms/right_arm/grasp/attached/brain/lime_1/shape_0_sphere")[0]
    else {
        panic!("the carried lime is a shape")
    };
    assert_eq!(
        shape.pose.origin(),
        [0.0, 0.0, 0.05],
        "in the grasp link's frame"
    );
    assert!(matches!(
        &canvas.at("world/scene/objects/brain/lime_1/shape_0_sphere")[..],
        [Drawn::Shape { .. }, Drawn::Clear { .. }]
    ));
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_read_that_changes_nothing_draws_nothing() -> Result<(), String> {
    let (harness, mocks, canvas) = start(bound(&["motion_planning"], default_parameters())).await?;
    let planner = mocks
        .deps
        .motion_planning
        .as_ref()
        .expect("the planner is linked");
    planner
        .describe_arms
        .next_request(TIMEOUT)
        .await
        .map_err(|e| e.to_string())?
        .respond(right_arm())
        .await
        .map_err(|e| e.to_string())?;
    let reads = [
        scene(vec![free_lime([0.4, -0.2, 0.1])], vec![]),
        scene(vec![free_lime([0.4, -0.2, 0.1])], vec![]),
        scene(vec![free_lime([0.4, -0.2, 0.2])], vec![]),
    ];
    for read in reads {
        planner
            .get_planning_scene
            .next_request(TIMEOUT)
            .await
            .map_err(|e| e.to_string())?
            .respond(read)
            .await
            .map_err(|e| e.to_string())?;
    }
    canvas
        .wait_until(TIMEOUT, "the move", |drawn| {
            drawn.iter().any(
                |call| matches!(call, Drawn::LogLine { line, .. } if line == "moved brain/lime_1"),
            )
        })
        .await?;
    assert_eq!(
        canvas.log_lines(CHANGES_PATH),
        ["added brain/lime_1", "moved brain/lime_1"]
    );
    assert_eq!(
        canvas.at("scene/allowed_pairs").len(),
        2,
        "the first read and the move are drawn, the read between them is not"
    );
    harness.shutdown().await.map_err(|e| e.to_string())
}
