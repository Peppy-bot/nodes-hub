//! room_map:v1's estimate over the wire, on the rig of `common`, in a map of the room's
//! loop loaded: set_initial_pose matches near the base, refuses a target in odom, fails
//! `no_match` far from the base with the estimate unchanged, runs a second call once the
//! first has answered, and fails `resource_unavailable` when no scan arrives; get_map_state
//! reports what holds; a member that tests the odometry waits for describe_base after the
//! odometry starts over; map to odom places the base across that start over and starts over
//! at a load_map; a stamped target that the odometry does not hold fails `extrapolation`;
//! and a search localizes the base as it turns in place, and ends when no scan arrives.

use peppygen::fixtures::emitted_topics::room_map::map_poses;
use peppygen::fixtures::emitted_topics::transforms::transforms;
use peppygen::fixtures::exposed_services::room_map::{
    get_map_state, load_map, reinitialize_global_localization, set_initial_pose, start_mapping,
};

mod common;
use common::{HANG, Rig, WIRE, at, first_after};

fn kind(answer: Result<set_initial_pose::Response, set_initial_pose::Failure>) -> String {
    format!("{:?}", answer.expect_err("a refusal").kind)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn set_initial_pose_matches_near_the_base_and_refuses_a_target_in_odom() {
    let rig = Rig::mapped_and_loaded("match").await;
    assert_eq!(
        kind(rig.set_initial_pose(&rig.target("odom", 0.0, 0.3)).await),
        "Rejected"
    );
    // The map frame is the odometry frame where mapping started, so the base's true pose
    // in odom is its pose in the map.
    let matched = rig
        .set_initial_pose(&rig.target("map", 0.1, 0.3))
        .await
        .expect("the scan matches near the base");
    let [x, y, _] = rig.pose.get();
    let [mx, my, _] = matched.current_pose.position;
    assert!((mx - x).hypot(my - y) < 0.1, "{matched:?} at ({x}, {y})");
    assert_eq!(matched.current_pose.frame, "map");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_target_far_from_the_base_fails_no_match_and_the_estimate_stays() {
    let rig = Rig::mapped_and_loaded("no_match").await;
    rig.set_initial_pose(&rig.target("map", 0.1, 0.3))
        .await
        .expect("the scan matches near the base");
    assert_eq!(
        kind(rig.set_initial_pose(&rig.target("map", 2.0, 0.3)).await),
        "NoMatch"
    );
    let state = get_map_state::poll(&rig.harness, WIRE)
        .await
        .expect("a state");
    assert!(state.localized, "the estimate stays: {state:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_match_past_the_matching_limits_names_its_deviations_and_the_limits() {
    // No match's deviations lie within limits of a micrometre and a microradian.
    let rig = Rig::mapped_and_loaded("no_match_limits")
        .await
        .restart_tuned("measured", "", 8, |parameters| {
            parameters.matching_limits.position_std_m = 1e-6;
            parameters.matching_limits.orientation_std_rad = 1e-6;
        })
        .await;
    rig.mocks
        .deps
        .transform_lookup
        .lookup_transform
        .enqueue_response(common::mount_answer())
        .expect("a lookup answer");
    rig.measure_base().await;
    load_map::poll(
        &rig.harness,
        &load_map::RequestData {
            map_name: "lab".into(),
        },
        WIRE,
    )
    .await
    .expect("the saved map loads");
    let failure = rig
        .set_initial_pose(&rig.target("map", 0.1, 0.3))
        .await
        .expect_err("no match lies within the limits");
    assert_eq!(
        failure.kind,
        set_initial_pose::FailureKind::NoMatch,
        "{failure:?}"
    );
    assert!(
        failure
            .message
            .contains("past the matching's limits, 0.000001 m and 0.000001 rad"),
        "{failure:?}"
    );
    assert!(
        failure
            .message
            .contains("launch with larger matching_limits"),
        "{failure:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_set_initial_pose_waits_for_the_first_then_runs() {
    let rig = Rig::mapped_and_loaded("second_call").await;
    let (first, second) = (rig.target("map", 0.1, 0.3), rig.target("map", 2.0, 0.3));
    rig.measure_base().await;
    let answered = &std::cell::Cell::new(0);
    let harness = &rig.harness;
    let call = move |request| async move {
        let answer = set_initial_pose::poll(harness, request, WIRE).await;
        answered.set(answered.get() + 1);
        answer.map_err(|error| set_initial_pose::Failure::of(&error).expect("an error reply"))
    };
    let (first, second, ()) = tokio::join!(
        call(&first),
        // The second call is made while the first waits.
        async {
            rig.until_state(|state| state.waiting).await;
            call(&second).await
        },
        // Each scan is sent once a call waits and the calls before it answered.
        async {
            for answers in 0..2 {
                rig.until_state(|state| state.waiting).await;
                rig.scan().await;
                while answered.get() <= answers {
                    tokio::task::yield_now().await;
                }
            }
        }
    );
    first.expect("the first call matches");
    assert_eq!(
        kind(second),
        "NoMatch",
        "the second call ran after the first"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn set_initial_pose_fails_resource_unavailable_when_no_scan_arrives() {
    let rig = Rig::mapped_and_loaded("no_scan").await;
    let request = rig.target("map", 0.0, 0.3);
    let answered = std::cell::Cell::new(false);
    let (answer, ()) = tokio::join!(
        async {
            let answer = set_initial_pose::poll(&rig.harness, &request, WIRE).await;
            answered.set(true);
            answer
        },
        async {
            // Once the call waits, the base is heard and the lidar silent while the clock
            // passes the source timeout, 2 s.
            rig.until_state(|state| state.waiting).await;
            for _ in 0..30 {
                if answered.get() {
                    break;
                }
                rig.tick(0.1).await;
                rig.measure_base().await;
                tokio::task::yield_now().await;
            }
        }
    );
    let failure =
        set_initial_pose::Failure::of(&answer.expect_err("a refusal")).expect("an error reply");
    assert_eq!(
        failure.kind,
        set_initial_pose::FailureKind::ResourceUnavailable
    );
    let state = get_map_state::poll(&rig.harness, WIRE)
        .await
        .expect("a state");
    assert!(!state.waiting && !state.localized, "{state:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn get_map_state_reports_searching_waiting_and_localized_as_each_holds() {
    let rig = Rig::mapped_and_loaded("state").await;
    let state = || async {
        get_map_state::poll(&rig.harness, WIRE)
            .await
            .expect("a state")
    };
    let loaded = state().await;
    assert!(
        !loaded.mapping && !loaded.searching && !loaded.waiting && !loaded.localized,
        "{loaded:?}"
    );
    let request = rig.target("map", 0.1, 0.3);
    let pending = set_initial_pose::poll(&rig.harness, &request, WIRE);
    tokio::pin!(pending);
    let waiting = tokio::select! {
        answer = &mut pending => panic!("answered before a scan was sent: {answer:?}"),
        waiting = rig.until_state(|state| state.waiting) => waiting,
    };
    assert!(waiting.waiting && !waiting.localized, "{waiting:?}");
    rig.scan().await;
    pending.await.expect("the scan matches");
    let localized = state().await;
    assert!(localized.localized && !localized.waiting, "{localized:?}");
    reinitialize_global_localization::poll(&rig.harness, WIRE)
        .await
        .expect("a search starts");
    let searching = state().await;
    assert!(searching.searching && !searching.localized, "{searching:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn after_the_odometry_starts_over_a_member_waits_for_describe_base() {
    // describe_base answers the first start alone.
    let rig = Rig::start_with("new_start", "measured", "", 1).await;
    rig.mocks
        .deps
        .transform_lookup
        .lookup_transform
        .enqueue_response(common::mount_answer())
        .expect("a lookup answer");
    rig.start_mapping("lab").await.expect("mapping starts");
    rig.odometry_start_s.set(2_000.0);
    rig.scan().await;
    let refused = start_mapping::poll(
        &rig.harness,
        &start_mapping::RequestData {
            map_name: "kitchen".into(),
        },
        WIRE,
    )
    .await
    .expect_err("describe_base does not answer");
    let failure = start_mapping::Failure::of(&refused).expect("an error reply");
    assert_eq!(
        failure.kind,
        start_mapping::FailureKind::ResourceUnavailable
    );
    assert!(failure.message.contains("describe_base"), "{failure:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn map_to_odom_places_the_base_after_its_odometry_starts_over() {
    let mut rig = Rig::mapped_and_loaded("start_over").await;
    rig.set_initial_pose(&rig.target("map", 0.1, 0.3))
        .await
        .expect("the scan matches near the base");
    // The base's odometry starts over where it stands: base_states give odom's origin
    // while the base stays where it was in the room.
    let [x, y, yaw] = rig.pose.get();
    let restarted = common::at(rig.now_s.get());
    rig.odometry_start_s.set(2_000.0);
    rig.odometry_offset.set([x, y, yaw]);
    for _ in 0..3 {
        rig.tick(1.0).await;
        rig.scan().await;
    }
    let pose = first_after(
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
        restarted,
    )
    .await;
    let [px, py, _] = pose.position;
    assert!((px - x).hypot(py - y) < 0.1, "{pose:?} at ({x}, {y})");
    let placed = first_after(
        async || {
            rig.harness
                .emitted
                .transforms_transforms
                .next()
                .await
                .ok()
                .flatten()
        },
        |placed: &transforms::Message| placed.timestamp,
        restarted,
    )
    .await;
    let map_to_odom = &placed.transforms[0];
    assert_eq!(
        (map_to_odom.frame.as_str(), map_to_odom.child_frame.as_str()),
        ("map", "odom")
    );
    // odom's origin, where the base's odometry now places it, lies at the base in the map.
    let [ox, oy, _] = map_to_odom.position;
    assert!((ox - x).hypot(oy - y) < 0.1, "{placed:?} at ({x}, {y})");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stamped_target_that_the_odometry_does_not_hold_fails_extrapolation() {
    let rig = Rig::mapped_and_loaded("extrapolation").await;
    let mut request = rig.target("map", 0.1, 0.3);
    // The odometry keeps 10 s of base_states; the drive took 30 s.
    request.target.timestamp = Some(at(1.0));
    assert_eq!(kind(rig.set_initial_pose(&request).await), "Extrapolation");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn map_to_odom_starts_over_at_a_load_map() {
    let mut rig = Rig::mapped_and_loaded("transform_start").await;
    rig.set_initial_pose(&rig.target("map", 0.1, 0.3))
        .await
        .expect("the scan matches near the base");
    let before = first_after(
        async || {
            rig.harness
                .emitted
                .transforms_transforms
                .next()
                .await
                .ok()
                .flatten()
        },
        |placed: &transforms::Message| placed.timestamp,
        at(0.0),
    )
    .await;
    load_map::poll(
        &rig.harness,
        &load_map::RequestData {
            map_name: "lab".into(),
        },
        WIRE,
    )
    .await
    .expect("the map loads again");
    let loaded = at(rig.now_s.get());
    rig.set_initial_pose(&rig.target("map", 0.1, 0.3))
        .await
        .expect("the scan matches near the base");
    let after = first_after(
        async || {
            rig.harness
                .emitted
                .transforms_transforms
                .next()
                .await
                .ok()
                .flatten()
        },
        |placed: &transforms::Message| placed.timestamp,
        loaded,
    )
    .await;
    assert!(
        after.start_timestamp > before.start_timestamp,
        "{before:?} then {after:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_search_localizes_the_base_as_it_turns_in_place() {
    let mut rig = Rig::mapped_and_loaded("search").await;
    reinitialize_global_localization::poll(&rig.harness, WIRE)
        .await
        .expect("a search starts");
    let localized = async {
        loop {
            // Each turn passes AMCL's update_min_a, 0.2 rad, so the search updates, resamples
            // and answers each scan; the next turn waits for that answer, and the search
            // takes the same scans on every host.
            let [x, y, yaw] = rig.pose.get();
            rig.pose.set([x, y, yaw + 0.3]);
            rig.tick(1.0).await;
            rig.scan().await;
            let answer = first_after(
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
                // The scan is stamped 0.075 s before the clock's time after it.
                at(rig.now_s.get() - 0.08),
            )
            .await;
            if answer.localized {
                return;
            }
        }
    };
    tokio::time::timeout(HANG, localized)
        .await
        .expect("the search localizes the base before the hang guard");
    let state = get_map_state::poll(&rig.harness, WIRE)
        .await
        .expect("a state");
    assert!(state.localized && !state.searching, "{state:?}");
    let matched = rig
        .set_initial_pose(&rig.target("map", 0.0, 0.1))
        .await
        .expect("the localization holds the base where it stands");
    let [x, y, _] = rig.pose.get();
    let [mx, my, _] = matched.current_pose.position;
    assert!((mx - x).hypot(my - y) < 0.1, "{matched:?} at ({x}, {y})");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_search_ends_when_no_scan_arrives_within_the_source_timeout() {
    let rig = Rig::mapped_and_loaded("search_silent").await;
    reinitialize_global_localization::poll(&rig.harness, WIRE)
        .await
        .expect("a search starts");
    let searching = get_map_state::poll(&rig.harness, WIRE)
        .await
        .expect("a state");
    assert!(searching.searching, "{searching:?}");
    // The lidar falls silent past the source timeout, 2 s; the search then ends.
    for _ in 0..30 {
        rig.tick(0.1).await;
        rig.measure_base().await;
    }
    let ended = rig.until_state(|state| !state.searching).await;
    assert!(!ended.localized, "{ended:?}");
}
