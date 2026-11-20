//! room_map:v1's refusals over the wire, each of the kind room_map gives it, on the rig of
//! `common`: with no map loaded; set_initial_pose's targets that no member takes; the
//! members refused while a set_initial_pose waits and while mapping runs; a base whose
//! odometry is commanded; a base whose first base_states does not arrive within the source
//! timeout; and a base silent for more than 0.5 s.

use peppygen::fixtures::emitted_topics::room_map::map_grids;
use peppygen::fixtures::exposed_services::room_map::{
    finish_mapping, load_map, reinitialize_global_localization, remove_map, set_initial_pose,
    start_mapping,
};

mod common;
use common::{Rig, WIRE, at, first_after, mount_answer};

/// The failure kind of `$answer`, an answer of the member `$member`, as its name, and its
/// message.
macro_rules! refused {
    (set_initial_pose, $answer:expr) => {{
        let failure = $answer.expect_err("set_initial_pose is refused");
        (format!("{:?}", failure.kind), failure.message)
    }};
    ($member:ident, $answer:expr) => {{
        let error = $answer.expect_err(concat!(stringify!($member), " is refused"));
        let failure = $member::Failure::of(&error).expect("an error reply");
        (format!("{:?}", failure.kind), failure.message)
    }};
}

fn named(map_name: &str) -> load_map::RequestData {
    load_map::RequestData {
        map_name: map_name.into(),
    }
}

fn removing(map_name: &str) -> remove_map::RequestData {
    remove_map::RequestData {
        map_name: map_name.into(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_no_map_loaded_the_members_that_need_one_are_refused() {
    let rig = Rig::start("no_map", "measured", "").await;
    rig.mocks
        .deps
        .transform_lookup
        .lookup_transform
        .enqueue_response(mount_answer())
        .expect("a lookup answer");
    rig.measure_base().await;
    let (kind, _) = refused!(
        finish_mapping,
        finish_mapping::poll(&rig.harness, WIRE).await
    );
    assert_eq!(kind, "PreconditionFailed");
    for (name, expected) in [("Lab", "Rejected"), ("lab", "Rejected")] {
        let (kind, message) = refused!(
            load_map,
            load_map::poll(&rig.harness, &named(name), WIRE).await
        );
        assert_eq!(kind, expected, "{name}: {message}");
    }
    let (kind, _) = refused!(
        reinitialize_global_localization,
        reinitialize_global_localization::poll(&rig.harness, WIRE).await
    );
    assert_eq!(kind, "PreconditionFailed");
    let (kind, message) = refused!(
        set_initial_pose,
        rig.set_initial_pose(&rig.target("map", 0.0, 0.3)).await
    );
    assert_eq!(kind, "PreconditionFailed", "{message}");
    // A map started and finished before its first scan holds no scan. Before that scan it
    // covers no point, so a pose in it lies outside the loaded map: rejected, before the
    // precondition of the mapping that runs.
    rig.start_mapping("empty").await.expect("mapping starts");
    let (kind, message) = refused!(
        set_initial_pose,
        rig.set_initial_pose(&rig.target("map", 0.0, 0.3)).await
    );
    assert_eq!(kind, "Rejected", "{message}");
    assert!(message.contains("has read no scan"), "{message}");
    let (kind, message) = refused!(
        finish_mapping,
        finish_mapping::poll(&rig.harness, WIRE).await
    );
    assert_eq!(kind, "AttemptFailed", "{message}");
    assert!(message.contains("holds no scan"), "{message}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn set_initial_pose_refuses_a_target_no_member_takes() {
    let rig = Rig::mapped_and_loaded("targets").await;
    let target = || rig.target("map", 0.0, 0.3);
    let half = 0.05_f64;
    let cases: [(&str, set_initial_pose::RequestData, &str); 5] = [
        (
            "off the floor",
            {
                let mut request = target();
                request.target.position[2] = 0.1;
                request
            },
            "[x, y, 0]",
        ),
        (
            "tilted",
            {
                let mut request = target();
                request.target.orientation = [half.sin(), 0.0, 0.0, half.cos()];
                request
            },
            "sin(yaw/2)",
        ),
        (
            "not a unit quaternion",
            {
                let mut request = target();
                request.target.orientation = [0.0, 0.0, 0.0, 2.0];
                request
            },
            "unit quaternion",
        ),
        (
            "outside the map",
            {
                let mut request = target();
                request.target.position[0] = 100.0;
                request
            },
            "outside the loaded map",
        ),
        (
            "no search",
            {
                let mut request = target();
                request.search_radius_m = 0.0;
                request
            },
            "search_radius_m",
        ),
    ];
    for (case, request, says) in cases {
        let failure = rig.set_initial_pose(&request).await.expect_err(case);
        assert_eq!(
            failure.kind,
            set_initial_pose::FailureKind::Rejected,
            "{case}: {failure:?}"
        );
        assert!(failure.message.contains(says), "{case}: {failure:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn while_a_set_initial_pose_waits_the_members_that_change_the_room_are_refused() {
    let rig = Rig::mapped_and_loaded("while_waiting").await;
    rig.measure_base().await;
    let target = rig.target("map", 0.1, 0.3);
    let pending = set_initial_pose::poll(&rig.harness, &target, WIRE);
    tokio::pin!(pending);
    tokio::select! {
        answer = &mut pending => panic!("answered before a scan was sent: {answer:?}"),
        _ = rig.until_state(|state| state.waiting) => {}
    }
    let refusals = [
        refused!(
            start_mapping,
            start_mapping::poll(
                &rig.harness,
                &start_mapping::RequestData {
                    map_name: "kitchen".into()
                },
                WIRE
            )
            .await
        ),
        refused!(
            load_map,
            load_map::poll(&rig.harness, &named("lab"), WIRE).await
        ),
        refused!(
            remove_map,
            remove_map::poll(&rig.harness, &removing("lab"), WIRE).await
        ),
        refused!(
            reinitialize_global_localization,
            reinitialize_global_localization::poll(&rig.harness, WIRE).await
        ),
    ];
    for (kind, message) in refusals {
        assert_eq!(kind, "PreconditionFailed", "{message}");
        assert!(message.contains("set_initial_pose waits"), "{message}");
    }
    rig.scan().await;
    pending.await.expect("the waiting call matches");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn while_mapping_runs_the_members_of_a_saved_map_are_refused() {
    let rig = Rig::mapped_and_loaded("while_mapping").await;
    rig.start_mapping("kitchen").await.expect("mapping starts");
    // A scan the mapping reads, so set_initial_pose's target lies inside the new map.
    rig.scan().await;
    rig.wait_for_scans(1).await;
    let refusals = [
        refused!(
            load_map,
            load_map::poll(&rig.harness, &named("lab"), WIRE).await
        ),
        refused!(
            reinitialize_global_localization,
            reinitialize_global_localization::poll(&rig.harness, WIRE).await
        ),
        refused!(
            set_initial_pose,
            rig.set_initial_pose(&rig.target("map", 0.0, 0.3)).await
        ),
    ];
    for (kind, message) in refusals {
        assert_eq!(kind, "PreconditionFailed", "{message}");
        assert!(message.contains("mapping runs"), "{message}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn against_a_base_whose_odometry_is_commanded_the_estimate_is_refused() {
    let rig = Rig::mapped_and_loaded("commanded_saved")
        .await
        .restart("commanded", "", 8)
        .await;
    rig.measure_base().await;
    load_map::poll(&rig.harness, &named("lab"), WIRE)
        .await
        .expect("the saved map loads");
    let refusals = [
        refused!(
            reinitialize_global_localization,
            reinitialize_global_localization::poll(&rig.harness, WIRE).await
        ),
        refused!(
            set_initial_pose,
            rig.set_initial_pose(&rig.target("map", 0.0, 0.3)).await
        ),
    ];
    for (kind, message) in refusals {
        assert_eq!(kind, "PreconditionFailed", "{message}");
        assert!(message.contains("commanded odometry"), "{message}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_silent_base_refuses_the_members_that_read_its_odometry() {
    let mut rig = Rig::mapped_and_loaded("silent_base").await;
    // More than 0.5 s of the clock since the newest base_states. The grid that a load
    // publishes is stamped at the node's clock, which is past the tick once that grid is.
    let target = rig.target("map", 0.0, 0.3);
    rig.tick(1.0).await;
    let ticked = at(rig.now_s.get() - 1e-6);
    load_map::poll(&rig.harness, &named("lab"), WIRE)
        .await
        .expect("the map loads again");
    first_after(
        async || {
            rig.harness
                .emitted
                .room_map_map_grids
                .next()
                .await
                .ok()
                .flatten()
        },
        |grid: &map_grids::Message| grid.timestamp,
        ticked,
    )
    .await;
    let (kind, message) = refused!(
        set_initial_pose,
        set_initial_pose::poll(&rig.harness, &target, WIRE)
            .await
            .map_err(|error| set_initial_pose::Failure::of(&error).expect("an error reply"))
    );
    assert_eq!(kind, "ResourceUnavailable", "{message}");
    assert!(message.contains("0.5 s"), "{message}");
    let (kind, message) = refused!(
        start_mapping,
        start_mapping::poll(
            &rig.harness,
            &start_mapping::RequestData {
                map_name: "kitchen".into()
            },
            WIRE
        )
        .await
    );
    assert_eq!(kind, "ResourceUnavailable", "{message}");
    assert!(message.contains("0.5 s"), "{message}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_waits_the_source_timeout_for_the_first_base_states_then_is_refused() {
    let rig = Rig::start("no_base_yet", "measured", "").await;
    let request = start_mapping::RequestData {
        map_name: "kitchen".into(),
    };
    let pending = start_mapping::poll(&rig.harness, &request, WIRE);
    tokio::pin!(pending);
    // No base_states arrives while the clock runs past the source timeout, 2 s, a tenth of
    // a second at a time.
    let answer = loop {
        tokio::select! {
            answer = &mut pending => break answer,
            () = rig.tick(0.1) => {}
        }
    };
    let (kind, message) = refused!(start_mapping, answer);
    assert_eq!(kind, "ResourceUnavailable", "{message}");
    assert!(
        message.contains("no base_states arrived within the source timeout (2 s)"),
        "{message}"
    );
}
