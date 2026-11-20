//! room_map:v1's maps over the wire, on the rig of `common`: a map built while the base
//! drives is saved, listed and loaded; the saved maps list by name, and removing the loaded
//! map unloads it; map_grids carries the loaded map's grid; a ray that ends on the base's
//! own body marks nothing; start_mapping refuses in room_map's order; a base whose odometry
//! is commanded is refused; no scan is taken until the lidar's place on the base is known;
//! and a lidar that a part carries stops the node.

use peppygen::consumed_services::transform_lookup::lookup_transform;
use peppygen::fixtures::emitted_topics::room_map::map_grids;
use peppygen::fixtures::exposed_services::room_map::{
    finish_mapping, get_map_state, list_maps, load_map, remove_map, start_mapping,
};

mod common;
use common::{HANG, LOOP_SCANS, Rig, WIRE, at, distance_to_walls, first_after, mount_answer};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_room_mapped_while_the_base_drives_is_saved_listed_and_loaded() {
    let rig = Rig::start("round_trip", "measured", "").await;
    rig.mocks
        .deps
        .transform_lookup
        .lookup_transform
        .enqueue_response(mount_answer())
        .expect("a lookup answer");
    rig.start_mapping("lab").await.expect("mapping starts");
    let mapping = get_map_state::poll(&rig.harness, WIRE)
        .await
        .expect("a state");
    assert!(mapping.mapping && mapping.localized, "{mapping:?}");
    rig.drive_a_loop().await;
    let finished = finish_mapping::poll(&rig.harness, WIRE)
        .await
        .expect("the map is saved");
    assert_eq!(finished.map.map_name, "lab");
    assert!(finished.map.scans >= LOOP_SCANS, "{:?}", finished.map);
    let listed = list_maps::poll(&rig.harness, WIRE).await.expect("the maps");
    assert_eq!(
        listed
            .maps
            .iter()
            .map(|map| map.map_name.as_str())
            .collect::<Vec<_>>(),
        ["lab"]
    );
    let loaded = load_map::poll(
        &rig.harness,
        &load_map::RequestData {
            map_name: "lab".into(),
        },
        WIRE,
    )
    .await
    .expect("the map loads");
    assert_eq!(loaded.map.extent, finished.map.extent);
    let state = get_map_state::poll(&rig.harness, WIRE)
        .await
        .expect("a state");
    assert!(
        !state.mapping && !state.localized && state.map.is_some(),
        "{state:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn start_mapping_refuses_in_room_maps_order() {
    let rig = Rig::start("refusals", "measured", "").await;
    rig.mocks
        .deps
        .transform_lookup
        .lookup_transform
        .enqueue_response(mount_answer())
        .expect("a lookup answer");
    let kind = |result: Result<(), start_mapping::Failure>| result.expect_err("a refusal").kind;
    assert_eq!(
        kind(rig.start_mapping("Lab").await),
        start_mapping::FailureKind::Rejected,
        "Lab is no map name"
    );
    rig.start_mapping("lab").await.expect("mapping starts");
    assert_eq!(
        kind(rig.start_mapping("kitchen").await),
        start_mapping::FailureKind::PreconditionFailed,
        "mapping runs"
    );
    rig.drive_a_loop().await;
    finish_mapping::poll(&rig.harness, WIRE)
        .await
        .expect("the map is saved");
    assert_eq!(
        kind(rig.start_mapping("lab").await),
        start_mapping::FailureKind::Rejected,
        "lab is saved"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_base_whose_odometry_is_commanded_maps_nothing() {
    let rig = Rig::start("commanded", "commanded", "").await;
    rig.mocks
        .deps
        .transform_lookup
        .lookup_transform
        .enqueue_response(mount_answer())
        .expect("a lookup answer");
    let refused = rig.start_mapping("lab").await.expect_err("a refusal");
    assert_eq!(refused.kind, start_mapping::FailureKind::PreconditionFailed);
    assert!(
        refused.message.contains("commanded odometry"),
        "{refused:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_scan_is_taken_until_the_lidars_place_is_known() {
    let rig = Rig::start("lookup", "measured", "").await;
    rig.scan().await;
    let (_, first) = rig
        .mocks
        .deps
        .transform_lookup
        .lookup_transform
        .next_request(WIRE)
        .await
        .expect("a lookup of the lidar's frame");
    first
        .respond_failure(
            lookup_transform::FailureKind::UnknownFrame,
            "no transform names lidar_frame",
        )
        .await
        .expect("the failure is answered");
    rig.measure_base().await;
    let refused = start_mapping::poll(
        &rig.harness,
        &start_mapping::RequestData {
            map_name: "lab".into(),
        },
        WIRE,
    )
    .await
    .expect_err("the lidar's place is unknown");
    let failure = start_mapping::Failure::of(&refused).expect("an error reply");
    assert_eq!(
        failure.kind,
        start_mapping::FailureKind::ResourceUnavailable
    );
    assert!(failure.message.contains("unknown_frame"), "{failure:?}");
    rig.mocks
        .deps
        .transform_lookup
        .lookup_transform
        .enqueue_response(mount_answer())
        .expect("a lookup answer");
    rig.start_mapping("lab")
        .await
        .expect("mapping starts once the lookup answers");
}

/// The names `list_maps` gives.
async fn listed(rig: &Rig) -> Vec<String> {
    list_maps::poll(&rig.harness, WIRE)
        .await
        .expect("the maps")
        .maps
        .into_iter()
        .map(|map| map.map_name)
        .collect()
}

/// A copy of the saved map `from` saved as `to`.
fn copy_map(rig: &Rig, from: &str, to: &str) {
    let target = rig.maps.join(to);
    std::fs::create_dir_all(&target).expect("a map's directory");
    for entry in std::fs::read_dir(rig.maps.join(from)).expect("the saved map") {
        let entry = entry.expect("a file of the map");
        std::fs::copy(entry.path(), target.join(entry.file_name())).expect("a copied file");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_saved_maps_list_by_name_and_removing_the_loaded_map_unloads_it() {
    let rig = Rig::mapped_and_loaded("remove").await;
    copy_map(&rig, "lab", "zoo");
    copy_map(&rig, "lab", "attic");
    assert_eq!(listed(&rig).await, ["attic", "lab", "zoo"]);
    let remove = |name: &str| remove_map::RequestData {
        map_name: name.into(),
    };
    remove_map::poll(&rig.harness, &remove("lab"), WIRE)
        .await
        .expect("the loaded map is removed");
    let state = get_map_state::poll(&rig.harness, WIRE)
        .await
        .expect("a state");
    assert!(state.map.is_none() && !state.localized, "{state:?}");
    let refused = remove_map::poll(&rig.harness, &remove("lab"), WIRE)
        .await
        .expect_err("no map named lab is saved");
    assert_eq!(
        remove_map::Failure::of(&refused)
            .expect("an error reply")
            .kind,
        remove_map::FailureKind::Rejected
    );
    assert_eq!(listed(&rig).await, ["attic", "zoo"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn map_grids_carries_the_loaded_maps_grid() {
    let mut rig = Rig::mapped_and_loaded("grids").await;
    let state = get_map_state::poll(&rig.harness, WIRE)
        .await
        .expect("a state");
    let extent = state
        .map
        .and_then(|map| map.extent)
        .expect("the loaded map's extent");
    let grid = first_after(
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
        // The grids of the finished and the loaded map, stamped at the clock's time then;
        // the grids of the map being built come before.
        at(rig.now_s.get() - 0.01),
    )
    .await;
    assert_eq!(
        (grid.frame.as_str(), grid.map_name.as_str()),
        ("map", "lab")
    );
    assert_eq!(grid.extent, extent);
    assert_eq!(grid.cells.len(), (grid.columns * grid.rows) as usize);
    for state in [0, 100, 255] {
        assert!(grid.cells.contains(&state), "a cell of {state}");
    }
    assert!(grid.cells.iter().all(|cell| [0, 100, 255].contains(cell)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_ray_that_ends_on_the_bases_own_body_marks_nothing() {
    let mut rig = Rig::start("own_body", "measured", "").await;
    rig.mocks
        .deps
        .transform_lookup
        .lookup_transform
        .enqueue_response(mount_answer())
        .expect("a lookup answer");
    // A pole on the base, behind the lidar, ends the rays behind the base at 0.2 m.
    rig.own_body.set(Some([2.6, 3.1]));
    rig.start_mapping("lab").await.expect("mapping starts");
    rig.drive_a_loop().await;
    finish_mapping::poll(&rig.harness, WIRE)
        .await
        .expect("the map is saved");
    let grid = first_after(
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
        // The grids of the finished and the loaded map, stamped at the clock's time then;
        // the grids of the map being built come before.
        at(rig.now_s.get() - 0.01),
    )
    .await;
    let [x_min, _, y_min, _] = grid.extent;
    let away: Vec<[f64; 2]> = (0..grid.rows)
        .flat_map(|row| (0..grid.columns).map(move |column| (column, row)))
        .filter(|&(column, row)| grid.cells[(row * grid.columns + column) as usize] == 100)
        .map(|(column, row)| {
            [
                x_min + (f64::from(column) + 0.5) * grid.cell_size_m,
                y_min + (f64::from(row) + 0.5) * grid.cell_size_m,
            ]
        })
        .filter(|&center| distance_to_walls(center) > 0.15)
        .collect();
    assert!(
        away.is_empty(),
        "occupied cells away from every wall: {away:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lidar_that_a_part_carries_stops_the_node() {
    let rig = Rig::start("carried", "measured", "left_arm").await;
    let stopped = rig.harness.node_runner().cancellation_token().clone();
    // A scan is sent again until one reaches the node, as the lidar streams them.
    tokio::time::timeout(HANG, async {
        loop {
            rig.scan().await;
            tokio::select! {
                () = stopped.cancelled() => break,
                () = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
            }
        }
    })
    .await
    .expect("the fault stops the node");
    let fault = rig.fault.take().expect("the node ends with a fault");
    assert!(
        fault
            .to_string()
            .starts_with("the lidar lidar is carried by left_arm: room_mapper maps with a lidar"),
        "{fault}"
    );
}
