//! The checker against synthetic contracts: the rules that read the
//! contracts of a workflow, and the binding of its roles to robots.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value as Json, json};
use workflow_documents::run::{Attendance, Incidents};
use workflow_documents::{Name, RobotName};

use super::{RunChoices, check, plan};
use crate::contract::ContractShape;
use crate::contract::tests::TEST_ARM;
use crate::engine::plan::{BoundArg, BoundCall, CallSite, Plan, Watch};
use crate::fleet::{Instance, Robot, Snapshot};
use crate::frames::{Crossing, FrameOfValues, Placed, Stamp};
use crate::testing::{
    ARM_PIN, LINKS, allowed, name, one_arm, robot, robot_name, snapshot, with_beta, workflow,
};

fn grab_tree(item_id: Json) -> Json {
    json!({ "id": "pick", "sequence": [
        { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
        { "id": "grab", "timeout_s": 30, "call": {
            "member": "arm.grab",
            "args": { "item_id": item_id, "target": { "$ref": "seen.item_point" }, "speed": 1 },
        } },
    ] })
}

fn picker() -> Json {
    one_arm(
        "alpha",
        grab_tree(json!({ "$ref": "seen.item_id" })),
        json!([]),
    )
}

fn alpha() -> BTreeMap<Name, RobotName> {
    BTreeMap::from([(name("alpha"), robot_name("alpha"))])
}

/// The choices of a run of `attendance` with no monitor.
fn choices(attendance: Attendance) -> RunChoices<'static> {
    RunChoices {
        attendance,
        confirmed: Default::default(),
        incidents: &Incidents::None,
        poll: workflow_documents::Bound::from_seconds(0.5).unwrap(),
        cancel_timeout: Duration::from_secs(10),
        watch: Watch::default(),
        of_a_trial: false,
    }
}

/// The bound call of the step `step` of `role`.
fn call_of<'p>(planned: &'p Plan, role: &str, step: &str) -> &'p BoundCall {
    &planned.calls[&CallSite {
        role: name(role),
        step: workflow_documents::StepId::parse(step).unwrap(),
    }]
}

/// The errors of a check, each as `<at>: <message>`.
fn errors(
    value: &Json,
    snapshot: &Snapshot,
    binding: Option<&BTreeMap<Name, RobotName>>,
) -> Vec<String> {
    check(&workflow(value), snapshot, &allowed(), binding)
        .iter()
        .map(ToString::to_string)
        .collect()
}

fn refuses(found: &[String], part: &str) {
    assert!(
        found.iter().any(|error| error.contains(part)),
        "no error holds {part:?}: {found:#?}"
    );
}

/// The errors of the plan of a run of `attendance` on `robots`.
fn plan_errors(value: &Json, robots: Vec<Robot>, attendance: Attendance) -> Vec<String> {
    plan_errors_watching(value, robots, attendance, Watch::default())
}

/// The errors of the plan of a run of `attendance` on `robots` that reads
/// the watch parameters of `watch`.
fn plan_errors_watching(
    value: &Json,
    robots: Vec<Robot>,
    attendance: Attendance,
    watch: Watch,
) -> Vec<String> {
    match plan(
        Arc::new(workflow(value)),
        &snapshot(robots),
        &allowed(),
        &alpha(),
        RunChoices {
            watch,
            ..choices(attendance)
        },
    ) {
        Ok(_) => Vec::new(),
        Err(errors) => errors.iter().map(ToString::to_string).collect(),
    }
}

#[test]
fn a_workflow_that_holds_passes_with_and_without_roles() {
    let fleet = snapshot(vec![robot("alpha")]);
    assert_eq!(errors(&picker(), &fleet, None), Vec::<String>::new());
    assert_eq!(
        errors(&picker(), &fleet, Some(&alpha())),
        Vec::<String>::new()
    );
}

#[test]
fn an_item_id_takes_only_a_ref() {
    let literal = one_arm("alpha", grab_tree(json!("b1")), json!([]));
    refuses(
        &errors(&literal, &snapshot(vec![robot("alpha")]), None),
        "alpha.grab: `item_id` holds an item id; give it a `$ref`",
    );
}

/// A literal pose in `frame`, at the latest time, with no orientation.
fn literal_pose(frame: &str) -> Json {
    json!({ "frame": frame, "position": [1.2, 0.4, 0.0] })
}

/// The picker with the target of its grab given as `target`.
fn grabbing_at(target: Json) -> Json {
    let mut value = picker();
    value["actions"]["alpha"]["tree"]["sequence"][1]["call"]["args"]["target"] = target;
    value
}

/// A literal pose in the robot frame by its name, or in a frame that is no link of the robot, such as map or odom,
/// is taken; one in a link of the robot other than its robot frame is refused, with and without roles.
#[test]
fn a_literal_pose_is_taken_in_the_robot_frame_or_map_and_refused_in_another_link_of_the_robot() {
    let fleet = snapshot(vec![robot("alpha")]);
    for frame in [LINKS[0], "map", "odom", "world"] {
        let taken = grabbing_at(literal_pose(frame));
        assert_eq!(
            errors(&taken, &fleet, None),
            Vec::<String>::new(),
            "{frame}"
        );
        assert_eq!(
            errors(&taken, &fleet, Some(&alpha())),
            Vec::<String>::new(),
            "{frame}"
        );
    }
    for link in &LINKS[1..] {
        let refused = grabbing_at(literal_pose(link));
        for binding in [None, Some(&alpha())] {
            refuses(
                &errors(&refused, &fleet, binding),
                &format!(
                    "alpha.grab: a literal in `{link}`, a link of `alpha` other than its robot frame `base_link`: a literal names the robot frame, by its name or by leaving `frame` out, or a frame that the robot's tree holds above its robot frame, odom, map, or world where a node publishes it, and no node publishes one into a robot's tree: pass the record from a report of a skill of the robot by `$ref`, which brings its frame and its timestamp"
                ),
            );
        }
    }
}

/// For a robot whose description gives no links, the director cannot tell its robot frame from its other links and
/// from map: a literal that names a frame, its robot frame by name among them, is refused saying to leave the frame
/// out or to pass the record by `$ref`, and a `$ref` is taken.
#[test]
fn a_literal_pose_needs_the_links_of_the_robot() {
    let mut fleet = snapshot(vec![robot("alpha")]);
    fleet.links.clear();
    for frame in ["map", LINKS[0]] {
        refuses(
            &errors(&grabbing_at(literal_pose(frame)), &fleet, Some(&alpha())),
            &format!(
                "`alpha` gives no URDF through robot_identity:v1, so the director cannot tell whether `{frame}` is its robot frame, another link of it or a frame its tree holds above its robot frame: leave `frame` out for the robot frame, or pass the record from a report of a skill of the robot by `$ref`, which brings its frame and its timestamp"
            ),
        );
    }
    assert_eq!(
        errors(&picker(), &fleet, Some(&alpha())),
        Vec::<String>::new()
    );
}

/// A literal in an empty frame names no frame: it is refused, with and without roles, saying what to write.
#[test]
fn a_literal_in_an_empty_frame_is_refused() {
    let fleet = snapshot(vec![robot("alpha")]);
    for binding in [None, Some(&alpha())] {
        assert_eq!(
            errors(&grabbing_at(literal_pose("")), &fleet, binding),
            [
                "alpha.grab: a literal in an empty frame, which names no frame: a literal names the robot frame, by its name or by leaving `frame` out, or a frame that the robot's tree holds above its robot frame, odom, map, or world where a node publishes it, and no node publishes one into a robot's tree: pass the record from a report of a skill of the robot by `$ref`, which brings its frame and its timestamp"
            ]
        );
    }
}

/// A literal that a lane group gives as a parameter is a literal: in a link of the robot other than its robot frame,
/// it is refused.
#[test]
fn a_pose_given_as_a_parameter_is_a_literal() {
    let fleet = snapshot(vec![robot("alpha")]);
    let by_param = |spot: Json| {
        let mut value = picker();
        let tree = value["actions"]["alpha"]["tree"].clone();
        value["actions"] = json!({ "pickers": {
            "roles": ["alpha"],
            "params": { "alpha": { "spot": spot } },
            "tree": tree,
        } });
        value["actions"]["pickers"]["tree"]["sequence"][1]["call"]["args"]["target"] =
            json!({ "$param": "spot" });
        value
    };
    assert_eq!(
        errors(&by_param(literal_pose("odom")), &fleet, None),
        Vec::<String>::new()
    );
    refuses(
        &errors(&by_param(literal_pose(LINKS[2])), &fleet, None),
        "a literal in `grasp_link`, a link of `alpha` other than its robot frame `base_link`",
    );
}

/// A record that names its frame goes whole: a part of it read by `$ref` or from a table, beside a literal frame,
/// is refused.
#[test]
fn a_pose_takes_no_part_by_ref_or_by_table() {
    let fleet = snapshot(vec![robot("alpha")]);
    let mut by_ref = literal_pose("map");
    by_ref["position"] = json!({ "$ref": "seen.item_point.position" });
    refuses(
        &errors(&grabbing_at(by_ref), &fleet, None),
        "alpha.grab: `position` is a part of a record that names its frame",
    );
    let mut by_frame = literal_pose("map");
    by_frame["frame"] = json!({ "$ref": "seen.item_point.frame" });
    refuses(
        &errors(&grabbing_at(by_frame), &fleet, None),
        "alpha.grab: `frame` is a part of a record that names its frame",
    );
    let mut by_table = grabbing_at(literal_pose("map"));
    by_table["tables"] = json!({ "spots": {
        "block": [0.3, 0.0, 0.1], "cup": [0.3, 0.1, 0.1], "unknown": [0.3, 0.2, 0.1],
    } });
    by_table["actions"]["alpha"]["tree"]["sequence"][1]["call"]["args"]["target"]["position"] =
        json!({ "$table": "spots", "key": { "$ref": "seen.kind" } });
    refuses(
        &errors(&by_table, &fleet, None),
        "`position` is a part of a record that names its frame",
    );
}

/// A table row holds no record: a pose row is refused in map and in a link of the robot alike, so no row gives a
/// call a frame.
#[test]
fn a_table_row_holds_no_pose() {
    let fleet = snapshot(vec![robot("alpha")]);
    for frame in ["map", LINKS[0]] {
        let mut value = grabbing_at(json!({ "$table": "spots", "key": { "$ref": "seen.kind" } }));
        value["tables"] = json!({ "spots": {
            "block": literal_pose(frame), "cup": literal_pose(frame), "unknown": literal_pose(frame),
        } });
        let found = errors(&value, &fleet, Some(&alpha()));
        refuses(
            &found,
            "tables.spots.cup: a row is a bool, a number, a string or a list of them",
        );
        refuses(
            &found,
            "alpha.grab: `$table` names `spots`, which has no rows of one type",
        );
    }
}

/// A workflow whose lane looks at each item of a scan and goes to `place`, with the table `places` and the
/// locations `bin` and `shelf`.
fn sorting(places: Json, place: Json) -> Json {
    let tree = json!({ "id": "sort", "sequence": [
        { "id": "scan", "timeout_s": 5, "call": { "member": "eyes.scan", "args": { "description": "blocks" }, "as": "scanned" } },
        { "id": "each", "for_each": { "list": "scanned.item_ids", "as": "item" }, "max_items": 5, "body": { "id": "one", "sequence": [
            { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
            { "id": "go", "timeout_s": 5, "call": { "member": "arm.go_to", "args": { "place": place } } },
        ] } },
    ] });
    let mut value = one_arm("alpha", tree, json!([]));
    value["locations"] = json!(["bin", "shelf"]);
    value["tables"] = json!({ "places": places });
    value
}

/// A row of `places` for each kind of a find.
fn places() -> Json {
    json!({ "block": { "$location": "bin" }, "cup": { "$location": "shelf" }, "unknown": { "$location": "bin" } })
}

/// A table whose every row is a `{ $location }` gives a field with the mark location the name of the row its key
/// selects, in a loop over a scan: the skill gets the name, and the run checks that the skill knows each name of
/// the table.
#[test]
fn a_location_table_gives_a_marked_field_the_location_its_key_selects() {
    let fleet = snapshot(vec![robot("alpha")]);
    let by_kind = json!({ "$table": "places", "key": { "$ref": "seen.kind" } });
    let value = sorting(places(), by_kind);
    assert_eq!(errors(&value, &fleet, Some(&alpha())), Vec::<String>::new());
    let planned = planned_on_alpha(&value);
    assert_eq!(
        call_of(&planned, "alpha", "go").args["place"],
        BoundArg::Table {
            rows: [("block", "bin"), ("cup", "shelf"), ("unknown", "bin")]
                .map(|(key, location)| (key.to_owned(), json!(location)))
                .into(),
            key: Box::new(BoundArg::Ref(path("seen.kind"))),
        }
    );
    assert_eq!(planned.locations.len(), 1);
    assert_eq!(
        planned.locations[0].names,
        [name("bin"), name("shelf")].into()
    );
}

/// A `{ $location }` row beside a row of another kind, a table of locations for a field without the mark location,
/// a table of literals for a field with it, and an expression that reads a table of locations are refused.
#[test]
fn a_location_table_holds_locations_alone_and_goes_to_a_marked_field_alone() {
    let fleet = snapshot(vec![robot("alpha")]);
    let by_kind = json!({ "$table": "places", "key": { "$ref": "seen.kind" } });
    let mut mixed = places();
    mixed["cup"] = json!("shelf");
    refuses(
        &errors(&sorting(mixed, by_kind.clone()), &fleet, Some(&alpha())),
        "tables.places.block: a row is a bool, a number, a string or a list of them, or `{ $location: \"<name>\" }` in a table whose every row is one",
    );
    let mut unmarked = sorting(places(), by_kind.clone());
    unmarked["actions"]["alpha"]["tree"]["sequence"][1]["body"]["sequence"][1]["call"] =
        json!({ "member": "arm.find", "args": { "description": by_kind.clone() } });
    refuses(
        &errors(&unmarked, &fleet, Some(&alpha())),
        "alpha.go: `$table` names `places`, a table of locations, and `description` takes no location",
    );
    let names = json!({ "block": "bin", "cup": "shelf", "unknown": "bin" });
    refuses(
        &errors(&sorting(names, by_kind.clone()), &fleet, Some(&alpha())),
        "alpha.go: `place` takes a location, and `$table` names `places`, whose rows are not `{ $location }`s",
    );
    let mut read = sorting(places(), by_kind);
    read["success"] = json!("places.block == 'bin'");
    let found = errors(&read, &fleet, Some(&alpha()));
    assert!(
        found.iter().any(|error| error.starts_with("success:")),
        "{found:#?}"
    );
}

/// The `signal` field of a `wait_for` result is an enum of the signals it waits for: a comparison with another
/// literal is refused.
#[test]
fn the_signal_a_wait_for_heard_is_an_enum_of_its_signals() {
    let fleet = snapshot(vec![robot("alpha"), robot("beta")]);
    let heard = |literal: &str| {
        json!({
            "workflow_schema": "workflow/v1",
            "manifest": { "name": "relay", "revision": 1 },
            "task": { "objective": "One arm rests once the other tells it which way" },
            "contracts": { "arm": "test_arm:v1" },
            "roles": { "alpha": {}, "beta": {} },
            "success": "true",
            "actions": {
                "alpha": { "tree": { "id": "first", "sequence": [
                    { "id": "hear", "wait_for": ["left", "right"], "as": "heard", "timeout_s": 5 },
                    { "id": "which", "if": format!("heard.signal == '{literal}'"), "then":
                        { "id": "rest", "timeout_s": 10, "call": { "member": "arm.home", "args": { "duration_s": 1 } } } },
                ] } },
                "beta": { "tree": { "id": "tell", "sequence": [
                    { "id": "go_left", "signal": "left" },
                    { "id": "go_right", "signal": "right" },
                ] } },
            },
        })
    };
    assert_eq!(errors(&heard("right"), &fleet, None), Vec::<String>::new());
    refuses(
        &errors(&heard("up"), &fleet, None),
        "'up' is not a value of",
    );
}

#[test]
fn an_item_id_is_valid_only_on_the_instance_that_gives_it() {
    // The robot serves perception on one instance and the arm on another.
    let mut split = robot("alpha");
    split.instances.push(Instance {
        name: "eyes".to_owned(),
        instance: "alpha_eyes".to_owned(),
        ..split.instances[0].clone()
    });
    split.contracts[1].instances = vec!["eyes".to_owned()];
    let tree = json!({ "id": "pick", "sequence": [
        { "id": "scan", "timeout_s": 5, "call": { "member": "eyes.scan", "args": { "description": "blocks" }, "as": "scanned" } },
        { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
        { "id": "grab", "timeout_s": 30, "call": {
            "member": "arm.grab",
            "args": { "item_id": { "$ref": "scanned.item_ids[0]" }, "target": { "$ref": "seen.item_point" }, "speed": 1 },
        } },
    ] });
    let value = one_arm("alpha", tree, json!([]));
    refuses(
        &errors(&value, &snapshot(vec![split]), Some(&alpha())),
        "an item id is valid only on the instance that gives it",
    );
}

/// The plan of `value` with the roles `alpha` and `beta` bound to the robots of their names.
fn planned_on_alpha_and_beta(value: &Json) -> Plan {
    super::plan(
        Arc::new(workflow(value)),
        &snapshot(vec![robot("alpha"), robot("beta")]),
        &allowed(),
        &BTreeMap::from([
            (name("alpha"), robot_name("alpha")),
            (name("beta"), robot_name("beta")),
        ]),
        choices(Attendance::Attended),
    )
    .unwrap_or_else(|errors| panic!("{errors:#?}"))
}

/// The roles that a value of `alpha` passes between to a call of `beta`.
fn alpha_to_beta() -> Option<Crossing> {
    Some(Crossing {
        from: name("alpha"),
        to: name("beta"),
    })
}

/// The observation `target` reads the arm of `alpha`, so its point is in a frame of that robot unless it is
/// world: the check holds, and the call of `beta` checks the frame of the point it reads in the run.
#[test]
fn a_point_of_another_role_is_bound_with_the_roles_it_passes_between() {
    let value = with_beta(
        picker(),
        json!({ "id": "reach", "timeout_s": 5, "call": {
            "member": "arm.check_reach", "args": { "points": [{ "$ref": "target.item_point" }] },
        } }),
    );
    let fleet = snapshot(vec![robot("alpha"), robot("beta")]);
    assert_eq!(errors(&value, &fleet, None), Vec::<String>::new());
    let planned = planned_on_alpha_and_beta(&value);
    let reach = call_of(&planned, "beta", "reach");
    let BoundArg::List(points) = &reach.args["points"] else {
        panic!("{:?}", reach.args);
    };
    let [BoundArg::Placed(point)] = points.as_slice() else {
        panic!("{points:?}");
    };
    assert_eq!(point.crossing, alpha_to_beta());
}

/// A request in one frame of `beta` takes the point that the observation `target` reads from the arm of `alpha`:
/// the check holds, and the frame of the request goes with the roles the point passes between, so the run checks
/// that it is world.
#[test]
fn a_request_in_one_frame_takes_a_point_of_another_role_with_the_roles_it_passes_between() {
    let value = with_beta(
        picker(),
        json!({ "id": "near", "timeout_s": 5, "call": {
            "member": "arm.check_positions", "args": { "points": [{ "$ref": "target.item_point" }] },
        } }),
    );
    let fleet = snapshot(vec![robot("alpha"), robot("beta")]);
    assert_eq!(errors(&value, &fleet, None), Vec::<String>::new());
    let planned = planned_on_alpha_and_beta(&value);
    assert_eq!(
        call_of(&planned, "beta", "near").args["frame"],
        BoundArg::Frame(FrameOfValues {
            frame: path("target.item_point.frame"),
            values: path("target.item_point"),
            crossing: alpha_to_beta(),
        })
    );
}

/// A `$ref` of `beta` to a record of `alpha` that holds a point is refused, saying to give the record field by
/// field; given so, its point goes with the roles it passes between.
#[test]
fn a_record_of_another_role_that_holds_a_point_is_given_field_by_field() {
    let fleet = snapshot(vec![robot("alpha"), robot("beta")]);
    let following = |found: Json| {
        with_beta(
            picker(),
            json!({ "id": "follow", "timeout_s": 5, "call": { "member": "eyes.follow", "args": { "found": found } } }),
        )
    };
    assert_eq!(
        errors(&following(json!({ "$ref": "target" })), &fleet, None),
        [
            "beta.follow: `found` holds points, poses or twists, and `target` comes from the role `alpha`: a point, a pose or a twist passes between robots in world alone, a frame their trees hold above their robot frames where a node publishes it, and no node publishes it into a robot's tree; give `found` field by field, with a `$ref` of its own to each point, pose or twist, which the run sends in world alone"
        ]
    );
    let by_field = following(json!({ "item_point": { "$ref": "target.item_point" } }));
    assert_eq!(errors(&by_field, &fleet, None), Vec::<String>::new());
    let planned = planned_on_alpha_and_beta(&by_field);
    let BoundArg::Object(found) = &call_of(&planned, "beta", "follow").args["found"] else {
        panic!("{planned:?}");
    };
    let BoundArg::Placed(point) = &found["item_point"] else {
        panic!("{found:?}");
    };
    assert_eq!(point.crossing, alpha_to_beta());
}

#[test]
fn a_lane_that_reaches_for_an_item_needs_a_reach_assumption() {
    let mut value = picker();
    value.as_object_mut().unwrap().remove("assumptions");
    refuses(
        &errors(&value, &snapshot(vec![robot("alpha")]), None),
        "alpha.grab: `arm.grab` reaches for an item",
    );
}

/// The error of a grab that no reach assumption and no reach check covers.
const UNCHECKED_GRAB: &str = "alpha.grab: `arm.grab` reaches for an item; give the workflow a `kind: \"reach\"` assumption whose check reads an observation of `alpha`, or put a reach check before it on every path";

/// The lane of `picker` with `checks` between the find and the grab.
fn picker_checking(checks: &[Json]) -> Json {
    let mut value = picker();
    value.as_object_mut().unwrap().remove("assumptions");
    let steps = value["actions"]["alpha"]["tree"]["sequence"]
        .as_array_mut()
        .unwrap();
    for (at, check) in checks.iter().enumerate() {
        steps.insert(1 + at, check.clone());
    }
    value
}

/// The find of a second item, a cup, as `other`.
fn find_other() -> Json {
    json!({ "id": "look_other", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "cup" }, "as": "other" } })
}

/// A call of `arm.check_reach` on the point of `of`, as `reach`, which
/// ends when `done_when` holds.
fn reach_check(of: &str, done_when: Option<&str>) -> Json {
    let mut call = json!({
        "member": "arm.check_reach", "args": { "points": [{ "$ref": format!("{of}.item_point") }] }, "as": "reach",
    });
    if let Some(done_when) = done_when {
        call["done_when"] = json!(done_when);
    }
    json!({ "id": "reach", "timeout_s": 5, "call": call })
}

/// A reach check of the point of the item that the grab reaches, named
/// before the grab, whose `done_when` is the field with the mark
/// `checks_reach` of its result, alone or as a term of `&&`.
#[test]
fn a_lane_that_checks_the_reach_of_the_item_before_it_reaches_needs_no_reach_assumption() {
    for done_when in [
        "reach.reachable",
        "reach.reachable && reach.margin > 0.01",
        "seen.found && (reach.margin > 0.01 && reach.reachable)",
    ] {
        let checked = picker_checking(&[reach_check("seen", Some(done_when))]);
        assert_eq!(
            errors(&checked, &snapshot(vec![robot("alpha")]), Some(&alpha())),
            Vec::<String>::new(),
            "{done_when}"
        );
    }
}

/// A check counts only from a member whose result has a field with the mark
/// `checks_reach`, with a `done_when` that is false whenever that field is
/// false, named before the reaching call on every path.
#[test]
fn a_reach_check_counts_only_when_it_gates_the_reach_on_every_path() {
    let unmarked = picker_checking(&[json!({ "id": "reach", "timeout_s": 5, "call": {
        "member": "arm.find", "args": { "description": "block" }, "as": "reach", "done_when": "reach.found",
    } })]);
    let in_one_branch = picker_checking(&[json!({
        "id": "maybe", "if": "seen.found", "then": reach_check("seen", Some("reach.reachable")),
    })]);
    let mut after = picker();
    after.as_object_mut().unwrap().remove("assumptions");
    after["actions"]["alpha"]["tree"]["sequence"]
        .as_array_mut()
        .unwrap()
        .push(reach_check("seen", Some("reach.reachable")));
    let mut cases = vec![
        ("an unmarked member", unmarked),
        (
            "no done_when",
            picker_checking(&[reach_check("seen", None)]),
        ),
        ("in one branch", in_one_branch),
        ("after the grab", after),
    ];
    for done_when in [
        "seen.found",
        "reach.margin > 0.01",
        "!reach.reachable",
        "reach.reachable || seen.found",
        "reach.reachable == true",
    ] {
        cases.push((
            done_when,
            picker_checking(&[reach_check("seen", Some(done_when))]),
        ));
    }
    for (case, value) in cases {
        let found = errors(&value, &snapshot(vec![robot("alpha")]), Some(&alpha()));
        assert!(
            found.iter().any(|error| error.contains(UNCHECKED_GRAB)),
            "{case}: {found:#?}"
        );
    }
}

/// A reach check counts only for the item whose point it reads: a check
/// of another item before the grab leaves the grab unchecked, and the error
/// names the source of the item that the grab reaches.
#[test]
fn a_reach_check_covers_only_the_item_whose_position_it_reads() {
    let fleet = snapshot(vec![robot("alpha")]);
    let of_another =
        picker_checking(&[find_other(), reach_check("other", Some("reach.reachable"))]);
    let found = errors(&of_another, &fleet, Some(&alpha()));
    assert_eq!(
        found,
        [
            "alpha.grab: `arm.grab` reaches for an item of `seen`, and no reach check before it on every path reads a position of it; give a reach check the position of each item that the call reaches, read from the same result or observation"
        ],
    );

    let mut both = picker_checking(&[find_other(), reach_check("other", Some("reach.reachable"))]);
    let mut second = reach_check("seen", Some("near.reachable"));
    second["id"] = json!("near");
    second["call"]["as"] = json!("near");
    both["actions"]["alpha"]["tree"]["sequence"]
        .as_array_mut()
        .unwrap()
        .insert(3, second);
    assert_eq!(errors(&both, &fleet, Some(&alpha())), Vec::<String>::new());
}

/// A loop element takes the source of the list of its loop: a grab of each
/// item of a scan is checked by a reach check of the points of that scan,
/// and not by a check of another item.
#[test]
fn a_loop_element_is_checked_by_a_reach_check_of_the_list_of_its_loop() {
    let fleet = snapshot(vec![robot("alpha")]);
    let each_scanned = |checked: Json| {
        let tree = json!({ "id": "pick", "sequence": [
            { "id": "scan", "timeout_s": 5, "call": { "member": "eyes.scan", "args": { "description": "blocks" }, "as": "scanned" } },
            { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
            { "id": "reach", "timeout_s": 5, "call": {
                "member": "arm.check_reach", "args": { "points": checked }, "as": "reach", "done_when": "reach.reachable",
            } },
            { "id": "each", "for_each": { "list": "scanned.points", "as": "point" }, "max_items": 5, "body": {
                "id": "grab", "timeout_s": 30, "call": {
                    "member": "arm.grab",
                    "args": { "item_id": { "$ref": "scanned.item_ids[0]" }, "target": { "$ref": "point" }, "speed": 1 },
                },
            } },
        ] });
        let mut value = one_arm("alpha", tree, json!([]));
        value.as_object_mut().unwrap().remove("assumptions");
        value
    };
    assert_eq!(
        errors(
            &each_scanned(json!({ "$ref": "scanned.points" })),
            &fleet,
            Some(&alpha())
        ),
        Vec::<String>::new()
    );
    refuses(
        &errors(
            &each_scanned(json!([{ "$ref": "seen.item_point" }])),
            &fleet,
            Some(&alpha()),
        ),
        "alpha.grab: `arm.grab` reaches for an item of `scanned`, and no reach check before it",
    );
}

/// The picker of scanned items whose reach check gives `args` to check_positions, a request that names its
/// frame at its level, and whose grab takes the first item and point of the scan `scanned`. A second scan, `again`,
/// gives values of another record.
fn checking_positions(args: Json) -> Json {
    let tree = json!({ "id": "pick", "sequence": [
        { "id": "scan", "timeout_s": 5, "call": { "member": "eyes.scan", "args": { "description": "blocks" }, "as": "scanned" } },
        { "id": "again", "timeout_s": 5, "call": { "member": "eyes.scan", "args": { "description": "blocks" }, "as": "again" } },
        { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
        { "id": "near", "timeout_s": 5, "call": {
            "member": "arm.check_positions", "args": args, "as": "near", "done_when": "near.all_workable",
        } },
        { "id": "grab", "timeout_s": 30, "call": {
            "member": "arm.grab",
            "args": { "item_id": { "$ref": "scanned.item_ids[0]" }, "target": { "$ref": "scanned.points[0]" }, "speed": 1 },
        } },
    ] });
    let mut value = one_arm("alpha", tree, json!([]));
    value.as_object_mut().unwrap().remove("assumptions");
    value
}

fn path(text: &str) -> workflow_documents::expression::RefPath {
    workflow_documents::expression::RefPath::parse(text).unwrap()
}

/// A request that names its frame at its level takes its points by `$ref` alone: the record they come from gives
/// the request its frame and its timestamp, each point goes with the fields the request takes, and the call checks
/// the reach of the grab of an item of that record.
#[test]
fn a_request_in_one_frame_takes_the_frame_and_the_time_of_the_record_of_its_values() {
    let fleet = snapshot(vec![robot("alpha")]);
    let by_ref = checking_positions(json!({ "points": { "$ref": "scanned.points" } }));
    assert_eq!(
        errors(&by_ref, &fleet, Some(&alpha())),
        Vec::<String>::new()
    );
    let planned = planned_on_alpha(&by_ref);
    let args = &call_of(&planned, "alpha", "near").args;
    assert_eq!(
        args["frame"],
        BoundArg::Frame(FrameOfValues {
            frame: path("scanned.frame"),
            values: path("scanned.points"),
            crossing: None,
        })
    );
    assert_eq!(
        args["timestamp"],
        BoundArg::Timestamp(path("scanned.timestamp"))
    );
    assert_eq!(
        args["points"],
        BoundArg::Placed(Placed {
            value: path("scanned.points"),
            each: true,
            stamp: None,
            fields: ["position".to_owned()].into(),
            crossing: None,
        })
    );
    let by_item = checking_positions(json!({
        "points": [{ "$ref": "scanned.points[0]" }, { "$ref": "scanned.points[1]" }],
        "frame": { "$ref": "scanned.frame" },
    }));
    assert_eq!(
        errors(&by_item, &fleet, Some(&alpha())),
        Vec::<String>::new()
    );
    refuses(
        &errors(
            &checking_positions(json!({ "points": { "$ref": "again.points" } })),
            &fleet,
            Some(&alpha()),
        ),
        "alpha.grab: `arm.grab` reaches for an item of `scanned`, and no reach check before it",
    );
}

/// A frame or a timestamp given beside values by `$ref` that is not the `$ref` of theirs, values of two records, a
/// literal value beside them, a list of values that each name their own frame and one value given to the list are
/// refused, each saying what to write.
#[test]
fn a_request_in_one_frame_refuses_values_of_two_records_and_a_frame_beside_its_values() {
    let fleet = snapshot(vec![robot("alpha")]);
    let refused = |args: Json, part: &str| {
        refuses(
            &errors(&checking_positions(args), &fleet, Some(&alpha())),
            part,
        )
    };
    refused(
        json!({ "points": { "$ref": "scanned.points" }, "frame": "map" }),
        "alpha.near: remove `frame`: the values of the record come by `$ref` from a record that gives `scanned.frame`, which the record takes with them",
    );
    refused(
        json!({ "points": { "$ref": "scanned.points" }, "timestamp": { "$ref": "again.timestamp" } }),
        "alpha.near: remove `timestamp`: the values of the record come by `$ref` from a record that gives `scanned.timestamp`",
    );
    refused(
        json!({ "points": [{ "$ref": "scanned.points[0]" }, { "$ref": "again.points[0]" }] }),
        "alpha.near: `scanned.points[0]` and `again.points[0]` come from two records: give values of one record that names one `frame`, or one value that names its frame",
    );
    refused(
        json!({ "points": [{ "$ref": "scanned.points[0]" }, { "position": [0.3, 0.0, 0.1] }] }),
        "alpha.near: `points` is a literal beside values given by `$ref`",
    );
    refused(
        json!({ "points": [{ "$ref": "seen.item_point" }, { "$ref": "scanned.points[0]" }] }),
        "alpha.near: `seen.item_point` and `scanned.points[0]` come from two records: give values of one record that names one `frame`, or one value that names its frame",
    );
    refused(
        json!({ "frame": { "$ref": "scanned.frame" }, "timestamp": { "$ref": "scanned.timestamp" }, "points": [{ "position": [0.3, 0.0, 0.1] }] }),
        "alpha.near: `frame` is a part of a record that names its frame",
    );
    refused(
        json!({ "points": { "$ref": "seen.item_point" } }),
        "alpha.near: `$ref: seen.item_point` is object, and `points` takes list of object: give `points` a `$ref` to a list of points",
    );
    let mut located = checking_positions(json!({ "points": { "$ref": "located.item_points" } }));
    located["actions"]["alpha"]["tree"]["sequence"]
        .as_array_mut()
        .unwrap()
        .insert(
            0,
            json!({ "id": "all", "timeout_s": 5, "call": { "member": "eyes.locate_all", "args": { "description": "blocks" }, "as": "located" } }),
        );
    refuses(
        &errors(&located, &fleet, Some(&alpha())),
        "alpha.near: `located.item_points` holds values that each name a frame of their own, and `points` takes values in the one frame of the request: give values of one record that names one `frame`, or one value that names its frame",
    );
}

/// The picker whose reach check gives `args` to check_positions, a request that names its frame at its level, and
/// whose grab takes the item that its look found, as an identify, a check and a grab go.
fn identifying(args: Json) -> Json {
    let tree = json!({ "id": "pick", "sequence": [
        { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
        { "id": "near", "timeout_s": 5, "call": {
            "member": "arm.check_positions", "args": args, "as": "near", "done_when": "near.all_workable",
        } },
        { "id": "grab", "timeout_s": 30, "call": {
            "member": "arm.grab",
            "args": { "item_id": { "$ref": "seen.item_id" }, "target": { "$ref": "seen.item_point" }, "speed": 1 },
        } },
    ] });
    let mut value = one_arm("alpha", tree, json!([]));
    value.as_object_mut().unwrap().remove("assumptions");
    value
}

/// A request in one frame takes a point that names its own frame and time by `$ref` in its list: the point goes
/// with its position, the request takes its frame and its timestamp, and the check counts as the reach check of the
/// grab of the item that the look found. A frame given beside it is refused unless it is the point's own.
#[test]
fn a_request_in_one_frame_takes_the_frame_and_the_time_of_a_stamped_value() {
    let fleet = snapshot(vec![robot("alpha")]);
    let one = identifying(json!({ "points": [{ "$ref": "seen.item_point" }] }));
    assert_eq!(errors(&one, &fleet, Some(&alpha())), Vec::<String>::new());
    let planned = planned_on_alpha(&one);
    let args = &call_of(&planned, "alpha", "near").args;
    assert_eq!(
        args["frame"],
        BoundArg::Frame(FrameOfValues {
            frame: path("seen.item_point.frame"),
            values: path("seen.item_point"),
            crossing: None,
        })
    );
    assert_eq!(
        args["timestamp"],
        BoundArg::Timestamp(path("seen.item_point.timestamp"))
    );
    assert_eq!(
        args["points"],
        BoundArg::List(vec![BoundArg::Placed(Placed {
            value: path("seen.item_point"),
            each: false,
            stamp: None,
            fields: ["position".to_owned()].into(),
            crossing: None,
        })])
    );
    let own_frame = identifying(json!({
        "points": [{ "$ref": "seen.item_point" }],
        "frame": { "$ref": "seen.item_point.frame" },
    }));
    assert_eq!(
        errors(&own_frame, &fleet, Some(&alpha())),
        Vec::<String>::new()
    );
    refuses(
        &errors(
            &identifying(json!({ "points": [{ "$ref": "seen.item_point" }], "frame": "map" })),
            &fleet,
            Some(&alpha()),
        ),
        "alpha.near: remove `frame`: the values of the record come by `$ref` from a record that gives `seen.item_point.frame`, which the record takes with them",
    );
}

/// A reach check counts only when every pin that the role reads gives the
/// field of the verdict: before a binding, the role reads each pin of the
/// fleet, and a pin of the arm whose check gives no verdict leaves the grab
/// unchecked. Bound to the robot whose pin gives it, the check counts.
#[test]
fn a_reach_check_counts_only_when_every_pin_gives_its_verdict() {
    const UNMARKED_ARM_PIN: &str =
        "4444444444444444444444444444444444444444444444444444444444444444";
    let unmarked = TEST_ARM.replace(
        r#"$type: "bool", $marks: ["checks_reach"]"#,
        r#"$type: "bool""#,
    );
    assert_ne!(unmarked, TEST_ARM);
    let mut beta = robot("beta");
    beta.contracts[0].sha256 = UNMARKED_ARM_PIN.to_owned();
    let mut fleet = snapshot(vec![robot("alpha"), beta]);
    fleet.contracts.insert(
        UNMARKED_ARM_PIN.to_owned(),
        Arc::new(ContractShape::parse(&unmarked, UNMARKED_ARM_PIN).unwrap()),
    );
    let checked = picker_checking(&[reach_check("seen", Some("reach.reachable"))]);

    refuses(&errors(&checked, &fleet, None), UNCHECKED_GRAB);
    assert_eq!(
        errors(&checked, &fleet, Some(&alpha())),
        Vec::<String>::new()
    );
}

#[test]
fn an_observation_reads_a_service_or_a_topic() {
    let mut value = picker();
    value["observations"]["target"] = json!({ "member": "arm.grab", "role": "alpha", "args": {} });
    refuses(
        &errors(&value, &snapshot(vec![robot("alpha")]), None),
        "is an action; an observation reads a service or a topic",
    );
}

#[test]
fn a_contract_the_director_does_not_allow_is_refused() {
    let found = check(
        &workflow(&picker()),
        &snapshot(vec![robot("alpha")]),
        &crate::config::AllowedContracts::parse("test_arm:v1").unwrap(),
        None,
    );
    let messages: Vec<String> = found.iter().map(ToString::to_string).collect();
    refuses(
        &messages,
        "contracts.eyes: `test_perception:v1` is not an allowed contract",
    );
}

#[test]
fn a_pinned_document_that_does_not_read_is_named() {
    let mut fleet = snapshot(vec![robot("alpha")]);
    fleet.contracts.remove(ARM_PIN);
    fleet
        .unreadable
        .insert(ARM_PIN.to_owned(), "test_arm:v1: a bad document".to_owned());
    refuses(
        &errors(&picker(), &fleet, None),
        "a pinned document of `test_arm:v1` does not read: test_arm:v1: a bad document",
    );
}

#[test]
fn a_member_on_several_instances_names_its_instance() {
    let mut twice = robot("alpha");
    twice.instances.push(Instance {
        name: "spare".to_owned(),
        instance: "alpha_spare".to_owned(),
        ..twice.instances[0].clone()
    });
    twice.contracts[0].instances.push("spare".to_owned());
    refuses(
        &errors(&picker(), &snapshot(vec![twice]), Some(&alpha())),
        "has several instances of `test_arm:v1` (brain, spare); name one with `instance`",
    );
}

#[test]
fn the_binding_refuses_each_robot_that_cannot_take_the_role() {
    let unattended = |change: fn(&mut Robot)| {
        let mut robot = robot("alpha");
        change(&mut robot);
        plan_errors(&picker(), vec![robot], Attendance::Unattended)
    };
    // The gates of an unattended run on a physical robot follow its plan.
    assert_eq!(
        unattended(|r| r.kind = "physical".to_owned()),
        Vec::<String>::new()
    );
    refuses(
        &unattended(|r| r.robot_commander = "headset".to_owned()),
        "has the robot commander `headset`",
    );
    refuses(
        &unattended(|r| r.software = "stopped".to_owned()),
        "the software of `alpha` does not run",
    );
    refuses(
        &unattended(|r| drop(r.contracts.remove(0))),
        "`alpha` does not implement `test_arm:v1`",
    );
    // The picker calls no member of `eyes`, so a robot without it fits.
    assert_eq!(
        unattended(|r| r.contracts.truncate(1)),
        Vec::<String>::new()
    );
    refuses(
        &plan_errors(&picker(), vec![robot("beta")], Attendance::Attended),
        "`alpha` is no robot of the fleet listing",
    );
    assert_eq!(
        plan_errors(&picker(), vec![robot("alpha")], Attendance::Unattended),
        Vec::<String>::new()
    );
}

/// A field with the mark location takes a `{ $location }` of a declared location, on an instance that implements
/// locations:v1, and the skill gets the name; a literal, a `$location` to another field and an instance without
/// locations:v1 are refused.
#[test]
fn a_location_goes_by_its_name_to_a_marked_field_of_a_skill_that_knows_locations() {
    let going = |place: Json| {
        let mut value = one_arm(
            "alpha",
            json!({ "id": "go", "timeout_s": 5, "call": { "member": "arm.go_to", "args": { "place": place } } }),
            json!([]),
        );
        value["locations"] = json!(["bin"]);
        value
    };
    let fine = going(json!({ "$location": "bin" }));
    assert!(errors(&fine, &snapshot(vec![robot("alpha")]), Some(&alpha())).is_empty());
    let planned = super::plan(
        Arc::new(workflow(&fine)),
        &snapshot(vec![robot("alpha")]),
        &allowed(),
        &alpha(),
        choices(Attendance::Unattended),
    )
    .unwrap();
    assert_eq!(planned.locations.len(), 1);
    assert_eq!(planned.locations[0].role, name("alpha"));
    assert_eq!(planned.locations[0].names, [name("bin")].into());
    refuses(
        &errors(
            &going(json!("bin")),
            &snapshot(vec![robot("alpha")]),
            Some(&alpha()),
        ),
        "`place` takes a location; give it a `{ $location }`",
    );
    let mut elsewhere = one_arm(
        "alpha",
        json!({ "id": "home", "timeout_s": 5, "call": { "member": "arm.home", "args": { "duration_s": { "$location": "bin" } } } }),
        json!([]),
    );
    elsewhere["locations"] = json!(["bin"]);
    refuses(
        &errors(&elsewhere, &snapshot(vec![robot("alpha")]), Some(&alpha())),
        "`duration_s` takes no location",
    );
    let mut unknowing = robot("alpha");
    unknowing
        .contracts
        .retain(|contract| contract.name != "locations");
    assert_eq!(
        errors(&fine, &snapshot(vec![unknowing]), Some(&alpha())),
        [
            "roles.alpha: the instance `brain` of `alpha` serves `arm`, to which the role `alpha` gives a location, and implements no locations:v1"
        ],
        "one rule, one error"
    );
}

/// A field with the mark location takes the name of the location in its own form: a field that takes no string
/// refuses a `{ $location }`.
#[test]
fn a_location_goes_only_to_a_field_that_takes_its_name() {
    let mut value = one_arm(
        "alpha",
        json!({ "id": "go", "timeout_s": 5, "call": { "member": "arm.go_to", "args": { "place": { "$location": "bin" } } } }),
        json!([]),
    );
    value["locations"] = json!(["bin"]);
    let numbered = TEST_ARM.replace(
        r#"place: { $type: "string", $marks: ["location"] }"#,
        r#"place: { $type: "u32", $marks: ["location"] }"#,
    );
    assert_ne!(numbered, TEST_ARM);
    let mut fleet = snapshot(vec![robot("alpha")]);
    fleet.contracts.insert(
        ARM_PIN.to_owned(),
        Arc::new(ContractShape::parse(&numbered, ARM_PIN).unwrap()),
    );
    refuses(
        &errors(&value, &fleet, Some(&alpha())),
        "`place` has the mark location and takes no location name",
    );
}

/// A check whose roles name one robot twice refuses the second role, naming the first.
#[test]
fn a_check_that_binds_one_robot_to_two_roles_refuses_the_second() {
    let mut value = picker();
    value["roles"] = json!({ "left": {}, "right": {} });
    let tree = value["actions"]["alpha"]["tree"].clone();
    value["actions"] = json!({ "left": { "tree": tree.clone() }, "right": { "tree": tree } });
    value["observations"]["target"]["role"] = json!("left");
    let binding = BTreeMap::from([
        (name("left"), robot_name("alpha")),
        (name("right"), robot_name("alpha")),
    ]);
    refuses(
        &errors(&value, &snapshot(vec![robot("alpha")]), Some(&binding)),
        "roles.right: `alpha` is the robot of the role `left` too: name another robot for each role",
    );
}

/// A watch parameter reads a number: one that reads a bool or a string
/// refuses the plan at `watch.<name>`, and one that reads a count or a
/// coordinate holds.
#[test]
fn a_watch_parameter_reads_a_number() {
    let watching = |pairs: &[(&str, &str)]| Watch {
        observations: pairs
            .iter()
            .map(|(name, text)| {
                let observation = serde_json::from_value(json!(text)).unwrap();
                (self::name(name), observation)
            })
            .collect(),
        validation: None,
    };
    let numbers = watching(&[
        ("loose", "target.count"),
        ("x", "target.item_point.position[0]"),
    ]);
    let found = plan_errors_watching(
        &picker(),
        vec![robot("alpha")],
        Attendance::Attended,
        numbers,
    );
    assert!(found.is_empty(), "{found:#?}");
    let others = watching(&[("found", "target.found"), ("kind", "target.kind")]);
    let found = plan_errors_watching(
        &picker(),
        vec![robot("alpha")],
        Attendance::Attended,
        others,
    );
    assert_eq!(found.len(), 2, "{found:#?}");
    refuses(
        &found,
        "watch.found: `target.found` is bool; a watch parameter reads a number",
    );
    refuses(&found, "watch.kind: `target.kind`");
}

/// The plan of `value` with its one role `alpha` bound to the robot `alpha`.
fn planned_on_alpha(value: &Json) -> Plan {
    plan(
        Arc::new(workflow(value)),
        &snapshot(vec![robot("alpha")]),
        &allowed(),
        &alpha(),
        choices(Attendance::Attended),
    )
    .unwrap_or_else(|errors| panic!("{errors:#?}"))
}

/// The picker whose grab takes `target` after the steps `before`, with the observation `arms` of the grasp points
/// of `alpha`.
fn grabbing_after(before: &[Json], target: Json) -> Json {
    let mut value = grabbing_at(target);
    value["observations"]["arms"] =
        json!({ "topic": "arm.limb_states", "role": "alpha", "max_age_s": 5 });
    let steps = value["actions"]["alpha"]["tree"]["sequence"]
        .as_array_mut()
        .unwrap();
    for (at, step) in before.iter().enumerate() {
        steps.insert(1 + at, step.clone());
    }
    value
}

/// The call of the planning scene, as `scene`.
fn scene_call() -> Json {
    json!({ "id": "scene", "timeout_s": 5, "call": { "member": "arm.get_planning_scene", "as": "scene" } })
}

/// The placed records that the grab of `value` takes as its target.
fn grab_target(value: &Json) -> Placed {
    let planned = planned_on_alpha(value);
    match &call_of(&planned, "alpha", "grab").args["target"] {
        BoundArg::Placed(placed) => placed.clone(),
        other => panic!("{other:?}"),
    }
}

/// A `$ref` to a stamped value brings its own frame and time; one to a grasp point of limb_states reaches the
/// grab as a pose_stamped with the frame and the timestamp of its message; one to an object of the planning scene
/// takes the frame and the time of the scene.
#[test]
fn a_ref_to_a_plain_value_takes_the_frame_and_the_timestamp_of_its_record() {
    let own = grab_target(&picker());
    assert_eq!((own.stamp, own.crossing), (None, None));
    let grasp = grab_target(&grabbing_after(
        &[],
        json!({ "$ref": "arms.grasp_points[1]" }),
    ));
    assert_eq!(
        grasp.stamp,
        Some(Stamp {
            frame: workflow_documents::expression::RefPath::parse("arms.frame").unwrap(),
            timestamp: Some(
                workflow_documents::expression::RefPath::parse("arms.timestamp").unwrap()
            ),
        })
    );
    assert_eq!(
        grasp.fields,
        ["frame", "orientation", "position", "timestamp"]
            .map(str::to_owned)
            .into()
    );
    let object = grab_target(&grabbing_after(
        &[scene_call()],
        json!({ "$ref": "scene.objects[0]" }),
    ));
    assert_eq!(
        object.stamp.map(|stamp| stamp.frame.to_string()),
        Some("scene.frame".to_owned())
    );
}

/// The call of a plan of the path of the left arm, as `plan`.
fn plan_call() -> Json {
    json!({ "id": "plan", "timeout_s": 5, "call": { "member": "arm.plan_path", "args": { "arm_name": "left_arm" }, "as": "plan" } })
}

/// A record whose values hold at every time holds no timestamp: a `$ref` to an attached object, which names its
/// frame, goes as it is; a waypoint of a plan takes the frame of the plan and no timestamp; a request in one frame
/// takes the frame of the plan of its points and no timestamp, and refuses a timestamp given beside them.
#[test]
fn a_record_whose_values_hold_at_every_time_gives_its_frame_alone() {
    let attached = grab_target(&grabbing_after(
        &[scene_call()],
        json!({ "$ref": "scene.attached_objects[0]" }),
    ));
    assert_eq!(attached.stamp, None);
    let waypoint = grab_target(&grabbing_after(
        &[plan_call()],
        json!({ "$ref": "plan.waypoints[1]" }),
    ));
    assert_eq!(
        waypoint.stamp,
        Some(Stamp {
            frame: path("plan.frame"),
            timestamp: None,
        })
    );
    let checking = |args: Json| {
        let tree = json!({ "id": "check", "sequence": [
            plan_call(),
            { "id": "near", "timeout_s": 5, "call": {
                "member": "arm.check_positions", "args": args, "as": "near", "done_when": "near.all_workable",
            } },
        ] });
        let mut value = one_arm("alpha", tree, json!([]));
        value.as_object_mut().unwrap().remove("assumptions");
        value
    };
    let planned = planned_on_alpha(&checking(json!({ "points": { "$ref": "plan.waypoints" } })));
    let args = &call_of(&planned, "alpha", "near").args;
    assert_eq!(
        args["frame"],
        BoundArg::Frame(FrameOfValues {
            frame: path("plan.frame"),
            values: path("plan.waypoints"),
            crossing: None,
        })
    );
    assert!(!args.contains_key("timestamp"), "{args:?}");
    refuses(
        &errors(
            &checking(
                json!({ "points": { "$ref": "plan.waypoints" }, "timestamp": "2026-11-18T12:00:00Z" }),
            ),
            &snapshot(vec![robot("alpha")]),
            Some(&alpha()),
        ),
        "alpha.near: remove `timestamp`: the values of the record come by `$ref` from a record whose values hold at every time, which gives none",
    );
}

/// A literal record that leaves its frame out lies in the frame of the robot of its call, which holds no link to
/// refuse: it is taken, whole or field by field, with and without roles, and goes with no frame, as the zero target
/// of a grab by an item id does; the links of its robot are not read for it.
#[test]
fn a_literal_with_no_frame_is_taken_and_goes_with_no_frame() {
    let mut fleet = snapshot(vec![robot("alpha")]);
    fleet.links.clear();
    let zero = grabbing_at(json!({ "position": [0.0, 0.0, 0.0] }));
    let tree = json!({ "id": "near", "timeout_s": 5, "call": {
        "member": "arm.check_positions", "args": { "points": [{ "position": [0.3, 0.0, 0.1] }] },
        "as": "near", "done_when": "near.all_workable",
    } });
    let mut in_parts = one_arm("alpha", tree, json!([]));
    in_parts.as_object_mut().unwrap().remove("assumptions");
    for value in [&zero, &in_parts] {
        for binding in [None, Some(&alpha())] {
            assert_eq!(errors(value, &fleet, binding), Vec::<String>::new());
        }
    }
    let planned = |value: &Json| {
        plan(
            Arc::new(workflow(value)),
            &fleet,
            &allowed(),
            &alpha(),
            choices(Attendance::Attended),
        )
        .unwrap_or_else(|errors| panic!("{errors:#?}"))
    };
    let grab = planned(&zero);
    let BoundArg::Object(target) = &call_of(&grab, "alpha", "grab").args["target"] else {
        panic!("{grab:?}");
    };
    assert_eq!(target.keys().collect::<Vec<_>>(), ["position"]);
    assert!(grab.reads_links.is_empty(), "{:?}", grab.reads_links);
    let near = planned(&in_parts);
    assert!(
        !call_of(&near, "alpha", "near").args.contains_key("frame"),
        "{near:?}"
    );
}

/// A point goes to a target that takes a pose whose orientation may be left out, with no orientation, and a target
/// whose orientation the member needs refuses it; a pose goes to both.
#[test]
fn a_point_goes_only_to_a_pose_whose_orientation_may_be_left_out() {
    let fleet = snapshot(vec![robot("alpha")]);
    assert_eq!(
        errors(&picker(), &fleet, Some(&alpha())),
        Vec::<String>::new()
    );
    let aiming = |target: Json| {
        let aim = json!({ "id": "aim", "timeout_s": 5, "call": { "member": "eyes.aim", "args": { "target": target } } });
        grabbing_after(&[aim], json!({ "$ref": "seen.item_point" }))
    };
    assert_eq!(
        errors(
            &aiming(json!({ "$ref": "seen.item_point" })),
            &fleet,
            Some(&alpha()),
        ),
        [
            "alpha.aim: `seen.item_point` gives no `orientation` of the kind `target` takes: give `target` a `$ref` to a pose that gives `orientation`"
        ],
    );
    assert_eq!(
        errors(
            &aiming(json!({ "$ref": "arms.grasp_points[1]" })),
            &fleet,
            Some(&alpha())
        ),
        Vec::<String>::new()
    );
}

/// A shape of a collision object is posed relative to its object, a vertex of a mesh relative to its shape, and a
/// shape of an attached object relative to that object, which names its frame: a `$ref` to each is refused.
#[test]
fn a_ref_to_a_pose_in_a_record_without_one_frame_is_refused() {
    let fleet = snapshot(vec![robot("alpha")]);
    for (value, record) in [
        ("scene.objects[0].shapes[0]", "scene.objects[0]"),
        (
            "scene.objects[0].shapes[0].mesh_vertices[1]",
            "scene.objects[0].shapes[0]",
        ),
        (
            "scene.attached_objects[0].shapes[0]",
            "scene.attached_objects[0]",
        ),
    ] {
        let relative = grabbing_after(&[scene_call()], json!({ "$ref": value }));
        refuses(
            &errors(&relative, &fleet, Some(&alpha())),
            &format!(
                "alpha.grab: `{value}` is posed relative to the position and the orientation of `{record}`"
            ),
        );
    }
    let item_ids = grabbing_after(&[], json!({ "$ref": "seen.item_id" }));
    assert_eq!(
        errors(&item_ids, &fleet, Some(&alpha())),
        [
            "alpha.grab: `$ref: seen.item_id` is string, and `target` takes object: give `target` a `$ref` to a pose"
        ],
    );
    let twist = grabbing_after(
        &[
            json!({ "id": "where", "timeout_s": 5, "call": { "member": "arm.odometry", "as": "odo" } }),
        ],
        json!({ "$ref": "odo.twist" }),
    );
    assert_eq!(
        errors(&twist, &fleet, Some(&alpha())),
        [
            "alpha.grab: `odo.twist` gives none of the values that `target` takes: give `target` a `$ref` to a pose"
        ],
    );
}

/// A loop over a list of plain values of a result gives elements that take the frame and the timestamp of that
/// result; a loop over the list of an observation reads its message once, so its elements take none.
#[test]
fn a_loop_element_of_a_result_takes_the_frame_of_the_result_and_one_of_an_observation_none() {
    let looping = |list: &str| {
        let tree = json!({ "id": "pick", "sequence": [
            { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
            { "id": "scan", "timeout_s": 5, "call": { "member": "eyes.scan", "args": { "description": "blocks" }, "as": "scanned" } },
            { "id": "each", "for_each": { "list": list, "as": "point" }, "max_items": 5, "body": {
                "id": "grab", "timeout_s": 30, "call": {
                    "member": "arm.grab",
                    "args": { "item_id": { "$ref": "seen.item_id" }, "target": { "$ref": "point" }, "speed": 1 },
                },
            } },
        ] });
        let mut value = one_arm("alpha", tree, json!([]));
        value["observations"]["arms"] =
            json!({ "topic": "arm.limb_states", "role": "alpha", "max_age_s": 5 });
        value
    };
    let of_a_result = grab_target(&looping("scanned.points"));
    assert_eq!(
        of_a_result
            .stamp
            .and_then(|stamp| stamp.timestamp)
            .map(|timestamp| timestamp.to_string()),
        Some("scanned.timestamp".to_owned())
    );
    refuses(
        &errors(
            &looping("arms.grasp_points"),
            &snapshot(vec![robot("alpha")]),
            Some(&alpha()),
        ),
        "alpha.grab: `point` is an element of `arms.grasp_points`, a list of an observation, whose message its loop read once",
    );
}

/// The anomaly monitor answers its picture as `data`: a service that answers `frame` is no monitor.
#[test]
fn an_anomaly_monitor_answers_its_picture_as_data() {
    const MONITOR_PIN: &str = "5555555555555555555555555555555555555555555555555555555555555555";
    let monitor = |picture: &str| {
        format!(
            r#"{{
              peppy_schema: "contract/v1",
              manifest: {{ name: "test_monitor", tag: "v1", description: ["A monitor for the tests of the director."] }},
              interfaces: {{ services: [{{
                name: "check",
                description: "Whether a step went wrong.",
                request_message_format: {{ robot: "string", step: "string" }},
                response_message_format: {{
                  anomaly: "bool", kind: "string", severity: {{ $type: "string", $enum: ["low", "medium", "high"] }},
                  description: "string", {picture}: "bytes",
                }},
              }}] }},
            }}"#
        )
    };
    let planned = |picture: &str| {
        let mut alpha_robot = robot("alpha");
        alpha_robot.contracts.push(crate::fleet::RobotContract {
            name: "test_monitor".to_owned(),
            tag: "v1".to_owned(),
            sha256: MONITOR_PIN.to_owned(),
            instances: vec!["brain".to_owned()],
        });
        let mut fleet = snapshot(vec![alpha_robot]);
        fleet.contracts.insert(
            MONITOR_PIN.to_owned(),
            Arc::new(ContractShape::parse(&monitor(picture), MONITOR_PIN).unwrap()),
        );
        fleet.fleet.nodes.push(crate::fleet::Node {
            name: "monitor".to_owned(),
            tag: "v1".to_owned(),
            machines: Vec::new(),
            instances: vec![(
                "alpha_brain".to_owned(),
                "core".to_owned(),
                "alpha".to_owned(),
            )],
        });
        let incidents: Incidents = serde_json::from_value(json!({
            "monitor": { "contract": "test_monitor:v1", "member": "check", "stack_instance": "alpha_brain" },
        }))
        .unwrap();
        plan(
            Arc::new(workflow(&picker())),
            &fleet,
            &crate::config::AllowedContracts::parse(
                "test_arm:v1, test_perception:v1, test_monitor:v1",
            )
            .unwrap(),
            &alpha(),
            RunChoices {
                incidents: &incidents,
                ..choices(Attendance::Attended)
            },
        )
        .map(|planned| planned.monitor.is_some())
        .map_err(|errors| errors.iter().map(ToString::to_string).collect::<Vec<_>>())
    };
    assert_eq!(planned("data"), Ok(true));
    refuses(
        &planned("frame").unwrap_err(),
        "is no anomaly monitor: a service of { robot, step } that answers { anomaly, kind, severity, description, data }",
    );
}

/// A run whose role gives a literal record that names its frame reads the links of its robot, and its plan says
/// so; a run that gives none reads them not.
#[test]
fn a_plan_names_the_roles_whose_links_it_read() {
    assert!(planned_on_alpha(&picker()).reads_links.is_empty());
    assert_eq!(
        planned_on_alpha(&grabbing_at(literal_pose("map"))).reads_links,
        [name("alpha")].into()
    );
}

/// A workflow that gives a member an argument it does not take, as the `position` of a move_arm that takes a
/// `target`, is refused naming the argument and each argument of the member, with what each record that names its
/// frame takes.
#[test]
fn an_unknown_argument_is_refused_naming_the_arguments_of_the_member() {
    let fleet = snapshot(vec![robot("alpha")]);
    let moving = one_arm(
        "alpha",
        json!({ "id": "move", "timeout_s": 30, "call": { "member": "arm.move_arm", "args": {
            "arm_name": "left_arm", "position": [0.3, 0.0, 0.2], "duration_s": 0, "settle_position_tolerance_m": 0.01,
            "settle_orientation_tolerance_rad": 0.05, "settle_timeout_s": 1, "contact_gap_m": 0.02,
        } } }),
        json!([]),
    );
    refuses(
        &errors(&moving, &fleet, Some(&alpha())),
        "alpha.move: `arm.move_arm` has no argument `position`: its arguments are `arm_name`, `contact_gap_m`, `duration_s`, `settle_orientation_tolerance_rad`, `settle_position_tolerance_m`, `settle_timeout_s`, `target` (a pose with frame and timestamp)",
    );
    let mut reaching = picker();
    reaching["actions"]["alpha"]["tree"]["sequence"][0] = json!({ "id": "look", "timeout_s": 5, "call": {
        "member": "arm.check_reach", "args": { "positions": [] }, "as": "seen",
    } });
    refuses(
        &errors(&reaching, &fleet, Some(&alpha())),
        "`arm.check_reach` has no argument `positions`: its arguments are `points` (a list of points each with frame and timestamp)",
    );
}
