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
use crate::fleet::{Instance, Robot, Snapshot};
use crate::testing::{ARM_PIN, allowed, name, one_arm, robot, robot_name, snapshot, workflow};

fn grab_tree(item_id: Json) -> Json {
    json!({ "id": "pick", "sequence": [
        { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
        { "id": "grab", "timeout_s": 30, "call": {
            "member": "arm.grab",
            "args": { "item_id": item_id, "position": { "$ref": "seen.position" }, "speed": 1 },
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
    let choices = RunChoices {
        attendance,
        confirmed: Default::default(),
        incidents: &Incidents::None,
        poll: workflow_documents::Bound::from_seconds(0.5).unwrap(),
        cancel_timeout: Duration::from_secs(10),
    };
    match plan(
        Arc::new(workflow(value)),
        &snapshot(robots),
        &allowed(),
        &alpha(),
        choices,
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
fn an_item_id_or_a_position_takes_only_a_ref() {
    let literal = one_arm("alpha", grab_tree(json!("b1")), json!([]));
    refuses(
        &errors(&literal, &snapshot(vec![robot("alpha")]), None),
        "alpha.grab: `item_id` holds a position, a pose or an item id",
    );
}

/// A position reaches a marked field only by a `$ref`: a literal row of a table, or a literal that a lane group
/// gives as a parameter, is refused like a literal.
#[test]
fn a_position_takes_no_table_row_and_no_parameter() {
    let fleet = snapshot(vec![robot("alpha")]);
    let mut by_table = picker();
    by_table["tables"] = json!({ "spots": {
        "block": [0.3, 0.0, 0.1], "cup": [0.3, 0.1, 0.1], "unknown": [0.3, 0.2, 0.1],
    } });
    by_table["actions"]["alpha"]["tree"]["sequence"][1]["call"]["args"]["position"] =
        json!({ "$table": "spots", "key": { "$ref": "seen.kind" } });
    refuses(
        &errors(&by_table, &fleet, None),
        "`position` holds a position, a pose or an item id; give it a `$ref`",
    );
    let mut by_param = picker();
    let tree = by_param["actions"]["alpha"]["tree"].clone();
    by_param["actions"] = json!({ "pickers": {
        "roles": ["alpha"],
        "params": { "alpha": { "spot": [0.3, 0.0, 0.1] } },
        "tree": tree,
    } });
    by_param["actions"]["pickers"]["tree"]["sequence"][1]["call"]["args"]["position"] =
        json!({ "$param": "spot" });
    refuses(
        &errors(&by_param, &fleet, None),
        "`position` holds a position, a pose or an item id",
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
            "args": { "item_id": { "$ref": "scanned.item_ids[0]" }, "position": { "$ref": "seen.position" }, "speed": 1 },
        } },
    ] });
    let value = one_arm("alpha", tree, json!([]));
    refuses(
        &errors(&value, &snapshot(vec![split]), Some(&alpha())),
        "an item id is valid only on the instance that gives it",
    );
}

/// The observation `target` reads the arm of `alpha`, so its position is in the frame of that robot.
#[test]
fn a_position_goes_only_to_the_robot_whose_frame_it_is_in() {
    let mut value = picker();
    value["roles"]["beta"] = json!({});
    value["actions"]["beta"] = json!({ "tree": { "id": "reach", "timeout_s": 5, "call": {
        "member": "arm.check_reach", "args": { "position": { "$ref": "target.position" } },
    } } });
    let found = errors(&value, &snapshot(vec![robot("alpha"), robot("beta")]), None);
    refuses(
        &found,
        "beta.reach: `target.position` is in the frame of the robot of `alpha`, and `position` goes to the robot of `beta`",
    );
    assert!(
        !found.iter().any(|error| error.starts_with("alpha.")),
        "{found:#?}"
    );
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
    refuses(
        &unattended(|r| r.kind = "physical".to_owned()),
        "`alpha` is physical",
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
        RunChoices {
            attendance: Attendance::Unattended,
            confirmed: Default::default(),
            incidents: &Incidents::None,
            poll: workflow_documents::Bound::from_seconds(0.5).unwrap(),
            cancel_timeout: Duration::from_secs(10),
        },
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
