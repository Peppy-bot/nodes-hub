//! The director over the wire, against a scripted fleet and a scripted arm, for the validation of workflows: the
//! work cell of a run and the values that no test covered, and the gates of an unattended run on a physical robot.

use peppygen::DEFAULT_FEEDBACK_BOUND;
use peppygen::exposed_actions::director::run::GoalBundleItem;
use peppygen::fixtures::exposed_actions::director::run;
use peppygen::fixtures::exposed_services::director::forget_robot;
use serde_json::{Value, json};
use workflow_documents::lock::{Lock, LockedNode};
use workflow_documents::{Bundle, NameTag, Sha256, Validation, Workflow};

mod common;
use common::{ARM_PIN, Home, ROBOT, Rig, WIRE, home_workflow, listed_robots, on_alpha};

/// The node of the arm, its sha256 and its commit, as the scripted listing gives them.
const ARM_NODE: (&str, &str, &str) = (
    "test_arm_node:v1",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "0123456",
);

/// The lock of the scripted stack: its peppy version, the pin of the arm and the node of `alpha`, at `commit`.
fn lock_at(commit: &str) -> Lock {
    let (node, sha256, _) = ARM_NODE;
    Lock {
        contracts: [(
            NameTag::parse("test_arm:v1").unwrap(),
            Sha256::parse(ARM_PIN).unwrap(),
        )]
        .into(),
        peppy: "0.36.0".to_owned(),
        nodes: [(
            NameTag::parse(node).unwrap(),
            LockedNode {
                sha256: Sha256::parse(sha256).unwrap(),
                commit: Some(commit.to_owned()),
            },
        )]
        .into(),
    }
}

/// The validation of a sweep of `home_workflow` that tested a belt speed from 0.05 to 0.25 m/s.
fn belt_speed_tested() -> Validation {
    let thresholds = json!({
        "completed_at_least": 1.0, "unsafe_events_at_most": 0,
        "false_successes_at_most": 0.0, "false_assumption_checks_at_most": 0,
    });
    Validation::new(
        serde_json::from_value(thresholds).unwrap(),
        serde_json::from_value(json!({ "belt_speed_mps": { "unit": "m/s", "kind": "control" } }))
            .unwrap(),
        serde_json::from_value(json!({ "belt_speed_mps": [0.05, 0.25] })).unwrap(),
    )
    .unwrap()
}

/// The files of the bundle of `home_workflow` with `lock`, signed off with `validation` when it gives one.
fn bundle_of_home(lock: Lock, validation: Option<Validation>) -> Vec<GoalBundleItem> {
    let workflow = Workflow::parse(&home_workflow()).unwrap();
    let files = match validation {
        Some(validation) => Bundle::write_signed_off(&workflow, Vec::new(), lock, validation),
        None => Bundle::write(&workflow, Vec::new(), lock),
    };
    files
        .unwrap()
        .into_iter()
        .map(|file| GoalBundleItem {
            path: file.path,
            content: file.content,
        })
        .collect()
}

/// The digest of the bundle of `files`.
fn digest_of(files: &[GoalBundleItem]) -> String {
    let files = files
        .iter()
        .map(|file| workflow_documents::BundleFile {
            path: file.path.clone(),
            content: file.content.clone(),
        })
        .collect();
    Bundle::open(files).unwrap().document.digest().to_string()
}

/// The run document of an unattended run of `home_workflow` that binds `arm` as `roles` says, with the work cell
/// block `work_cell`.
fn unattended(roles: Value, work_cell: Value) -> String {
    json!({
        "workflow_schema": "run/v1",
        "workflow": { "name": "go_home", "revision": 1 },
        "roles": roles,
        "attendance": "unattended",
        "work_cell": work_cell,
    })
    .to_string()
}

/// The run result of a run of the bundle `bundle` by the run document `document`, and the events of its feedback.
async fn run_bundle(
    rig: &Rig,
    bundle: Vec<GoalBundleItem>,
    document: String,
) -> peppygen::Result<(Value, Vec<run::Feedback>)> {
    let request = run::GoalRequestData {
        workflow: None,
        bundle: Some(bundle),
        run_document: Some(document),
        ..home_request()
    };
    let goal = run::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(goal.accepted, "{:?}", goal.refusal);
    ended(goal).await
}

/// The message of the refusal of the run result `result`.
fn refusal_of(result: &Value) -> String {
    assert_eq!(result["ending"], "refused", "{result}");
    result["refusal"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

/// A `run` request of `home_workflow` with no binding, which a test completes.
fn home_request() -> run::GoalRequestData {
    run::GoalRequestData {
        workflow: Some(home_workflow()),
        bundle: None,
        run_document: None,
        roles: None,
        work_cell: None,
        started_by: Some("tests".to_owned()),
        caller_data: None,
    }
}

/// The run result of the goal `goal` once its run ended, with the events of its feedback.
async fn ended(mut goal: run::GoalHandle) -> peppygen::Result<(Value, Vec<run::Feedback>)> {
    let mut feedback = Vec::new();
    while let Ok(next) = goal.on_next_feedback().await {
        feedback.push(next);
    }
    let ended = goal.get_result(WIRE).await?;
    let text = match ended.outcome {
        run::ResultOutcome::Completed(result) => result.result,
        other => {
            let last = feedback
                .iter()
                .rev()
                .find(|item| item.event == "run_ends")
                .unwrap_or_else(|| panic!("no run_ends: {other:?}"));
            last.details.clone()
        }
    };
    Ok((serde_json::from_str(&text).expect("a run result"), feedback))
}

/// A run made from roles takes its work cell block: the run result gives it, and the event `uncovered` and the run
/// result name each of its values, which no envelope covers with no bundle.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_made_from_roles_gives_its_work_cell_and_names_each_value_no_test_covered()
-> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    listed_robots(&rig).await?;
    let request = run::GoalRequestData {
        roles: Some(on_alpha()),
        work_cell: Some("{ belt_speed_mps: 0.2 }".to_owned()),
        ..home_request()
    };
    let goal = run::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(goal.accepted, "{:?}", goal.refusal);
    let (result, feedback) = ended(goal).await?;
    assert_eq!(result["ending"], "completed", "{result}");
    assert_eq!(result["work_cell"], json!({ "belt_speed_mps": 0.2 }));
    assert_eq!(
        result["uncovered"],
        json!([{ "parameter": "belt_speed_mps", "value": 0.2, "why": "no_envelope" }])
    );
    let uncovered = feedback
        .iter()
        .find(|item| item.event == "uncovered")
        .expect("the event uncovered");
    assert_eq!(
        serde_json::from_str::<Value>(&uncovered.details).unwrap(),
        result["uncovered"]
    );
    rig.harness.shutdown().await
}

/// A run document gives its own work cell block: a request that gives one with roles or with `work_cell`, or a
/// work cell block that does not parse, is refused with `rejected` and starts no run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_document_with_roles_or_a_work_cell_is_rejected() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    listed_robots(&rig).await?;
    let document = json!({
        "workflow_schema": "run/v1",
        "workflow": { "name": "go_home", "revision": 1 },
        "roles": { "arm": "alpha" },
        "attendance": "attended",
    })
    .to_string();
    let requests = [
        run::GoalRequestData {
            run_document: Some(document.clone()),
            roles: Some(on_alpha()),
            ..home_request()
        },
        run::GoalRequestData {
            run_document: Some(document),
            work_cell: Some("{ belt_speed_mps: 0.2 }".to_owned()),
            ..home_request()
        },
        run::GoalRequestData {
            roles: Some(on_alpha()),
            work_cell: Some("{ belt_speed_mps: ".to_owned()),
            ..home_request()
        },
    ];
    for request in requests {
        let goal = run::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
        assert!(!goal.accepted, "{request:?}");
        let refusal = goal.refusal.expect("a refusal");
        assert_eq!(refusal.kind.to_string(), "rejected", "{refusal:?}");
    }
    rig.harness.shutdown().await
}

/// The director refuses an unattended run on a physical robot, before it takes a lease, and names the cause: no
/// bundle, a bundle with no envelope, a work cell value missing or outside the envelope, `versions: "warn"`, a
/// difference from the lock, and no sign-off record of a trial of the bundle.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unattended_run_on_a_physical_robot_is_refused_naming_each_cause() -> peppygen::Result<()>
{
    let rig = Rig::start(Home::AtOnce).await?;
    rig.fleet.state.send_modify(|state| state.physical = true);
    listed_robots(&rig).await?;
    let inside = json!({ "belt_speed_mps": 0.2 });
    let signed_off = bundle_of_home(lock_at(ARM_NODE.2), Some(belt_speed_tested()));
    let request = run::GoalRequestData {
        run_document: Some(unattended(json!({ "arm": ROBOT }), inside.clone())),
        ..home_request()
    };
    let goal = run::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    let (no_bundle, _) = ended(goal).await?;
    assert!(
        refusal_of(&no_bundle).contains("give the bundle in place of the workflow"),
        "{no_bundle}"
    );
    let (no_envelope, _) = run_bundle(
        &rig,
        bundle_of_home(lock_at(ARM_NODE.2), None),
        unattended(json!({ "arm": ROBOT }), inside.clone()),
    )
    .await?;
    assert!(refusal_of(&no_envelope).starts_with("the bundle has no envelope"));
    let (missing, feedback) = run_bundle(
        &rig,
        signed_off.clone(),
        unattended(json!({ "arm": ROBOT }), json!({})),
    )
    .await?;
    assert!(
        refusal_of(&missing)
            .ends_with("no test covered `belt_speed_mps` has no value in the work cell block"),
        "{missing}"
    );
    assert!(feedback.iter().any(|item| item.event == "uncovered"));
    let (outside, _) = run_bundle(
        &rig,
        signed_off.clone(),
        unattended(json!({ "arm": ROBOT }), json!({ "belt_speed_mps": 0.3 })),
    )
    .await?;
    assert!(
        refusal_of(&outside)
            .ends_with("`belt_speed_mps` is 0.3, outside the tested range [0.05, 0.25]"),
        "{outside}"
    );
    let mut warned: Value =
        serde_json::from_str(&unattended(json!({ "arm": ROBOT }), inside.clone())).unwrap();
    warned["versions"] = json!("warn");
    let (warn, _) = run_bundle(&rig, signed_off.clone(), warned.to_string()).await?;
    assert!(
        refusal_of(&warn).contains("give its run document `versions: \"refuse\"`"),
        "{warn}"
    );
    let (differs, _) = run_bundle(
        &rig,
        bundle_of_home(lock_at("7654321"), Some(belt_speed_tested())),
        unattended(json!({ "arm": ROBOT }), inside.clone()),
    )
    .await?;
    assert!(
        refusal_of(&differs).starts_with("the stack differs from the lock of the bundle in 1 ways")
    );
    let (no_trial, _) = run_bundle(
        &rig,
        signed_off,
        unattended(json!({ "arm": ROBOT }), inside),
    )
    .await?;
    assert!(
        refusal_of(&no_trial).starts_with("no trial signed off the bundle"),
        "{no_trial}"
    );
    assert!(
        !rig.fleet
            .calls()
            .iter()
            .any(|call| call.starts_with("take")),
        "no refused run takes a lease: {:?}",
        rig.fleet.calls()
    );
    rig.harness.shutdown().await
}

/// A sign-off record that the director reads at its start admits an unattended run of its bundle on its robot, and
/// `"auto"` binds that physical robot as the record binds it; `forget_robot` removes the record, and the same run
/// is refused after.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sign_off_record_admits_an_unattended_run_until_forget_robot_removes_it()
-> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    let core_node = rig
        .harness
        .node_runner()
        .processor()
        .bound_core_node()
        .to_owned();
    let Rig {
        harness, journals, ..
    } = rig;
    harness.shutdown().await?;
    let bundle = bundle_of_home(lock_at(ARM_NODE.2), Some(belt_speed_tested()));
    let (node, sha256, commit) = ARM_NODE;
    let record = json!({
        "trial": "t1",
        "sign_off": {
            "bundle": digest_of(&bundle),
            "core_node": core_node,
            "binding": { "arm": { "name": ROBOT, "model": "", "core_node": core_node } },
            "nodes": { node: { "sha256": sha256, "commit": commit } },
            "incidents": {},
            "peppy": "0.36.0",
        },
    });
    let records = journals.path().join("workflow_director").join("sign_offs");
    std::fs::create_dir_all(&records).unwrap();
    std::fs::write(records.join("t1.json"), record.to_string()).unwrap();

    let rig = Rig::start_on(Home::AtOnce, &[], journals).await?;
    rig.fleet.state.send_modify(|state| state.physical = true);
    listed_robots(&rig).await?;
    let document = || unattended(json!({ "arm": "auto" }), json!({ "belt_speed_mps": 0.2 }));
    let (admitted, _) = run_bundle(&rig, bundle.clone(), document()).await?;
    assert_eq!(admitted["ending"], "completed", "{admitted}");
    assert_eq!(admitted["lanes"][0]["robot"], ROBOT);
    let request = forget_robot::RequestData {
        robot: ROBOT.to_owned(),
    };
    assert_eq!(
        forget_robot::poll(&rig.harness, &request, WIRE)
            .await?
            .removed,
        1
    );
    assert_eq!(
        forget_robot::poll(&rig.harness, &request, WIRE)
            .await?
            .removed,
        0
    );
    let (refused, _) = run_bundle(&rig, bundle, document()).await?;
    assert_eq!(refusal_of(&refused), "the roles do not bind");
    let why = refused["refusal"]["errors"][0]["message"]
        .as_str()
        .unwrap_or_default();
    assert!(
        why.contains(
            "\"auto\" binds a physical robot only as a sign-off record of the bundle binds it"
        ),
        "{refused}"
    );
    rig.harness.shutdown().await
}
