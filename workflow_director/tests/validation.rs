//! The director over the wire, against a scripted fleet and a scripted arm, for the validation of workflows: the
//! work cell of a run and the values that no test covered, the gates of an unattended run on a physical robot, and
//! the trials that sign off a bundle.

use peppygen::DEFAULT_FEEDBACK_BOUND;
use peppygen::exposed_actions::director::run::GoalBundleItem;
use peppygen::fixtures::exposed_actions::director::{next_run, run, trial};
use peppygen::fixtures::exposed_services::director::{
    forget_robot, judge_checks, judge_ending, runs, status, suspend,
};
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

/// The validation of a sweep of `home_workflow` that tested a belt speed from 0.05 to 0.25 m/s, whose thresholds a
/// trial meets only when each of its runs completes, with no unsafe event, false success or false check.
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
    bundle_of(&home_workflow(), lock, validation)
}

/// The files of the bundle of the workflow `text` with `lock`, signed off with `validation` when it gives one.
fn bundle_of(text: &str, lock: Lock, validation: Option<Validation>) -> Vec<GoalBundleItem> {
    let workflow = Workflow::parse(text).unwrap();
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

/// `home_workflow` with the assumption `belt_runs`, whose check is `check`.
fn home_checking(check: &str) -> String {
    let mut workflow: Value = serde_json::from_str(&home_workflow()).unwrap();
    workflow["assumptions"] = json!({ "belt_runs": { "says": "The belt runs", "check": check } });
    workflow.to_string()
}

/// The trial document of `runs` runs of `home_workflow` on `alpha`, in a work cell whose belt runs at 0.2 m/s.
fn trial_of_home(runs: u32) -> String {
    json!({
        "workflow_schema": "trial/v1",
        "manifest": { "name": "home_line", "revision": 1 },
        "workflow": { "name": "go_home", "revision": 1 },
        "roles": { "arm": ROBOT },
        "work_cell": { "belt_speed_mps": 0.2 },
        "runs": runs,
    })
    .to_string()
}

/// The goal of a trial of `runs` runs of the bundle `bundle`, which `trial_ends` and its result end.
async fn start_trial(
    rig: &Rig,
    runs: u32,
    bundle: Vec<GoalBundleItem>,
) -> peppygen::Result<trial::GoalHandle> {
    let request = trial::GoalRequestData {
        trial: trial_of_home(runs),
        bundle: bundle
            .into_iter()
            .map(
                |file| peppygen::exposed_actions::director::trial::GoalBundleItem {
                    path: file.path,
                    content: file.content,
                },
            )
            .collect(),
        started_by: Some("tests".to_owned()),
    };
    let goal = trial::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(goal.accepted, "{:?}", goal.refusal);
    Ok(goal)
}

/// The next feedback of the trial `goal` whose event is `event`.
async fn until_trial(
    goal: &mut trial::GoalHandle,
    event: &str,
) -> peppygen::Result<trial::Feedback> {
    loop {
        let feedback = goal.on_next_feedback().await?;
        if feedback.event == event {
            return Ok(feedback);
        }
    }
}

/// The goal of the next run of the trial `trial`, once its run waits for the judgement of its checks, with the run.
async fn next_run_judging(
    rig: &Rig,
    trial: &str,
) -> peppygen::Result<(next_run::GoalHandle, String)> {
    let request = next_run::GoalRequestData {
        trial: trial.to_owned(),
    };
    let mut goal =
        next_run::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(goal.accepted, "{:?}", goal.refusal);
    loop {
        let feedback = goal.on_next_feedback().await?;
        if feedback.event == "checks_to_judge" {
            return Ok((goal, feedback.run));
        }
    }
}

/// The outcome that `judge_checks` gives for `wrong` in the trial `trial`, or the kind of its refusal.
async fn judge_checks_of(rig: &Rig, trial: &str, wrong: &[&str]) -> Result<String, String> {
    let request = judge_checks::RequestData {
        trial: trial.to_owned(),
        wrong: wrong.iter().map(|name| (*name).to_owned()).collect(),
    };
    match judge_checks::poll(&rig.harness, &request, WIRE).await {
        Ok(response) => Ok(response.outcome),
        Err(error) => Err(judge_checks::Failure::of(&error)
            .map(|failure| failure.kind.to_string())
            .unwrap_or_else(|| error.to_string())),
    }
}

/// The answer of `judge_ending` for a run of the trial `trial` whose task is done as `task_done`.
async fn judge_ending_of(
    rig: &Rig,
    trial: &str,
    task_done: bool,
) -> peppygen::Result<judge_ending::Response> {
    let request = judge_ending::RequestData {
        trial: trial.to_owned(),
        task_done,
        emergency_stop: false,
        unsafe_events: Vec::new(),
    };
    judge_ending::poll(&rig.harness, &request, WIRE).await
}

/// The report of the trial that `goal` gave at its end, and how the goal ended.
async fn trial_ended(
    mut goal: trial::GoalHandle,
) -> peppygen::Result<(Value, trial::ResultOutcome)> {
    let ends = until_trial(&mut goal, "trial_ends").await?;
    let ended = goal.get_result(WIRE).await?;
    Ok((serde_json::from_str(&ends.details).unwrap(), ended.outcome))
}

/// A rig over a physical `alpha`, which the director lists.
async fn physical_rig() -> peppygen::Result<Rig> {
    let rig = Rig::start(Home::AtOnce).await?;
    rig.fleet.state.send_modify(|state| state.physical = true);
    listed_robots(&rig).await?;
    Ok(rig)
}

/// A trial takes the lease of its robot, opens it while the operator prepares the work cell, and closes it at
/// `next_run`; its run gives each check to judge, runs once the operator judges that it starts, and counts by the
/// judgement of its ending. After its last run, the trial signs off its bundle: its goal completes with its report,
/// the director keeps a sign-off record, and an unattended run of the bundle on that robot then runs. The runs of
/// the director list the trial, its run has the trial as its `started_by`, and `status` gives its report: without
/// the journals while the trial runs, with them once it ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_trial_whose_runs_meet_the_thresholds_signs_off_its_bundle() -> peppygen::Result<()> {
    let rig = physical_rig().await?;
    let bundle = bundle_of(
        &home_checking("true"),
        lock_at(ARM_NODE.2),
        Some(belt_speed_tested()),
    );
    let mut goal = start_trial(&rig, 1, bundle.clone()).await?;
    let starts = goal.on_next_feedback().await?;
    assert_eq!(
        (starts.event.as_str(), starts.trial.as_str()),
        ("trial_starts", "t1")
    );
    let prepares = until_trial(&mut goal, "prepares").await?;
    assert_eq!(
        serde_json::from_str::<Value>(&prepares.details).unwrap()["number"],
        1
    );
    assert_eq!(
        judge_checks_of(&rig, "t1", &[]).await,
        Err("precondition_failed".to_owned())
    );
    let (mut run_goal, run) = next_run_judging(&rig, "t1").await?;
    assert_eq!(
        judge_checks_of(&rig, "t1", &["no_such_check"]).await,
        Err("rejected".to_owned())
    );
    let request = suspend::RequestData { run: run.clone() };
    let refused = suspend::poll(&rig.harness, &request, WIRE)
        .await
        .unwrap_err();
    let failure = suspend::Failure::of(&refused).expect("a refusal of suspend");
    assert!(
        failure.message.contains("a run of the trial `t1`"),
        "{failure:?}"
    );
    assert_eq!(
        judge_checks_of(&rig, "t1", &[]).await,
        Ok("starts".to_owned())
    );
    while run_goal.on_next_feedback().await.is_ok() {}
    let ran = run_goal.get_result(WIRE).await?;
    assert!(
        matches!(ran.outcome, next_run::ResultOutcome::Completed(_)),
        "{:?}",
        ran.outcome
    );
    let running = status::RequestData {
        run: "t1".to_owned(),
    };
    let running = status::poll(&rig.harness, &running, WIRE).await?;
    assert_eq!(running.state, "running");
    let so_far: Value = serde_json::from_str(&running.details).unwrap();
    assert_eq!(so_far["runs"][0]["run"], run.as_str(), "{so_far}");
    assert!(so_far["runs"][0].get("journal").is_none(), "{so_far}");
    let counted = judge_ending_of(&rig, "t1", true).await?;
    assert_eq!((counted.runs_counted, counted.goes_on), (1, false));
    let (report, outcome) = trial_ended(goal).await?;
    assert!(
        matches!(outcome, trial::ResultOutcome::Completed(_)),
        "{outcome:?}"
    );
    assert_eq!(report["signs_off"], true, "{report}");
    assert_eq!(report["sign_off"]["trial"], "t1");
    assert_eq!(report["runs"][0]["count"], "completed");
    assert!(
        report["runs"][0]["journal"]
            .as_array()
            .is_some_and(|journal| !journal.is_empty())
    );
    let calls = rig.fleet.calls();
    let order: Vec<&str> = calls
        .iter()
        .map(String::as_str)
        .filter(|call| {
            [
                "take alpha trial t1",
                "open alpha",
                "close alpha",
                "release alpha",
            ]
            .contains(call)
        })
        .collect();
    assert_eq!(
        order,
        [
            "take alpha trial t1",
            "open alpha",
            "close alpha",
            "release alpha"
        ],
        "{calls:?}"
    );
    assert!(
        !calls.iter().any(|call| call.starts_with("take alpha run")),
        "a run of a trial takes no lease"
    );

    let document = unattended(json!({ "arm": ROBOT }), json!({ "belt_speed_mps": 0.2 }));
    let (admitted, _) = run_bundle(&rig, bundle, document).await?;
    assert_eq!(admitted["ending"], "completed", "{admitted}");
    let listed = runs::poll(&rig.harness, WIRE).await?;
    let trials: Vec<(&str, &str, u32)> = listed
        .trials
        .iter()
        .map(|trial| {
            (
                trial.trial.as_str(),
                trial.state.as_str(),
                trial.runs_counted,
            )
        })
        .collect();
    assert_eq!(trials, [("t1", "signed_off", 1)]);
    let of_the_trial = listed
        .runs
        .iter()
        .find(|item| item.run == run)
        .expect("the run of the trial");
    assert_eq!(of_the_trial.started_by, "t1");
    let request = status::RequestData {
        run: "t1".to_owned(),
    };
    let status = status::poll(&rig.harness, &request, WIRE).await?;
    assert_eq!(status.state, "signed_off");
    assert!(status.details.contains("\"journal\""), "{}", status.details);
    rig.harness.shutdown().await
}

/// The director refuses a trial before its first run, naming each cause: a simulated robot, a bundle with no
/// envelope, a stack that differs from the lock, a robot that another client leases, and a halted stack. A refused
/// trial takes no lease it keeps.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_trial_is_refused_naming_each_cause() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    listed_robots(&rig).await?;
    let signed_off = || bundle_of_home(lock_at(ARM_NODE.2), Some(belt_speed_tested()));
    let refusal = |outcome: trial::ResultOutcome| match outcome {
        trial::ResultOutcome::Failed(failure) => {
            assert_eq!(failure.kind.to_string(), "trial_refused", "{failure:?}");
            failure.message
        }
        other => panic!("{other:?}"),
    };
    let (_, simulated) = trial_ended(start_trial(&rig, 1, signed_off()).await?).await?;
    assert!(
        refusal(simulated).contains("`alpha` is simulated; a trial binds only physical robots")
    );
    rig.fleet.state.send_modify(|state| state.physical = true);
    let no_envelope = bundle_of_home(lock_at(ARM_NODE.2), None);
    let (_, outcome) = trial_ended(start_trial(&rig, 1, no_envelope).await?).await?;
    assert!(refusal(outcome).contains("the bundle has no envelope"));
    let other_commit = bundle_of_home(lock_at("7654321"), Some(belt_speed_tested()));
    let (_, outcome) = trial_ended(start_trial(&rig, 1, other_commit).await?).await?;
    assert!(
        refusal(outcome).starts_with("the stack differs from the lock of the bundle in 1 ways")
    );
    rig.fleet.state.send_modify(|state| {
        state.leases.push(common::Lease {
            robot: ROBOT.to_owned(),
            holder: "headset".to_owned(),
            holder_core_node: "elsewhere".to_owned(),
            name: "teleop".to_owned(),
            open: false,
        })
    });
    let (report, outcome) = trial_ended(start_trial(&rig, 1, signed_off()).await?).await?;
    let leased = refusal(outcome);
    assert!(
        leased.starts_with("the lease of `alpha` was refused (leased)"),
        "{leased}"
    );
    assert_eq!(report["state"], "refused");
    rig.fleet.state.send_modify(|state| state.leases.clear());
    rig.fleet.halt();
    let (_, outcome) = trial_ended(start_trial(&rig, 1, signed_off()).await?).await?;
    assert!(refusal(outcome).starts_with("the stack is halted"));
    rig.harness.shutdown().await
}

/// Runs the trial `trial` of one run of `bundle` to its sign-off: the operator judges that its run starts and that
/// its task is done. Gives the report.
async fn sign_off_in_one_run(
    rig: &Rig,
    trial: &str,
    bundle: Vec<GoalBundleItem>,
) -> peppygen::Result<Value> {
    let mut goal = start_trial(rig, 1, bundle).await?;
    until_trial(&mut goal, "prepares").await?;
    let (mut run_goal, _) = next_run_judging(rig, trial).await?;
    assert_eq!(
        judge_checks_of(rig, trial, &[]).await,
        Ok("starts".to_owned())
    );
    while run_goal.on_next_feedback().await.is_ok() {}
    judge_ending_of(rig, trial, true).await?;
    let (report, outcome) = trial_ended(goal).await?;
    assert!(
        matches!(outcome, trial::ResultOutcome::Completed(_)),
        "{outcome:?}"
    );
    Ok(report)
}

/// The kind and the message of the failure that ended the trial goal with `outcome`.
fn failed(outcome: trial::ResultOutcome) -> (String, String) {
    match outcome {
        trial::ResultOutcome::Failed(failure) => (failure.kind.to_string(), failure.message),
        other => panic!("{other:?}"),
    }
}

/// A trial removes, at its start, the sign-off record of its bundle and binding. A cancel of `next_run` aborts its
/// run, which counts as not completed; the trial then ends, as the runs left cannot meet the thresholds, and signs
/// off nothing, so an unattended run of the bundle is refused after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_trial_that_ends_early_signs_off_nothing_and_leaves_no_record() -> peppygen::Result<()> {
    let rig = physical_rig().await?;
    let bundle = bundle_of(
        &home_checking("true"),
        lock_at(ARM_NODE.2),
        Some(belt_speed_tested()),
    );
    sign_off_in_one_run(&rig, "t1", bundle.clone()).await?;
    let mut goal = start_trial(&rig, 2, bundle.clone()).await?;
    until_trial(&mut goal, "prepares").await?;
    let (run_goal, run) = next_run_judging(&rig, "t2").await?;
    run_goal.cancel_goal(WIRE).await?;
    let aborted = run_goal.get_result(WIRE).await?;
    assert!(
        matches!(aborted.outcome, next_run::ResultOutcome::Cancelled { .. }),
        "{:?}",
        aborted.outcome
    );
    let counted = until_trial(&mut goal, "run_counted").await?;
    assert_eq!(counted.run, run);
    assert_eq!(
        serde_json::from_str::<Value>(&counted.details).unwrap()["count"],
        "not_completed"
    );
    let (report, outcome) = trial_ended(goal).await?;
    let (kind, message) = failed(outcome);
    assert_eq!(kind, "trial_not_signed_off");
    assert!(
        message.contains("cannot meet the thresholds of the bundle any more"),
        "{message}"
    );
    assert_eq!(report["signs_off"], false);
    assert!(report["sign_off"].is_null(), "{report}");
    let document = unattended(json!({ "arm": ROBOT }), json!({ "belt_speed_mps": 0.2 }));
    let (refused, _) = run_bundle(&rig, bundle, document).await?;
    assert!(
        refusal_of(&refused).starts_with("no trial signed off the bundle"),
        "{refused}"
    );
    rig.harness.shutdown().await
}

/// A run whose work cell the operator judges not ready does not count, and the trial waits for the next run; a
/// check that the operator names wrong counts the run as not completed and the check as false.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_not_ready_does_not_count_and_a_wrong_check_counts_as_false() -> peppygen::Result<()>
{
    let rig = physical_rig().await?;
    let bundle = bundle_of(
        &home_checking("false"),
        lock_at(ARM_NODE.2),
        Some(belt_speed_tested()),
    );
    let mut goal = start_trial(&rig, 3, bundle).await?;
    until_trial(&mut goal, "prepares").await?;
    let (not_ready, _) = next_run_judging(&rig, "t1").await?;
    assert_eq!(
        judge_checks_of(&rig, "t1", &[]).await,
        Ok("not_ready".to_owned())
    );
    let refused = not_ready.get_result(WIRE).await?;
    match refused.outcome {
        next_run::ResultOutcome::Failed(failure) => {
            assert_eq!(failure.kind.to_string(), "run_refused")
        }
        other => panic!("{other:?}"),
    }
    let not_counted = until_trial(&mut goal, "run_not_counted").await?;
    assert_eq!(
        serde_json::from_str::<Value>(&not_counted.details).unwrap()["measures"]["runs"],
        0
    );
    let prepares = until_trial(&mut goal, "prepares").await?;
    assert_eq!(
        serde_json::from_str::<Value>(&prepares.details).unwrap()["number"],
        1
    );
    let (_wrong, _) = next_run_judging(&rig, "t1").await?;
    assert_eq!(
        judge_checks_of(&rig, "t1", &["belt_runs"]).await,
        Ok("false_check".to_owned())
    );
    let counted = until_trial(&mut goal, "run_counted").await?;
    let measures = &serde_json::from_str::<Value>(&counted.details).unwrap()["measures"];
    assert_eq!(
        (
            measures["runs"].as_u64(),
            measures["false_assumption_checks"].as_u64()
        ),
        (Some(1), Some(1))
    );
    let (_, outcome) = trial_ended(goal).await?;
    assert_eq!(failed(outcome).0, "trial_not_signed_off");
    rig.harness.shutdown().await
}

/// A next run that does not start, as when the journal of its run does not open, fails with `attempt_failed` and
/// opens the lease of the trial again, so the operator can go on preparing the work cell; the next `next_run` then
/// starts the run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_next_run_that_does_not_start_opens_the_leases_again() -> peppygen::Result<()> {
    let rig = physical_rig().await?;
    let bundle = bundle_of_home(lock_at(ARM_NODE.2), Some(belt_speed_tested()));
    let mut goal = start_trial(&rig, 1, bundle).await?;
    until_trial(&mut goal, "prepares").await?;
    // A directory where the journals keep the number of the next run: the journal of the run cannot open.
    let next_run_number = rig
        .journals
        .path()
        .join("workflow_director")
        .join("journals")
        .join("next_run");
    let _ = std::fs::remove_file(&next_run_number);
    std::fs::create_dir(&next_run_number).unwrap();
    let request = next_run::GoalRequestData {
        trial: "t1".to_owned(),
    };
    let not_started =
        next_run::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    match not_started.get_result(WIRE).await?.outcome {
        next_run::ResultOutcome::Failed(failure) => {
            assert_eq!(failure.kind.to_string(), "attempt_failed");
            assert!(
                failure
                    .message
                    .starts_with("the journal of the run did not open"),
                "{failure:?}"
            );
        }
        other => panic!("{other:?}"),
    }
    let leases = |calls: Vec<String>| -> Vec<String> {
        calls
            .into_iter()
            .filter(|call| [format!("open {ROBOT}"), format!("close {ROBOT}")].contains(call))
            .collect()
    };
    assert_eq!(
        leases(rig.fleet.calls()),
        [
            format!("open {ROBOT}"),
            format!("close {ROBOT}"),
            format!("open {ROBOT}")
        ]
    );
    std::fs::remove_dir(&next_run_number).unwrap();
    let (_run_goal, run) = next_run_judging(&rig, "t1").await?;
    assert!(!run.is_empty());
    goal.cancel_goal(WIRE).await?;
    let (report, _) = trial_ended(goal).await?;
    assert_eq!(report["state"], "aborted", "{report}");
    rig.harness.shutdown().await
}

/// A halt ends a trial that waits for its next run halted, signing off nothing, and refuses every new trial.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_halt_ends_a_running_trial_halted_and_refuses_a_new_one() -> peppygen::Result<()> {
    let rig = physical_rig().await?;
    let bundle = bundle_of_home(lock_at(ARM_NODE.2), Some(belt_speed_tested()));
    let mut goal = start_trial(&rig, 1, bundle.clone()).await?;
    until_trial(&mut goal, "prepares").await?;
    rig.fleet.halt();
    let (report, outcome) = trial_ended(goal).await?;
    assert_eq!(failed(outcome).0, "trial_halted");
    assert_eq!(report["state"], "halted");
    let (_, refused) = trial_ended(start_trial(&rig, 1, bundle).await?).await?;
    let (kind, message) = failed(refused);
    assert_eq!(kind, "trial_refused");
    assert!(message.starts_with("the stack is halted"), "{message}");
    rig.harness.shutdown().await
}

/// A cancel of the goal of a trial aborts it: the goal ends cancelled with the report, which signs off nothing, and
/// the trial gives its lease back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cancel_of_a_trial_aborts_it_and_it_signs_off_nothing() -> peppygen::Result<()> {
    let rig = physical_rig().await?;
    let bundle = bundle_of_home(lock_at(ARM_NODE.2), Some(belt_speed_tested()));
    let mut goal = start_trial(&rig, 1, bundle).await?;
    until_trial(&mut goal, "prepares").await?;
    goal.cancel_goal(WIRE).await?;
    let (report, outcome) = trial_ended(goal).await?;
    assert!(
        matches!(outcome, trial::ResultOutcome::Cancelled { .. }),
        "{outcome:?}"
    );
    assert_eq!(
        (report["state"].as_str(), report["signs_off"].as_bool()),
        (Some("aborted"), Some(false))
    );
    assert!(rig.fleet.calls().contains(&format!("release {ROBOT}")));
    rig.harness.shutdown().await
}

/// A restart of the director ends a running trial interrupted: the director that starts again lists it so, and
/// its report signs off nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restart_of_the_director_ends_a_running_trial_interrupted() -> peppygen::Result<()> {
    let rig = physical_rig().await?;
    let bundle = bundle_of_home(lock_at(ARM_NODE.2), Some(belt_speed_tested()));
    let mut goal = start_trial(&rig, 1, bundle).await?;
    until_trial(&mut goal, "prepares").await?;
    let Rig {
        harness, journals, ..
    } = rig;
    harness.shutdown().await?;
    let rig = Rig::start_on(Home::AtOnce, &[], journals).await?;
    let listed = runs::poll(&rig.harness, WIRE).await?;
    assert_eq!(listed.trials.len(), 1);
    assert_eq!(
        (
            listed.trials[0].trial.as_str(),
            listed.trials[0].state.as_str()
        ),
        ("t1", "interrupted")
    );
    let request = status::RequestData {
        run: "t1".to_owned(),
    };
    let status = status::poll(&rig.harness, &request, WIRE).await?;
    assert_eq!(status.state, "interrupted");
    let report: Value = serde_json::from_str(&status.details).unwrap();
    assert_eq!(report["signs_off"], false);
    rig.harness.shutdown().await
}
