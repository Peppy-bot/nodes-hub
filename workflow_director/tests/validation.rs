//! The director over the wire, against a scripted fleet and a scripted arm, for the validation of workflows: the
//! work cell of a run and the values that no test covered.

use peppygen::DEFAULT_FEEDBACK_BOUND;
use peppygen::fixtures::exposed_actions::director::run;
use serde_json::{Value, json};

mod common;
use common::{Home, Rig, WIRE, home_workflow, listed_robots, on_alpha};

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
