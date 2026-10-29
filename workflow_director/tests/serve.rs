//! The director over the wire, against a scripted fleet and a scripted arm: the robots it lists, and a run of a
//! workflow from its goal to its end, with the calls it makes of the fleet and of the robot in order.

use std::sync::Arc;
use std::time::Duration;

use peppygen::DEFAULT_FEEDBACK_BOUND;
use peppygen::fixtures::exposed_actions::director::{abort, locations, resume, run};
use peppygen::fixtures::exposed_services::director::{confirm, robots, runs, status, suspend};
use serde_json::{Value, json};

mod common;
use common::{ARM_INSTANCE, ARM_PIN, Home, Lease, ROBOT, Rig, WIRE, home_workflow, on_alpha};

/// How often a test reads the director again while it waits for the director to see the fleet.
const SETTLE: Duration = Duration::from_millis(50);

/// The robots of the director, once it lists `alpha`: before the first feedback of the fleet, `robots` answers
/// `resource_unavailable`.
async fn listed_robots(rig: &Rig) -> peppygen::Result<robots::Response> {
    loop {
        match robots::poll(&rig.harness, WIRE).await {
            Ok(listed) if !listed.robots.is_empty() => return Ok(listed),
            Ok(_) => {}
            Err(error)
                if robots::Failure::of(&error)
                    .is_some_and(|failure| failure.kind.to_string() == "resource_unavailable") => {}
            Err(error) => return Err(error),
        }
        tokio::time::sleep(SETTLE).await;
    }
}

/// The goal of a run of `home_workflow` on `alpha`, once the director lists `alpha`.
async fn run_home(rig: &Rig) -> peppygen::Result<run::GoalHandle> {
    run_on_alpha(rig, home_workflow()).await
}

/// `home_workflow` whose lane goes to the location `bin` with `arm.go_to`.
fn go_to_workflow() -> String {
    let mut workflow: Value = serde_json::from_str(&home_workflow()).expect("the workflow is JSON");
    workflow["locations"] = json!(["bin"]);
    workflow["actions"]["arm"]["tree"] = json!({ "id": "go", "timeout_s": 5, "call": { "member": "arm.go_to", "args": { "place": { "$location": "bin" } } } });
    workflow.to_string()
}

/// A `locations` goal of `alpha` that teaches `taught`, or lists with `None`: the places each skill knows after.
async fn locations_of(
    rig: &Rig,
    taught: Option<&[&str]>,
) -> peppygen::Result<locations::GoalHandle> {
    let teach = taught.map(|places| {
        places
            .iter()
            .map(
                |place| peppygen::exposed_actions::director::locations::GoalTeachItem {
                    name: (*place).to_owned(),
                    kind: "surface".to_owned(),
                    position: [0.0; 3],
                    orientation: [1.0, 0.0, 0.0, 0.0],
                    size: [0.0; 3],
                },
            )
            .collect()
    });
    let request = locations::GoalRequestData {
        robot: ROBOT.to_owned(),
        teach,
    };
    locations::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await
}

/// `home_workflow` with the assumption `clear`, which has no check, so that a run of it waits for its confirmation.
fn home_workflow_to_confirm() -> String {
    let mut workflow: Value = serde_json::from_str(&home_workflow()).expect("the workflow is JSON");
    workflow["assumptions"] = json!({ "clear": { "says": "The table is clear" } });
    workflow.to_string()
}

/// The goal of a run of `workflow` on `alpha`, once the director lists `alpha`.
async fn run_on_alpha(rig: &Rig, workflow: String) -> peppygen::Result<run::GoalHandle> {
    listed_robots(rig).await?;
    let request = run::GoalRequestData {
        workflow: Some(workflow),
        bundle: None,
        run_document: None,
        roles: Some(on_alpha()),
        started_by: Some("tests".to_owned()),
        caller_data: None,
    };
    let goal = run::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(goal.accepted, "{:?}", goal.refusal);
    Ok(goal)
}

/// Reads the feedback of the run `goal` up to its `to_confirm`.
async fn until_it_waits_for_its_confirmation(goal: &mut run::GoalHandle) -> peppygen::Result<()> {
    while goal.on_next_feedback().await?.event != "to_confirm" {}
    Ok(())
}

/// The failure kind that `confirm` gives for `assumption` of `run`, or `None` when the director admits it.
async fn kind_of_confirm(rig: &Rig, run: &str, assumption: &str) -> Option<String> {
    let request = confirm::RequestData {
        run: run.to_owned(),
        assumption: assumption.to_owned(),
    };
    let error = confirm::poll(&rig.harness, &request, WIRE).await.err()?;
    let failure = confirm::Failure::of(&error).unwrap_or_else(|| panic!("{error}"));
    Some(failure.kind.to_string())
}

/// The failure kind that `suspend` gives for `run`, or `None` when the director admits it.
async fn kind_of_suspend(rig: &Rig, run: &str) -> Option<String> {
    let request = suspend::RequestData {
        run: run.to_owned(),
    };
    let error = suspend::poll(&rig.harness, &request, WIRE).await.err()?;
    let failure = suspend::Failure::of(&error).unwrap_or_else(|| panic!("{error}"));
    Some(failure.kind.to_string())
}

/// The events of the feedback of the run `goal` up to the end of its stream.
async fn feedback_of(goal: &mut run::GoalHandle) -> Vec<String> {
    let mut events = Vec::new();
    while let Ok(feedback) = goal.on_next_feedback().await {
        events.push(feedback.event);
    }
    events
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_robots_are_the_robots_of_the_listing_with_the_allowed_contracts_and_their_documents()
-> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    let listed = listed_robots(&rig).await?;
    let [alpha] = &listed.robots[..] else {
        panic!("{:?}", listed.robots)
    };
    assert_eq!(
        (
            alpha.name.as_str(),
            alpha.kind.as_str(),
            alpha.lease_holder.as_str()
        ),
        (ROBOT, "simulated", "")
    );
    let contracts: Vec<(&str, &str)> = alpha
        .contracts
        .iter()
        .map(|contract| (contract.name.as_str(), contract.sha256.as_str()))
        .collect();
    assert_eq!(contracts, [("test_arm", ARM_PIN)]);
    assert_eq!(
        listed
            .documents
            .iter()
            .map(|document| document.sha256.as_str())
            .collect::<Vec<_>>(),
        [ARM_PIN]
    );
    assert!(!listed.halted);
    rig.harness.shutdown().await
}

/// A run takes the lease of its robot under its name, asks the fleet before the move, moves the robot, gives the
/// lease back, then sends `run_ends`, and only then completes its goal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_takes_its_lease_moves_after_the_move_check_and_gives_the_lease_back_before_it_ends()
-> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    let mut goal = run_home(&rig).await?;
    let events = feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Completed(result) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(result.run, "r1");
    assert!(
        result.result.contains("\"ending\":\"completed\""),
        "{}",
        result.result
    );
    let moving: Vec<&String> = events.iter().filter(|event| *event != "progress").collect();
    assert_eq!(
        moving.first().map(|event| event.as_str()),
        Some("run_starts")
    );
    assert_eq!(
        moving.last().map(|event| event.as_str()),
        Some("run_ends"),
        "{events:?}"
    );
    let calls = rig.fleet.calls();
    let take = format!("take {ROBOT} run r1");
    let order: Vec<&str> = calls
        .iter()
        .map(String::as_str)
        .filter(|call| [take.as_str(), "check_move alpha_arm", "release alpha"].contains(call))
        .collect();
    assert_eq!(
        order,
        [take.as_str(), "check_move alpha_arm", "release alpha"],
        "{calls:?}"
    );
    assert_eq!(*rig.arm.goals.lock().unwrap(), 1);
    assert!(rig.fleet.state.borrow().leases.is_empty());
    rig.harness.shutdown().await
}

/// A halt during a move ends the run halted: the director sends the robot nothing more, and the goal fails with
/// `run_halted`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_halt_during_a_move_ends_the_run_halted() -> peppygen::Result<()> {
    let rig = Rig::start(Home::Never).await?;
    let mut goal = run_home(&rig).await?;
    rig.arm.arrived.notified().await;
    rig.fleet.halt();
    feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Failed(failure) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(failure.kind.to_string(), "run_halted");
    assert_eq!(
        *rig.arm.goals.lock().unwrap(),
        1,
        "no move after the halt, the safe state included"
    );
    rig.harness.shutdown().await
}

/// An abort of a run that moves its robot ends it aborted: the abort answers with the result, and the goal of the
/// run ends cancelled with the same result; the lease goes back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_abort_ends_a_moving_run_aborted_and_gives_its_lease_back() -> peppygen::Result<()> {
    let rig = Rig::start(Home::Never).await?;
    let goal = run_home(&rig).await?;
    rig.arm.arrived.notified().await;
    let request = abort::GoalRequestData {
        run: "r1".to_owned(),
    };
    let aborting = abort::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(aborting.accepted, "{:?}", aborting.refusal);
    let aborted = aborting.get_result(WIRE).await?;
    let abort::ResultOutcome::Completed(result) = aborted.outcome else {
        panic!("{:?}", aborted.outcome)
    };
    assert!(
        result.result.contains("\"ending\":\"aborted\""),
        "{}",
        result.result
    );
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Cancelled { reason, .. } = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(reason, "aborted");
    assert!(rig.fleet.calls().contains(&format!("release {ROBOT}")));
    rig.harness.shutdown().await
}

/// A resume of a run that the director does not keep is refused with `rejected`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_resume_of_an_unknown_run_is_refused() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    listed_robots(&rig).await?;
    let request = resume::GoalRequestData {
        run: "r9".to_owned(),
    };
    let refused = resume::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(!refused.accepted);
    assert_eq!(
        refused
            .refusal
            .map(|failure| failure.kind.to_string())
            .as_deref(),
        Some("rejected")
    );
    rig.harness.shutdown().await
}

/// A stop of the robot during a step suspends the attended run: the lane runs its safe state, the lease opens for a
/// person, and the goal fails with `run_suspended`. A resume closes the lease, runs the step again and completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stopped_step_suspends_the_run_and_a_resume_closes_the_lease_and_runs_it_again()
-> peppygen::Result<()> {
    let rig = Rig::start(Home::StoppedFirst).await?;
    let mut goal = run_home(&rig).await?;
    let events = feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Failed(failure) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(failure.kind.to_string(), "run_suspended");
    for event in ["incident", "lane_suspends", "run_suspends"] {
        assert!(
            events.iter().any(|seen| seen == event),
            "{event}: {events:?}"
        );
    }
    assert_eq!(
        *rig.arm.goals.lock().unwrap(),
        2,
        "the step, then the safe state"
    );
    assert_eq!(
        rig.fleet
            .state
            .borrow()
            .leases
            .first()
            .map(|lease| lease.open),
        Some(true)
    );
    let request = resume::GoalRequestData {
        run: "r1".to_owned(),
    };
    let mut resuming =
        resume::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(resuming.accepted, "{:?}", resuming.refusal);
    let mut resumed = Vec::new();
    while let Ok(feedback) = resuming.on_next_feedback().await {
        resumed.push(feedback.event);
    }
    let ended = resuming.get_result(WIRE).await?;
    let resume::ResultOutcome::Completed(result) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert!(
        result.result.contains("\"ending\":\"completed\""),
        "{}",
        result.result
    );
    assert_eq!(
        resumed.first().map(String::as_str),
        Some("lane_resumes"),
        "{resumed:?}"
    );
    let calls = rig.fleet.calls();
    let lease_calls: Vec<&str> = calls
        .iter()
        .map(String::as_str)
        .filter(|call| call.starts_with("open") || call.starts_with("close"))
        .collect();
    assert_eq!(lease_calls, ["open alpha", "close alpha"]);
    assert_eq!(*rig.arm.goals.lock().unwrap(), 3);
    assert!(rig.fleet.state.borrow().leases.is_empty());
    rig.harness.shutdown().await
}

/// A run that names a robot that another client leases and did not offer is refused before any take, naming the
/// holder.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_on_a_robot_that_another_client_leases_is_refused_naming_the_holder()
-> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    rig.fleet.state.send_modify(|state| {
        state.leases.push(Lease {
            robot: ROBOT.to_owned(),
            holder: "headset_inst".to_owned(),
            holder_core_node: "cn-desk".to_owned(),
            name: "teleop".to_owned(),
            open: false,
        })
    });
    let mut goal = run_home(&rig).await?;
    feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Failed(failure) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(failure.kind.to_string(), "run_refused");
    assert!(
        failure.message.contains("headset_inst"),
        "{}",
        failure.message
    );
    assert!(
        !rig.fleet
            .calls()
            .iter()
            .any(|call| call.starts_with("take")),
        "{:?}",
        rig.fleet.calls()
    );
    assert_eq!(*rig.arm.goals.lock().unwrap(), 0);
    rig.harness.shutdown().await
}

/// A workflow that reads as JSON5 but breaks its schema starts a run, which its check refuses: the run's first
/// event is run_starts, with no run document, and its end is run_refused, with a run result that names no workflow.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_whose_workflow_breaks_its_schema_starts_and_is_refused() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    listed_robots(&rig).await?;
    let request = run::GoalRequestData {
        workflow: Some("{ workflow_schema: \"workflow/v1\" }".to_owned()),
        bundle: None,
        run_document: None,
        roles: Some(on_alpha()),
        started_by: Some("tests".to_owned()),
        caller_data: None,
    };
    let mut goal = run::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(goal.accepted, "{:?}", goal.refusal);
    let first = goal.on_next_feedback().await?;
    assert_eq!(first.event, "run_starts");
    assert!(first.run.starts_with('r'), "{}", first.run);
    assert_eq!(first.details, "null");
    let mut last = first;
    while let Ok(feedback) = goal.on_next_feedback().await {
        last = feedback;
    }
    assert_eq!(last.event, "run_ends");
    let result: Value = serde_json::from_str(&last.details).expect("the run result is JSON");
    assert_eq!(result["ending"], "refused");
    assert!(
        result.get("workflow").is_none(),
        "a workflow that does not read names no workflow: {result}"
    );
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Failed(failure) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(failure.kind.to_string(), "run_refused");
    rig.harness.shutdown().await
}

/// The registry gives a run ended before its goal ends: a run of the same robot right after the end of the first
/// binds it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_right_after_the_end_of_another_binds_its_robot() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    for run in ["r1", "r2"] {
        let mut goal = run_home(&rig).await?;
        feedback_of(&mut goal).await;
        let ended = goal.get_result(WIRE).await?;
        let run::ResultOutcome::Completed(result) = ended.outcome else {
            panic!("{run}: {:?}", ended.outcome)
        };
        assert_eq!(result.run, run);
    }
    rig.harness.shutdown().await
}

/// A director that starts again over the journals of its runs keeps those runs: `runs` and `status` give the run that
/// ended before the restart as it ended, and the next run takes the next id.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_director_that_starts_again_keeps_the_runs_of_its_journals() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    let mut goal = run_home(&rig).await?;
    feedback_of(&mut goal).await;
    goal.get_result(WIRE).await?;
    let Rig {
        harness, journals, ..
    } = rig;
    harness.shutdown().await?;

    let rig = Rig::start_on(Home::AtOnce, &[], journals).await?;
    let listed = runs::poll(&rig.harness, WIRE).await?;
    let kept: Vec<(&str, &str)> = listed
        .runs
        .iter()
        .map(|run| (run.run.as_str(), run.state.as_str()))
        .collect();
    assert_eq!(kept, [("r1", "completed")]);
    let ended = status::poll(
        &rig.harness,
        &status::RequestData {
            run: "r1".to_owned(),
        },
        WIRE,
    )
    .await?;
    assert_eq!(ended.state, "completed");
    assert!(
        ended.details.contains("\"ending\":\"completed\""),
        "{}",
        ended.details
    );

    let mut goal = run_home(&rig).await?;
    feedback_of(&mut goal).await;
    let next = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Completed(result) = next.outcome else {
        panic!("{:?}", next.outcome)
    };
    assert_eq!(result.run, "r2", "run ids go on across a restart");
    rig.harness.shutdown().await
}

/// A resume that the director refuses, here as the lease of the robot does not close, fails `run_suspended`: its
/// caller reads the suspension that names the refusal as its last progress, and the run stays suspended with it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_resume_gives_its_caller_the_suspension_that_names_the_refusal()
-> peppygen::Result<()> {
    let rig = Rig::start(Home::StoppedFirst).await?;
    let mut goal = run_home(&rig).await?;
    feedback_of(&mut goal).await;
    goal.get_result(WIRE).await?;
    rig.fleet
        .state
        .send_modify(|state| state.close_refused = true);

    let request = resume::GoalRequestData {
        run: "r1".to_owned(),
    };
    let mut resuming =
        resume::send_goal(&rig.harness, &request, DEFAULT_FEEDBACK_BOUND, WIRE).await?;
    assert!(resuming.accepted, "{:?}", resuming.refusal);
    let mut feedback = Vec::new();
    while let Ok(progress) = resuming.on_next_feedback().await {
        feedback.push(progress);
    }
    let ended = resuming.get_result(WIRE).await?;
    let resume::ResultOutcome::Failed(failure) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(failure.kind.to_string(), "run_suspended");
    let last = feedback.last().expect("the caller reads the suspension");
    assert_eq!(last.event, "run_suspends");
    assert!(last.details.contains("did not close"), "{}", last.details);
    let kept = status::poll(
        &rig.harness,
        &status::RequestData {
            run: "r1".to_owned(),
        },
        WIRE,
    )
    .await?;
    assert_eq!(kept.state, "suspended");
    assert!(kept.details.contains("did not close"), "{}", kept.details);
    rig.harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_end_of_the_lease_goal_shuts_the_director_down_with_a_fault() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    listed_robots(&rig).await?;
    assert!(rig.fault.end().is_ok());
    rig.fleet.end_lease_goal();
    rig.harness
        .node_runner()
        .cancellation_token()
        .cancelled()
        .await;
    let fault = rig.fault.end().expect_err("the director ends with a fault");
    assert!(
        fault
            .to_string()
            .contains("the lease goal of the director ended"),
        "{fault}"
    );
    rig.harness.shutdown().await
}

/// `confirm` admits the confirmation of an assumption that the run waits for, after which the run goes on. It refuses
/// an unknown run and an assumption that the workflow does not have with `rejected`, and a confirmation that the run
/// does not wait for with `precondition_failed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_confirm_admits_only_an_assumption_that_the_run_waits_for() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    let mut goal = run_on_alpha(&rig, home_workflow_to_confirm()).await?;
    until_it_waits_for_its_confirmation(&mut goal).await?;
    let rejected = Some("rejected".to_owned());
    assert_eq!(kind_of_confirm(&rig, "r9", "clear").await, rejected);
    assert_eq!(kind_of_confirm(&rig, "r1", "dry").await, rejected);
    assert_eq!(kind_of_confirm(&rig, "r1", "clear").await, None);
    let events = feedback_of(&mut goal).await;
    assert!(events.iter().any(|event| event == "run_ends"), "{events:?}");
    let ended = goal.get_result(WIRE).await?;
    assert!(
        matches!(ended.outcome, run::ResultOutcome::Completed(_)),
        "{:?}",
        ended.outcome
    );
    assert_eq!(
        kind_of_confirm(&rig, "r1", "clear").await,
        Some("precondition_failed".to_owned())
    );
    rig.harness.shutdown().await
}

/// `suspend` admits a run whose lanes run, which then suspends. It refuses an unknown run with `rejected` and a run
/// whose lanes do not run with `precondition_failed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_suspend_admits_only_a_run_whose_lanes_run() -> peppygen::Result<()> {
    let rig = Rig::start(Home::NeverFirst).await?;
    listed_robots(&rig).await?;
    assert_eq!(
        kind_of_suspend(&rig, "r9").await,
        Some("rejected".to_owned())
    );
    let mut goal = run_on_alpha(&rig, home_workflow_to_confirm()).await?;
    until_it_waits_for_its_confirmation(&mut goal).await?;
    assert_eq!(
        kind_of_suspend(&rig, "r1").await,
        Some("precondition_failed".to_owned())
    );
    assert_eq!(kind_of_confirm(&rig, "r1", "clear").await, None);
    rig.arm.arrived.notified().await;
    assert_eq!(kind_of_suspend(&rig, "r1").await, None);
    let events = feedback_of(&mut goal).await;
    assert!(
        events.iter().any(|event| event == "run_suspends"),
        "{events:?}"
    );
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Failed(failure) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(failure.kind.to_string(), "run_suspended");
    rig.harness.shutdown().await
}

/// A run gives a location to a skill by its name, once the skill knows it: the arm goes to `bin`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_gives_a_location_the_skill_knows_by_its_name() -> peppygen::Result<()> {
    let rig = Rig::knowing(Home::AtOnce, &["bin"]).await?;
    let mut goal = run_on_alpha(&rig, go_to_workflow()).await?;
    feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    assert!(
        matches!(ended.outcome, run::ResultOutcome::Completed(_)),
        "{:?}",
        ended.outcome
    );
    assert_eq!(*rig.arm.visited.lock().unwrap(), ["bin"]);
    rig.harness.shutdown().await
}

/// A run that gives a skill a location it does not know is refused before the assumptions, naming the skill and
/// the location, and gives back the lease it took. Its assumption reads a topic that never publishes, and a check of
/// it refuses a run naming the assumption: the refusal names the location.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_that_names_a_location_the_skill_does_not_know_is_refused_naming_both()
-> peppygen::Result<()> {
    let rig = Rig::knowing(Home::AtOnce, &["shelf"]).await?;
    let mut workflow: Value =
        serde_json::from_str(&go_to_workflow()).expect("the workflow is JSON");
    workflow["observations"] =
        json!({ "state": { "topic": "arm.state", "role": "arm", "max_age_s": 1 } });
    workflow["assumptions"] =
        json!({ "idle": { "says": "The arm is idle", "check": "!state.busy" } });
    let mut goal = run_on_alpha(&rig, workflow.to_string()).await?;
    feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Failed(failure) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(failure.kind.to_string(), "run_refused");
    assert!(
        failure.message.contains("knows no location `bin`")
            && failure.message.contains(ARM_INSTANCE)
            && failure.message.contains("locations"),
        "{}",
        failure.message
    );
    assert!(rig.arm.visited.lock().unwrap().is_empty());
    let calls = rig.fleet.calls();
    let take = format!("take {ROBOT} run r1");
    let leases: Vec<&str> = calls
        .iter()
        .map(String::as_str)
        .filter(|call| [take.as_str(), "release alpha"].contains(call))
        .collect();
    assert_eq!(leases, [take.as_str(), "release alpha"], "{calls:?}");
    rig.harness.shutdown().await
}

/// A run whose assumption reads a topic waits for the first message of the topic: the topic publishes every 0.5 s,
/// its first message after the subscription of the director, and the check reads it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_waits_for_the_first_message_of_a_topic_it_reads() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    let publishing = common::publish_state_every(&rig.harness, Duration::from_millis(500)).await?;
    let mut workflow: Value = serde_json::from_str(&home_workflow()).expect("the workflow is JSON");
    workflow["observations"] =
        json!({ "state": { "topic": "arm.state", "role": "arm", "max_age_s": 5 } });
    workflow["assumptions"] =
        json!({ "idle": { "says": "The arm is idle", "check": "!state.busy" } });
    let mut goal = run_on_alpha(&rig, workflow.to_string()).await?;
    feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    assert!(
        matches!(ended.outcome, run::ResultOutcome::Completed(_)),
        "{:?}",
        ended.outcome
    );
    publishing.abort();
    rig.harness.shutdown().await
}

/// A lease that ends between its take and the start of the run, with no event of it, stops the run before its
/// first step: the run ends interrupted, and the arm gets no goal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lease_that_ends_before_the_run_starts_interrupts_it() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    rig.fleet
        .state
        .send_modify(|state| state.lease_ends_at_take = true);
    let mut goal = run_home(&rig).await?;
    feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Failed(failure) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(failure.kind.to_string(), "run_interrupted");
    assert_eq!(*rig.arm.goals.lock().unwrap(), 0);
    rig.harness.shutdown().await
}

/// A director that starts again over the journal of a run that did not end, older than the runs it keeps, ends that
/// run interrupted and prunes its journal with its entry: `runs` gives only the kept runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_run_that_a_restart_ends_and_prunes_leaves_no_entry() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    for _ in 0..3 {
        let mut goal = run_home(&rig).await?;
        feedback_of(&mut goal).await;
        goal.get_result(WIRE).await?;
    }
    let Rig {
        harness, journals, ..
    } = rig;
    harness.shutdown().await?;
    let opened = json!({ "seq": 0, "run_ns": 0, "wall_ns": 0, "type": "opened", "body": { "workflow": home_workflow(), "started_by": "tests" } });
    std::fs::write(
        journals.path().join("workflow_director/journals/r0.jsonl"),
        format!("{opened}\n"),
    )
    .expect("the journal of r0 is written");

    let rig = Rig::start_keeping(Home::AtOnce, &[], journals, 3).await?;
    let listed = runs::poll(&rig.harness, WIRE).await?;
    let kept: Vec<&str> = listed.runs.iter().map(|run| run.run.as_str()).collect();
    assert_eq!(kept, ["r3", "r2", "r1"]);
    rig.harness.shutdown().await
}

/// `locations` teaches each locations:v1 skill of a robot and lists what it knows; a teach on a leased robot is
/// refused, naming the holder.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn locations_teaches_the_skills_of_a_robot_and_lists_what_they_know() -> peppygen::Result<()>
{
    let rig = Rig::knowing(Home::AtOnce, &["bin"]).await?;
    listed_robots(&rig).await?;
    let taught = locations_of(&rig, Some(&["shelf", "crate"])).await?;
    assert!(taught.accepted, "{:?}", taught.refusal);
    let ended = taught.get_result(WIRE).await?;
    let locations::ResultOutcome::Completed(result) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(result.skills.len(), 1);
    assert_eq!(result.skills[0].instance, ARM_INSTANCE);
    let names: Vec<&str> = result.skills[0]
        .locations
        .iter()
        .map(|location| location.name.as_str())
        .collect();
    assert_eq!(names, ["shelf", "crate"]);
    assert_eq!(*rig.arm.known.lock().unwrap(), ["shelf", "crate"]);
    assert_eq!(
        rig.fleet.calls(),
        [
            format!("take {ROBOT} locations"),
            format!("release {ROBOT}")
        ],
        "the teach holds the lease of the robot while it teaches"
    );
    let listed = locations_of(&rig, None).await?;
    let locations::ResultOutcome::Completed(result) = listed.get_result(WIRE).await?.outcome else {
        panic!("the list completes")
    };
    assert_eq!(result.skills[0].locations.len(), 2);

    rig.fleet.state.send_modify(|state| {
        state.leases.push(Lease {
            robot: ROBOT.to_owned(),
            holder: "headset_inst".to_owned(),
            holder_core_node: "cn-desk".to_owned(),
            name: "teleop".to_owned(),
            open: false,
        })
    });
    loop {
        let listed = robots::poll(&rig.harness, WIRE).await?;
        if listed
            .robots
            .iter()
            .any(|robot| robot.lease_holder == "headset_inst")
        {
            break;
        }
        tokio::time::sleep(SETTLE).await;
    }
    let refused = locations_of(&rig, Some(&["bench"])).await?;
    let refusal = refused
        .refusal
        .expect("a teach on a leased robot is refused");
    assert_eq!(refusal.kind.to_string(), "precondition_failed");
    assert!(
        refusal.message.contains("headset_inst"),
        "{}",
        refusal.message
    );
    assert_eq!(*rig.arm.known.lock().unwrap(), ["shelf", "crate"]);
    rig.harness.shutdown().await
}

/// A service call that the robot answers with an error reply fails its step with the kind and the message of the
/// reply.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_service_call_answered_by_an_error_reply_fails_its_step_with_its_kind()
-> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    *rig.arm.find_fails.lock().unwrap() = Some((
        "precondition_failed".to_owned(),
        "the camera is covered".to_owned(),
    ));
    rig.arm.finds.add_permits(1);
    let mut workflow: Value = serde_json::from_str(&home_workflow()).expect("the workflow is JSON");
    workflow["actions"]["arm"]["tree"] = json!({ "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" } } });
    let mut goal = run_on_alpha(&rig, workflow.to_string()).await?;
    let mut step_kind = None;
    let mut suspension = None;
    while let Ok(feedback) = goal.on_next_feedback().await {
        if feedback.event == "step_ends" && feedback.step == "arm.look" {
            step_kind = Some(feedback.failure_kind);
        }
        if feedback.event == "run_suspends" {
            suspension = Some(feedback.details);
            break;
        }
    }
    assert_eq!(step_kind.as_deref(), Some("precondition_failed"));
    // The attended run suspends on the incident of the step, which keeps the message of the reply.
    let suspension: Value = serde_json::from_str(&suspension.expect("the run suspends"))
        .expect("the suspension is JSON");
    let incident = &suspension["incidents"][0];
    assert_eq!(
        (
            incident["step"].as_str(),
            incident["failure_kind"].as_str(),
            incident["message"].as_str()
        ),
        (
            Some("arm.look"),
            Some("precondition_failed"),
            Some("the camera is covered")
        ),
        "{suspension}"
    );
    rig.harness.shutdown().await
}

/// A suspend of a run whose lanes ended, while it checks its `success`, is refused with `precondition_failed`: the
/// run is ending. Its `success` reads `find`, which the arm answers only once the test lets it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_suspend_of_a_run_whose_lanes_ended_is_refused() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    let mut workflow: Value = serde_json::from_str(&home_workflow()).expect("the workflow is JSON");
    workflow["observations"] = json!({ "seen": { "member": "arm.find", "role": "arm", "args": { "description": "block" } } });
    workflow["success"] = json!("seen.found");
    let mut goal = run_on_alpha(&rig, workflow.to_string()).await?;
    while goal.on_next_feedback().await?.event != "lane_ends" {}
    assert_eq!(
        kind_of_suspend(&rig, "r1").await.as_deref(),
        Some("precondition_failed")
    );
    rig.arm.finds.add_permits(1);
    feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    assert!(
        matches!(ended.outcome, run::ResultOutcome::Completed(_)),
        "{:?}",
        ended.outcome
    );
    rig.harness.shutdown().await
}

/// A wait on a topic that gives its first message after the wait started wakes at that message: the topic
/// publishes from the start of the wait, and the wait ends well within its bound of 10 s, where the max_age of the
/// topic is 600 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wait_on_a_topic_wakes_at_its_first_message() -> peppygen::Result<()> {
    let rig = Rig::start(Home::AtOnce).await?;
    let mut workflow: Value = serde_json::from_str(&home_workflow()).expect("the workflow is JSON");
    workflow["observations"] =
        json!({ "state": { "topic": "arm.state", "role": "arm", "max_age_s": 600 } });
    let home = workflow["actions"]["arm"]["tree"].clone();
    workflow["actions"]["arm"]["tree"] = json!({ "id": "go", "sequence": [
        { "id": "idle", "timeout_s": 10, "wait_until": "!state.busy" },
        home,
    ] });
    let mut goal = run_on_alpha(&rig, workflow.to_string()).await?;
    loop {
        let feedback = goal.on_next_feedback().await?;
        if feedback.event == "step_starts" && feedback.step == "arm.idle" {
            break;
        }
    }
    let publishing = common::publish_state_every(&rig.harness, Duration::from_millis(500)).await?;
    feedback_of(&mut goal).await;
    let ended = goal.get_result(WIRE).await?;
    assert!(
        matches!(ended.outcome, run::ResultOutcome::Completed(_)),
        "{:?}",
        ended.outcome
    );
    publishing.abort();
    rig.harness.shutdown().await
}

/// The producer of a goal that dies while the director cancels it leaves the robot in an unknown state: once the
/// cancel reached the arm, the step fails with `resource_unavailable`, a failure of the lane, and the attended run
/// suspends on that incident.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_producer_that_dies_during_a_cancel_leaves_its_robot_unknown() -> peppygen::Result<()> {
    let rig = Rig::start(Home::DiesOnCancel).await?;
    let cancel = {
        let cancelled = Arc::clone(&rig.arm.cancelled);
        tokio::spawn(async move { cancelled.notified().await })
    };
    let mut goal = run_home(&rig).await?;
    let mut events = Vec::new();
    while let Ok(feedback) = goal.on_next_feedback().await {
        if feedback.event == "incident" || feedback.event == "step_ends" {
            events.push((feedback.event, feedback.failure_kind));
        }
    }
    cancel.await.expect("the arm got the cancel");
    assert_eq!(
        events,
        [
            ("step_ends".to_owned(), "resource_unavailable".to_owned()),
            ("incident".to_owned(), "resource_unavailable".to_owned()),
        ]
    );
    let ended = goal.get_result(WIRE).await?;
    let run::ResultOutcome::Failed(failure) = ended.outcome else {
        panic!("{:?}", ended.outcome)
    };
    assert_eq!(failure.kind.to_string(), "run_suspended");
    rig.harness.shutdown().await
}
