//! The engine against a scripted host: each test runs a whole run on a
//! paused clock, and reads the calls it sent, the events it gave and its
//! result.

use std::time::Duration;

use serde_json::{Value as Json, json};
use tokio::sync::{mpsc, oneshot, watch};
use workflow_documents::EventKind;
use workflow_documents::result::{
    AlertCause, Anomaly, Ending, LaneState, RunResult, SafeStateOutcome,
};
use workflow_documents::run::{Attendance, Severity};

use super::failure::{Cause, Failure};
use super::host::{CallEnd, Host};
use super::plan::{MonitorPlan, Plan, Target};
use super::{Phase, ResultBase, ResumeRefusal, RunCommand, drive};
use crate::testing::{Answer, ScriptedHost, found, home, name, one_arm, plan, rest, with_beta};

/// The refusal of a resume that a check of it refused.
fn refused_by_a_check(answer: Result<(), ResumeRefusal>) -> workflow_documents::Located {
    match answer {
        Err(ResumeRefusal::Check(refusal)) => refusal,
        other => panic!("a check refuses the resume: {other:?}"),
    }
}

fn grab(retry: u32, done_when: &str) -> Json {
    json!({ "id": "grab", "timeout_s": 30, "retry": retry, "call": {
        "member": "arm.grab",
        "args": { "item_id": { "$ref": "seen.item_id" }, "position": { "$ref": "seen.position" }, "speed": 1 },
        "as": "grabbed", "done_when": done_when,
    } })
}

fn find_then_grab(grab: Json) -> Json {
    json!({ "id": "pick", "sequence": [
        { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
        grab,
    ] })
}

/// Runs `plan` on `host` to its end, sending each command of `commands` at
/// its time in seconds of the run.
async fn run(
    plan: &Plan,
    host: &ScriptedHost,
    commands: Vec<(u64, RunCommand)>,
) -> (RunResult, Vec<Phase>) {
    let (commands_tx, commands_rx) = mpsc::unbounded_channel();
    let (phase_tx, mut phase_rx) = watch::channel(Phase::Preparing);
    let base = ResultBase {
        result: crate::director::empty_result(),
    };
    let phases = tokio::spawn(async move {
        let mut seen = Vec::new();
        while phase_rx.changed().await.is_ok() {
            seen.push(phase_rx.borrow().clone());
        }
        seen
    });
    let sender = tokio::spawn(async move {
        let start = tokio::time::Instant::now();
        for (at, command) in commands {
            tokio::time::sleep_until(start + Duration::from_secs(at)).await;
            let _ = commands_tx.send(command);
        }
        std::future::pending::<()>().await
    });
    let result = drive(plan, host, base, commands_rx, phase_tx, host.now()).await;
    sender.abort();
    (result, phases.await.unwrap())
}

fn arm() -> ScriptedHost {
    ScriptedHost::new()
        .answer("find", vec![Answer::now(found())])
        .answer("home", vec![Answer::now(json!({ "at_home": true }))])
}

#[tokio::test(start_paused = true)]
async fn a_run_calls_its_steps_in_order_and_completes() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm().answer(
        "grab",
        vec![Answer::after(
            2,
            json!({ "success": true, "item_id": "b1" }),
        )],
    );
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    // The reach assumption reads `find` once, then the lane looks and grabs.
    assert_eq!(host.members(), ["find", "find", "grab"]);
    let grab = &host.calls_of("grab")[0];
    assert_eq!(grab.request["item_id"], "b1");
    assert_eq!(grab.request["position"], json!([0.3, 0.0, 0.1]));
    let kinds: Vec<EventKind> = host
        .event_kinds()
        .into_iter()
        .map(|(kind, _)| kind)
        .collect();
    let order = [
        EventKind::InitializationStarts,
        EventKind::ActionsStart,
        EventKind::StepStarts,
        EventKind::LaneEnds,
    ];
    let positions: Vec<usize> = order
        .iter()
        .map(|kind| kinds.iter().position(|k| k == kind).unwrap())
        .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "{kinds:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_result_that_fails_done_when_takes_the_retries() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(1, "grabbed.success")), home()),
        Attendance::Unattended,
    );
    let missed = Answer::now(json!({ "success": false, "item_id": "b1" }));
    let host = arm().answer(
        "grab",
        vec![
            missed,
            Answer::now(json!({ "success": true, "item_id": "b1" })),
        ],
    );
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(host.calls_of("grab").len(), 2);
}

#[tokio::test(start_paused = true)]
async fn an_unattended_lane_that_fails_runs_its_safe_state_and_the_run_fails() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Unattended,
    );
    let host = arm().answer(
        "grab",
        vec![Answer::now(json!({ "success": false, "item_id": "b1" }))],
    );
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result.failed.as_ref().expect("a failed step");
    assert_eq!(failed.failure_kind, "attempt_failed");
    assert_eq!(
        host.members().last().map(String::as_str),
        Some("home"),
        "the safe state runs last"
    );
}

#[tokio::test(start_paused = true)]
async fn a_bound_that_passes_cancels_the_goal_and_fails_with_timeout() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Unattended,
    );
    let host = arm().answer("grab", vec![Answer::hangs()]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    assert_eq!(result.failed.as_ref().unwrap().failure_kind, "timeout");
    let grab = &host.calls_of("grab")[0];
    assert!(grab.cancelled, "the goal of the passed bound is cancelled");
    let home = &host.calls_of("home")[0];
    assert!(
        (home.at_s - grab.at_s - 30.0).abs() < 1.0,
        "the safe state follows the bound of 30 s: {home:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_bound_that_passes_with_the_bound_inside_it_fails_the_outer_step() {
    // The look answers at once, so both bounds pass at 5 s.
    let tree = json!({ "id": "pick", "timeout_s": 5, "sequence": [
        { "id": "look", "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
        { "id": "grab", "timeout_s": 5, "call": {
            "member": "arm.grab",
            "args": { "item_id": { "$ref": "seen.item_id" }, "position": { "$ref": "seen.position" }, "speed": 1 },
        } },
    ] });
    let plan = plan(&one_arm("alpha", tree, home()), Attendance::Unattended);
    let host = arm().answer("grab", vec![Answer::hangs()]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result.failed.as_ref().unwrap();
    assert_eq!(failed.failure_kind, "timeout");
    assert_eq!(failed.step, "alpha.pick", "{failed:?}");
}

#[tokio::test(start_paused = true)]
async fn a_goal_that_outlives_its_cancel_fails_the_step_with_the_robot_unknown() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Unattended,
    );
    let host = arm().answer("grab", vec![Answer::hangs_past_its_cancel()]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    assert_eq!(
        result.failed.as_ref().unwrap().failure_kind,
        "resource_unavailable",
        "the robot is in an unknown state, which comes before the passed bound"
    );
    assert!(host.calls_of("grab")[0].cancelled);
}

#[tokio::test(start_paused = true)]
async fn an_attended_lane_suspends_on_an_incident_and_a_resume_runs_the_failed_step_again() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm().answer(
        "grab",
        vec![
            Answer::failing("unreachable"),
            Answer::now(json!({ "success": true, "item_id": "b1" })),
        ],
    );
    let (reply, resumed) = oneshot::channel();
    let (result, phases) = run(&plan, &host, vec![(60, RunCommand::Resume { reply })]).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(resumed.await.unwrap(), Ok(()));
    assert!(
        phases
            .iter()
            .any(|phase| matches!(phase, Phase::Suspended(_))),
        "{phases:?}"
    );
    assert!(
        !phases.iter().any(|phase| matches!(phase, Phase::Ended(_))),
        "the director marks the run ended after its leases and run_ends: {phases:?}"
    );
    assert_eq!(host.calls_of("grab").len(), 2, "the failed step runs again");
    assert_eq!(
        host.calls_of("home").len(),
        1,
        "the suspended lane runs its safe state"
    );
    assert_eq!(
        *host.leases.lock().unwrap(),
        [("alpha".to_owned(), true), ("alpha".to_owned(), false)],
        "the lease opens, then closes"
    );
    assert_eq!(result.incidents.len(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_abort_cancels_the_goal_and_runs_the_safe_state() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm().answer("grab", vec![Answer::hangs()]);
    let (result, _) = run(&plan, &host, vec![(3, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert!(host.calls_of("grab")[0].cancelled);
    assert_eq!(host.calls_of("home").len(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_abort_of_a_suspended_run_closes_its_leases_and_runs_no_more_safe_state() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm()
        .answer("grab", vec![Answer::failing("unreachable")])
        .closes_leases_after(Duration::from_secs(5));
    let (result, _) = run(&plan, &host, vec![(60, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert_eq!(
        host.calls_of("home").len(),
        1,
        "the safe state ran at the suspension only"
    );
    assert_eq!(
        *host.leases.lock().unwrap(),
        [("alpha".to_owned(), true), ("alpha".to_owned(), false)]
    );
}

#[tokio::test(start_paused = true)]
async fn a_halt_cancels_the_goal_and_sends_nothing_more() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm().answer("grab", vec![Answer::hangs()]);
    let stop = RunCommand::Stop {
        halted: true,
        reason: "the stack is halted".to_owned(),
    };
    let (result, _) = run(&plan, &host, vec![(3, stop)]).await;
    assert_eq!(result.ending, Ending::Halted, "{result:#?}");
    assert!(host.calls_of("grab")[0].cancelled);
    assert!(
        host.calls_of("home").is_empty(),
        "no safe state after a halt"
    );
}

#[tokio::test(start_paused = true)]
async fn a_leader_streaming_refusal_is_sent_again_each_second_within_the_bound() {
    let tree = rest("rest");
    let plan = plan(&one_arm("alpha", tree, json!([])), Attendance::Attended);
    let refused = Answer {
        after: Duration::ZERO,
        end: CallEnd::Refused(Failure::new("leader_streaming", "a leader streams")),
        on_cancel: None,
    };
    let host = ScriptedHost::new()
        .answer("find", vec![Answer::now(found())])
        .answer(
            "home",
            vec![
                refused.clone(),
                refused,
                Answer::now(json!({ "at_home": true })),
            ],
        );
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let times: Vec<f64> = host.calls_of("home").iter().map(|sent| sent.at_s).collect();
    assert_eq!(times.len(), 3);
    assert!(
        times
            .windows(2)
            .all(|pair| (pair[1] - pair[0] - 1.0).abs() < 1e-6),
        "{times:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_failed_reach_assumption_refuses_the_run_before_any_command() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let missing = json!({ "found": false, "item_id": "", "position": [0.0, 0.0, 0.0], "kind": "unknown", "count": "0" });
    let host = ScriptedHost::new().answer("find", vec![Answer::now(missing)]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Refused, "{result:#?}");
    let refusal = result.refusal.expect("a refusal");
    assert_eq!(refusal.assumption, Some(name("in_reach")));
    assert_eq!(host.members(), ["find"], "nothing but the check is called");
}

#[tokio::test(start_paused = true)]
async fn a_for_each_runs_its_body_per_item_and_a_parallel_any_cancels_the_rest() {
    let tree = json!({ "id": "all", "sequence": [
        { "id": "scan", "timeout_s": 5, "call": { "member": "eyes.scan", "args": { "description": "blocks" }, "as": "scanned" } },
        { "id": "each", "for_each": { "list": "scanned.item_ids", "as": "item" }, "max_items": 5, "body": {
            "id": "one", "sequence": [
                { "id": "where", "timeout_s": 5, "call": { "member": "eyes.locate", "args": { "item_id": { "$ref": "item" } }, "as": "at" } },
                { "id": "race", "timeout_s": 20, "policy": "any", "parallel": [
                    { "id": "take", "call": { "member": "arm.grab", "args": { "item_id": { "$ref": "item" }, "position": { "$ref": "at.position" }, "speed": 2 }, "as": "took", "done_when": "took.success" } },
                    { "id": "rest", "call": { "member": "arm.home", "args": { "duration_s": 9 } } },
                ] },
            ],
        } },
    ] });
    let plan = plan(&one_arm("alpha", tree, json!([])), Attendance::Attended);
    let host = ScriptedHost::new()
        .answer("find", vec![Answer::now(found())])
        .answer(
            "scan",
            vec![Answer::now(json!({ "item_ids": ["b1", "b2", "b3"] }))],
        )
        .answer(
            "locate",
            vec![Answer::now(
                json!({ "found": true, "position": [0.2, 0.1, 0.0] }),
            )],
        )
        .answer(
            "grab",
            vec![Answer::after(1, json!({ "success": true, "item_id": "b" }))],
        )
        .answer("home", vec![Answer::hangs()]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let items: Vec<Json> = host
        .calls_of("grab")
        .iter()
        .map(|sent| sent.request["item_id"].clone())
        .collect();
    assert_eq!(items, [json!("b1"), json!("b2"), json!("b3")]);
    assert!(
        host.calls_of("home").iter().all(|sent| sent.cancelled),
        "the slower child of `any` is cancelled"
    );
}

#[tokio::test(start_paused = true)]
async fn a_repeat_until_that_never_holds_ends_at_its_bound() {
    let mut body = rest("rest");
    body.as_object_mut().unwrap().remove("timeout_s");
    let tree = json!({ "id": "again", "timeout_s": 60, "repeat_until": "false", "max_iterations": 3, "body": body });
    let plan = plan(&one_arm("alpha", tree, json!([])), Attendance::Unattended);
    let host = arm();
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    assert_eq!(
        result.failed.as_ref().unwrap().failure_kind,
        "bound_reached"
    );
    assert_eq!(host.calls_of("home").len(), 3);
    assert_eq!(result.loops[0].passes, 3);
}

#[tokio::test(start_paused = true)]
async fn a_wait_until_polls_its_topic_until_it_holds() {
    let mut value = one_arm(
        "alpha",
        json!({ "id": "go", "sequence": [
        { "id": "idle", "wait_until": "!state.busy", "poll_s": 0.5, "timeout_s": 20 },
        rest("rest"),
    ] }),
        json!([]),
    );
    value["observations"]["state"] =
        json!({ "topic": "arm.state", "role": "alpha", "max_age_s": 5 });
    let plan = plan(&value, Attendance::Attended);
    let host = arm();
    host.publish("state", json!({ "busy": true, "holding": false }));
    let idle = async {
        tokio::time::sleep(Duration::from_secs(3)).await;
        host.publish("state", json!({ "busy": false, "holding": false }));
    };
    let ((result, _), ()) = tokio::join!(run(&plan, &host, Vec::new()), idle);
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let rest = &host.calls_of("home")[0];
    assert!(
        (3.0..=3.5).contains(&rest.at_s),
        "the wait ends at the first poll after the topic changed: {rest:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_wait_until_on_a_stale_topic_fails_the_lane() {
    let mut value = one_arm(
        "alpha",
        json!({ "id": "idle", "wait_until": "!state.busy", "poll_s": 0.5, "timeout_s": 20 }),
        json!([]),
    );
    value["observations"]["state"] =
        json!({ "topic": "arm.state", "role": "alpha", "max_age_s": 1 });
    let plan = plan(&value, Attendance::Unattended);
    let host = arm();
    host.publish("state", json!({ "busy": true, "holding": false }));
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    assert_eq!(
        result.failed.as_ref().unwrap().failure_kind,
        "stale_observation"
    );
}

#[tokio::test(start_paused = true)]
async fn a_lane_waits_for_the_signal_of_another() {
    let value = json!({
        "workflow_schema": "workflow/v1",
        "manifest": { "name": "relay", "revision": 1 },
        "task": { "objective": "One arm rests after the other" },
        "contracts": { "arm": "test_arm:v1" },
        "roles": { "alpha": {}, "beta": {} },
        "success": "true",
        "actions": {
            "alpha": { "tree": { "id": "first", "sequence": [rest("rest"), { "id": "tell", "signal": "rested" }] } },
            "beta": { "tree": { "id": "second", "sequence": [{ "id": "hear", "wait_for": ["rested"], "timeout_s": 30 }, rest("rest")] } },
        },
    });
    let plan = plan(&value, Attendance::Attended);
    let host =
        ScriptedHost::new().answer("home", vec![Answer::after(2, json!({ "at_home": true }))]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let calls = host.calls_of("home");
    let at = |role: &str| calls.iter().find(|sent| sent.role == role).unwrap().at_s;
    assert_eq!(at("alpha"), 0.0);
    assert!(at("beta") >= 2.0, "beta rests once alpha has: {calls:?}");
}

/// A look, then a grab beside a rest of the same arm.
fn grab_beside_a_rest() -> Json {
    json!({ "id": "pick", "sequence": [
        { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
        { "id": "both", "policy": "all", "parallel": [
            grab(0, "grabbed.success"),
            { "id": "rest", "timeout_s": 30, "call": { "member": "arm.home", "args": { "duration_s": 9 } } },
        ] },
    ] })
}

/// A grab that fails after 1 s, while the rest beside it moves.
fn failing_after_a_second() -> Answer {
    Answer {
        after: Duration::from_secs(1),
        ..Answer::failing("unreachable")
    }
}

fn durations_of_home(host: &ScriptedHost) -> Vec<(Json, bool)> {
    host.calls_of("home")
        .iter()
        .map(|sent| (sent.request["duration_s"].clone(), sent.cancelled))
        .collect()
}

fn count_of(host: &ScriptedHost, kind: EventKind) -> usize {
    host.event_kinds()
        .iter()
        .filter(|(given, _)| *given == kind)
        .count()
}

#[tokio::test(start_paused = true)]
async fn an_incident_in_a_parallel_step_parks_the_whole_lane_once() {
    let plan = plan(
        &one_arm("alpha", grab_beside_a_rest(), home()),
        Attendance::Attended,
    );
    let host = arm()
        .answer(
            "grab",
            vec![
                failing_after_a_second(),
                Answer::now(json!({ "success": true, "item_id": "b1" })),
            ],
        )
        .answer("home", vec![Answer::after(5, json!({ "at_home": true }))]);
    let (reply, resumed) = oneshot::channel();
    let (result, _) = run(&plan, &host, vec![(60, RunCommand::Resume { reply })]).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(resumed.await.unwrap(), Ok(()));
    // The rest beside the failed grab is cancelled before the safe state,
    // which runs once, and both branches go on at the resume.
    assert_eq!(
        durations_of_home(&host),
        [(json!(9), true), (json!(2), false), (json!(9), false)]
    );
    assert_eq!(host.calls_of("grab").len(), 2);
    assert_eq!(
        *host.leases.lock().unwrap(),
        [("alpha".to_owned(), true), ("alpha".to_owned(), false)],
        "the lease opens once and closes once"
    );
    assert_eq!(count_of(&host, EventKind::LaneSuspends), 1);
    assert_eq!(count_of(&host, EventKind::LaneResumes), 1);
    assert_eq!(result.incidents.len(), 1);
    assert_eq!(result.incidents[0].step, "alpha.grab");
    assert_eq!(result.incidents[0].safe_state.len(), 1);
}

#[tokio::test(start_paused = true)]
async fn an_unattended_incident_in_a_parallel_step_runs_the_safe_state_after_its_siblings() {
    let plan = plan(
        &one_arm("alpha", grab_beside_a_rest(), home()),
        Attendance::Unattended,
    );
    let host = arm()
        .answer("grab", vec![failing_after_a_second()])
        .answer("home", vec![Answer::after(5, json!({ "at_home": true }))]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    assert_eq!(
        durations_of_home(&host),
        [(json!(9), true), (json!(2), false)],
        "the rest is cancelled, then the safe state runs once"
    );
    assert_eq!(result.incidents.len(), 1);
    assert_eq!(result.incidents[0].safe_state.len(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_suspension_asked_between_two_steps_parks_the_lane_at_the_next() {
    // The lane reads `target` for its `if` from 5 s to 10 s, and the run
    // asks it to suspend at 7 s, between two of its calls.
    let tree = json!({ "id": "maybe", "if": "target.found", "then": rest("rest") });
    let plan = plan(&one_arm("alpha", tree, home()), Attendance::Attended);
    let host = ScriptedHost::new()
        .answer("find", vec![Answer::after(5, found())])
        .answer("home", vec![Answer::now(json!({ "at_home": true }))]);
    let (reply, resumed) = oneshot::channel();
    let commands = vec![(7, RunCommand::Suspend), (60, RunCommand::Resume { reply })];
    let (result, phases) = run(&plan, &host, commands).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(resumed.await.unwrap(), Ok(()));
    assert!(
        phases
            .iter()
            .any(|phase| matches!(phase, Phase::Suspended(_))),
        "{phases:?}"
    );
    let homes = host.calls_of("home");
    assert_eq!(
        durations_of_home(&host),
        [(json!(2), false), (json!(1), false)],
        "the safe state runs before the rest, which waits for the resume"
    );
    assert!(homes[1].at_s >= 60.0, "{homes:?}");
}

#[tokio::test(start_paused = true)]
async fn the_assumptions_that_wait_are_sent_again_after_each_confirmation() {
    let mut value = one_arm("alpha", rest("rest"), home());
    value["assumptions"] = json!({
        "clear": { "says": "The table is clear" },
        "dry": { "says": "The table is dry" },
    });
    let plan = plan(&value, Attendance::Attended);
    let host = arm();
    let confirm = |assumption: &str| RunCommand::Confirm {
        assumption: name(assumption),
    };
    let commands = vec![(1, confirm("dry")), (2, confirm("clear"))];
    let (result, phases) = run(&plan, &host, commands).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let waiting: Vec<Vec<String>> = phases
        .iter()
        .filter_map(|phase| match phase {
            Phase::Confirming(to_confirm) => Some(
                to_confirm
                    .iter()
                    .map(|item| item.assumption.to_string())
                    .collect(),
            ),
            _ => None,
        })
        .collect();
    assert_eq!(
        waiting,
        [
            vec!["clear".to_owned(), "dry".to_owned()],
            vec!["clear".to_owned()]
        ],
        "the second to_confirm names the assumption still waiting"
    );
    let sent: Vec<Json> = host
        .events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.kind == EventKind::ToConfirm)
        .map(|event| event.details.clone())
        .collect();
    assert_eq!(
        sent,
        [
            json!([
                { "assumption": "clear", "says": "The table is clear" },
                { "assumption": "dry", "says": "The table is dry" },
            ]),
            json!([{ "assumption": "clear", "says": "The table is clear" }]),
        ],
        "a to_confirm event goes again after the confirmation, naming what still waits"
    );
}

#[tokio::test(start_paused = true)]
async fn a_fault_before_the_lanes_is_met_once_by_the_first_call() {
    let mut value = one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home());
    value["assumptions"]["clear"] = json!({ "says": "The table is clear" });
    let plan = plan(&value, Attendance::Attended);
    let host = arm().answer(
        "grab",
        vec![Answer::now(json!({ "success": true, "item_id": "b1" }))],
    );
    let (reply, resumed) = oneshot::channel();
    let commands = vec![
        (
            1,
            RunCommand::Fault {
                role: name("alpha"),
                failure: Failure::new("alert", "the left gripper is hot").because(Cause::Alert(
                    AlertCause {
                        source: "gripper_left".to_owned(),
                        kind: "overheat".to_owned(),
                        severity: Some(2),
                        stale: false,
                    },
                )),
            },
        ),
        (
            2,
            RunCommand::Confirm {
                assumption: name("clear"),
            },
        ),
        (60, RunCommand::Resume { reply }),
    ];
    let (result, _) = run(&plan, &host, commands).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(resumed.await.unwrap(), Ok(()));
    assert_eq!(result.incidents.len(), 1);
    assert_eq!(result.incidents[0].failure_kind, "alert");
    assert_eq!(result.incidents[0].step, "alpha.look");
    assert_eq!(
        result.incidents[0]
            .alert
            .as_ref()
            .map(|alert| (alert.source.as_str(), alert.severity)),
        Some(("gripper_left", Some(2)))
    );
    // The fault arrived while the run waited for its confirmation, and
    // the resumed lane meets it no more.
    assert_eq!(host.calls_of("grab").len(), 1);
}

#[tokio::test(start_paused = true)]
async fn a_suspended_lane_opens_its_lease_before_it_reports_its_suspension() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm().answer("grab", vec![Answer::failing("unreachable")]);
    let (result, _) = run(&plan, &host, vec![(60, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    let suspends = host
        .event_kinds()
        .iter()
        .position(|(kind, _)| *kind == EventKind::LaneSuspends)
        .expect("the lane suspends");
    let opened = host.events_before_lease.lock().unwrap()[0];
    assert!(
        opened <= suspends,
        "{opened} events before the lease opened, the suspension is event {suspends}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_resume_whose_lease_does_not_close_is_refused() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm().answer("grab", vec![Answer::failing("unreachable")]);
    *host.close_refusal.lock().unwrap() = Some("the daemon is gone".to_owned());
    let (reply, resumed) = oneshot::channel();
    let commands = vec![(60, RunCommand::Resume { reply }), (70, RunCommand::Abort)];
    let (result, _) = run(&plan, &host, commands).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    let refusal = refused_by_a_check(resumed.await.unwrap());
    assert_eq!(refusal.at, "alpha");
    assert!(refusal.message.contains("did not close"), "{refusal:?}");
    assert_eq!(host.calls_of("grab").len(), 1, "the lane does not go on");
}

#[tokio::test(start_paused = true)]
async fn a_halt_of_a_suspended_run_interrupts_its_lanes() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm().answer("grab", vec![Answer::failing("unreachable")]);
    let stop = RunCommand::Stop {
        halted: true,
        reason: "the stack is halted".to_owned(),
    };
    let (result, _) = run(&plan, &host, vec![(60, stop)]).await;
    assert_eq!(result.ending, Ending::Halted, "{result:#?}");
    assert_eq!(result.lanes[0].state, LaneState::Interrupted);
    assert_eq!(
        host.calls_of("home").len(),
        1,
        "the safe state ran at the suspension only"
    );
}

/// `one_arm` with a second observation of `arm.find`, `again`, which the
/// tree reads after the assumption check read `target`.
fn reading_again(tree: Json) -> Json {
    let mut value = one_arm("alpha", tree, json!([]));
    value["observations"]["again"] =
        json!({ "member": "arm.find", "role": "alpha", "args": { "description": "block" } });
    value
}

#[tokio::test(start_paused = true)]
async fn a_read_that_gives_no_answer_fails_its_step_at_its_bound() {
    let tree = json!({ "id": "check", "timeout_s": 5, "if": "again.found", "then": rest("rest") });
    let plan = plan(&reading_again(tree), Attendance::Unattended);
    let host = arm().answer("find", vec![Answer::now(found()), Answer::hangs()]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result.failed.as_ref().unwrap();
    assert_eq!(
        (failed.step.as_str(), failed.failure_kind.as_str()),
        ("alpha.check", "timeout")
    );
    assert!(
        (result.duration.run_s - 5.0).abs() < 0.5,
        "{:?}",
        result.duration
    );
    assert!(host.calls_of("home").is_empty());
}

/// A sibling that fails ends the parallel step with its failure; the read
/// of the branch it cancels ends cancelled, whatever the order of the
/// branches.
#[tokio::test(start_paused = true)]
async fn a_read_of_a_branch_that_a_sibling_ends_ends_cancelled() {
    let tree = json!({ "id": "both", "timeout_s": 30, "policy": "all", "parallel": [
        { "id": "check", "if": "again.found", "then": rest("after") },
        rest("fail"),
    ] });
    let plan = plan(&reading_again(tree), Attendance::Unattended);
    let host = arm()
        .answer("find", vec![Answer::now(found()), Answer::hangs()])
        .answer("home", vec![failing_after_a_second()]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result.failed.as_ref().unwrap();
    assert_eq!(failed.failure_kind, "unreachable", "{failed:?}");
    assert!(
        (result.duration.run_s - 1.0).abs() < 0.5,
        "{:?}",
        result.duration
    );
}

#[tokio::test(start_paused = true)]
async fn an_assumption_check_that_a_leader_stream_refuses_is_asked_again_each_second() {
    let plan = plan(
        &one_arm("alpha", rest("rest"), json!([])),
        Attendance::Unattended,
    );
    let host = arm().answer(
        "find",
        vec![Answer::failing("leader_streaming"), Answer::now(found())],
    );
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let asked: Vec<f64> = host.calls_of("find").iter().map(|sent| sent.at_s).collect();
    assert_eq!(asked, [0.0, 1.0]);
}

#[tokio::test(start_paused = true)]
async fn a_success_that_gives_no_answer_within_its_bound_ends_the_run_failed() {
    let mut value = reading_again(rest("rest"));
    value["success"] = json!("again.found");
    let plan = plan(&value, Attendance::Unattended);
    let host = arm().answer("find", vec![Answer::now(found()), Answer::hangs()]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result
        .failed
        .as_ref()
        .expect("the success check failed the run");
    assert_eq!(
        (failed.step.as_str(), failed.failure_kind.as_str()),
        ("success", "timeout"),
        "{failed:?}"
    );
    assert!(
        (result.duration.run_s - 30.0).abs() < 0.5,
        "{:?}",
        result.duration
    );
}

#[tokio::test(start_paused = true)]
async fn a_reach_check_of_a_resume_that_gives_no_answer_refuses_the_resume_at_its_bound() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm()
        .answer(
            "find",
            vec![Answer::now(found()), Answer::now(found()), Answer::hangs()],
        )
        .answer("grab", vec![Answer::failing("unreachable")]);
    let (reply, resumed) = oneshot::channel();
    let commands = vec![(60, RunCommand::Resume { reply }), (120, RunCommand::Abort)];
    let (result, _) = run(&plan, &host, commands).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    let refusal = refused_by_a_check(resumed.await.unwrap());
    assert_eq!(refusal.at, "assumptions.in_reach");
    assert!(
        refusal.message.contains("within its bound of 30 s"),
        "{refusal:?}"
    );
    let asked = host.calls_of("find");
    assert!((asked[2].at_s - 60.0).abs() < 0.5, "{asked:?}");
    assert_eq!(host.calls_of("grab").len(), 1, "the lane does not go on");
}

#[tokio::test(start_paused = true)]
async fn an_abort_during_the_checks_of_a_resume_ends_the_run_at_once() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm()
        .answer(
            "find",
            vec![Answer::now(found()), Answer::now(found()), Answer::hangs()],
        )
        .answer("grab", vec![Answer::failing("unreachable")]);
    let (reply, resumed) = oneshot::channel();
    let commands = vec![(60, RunCommand::Resume { reply }), (65, RunCommand::Abort)];
    let (result, _) = run(&plan, &host, commands).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert!(
        (result.duration.run_s - 65.0).abs() < 0.5,
        "the abort does not wait for the reach check: {:?}",
        result.duration
    );
    assert!(
        resumed.await.is_err(),
        "the resume has no answer, and its caller follows the run to its end"
    );
}

#[tokio::test(start_paused = true)]
async fn a_resume_during_the_checks_of_another_takes_its_answer() {
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm()
        .answer(
            "find",
            vec![
                Answer::now(found()),
                Answer::now(found()),
                Answer::after(5, found()),
            ],
        )
        .answer(
            "grab",
            vec![
                Answer::failing("unreachable"),
                Answer::now(json!({ "success": true, "item_id": "b1" })),
            ],
        );
    let (first, first_resumed) = oneshot::channel();
    let (second, second_resumed) = oneshot::channel();
    let commands = vec![
        (60, RunCommand::Resume { reply: first }),
        (62, RunCommand::Resume { reply: second }),
    ];
    let (result, _) = run(&plan, &host, commands).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(first_resumed.await.unwrap(), Ok(()));
    assert_eq!(second_resumed.await.unwrap(), Ok(()));
    assert_eq!(host.calls_of("grab").len(), 2, "the lane goes on once");
}

#[tokio::test(start_paused = true)]
async fn an_abort_during_the_assumption_checks_ends_the_run_at_once() {
    let plan = plan(
        &one_arm("alpha", rest("rest"), json!([])),
        Attendance::Unattended,
    );
    let host = arm().answer("find", vec![Answer::hangs()]);
    let (result, _) = run(&plan, &host, vec![(5, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert!(
        (result.duration.run_s - 5.0).abs() < 0.5,
        "{:?}",
        result.duration
    );
    assert_eq!(host.members(), ["find"], "no lane runs");
}

/// A sibling that ends the parallel step before a branch starts leaves the
/// branch cancelled with nothing sent.
#[tokio::test(start_paused = true)]
async fn a_branch_that_a_sibling_ends_before_it_starts_sends_no_goal() {
    let tree = json!({ "id": "either", "timeout_s": 30, "policy": "any", "parallel": [
        { "id": "go", "signal": "go" },
        rest("rest"),
    ] });
    let plan = plan(&one_arm("alpha", tree, json!([])), Attendance::Unattended);
    let host = arm();
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert!(host.calls_of("home").is_empty(), "{:?}", host.members());
}

/// `plan` with the anomaly monitor `check` beside the arm, which lets an
/// unattended run go on past an anomaly of each severity of `continues`.
fn monitored(mut plan: Plan, continues: &[Severity]) -> Plan {
    let arm = plan.calls.values().next().unwrap().target.clone();
    plan.monitor = Some(MonitorPlan {
        target: Target {
            member: name("check"),
            ..arm
        },
        continue_on_severity: continues.iter().copied().collect(),
    });
    plan
}

fn anomaly(severity: &str) -> Answer {
    Answer::now(
        json!({ "anomaly": true, "kind": "slip", "severity": severity, "description": "the block slipped" }),
    )
}

#[tokio::test(start_paused = true)]
async fn the_monitor_checks_each_step_whose_done_when_holds_and_its_anomaly_fails_the_step() {
    let plan = monitored(
        plan(
            &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
            Attendance::Unattended,
        ),
        &[Severity::Low],
    );
    let host = arm()
        .answer(
            "grab",
            vec![Answer::now(json!({ "success": true, "item_id": "b1" }))],
        )
        .answer("check", vec![anomaly("high")]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result.failed.as_ref().unwrap();
    assert_eq!(
        (
            failed.step.as_str(),
            failed.failure_kind.as_str(),
            failed.message.as_str()
        ),
        ("alpha.grab", "anomaly", "high anomaly: the block slipped")
    );
    assert_eq!(
        result.incidents[0].anomaly,
        Some(Anomaly {
            kind: "slip".to_owned(),
            severity: Severity::High,
            description: "the block slipped".to_owned(),
        })
    );
    let checks = host.calls_of("check");
    assert_eq!(
        checks.iter().map(|sent| &sent.request).collect::<Vec<_>>(),
        [&json!({ "robot": "alpha", "step": "alpha.grab" })],
        "the look has no done_when, so only the grab is checked"
    );
}

#[tokio::test(start_paused = true)]
async fn an_unattended_run_goes_on_past_no_anomaly_and_an_anomaly_of_a_severity_it_lets_pass() {
    let plan = monitored(
        plan(
            &one_arm("alpha", find_then_grab(grab(1, "grabbed.success")), home()),
            Attendance::Unattended,
        ),
        &[Severity::Low],
    );
    for check in [anomaly("low"), Answer::now(json!({ "anomaly": false }))] {
        let host = arm()
            .answer(
                "grab",
                vec![Answer::now(json!({ "success": true, "item_id": "b1" }))],
            )
            .answer("check", vec![check]);
        let (result, _) = run(&plan, &host, Vec::new()).await;
        assert_eq!(result.ending, Ending::Completed, "{result:#?}");
        assert_eq!(
            host.calls_of("grab").len(),
            1,
            "a passed check takes no retry"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn an_attended_run_suspends_on_any_anomaly_and_a_monitor_with_no_answer_fails_the_step() {
    let attended = monitored(
        plan(
            &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
            Attendance::Attended,
        ),
        &[Severity::Low],
    );
    let host = arm()
        .answer(
            "grab",
            vec![Answer::now(json!({ "success": true, "item_id": "b1" }))],
        )
        .answer("check", vec![anomaly("low")]);
    let (result, phases) = run(&attended, &host, vec![(60, RunCommand::Abort)]).await;
    assert!(
        phases
            .iter()
            .any(|phase| matches!(phase, Phase::Suspended(_))),
        "an attended run lets no severity pass: {phases:?}"
    );
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert_eq!(result.incidents[0].step, "alpha.grab");
    assert_eq!(
        result.incidents[0]
            .anomaly
            .as_ref()
            .map(|anomaly| anomaly.severity),
        Some(Severity::Low)
    );

    let unattended = monitored(
        plan(
            &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
            Attendance::Unattended,
        ),
        &[],
    );
    let no_answer = Answer {
        end: CallEnd::Unknown,
        ..Answer::now(Json::Null)
    };
    let host = arm()
        .answer(
            "grab",
            vec![Answer::now(json!({ "success": true, "item_id": "b1" }))],
        )
        .answer("check", vec![no_answer]);
    let (result, _) = run(&unattended, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result.failed.as_ref().unwrap();
    assert_eq!(
        (failed.failure_kind.as_str(), failed.message.as_str()),
        ("resource_unavailable", "the anomaly monitor gave no answer")
    );
}

#[tokio::test(start_paused = true)]
async fn an_anomaly_whose_severity_is_none_of_low_medium_and_high_fails_the_step() {
    let plan = monitored(
        plan(
            &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
            Attendance::Unattended,
        ),
        &[Severity::Low],
    );
    let host = arm()
        .answer(
            "grab",
            vec![Answer::now(json!({ "success": true, "item_id": "b1" }))],
        )
        .answer("check", vec![anomaly("urgent")]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result.failed.as_ref().unwrap();
    assert_eq!(failed.failure_kind, "resource_unavailable");
    assert!(
        failed.message.contains("does not read"),
        "{}",
        failed.message
    );
    assert_eq!(result.incidents[0].anomaly, None);
}

/// A refusal of a leader that streams, `after` it was asked.
fn leader_streams(after: Duration) -> Answer {
    Answer {
        after,
        end: CallEnd::Refused(Failure::new("leader_streaming", "a leader streams")),
        on_cancel: None,
    }
}

#[tokio::test(start_paused = true)]
async fn a_suspension_during_a_leader_retry_parks_the_lane_with_no_incident() {
    // The refusals come 0.3 s after each try, so the lane waits for its next
    // try from 1.6 s to 2.6 s, and the suspension at 2 s comes in that wait.
    let mut tree = rest("rest");
    tree["timeout_s"] = json!(30);
    let plan = plan(&one_arm("alpha", tree, json!([])), Attendance::Attended);
    let refused = leader_streams(Duration::from_millis(300));
    let host = ScriptedHost::new()
        .answer("find", vec![Answer::now(found())])
        .answer(
            "home",
            vec![
                refused.clone(),
                refused,
                Answer::now(json!({ "at_home": true })),
            ],
        );
    let (reply, resumed) = oneshot::channel();
    let commands = vec![(2, RunCommand::Suspend), (60, RunCommand::Resume { reply })];
    let (result, phases) = run(&plan, &host, commands).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(resumed.await.unwrap(), Ok(()));
    assert!(
        phases
            .iter()
            .any(|phase| matches!(phase, Phase::Suspended(_))),
        "{phases:?}"
    );
    assert!(
        result.incidents.is_empty(),
        "a suspension is no incident: {:?}",
        result.incidents
    );
    assert_eq!(
        host.calls_of("home").len(),
        3,
        "the step runs again after the resume"
    );
}

#[tokio::test(start_paused = true)]
async fn a_branch_in_a_leader_retry_ends_cancelled_when_its_sibling_wins() {
    let tree = json!({ "id": "either", "policy": "any", "parallel": [
        { "id": "rest", "timeout_s": 30, "call": { "member": "arm.home", "args": { "duration_s": 9 } } },
        { "id": "look", "timeout_s": 30, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
    ] });
    let plan = plan(&one_arm("alpha", tree, json!([])), Attendance::Unattended);
    let refused = leader_streams(Duration::from_millis(300));
    let host = ScriptedHost::new()
        .answer(
            "find",
            vec![
                Answer::now(found()),
                Answer {
                    after: Duration::from_millis(1800),
                    ..Answer::now(found())
                },
            ],
        )
        .answer("home", vec![refused.clone(), refused.clone(), refused]);
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert!(result.failed.is_none(), "{:?}", result.failed);
}

#[tokio::test(start_paused = true)]
async fn an_abort_ends_a_read_that_hangs_outside_a_call_step() {
    // The `if` has no bound, and its read of `target`, a second after the
    // read of the assumption check, hangs.
    let tree = json!({ "id": "steps", "sequence": [
        rest("first"),
        { "id": "maybe", "if": "target.found", "then": rest("rest") },
    ] });
    let plan = plan(&one_arm("alpha", tree, home()), Attendance::Attended);
    let at_home = json!({ "at_home": true });
    let host = ScriptedHost::new()
        .answer("find", vec![Answer::now(found()), Answer::hangs()])
        .answer(
            "home",
            vec![Answer::after(1, at_home.clone()), Answer::now(at_home)],
        );
    let started = tokio::time::Instant::now();
    let (result, _) = run(&plan, &host, vec![(5, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the abort ends the read at once: {:?}",
        started.elapsed()
    );
}

#[tokio::test(start_paused = true)]
async fn an_abort_while_a_lease_opens_for_a_take_over_closes_it_again() {
    // The grab fails at once, the lane runs its safe state and asks the
    // daemon to open the lease, which takes 5 s; the abort comes at 3 s.
    let plan = plan(
        &one_arm("alpha", find_then_grab(grab(0, "grabbed.success")), home()),
        Attendance::Attended,
    );
    let host = arm()
        .answer("grab", vec![Answer::failing("unreachable")])
        .opens_leases_after(Duration::from_secs(5));
    let (result, _) = run(&plan, &host, vec![(3, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert_eq!(
        *host.leases.lock().unwrap(),
        [("alpha".to_owned(), true), ("alpha".to_owned(), false)],
        "the lease that opened closes again"
    );
    assert_eq!(count_of(&host, EventKind::LaneSuspends), 0);
}

#[tokio::test(start_paused = true)]
async fn an_abort_during_the_success_check_ends_the_run_aborted() {
    // Every lane completed at once; the read of `success` hangs, and the
    // abort comes at 5 s, within its 30 s bound.
    let mut value = reading_again(rest("rest"));
    value["success"] = json!("again.found");
    let plan = plan(&value, Attendance::Unattended);
    let host = arm().answer("find", vec![Answer::now(found()), Answer::hangs()]);
    let (result, _) = run(&plan, &host, vec![(5, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert!(result.duration.run_s < 10.0, "{:?}", result.duration);
}

/// The answer of `find` for a cup, which no block case of a `match` takes.
fn a_cup() -> Json {
    json!({ "found": true, "item_id": "c1", "position": [0.3, 0.0, 0.1], "kind": "cup", "count": "1" })
}

/// The answer of `find` for no item.
fn nothing() -> Json {
    json!({ "found": false, "item_id": "", "position": [0.0, 0.0, 0.0], "kind": "unknown", "count": "0" })
}

/// The incident alerts that stand on the robot of `alpha`, as the watch of its
/// alerts reports them.
fn standing_on_alpha(alerts: &[&str]) -> RunCommand {
    RunCommand::Alerts {
        role: name("alpha"),
        source: "gripper_inst".to_owned(),
        standing: alerts.iter().map(|alert| (*alert).to_owned()).collect(),
    }
}

#[tokio::test(start_paused = true)]
async fn a_resume_is_refused_while_an_alert_stands_on_the_robot_of_its_lane() {
    // The alert at 1 s suspends the lane during its 10 s rest. The resume at
    // 20 s is refused while the alert stands; the one at 40 s, after it
    // cleared at 30 s, runs the rest again.
    let plan = plan(
        &one_arm("alpha", rest("rest"), json!([])),
        Attendance::Attended,
    );
    let at_home = json!({ "at_home": true });
    let host = arm().answer(
        "home",
        vec![Answer::after(10, at_home.clone()), Answer::now(at_home)],
    );
    let (refused_reply, refused) = oneshot::channel();
    let (reply, resumed) = oneshot::channel();
    let commands = vec![
        (1, hot_gripper()),
        (1, standing_on_alpha(&["gripper_left (overheat)"])),
        (
            20,
            RunCommand::Resume {
                reply: refused_reply,
            },
        ),
        (30, standing_on_alpha(&[])),
        (40, RunCommand::Resume { reply }),
        (200, RunCommand::Abort),
    ];
    let (result, _) = run(&plan, &host, commands).await;
    let Err(ResumeRefusal::Check(refusal)) = refused.await.unwrap() else {
        panic!("the resume at 20 s is refused");
    };
    assert_eq!(refusal.at, "alpha");
    assert!(
        refusal.message.contains("gripper_left (overheat)"),
        "{}",
        refusal.message
    );
    assert_eq!(resumed.await.unwrap(), Ok(()));
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(
        host.calls_of("home").len(),
        2,
        "the resume runs the rest again"
    );
}

/// An alert of the gripper of `alpha`, as the watch of its alerts gives it.
fn hot_gripper() -> RunCommand {
    RunCommand::Fault {
        role: name("alpha"),
        failure: Failure::new("alert", "the left gripper is hot").because(Cause::Alert(
            AlertCause {
                source: "gripper_left".to_owned(),
                kind: "overheat".to_owned(),
                severity: Some(2),
                stale: false,
            },
        )),
    }
}

#[tokio::test(start_paused = true)]
async fn the_steps_of_a_for_each_item_name_their_item_and_its_loop_counts_its_fallbacks() {
    let tree = json!({ "id": "all", "sequence": [
        { "id": "scan", "timeout_s": 5, "call": { "member": "eyes.scan", "args": { "description": "blocks" }, "as": "scanned" } },
        { "id": "each", "for_each": { "list": "scanned.item_ids", "as": "item" }, "max_items": 5, "body": {
            "id": "one", "sequence": [
                { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
                { "id": "sort", "match": "seen.kind", "cases": { "block": rest("put") }, "otherwise": rest("drop") },
            ],
        } },
    ] });
    let plan = plan(&one_arm("alpha", tree, json!([])), Attendance::Unattended);
    let host = arm()
        .answer(
            "find",
            vec![
                Answer::now(found()),
                Answer::now(found()),
                Answer::now(a_cup()),
                Answer::now(a_cup()),
            ],
        )
        .answer(
            "scan",
            vec![Answer::now(json!({ "item_ids": ["b1", "b2", "b3"] }))],
        );
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let looked: Vec<String> = host
        .event_kinds()
        .into_iter()
        .filter(|(kind, step)| *kind == EventKind::StepStarts && step.starts_with("alpha.look"))
        .map(|(_, step)| step)
        .collect();
    assert_eq!(looked, ["alpha.look#0", "alpha.look#1", "alpha.look#2"]);
    let loops: Vec<(String, u32, u32, u32)> = result
        .loops
        .iter()
        .map(|count| {
            (
                count.step.clone(),
                count.passes,
                count.failed,
                count.fallback,
            )
        })
        .collect();
    assert_eq!(loops, [("alpha.each".to_owned(), 3, 0, 2)]);
}

#[tokio::test(start_paused = true)]
async fn an_abort_while_a_lane_opens_its_lease_and_another_closes_closes_both() {
    // alpha fails at once and suspends at 5 s, as its lease takes 5 s to
    // open; beta fails at 3 s and opens its lease until 8 s; the abort at 6 s
    // closes alpha's lease, which takes 5 s, while beta's opens.
    let value = with_beta(
        one_arm("alpha", rest("rest"), json!([])),
        json!({ "id": "look", "timeout_s": 30, "call": { "member": "arm.find", "args": { "description": "block" } } }),
    );
    let plan = plan(&value, Attendance::Attended);
    let late_failure = Answer {
        after: Duration::from_secs(3),
        ..Answer::failing("unreachable")
    };
    let host = ScriptedHost::new()
        .answer("find", vec![Answer::now(found()), late_failure])
        .answer("home", vec![Answer::failing("unreachable")])
        .opens_leases_after(Duration::from_secs(5))
        .closes_leases_after(Duration::from_secs(5));
    let (result, _) = run(&plan, &host, vec![(6, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    let leases = host.leases.lock().unwrap().clone();
    assert!(
        leases.contains(&("beta".to_owned(), false)),
        "the lease that opened during the abort closes: {leases:?}"
    );
    assert!(leases.contains(&("alpha".to_owned(), false)), "{leases:?}");
}

#[tokio::test(start_paused = true)]
async fn the_safe_state_of_a_suspension_counts_no_time_of_the_bounds_around_it() {
    // `grab` fails at 10 s with 2 s left of the bound of `carry`; the safe
    // state takes 5 s, and the resume at 60 s runs `grab` again within what
    // is left.
    let mut grab = rest("grab");
    grab["timeout_s"] = json!(11);
    let tree = json!({ "id": "carry", "timeout_s": 12, "sequence": [grab] });
    let plan = plan(&one_arm("alpha", tree, home()), Attendance::Attended);
    let failing_late = Answer {
        after: Duration::from_secs(10),
        ..Answer::failing("unreachable")
    };
    let at_home = json!({ "at_home": true });
    let host = arm().answer(
        "home",
        vec![
            failing_late,
            Answer::after(5, at_home.clone()),
            Answer::now(at_home),
        ],
    );
    let (reply, resumed) = oneshot::channel();
    let commands = vec![(60, RunCommand::Resume { reply }), (200, RunCommand::Abort)];
    let (result, _) = run(&plan, &host, commands).await;
    assert_eq!(resumed.await.unwrap(), Ok(()));
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(count_of(&host, EventKind::LaneSuspends), 1);
    assert_eq!(result.incidents.len(), 1, "{:?}", result.incidents);
}

#[tokio::test(start_paused = true)]
async fn a_fault_during_the_last_read_of_a_lane_fails_the_unattended_lane() {
    // The alert comes at 1 s, while `done_when` reads `again`, which answers
    // at 2 s; the lane then has no step left to meet it.
    let mut tree = rest("rest");
    tree["call"]["done_when"] = json!("again.found");
    let plan = plan(&reading_again(tree), Attendance::Unattended);
    let host = arm().answer(
        "find",
        vec![Answer::now(found()), Answer::after(2, found())],
    );
    let (result, _) = run(&plan, &host, vec![(1, hot_gripper())]).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    assert_eq!(
        result
            .failed
            .as_ref()
            .map(|failed| failed.failure_kind.as_str()),
        Some("alert")
    );
    assert_eq!(result.incidents.len(), 1, "{:?}", result.incidents);
    assert_eq!(result.incidents[0].step, "alpha.rest");
}

#[tokio::test(start_paused = true)]
async fn a_fault_after_the_lanes_ended_is_an_incident_of_the_run() {
    // Every lane completed at once; the alert comes at 1 s, while `success`
    // reads `again`, which answers at 2 s.
    let mut value = reading_again(rest("rest"));
    value["success"] = json!("again.found");
    let plan = plan(&value, Attendance::Unattended);
    let host = arm().answer(
        "find",
        vec![Answer::now(found()), Answer::after(2, found())],
    );
    let (result, _) = run(&plan, &host, vec![(1, hot_gripper())]).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(result.incidents.len(), 1, "{:?}", result.incidents);
    assert_eq!(result.incidents[0].failure_kind, "alert");
    assert_eq!(result.incidents[0].step, "alpha.rest");
}

#[tokio::test(start_paused = true)]
async fn a_halt_during_a_safe_state_cancels_its_call_and_sends_no_more() {
    // `rest` fails at once; the first call of the safe state takes 10 s, and
    // the halt comes at 3 s.
    let safe_state = json!([
        { "id": "home", "timeout_s": 30, "call": { "member": "arm.home", "args": { "duration_s": 2 } } },
        { "id": "again", "timeout_s": 30, "call": { "member": "arm.home", "args": { "duration_s": 3 } } },
    ]);
    let plan = plan(
        &one_arm("alpha", rest("rest"), safe_state),
        Attendance::Unattended,
    );
    let host = arm().answer(
        "home",
        vec![
            Answer::failing("unreachable"),
            Answer::after(10, json!({ "at_home": true })),
        ],
    );
    let halt = RunCommand::Stop {
        halted: true,
        reason: "the stack is halted".to_owned(),
    };
    let (result, _) = run(&plan, &host, vec![(3, halt)]).await;
    assert_eq!(result.ending, Ending::Halted, "{result:#?}");
    assert_eq!(
        durations_of_home(&host),
        [(json!(1), false), (json!(2), true)],
        "the halt cancels the call of the safe state, and the next is not sent"
    );
}

#[tokio::test(start_paused = true)]
async fn a_run_whose_lanes_ended_is_ending_and_takes_no_suspension() {
    // Every lane completed at once; `success` reads `again`, which answers at
    // 2 s, and the suspension at 1 s finds no lane to suspend.
    let mut value = reading_again(rest("rest"));
    value["success"] = json!("again.found");
    let plan = plan(&value, Attendance::Attended);
    let host = arm().answer(
        "find",
        vec![Answer::now(found()), Answer::after(2, found())],
    );
    let (result, phases) = run(&plan, &host, vec![(1, RunCommand::Suspend)]).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(phases.last(), Some(&Phase::Ending), "{phases:?}");
    assert!(
        !phases
            .iter()
            .any(|phase| matches!(phase, Phase::Suspended(_))),
        "{phases:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn the_poll_ticks_that_a_slow_read_passed_merge_into_the_next() {
    // Each read of `again` takes 2 s, longer than the poll of 0.5 s: the wait
    // reads at the next tick after each read.
    let tree = json!({ "id": "idle", "wait_until": "again.found", "poll_s": 0.5, "timeout_s": 20 });
    let plan = plan(&reading_again(tree), Attendance::Unattended);
    let host = arm().answer(
        "find",
        vec![
            Answer::now(found()),
            Answer::after(2, nothing()),
            Answer::after(2, nothing()),
            Answer::after(2, found()),
        ],
    );
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let asked: Vec<f64> = host.calls_of("find").iter().map(|sent| sent.at_s).collect();
    assert_eq!(asked, [0.0, 0.0, 2.5, 5.0]);
}

#[tokio::test(start_paused = true)]
async fn two_lanes_that_read_one_observation_at_one_instant_share_its_read() {
    let wait = json!({ "id": "idle", "wait_until": "again.found", "poll_s": 0.5, "timeout_s": 20 });
    let plan = plan(
        &with_beta(reading_again(wait.clone()), wait),
        Attendance::Unattended,
    );
    let host = arm().answer(
        "find",
        vec![Answer::now(found()), Answer::after(1, found())],
    );
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(
        host.calls_of("find").len(),
        2,
        "the check of the assumption, and one read for both lanes"
    );
}

/// `rest` on `alpha`, with the assumption `idle` whose check reads the topic
/// `state`, of `max_age_s`.
fn reading_state(max_age_s: u64) -> Json {
    let mut value = one_arm("alpha", rest("rest"), json!([]));
    value["observations"]["state"] =
        json!({ "topic": "arm.state", "role": "alpha", "max_age_s": max_age_s });
    value["assumptions"] = json!({ "idle": { "says": "The arm is idle", "check": "!state.busy" } });
    value
}

#[tokio::test(start_paused = true)]
async fn a_read_before_the_first_message_of_a_topic_waits_for_it() {
    // The topic gives its first message 2 s after the run starts, within its
    // max_age of 5 s: the check reads it then.
    let plan = plan(&reading_state(5), Attendance::Unattended);
    let host = arm();
    let first = async {
        tokio::time::sleep(Duration::from_secs(2)).await;
        host.publish("state", json!({ "busy": false, "holding": false }));
    };
    let ((result, _), ()) = tokio::join!(run(&plan, &host, Vec::new()), first);
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let rest = &host.calls_of("home")[0];
    assert!((rest.at_s - 2.0).abs() < 0.1, "{rest:?}");
}

#[tokio::test(start_paused = true)]
async fn an_abort_during_the_wait_for_the_first_message_of_a_topic_ends_the_run_at_once() {
    // The topic gives no message, and its max_age of 600 s is far: the abort
    // at 5 s ends the wait.
    let plan = plan(&reading_state(600), Attendance::Unattended);
    let host = arm();
    let (result, _) = run(&plan, &host, vec![(5, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert!(
        (result.duration.run_s - 5.0).abs() < 0.5,
        "{:?}",
        result.duration
    );
    assert!(host.calls_of("home").is_empty());
}

#[tokio::test(start_paused = true)]
async fn an_abort_of_two_suspended_lanes_closes_both_leases_before_the_run_ends() {
    // Both lanes fail at once and suspend; each close of a lease takes 5 s.
    let value = with_beta(
        one_arm("alpha", rest("rest"), json!([])),
        json!({ "id": "look", "timeout_s": 30, "call": { "member": "arm.find", "args": { "description": "block" } } }),
    );
    let plan = plan(&value, Attendance::Attended);
    let host = ScriptedHost::new()
        .answer(
            "find",
            vec![Answer::now(found()), Answer::failing("unreachable")],
        )
        .answer("home", vec![Answer::failing("unreachable")])
        .closes_leases_after(Duration::from_secs(5));
    let (result, _) = run(&plan, &host, vec![(10, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    let leases = host.leases.lock().unwrap().clone();
    for robot in ["alpha", "beta"] {
        assert!(
            leases.contains(&(robot.to_owned(), false)),
            "the lease of {robot} closes: {leases:?}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn a_safe_state_call_that_a_leader_stream_refuses_in_a_park_goes_again_each_second() {
    // `rest` fails at once; the safe state of the park is refused once while
    // a leader streams, and goes again a second later, while lane time stands.
    let plan = plan(
        &one_arm("alpha", rest("rest"), home()),
        Attendance::Attended,
    );
    let host = arm().answer(
        "home",
        vec![
            Answer::failing("unreachable"),
            leader_streams(Duration::from_millis(300)),
            Answer::now(json!({ "at_home": true })),
        ],
    );
    let (result, _) = run(&plan, &host, vec![(60, RunCommand::Abort)]).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert_eq!(
        result.incidents[0].safe_state[0].outcome,
        SafeStateOutcome::Completed,
        "{:?}",
        result.incidents
    );
    let asked: Vec<f64> = host.calls_of("home").iter().map(|sent| sent.at_s).collect();
    assert_eq!(asked.len(), 3, "{asked:?}");
    assert!((asked[2] - 1.3).abs() < 0.1, "{asked:?}");
}

#[tokio::test(start_paused = true)]
async fn a_safe_state_call_that_hangs_in_a_park_fails_at_its_bound_on_the_run_clock() {
    // The lane time of the park stands; the bound of 10 s of the safe-state
    // call passes on the run clock, and the lane suspends.
    let plan = plan(
        &one_arm("alpha", rest("rest"), home()),
        Attendance::Attended,
    );
    let host = arm().answer(
        "home",
        vec![Answer::failing("unreachable"), Answer::hangs()],
    );
    let ran = tokio::time::timeout(
        Duration::from_secs(300),
        run(&plan, &host, vec![(60, RunCommand::Abort)]),
    )
    .await;
    let (result, phases) = ran.expect("the bound of the safe-state call passes");
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert!(
        phases
            .iter()
            .any(|phase| matches!(phase, Phase::Suspended(_))),
        "{phases:?}"
    );
    assert_eq!(
        result.incidents[0].safe_state[0].outcome,
        SafeStateOutcome::Failed {
            failure_kind: "timeout".to_owned()
        }
    );
}

/// `one_arm` of `alpha` whose lane is `tree`, with the topic `state` of
/// `max_age_s` beside the observation `target` of its assumption.
fn with_state(tree: Json, max_age_s: u64) -> Json {
    let mut value = one_arm("alpha", tree, json!([]));
    value["observations"]["state"] =
        json!({ "topic": "arm.state", "role": "alpha", "max_age_s": max_age_s });
    value
}

/// `rest` under the invariant `!state.busy`, of `duration_s`, within
/// `timeout_s`.
fn rest_while_idle(duration_s: u64, timeout_s: u64) -> Json {
    let mut step = rest("rest");
    step["invariant"] = json!("!state.busy");
    step["timeout_s"] = json!(timeout_s);
    step["call"]["args"]["duration_s"] = json!(duration_s);
    step
}

#[tokio::test(start_paused = true)]
async fn an_invariant_that_is_false_at_the_start_fails_the_step_with_no_call() {
    let plan = plan(
        &with_state(rest_while_idle(1, 30), 5),
        Attendance::Unattended,
    );
    let host = arm();
    host.publish("state", json!({ "busy": true, "holding": false }));
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result.failed.as_ref().unwrap();
    assert_eq!(
        (failed.step.as_str(), failed.failure_kind.as_str()),
        ("alpha.rest", "invariant_violated")
    );
    assert!(host.calls_of("home").is_empty(), "{:?}", host.members());
}

#[tokio::test(start_paused = true)]
async fn an_invariant_that_turns_false_at_a_tick_cancels_the_call_and_fails_the_step() {
    // The arm turns busy at 2 s, while `rest` runs a goal that does not end.
    let plan = plan(
        &with_state(rest_while_idle(9, 30), 5),
        Attendance::Unattended,
    );
    let host = arm().answer("home", vec![Answer::hangs()]).publishes_at(
        2,
        "state",
        json!({ "busy": true, "holding": false }),
    );
    host.publish("state", json!({ "busy": false, "holding": false }));
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    assert_eq!(
        result
            .failed
            .as_ref()
            .map(|failed| failed.failure_kind.as_str()),
        Some("invariant_violated")
    );
    let homes = host.calls_of("home");
    assert!(
        homes[0].cancelled,
        "the call under the invariant is cancelled: {homes:?}"
    );
    assert!(
        (result.duration.run_s - 2.0).abs() < 0.6,
        "{:?}",
        result.duration
    );
}

#[tokio::test(start_paused = true)]
async fn an_invariant_read_that_waits_for_a_first_message_ends_at_the_bound_of_its_step() {
    // The topic gives no message, and its max_age of 600 s is far: the read
    // of the invariant at the start of `rest` ends at its bound of 5 s.
    let plan = plan(
        &with_state(rest_while_idle(1, 5), 600),
        Attendance::Unattended,
    );
    let host = arm();
    let (result, _) = run(&plan, &host, Vec::new()).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    let failed = result.failed.as_ref().unwrap();
    assert_eq!(
        (failed.step.as_str(), failed.failure_kind.as_str()),
        ("alpha.rest", "timeout")
    );
    assert!(
        (result.duration.run_s - 5.0).abs() < 0.5,
        "{:?}",
        result.duration
    );
}

/// An `if` whose read waits for the first message of a topic, of max_age
/// 600 s, at the start of an attended lane.
fn checking_the_state() -> Plan {
    let tree = json!({ "id": "check", "if": "state.busy", "then": rest("rest") });
    plan(&with_state(tree, 600), Attendance::Attended)
}

#[tokio::test(start_paused = true)]
async fn a_suspension_during_a_read_that_waits_for_a_first_message_parks_the_lane() {
    let host = arm();
    let commands = vec![(5, RunCommand::Suspend), (60, RunCommand::Abort)];
    let (result, phases) = run(&checking_the_state(), &host, commands).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert!(
        phases
            .iter()
            .any(|phase| matches!(phase, Phase::Suspended(_))),
        "the suspension parks the lane during the read: {phases:?}"
    );
    assert!(result.incidents.is_empty(), "{:?}", result.incidents);
}

#[tokio::test(start_paused = true)]
async fn an_alert_during_a_read_that_waits_for_a_first_message_suspends_the_lane() {
    let host = arm();
    let commands = vec![(5, hot_gripper()), (60, RunCommand::Abort)];
    let (result, _) = run(&checking_the_state(), &host, commands).await;
    assert_eq!(result.ending, Ending::Aborted, "{result:#?}");
    assert_eq!(count_of(&host, EventKind::LaneSuspends), 1);
    assert_eq!(result.incidents.len(), 1, "{:?}", result.incidents);
    assert_eq!(
        (
            result.incidents[0].step.as_str(),
            result.incidents[0].failure_kind.as_str()
        ),
        ("alpha.check", "alert")
    );
}

#[tokio::test(start_paused = true)]
async fn an_alert_during_a_done_when_check_that_waits_for_a_first_message_ends_the_step() {
    // `rest` fails `attempt_failed`, so its `done_when` is read before the
    // retry; `state` gives no message, and the alert at 2 s ends that wait.
    let mut tree = rest("rest");
    tree["timeout_s"] = json!(30);
    tree["retry"] = json!(1);
    tree["call"]["done_when"] = json!("!state.busy");
    let plan = plan(&with_state(tree, 600), Attendance::Unattended);
    let host = arm().answer("home", vec![Answer::failing("attempt_failed")]);
    let (result, _) = run(&plan, &host, vec![(2, hot_gripper())]).await;
    assert_eq!(result.ending, Ending::Failed, "{result:#?}");
    assert_eq!(
        result
            .failed
            .as_ref()
            .map(|failed| failed.failure_kind.as_str()),
        Some("alert")
    );
    assert_eq!(result.incidents.len(), 1, "{:?}", result.incidents);
    assert_eq!(host.calls_of("home").len(), 1, "the alert sends no retry");
}

#[tokio::test(start_paused = true)]
async fn a_suspension_of_a_step_under_an_invariant_parks_its_running_child_once() {
    // `carry` holds `lift`, done at once, then `move`, which runs from 0 s to
    // 10 s; the suspension comes at 5 s, at a tick of the invariant of
    // `carry`, and the resume at 60 s runs `move` again.
    let mut moving = rest("move");
    moving["timeout_s"] = json!(30);
    moving["call"]["args"]["duration_s"] = json!(9);
    let tree = json!({ "id": "carry", "timeout_s": 120, "invariant": "!state.busy", "sequence": [rest("lift"), moving] });
    let plan = plan(&with_state(tree, 600), Attendance::Attended);
    let at_home = json!({ "at_home": true });
    let host = arm().answer(
        "home",
        vec![
            Answer::now(at_home.clone()),
            Answer::after(10, at_home.clone()),
            Answer::now(at_home),
        ],
    );
    host.publish("state", json!({ "busy": false, "holding": false }));
    let (reply, resumed) = oneshot::channel();
    let commands = vec![
        (5, RunCommand::Suspend),
        (60, RunCommand::Resume { reply }),
        (200, RunCommand::Abort),
    ];
    let (result, _) = run(&plan, &host, commands).await;
    assert_eq!(resumed.await.unwrap(), Ok(()));
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    assert_eq!(count_of(&host, EventKind::LaneSuspends), 1);
    assert_eq!(
        durations_of_home(&host),
        [(json!(1), false), (json!(9), true), (json!(9), false)],
        "`lift` goes once, and the resume sends `move` again"
    );
}

#[tokio::test(start_paused = true)]
async fn a_suspension_during_the_test_of_a_loop_keeps_its_passes() {
    // The clock moves between the reads of `again`, so each test of the
    // loop reads it itself. The second test's read takes 2 s, and the
    // suspension at 1 s comes during it: the read answers, and the pass
    // parks at `place`. The resume at 60 s reads the reach check of
    // `target`, then goes on with the pass.
    let tree = json!({ "id": "fill", "repeat_until": "again.found", "max_iterations": 3, "body": rest("place") });
    let plan = plan(&reading_again(tree), Attendance::Attended);
    let host = arm().moving_at_each_reading().answer(
        "find",
        vec![
            Answer::now(found()),
            Answer::now(nothing()),
            Answer::after(2, nothing()),
            Answer::now(found()),
            Answer::now(nothing()),
            Answer::now(found()),
        ],
    );
    let (reply, resumed) = oneshot::channel();
    let commands = vec![
        (1, RunCommand::Suspend),
        (60, RunCommand::Resume { reply }),
        (200, RunCommand::Abort),
    ];
    let (result, _) = run(&plan, &host, commands).await;
    assert_eq!(resumed.await.unwrap(), Ok(()));
    assert_eq!(result.ending, Ending::Completed, "{result:#?}");
    let loops: Vec<(String, u32, u32)> = result
        .loops
        .iter()
        .map(|count| (count.step.clone(), count.passes, count.failed))
        .collect();
    assert_eq!(loops, [("alpha.fill".to_owned(), 3, 0)]);
    assert_eq!(host.calls_of("home").len(), 3, "one `place` per pass");
}
