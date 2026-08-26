//! Integration tests over the generated harness: the node in-process, the
//! four hardware components played by generated mocks over the real wire.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use peppygen::fixtures::exposed_services::robot_ready::is_ready as robot_is_ready;
use peppygen::fixtures::harness::Harness;

/// How long each mock waits parked for the node's next poll. The node polls
/// every 500ms, so this only expires once the harness is gone.
const PUMP_TIMEOUT: Duration = Duration::from_secs(60);

/// Answers every `is_ready` poll from the node with the current value of
/// `flag`, until the mock's session closes.
macro_rules! pump_is_ready {
    ($service:expr, $flag:expr, $response:path) => {{
        let service = $service;
        let flag = Arc::clone(&$flag);
        tokio::spawn(async move {
            type Response = $response;
            while let Ok(responder) = service.next_request(PUMP_TIMEOUT).await {
                let ready = flag.load(Ordering::SeqCst);
                if responder.respond(Response { ready }).await.is_err() {
                    break;
                }
            }
        })
    }};
}

/// Polls the node's exposed `is_ready` until it reports `want`. The node
/// re-checks its components every 500ms, so the deadline bounds a handful of
/// its poll passes, not a fixed sleep.
async fn poll_until(harness: &Harness, want: bool, deadline: Duration) -> peppygen::Result<()> {
    let end = tokio::time::Instant::now() + deadline;
    loop {
        let response = robot_is_ready::poll(harness, Duration::from_secs(2)).await?;
        if response.ready == want {
            return Ok(());
        }
        assert!(
            tokio::time::Instant::now() < end,
            "node never reported ready={want}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reports_ready_only_when_every_component_is() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(openarm_robot_initializer::setup).await?;

    let left_arm = Arc::new(AtomicBool::new(true));
    let right_arm = Arc::new(AtomicBool::new(true));
    let left_gripper = Arc::new(AtomicBool::new(false));
    let right_gripper = Arc::new(AtomicBool::new(true));
    pump_is_ready!(
        mocks.deps.left_arm.is_ready,
        left_arm,
        peppygen::mock::deps::left_arm::is_ready::ResponseData
    );
    pump_is_ready!(
        mocks.deps.right_arm.is_ready,
        right_arm,
        peppygen::mock::deps::right_arm::is_ready::ResponseData
    );
    pump_is_ready!(
        mocks.deps.left_gripper.is_ready,
        left_gripper,
        peppygen::mock::deps::left_gripper::is_ready::ResponseData
    );
    pump_is_ready!(
        mocks.deps.right_gripper.is_ready,
        right_gripper,
        peppygen::mock::deps::right_gripper::is_ready::ResponseData
    );

    // Three of four components ready: the robot must keep reporting not-ready
    // across full poll passes (a pass exists once every component was polled).
    let response = robot_is_ready::poll(&harness, Duration::from_secs(2)).await?;
    assert!(!response.ready);

    // The last component comes up: the aggregate flips within a poll pass.
    left_gripper.store(true, Ordering::SeqCst);
    poll_until(&harness, true, Duration::from_secs(10)).await?;

    // A component reporting not-ready flips the robot back: readiness is
    // re-polled every pass, not latched.
    right_arm.store(false, Ordering::SeqCst);
    poll_until(&harness, false, Duration::from_secs(10)).await?;

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn losing_a_component_flips_back_to_not_ready() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(openarm_robot_initializer::setup).await?;

    let always = Arc::new(AtomicBool::new(true));
    pump_is_ready!(
        mocks.deps.left_arm.is_ready,
        always,
        peppygen::mock::deps::left_arm::is_ready::ResponseData
    );
    pump_is_ready!(
        mocks.deps.left_gripper.is_ready,
        always,
        peppygen::mock::deps::left_gripper::is_ready::ResponseData
    );
    pump_is_ready!(
        mocks.deps.right_gripper.is_ready,
        always,
        peppygen::mock::deps::right_gripper::is_ready::ResponseData
    );
    // The right arm serves from a scripted queue instead of a pump so the
    // whole Mock stays intact for `stop()` below.
    use peppygen::mock::deps::right_arm::is_ready::ResponseData as RightArmReady;
    for _ in 0..60 {
        mocks
            .deps
            .right_arm
            .is_ready
            .enqueue_response(RightArmReady { ready: true })?;
    }

    poll_until(&harness, true, Duration::from_secs(10)).await?;

    // The component dies: its queryable disappears with its session, the
    // node's next poll of it fails, and the aggregate must drop.
    mocks.deps.right_arm.stop();
    poll_until(&harness, false, Duration::from_secs(15)).await?;

    harness.shutdown().await
}
