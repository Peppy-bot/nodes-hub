//! Integration tests over the generated harness: the node in-process, both
//! gripper_link peers (backbone leader, engine follower) played by generated
//! pairing mocks over the real wire.

use std::time::{Duration, SystemTime};

use control_core::motor_health::{HEALTH_PERIOD, STATE_STALE_AFTER};
use peppygen::fixtures::exposed_services::ready::is_ready;
use peppygen::fixtures::harness::Harness;
use peppygen::mock::pairings::backbone::gripper_setpoints as backbone_setpoints;
use peppygen::mock::pairings::engine::gripper_states as engine_states;

/// Motors per gripper, matching the node's health `level` vector length.
const MOTORS: usize = 1;

/// Polls the node's exposed `is_ready` until it reports `want`, bounded by
/// `deadline`: the answer flips on the node's relay loop, not on a timer, so
/// the loop observes convergence instead of sleeping for it.
async fn poll_until(harness: &Harness, want: bool, deadline: Duration) -> peppygen::Result<()> {
    let end = tokio::time::Instant::now() + deadline;
    loop {
        let response = is_ready::poll(harness, Duration::from_secs(2)).await?;
        if response.ready == want {
            return Ok(());
        }
        assert!(
            tokio::time::Instant::now() < end,
            "node never reported ready={want}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn engine_state(timestamp: SystemTime) -> engine_states::Message {
    engine_states::Message {
        timestamp,
        opening: 0.03,
        effort: 1.25,
        max_effort: 10.0,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relays_setpoints_and_states_between_backbone_and_engine() -> peppygen::Result<()> {
    let (harness, mut mocks) = Harness::start(openarm_gripper_sim::setup).await?;

    // Backbone -> engine: a non-finite setpoint first, which the relay must
    // drop, then a finite one that must arrive verbatim, timestamp untouched.
    // The two publishes share one mock publisher, so their order holds and the
    // engine peer's first delivery proves both the relay and the drop.
    let setpoint_ts = SystemTime::now();
    mocks
        .pairings
        .backbone
        .gripper_setpoints
        .publish(&backbone_setpoints::Message {
            timestamp: setpoint_ts,
            opening: f64::NAN,
            max_effort: 5.0,
        })
        .await?;
    mocks
        .pairings
        .backbone
        .gripper_setpoints
        .publish(&backbone_setpoints::Message {
            timestamp: setpoint_ts,
            opening: 0.07,
            max_effort: 6.5,
        })
        .await?;
    let relayed = mocks
        .pairings
        .engine
        .gripper_setpoints
        .next()
        .await?
        .expect("engine setpoints subscription should be open");
    assert_eq!(relayed.opening, 0.07);
    assert_eq!(relayed.max_effort, 6.5);
    assert_eq!(relayed.timestamp, setpoint_ts);

    // Engine -> backbone: the measured state forwards back unchanged.
    let state_ts = SystemTime::now();
    mocks
        .pairings
        .engine
        .gripper_states
        .publish(&engine_states::Message {
            timestamp: state_ts,
            opening: 0.045,
            effort: -0.75,
            max_effort: 6.5,
        })
        .await?;
    let relayed = mocks
        .pairings
        .backbone
        .gripper_states
        .next()
        .await?
        .expect("backbone states subscription should be open");
    assert_eq!(relayed.opening, 0.045);
    assert_eq!(relayed.effort, -0.75);
    assert_eq!(relayed.max_effort, 6.5);
    assert_eq!(relayed.timestamp, state_ts);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ready_latches_on_first_state_and_health_flows_only_while_fresh() -> peppygen::Result<()> {
    let (mut harness, mut mocks) = Harness::start(openarm_gripper_sim::setup).await?;

    // Before any engine state, nothing is known about the limb: not ready.
    let response = is_ready::poll(&harness, Duration::from_secs(10)).await?;
    assert!(!response.ready);

    // One engine state flows through; the backbone peer receiving the relayed
    // copy proves the node ingested it.
    let state_ts = SystemTime::now();
    mocks
        .pairings
        .engine
        .gripper_states
        .publish(&engine_state(state_ts))
        .await?;
    let relayed = mocks
        .pairings
        .backbone
        .gripper_states
        .next()
        .await?
        .expect("backbone states subscription should be open");
    assert_eq!(relayed.timestamp, state_ts);

    // Readiness latches on the first relayed state.
    poll_until(&harness, true, Duration::from_secs(10)).await?;

    // The heartbeat now vouches for the limb: "present, not sensed" — a
    // nominal level for the single motor, empty reading vectors.
    let health = tokio::time::timeout(
        Duration::from_secs(5),
        harness.emitted.motor_health_motor_health.next(),
    )
    .await
    .expect("motor_health heartbeat never arrived")?
    .expect("motor_health subscription should be open");
    assert_eq!(health.level, vec![0u8; MOTORS]);
    assert!(health.effort_fraction_rated.is_empty());
    assert!(health.effort_fraction_rated_sustained.is_empty());
    assert!(health.effort_fraction_peak.is_empty());
    assert!(health.driver_temp_c.is_empty());
    assert!(health.winding_temp_c.is_empty());

    // With states stopped, the heartbeat must go silent once the only state
    // ages past STATE_STALE_AFTER. Drain until the topic stays quiet for
    // several health periods; every drained heartbeat passed the recency gate
    // at stamp time, so its stamp must sit inside the state's freshness
    // window — a later stamp would mean the node vouched for a stale limb.
    let silence = HEALTH_PERIOD * 3 + Duration::from_millis(100);
    let drain_deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match tokio::time::timeout(silence, harness.emitted.motor_health_motor_health.next()).await
        {
            // Silent for three-plus periods: the heartbeat is held.
            Err(_) => break,
            Ok(next) => {
                let msg = next?.expect("motor_health subscription should be open");
                let age = msg
                    .timestamp
                    .duration_since(state_ts)
                    .unwrap_or(Duration::ZERO);
                assert!(
                    age < STATE_STALE_AFTER,
                    "heartbeat stamped {age:?} after the last state, past the \
                     {STATE_STALE_AFTER:?} staleness window"
                );
            }
        }
        assert!(
            tokio::time::Instant::now() < drain_deadline,
            "heartbeat never went silent after states stopped"
        );
    }

    // Readiness is a latch: the state is long stale, but ready stays true
    // (mid-session recovery is the runtime's restart, not a ready flap).
    let response = is_ready::poll(&harness, Duration::from_secs(10)).await?;
    assert!(response.ready);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serves_ready_and_ingests_engine_states_without_backbone_peer() -> peppygen::Result<()> {
    // The manifest marks the backbone slot optional: boot it truly unpaired.
    // The peer pin is never seeded, so the node's own `paired()` stays None
    // and its publishes into the slot are legal no-ops.
    let config = peppygen::fixtures::harness::Config {
        backbone_vacant: true,
        ..Default::default()
    };
    let (mut harness, mocks) = Harness::start_with(config, openarm_gripper_sim::setup).await?;
    assert!(
        peppygen::paired_topics::backbone::gripper_states::paired(harness.node_runner())?.is_none(),
        "a vacant boot must leave the backbone slot unpaired"
    );

    // The ready service answers with no backbone peer present: not ready yet.
    let response = is_ready::poll(&harness, Duration::from_secs(10)).await?;
    assert!(!response.ready);

    // Engine states still flow: the relay publishes into the unpaired
    // backbone slot (a legal no-op), records the state, and latches ready.
    mocks
        .pairings
        .engine
        .gripper_states
        .publish(&engine_state(SystemTime::now()))
        .await?;
    poll_until(&harness, true, Duration::from_secs(10)).await?;

    // And the heartbeat vouches for the limb, backbone or not.
    let health = tokio::time::timeout(
        Duration::from_secs(5),
        harness.emitted.motor_health_motor_health.next(),
    )
    .await
    .expect("motor_health heartbeat never arrived")?
    .expect("motor_health subscription should be open");
    assert_eq!(health.level, vec![0u8; MOTORS]);

    harness.shutdown().await
}
