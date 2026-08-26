//! Every launcher value the backbone refuses, proven through the real node
//! entry point rather than through the validators alone.
//!
//! The point of each case is the shape of the failure, not just that it fails:
//! `setup` returns `Result`, so a bad parameter has to come back as an `Err`
//! that names it. A panic would unwind past the runtime's shutdown hooks and
//! reach the daemon as a backtrace, where the parameter's name belongs.

use peppygen::fixtures::harness::{Config, Harness};

mod helpers;
use helpers::params;

/// Boot the node with one parameter spoiled and return the refusal as an
/// operator reads it: the whole Display chain, since a wrapping variant names
/// the step and its source carries the detail. Fails the test if the node
/// accepts the value instead.
///
/// `start_with` spawns `setup` rather than awaiting it, so a refusal can
/// surface at either end: from the start itself when the node dies before the
/// fixture finishes converging, or from `shutdown`, which awaits `setup` and
/// propagates its error. Reading only one end makes the test race the node.
async fn refusal(spoil: impl FnOnce(&mut peppygen::Parameters)) -> String {
    let refused = refusal_error(spoil).await;
    let mut chain = vec![refused.to_string()];
    let mut cause = std::error::Error::source(&refused);
    while let Some(error) = cause {
        chain.push(error.to_string());
        cause = error.source();
    }
    chain.join(": ")
}

/// The refusal as the runtime received it, for the cases that care which
/// variant it arrived as rather than only what it says.
async fn refusal_error(spoil: impl FnOnce(&mut peppygen::Parameters)) -> peppygen::Error {
    let mut parameters = params();
    spoil(&mut parameters);
    let started = Harness::start_with(
        Config {
            parameters: Some(parameters),
            collision_ctrl_vacant: true,
            ..Default::default()
        },
        openarm_backbone::setup,
    )
    .await;
    match started {
        Err(e) => e,
        Ok((harness, _mocks)) => harness
            .shutdown()
            .await
            .expect_err("the backbone accepted a parameter it must refuse"),
    }
}

/// Asserts the refusal names the parameter, which is the whole reason it is a
/// refusal and not a panic.
async fn refusal_names(parameter: &str, spoil: impl FnOnce(&mut peppygen::Parameters)) {
    let text = refusal(spoil).await;
    assert!(
        text.contains(parameter),
        "the refusal must name {parameter}, got: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_zero_control_rate_is_refused_by_name() {
    // Zero would divide into a 0 us period and spin the control loop.
    refusal_names("control_rate_hz", |p| p.control_rate_hz = 0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_non_finite_joint_velocity_cap_is_refused_by_name() {
    // NaN reaches the chase as a clamp bound, where it disables the clamp
    // silently rather than tripping anything.
    refusal_names("max_joint_velocity_rad_s_4", |p| {
        p.max_joint_velocity_rad_s_4 = f64::NAN
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_non_finite_ee_speed_cap_is_refused_by_name() {
    refusal_names("max_ee_velocity_m_s", |p| {
        p.max_ee_velocity_m_s = f64::INFINITY
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_inverted_collision_band_is_refused_by_name() {
    // d_stop above d_safe inverts the throttle: the governor would read every
    // clearance as inside the stop band.
    refusal_names("d_stop_m", |p| {
        p.d_stop_m = 0.05;
        p.d_safe_m = 0.02;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_filter_cutoff_at_or_above_nyquist_is_refused_by_name() {
    // At half the control rate the low-pass stops attenuating; above it the
    // cutoff has no meaning at all.
    refusal_names("velocity_filter_cutoff_hz", |p| {
        p.velocity_filter_cutoff_hz = p.control_rate_hz as f64 / 2.0
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unknown_hardware_version_is_refused_by_name() {
    let text = refusal(|p| p.hardware_version = "v3".to_string()).await;
    assert!(
        text.contains("v3"),
        "the refusal must quote the offending value, got: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unknown_upstream_mode_is_refused_by_name() {
    let text = refusal(|p| p.upstream_mode = "telepathy".to_string()).await;
    assert!(
        text.contains("telepathy"),
        "the refusal must quote the offending value, got: {text}"
    );
}

/// The refusal reaches the runtime as `Error::Node` carrying this node's own
/// error, which is what `#[source]` on that variant buys: the operator reads
/// the message, and a caller can still recover the typed cause. Wrapping it in
/// a runtime variant instead would flatten it to text here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refusal_reaches_the_runtime_with_its_source_intact() {
    let refused = refusal_error(|p| p.control_rate_hz = 0).await;
    assert!(
        matches!(refused, peppygen::Error::Node(_)),
        "the refusal must arrive as Error::Node, got: {refused:?}"
    );
    let source = std::error::Error::source(&refused).expect("the boxed cause stays reachable");
    assert!(
        source
            .downcast_ref::<openarm_backbone::NodeError>()
            .is_some(),
        "the cause must still be this node's NodeError, got: {source}"
    );
}

/// What the operator actually reads. `Termination` renders a `main` that
/// returns `Result` with `Debug`, so the sentence a variant defines reaches
/// the daemon log only if something walks `Display`. This pins both halves:
/// the parameter's name is in the Display chain, and it is absent from the
/// `Debug` rendering that a bare `Result`-returning `main` would have printed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_refusal_an_operator_reads_names_the_parameter() {
    let refused = refusal_error(|p| p.control_rate_hz = 0).await;

    let mut chain = vec![refused.to_string()];
    let mut cause = std::error::Error::source(&refused);
    while let Some(error) = cause {
        chain.push(error.to_string());
        cause = error.source();
    }
    let displayed = chain.join(" | ");
    assert!(
        displayed.contains("control_rate_hz"),
        "the Display chain must name the parameter, got: {displayed}"
    );

    let debugged = format!("{refused:?}");
    assert!(
        !debugged.contains("must be in"),
        "Debug is not expected to carry the message; if it now does, this \
         test no longer proves the reporter is what surfaces it: {debugged}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rate_too_slow_for_the_servo_smoother_is_refused_by_name() {
    // 30 Hz puts Nyquist (15 Hz) under the servo's fixed cutoff. Before this
    // was a refusal it was an expect inside the first Cartesian move: the node
    // booted healthy and panicked when someone commanded a pose.
    refusal_names("control_rate_hz", |p| p.control_rate_hz = 30).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_non_finite_ee_angular_cap_is_refused_by_name() {
    refusal_names("max_ee_angular_velocity_rad_s", |p| {
        p.max_ee_angular_velocity_rad_s = f64::NAN;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gripper_rate_too_slow_to_make_progress_is_refused_by_name() {
    // Below the governor's floor a full stroke would take over a minute; the
    // governor is the single authority for this bound, so the refusal arrives
    // through its error rather than a duplicate node-side check.
    refusal_names("max_gripper_rate_frac_s", |p| {
        p.max_gripper_rate_frac_s = 1e-6;
    })
    .await;
}
