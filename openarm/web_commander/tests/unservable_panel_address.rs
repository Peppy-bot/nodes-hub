//! A launch whose panel address this host cannot serve: the node must fail its
//! launch naming what to fix. No other port can rescue this one, so it must
//! not be retried.
//!
//! One booting test per binary: `ui::init_limits` is once-per-process.

mod helpers;

use peppygen::fixtures::harness::{Config, Harness};

// An address of TEST-NET-1 (RFC 5737), which no host holds.
const UNSERVABLE_HOST: &str = "192.0.2.1";
const UNSERVABLE_PORT: u16 = 18765;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panel_address_this_host_cannot_serve_fails_the_launch() {
    if tokio::net::TcpListener::bind((UNSERVABLE_HOST, UNSERVABLE_PORT))
        .await
        .is_ok()
    {
        panic!(
            "this host permits binding {UNSERVABLE_HOST} (ip_nonlocal_bind), so the refusal path cannot be exercised here"
        );
    }

    let mut parameters = helpers::test_parameters(UNSERVABLE_PORT);
    parameters.http_host = UNSERVABLE_HOST.to_string();

    // The harness runs `setup` in the background and hands its error back at
    // teardown, so the refusal surfaces there.
    let (harness, _mocks) = Harness::start_with(
        Config {
            parameters: Some(parameters),
            ..Config::default()
        },
        openarm_web_commander::setup,
    )
    .await
    .expect("the harness starts the node");

    let refused = harness
        .shutdown()
        .await
        .expect_err("a commander that cannot serve must fail its launch")
        .to_string();

    assert!(
        refused.contains(UNSERVABLE_HOST),
        "the refusal must name the address that failed, got: {refused}"
    );
    assert!(
        refused.contains("http_host and http_port"),
        "the refusal must name the parameters to change, got: {refused}"
    );
}
