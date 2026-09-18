//! A launch whose panel port another process already holds: the node must come
//! up, serve its panel on a port it could take, announce that port as its
//! `panel` endpoint, and say in its log that the launcher's port was taken.
//! The announcement is what the daemon turns into the URLs an operator opens,
//! so this test reads the panel's address the way the daemon does.
//!
//! One booting test per binary: `ui::init_limits` is once-per-process.

mod helpers;

use std::io;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use peppygen::NodeRunner;
use peppygen::fixtures::harness::{Config, Harness};
use tokio::net::TcpListener;

/// The node's log, captured in memory.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().expect("the log mutex is never poisoned"))
            .into_owned()
    }
}

impl io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("the log mutex is never poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Waits for the node to announce its `panel` endpoint, which `setup` does
/// once it holds the socket. The harness runs `setup` in the background, so
/// the announcement is the readiness signal the daemon gets too. Answers the
/// announced socket address.
async fn await_announcement(node_runner: &NodeRunner) -> std::net::SocketAddr {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(endpoint) = node_runner
            .announced_endpoints()
            .into_iter()
            .find(|endpoint| endpoint.label == "panel")
        {
            assert_eq!(endpoint.binding.scheme, "http");
            assert_eq!(endpoint.binding.path, "");
            return endpoint.binding.address;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the node must announce its panel endpoint"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_held_panel_port_moves_the_panel_and_the_log_says_where() -> peppygen::Result<()> {
    let capture = Capture::default();
    tracing_subscriber::fmt()
        .with_writer({
            let capture = capture.clone();
            move || capture.clone()
        })
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .init();

    // Whatever else on this host owns the port the launcher asked for: another
    // copy of this commander, or a process that has nothing to do with Peppy.
    let held = TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("an operating-system port is always available");
    let taken = held.local_addr().expect("a bound listener has an address");

    // The launch names the held port. Booting at all is the first assertion:
    // the node binds during setup, so a port it cannot take would refuse here.
    let (harness, _mocks) = Harness::start_with(
        Config {
            parameters: Some(helpers::test_parameters(taken.port())),
            ..Config::default()
        },
        openarm_web_commander::setup,
    )
    .await?;

    let bound = await_announcement(harness.node_runner()).await;
    let log = capture.text();
    assert!(
        log.contains(&format!("{taken} is already in use")),
        "the operator must be told the launcher's port was taken; log:\n{log}"
    );
    assert!(
        log.contains(&format!("operator panel bound at {bound}")),
        "the log names the address the panel took; log:\n{log}"
    );

    assert_eq!(
        bound.ip(),
        taken.ip(),
        "the fallback stays on the launcher's host address"
    );
    let served = bound.port();
    assert_ne!(
        served,
        taken.port(),
        "the panel must not claim the port another process holds"
    );

    // The panel the operator reaches at the announced address is this node's:
    // it answers the websocket and publishes the owner's snapshots.
    let mut ws = helpers::WsClient::connect(served).await;
    let snapshot = ws
        .next_snapshot(Duration::from_secs(10), "snapshot from the moved panel")
        .await;
    assert_eq!(snapshot["left_enabled"], false);
    assert_eq!(snapshot["health"]["bound"], false);

    // The holder kept its port throughout.
    let (_connected, accepted) = tokio::join!(
        tokio::net::TcpStream::connect(taken),
        tokio::time::timeout(Duration::from_secs(5), held.accept()),
    );
    accepted
        .expect("the holder must still be accepting on its port")
        .expect("the holder's port is still its own");

    harness.shutdown().await
}
