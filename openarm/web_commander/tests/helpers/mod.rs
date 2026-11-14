//! Shared plumbing for the integration tests: launch parameters, the node's
//! boot with its robot's initializer answering, and a minimal WebSocket
//! client for the commander's real operator surface (the HTTP+WS panel). No
//! WS client crate is in the node's dependency set, so this speaks
//! just enough RFC 6455 for the panel's unfragmented text frames.
//!
//! NOTE: each test binary boots the node at most once. `limits::init_limits`
//! asserts it runs exactly once per process, so a second `Harness::start` in
//! the same binary would panic; every tests/*.rs file therefore holds exactly
//! one booting test.
#![allow(dead_code)]

use std::time::Duration;

use control_core::fault::Fault;
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// The model of the tests' robot, as its initializer names it.
pub const MODEL: &str = "openarm_v2";

/// How long the robot's initializer mock waits for the node to ask for the
/// robot's model.
const IDENTITY_BUDGET: Duration = Duration::from_secs(30);

/// How long one handshake attempt may take before it is retried.
const HANDSHAKE_BUDGET: Duration = Duration::from_secs(5);

/// How long a test waits for the node's `setup` to return.
const SETUP_BUDGET: Duration = Duration::from_secs(30);

/// How often a test checks whether the node's `setup` has returned.
const SETUP_POLL: Duration = Duration::from_millis(50);

/// Valid launch parameters (every required field of the manifest schema):
/// governor on with the 0.005/0.02 band, a 0.5 m/s speed cap, v2 ranges, and
/// a 50 Hz command tick. The panel serves loopback on the caller's port, one
/// per test binary (the production default 8765 could collide with anything
/// on the host).
pub fn test_parameters(port: u16) -> peppygen::Parameters {
    peppygen::Parameters {
        collision_governor_enabled: true,
        command_rate_hz: 50,
        d_safe: 0.02,
        d_stop: 0.005,
        http_host: "127.0.0.1".to_string(),
        http_port: port,
        joint_jog_acceleration_rad_s2: 10.0,
        joint_jog_velocity_rad_s: 1.0,
        max_stream_ee_velocity_m_s: 0.5,
        max_gripper_rate_frac_s: 6.0,
    }
}

fn other(message: impl Into<String>) -> std::io::Error {
    std::io::Error::other(message.into())
}

/// A WebSocket connection to the panel's `/ws` route.
pub struct WsClient {
    stream: TcpStream,
}

impl WsClient {
    async fn connect_attempt(port: u16) -> std::io::Result<Self> {
        tokio::time::timeout(HANDSHAKE_BUDGET, Self::connect_once(port))
            .await
            .unwrap_or_else(|_| {
                Err(other(format!(
                    "no websocket handshake from port {port} within {HANDSHAKE_BUDGET:?}"
                )))
            })
    }

    async fn connect_once(port: u16) -> std::io::Result<Self> {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).await?;
        // A fixed Sec-WebSocket-Key is legal: the server only needs it to
        // derive its accept hash, which this client does not verify.
        let request = format!(
            "GET /ws HTTP/1.1\r\n\
             Host: 127.0.0.1:{port}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: AAAAAAAAAAAAAAAAAAAAAA==\r\n\
             Sec-WebSocket-Version: 13\r\n\r\n"
        );
        stream.write_all(request.as_bytes()).await?;
        let mut response = Vec::new();
        let mut byte = [0u8; 1];
        while !response.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).await?;
            response.push(byte[0]);
            if response.len() > 16 * 1024 {
                return Err(other("oversized websocket handshake response"));
            }
        }
        let head = String::from_utf8_lossy(&response);
        if !head.starts_with("HTTP/1.1 101") {
            return Err(other(format!(
                "websocket handshake refused: {}",
                head.lines().next().unwrap_or_default()
            )));
        }
        Ok(Self { stream })
    }

    /// Connects to the panel, retrying while its server task is still coming
    /// up on the already-bound socket. Bounded: panics if it never accepts.
    ///
    /// Each attempt is bounded too. A process that accepts the connection and
    /// then says nothing is what a panel moved off this port looks like from
    /// here, and the handshake read would otherwise block past the deadline.
    pub async fn connect(port: u16) -> Self {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            match Self::connect_attempt(port).await {
                Ok(ws) => return ws,
                Err(e) => {
                    assert!(
                        tokio::time::Instant::now() < deadline,
                        "panel on port {port} never accepted a websocket: {e}"
                    );
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    /// Sends one text frame (a panel `Command` as JSON). Client frames must
    /// carry a mask; the all-zero key leaves the payload bytes unchanged.
    pub async fn send_text(&mut self, text: &str) -> std::io::Result<()> {
        let payload = text.as_bytes();
        let mut frame = vec![0x81u8]; // FIN + text opcode
        match payload.len() {
            n if n < 126 => frame.push(0x80 | n as u8),
            n if n <= u16::MAX as usize => {
                frame.push(0x80 | 126);
                frame.extend((n as u16).to_be_bytes());
            }
            n => {
                frame.push(0x80 | 127);
                frame.extend((n as u64).to_be_bytes());
            }
        }
        frame.extend([0u8; 4]); // masking key
        frame.extend_from_slice(payload);
        self.stream.write_all(&frame).await
    }

    /// Awaits the next text frame's payload, answering pings and skipping
    /// pongs/binary along the way. Bounded by `timeout`.
    pub async fn next_text(&mut self, timeout: Duration) -> std::io::Result<String> {
        tokio::time::timeout(timeout, self.next_text_inner())
            .await
            .map_err(|_| other("timed out waiting for a websocket text frame"))?
    }

    async fn next_text_inner(&mut self) -> std::io::Result<String> {
        loop {
            let mut header = [0u8; 2];
            self.stream.read_exact(&mut header).await?;
            let fin = header[0] & 0x80 != 0;
            let opcode = header[0] & 0x0f;
            if header[1] & 0x80 != 0 {
                return Err(other("server frames must not be masked"));
            }
            let mut len = u64::from(header[1] & 0x7f);
            if len == 126 {
                let mut ext = [0u8; 2];
                self.stream.read_exact(&mut ext).await?;
                len = u64::from(u16::from_be_bytes(ext));
            } else if len == 127 {
                let mut ext = [0u8; 8];
                self.stream.read_exact(&mut ext).await?;
                len = u64::from_be_bytes(ext);
            }
            if !fin {
                return Err(other("fragmented websocket frames are unsupported"));
            }
            if len > 16 * 1024 * 1024 {
                return Err(other("oversized websocket frame"));
            }
            let mut payload = vec![0u8; len as usize];
            self.stream.read_exact(&mut payload).await?;
            match opcode {
                0x1 => {
                    return String::from_utf8(payload)
                        .map_err(|e| other(format!("non-utf8 text frame: {e}")));
                }
                0x8 => return Err(other("server closed the websocket")),
                0x9 => {
                    // ping -> pong, echoing the (always small) payload.
                    let mut pong = vec![0x8au8, 0x80 | payload.len() as u8];
                    pong.extend([0u8; 4]);
                    pong.extend_from_slice(&payload);
                    self.stream.write_all(&pong).await?;
                }
                _ => {} // pong / binary: not part of the panel protocol, skip
            }
        }
    }

    /// [`snapshot_until`](Self::snapshot_until) with no predicate: awaits
    /// the panel's next snapshot as-is.
    pub async fn next_snapshot(&mut self, deadline: Duration, what: &str) -> serde_json::Value {
        self.snapshot_until(deadline, what, |_| true).await
    }

    /// Reads owner snapshots (published at 10 Hz) until `predicate` holds,
    /// bounded by `deadline`. The loop is paced by the snapshot stream itself,
    /// never by sleeps; on expiry it panics with the last snapshot seen.
    pub async fn snapshot_until(
        &mut self,
        deadline: Duration,
        what: &str,
        mut predicate: impl FnMut(&serde_json::Value) -> bool,
    ) -> serde_json::Value {
        let end = tokio::time::Instant::now() + deadline;
        let mut last = serde_json::Value::Null;
        while tokio::time::Instant::now() < end {
            let text = self
                .next_text(Duration::from_secs(5))
                .await
                .unwrap_or_else(|e| panic!("{what}: snapshot stream failed: {e}"));
            let snapshot: serde_json::Value =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{what}: bad JSON: {e}"));
            if predicate(&snapshot) {
                return snapshot;
            }
            last = snapshot;
        }
        panic!("{what}: not observed before the deadline; last snapshot: {last}");
    }
}

/// The snapshot's entry for the arm named `name`: the arm, its gripper, its
/// deadman and its health. Null when the snapshot names no such arm, so a
/// predicate over it reads false.
pub fn arm<'a>(snapshot: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    static NO_ARM: serde_json::Value = serde_json::Value::Null;
    snapshot["arms"]
        .as_array()
        .and_then(|arms| arms.iter().find(|entry| entry["name"] == name))
        .unwrap_or(&NO_ARM)
}

/// Element-wise approximate equality for joint vectors crossing the wire.
pub fn approx_eq(actual: &[f64], expected: &[f64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-9)
}

/// Boots the node over the harness as `config` binds it, with the fault that
/// ends it once its setup has returned. Nothing answers the node's ask for the
/// robot's model.
pub async fn boot(config: Config) -> peppygen::Result<(Harness, Mocks, Fault)> {
    let fault = Fault::default();
    let raised = fault.clone();
    let (harness, mocks) = Harness::start_with(config, move |params, runner| {
        openarm_web_commander::setup(params, runner, raised)
    })
    .await?;
    Ok((harness, mocks, fault))
}

/// Answers the node's ask for the robot's model with `model`.
pub async fn answer_identity(mocks: &Mocks, model: &str) -> peppygen::Result<()> {
    use peppygen::mock::deps::robot_identity::get_identity::ResponseData;
    mocks
        .deps
        .robot_identity
        .get_identity
        .next_request(IDENTITY_BUDGET)
        .await?
        .respond(ResponseData {
            robot: "alpha".to_owned(),
            model: model.to_owned(),
            core_node: "harness".to_owned(),
        })
        .await
}

/// Boots the node over the harness as `config` binds it, its robot's
/// initializer naming [`MODEL`].
pub async fn start(config: Config) -> peppygen::Result<(Harness, Mocks)> {
    let (harness, mocks, _fault) = boot(config).await?;
    answer_identity(&mocks, MODEL).await?;
    Ok((harness, mocks))
}

/// Waits for the node's `setup` to return, failing the test after
/// `SETUP_BUDGET`.
pub async fn await_setup_return(harness: &Harness) {
    let deadline = tokio::time::Instant::now() + SETUP_BUDGET;
    while !harness.setup_finished() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "setup must return within {SETUP_BUDGET:?}"
        );
        tokio::time::sleep(SETUP_POLL).await;
    }
}

/// Tears the harness down once the node's `setup` has returned, so a setup
/// error reaches the caller. Teardown aborts a setup still running after the
/// shutdown grace and reports it as a clean stop.
pub async fn shutdown_once_setup_returns(harness: Harness) -> peppygen::Result<()> {
    await_setup_return(&harness).await;
    harness.shutdown().await
}
