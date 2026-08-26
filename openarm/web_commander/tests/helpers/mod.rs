//! Shared plumbing for the integration tests: launch parameters, the panel's
//! port selection, and a minimal WebSocket client for the commander's real
//! operator surface (the HTTP+WS panel). No WS client crate is in the node's
//! dependency set, so this speaks just enough RFC 6455 for the panel's
//! unfragmented text frames.
//!
//! NOTE: each test binary boots the node at most once. `ui::init_limits`
//! asserts it runs exactly once per process, so a second `Harness::start` in
//! the same binary would panic; every tests/*.rs file therefore holds exactly
//! one booting test.
#![allow(dead_code)]

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Valid launch parameters (every required field of the manifest schema):
/// governor on with the 0.005/0.02 band, a 0.5 m/s speed cap, v2 ranges, and
/// a 50 Hz command tick.
pub fn test_parameters() -> peppygen::Parameters {
    peppygen::Parameters {
        collision_governor_enabled: true,
        command_rate_hz: 50,
        d_safe: 0.02,
        d_stop: 0.005,
        hardware_version: "v2".to_string(),
        joint_jog_acceleration_rad_s2: 10.0,
        max_ee_velocity_m_s: 0.5,
        max_gripper_rate_frac_s: 6.0,
    }
}

/// Point the panel at loopback on a per-test-binary port (the production
/// default 8765 could collide with anything on the host). Must run before
/// `Harness::start`, which spawns the server task that reads these.
pub fn set_panel_env(port: u16) {
    // SAFETY: called at the very start of the test, before the harness boots
    // the node; nothing else in this process reads or writes the environment
    // concurrently at that point.
    unsafe {
        std::env::set_var("PEPPY_JC_PORT", port.to_string());
        std::env::set_var("PEPPY_JC_BIND_IP", "127.0.0.1");
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

    /// Connects to the panel, retrying while the spawned server task is still
    /// binding its listener. Bounded: panics if the panel never accepts.
    pub async fn connect(port: u16) -> Self {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            match Self::connect_once(port).await {
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

/// Element-wise approximate equality for joint vectors crossing the wire.
pub fn approx_eq(actual: &[f64], expected: &[f64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-9)
}
