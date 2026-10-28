//! HTTP+WS UI on http_host, on http_port when it is free and on an
//! operating-system port otherwise. The WS exposes unauthenticated motion
//! control, so only run on a trusted network; bind http_host to 127.0.0.1 to
//! restrict it to loopback.
//!
//! This is only the transport: every text frame is decoded to a [`Command`] and sent to
//! the state owner, and every snapshot the owner publishes is forwarded to the browser.
//! The owner (see [`crate::owner`]) is the sole reader/writer of `UiState`.

use std::io::ErrorKind;
use std::net::SocketAddr;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use openarm_description::Side;
use peppylib::runtime::CancellationToken;
use serde::Deserialize;
use srs_model::JointVec;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tracing::warn;

use crate::owner::UiMsg;

/// Why the operator panel never opened, or stopped serving. Every variant
/// wraps an `io::Error` and names the step that produced it.
#[derive(Debug, thiserror::Error)]
pub enum UiError {
    #[error(
        "bind the operator panel to {addr}: {source}. Set http_host and http_port to an address this machine can serve."
    )]
    Bind {
        addr: SocketAddr,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "bind the operator panel to a port chosen by the operating system, after {preferred} was already in use: {source}. Free ports on this host, or set http_port to one this machine can bind."
    )]
    Fallback {
        preferred: SocketAddr,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "read back the address the operator panel bound: {0}. Set http_host and http_port to an address this machine can serve."
    )]
    BoundAddress(#[source] std::io::Error),

    #[error("serve the operator panel: {0}")]
    Serve(#[source] std::io::Error),
}

const INDEX_HTML: &str = include_str!("../static/index.html");

// Binding this port has the operating system choose a free one, which is what
// the panel falls back to when the launcher's port is already taken.
pub const ANY_PORT: u16 = 0;

#[derive(Clone)]
struct AppState {
    // Operator input to the owner: decoded commands and the disconnect signal.
    command_tx: mpsc::Sender<UiMsg>,
    // The owner's latest pre-serialized snapshot; forwarded verbatim to the browser.
    snapshot_rx: watch::Receiver<String>,
    token: CancellationToken,
}

/// The panel's listening socket, and the address it answers on.
///
/// Holding the bound socket is what makes the address true: it is taken before
/// the node reports ready and stays owned until [`Panel::serve`] returns, so
/// nothing between the two can claim the port the operator was told about.
#[derive(Debug)]
pub struct Panel {
    listener: TcpListener,
    address: SocketAddr,
}

impl Panel {
    /// Take `preferred`, or a port the operating system picks on the same host
    /// address when another process already holds `preferred`. [`ANY_PORT`]
    /// asks for an operating-system port outright; the node refuses it as a
    /// launch parameter, in [`crate::node`], before it reaches here.
    ///
    /// Only a port conflict falls back. A mistyped host, a privileged port, or
    /// any other bind failure is returned, so it reaches the operator as a
    /// refusal naming what to fix.
    pub async fn bind(preferred: SocketAddr) -> std::result::Result<Self, UiError> {
        let listener = match TcpListener::bind(preferred).await {
            Ok(listener) => listener,
            Err(source) if source.kind() == ErrorKind::AddrInUse => {
                warn!("{preferred} is already in use; taking a port from the operating system");
                TcpListener::bind(SocketAddr::new(preferred.ip(), ANY_PORT))
                    .await
                    .map_err(|source| UiError::Fallback { preferred, source })?
            }
            Err(source) => {
                return Err(UiError::Bind {
                    addr: preferred,
                    source,
                });
            }
        };
        let address = listener.local_addr().map_err(UiError::BoundAddress)?;
        Ok(Self { listener, address })
    }

    /// Where the panel is answering: the address every log line and browser URL
    /// is built from.
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Serve the panel on the bound socket until the token is cancelled.
    pub async fn serve(
        self,
        command_tx: mpsc::Sender<UiMsg>,
        snapshot_rx: watch::Receiver<String>,
        token: CancellationToken,
    ) -> std::result::Result<(), UiError> {
        let app_state = AppState {
            command_tx,
            snapshot_rx,
            token: token.clone(),
        };
        let app = Router::new()
            .route("/", get(index))
            .route("/ws", get(ws_upgrade))
            .with_state(app_state);

        axum::serve(self.listener, app)
            .with_graceful_shutdown(async move { token.cancelled().await })
            .await
            .map_err(UiError::Serve)?;
        Ok(())
    }
}

async fn index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(app): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_handle(socket, app))
}

async fn ws_handle(mut socket: WebSocket, app: AppState) {
    let mut snapshots = app.snapshot_rx.clone();
    // Send the latest snapshot immediately so a fresh connection paints at once, then
    // follow the owner's updates. `borrow_and_update` marks it seen, so `changed` next
    // waits for the following snapshot.
    let initial = snapshots.borrow_and_update().clone();
    if !initial.is_empty()
        && socket
            .send(Message::Text(Utf8Bytes::from(initial)))
            .await
            .is_err()
    {
        return;
    }
    loop {
        tokio::select! {
            _ = app.token.cancelled() => break,
            changed = snapshots.changed() => {
                if changed.is_err() {
                    break; // the owner is gone
                }
                let json = snapshots.borrow_and_update().clone();
                if !json.is_empty()
                    && socket.send(Message::Text(Utf8Bytes::from(json))).await.is_err()
                {
                    break;
                }
            }
            msg = socket.recv() => match msg {
                Some(Ok(Message::Text(text))) => match serde_json::from_str::<Command>(text.as_str()) {
                    Ok(cmd) => { let _ = app.command_tx.send(UiMsg::Command(cmd)).await; }
                    // Payload withheld: it carries operator free text (task
                    // names) and arbitrary bytes from the wire. The status line
                    // says the command was not read.
                    Err(e) => {
                        warn!(error = %e, payload_bytes = text.len(), "ws: bad command");
                        let _ = app.command_tx.send(UiMsg::Unreadable(e.to_string())).await;
                    }
                },
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(e)) => { warn!(error = %e, "ws: recv"); break; }
                _ => {}
            }
        }
    }
    // Releasing the panel drops the deadman and restores the governor default.
    let _ = app.command_tx.send(UiMsg::Disconnect).await;
}

// --------------------------- wire protocol ---------------------------

#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub(crate) enum Command {
    FireArm {
        side: Side,
        joints: JointVec,
        // Requested move duration (s); 0 = fastest safe.
        duration_s: f64,
        // The settle tolerance (rad, held by every joint) and timeout (s) the
        // backbone completes the move under.
        settle_tolerance_rad: f64,
        settle_timeout_s: f64,
    },
    // Toggle the streaming deadman for one side. While enabled, the command stream
    // emits that side's arm target and gripper opening on its pairing slots;
    // while disabled it emits nothing for the side.
    SetEnabled {
        side: Side,
        on: bool,
    },
    // Update an enabled arm's streamed target: the joint jog ramps the setpoint
    // toward it. Ignored while disabled.
    SetArmTarget {
        side: Side,
        joints: JointVec,
    },
    // Fire the backbone's planned Cartesian move_arm to a composed world-frame pose
    // (Actions-mode Execute): a governed straight-line move, not a jog. Refused
    // while the side streams, like fire_arm.
    FireArmPose {
        side: Side,
        position: [f64; 3],
        orientation: [f64; 4],
        // Requested move duration (s); 0 = fastest safe.
        duration_s: f64,
        // The settle tolerances (m, rad) and timeout (s) the backbone completes
        // the move under.
        settle_position_tolerance_m: f64,
        settle_orientation_tolerance_rad: f64,
        settle_timeout_s: f64,
    },
    // Update an enabled gripper's streamed opening. Ignored while disabled.
    SetGripperTarget {
        side: Side,
        position: f64,
    },
    // Set the gripper's effort cap, applied to both the streamed opening and
    // discrete moves. Not deadman-gated: Actions mode fires while disabled.
    SetGripperEffort {
        side: Side,
        max_effort: f64,
    },
    // Fire the backbone's discrete move_gripper (Actions-mode gripper Execute): a governed
    // open/close to the opening fraction `position`, not the streamed opening. Refused while the side
    // streams or a prior gripper move is in flight.
    FireGripper {
        side: Side,
        position: f64,
        // The longest (s) the backbone waits after the planned end of the travel
        // for the jaws to stand still.
        settle_timeout_s: f64,
    },
    // Set the backbone's self-collision-avoidance toggle (streamed continuously).
    SetCollision {
        enabled: bool,
    },
    // Retune the backbone's governor band and stream speed cap (streamed continuously).
    SetGovernorParams {
        d_stop: f64,
        d_safe: f64,
        max_ee_velocity_m_s: f64,
        max_gripper_rate_frac_s: f64,
    },
    // Play a named gesture from the baked library. Refused while its sides
    // stream, a move is in flight, or another gesture is playing.
    RunGesture {
        name: String,
    },
    // Stop the playing gesture; the involved sides hold where they are.
    StopGesture,
    // Start recording a dataset episode labeled with the operator's task text.
    // Refused when no recorder is bound or an episode is already recording.
    StartRecording {
        task: String,
    },
    // End the in-flight episode with a save (the recorder's cancel semantics).
    StopRecording,
    // Finalize and mirror the current session's dataset and open a fresh one;
    // the finished directory becomes replayable. Refused while recording.
    FinishSession,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use openarm_description::ARM_DOF;

    /// The value a panel input with `id` is prefilled with.
    fn prefilled(id: &str) -> f64 {
        let at = INDEX_HTML
            .find(&format!("id=\"${{side}}_{id}\" value=\""))
            .unwrap_or_else(|| panic!("the panel has no {id} input"));
        let value = &INDEX_HTML[at..]
            .split('"')
            .nth(3)
            .expect("a value attribute");
        value
            .parse()
            .unwrap_or_else(|e| panic!("{id} prefill {value}: {e}"))
    }

    // The backbone refuses a pose goal whose settle tolerance is below the
    // plan tolerance of its axis, so the panel's prefilled settle must cover
    // the plan tolerances it sends.
    #[test]
    fn the_prefilled_pose_settle_covers_the_plan_tolerances_the_panel_sends() {
        use crate::pose::{PLAN_ANGLE_TOL_RAD, PLAN_POS_TOL_M};
        assert!(prefilled("settle_pos") >= PLAN_POS_TOL_M);
        assert!(prefilled("settle_rot") >= PLAN_ANGLE_TOL_RAD);
        assert!(prefilled("settle_joints") > 0.0);
    }

    /// Every key of each object literal the page sends with `cmd: '<cmd>',`,
    /// in order: the shorthand keys and the `key:` keys alike. Each literal
    /// ends where its `ws.send(JSON.stringify({...}))` closes.
    fn page_payload_keys(cmd: &str) -> Vec<Vec<String>> {
        let marker = format!("cmd: '{cmd}',");
        INDEX_HTML
            .match_indices(&marker)
            .map(|(at, _)| {
                let end = INDEX_HTML[at..].find("}))").expect("the send closes");
                let literal = &INDEX_HTML[at..at + end];
                literal
                    .split(',')
                    .map(|entry| entry.split(':').next().expect("a key").trim().to_string())
                    .filter(|key| !key.is_empty())
                    .collect()
            })
            .collect()
    }

    // Every move command the page sends carries every field the server reads:
    // each literal, filled with a value of the right shape per key, parses as a
    // Command. A send site that leaves a field out is dropped unread.
    #[test]
    fn every_move_command_the_page_sends_parses_as_a_command() {
        for (cmd, shaped) in [
            (
                "fire_arm",
                vec![("joints", serde_json::json!(vec![0.0; ARM_DOF]))],
            ),
            (
                "fire_arm_pose",
                vec![
                    ("position", serde_json::json!([0.4, 0.1, 0.8])),
                    ("orientation", serde_json::json!([0.0, 0.0, 0.0, 1.0])),
                ],
            ),
            ("fire_gripper", vec![("position", serde_json::json!(0.5))]),
        ] {
            let sites = page_payload_keys(cmd);
            assert_eq!(sites.len(), 1, "the page sends {cmd} from one place");
            let payload: serde_json::Map<String, serde_json::Value> = sites[0]
                .iter()
                .map(|key| {
                    let value = match key.as_str() {
                        "cmd" => serde_json::json!(cmd),
                        "side" => serde_json::json!("left"),
                        other => shaped
                            .iter()
                            .find(|(name, _)| *name == other)
                            .map_or(serde_json::json!(1.0), |(_, v)| v.clone()),
                    };
                    (key.clone(), value)
                })
                .collect();
            serde_json::from_value::<Command>(serde_json::Value::Object(payload.clone()))
                .unwrap_or_else(|e| panic!("the page's {cmd} {payload:?} does not parse: {e}"));
        }
    }

    #[test]
    fn a_command_names_its_side_in_lowercase_only() {
        let set_enabled = |side: &str| {
            serde_json::from_str::<Command>(&format!(
                r#"{{"cmd":"set_enabled","side":"{side}","on":true}}"#
            ))
        };
        assert!(matches!(
            set_enabled("right"),
            Ok(Command::SetEnabled {
                side: Side::Right,
                on: true
            })
        ));
        let refused = set_enabled("Left").err().expect("refused").to_string();
        assert!(
            refused.starts_with("unknown variant `Left`, expected `left` or `right`"),
            "{refused}"
        );
    }

    // A panel that has not answered within this is not going to.
    const REPLY_BUDGET: Duration = Duration::from_secs(5);

    // Copies of one robot, all preferring one port.
    const COPIES: usize = 4;

    // -- the panel's socket ------------------------------------------------

    /// A listener on a free port, standing in for whatever else on the host
    /// holds the port a launcher asked for.
    async fn holder() -> (TcpListener, SocketAddr) {
        let listener = TcpListener::bind(loopback(ANY_PORT))
            .await
            .expect("an operating-system port is always available");
        let addr = listener
            .local_addr()
            .expect("a bound listener has an address");
        (listener, addr)
    }

    fn loopback(port: u16) -> SocketAddr {
        SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), port)
    }

    /// The panel answers a plain GET on its bound address, so a port is only
    /// reported once something is actually serving there.
    async fn get_index(addr: SocketAddr) -> String {
        tokio::time::timeout(REPLY_BUDGET, get_index_inner(addr))
            .await
            .unwrap_or_else(|_| panic!("the panel at {addr} must answer a GET"))
    }

    async fn get_index_inner(addr: SocketAddr) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(addr)
            .await
            .expect("the panel accepts connections on the address it reports");
        stream
            .write_all(
                format!("GET / HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").as_bytes(),
            )
            .await
            .expect("write the request");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .expect("read the response");
        String::from_utf8_lossy(&response).into_owned()
    }

    /// A panel serving until [`Serving::shut_down`] is awaited. The channel
    /// senders ride along so the panel's receivers stay open under test.
    struct Serving {
        address: SocketAddr,
        token: CancellationToken,
        server: tokio::task::JoinHandle<std::result::Result<(), UiError>>,
        _command_rx: mpsc::Receiver<UiMsg>,
        _snapshot_tx: watch::Sender<String>,
    }

    impl Serving {
        fn start(panel: Panel) -> Self {
            let (command_tx, _command_rx) = mpsc::channel(1);
            let (_snapshot_tx, snapshot_rx) = watch::channel(String::new());
            let token = CancellationToken::new();
            let address = panel.address();
            let server = tokio::spawn(panel.serve(command_tx, snapshot_rx, token.clone()));
            Self {
                address,
                token,
                server,
                _command_rx,
                _snapshot_tx,
            }
        }

        /// Stop serving and wait for the server to finish, which is when the
        /// socket is released.
        async fn shut_down(self) {
            self.token.cancel();
            self.server
                .await
                .expect("the server task runs")
                .expect("a cancelled panel stops cleanly");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn copies_racing_for_a_free_port_all_survive() {
        // No holder: the copies contend for the preferred port with each
        // other, so whoever loses the bind must fall back rather than fail.
        let preferred = {
            let (listener, addr) = holder().await;
            drop(listener);
            addr
        };

        let panels = bind_concurrently(COPIES, preferred).await;
        let ports: std::collections::BTreeSet<u16> =
            panels.iter().map(|p| p.address().port()).collect();
        assert_eq!(
            ports.len(),
            COPIES,
            "every copy needs its own socket: {ports:?}"
        );
        assert!(
            ports.contains(&preferred.port()),
            "one copy must win the port they all prefer: {ports:?}"
        );
    }

    #[tokio::test]
    async fn the_preferred_port_is_the_one_taken() {
        // Free a port this process just held, so the preference under test is
        // a port this host allowed a moment ago.
        let (listener, addr) = holder().await;
        drop(listener);

        let panel = Panel::bind(addr).await.expect("a free port binds");
        assert_eq!(panel.address(), addr);
    }

    #[tokio::test]
    async fn a_held_port_moves_the_panel_and_leaves_the_holder_serving() {
        let (listener, addr) = holder().await;

        let panel = Panel::bind(addr).await.expect("a held port falls back");
        assert_ne!(
            panel.address().port(),
            addr.port(),
            "the panel must not claim the port another process holds"
        );
        assert_eq!(panel.address().ip(), addr.ip(), "only the port moves");

        // The panel answers where it moved to.
        let serving = Serving::start(panel);
        assert!(get_index(serving.address).await.starts_with("HTTP/1.1 200"));

        // The holder is untouched: it still accepts on the port it owns.
        let connected = tokio::net::TcpStream::connect(addr);
        let accepted = tokio::time::timeout(REPLY_BUDGET, listener.accept());
        let (accepted, connected) = tokio::join!(accepted, connected);
        accepted
            .expect("the holder must still be accepting on its port")
            .expect("the holder still owns its port");
        connected.expect("and still accepts connections");

        serving.shut_down().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_copies_each_serve_their_own_socket() {
        let (_listener, addr) = holder().await;

        // Every copy prefers the held port, as copies of one robot launcher do.
        let panels = bind_concurrently(COPIES, addr).await;
        let ports: std::collections::BTreeSet<u16> =
            panels.iter().map(|p| p.address().port()).collect();
        assert_eq!(
            ports.len(),
            COPIES,
            "every copy needs its own socket: {ports:?}"
        );
        assert!(
            !ports.contains(&addr.port()),
            "no copy may take the held port"
        );

        for panel in panels {
            let serving = Serving::start(panel);
            assert!(
                get_index(serving.address).await.starts_with("HTTP/1.1 200"),
                "every copy serves its own panel at {}",
                serving.address
            );
            serving.shut_down().await;
        }
    }

    /// Bind `count` panels at once, all preferring `addr`.
    async fn bind_concurrently(count: usize, addr: SocketAddr) -> Vec<Panel> {
        let together = std::sync::Arc::new(tokio::sync::Barrier::new(count));
        let binds: Vec<_> = (0..count)
            .map(|_| {
                let together = together.clone();
                tokio::spawn(async move {
                    together.wait().await;
                    Panel::bind(addr).await
                })
            })
            .collect();
        let mut panels = Vec::with_capacity(count);
        for bind in binds {
            panels.push(
                bind.await
                    .expect("the bind task runs")
                    .expect("a held port falls back"),
            );
        }
        panels
    }

    #[tokio::test]
    async fn a_failure_that_is_not_a_conflict_is_refused() {
        // TEST-NET-1 is not an address of this host, so the bind fails for a
        // reason no other port can fix.
        let unbindable = SocketAddr::new(
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(192, 0, 2, 1)),
            34567,
        );
        if TcpListener::bind(unbindable).await.is_ok() {
            panic!(
                "this host permits binding {unbindable} (ip_nonlocal_bind), so the refusal path cannot be exercised here"
            );
        }
        let refused = Panel::bind(unbindable)
            .await
            .expect_err("an address this host does not hold cannot be served");
        assert!(
            matches!(refused, UiError::Bind { addr, .. } if addr == unbindable),
            "the refusal must name the address that failed, got: {refused}"
        );
        assert!(
            refused.to_string().contains("http_host and http_port"),
            "the refusal must name the parameters to change, got: {refused}"
        );
    }

    #[tokio::test]
    async fn a_stopped_panel_releases_its_port() {
        let panel = Panel::bind(loopback(ANY_PORT))
            .await
            .expect("an operating-system port is always available");
        let serving = Serving::start(panel);
        let served = serving.address;
        assert!(get_index(served).await.starts_with("HTTP/1.1 200"));

        serving.shut_down().await;

        // A copy that leaves the stack gives its port back, so the one that
        // rejoins can take it.
        let rebound = Panel::bind(served)
            .await
            .expect("the released port binds again");
        assert_eq!(rebound.address(), served);
    }
}
