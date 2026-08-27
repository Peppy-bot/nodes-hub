//! The workspace-obstacle services: convex bodies an operator puts in the arms'
//! way at runtime, which the governor then keeps them off exactly as it keeps
//! them off the robot's own geometry.
//!
//! The collision model lives in the coordination loop, and an insertion has to
//! be weighed against the configurations only the loop holds, so a handler here
//! does no model work of its own: it parses the request, fits the hull (the
//! expensive half, deliberately paid off the control tick), and hands the
//! result to the loop over [`ObstacleRequest`], which answers on the request's
//! own reply channel.

use std::result::Result as StdResult;
use std::sync::Arc;
use std::sync::mpsc::{RecvTimeoutError, SyncSender, sync_channel};
use std::time::{Duration, Instant};

use bimanual_collision_model::{Obstacle, Point3, parse_binary_stl};
use peppygen::exposed_services::obstacles::{
    add_obstacle, clear_obstacles, list_obstacles, remove_obstacle,
};
use peppygen::{NodeRunner, Result};
use peppylib::runtime::CancellationToken;

use crate::streams::{RECEIVE_ERROR_BACKOFF, warn_throttled};
use tokio::sync::mpsc;
use tracing::warn;

/// Longest an obstacle name may be. Names are operator-facing labels, not
/// payloads.
const MAX_NAME_LEN: usize = 64;
/// Most points one obstacle may be fitted from. A hull fit is superlinear in
/// the cloud, and the fit runs while an operator waits on the reply.
const MAX_POINTS: usize = 100_000;
/// Under this, a serve call cannot have waited on a caller.
const INSTANT_RETURN: Duration = Duration::from_millis(1);

/// How long a handler waits for the coordination loop to answer. The loop
/// drains these every tick, so this is only ever hit when it is not ticking.
const REPLY_TIMEOUT: Duration = Duration::from_secs(2);

/// A request that was queued but never answered may still be applied by a loop
/// that resumes ticking, so the reply says so rather than reporting a failure
/// the caller cannot rely on.
fn no_answer() -> String {
    format!(
        "no answer within {:?}; the request may still take effect, so check list_obstacles",
        REPLY_TIMEOUT
    )
}
const LOOP_GONE: &str = "the coordination loop is gone";
const TOO_MANY: &str = "too many obstacle requests in flight";

/// What the coordination loop made of a request: an operator-facing message
/// either way, and whether it happened.
pub type ObstacleOutcome = StdResult<String, String>;

/// A request the coordination loop answers on its way through a tick, with the
/// channel to answer on. The loop owns the collision model and the measured
/// state, so every one of these is decided there and nowhere else.
pub enum ObstacleRequest {
    /// Boxed because a fitted obstacle carries its whole hull, and the other
    /// variants would otherwise pay for it in every queued message.
    Add {
        obstacle: Box<Obstacle>,
        reply: SyncSender<ObstacleOutcome>,
    },
    Remove {
        name: String,
        reply: SyncSender<ObstacleOutcome>,
    },
    Clear {
        reply: SyncSender<String>,
    },
    List {
        reply: SyncSender<Vec<String>>,
    },
}

pub async fn run_add_obstacle(
    runner: Arc<NodeRunner>,
    requests: mpsc::Sender<ObstacleRequest>,
    token: CancellationToken,
) -> Result<()> {
    serve(&token, "add_obstacle", || {
        add_obstacle::handle_next_request(&runner, |req| {
            let outcome = parse(&req.data).and_then(|obstacle| {
                ask_outcome(&requests, |reply| ObstacleRequest::Add {
                    obstacle: Box::new(obstacle),
                    reply,
                })
            });
            let (success, message) = reported(outcome);
            Ok(add_obstacle::Response::new(success, message))
        })
    })
    .await
}

pub async fn run_remove_obstacle(
    runner: Arc<NodeRunner>,
    requests: mpsc::Sender<ObstacleRequest>,
    token: CancellationToken,
) -> Result<()> {
    serve(&token, "remove_obstacle", || {
        remove_obstacle::handle_next_request(&runner, |req| {
            let outcome = parse_name(&req.data.name).and_then(|name| {
                ask_outcome(&requests, |reply| ObstacleRequest::Remove { name, reply })
            });
            let (success, message) = reported(outcome);
            Ok(remove_obstacle::Response::new(success, message))
        })
    })
    .await
}

pub async fn run_clear_obstacles(
    runner: Arc<NodeRunner>,
    requests: mpsc::Sender<ObstacleRequest>,
    token: CancellationToken,
) -> Result<()> {
    serve(&token, "clear_obstacles", || {
        clear_obstacles::handle_next_request(&runner, |_| {
            let (success, message) =
                reported(ask(&requests, |reply| ObstacleRequest::Clear { reply }));
            Ok(clear_obstacles::Response::new(success, message))
        })
    })
    .await
}

pub async fn run_list_obstacles(
    runner: Arc<NodeRunner>,
    requests: mpsc::Sender<ObstacleRequest>,
    token: CancellationToken,
) -> Result<()> {
    serve(&token, "list_obstacles", || {
        list_obstacles::handle_next_request(&runner, |_| {
            let response = match ask(&requests, |reply| ObstacleRequest::List { reply }) {
                Ok(names) => list_obstacles::Response::new(
                    true,
                    format!("{} obstacle(s) in force", names.len()),
                    names,
                ),
                Err(message) => list_obstacles::Response::new(false, message, Vec::new()),
            };
            Ok(response)
        })
    })
    .await
}

/// How many back-to-back instant returns read as a fault rather than traffic.
/// These services are driven by an operator, so a hundred requests inside one
/// backoff period is not a burst anyone is producing.
const INSTANT_RETURNS_BEFORE_BACKOFF: u32 = 100;

/// Serve one request at a time until the node shuts down. A handler error is
/// one request going wrong, not a reason to take down two arms mid-motion, so
/// it is logged and the next request is served.
///
/// Two faults would otherwise spin this task at full rate beside a 100 Hz
/// control loop: a call that fails immediately every time, and a closed
/// subscription, which the generated handler reports as an ordinary `Ok`
/// having served nothing. Neither can be waited on, so both are caught the
/// same way, by how fast the call came back, and answered with the backoff and
/// throttled warning the inbound listeners already use.
async fn serve<F, Fut>(token: &CancellationToken, service: &str, mut next: F) -> Result<()>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let mut last_warning = None;
    let mut instant_returns = 0;
    loop {
        let started = Instant::now();
        tokio::select! {
            // Biased so shutdown wins a race it would otherwise only usually
            // win, and a cancelled service cannot take one more request.
            biased;
            _ = token.cancelled() => return Ok(()),
            result = next() => {
                if let Err(e) = result {
                    warn_throttled(&mut last_warning, || {
                        warn!("{service} handler error: {e}");
                    });
                    tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                    continue;
                }
            }
        }
        // A served request took as long as its caller did; anything that
        // returned instantly, over and over, served nobody.
        instant_returns = if started.elapsed() < INSTANT_RETURN {
            instant_returns + 1
        } else {
            0
        };
        if instant_returns >= INSTANT_RETURNS_BEFORE_BACKOFF {
            warn_throttled(&mut last_warning, || {
                warn!(
                    "{service} returned {INSTANT_RETURNS_BEFORE_BACKOFF} times without \
                     waiting on a caller; backing off"
                );
            });
            instant_returns = 0;
            tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
        }
    }
}

/// [`ask`] for a request the loop answers with an outcome: a request that never
/// reached the loop and one the loop refused are the same refusal to the
/// caller, differing only in the message they carry.
fn ask_outcome(
    requests: &mpsc::Sender<ObstacleRequest>,
    request: impl FnOnce(SyncSender<ObstacleOutcome>) -> ObstacleRequest,
) -> ObstacleOutcome {
    ask(requests, request).unwrap_or_else(Err)
}

/// Put a request to the coordination loop and wait for its answer. The
/// generated handler is synchronous, so this is a blocking wait rather than an
/// await; `block_in_place` is what keeps it off the runtime's back. The loop
/// answers within a tick, so the timeout only ever fires on a loop that has
/// stopped ticking.
fn ask<T>(
    requests: &mpsc::Sender<ObstacleRequest>,
    request: impl FnOnce(SyncSender<T>) -> ObstacleRequest,
) -> StdResult<T, String> {
    let (reply, answer) = sync_channel(1);
    requests.try_send(request(reply)).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => TOO_MANY.to_string(),
        mpsc::error::TrySendError::Closed(_) => LOOP_GONE.to_string(),
    })?;
    tokio::task::block_in_place(|| answer.recv_timeout(REPLY_TIMEOUT)).map_err(|e| match e {
        RecvTimeoutError::Timeout => no_answer(),
        RecvTimeoutError::Disconnected => LOOP_GONE.to_string(),
    })
}

/// Split an outcome into the success flag and message every response carries.
fn reported(outcome: ObstacleOutcome) -> (bool, String) {
    match outcome {
        Ok(message) => (true, message),
        Err(message) => (false, message),
    }
}

/// Parse an add request into a fitted obstacle, or the reason it is not one.
/// The tolerance is the caller's, not ours: it is the cost the robot carries on
/// every tick this obstacle is live, so the contract requires it rather than
/// letting a default decide.
fn parse(request: &add_obstacle::RequestData) -> StdResult<Obstacle, String> {
    let name = parse_name(&request.name)?;
    let points = parse_points(&request.vertices, &request.stl)?;
    // Fitting a hull is far too slow to sit on an async worker: this is the
    // whole reason the loop is handed a fitted obstacle rather than a cloud.
    tokio::task::block_in_place(|| Obstacle::fit(&name, &points, request.tolerance_m))
        .map_err(|e| e.to_string())
}

/// Refuse a cloud over the point cap, which is what bounds how long an
/// operator waits on a fit.
fn within_cap(points: usize) -> StdResult<(), String> {
    if points > MAX_POINTS {
        return Err(format!(
            "{points} points, over the {MAX_POINTS} an obstacle may be fitted from"
        ));
    }
    Ok(())
}

fn parse_name(name: &str) -> StdResult<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("an obstacle needs a name".to_string());
    }
    let length = trimmed.chars().count();
    if length > MAX_NAME_LEN {
        return Err(format!(
            "obstacle name is {length} characters, over the {MAX_NAME_LEN} allowed"
        ));
    }
    Ok(trimmed.to_string())
}

/// Parse the two mutually exclusive geometry forms into one world-frame point
/// cloud. Whether that cloud bounds a solid is the fit's business, not this
/// one's; what is settled here is that exactly one form was sent and that it is
/// the shape and size it claims to be.
fn parse_points(vertices: &[f64], stl: &[u8]) -> StdResult<Vec<Point3<f64>>, String> {
    let points = match (vertices.is_empty(), stl.is_empty()) {
        (true, true) => return Err("send the obstacle as either vertices or an stl".to_string()),
        (false, false) => {
            return Err("send the obstacle as vertices or an stl, not both".to_string());
        }
        (false, true) => {
            if !vertices.len().is_multiple_of(3) {
                return Err(format!(
                    "{} vertex coordinates do not divide into x, y, z triples",
                    vertices.len()
                ));
            }
            // Counted from the request rather than from the cloud, so an
            // oversized vertex list is refused before it is assembled. The stl
            // branch below counts after parsing, which allocates its facets
            // first; that allocation is within a small factor of the payload
            // the transport already carried.
            within_cap(vertices.len() / 3)?;
            if vertices.iter().any(|c| !c.is_finite()) {
                return Err("a vertex coordinate is not a finite number".to_string());
            }
            vertices
                .chunks_exact(3)
                .map(|c| Point3::new(c[0], c[1], c[2]))
                .collect()
        }
        (true, false) => {
            let points = parse_binary_stl(stl)?;
            within_cap(points.len())?;
            points
        }
    };
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eight corners of a unit box, flat: the cloud an operator sends for a
    /// wall.
    fn box_vertices() -> Vec<f64> {
        let mut v = Vec::with_capacity(24);
        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    v.extend([x, y, z]);
                }
            }
        }
        v
    }

    #[test]
    fn one_geometry_form_is_required() {
        assert!(parse_points(&[], &[]).is_err(), "neither form sent");
        assert!(
            parse_points(&box_vertices(), &[1, 2, 3]).is_err(),
            "both forms sent"
        );
        assert!(parse_points(&box_vertices(), &[]).is_ok());
    }

    #[test]
    fn vertices_must_divide_into_triples() {
        let mut ragged = box_vertices();
        ragged.pop();
        assert!(parse_points(&ragged, &[]).is_err());
    }

    #[test]
    fn a_non_finite_coordinate_is_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut points = box_vertices();
            points[4] = bad;
            assert!(
                parse_points(&points, &[]).is_err(),
                "{bad} is not a coordinate"
            );
        }
    }

    #[test]
    fn a_cloud_over_the_cap_is_refused() {
        let too_many = vec![0.0; 3 * (MAX_POINTS + 1)];
        let err = parse_points(&too_many, &[]).expect_err("over the cap");
        assert!(err.contains("over the"), "{err}");
    }

    #[test]
    fn an_unparseable_stl_is_refused_with_its_reason() {
        let err = parse_points(&[], &[0u8; 10]).expect_err("not an stl");
        assert!(err.contains("too short"), "{err}");
    }

    #[test]
    fn a_name_must_be_present_and_bounded() {
        assert!(parse_name("").is_err());
        assert!(parse_name("   ").is_err());
        assert!(parse_name(&"w".repeat(MAX_NAME_LEN + 1)).is_err());
        // Counted in characters, not bytes, so a multi-byte name is not
        // refused for being long when it is not.
        assert!(parse_name(&"\u{00e9}".repeat(MAX_NAME_LEN)).is_ok());
        assert_eq!(parse_name("  wall  ").expect("named"), "wall");
    }

    /// `ask` runs a blocking wait on a runtime worker, so its tests need a
    /// real multi-thread runtime, exactly as the node has.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_request_reaches_the_loop_and_its_answer_comes_back() {
        let (requests, mut inbox) = mpsc::channel(8);
        tokio::spawn(async move {
            let Some(ObstacleRequest::List { reply }) = inbox.recv().await else {
                panic!("expected the list request");
            };
            let _ = reply.send(vec!["wall".to_string()]);
        });
        let names = ask(&requests, |reply| ObstacleRequest::List { reply }).expect("answered");
        assert_eq!(names, ["wall"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_loop_that_drops_the_answer_is_reported_not_waited_on() {
        let (requests, mut inbox) = mpsc::channel(8);
        tokio::spawn(async move {
            // Take the request and drop it, which drops the reply channel with
            // it: the loop died mid-request.
            let _ = inbox.recv().await;
        });
        let err = ask(&requests, |reply| ObstacleRequest::List { reply })
            .expect_err("no answer is coming");
        assert!(err.contains("coordination loop"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_silent_loop_is_reported_as_an_unknown_outcome() {
        // Nothing ever answers, so `ask` gives up on the deadline. The request
        // is still queued and a loop that resumes ticking would apply it, so
        // the message must not claim it failed.
        let (requests, _inbox) = mpsc::channel(8);
        let started = std::time::Instant::now();
        let err = ask(&requests, |reply| ObstacleRequest::List { reply })
            .expect_err("no answer is coming");
        assert!(started.elapsed() >= REPLY_TIMEOUT, "gave up early");
        assert!(err.contains("may still take effect"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_full_queue_is_refused_rather_than_queued_behind() {
        // Nothing drains, so the queue fills and the next request must come
        // back as a refusal instead of parking a handler on it.
        let (requests, _inbox) = mpsc::channel(1);
        requests
            .try_send(ObstacleRequest::List {
                reply: sync_channel(1).0,
            })
            .expect("the first fits");
        let err =
            ask(&requests, |reply| ObstacleRequest::List { reply }).expect_err("the queue is full");
        assert!(err.contains("in flight"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dead_loop_is_not_reported_as_a_busy_one() {
        let (requests, inbox) = mpsc::channel(8);
        drop(inbox);
        let err = ask(&requests, |reply| ObstacleRequest::List { reply })
            .expect_err("nothing is listening");
        assert!(err.contains("gone"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failing_service_is_retried_rather_than_fatal() {
        // A handler error must not take two arms down mid-motion, and a fault
        // that fails instantly every time must not spin the task either.
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let token = CancellationToken::new();
        let counted = calls.clone();
        let stop = token.clone();
        let served = tokio::spawn(async move {
            serve(&stop, "failing", || {
                let calls = counted.clone();
                async move {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Err(peppygen::Error::ServiceRequestStreamClosed)
                }
            })
            .await
        });
        tokio::time::sleep(RECEIVE_ERROR_BACKOFF * 3).await;
        token.cancel();
        assert!(
            served.await.expect("task").is_ok(),
            "a handler error was fatal"
        );
        let attempts = calls.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            attempts > 1,
            "the service gave up after {attempts} attempt(s)"
        );
        assert!(
            attempts < 50,
            "{attempts} attempts in three backoff periods is a busy loop"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_service_that_serves_nothing_stops_spinning() {
        // A closed subscription is reported as an ordinary Ok having served
        // nobody, so only the speed of the return gives it away.
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let token = CancellationToken::new();
        let counted = calls.clone();
        let stop = token.clone();
        let served = tokio::spawn(async move {
            serve(&stop, "empty", || {
                let calls = counted.clone();
                async move {
                    calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(())
                }
            })
            .await
        });
        tokio::time::sleep(RECEIVE_ERROR_BACKOFF * 3).await;
        token.cancel();
        assert!(served.await.expect("task").is_ok());
        let attempts = calls.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            attempts <= INSTANT_RETURNS_BEFORE_BACKOFF as usize * 6,
            "{attempts} calls in three backoff periods is still a busy loop"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancelled_service_stops_serving() {
        // The request never completes, so shutdown is the only way out.
        let token = CancellationToken::new();
        token.cancel();
        let result = tokio::time::timeout(
            REPLY_TIMEOUT,
            serve(&token, "cancelled", std::future::pending::<Result<()>>),
        )
        .await
        .expect("a cancelled service must return, not hang");
        assert!(result.is_ok());
    }

    #[test]
    fn a_cloud_at_the_cap_is_accepted() {
        // The refused side is covered above; this pins that the cap itself is
        // not off by one.
        assert!(parse_points(&vec![0.0; 3 * MAX_POINTS], &[]).is_ok());
    }
}
