//! transform_lookup's lookup_transform: requests answered one at a time in the order they
//! arrive, each checked as the contract orders it, `rejected` for a value the member does
//! not take, then the lookup in the robot's tree, which waits up to the request's timeout_s
//! on the node's clock, counted from when the server takes the request, for the tree to
//! answer. A lookup places the source frame at source_timestamp in the target frame at
//! target_timestamp, through the root of the source frame's chain when the two times differ,
//! as tf2's advanced lookupTransform does. A failed lookup gives the failure kind of its
//! first failure, its message naming the cause.

use std::sync::Arc;
use std::time::Duration;

use control_core::geometry::{lookup_time, report_timestamp};
use control_core::robot_tree::RobotTree;
use control_core::serving;
use control_core::sync::wait_for;
use control_core::time::wait_from_secs;
use frame_tree::{Found, FrameName, FrameNameError, LookupError, LookupTime};
use peppygen::NodeRunner;
use peppygen::exposed_services::transform_lookup::lookup_transform::{
    self, Failure, FailureKind, RequestData, Response,
};
use peppylib::runtime::CancellationToken;

/// The longest wait a request of lookup_transform takes (s): its timeout_s is from 0 to 1.
const MAX_TIMEOUT_S: f64 = 1.0;

/// A request of lookup_transform as the tree takes it.
#[derive(Debug, Clone, PartialEq)]
struct Lookup {
    target: FrameName,
    /// The time of the target frame's place.
    target_time: LookupTime,
    source: FrameName,
    /// The time of the source frame's place.
    source_time: LookupTime,
    /// How long the lookup waits for the tree to answer it, on the node's clock.
    timeout: Duration,
}

impl Lookup {
    /// The lookup `request` asks for, or `rejected` naming the first value of it the
    /// member does not take, in the order target_frame, source_frame, source_timestamp,
    /// target_timestamp, timeout_s. An absent timestamp is the latest time.
    fn parse(request: &RequestData) -> Result<Self, Failure> {
        let frame = |field: &str, name: &str| {
            FrameName::parse(name).map_err(|FrameNameError| {
                rejected(format!(
                    "{field} is empty: send the name of a link of the robot's URDF"
                ))
            })
        };
        let target = frame("target_frame", &request.target_frame)?;
        let source = frame("source_frame", &request.source_frame)?;
        let time = |field: &str, timestamp| {
            lookup_time(timestamp).map_err(|e| rejected(format!("{field}: {e}")))
        };
        let source_time = time("source_timestamp", request.source_timestamp)?;
        let target_time = time("target_timestamp", request.target_timestamp)?;
        if !(0.0..=MAX_TIMEOUT_S).contains(&request.timeout_s) {
            return Err(rejected(format!(
                "timeout_s is {} s: send a timeout_s from 0 to {MAX_TIMEOUT_S}",
                request.timeout_s
            )));
        }
        let timeout = wait_from_secs(request.timeout_s)
            .expect("a timeout_s from 0 to 1 is a wait wait_from_secs takes");
        Ok(Self {
            target,
            target_time,
            source,
            source_time,
            timeout,
        })
    }
}

fn rejected(message: String) -> Failure {
    Failure::new(FailureKind::Rejected, message)
}

/// The failure of a lookup, of the kind of the contract that names it.
fn failure_of(error: &LookupError) -> Failure {
    let kind = FailureKind::parse(error.failure_kind())
        .expect("lookup_transform names each failure kind of a lookup");
    Failure::new(kind, error.to_string())
}

/// The answer of `lookup`, a transform record: the place of its source frame
/// (child_frame) in its target frame (frame), stamped with the time of the target frame's
/// place, with no timestamp for an answer that holds at every time.
fn response_of(lookup: &Lookup, found: Found) -> Response {
    Response::new(
        lookup.target.to_string(),
        lookup.source.to_string(),
        report_timestamp(found.time),
        found.placement.position(),
        found.placement.orientation().to_xyzw(),
    )
}

/// The lookup of `lookup` in `tree`: once the tree answers it, or its failure once its
/// timeout has passed on the node's clock since the call, when the server takes the
/// request. A node that stops ends the wait at once with the lookup as the tree stands.
async fn look_up(
    tree: &RobotTree,
    lookup: &Lookup,
    stopping: &CancellationToken,
) -> Result<Found, LookupError> {
    let Lookup {
        target,
        target_time,
        source,
        source_time,
        timeout,
    } = lookup;
    let within =
        |timeout| tree.lookup_within_through(target, *target_time, source, *source_time, timeout);
    tokio::select! {
        biased;
        () = stopping.cancelled() => within(Duration::ZERO).await,
        found = within(*timeout) => found,
    }
}

/// Serves lookup_transform from `tree` for as long as the node runs, one request at a
/// time in the order they arrive: a request waits for the tree on the task that serves the
/// service, and the requests after it wait behind it.
pub(crate) fn serve(runner: Arc<NodeRunner>, tree: Arc<RobotTree>) {
    let token = runner.cancellation_token().clone();
    tokio::spawn(async move {
        serving::serve("lookup_transform", token.cancelled(), || {
            lookup_transform::handle_requests(&runner, |request| {
                let lookup = Lookup::parse(&request.data)?;
                wait_for(look_up(&tree, &lookup, &token))
                    .map(|found| response_of(&lookup, found))
                    .map_err(|e| failure_of(&e))
            })
        })
        .await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::UNIX_EPOCH;

    use control_core::node_clock::{NodeClock, TickFeed};
    use frame_tree::{
        BrokenTree, HeldStamps, MovingMessage, Placement, Publisher, Stamp, StaticMessage,
        TransformName, TransformRecord,
    };

    const NANOS_PER_SECOND: u64 = 1_000_000_000;
    const WINDOW: Duration = Duration::from_secs(10);
    /// The longest wait, as a request's lookup takes it.
    const LONGEST_WAIT: Duration = Duration::from_secs(1);
    /// The time on the runtime's paused clock a test grants a task to end. The runtime moves
    /// that clock only once every task waits, so it runs out at once for a task that waits
    /// on, and never for one that ends.
    const TO_END: Duration = Duration::from_secs(1);

    const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

    fn request() -> RequestData {
        RequestData {
            target_frame: "base_link".into(),
            source_frame: "camera".into(),
            source_timestamp: None,
            target_timestamp: None,
            timeout_s: 0.5,
        }
    }

    /// The record of `child` placed in `frame` at `position`, unturned.
    fn record(frame: &str, child: &str, position: [f64; 3]) -> TransformRecord {
        TransformRecord {
            frame: frame.into(),
            child_frame: child.into(),
            position,
            orientation: IDENTITY,
        }
    }

    fn frame(name: &str) -> FrameName {
        FrameName::parse(name).expect("a frame name")
    }

    fn nanos(seconds: u64) -> u64 {
        seconds * NANOS_PER_SECOND
    }

    fn stamp(seconds: u64) -> Stamp {
        Stamp::from_nanos(nanos(seconds))
    }

    fn publisher(instance_id: &str) -> Publisher {
        Publisher {
            core_node: "core".into(),
            instance_id: instance_id.into(),
        }
    }

    fn refusal(request: RequestData) -> Failure {
        Lookup::parse(&request).expect_err("a value the member does not take")
    }

    /// A tree on a simulation clock at `seconds`, holding the camera fixed 1 m above the
    /// arm link and the arm 1 m along the base link's x at 20 s.
    fn tree_at(seconds: u64) -> (Arc<RobotTree>, TickFeed) {
        let (clock, ticks) = NodeClock::simulation();
        ticks.tick(nanos(seconds));
        let tree = RobotTree::new(WINDOW, clock).expect("a window above 0");
        tree.take_static(
            &publisher("init_inst"),
            &StaticMessage {
                transforms: vec![record("arm", "camera", [0.0, 0.0, 1.0])],
            },
        )
        .expect("taken");
        tree.take_moving(
            &publisher("backbone"),
            &MovingMessage {
                timestamp: UNIX_EPOCH + Duration::from_secs(20),
                transforms: vec![record("base_link", "arm", [1.0, 0.0, 0.0])],
                start_timestamp: UNIX_EPOCH + Duration::from_secs(1),
            },
        )
        .expect("taken");
        (Arc::new(tree), ticks)
    }

    /// The lookup of the camera in the base link at `seconds`, waiting up to `LONGEST_WAIT`, on a
    /// task of its own that `stopping` stops.
    fn looking_up_at(
        tree: &Arc<RobotTree>,
        seconds: u64,
        stopping: &CancellationToken,
    ) -> tokio::task::JoinHandle<Result<Found, LookupError>> {
        let lookup = Lookup {
            target: frame("base_link"),
            target_time: LookupTime::At(stamp(seconds)),
            source: frame("camera"),
            source_time: LookupTime::At(stamp(seconds)),
            timeout: LONGEST_WAIT,
        };
        let tree = tree.clone();
        let stopping = stopping.clone();
        tokio::spawn(async move { look_up(&tree, &lookup, &stopping).await })
    }

    /// Whether `waiting` still waits once every other task waits.
    async fn still_waits<T>(waiting: &mut tokio::task::JoinHandle<T>) -> bool {
        tokio::time::timeout(TO_END, waiting).await.is_err()
    }

    /// What `waiting` ends with, which it must end with before every other task waits.
    async fn ended<T>(waiting: tokio::task::JoinHandle<T>) -> T {
        tokio::time::timeout(TO_END, waiting)
            .await
            .expect("the lookup ends")
            .expect("the lookup's task ends without a panic")
    }

    #[test]
    fn a_request_parses_into_its_frames_its_times_and_its_wait() {
        assert_eq!(
            Lookup::parse(&request()),
            Ok(Lookup {
                target: frame("base_link"),
                target_time: LookupTime::Latest,
                source: frame("camera"),
                source_time: LookupTime::Latest,
                timeout: Duration::from_millis(500),
            })
        );
        let timed = Lookup::parse(&RequestData {
            source_timestamp: Some(UNIX_EPOCH + Duration::from_secs(20)),
            target_timestamp: Some(UNIX_EPOCH + Duration::from_secs(21)),
            timeout_s: 0.0,
            ..request()
        })
        .expect("a lookup at two times");
        assert_eq!(timed.source_time, LookupTime::At(stamp(20)));
        assert_eq!(timed.target_time, LookupTime::At(stamp(21)));
        assert_eq!(timed.timeout, Duration::ZERO);
        let longest = Lookup::parse(&RequestData {
            timeout_s: 1.0,
            ..request()
        })
        .expect("the longest wait");
        assert_eq!(longest.timeout, LONGEST_WAIT);
    }

    #[test]
    fn a_value_the_member_does_not_take_is_rejected_naming_it_in_the_contract_order() {
        let both_empty = refusal(RequestData {
            target_frame: String::new(),
            source_frame: String::new(),
            ..request()
        });
        assert_eq!(both_empty.kind, FailureKind::Rejected);
        assert_eq!(
            both_empty.message,
            "target_frame is empty: send the name of a link of the robot's URDF"
        );
        let source = refusal(RequestData {
            source_frame: String::new(),
            ..request()
        });
        assert_eq!(
            source.message,
            "source_frame is empty: send the name of a link of the robot's URDF"
        );
        let source_before_the_epoch = refusal(RequestData {
            source_timestamp: Some(UNIX_EPOCH - Duration::from_secs(1)),
            target_timestamp: Some(UNIX_EPOCH - Duration::from_secs(1)),
            ..request()
        });
        assert!(
            source_before_the_epoch
                .message
                .starts_with("source_timestamp: "),
            "{}",
            source_before_the_epoch.message
        );
        let target_before_the_epoch = refusal(RequestData {
            target_timestamp: Some(UNIX_EPOCH - Duration::from_secs(1)),
            ..request()
        });
        assert!(
            target_before_the_epoch
                .message
                .starts_with("target_timestamp: "),
            "{}",
            target_before_the_epoch.message
        );
        // 1.0000000004 s rounds to the 1 s a Duration holds, and it lies above 1 all the same.
        for timeout_s in [
            -0.1,
            -f64::MIN_POSITIVE,
            f64::NAN,
            f64::INFINITY,
            1.000_000_000_4,
            30.0,
            f64::MAX,
        ] {
            let refused = refusal(RequestData {
                timeout_s,
                ..request()
            });
            assert_eq!(refused.kind, FailureKind::Rejected);
            assert_eq!(
                refused.message,
                format!("timeout_s is {timeout_s} s: send a timeout_s from 0 to 1")
            );
        }
        let after_the_frames = refusal(RequestData {
            source_frame: String::new(),
            timeout_s: 30.0,
            ..request()
        });
        assert_eq!(
            after_the_frames.message,
            "source_frame is empty: send the name of a link of the robot's URDF",
            "the frames are checked before the wait"
        );
    }

    #[test]
    fn each_failure_of_a_lookup_gives_its_kind_and_names_its_cause() {
        let transform = TransformName {
            parent: frame("base_link"),
            child: frame("arm"),
        };
        let failures = [
            (
                LookupError::UnknownFrame {
                    frame: frame("gripper"),
                    held: vec![frame("arm"), frame("base_link")],
                },
                FailureKind::UnknownFrame,
                "gripper: look up one of the frames it holds: arm, base_link",
            ),
            (
                LookupError::NoChain {
                    target_frame: frame("torso"),
                    source_frame: frame("camera"),
                },
                FailureKind::NoChain,
                "torso",
            ),
            (
                LookupError::Extrapolation {
                    transform: transform.clone(),
                    requested: LookupTime::At(stamp(30)),
                    held: HeldStamps::One(stamp(20)),
                },
                FailureKind::Extrapolation,
                "base_link -> arm",
            ),
            (
                LookupError::BrokenTree(BrokenTree::TwoPublishers {
                    child: frame("arm"),
                    first: publisher("backbone"),
                    second: publisher("init_inst"),
                }),
                FailureKind::BrokenTree,
                "instance init_inst of core",
            ),
            (
                LookupError::BrokenTree(BrokenTree::Cycle {
                    transforms: vec![transform],
                }),
                FailureKind::BrokenTree,
                "base_link -> arm",
            ),
        ];
        for (error, kind, cause) in failures {
            let failure = failure_of(&error);
            assert_eq!(failure.kind, kind, "{error}");
            assert_eq!(failure.message, error.to_string());
            assert!(failure.message.contains(cause), "{}", failure.message);
        }
    }

    #[test]
    fn an_answer_is_a_transform_record_stamped_at_its_time_or_unstamped_for_every_time() {
        let lookup = Lookup::parse(&request()).expect("a lookup at the latest time");
        let found = |time| Found {
            placement: Placement::IDENTITY,
            time,
        };
        let timed = response_of(&lookup, found(Some(stamp(20))));
        assert_eq!(
            (timed.frame.as_str(), timed.child_frame.as_str()),
            ("base_link", "camera")
        );
        assert_eq!(timed.position, [0.0; 3]);
        assert_eq!(timed.orientation, IDENTITY);
        assert_eq!(timed.timestamp, Some(UNIX_EPOCH + Duration::from_secs(20)));
        assert_eq!(response_of(&lookup, found(None)).timestamp, None);
    }

    #[tokio::test(start_paused = true)]
    async fn a_wait_fails_once_the_hand_ticked_clock_passes_its_deadline() {
        let (tree, ticks) = tree_at(21);
        let mut waiting = looking_up_at(&tree, 30, &CancellationToken::new());
        assert!(
            still_waits(&mut waiting).await,
            "the wait starts at 21 s on the clock"
        );
        ticks.tick(nanos(22) - 1);
        assert!(still_waits(&mut waiting).await, "1 ns of the wait is left");
        ticks.tick(nanos(22));
        assert!(matches!(
            ended(waiting).await,
            Err(LookupError::Extrapolation { .. })
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn a_node_that_stops_ends_the_wait_at_once_with_the_lookup_as_the_tree_stands() {
        let (tree, _ticks) = tree_at(21);
        let stopping = CancellationToken::new();
        let mut waiting = looking_up_at(&tree, 30, &stopping);
        assert!(still_waits(&mut waiting).await, "no stamp of 30 s");
        stopping.cancel();
        assert!(matches!(
            ended(waiting).await,
            Err(LookupError::Extrapolation { .. })
        ));
    }
}
