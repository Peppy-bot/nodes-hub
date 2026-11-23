//! The host of a live run: calls through the clients of the codec API, the
//! move check before each call of a member with the mark moves_robot, the
//! latest message of each topic that a run observes, the open and close of
//! its leases, and its run clock.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use control_core::node_clock::NodeClock;
use futures::{Stream, StreamExt};
use peppy_codec::consumer::{
    ActionClient, ConsumerError, ConsumerIdentity, GoalHandle, GoalOutcome, MemberBinding,
    ServiceClient, TopicConsumer, contract_topic_codec,
};
use peppygen::NodeRunner;
use peppylib::messaging::ProducerRef;
use serde_json::Value as Json;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
use workflow_documents::{Name, RobotName};

use crate::contract::ContractShape;
use crate::engine::failure::{Failure, RESOURCE_UNAVAILABLE};
use crate::engine::host::{CallEnd, Event, Feedback, Host, Join, Joining};
use crate::engine::plan::{MemberKind, ObservationRead, Plan, Target, TopicKey};
use crate::leases::Leases;

/// How long a goal has to be admitted.
const ADMISSION: Duration = Duration::from_secs(10);

/// How long the host waits for the result of a goal: past every bound a
/// workflow gives.
const RESULT_WAIT: Duration = Duration::from_secs(86_400 + 60);

/// How long the host waits, after the result of a goal, for the end of the
/// feedback of the goal. The producer of a goal gives its result and then
/// closes its feedback, so the end follows the result at once.
const FEEDBACK_END: Duration = Duration::from_secs(5);

/// A client of one member.
enum Client {
    Service(ServiceClient, MemberBinding),
    Action(ActionClient, MemberBinding),
}

/// The host of one live run.
pub struct LiveHost {
    runner: Arc<NodeRunner>,
    identity: ConsumerIdentity,
    clients: BTreeMap<(String, String), Client>,
    latest: Arc<Mutex<BTreeMap<TopicKey, (Json, u64)>>>,
    /// Counts the messages of the topics, so a read that waits for one wakes.
    arrivals: Arc<watch::Sender<u64>>,
    /// The run clock of the run.
    clock: NodeClock,
    leases: Arc<Leases>,
    events: mpsc::UnboundedSender<Event>,
    cancel_timeout: Duration,
    subscriptions: Vec<tokio::task::JoinHandle<()>>,
}

impl Drop for LiveHost {
    fn drop(&mut self) {
        self.subscriptions
            .iter()
            .for_each(tokio::task::JoinHandle::abort);
    }
}

fn key_of(target: &Target) -> (String, String) {
    (target.sha256.clone(), target.member.to_string())
}

fn producer_of(target: &Target) -> ProducerRef {
    ProducerRef::new(
        target.producer.core_node.clone(),
        target.producer.instance.clone(),
    )
}

/// The feedback of `handle` as the codec decodes it, to the end of the stream
/// of its goal.
fn feedback_of(handle: GoalHandle) -> impl Stream<Item = Result<Json, ConsumerError>> + Send {
    futures::stream::unfold(handle, |mut handle| async move {
        let fed = handle.next_feedback().await.transpose()?;
        Some((fed, handle))
    })
}

/// The feedback of one goal, which a task of its own reads. A handle whose
/// feedback is read once follows it, and cancels its goal when more messages
/// than its bound are unread. The task takes each message as it comes, so
/// the count of the unread messages does not depend on how often the engine
/// polls the call, and the call takes the messages from the task each time
/// the engine polls it.
struct FeedbackReader {
    /// The messages that the task read and the call did not take yet; closed
    /// once the task read the end of the stream.
    messages: mpsc::UnboundedReceiver<Json>,
    task: tokio::task::JoinHandle<()>,
}

impl FeedbackReader {
    /// Reads `feedback`, the feedback of a goal of `member`, to its end. A
    /// message that does not convert is logged and skipped; any other
    /// failure of the stream is logged and ends the read.
    fn start(
        feedback: impl Stream<Item = Result<Json, ConsumerError>> + Send + 'static,
        member: String,
    ) -> FeedbackReader {
        let (read, messages) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let mut feedback = std::pin::pin!(feedback);
            while let Some(fed) = feedback.next().await {
                match fed {
                    Ok(message) => {
                        if read.send(message).is_err() {
                            return;
                        }
                    }
                    Err(ConsumerError::Conversion(error)) => tracing::warn!(
                        "a feedback message of {member} does not convert, and gives no report: {error}"
                    ),
                    Err(error) => {
                        tracing::warn!(
                            "the feedback of {member} stops: {error}; the step gives no more reports"
                        );
                        return;
                    }
                }
            }
        });
        FeedbackReader { messages, task }
    }

    /// The next message; `None` once the stream ended and the call took each
    /// message before its end.
    async fn next(&mut self) -> Option<Json> {
        self.messages.recv().await
    }

    /// Hands `sink` each message that is left, up to the end of the stream:
    /// the last messages of a goal can come after its result. The wait ends
    /// at the cancel of `cancel` and after `FEEDBACK_END`, each after the
    /// messages that came before it.
    async fn rest(mut self, sink: Feedback<'_>, cancel: &CancellationToken, member: &str) {
        let to_the_end = async {
            while let Some(message) = self.next().await {
                sink(message);
            }
        };
        tokio::select! {
            biased;
            () = to_the_end => {}
            () = cancel.cancelled() => {}
            () = tokio::time::sleep(FEEDBACK_END) => tracing::warn!(
                "the feedback of {member} did not end within {} s of the result of its goal",
                FEEDBACK_END.as_secs()
            ),
        }
    }
}

impl Drop for FeedbackReader {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl LiveHost {
    /// The host of a run of `plan`: a client for each member it calls, laid
    /// out once, and a subscription for each topic it observes.
    pub async fn new(
        runner: Arc<NodeRunner>,
        plan: &Plan,
        contracts: &BTreeMap<String, Arc<ContractShape>>,
        leases: Arc<Leases>,
        clock: NodeClock,
        events: mpsc::UnboundedSender<Event>,
    ) -> Result<LiveHost, String> {
        let processor = runner.processor();
        let identity = ConsumerIdentity {
            core_node: processor.bound_core_node().to_owned(),
            instance_id: processor.bound_instance_id().to_owned(),
        };
        let mut targets: Vec<&Target> = plan.calls.values().map(|call| &call.target).collect();
        targets.extend(
            plan.observations
                .values()
                .filter(|o| matches!(o.read, ObservationRead::Service { .. }))
                .map(|o| &o.target),
        );
        targets.extend(plan.monitor.iter().map(|monitor| &monitor.target));
        let mut clients = BTreeMap::new();
        for target in targets {
            let key = key_of(target);
            if clients.contains_key(&key) {
                continue;
            }
            let shape = contracts
                .get(&target.sha256)
                .ok_or_else(|| format!("the document of `{}` is gone", target.contract))?;
            let member = target.member.as_str();
            let binding =
                MemberBinding::of_contract(&shape.parsed, member).map_err(|e| e.to_string())?;
            let client = match target.kind {
                MemberKind::Service => Client::Service(
                    ServiceClient::of_contract(&shape.parsed, member).map_err(|e| e.to_string())?,
                    binding,
                ),
                MemberKind::Action => Client::Action(
                    ActionClient::of_contract(&shape.parsed, member).map_err(|e| e.to_string())?,
                    binding,
                ),
            };
            clients.insert(key, client);
        }
        let latest = Arc::new(Mutex::new(BTreeMap::new()));
        let arrivals = Arc::new(watch::Sender::new(0));
        let mut subscriptions = Vec::new();
        for observation in plan.observations.values() {
            if !matches!(observation.read, ObservationRead::Topic { .. }) {
                continue;
            }
            let target = &observation.target;
            let shape = contracts
                .get(&target.sha256)
                .ok_or_else(|| format!("the document of `{}` is gone", target.contract))?;
            let topic = shape
                .parsed
                .topic(target.member.as_str())
                .ok_or_else(|| format!("`{}` has no topic `{}`", target.contract, target.member))?
                .clone();
            let codec = contract_topic_codec(&shape.parsed, target.member.as_str())
                .map_err(|e| e.to_string())?;
            let binding = MemberBinding::of_contract(&shape.parsed, target.member.as_str())
                .map_err(|e| e.to_string())?;
            let mut consumer = TopicConsumer::of_producer(
                runner.messenger(),
                &identity,
                &binding,
                &producer_of(target),
                topic.qos_profile,
                topic.retention,
                codec,
            )
            .await
            .map_err(|e| e.to_string())?;
            let key = target.topic_key();
            let latest = Arc::clone(&latest);
            let arrivals = Arc::clone(&arrivals);
            let clock = clock.clone();
            subscriptions.push(tokio::spawn(async move {
                while let Some((_, message)) = consumer.next_message().await {
                    if let Ok(value) = consumer.decode(&message) {
                        let at = clock.now_ns();
                        latest
                            .lock()
                            .expect("no reader panics holding the topics")
                            .insert(key.clone(), (value, at));
                        arrivals.send_modify(|count| *count += 1);
                    }
                }
            }));
        }
        Ok(LiveHost {
            runner,
            identity,
            clients,
            latest,
            arrivals,
            clock,
            leases,
            events,
            cancel_timeout: plan.cancel_timeout,
            subscriptions,
        })
    }

    fn failure_of(error: &ConsumerError) -> Failure {
        match error {
            ConsumerError::Messaging(error) => match error.failure() {
                Some(failure) => {
                    Failure::new(failure.kind().to_string(), failure.message().to_owned())
                }
                None => Failure::new(RESOURCE_UNAVAILABLE, error.to_string()),
            },
            ConsumerError::UndeclaredFailureKind { failure, .. } => Failure::new(
                failure.kind().to_string(),
                format!(
                    "{} (a kind the contract does not declare)",
                    failure.message()
                ),
            ),
            ConsumerError::Conversion(error) => {
                Failure::new(crate::engine::failure::REJECTED, error.to_string())
            }
        }
    }

    /// Sends a goal and follows it to its end. When the engine gives a
    /// `feedback` sink, each feedback message of the goal goes to it, in
    /// order, up to the end of the stream or the cancel of the goal.
    async fn goal(
        &self,
        client: &ActionClient,
        binding: &MemberBinding,
        target: &Target,
        request: Json,
        cancel: CancellationToken,
        feedback: Option<Feedback<'_>>,
    ) -> CallEnd {
        let messenger = self.runner.messenger();
        let producer = producer_of(target);
        let handle = client
            .fire_goal(
                messenger,
                &self.identity,
                binding,
                &producer,
                &request,
                peppylib::messaging::DEFAULT_FEEDBACK_BOUND,
                ADMISSION,
            )
            .await;
        let handle = match handle {
            Ok(handle) => handle,
            Err(error) => return CallEnd::Refused(Self::failure_of(&error)),
        };
        if let Some(refusal) = handle.refusal() {
            return CallEnd::Refused(Failure::new(
                refusal.kind().to_string(),
                refusal.message().to_owned(),
            ));
        }
        // The cancel and the result are asked through the requests of the
        // goal, which borrow nothing of the handle: the reader of the
        // feedback owns the handle. A goal with no sink keeps its handle
        // unread, so the handle never follows the feedback.
        let requests = handle.requests();
        let member = format!(
            "{}.{} on {}",
            target.contract, target.member, target.producer.instance
        );
        let (mut reports, _unread) = match feedback {
            Some(sink) => {
                let reader = FeedbackReader::start(feedback_of(handle), member.clone());
                (Some((sink, reader)), None)
            }
            None => (None, Some(handle)),
        };
        let result = requests.result(messenger, RESULT_WAIT);
        tokio::pin!(result);
        // The cancel first, so a stream of feedback never holds it back; then
        // each feedback message that came, so the reports of a goal come as
        // the goal gives them.
        let outcome = loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    let settling = async {
                        let _ = requests.cancel(messenger, self.cancel_timeout).await;
                        requests.result(messenger, RESULT_WAIT).await
                    };
                    let settled = tokio::time::timeout(self.cancel_timeout, settling).await;
                    return end_after_cancel(settled.ok());
                }
                message = next_report(&mut reports), if reports.is_some() => {
                    match (message, &reports) {
                        (Some(message), Some((sink, _))) => sink(message),
                        _ => reports = None,
                    }
                }
                outcome = &mut result => break outcome,
            }
        };
        if let Some((sink, reader)) = reports {
            reader.rest(sink, &cancel, &member).await;
        }
        match outcome {
            Ok(GoalOutcome::Completed(result)) => CallEnd::Completed(result),
            Ok(GoalOutcome::Cancelled { reason, .. }) => CallEnd::Cancelled {
                asked: false,
                reason,
            },
            Ok(GoalOutcome::Failed(failure)) => CallEnd::Failed(Failure::new(
                failure.kind().to_string(),
                failure.message().to_owned(),
            )),
            Ok(GoalOutcome::Abandoned) => CallEnd::Failed(Failure::new(
                RESOURCE_UNAVAILABLE,
                "the producer dropped the goal",
            )),
            Ok(GoalOutcome::Expired) => CallEnd::Failed(Failure::new(
                RESOURCE_UNAVAILABLE,
                "the producer no longer keeps the result",
            )),
            Err(error) => CallEnd::Failed(Self::failure_of(&error)),
        }
    }
}

/// How a goal that the director cancelled ended, from its outcome within
/// the cancel timeout (`None` when none came): a goal that ended cancelled,
/// completed or failed settled; any other end leaves its robot in an
/// unknown state.
fn end_after_cancel(settled: Option<Result<GoalOutcome, ConsumerError>>) -> CallEnd {
    match settled {
        Some(Ok(
            GoalOutcome::Cancelled { .. } | GoalOutcome::Completed(_) | GoalOutcome::Failed(_),
        )) => CallEnd::Cancelled {
            asked: true,
            reason: "the director cancelled it".to_owned(),
        },
        _ => CallEnd::Unknown,
    }
}

/// The next feedback message of `reports`; `None` at the end of the stream,
/// and with no reports.
async fn next_report(reports: &mut Option<(Feedback<'_>, FeedbackReader)>) -> Option<Json> {
    let (_, reader) = reports.as_mut()?;
    reader.next().await
}

impl Host for LiveHost {
    fn now(&self) -> u64 {
        self.clock.now_ns()
    }

    fn wall_now(&self) -> u64 {
        crate::journal::wall_ns()
    }

    async fn until(&self, at: u64) {
        self.clock.until(at).await
    }

    async fn call(
        &self,
        target: &Target,
        request: Json,
        cancel: CancellationToken,
        feedback: Option<Feedback<'_>>,
    ) -> CallEnd {
        if target.moves_robot {
            match self.leases.check_move(&target.producer.instance).await {
                Ok(None) => {}
                Ok(Some(refused)) => {
                    return CallEnd::MoveRefused {
                        halted: refused.halted,
                        message: refused.message,
                    };
                }
                Err(message) => {
                    return CallEnd::Refused(Failure::new(RESOURCE_UNAVAILABLE, message));
                }
            }
        }
        if cancel.is_cancelled() {
            return CallEnd::Cancelled {
                asked: true,
                reason: "the director cancelled it before it was sent".to_owned(),
            };
        }
        match self.clients.get(&key_of(target)) {
            Some(Client::Service(client, binding)) => {
                let messenger = self.runner.messenger();
                match client
                    .call(
                        messenger,
                        &self.identity,
                        binding,
                        &producer_of(target),
                        &request,
                        None,
                    )
                    .await
                {
                    Ok(response) => CallEnd::Completed(response),
                    Err(error) => CallEnd::Refused(Self::failure_of(&error)),
                }
            }
            // A goal follows its feedback, its result and its cancel, a large
            // future: on the heap, so the future of each call, a call of a
            // service too, stays small in the futures of the engine, whose
            // polls nest on the stack of a worker.
            Some(Client::Action(client, binding)) => {
                Box::pin(self.goal(client, binding, target, request, cancel, feedback)).await
            }
            None => CallEnd::Refused(Failure::new(
                RESOURCE_UNAVAILABLE,
                format!("no client of `{}.{}`", target.contract, target.member),
            )),
        }
    }

    fn latest(&self, target: &Target) -> Option<(Json, u64)> {
        self.latest
            .lock()
            .expect("no reader panics holding the topics")
            .get(&target.topic_key())
            .cloned()
    }

    async fn message_arrives(&self, target: &Target) {
        let key = target.topic_key();
        let mut arrivals = self.arrivals.subscribe();
        let arrived = arrivals
            .wait_for(|_| {
                self.latest
                    .lock()
                    .expect("no reader panics holding the topics")
                    .contains_key(&key)
            })
            .await
            .is_ok();
        if !arrived {
            std::future::pending::<()>().await;
        }
    }

    async fn joining(&self, _name: &Name, _nth: u64) -> Joining {
        Joining::AsFound
    }

    fn joined(&self, _name: &Name, _nth: u64, _join: &Join) {}

    async fn set_lease_open(&self, robot: &RobotName, open: bool) -> Result<(), String> {
        self.leases.set_open(robot.as_str(), open).await
    }

    fn event(&self, event: Event) {
        let _ = self.events.send(event);
    }
}

#[cfg(test)]
mod tests {
    use peppylib::PeppyError;
    use peppylib::messaging::DEFAULT_FEEDBACK_BOUND;
    use serde_json::json;

    use super::*;

    /// The member of the goals of these tests, as the log names it.
    const MEMBER: &str = "test_arm.grab on alpha_arm";

    /// A sink that keeps each message it takes in `taken`.
    fn keeping(taken: &Mutex<Vec<Json>>) -> impl Fn(Json) + Send + Sync + '_ {
        |message| taken.lock().expect("no sink panics").push(message)
    }

    /// The reader reads the feedback of a goal to its end with no call that
    /// takes a message, so the handle of the goal never holds more unread
    /// messages than its bound while the engine does not poll the call.
    #[tokio::test]
    async fn the_reader_reads_the_feedback_to_its_end_while_the_call_takes_no_message() {
        let count = 4 * DEFAULT_FEEDBACK_BOUND.get();
        let feedback = futures::stream::iter((0..count).map(|number| Ok(json!(number))));
        let mut reader = FeedbackReader::start(feedback, MEMBER.to_owned());
        (&mut reader.task)
            .await
            .expect("the reader reads to the end of the stream");
        for number in 0..count {
            assert_eq!(reader.next().await, Some(json!(number)));
        }
        assert_eq!(reader.next().await, None);
    }

    /// A message that does not convert gives nothing and the read goes on;
    /// any other failure of the stream ends the read.
    #[tokio::test]
    async fn the_reader_skips_a_message_that_does_not_convert_and_stops_at_another_failure() {
        let feedback = futures::stream::iter([
            Ok(json!("first")),
            Err(ConsumerError::Conversion(
                "the message has no stage".to_owned().into(),
            )),
            Ok(json!("second")),
            Err(ConsumerError::Messaging(
                PeppyError::ActionFeedbackChannelClosed,
            )),
            Ok(json!("after the failure")),
        ]);
        let mut reader = FeedbackReader::start(feedback, MEMBER.to_owned());
        assert_eq!(reader.next().await, Some(json!("first")));
        assert_eq!(reader.next().await, Some(json!("second")));
        assert_eq!(reader.next().await, None);
    }

    /// The feedback that comes after the result of its goal goes to the sink
    /// up to the end of the stream: the last message of this stream comes
    /// only once the call waits for the rest.
    #[tokio::test]
    async fn the_rest_of_the_feedback_goes_to_the_sink_up_to_the_end_of_the_stream() {
        let (last, coming) = futures::channel::mpsc::unbounded();
        let feedback = futures::stream::iter([Ok(json!("before the result"))]).chain(coming);
        let reader = FeedbackReader::start(feedback, MEMBER.to_owned());
        let taken = Mutex::new(Vec::new());
        let sink = keeping(&taken);
        let cancel = CancellationToken::new();
        tokio::join!(reader.rest(&sink, &cancel, MEMBER), async {
            last.unbounded_send(Ok(json!("after the result")))
                .expect("the reader reads the stream");
            drop(last);
        });
        assert_eq!(
            *taken.lock().unwrap(),
            [json!("before the result"), json!("after the result")]
        );
    }

    /// A stream that does not end holds the end of its call no longer than
    /// `FEEDBACK_END`, on the paused clock of tokio.
    #[tokio::test(start_paused = true)]
    async fn the_wait_for_the_end_of_the_feedback_ends_at_its_bound() {
        let feedback = futures::stream::iter([Ok(json!("only"))]).chain(futures::stream::pending());
        let reader = FeedbackReader::start(feedback, MEMBER.to_owned());
        let taken = Mutex::new(Vec::new());
        let sink = keeping(&taken);
        let started = tokio::time::Instant::now();
        reader.rest(&sink, &CancellationToken::new(), MEMBER).await;
        assert_eq!(started.elapsed(), FEEDBACK_END);
        assert_eq!(*taken.lock().unwrap(), [json!("only")]);
    }

    /// A cancel of the step ends the wait for the end of the feedback at
    /// once, on the paused clock of tokio.
    #[tokio::test(start_paused = true)]
    async fn a_cancel_ends_the_wait_for_the_end_of_the_feedback() {
        let feedback = futures::stream::pending::<Result<Json, ConsumerError>>();
        let reader = FeedbackReader::start(feedback, MEMBER.to_owned());
        let cancel = CancellationToken::new();
        cancel.cancel();
        let started = tokio::time::Instant::now();
        reader.rest(&|_| {}, &cancel, MEMBER).await;
        assert_eq!(started.elapsed(), Duration::ZERO);
    }

    /// A cancelled goal settled when it ended cancelled, completed or failed;
    /// a goal that its producer dropped, whose result expired, or that gave no
    /// end within the cancel timeout leaves its robot in an unknown state.
    #[test]
    fn a_cancelled_goal_settles_only_on_an_end_of_its_producer() {
        let settled = CallEnd::Cancelled {
            asked: true,
            reason: "the director cancelled it".to_owned(),
        };
        let cancelled = GoalOutcome::Cancelled {
            reason: "cancelled".to_owned(),
            result: json!({}),
        };
        assert_eq!(end_after_cancel(Some(Ok(cancelled))), settled);
        assert_eq!(
            end_after_cancel(Some(Ok(GoalOutcome::Completed(json!({}))))),
            settled
        );
        for unknown in [
            Some(Ok(GoalOutcome::Abandoned)),
            Some(Ok(GoalOutcome::Expired)),
            None,
        ] {
            assert_eq!(end_after_cancel(unknown), CallEnd::Unknown);
        }
    }
}
