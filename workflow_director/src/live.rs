//! The host of a live run: calls through the clients of the codec API, the
//! move check before each call of a member with the mark moves_robot, the
//! latest message of each topic that a run observes, the open and close of
//! its leases, and its run clock.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use peppy_codec::consumer::{
    ActionClient, ConsumerError, ConsumerIdentity, GoalOutcome, MemberBinding, ServiceClient,
    TopicConsumer, contract_topic_codec,
};
use peppygen::NodeRunner;
use peppylib::messaging::ProducerRef;
use serde_json::Value as Json;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
use workflow_documents::{Name, RobotName};

use crate::contract::ContractShape;
use crate::engine::failure::{Failure, RESOURCE_UNAVAILABLE};
use crate::engine::host::{CallEnd, Event, Host, Join, Joining};
use crate::engine::plan::{MemberKind, ObservationRead, Plan, Target, TopicKey};
use crate::leases::Leases;

/// How long a goal has to be admitted.
const ADMISSION: Duration = Duration::from_secs(10);

/// How long the host waits for the result of a goal: past every bound a
/// workflow gives.
const RESULT_WAIT: Duration = Duration::from_secs(86_400 + 60);

/// The run clock of a run.
#[derive(Clone)]
pub enum RunClock {
    Wall,
    /// The simulation clock of the director: the time of its last tick.
    Simulation(watch::Receiver<u64>),
}

fn wall_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

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
    clock: RunClock,
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

impl LiveHost {
    /// The host of a run of `plan`: a client for each member it calls, laid
    /// out once, and a subscription for each topic it observes.
    pub async fn new(
        runner: Arc<NodeRunner>,
        plan: &Plan,
        contracts: &BTreeMap<String, Arc<ContractShape>>,
        leases: Arc<Leases>,
        clock: RunClock,
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
                        let at = now_of(&clock);
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

    async fn goal(
        &self,
        client: &ActionClient,
        binding: &MemberBinding,
        target: &Target,
        request: Json,
        cancel: CancellationToken,
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
        let outcome = tokio::select! {
            outcome = handle.result(messenger, RESULT_WAIT) => outcome,
            _ = cancel.cancelled() => {
                let settling = async {
                    let _ = handle.cancel(messenger, self.cancel_timeout).await;
                    handle.result(messenger, RESULT_WAIT).await
                };
                let settled = tokio::time::timeout(self.cancel_timeout, settling).await;
                return end_after_cancel(settled.ok());
            }
        };
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

fn now_of(clock: &RunClock) -> u64 {
    match clock {
        RunClock::Wall => wall_ns(),
        RunClock::Simulation(ticks) => *ticks.borrow(),
    }
}

/// Resolves at the first instant of `clock` at or after `at`; a simulation
/// clock that closed ends no wait past its last tick.
async fn until_on(clock: &RunClock, at: u64) {
    match clock {
        RunClock::Wall => {
            let now = wall_ns();
            if at > now {
                tokio::time::sleep(Duration::from_nanos(at - now)).await;
            }
        }
        RunClock::Simulation(ticks) => crate::clock::until_tick(ticks, at).await,
    }
}

impl Host for LiveHost {
    fn now(&self) -> u64 {
        now_of(&self.clock)
    }

    fn wall_now(&self) -> u64 {
        wall_ns()
    }

    async fn until(&self, at: u64) {
        until_on(&self.clock, at).await
    }

    async fn call(&self, target: &Target, request: Json, cancel: CancellationToken) -> CallEnd {
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
            Some(Client::Action(client, binding)) => {
                self.goal(client, binding, target, request, cancel).await
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
    use serde_json::json;

    use super::*;

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
