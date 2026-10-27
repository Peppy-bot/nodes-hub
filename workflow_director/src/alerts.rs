//! The alerts of the physical robots of a run, from the `alert:v1` instances
//! of each robot in the fleet listing. An alert of severity 2 or 3 is an
//! incident of the lane of its robot, once while its `(source, kind)` stands;
//! an alert of severity 1 goes to the journal. A set that is not refreshed for
//! three cadence periods leaves the conditions of its robot unknown, which
//! is an incident too.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use peppy_codec::consumer::{ConsumerIdentity, MemberBinding, TopicConsumer, contract_topic_codec};
use peppygen::NodeRunner;
use peppylib::messaging::ProducerRef;
use serde_json::{Value as Json, json};
use tokio::sync::mpsc;
use workflow_documents::Name;
use workflow_documents::result::AlertCause;

use crate::engine::RunCommand;
use crate::engine::failure::{Cause, DirectorKind, Failure};
use crate::engine::plan::Plan;
use crate::fleet::Snapshot;

pub const ALERT: (&str, &str) = ("alert", "v1");
const ALERTS_TOPIC: &str = "alerts";

/// Three cadence periods of `alert:v1`.
const ALERTS_STALE: Duration = Duration::from_secs(6);

/// The lowest severity that is an incident.
const INCIDENT_SEVERITY: u64 = 2;

/// An alert of a producer by its `(source, kind)`, which `alert:v1` keeps for the life of the alert.
type AlertPair = (String, String);

/// The alerts of one producer that stand: those raised as incidents of the lane, at severity 2 or 3, and those noted
/// in the journal, at severity 1. A pair that rises from one to the other is a new alert of that severity.
#[derive(Default)]
struct Standing {
    raised: BTreeSet<AlertPair>,
    noted: BTreeSet<AlertPair>,
}

/// The watch of the alerts of a run; dropping it ends the watch.
pub struct Alerts {
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl Drop for Alerts {
    fn drop(&mut self) {
        self.tasks.iter().for_each(tokio::task::JoinHandle::abort);
    }
}

/// A journal entry the watch writes: an alert of severity 1.
pub type Note = Box<dyn Fn(Json) + Send + Sync>;

/// Subscribes to the alerts of each physical robot of `plan`, waits up to
/// three cadence periods for the first set of each producer, and watches
/// them for the run.
pub async fn watch(
    runner: &NodeRunner,
    snapshot: &Snapshot,
    plan: &Plan,
    commands: mpsc::UnboundedSender<RunCommand>,
    note: Note,
) -> Result<Alerts, String> {
    let note: Arc<Note> = Arc::new(note);
    let processor = runner.processor();
    let identity = ConsumerIdentity {
        core_node: processor.bound_core_node().to_owned(),
        instance_id: processor.bound_instance_id().to_owned(),
    };
    let mut tasks = Vec::new();
    for (role, bound) in plan.roles.iter().filter(|(_, bound)| bound.physical) {
        let Some(robot) = snapshot.fleet.robots.get(bound.robot.as_str()) else {
            continue;
        };
        let (name, tag) = ALERT;
        // Each producer reads with the pin it serves.
        for instance in robot.instances_of(name, tag) {
            let Some(pin) = robot.pin_of(name, tag, &instance.name) else {
                continue;
            };
            let shape = snapshot.contracts.get(&pin.sha256).ok_or_else(|| {
                format!("the document of alert:v1 of `{}` did not read", robot.name)
            })?;
            let topic = shape
                .parsed
                .topic(ALERTS_TOPIC)
                .ok_or("alert:v1 has no topic `alerts`")?
                .clone();
            let codec =
                contract_topic_codec(&shape.parsed, ALERTS_TOPIC).map_err(|e| e.to_string())?;
            let binding = MemberBinding::of_contract(&shape.parsed, ALERTS_TOPIC)
                .map_err(|e| e.to_string())?;
            let producer = ProducerRef::new(instance.core_node.clone(), instance.instance.clone());
            let mut consumer = TopicConsumer::of_producer(
                runner.messenger(),
                &identity,
                &binding,
                &producer,
                topic.qos_profile.clone(),
                topic.retention,
                codec,
            )
            .await
            .map_err(|e| e.to_string())?;
            let first = tokio::time::timeout(ALERTS_STALE, consumer.next_message())
                .await
                .ok()
                .flatten()
                .and_then(|(_, message)| consumer.decode(&message).ok());
            let Some(first) = first else {
                return Err(format!(
                    "the alerts of `{}` from `{}` did not come within {} s: its conditions are unknown",
                    robot.name,
                    instance.instance,
                    ALERTS_STALE.as_secs()
                ));
            };
            let role = role.clone();
            let commands = commands.clone();
            let note = Arc::clone(&note);
            let source = instance.instance.clone();
            tasks.push(tokio::spawn(async move {
                let mut standing = Standing::default();
                let mut set = first;
                let mut stale = false;
                loop {
                    standing = raise(&role, &source, &set, standing, &commands, &note);
                    match tokio::time::timeout(ALERTS_STALE, consumer.next_message()).await {
                        Ok(Some((_, message))) => {
                            stale = false;
                            if let Ok(next) = consumer.decode(&message) {
                                set = next;
                            }
                        }
                        Ok(None) => return,
                        Err(_) if stale => {}
                        Err(_) => {
                            stale = true;
                            let _ = commands.send(RunCommand::Fault {
                                role: role.clone(),
                                failure: stale_alerts(&source),
                            });
                        }
                    }
                }
            }));
        }
    }
    Ok(Alerts { tasks })
}

/// The failure of a producer whose alert set was not refreshed for
/// `ALERTS_STALE`.
fn stale_alerts(source: &str) -> Failure {
    let message = format!(
        "the alerts from `{source}` were not refreshed for {} s: its conditions are unknown",
        ALERTS_STALE.as_secs()
    );
    Failure::of(DirectorKind::Alert, message).because(Cause::Alert(AlertCause {
        source: source.to_owned(),
        kind: String::new(),
        severity: None,
        stale: true,
    }))
}

/// Raises each alert of `set` of severity 2 or 3 that is not raised yet,
/// notes each of severity 1 that is not noted yet, and gives the alerts that
/// stand after it.
fn raise(
    role: &Name,
    source: &str,
    set: &Json,
    standing: Standing,
    commands: &mpsc::UnboundedSender<RunCommand>,
    note: &Note,
) -> Standing {
    let active = set["active"].as_array().cloned().unwrap_or_default();
    let mut now = Standing::default();
    for alert in &active {
        let pair = (
            alert["source"].as_str().unwrap_or_default().to_owned(),
            alert["kind"].as_str().unwrap_or_default().to_owned(),
        );
        let severity = alert["severity"]
            .as_u64()
            .or_else(|| alert["severity"].as_str().and_then(|s| s.parse().ok()))
            .unwrap_or(0);
        if severity < INCIDENT_SEVERITY {
            if !standing.noted.contains(&pair) {
                note(json!({ "role": role, "producer": source, "alert": alert }));
            }
            now.noted.insert(pair);
            continue;
        }
        if !standing.raised.contains(&pair) {
            let message = format!(
                "{} ({}): {}",
                pair.0,
                pair.1,
                alert["message"].as_str().unwrap_or_default()
            );
            let cause = Cause::Alert(AlertCause {
                source: pair.0.clone(),
                kind: pair.1.clone(),
                severity: u8::try_from(severity).ok(),
                stale: false,
            });
            let _ = commands.send(RunCommand::Fault {
                role: role.clone(),
                failure: Failure::of(DirectorKind::Alert, message).because(cause),
            });
        }
        now.raised.insert(pair);
    }
    now
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::testing::name;

    /// The `(role, kind, message)` of each fault that the watch sent.
    fn faults(commands: &mut mpsc::UnboundedReceiver<RunCommand>) -> Vec<(String, String, String)> {
        std::iter::from_fn(|| commands.try_recv().ok())
            .map(|command| match command {
                RunCommand::Fault { role, failure } => {
                    (role.to_string(), failure.kind, failure.message)
                }
                _ => panic!("the watch sends faults alone"),
            })
            .collect()
    }

    fn set(active: Json) -> Json {
        json!({ "active": active })
    }

    fn fault(message: &str) -> (String, String, String) {
        ("arm".to_owned(), "alert".to_owned(), message.to_owned())
    }

    /// An alert of severity 2 or 3, its severity a number or its text, raises
    /// one fault of its lane while its `(source, kind)` stands, and one more
    /// once it went and came back.
    #[test]
    fn an_incident_alert_raises_one_fault_while_it_stands() {
        let (commands, mut sent) = mpsc::unbounded_channel();
        let note: Note = Box::new(|entry| panic!("no alert of severity 1: {entry}"));
        let raised = |set: Json, standing| {
            raise(
                &name("arm"),
                "health_inst",
                &set,
                standing,
                &commands,
                &note,
            )
        };
        let hot =
            json!({ "source": "motor_3", "kind": "overheat", "severity": 2, "message": "82 C" });

        let standing = raised(set(json!([hot])), Standing::default());
        assert_eq!(faults(&mut sent), [fault("motor_3 (overheat): 82 C")]);
        let standing = raised(set(json!([hot])), standing);
        assert!(
            faults(&mut sent).is_empty(),
            "a standing alert raises nothing"
        );
        let standing = raised(set(json!([])), standing);
        assert!(standing.raised.is_empty() && standing.noted.is_empty());
        let hotter =
            json!({ "source": "motor_3", "kind": "overheat", "severity": "3", "message": "90 C" });
        raised(set(json!([hotter])), standing);
        assert_eq!(faults(&mut sent), [fault("motor_3 (overheat): 90 C")]);
    }

    /// An alert of severity 1 goes to the journal once while it stands, and
    /// raises no fault.
    #[test]
    fn a_low_alert_is_noted_once_while_it_stands_and_raises_no_fault() {
        let (commands, mut sent) = mpsc::unbounded_channel();
        let notes = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&notes);
        let note: Note = Box::new(move |entry| kept.lock().unwrap().push(entry));
        let warm = json!({ "source": "motor_3", "kind": "warm", "severity": 1, "message": "60 C" });

        let standing = raise(
            &name("arm"),
            "health_inst",
            &set(json!([warm])),
            Standing::default(),
            &commands,
            &note,
        );
        raise(
            &name("arm"),
            "health_inst",
            &set(json!([warm])),
            standing,
            &commands,
            &note,
        );
        assert_eq!(
            *notes.lock().unwrap(),
            [json!({ "role": "arm", "producer": "health_inst", "alert": warm })]
        );
        assert!(faults(&mut sent).is_empty());
    }

    /// A raised alert names its source, its kind and its severity as the
    /// cause of its fault, and a stale alert set names its producer.
    #[test]
    fn the_fault_of_an_alert_names_its_cause() {
        let (commands, mut sent) = mpsc::unbounded_channel();
        let note: Note = Box::new(|entry| panic!("no alert of severity 1: {entry}"));
        let hot =
            json!({ "source": "motor_3", "kind": "overheat", "severity": 2, "message": "82 C" });
        raise(
            &name("arm"),
            "health_inst",
            &set(json!([hot])),
            Standing::default(),
            &commands,
            &note,
        );
        let Ok(RunCommand::Fault { failure, .. }) = sent.try_recv() else {
            panic!("the alert raises a fault")
        };
        assert_eq!(
            failure.cause.as_deref(),
            Some(&Cause::Alert(AlertCause {
                source: "motor_3".to_owned(),
                kind: "overheat".to_owned(),
                severity: Some(2),
                stale: false,
            }))
        );
        assert_eq!(
            stale_alerts("health_inst").cause.as_deref(),
            Some(&Cause::Alert(AlertCause {
                source: "health_inst".to_owned(),
                kind: String::new(),
                severity: None,
                stale: true,
            }))
        );
    }

    /// An alert that rises from severity 1 to 3 under its `(source, kind)`, as a motor that warns and then faults, is
    /// noted once and then raised once.
    #[test]
    fn an_alert_that_rises_to_an_incident_raises_its_fault() {
        let (commands, mut sent) = mpsc::unbounded_channel();
        let notes = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&notes);
        let note: Note = Box::new(move |entry| kept.lock().unwrap().push(entry));
        let raised = |set: Json, standing| {
            raise(
                &name("arm"),
                "health_inst",
                &set,
                standing,
                &commands,
                &note,
            )
        };
        let warns = json!({ "source": "motor_3", "kind": "motor_condition", "severity": 1, "message": "warm" });
        let faults_now = json!({ "source": "motor_3", "kind": "motor_condition", "severity": 3, "message": "fault" });

        let standing = raised(set(json!([warns])), Standing::default());
        let standing = raised(set(json!([faults_now])), standing);
        raised(set(json!([faults_now])), standing);
        assert_eq!(notes.lock().unwrap().len(), 1);
        assert_eq!(
            faults(&mut sent),
            [fault("motor_3 (motor_condition): fault")]
        );
    }
}
