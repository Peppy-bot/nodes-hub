//! The leases of the director: one `fleet.lease` goal for the life of the
//! node, the services of a lease holder, the move check before each call of
//! a member with the mark moves_robot, and the events of the lease goal for
//! the runs that hold leases.

use std::sync::Arc;

use peppygen::NodeRunner;
use peppygen::consumed_actions::fleet::lease;
use peppygen::consumed_services::fleet::{check_move, close, open, release, take};
use tokio::sync::broadcast;

use crate::Fault;
use crate::engine::failure::RESOURCE_UNAVAILABLE;
use crate::fleet::DAEMON_TIMEOUT;

/// What an event of the lease goal tells: an offer to the director and its
/// end, and the end or the give-back of a lease of the director.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseEventKind {
    Offer,
    OfferEnd,
    LeaseEnd,
    LeaseBack,
}

impl LeaseEventKind {
    fn parse(event: &str) -> Option<LeaseEventKind> {
        match event {
            "offer" => Some(LeaseEventKind::Offer),
            "offer_end" => Some(LeaseEventKind::OfferEnd),
            "lease_end" => Some(LeaseEventKind::LeaseEnd),
            "lease_back" => Some(LeaseEventKind::LeaseBack),
            _ => None,
        }
    }
}

/// An event of the lease goal, of `robot`, with its cause.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeaseEvent {
    pub kind: LeaseEventKind,
    pub robot: String,
    pub cause: String,
}

/// Why the daemon refused a take or a move: its kind, whether that kind is
/// the halt of the stack, and its message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    pub kind: String,
    pub halted: bool,
    pub message: String,
}

/// The kind of failure the daemon gives while the stack is halted.
const HALTED: &str = "halted";

/// The lease holder of the director.
pub struct Leases {
    runner: Arc<NodeRunner>,
    events: broadcast::Sender<LeaseEvent>,
}

impl Leases {
    /// Starts the lease goal and follows its feedback for the life of the
    /// node. The end of the goal raises `fault`.
    pub async fn start(runner: Arc<NodeRunner>, fault: Fault) -> peppygen::Result<Arc<Leases>> {
        let daemon = lease::bound_producer(&runner);
        let mut goal = lease::ActionHandle::fire_goal(
            &runner,
            daemon,
            DAEMON_TIMEOUT,
            peppygen::DEFAULT_FEEDBACK_BOUND,
        )
        .await?;
        if !goal.accepted {
            return Err(crate::refused(format!(
                "the daemon refused the lease goal of the director: {:?}",
                goal.refusal
            )));
        }
        let (events, _) = broadcast::channel(1024);
        let leases = Arc::new(Leases {
            runner: Arc::clone(&runner),
            events: events.clone(),
        });
        tokio::spawn(async move {
            while let Ok(feedback) = goal.on_next_feedback_message().await {
                let Some(kind) = LeaseEventKind::parse(&feedback.event) else {
                    tracing::warn!(
                        "the lease goal gave the event `{}`, which the director does not know",
                        feedback.event
                    );
                    continue;
                };
                let _ = events.send(LeaseEvent {
                    kind,
                    robot: feedback.robot,
                    cause: feedback.cause,
                });
            }
            fault.raise(
                &runner,
                "the lease goal of the director ended, and its leases with it; the node exits for its restart policy to start it with a new lease goal",
            );
        });
        Ok(leases)
    }

    pub fn events(&self) -> broadcast::Receiver<LeaseEvent> {
        self.events.subscribe()
    }

    /// Takes the lease of `robot` under `name`.
    pub async fn take(&self, robot: &str, name: &str) -> Result<(), Refused> {
        let daemon = take::bound_producer(&self.runner);
        let request = take::Request::new(robot.to_owned(), name.to_owned());
        take::poll(&self.runner, daemon, DAEMON_TIMEOUT, request)
            .await
            .map_err(|error| match take::Failure::of(&error) {
                Some(failure) => Refused {
                    kind: failure.kind.to_string(),
                    halted: failure.kind == take::FailureKind::Halted,
                    message: failure.message,
                },
                None => Refused {
                    kind: RESOURCE_UNAVAILABLE.to_owned(),
                    halted: false,
                    message: error.to_string(),
                },
            })
    }

    /// Releases the lease of `robot`; a lease that ended already is no error.
    pub async fn release(&self, robot: &str) {
        let daemon = release::bound_producer(&self.runner);
        let request = release::Request::new(robot.to_owned());
        let Err(error) = release::poll(&self.runner, daemon, DAEMON_TIMEOUT, request).await else {
            return;
        };
        match release::Failure::of(&error) {
            Some(failure) if failure.kind == release::FailureKind::NotLeased => {}
            _ => tracing::warn!("the lease of {robot} was not released: {error}"),
        }
    }

    /// Opens or closes the lease of `robot` for a take-over, or gives why
    /// the daemon did not.
    pub async fn set_open(&self, robot: &str, is_open: bool) -> Result<(), String> {
        let result = match is_open {
            true => {
                let daemon = open::bound_producer(&self.runner);
                open::poll(
                    &self.runner,
                    daemon,
                    DAEMON_TIMEOUT,
                    open::Request::new(robot.to_owned()),
                )
                .await
            }
            false => {
                let daemon = close::bound_producer(&self.runner);
                close::poll(
                    &self.runner,
                    daemon,
                    DAEMON_TIMEOUT,
                    close::Request::new(robot.to_owned()),
                )
                .await
            }
        };
        result.map(drop).map_err(|error| error.to_string())
    }

    /// Asks the daemon whether the director may call a member with the mark
    /// moves_robot on `instance`: `Ok(None)` when it may, else the kind and
    /// the message of the refusal.
    pub async fn check_move(&self, instance: &str) -> Result<Option<Refused>, String> {
        let daemon = check_move::bound_producer(&self.runner);
        let request = check_move::Request::new(Some(instance.to_owned()), None);
        let response = check_move::poll(&self.runner, daemon, DAEMON_TIMEOUT, request)
            .await
            .map_err(|error| error.to_string())?;
        Ok(response.data.refusal.map(|refusal| Refused {
            halted: refusal.failure_kind == HALTED,
            kind: refusal.failure_kind,
            message: refusal.message,
        }))
    }
}
