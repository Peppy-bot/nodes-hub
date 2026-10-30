//! The hold of one lane: how many of its branches run (the tree, and one
//! per child of each running parallel step), the incidents its branches met,
//! and its suspension. A suspension parks each branch at the step where it
//! stops; once no branch runs, the lane runs its safe state, records the
//! incidents and opens the lease of its robot, once for all its branches.

use tokio::sync::{mpsc, oneshot, watch};
use workflow_documents::Located;

use super::failure::Failure;
use super::lane::Stop;

/// What a lane tells a branch parked at a step.
pub(crate) enum BranchCommand {
    /// Check that the branch can go on: the invariant of each enclosing
    /// step, and whether the `done_when` of its call holds (`Ok(true)`).
    Check(oneshot::Sender<Result<bool, Located>>),
    Go,
    /// The run stopped.
    Stop,
}

/// An incident that a branch met at `step`, recorded once the lane settled.
pub(crate) struct Met {
    pub step: String,
    pub failure: Failure,
}

struct ParkedBranch {
    step: String,
    commands: mpsc::UnboundedSender<BranchCommand>,
}

#[derive(Default)]
struct Suspension {
    branches: Vec<ParkedBranch>,
    /// Whether the lane ran its safe state for the suspension.
    parked: bool,
}

struct State {
    running: usize,
    met: Vec<Met>,
    suspension: Option<Suspension>,
}

pub(crate) struct Hold {
    state: watch::Sender<State>,
}

impl Hold {
    /// The hold of a lane whose tree runs.
    pub fn new() -> Self {
        Self {
            state: watch::Sender::new(State {
                running: 1,
                met: Vec::new(),
                suspension: None,
            }),
        }
    }

    /// A parallel step splits its branch into one per child.
    pub fn split(&self, children: usize) {
        self.state
            .send_modify(|s| s.running = s.running + children - 1);
    }

    /// A child of a parallel step ended.
    pub fn branch_ends(&self) {
        self.state.send_modify(|s| s.running -= 1);
    }

    /// A parallel step whose children ended goes on as one branch.
    pub fn join(&self) {
        self.state.send_modify(|s| s.running += 1);
    }

    /// Asks the lane to suspend: each running branch parks at its next
    /// wait or call, or at once in one.
    pub fn request(&self) {
        self.state.send_if_modified(|s| match s.suspension {
            Some(_) => false,
            None => {
                s.suspension = Some(Suspension::default());
                true
            }
        });
    }

    pub fn is_suspending(&self) -> bool {
        self.state.borrow().suspension.is_some()
    }

    /// Resolves once the lane is asked to suspend.
    pub async fn suspending(&self) {
        let _ = self
            .state
            .subscribe()
            .wait_for(|s| s.suspension.is_some())
            .await;
    }

    /// Keeps an incident for the lane to record once it settled.
    pub fn meet(&self, met: Met) {
        self.state.send_if_modified(|s| {
            s.met.push(met);
            false
        });
    }

    /// The incidents met since the last call.
    pub fn take_met(&self) -> Vec<Met> {
        let mut met = Vec::new();
        self.state.send_if_modified(|s| {
            met = std::mem::take(&mut s.met);
            false
        });
        met
    }

    /// Parks a branch at `step` in the suspension of the lane, which it
    /// starts if none runs, and gives it the channel the lane tells it
    /// through; a run that stopped parks no branch, so this gives its stop.
    pub fn park(
        &self,
        step: String,
        stop: &watch::Sender<Option<Stop>>,
    ) -> Result<mpsc::UnboundedReceiver<BranchCommand>, Stop> {
        let (commands, receiver) = mpsc::unbounded_channel();
        let mut stopped = None;
        self.state.send_modify(|s| {
            stopped = *stop.borrow();
            if stopped.is_some() {
                return;
            }
            s.running -= 1;
            s.suspension
                .get_or_insert_with(Suspension::default)
                .branches
                .push(ParkedBranch { step, commands });
        });
        match stopped {
            Some(stop) => Err(stop),
            None => Ok(receiver),
        }
    }

    /// A branch leaves its park: at a resume, or at the stop of the run.
    pub fn unpark(&self) {
        self.state.send_modify(|s| s.running += 1);
    }

    /// Resolves once every branch of a suspension that has not ended is
    /// parked, and the lane has not run its safe state for it.
    pub async fn settled(&self) {
        let _ = self
            .state
            .subscribe()
            .wait_for(|s| {
                s.running == 0
                    && s.suspension
                        .as_ref()
                        .is_some_and(|p| !p.parked && !p.branches.is_empty())
            })
            .await;
    }

    /// Marks that the lane runs its safe state for its suspension, and
    /// gives the step the lane suspends at: that of its first incident, else
    /// that of its first parked branch.
    pub fn mark_parked(&self) -> String {
        let mut step = String::new();
        self.state.send_modify(|s| {
            let Some(suspension) = &mut s.suspension else {
                return;
            };
            suspension.parked = true;
            step = s
                .met
                .first()
                .map(|met| met.step.clone())
                .or_else(|| suspension.branches.first().map(|b| b.step.clone()))
                .unwrap_or_default();
        });
        step
    }

    /// Whether the lane ran its safe state for the suspension it is in.
    pub fn parked(&self) -> bool {
        self.state
            .borrow()
            .suspension
            .as_ref()
            .is_some_and(|p| p.parked)
    }

    /// The channels of the parked branches.
    pub fn branches(&self) -> Vec<mpsc::UnboundedSender<BranchCommand>> {
        self.state
            .borrow()
            .suspension
            .as_ref()
            .map(|p| p.branches.iter().map(|b| b.commands.clone()).collect())
            .unwrap_or_default()
    }

    /// Ends the suspension of the lane, and gives the step of its first
    /// parked branch and the channels of every parked branch.
    pub fn resume(&self) -> (String, Vec<mpsc::UnboundedSender<BranchCommand>>) {
        let mut ended = None;
        self.state.send_modify(|s| ended = s.suspension.take());
        let branches = ended.map(|p| p.branches).unwrap_or_default();
        let step = branches.first().map(|b| b.step.clone()).unwrap_or_default();
        (step, branches.into_iter().map(|b| b.commands).collect())
    }

    /// Tells each parked branch that the run stopped. The suspension stays,
    /// so a lane that ran its safe state for it runs none for the stop.
    pub fn stop_branches(&self) {
        let mut branches = Vec::new();
        self.state.send_modify(|s| {
            if let Some(suspension) = &mut s.suspension {
                branches = std::mem::take(&mut suspension.branches);
            }
        });
        for branch in branches {
            let _ = branch.commands.send(BranchCommand::Stop);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stop() -> watch::Sender<Option<Stop>> {
        watch::Sender::new(None)
    }

    #[test]
    fn a_lane_settles_once_every_running_branch_parked() {
        let hold = Hold::new();
        let stop = stop();
        hold.split(2);
        let _first = hold.park("arm.left".to_owned(), &stop).expect("parks");
        assert!(hold.is_suspending());
        assert_eq!(hold.state.borrow().running, 1);
        let _second = hold.park("arm.right".to_owned(), &stop).expect("parks");
        assert_eq!(hold.state.borrow().running, 0);
        assert_eq!(hold.mark_parked(), "arm.left");
        assert!(hold.parked());
        assert_eq!(hold.branches().len(), 2);
    }

    #[test]
    fn the_step_of_a_suspension_is_that_of_its_first_incident() {
        let hold = Hold::new();
        let stop = stop();
        hold.split(2);
        let _first = hold.park("arm.left".to_owned(), &stop).expect("parks");
        hold.meet(Met {
            step: "arm.right".to_owned(),
            failure: Failure::new("timeout", "the bound passed"),
        });
        let _second = hold.park("arm.right".to_owned(), &stop).expect("parks");
        assert_eq!(hold.mark_parked(), "arm.right");
        assert_eq!(hold.take_met().len(), 1);
        assert!(hold.take_met().is_empty());
    }

    #[test]
    fn a_stopped_run_parks_no_branch() {
        let hold = Hold::new();
        let stop = stop();
        stop.send_replace(Some(Stop::Halted));
        assert!(matches!(
            hold.park("arm.move".to_owned(), &stop),
            Err(Stop::Halted)
        ));
        assert_eq!(hold.state.borrow().running, 1);
        assert!(!hold.is_suspending());
    }

    #[test]
    fn a_resume_ends_the_suspension_and_the_branches_run_again() {
        let hold = Hold::new();
        let stop = stop();
        let _branch = hold.park("arm.move".to_owned(), &stop).expect("parks");
        hold.mark_parked();
        let (step, branches) = hold.resume();
        assert_eq!(step, "arm.move");
        assert_eq!(branches.len(), 1);
        assert!(!hold.is_suspending());
        assert!(!hold.parked());
        hold.unpark();
        assert_eq!(hold.state.borrow().running, 1);
    }

    #[test]
    fn a_stop_keeps_the_parked_suspension() {
        let hold = Hold::new();
        let stop = stop();
        let mut branch = hold.park("arm.move".to_owned(), &stop).expect("parks");
        hold.mark_parked();
        hold.stop_branches();
        assert!(matches!(branch.try_recv(), Ok(BranchCommand::Stop)));
        assert!(hold.parked());
        assert!(hold.branches().is_empty());
    }

    #[test]
    fn a_second_request_keeps_the_suspension() {
        let hold = Hold::new();
        let stop = stop();
        let _branch = hold.park("arm.move".to_owned(), &stop).expect("parks");
        hold.request();
        assert_eq!(hold.branches().len(), 1);
    }
}
