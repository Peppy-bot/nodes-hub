//! The judgments of joint paths, run beside the coordination loop. The shares
//! of an execute_joint_path goal hold their arms' commands while the path is
//! judged ([`crate::planner::start_path`]), and the loop hands the path to
//! [`PathChecks::judge`], which judges it on a blocking thread against the
//! guard as the governor held it then ([`judge_path`]). Each tick the loop
//! resumes the shares of every path whose judgment came and whose arms are
//! measured ([`PathChecks::resume_judged`]): they follow the judged path from
//! that tick on, each named arm's command moved onto the path's start, or the
//! path is judged again from where an arm that moved stands, or they end with
//! the failure, running none of the path. The loop
//! ticks on while a path is judged, which takes a long path many control
//! ticks, and stops the judgment of a path whose goal ended, which frees the
//! checker for the next path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use crate::robot::ARMS;
use robot_description::PerArm;
use tokio::sync::oneshot;
use tokio::sync::oneshot::error::TryRecvError;

use control_core::path_timing::RoundedPath;

use crate::failure::{Failure, FailureKind};
use crate::governor::{GuardAt, PathArms, PathChecker};
use crate::planner::{
    Measurement, PathGuard, PathJudgment, Planner, Resumed, judge_path, resume_judged_path,
};

impl PathGuard for PathChecker {
    fn check(
        &mut self,
        arms: PathArms,
        guard: &GuardAt,
        path: &RoundedPath,
        stop: &AtomicBool,
    ) -> Result<(), String> {
        self.check_path(arms, guard, path, stop)
            .map_err(|blocked| blocked.to_string())
    }
}

/// The words of the failure of a path whose checker failed.
const CHECKER_FAILED: &str = "the check of the path against the robot's body failed";

/// A path that waits for its judgment, the judgment's stop, and how far its
/// judgment has come.
struct Judging {
    judgment: PathJudgment,
    stop: Arc<AtomicBool>,
    state: JudgmentState,
}

/// How far the judgment of a path has come.
enum JudgmentState {
    /// The verdict has yet to come, on this receiver.
    Pending(oneshot::Receiver<Verdict>),
    /// The verdict came. A path judged to run holds it here while an arm of
    /// it has yet to be measured.
    Judged(Verdict),
}

impl JudgmentState {
    /// This state with the verdict taken in, once it has come: the failure
    /// of the checker when its thread dropped the verdict.
    fn received(self) -> Self {
        let Self::Pending(mut verdict) = self else {
            return self;
        };
        match verdict.try_recv() {
            Ok(verdict) => Self::Judged(verdict),
            Err(TryRecvError::Empty) => Self::Pending(verdict),
            Err(TryRecvError::Closed) => Self::Judged(checker_failed()),
        }
    }

    /// Wait for the verdict and take it in, as [`JudgmentState::received`]
    /// does: the verdict.
    #[cfg(test)]
    async fn come(&mut self) -> &Verdict {
        if let Self::Pending(verdict) = self {
            let came = verdict.await.unwrap_or_else(|_| checker_failed());
            *self = Self::Judged(came);
        }
        match self {
            Self::Judged(verdict) => verdict,
            Self::Pending(_) => unreachable!("the verdict came"),
        }
    }
}

impl Judging {
    /// The judgment of `judgment` against `guard` by `checker` on a blocking
    /// thread, which a raised `stop` ends at its next measured probe: it
    /// comes to a later tick.
    fn spawned(
        checker: &Arc<Mutex<dyn PathGuard>>,
        judgment: PathJudgment,
        guard: GuardAt,
        stop: Arc<AtomicBool>,
    ) -> Self {
        let (answer, verdict) = oneshot::channel();
        let checker = checker.clone();
        let (plan, arms, stopped) = (judgment.plan.clone(), judgment.arms, stop.clone());
        tokio::task::spawn_blocking(move || {
            let mut checker = checker.lock().unwrap_or_else(PoisonError::into_inner);
            let _ = answer.send(judge_path(&plan, arms, &guard, &mut *checker, &stopped));
        });
        Self {
            judgment,
            stop,
            state: JudgmentState::Pending(verdict),
        }
    }

    /// Resume the shares that wait for this judgment at `now`, the arms
    /// measuring `measured`: this judging back, or the next judgment of the
    /// path, while the path waits on.
    /// - The judgment stops once no share waits for it.
    /// - A failed path fails with its verdict.
    /// - A path judged to run starts once its arms are measured, or is
    ///   judged again by `checker` against `guard` from where they stand
    ///   ([`resume_judged_path`]).
    fn resume(
        self,
        planners: &mut PerArm<Planner, ARMS>,
        measured: &PerArm<Measurement, ARMS>,
        checker: &Arc<Mutex<dyn PathGuard>>,
        guard: GuardAt,
        now: Instant,
    ) -> Option<Self> {
        let Self {
            judgment,
            stop,
            state,
        } = self;
        if !judgment.is_awaited(planners) {
            stop.store(true, Ordering::Release);
            return None;
        }
        let state = state.received();
        let JudgmentState::Judged(verdict) = &state else {
            return Some(Self {
                judgment,
                stop,
                state,
            });
        };
        match resume_judged_path(planners, &judgment, verdict, measured, now) {
            Resumed::Done => None,
            Resumed::AwaitsMeasurement => Some(Self {
                judgment,
                stop,
                state,
            }),
            Resumed::JudgeAgain(again) => Some(Self::spawned(checker, *again, guard, stop)),
        }
    }
}

/// The verdict on a path: whether the judgment lets the path run, or the
/// failure of the path.
type Verdict = Result<(), Failure>;

/// The paths being judged beside the loop, and the checker that judges
/// them, one path at a time.
pub struct PathChecks {
    checker: Arc<Mutex<dyn PathGuard>>,
    judging: Vec<Judging>,
}

impl PathChecks {
    pub fn new(checker: Arc<Mutex<dyn PathGuard>>) -> Self {
        Self {
            checker,
            judging: Vec::new(),
        }
    }

    /// Judge the path of `judgment` against `guard` on a blocking thread,
    /// and return at once: the judgment comes to a later tick.
    pub fn judge(&mut self, judgment: PathJudgment, guard: GuardAt) {
        let stop = Arc::new(AtomicBool::new(false));
        self.judging
            .push(Judging::spawned(&self.checker, judgment, guard, stop));
    }

    /// Resume the shares of each path being judged at `now`, the arms
    /// measuring `measured`, as [`Judging::resume`] says: a path whose
    /// checker failed fails with `attempt_failed`, and a path judged again
    /// is judged against `guard`, the guard as the governor holds it this
    /// tick. A path still being judged, or judged to run while an arm of it
    /// has yet to be measured, waits on.
    pub fn resume_judged(
        &mut self,
        planners: &mut PerArm<Planner, ARMS>,
        measured: &PerArm<Measurement, ARMS>,
        guard: GuardAt,
        now: Instant,
    ) {
        let checker = self.checker.clone();
        self.judging = std::mem::take(&mut self.judging)
            .into_iter()
            .filter_map(|judging| judging.resume(planners, measured, &checker, guard, now))
            .collect();
    }

    /// Whether a path waits for its judgment.
    #[cfg(test)]
    pub fn is_judging(&self) -> bool {
        !self.judging.is_empty()
    }

    /// The verdict on the first path being judged, once it comes.
    #[cfg(test)]
    pub async fn first_verdict(&mut self) -> Verdict {
        let judging = self.judging.first_mut().expect("a path being judged");
        judging.state.come().await.clone()
    }

    /// Wait for the verdict on every path being judged, and resume its
    /// shares at `now`, each arm measured where it is held, as
    /// [`PathChecks::resume_judged`] does.
    #[cfg(test)]
    pub async fn resume_once_judged(&mut self, planners: &mut PerArm<Planner, ARMS>, now: Instant) {
        for judging in &mut self.judging {
            judging.state.come().await;
        }
        let held = crate::test_support::measured_at_setpoints(planners);
        self.resume_judged_by_test_guard(planners, &held, now);
    }

    /// Resume the shares of each path being judged at `now`, the arms
    /// measuring `measured`, as [`PathChecks::resume_judged`] does, a path
    /// judged again against the disabled guard of
    /// [`crate::test_support::test_guard`].
    #[cfg(test)]
    pub fn resume_judged_by_test_guard(
        &mut self,
        planners: &mut PerArm<Planner, ARMS>,
        measured: &PerArm<Measurement, ARMS>,
        now: Instant,
    ) {
        let guard = crate::test_support::test_guard(planners);
        self.resume_judged(planners, measured, guard, now);
    }
}

impl Drop for PathChecks {
    /// Stop every judgment still running or waiting for the checker, so
    /// each ends by its next measured probe.
    fn drop(&mut self) {
        for judging in &self.judging {
            judging.stop.store(true, Ordering::Release);
        }
    }
}

/// The verdict on a path whose judgment did not come: the checker failed.
fn checker_failed() -> Verdict {
    Err(Failure::new(FailureKind::AttemptFailed, CHECKER_FAILED))
}
