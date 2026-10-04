//! The loop that serves each service the backbone answers for the life of
//! the node, and the pause after a runtime error.

use std::fmt::Display;
use std::future::Future;
use std::time::Duration;

use peppylib::runtime::CancellationToken;
use tracing::error;

/// Pause after a service error before the loop listens again, so a broken
/// transport cannot hot-spin a loop or flood the log.
pub(crate) const RETRY_BACKOFF: Duration = Duration::from_secs(1);

/// Serves `service` one request at a time until `token` is cancelled:
/// `next_request` listens for the next request and answers it (a generated
/// `handle_next_request`). A cancel ends the loop at once, while it waits for
/// a request or waits out an error. An error is logged with the service's
/// name and waited out for [`RETRY_BACKOFF`] before the loop listens again.
pub(crate) async fn serve<F, E>(
    service: &str,
    token: &CancellationToken,
    mut next_request: impl FnMut() -> F,
) where
    F: Future<Output = Result<(), E>>,
    E: Display,
{
    loop {
        let served = tokio::select! {
            biased;
            () = token.cancelled() => return,
            served = next_request() => served,
        };
        match served {
            Ok(()) => continue,
            Err(e) => error!("{service}: {e}"),
        }
        tokio::select! {
            biased;
            () = token.cancelled() => return,
            () = tokio::time::sleep(RETRY_BACKOFF) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;

    use tokio::time::Instant;

    /// What a scripted service saw: when each request was listened for, on
    /// the test's clock. A listen past the most the test expects fails the
    /// test at once, so a loop that runs on past a cancel fails the test in
    /// place of spinning with no end.
    struct Listened {
        most: usize,
        at: RefCell<Vec<Instant>>,
    }

    impl Listened {
        fn at_most(most: usize) -> Self {
            Self {
                most,
                at: RefCell::default(),
            }
        }

        /// Notes one listen and gives how many there were.
        fn note(&self) -> usize {
            let mut listens = self.at.borrow_mut();
            listens.push(Instant::now());
            assert!(
                listens.len() <= self.most,
                "listen {} is past the {} the test expects",
                listens.len(),
                self.most
            );
            listens.len()
        }

        fn count(&self) -> usize {
            self.at.borrow().len()
        }
    }

    #[tokio::test]
    async fn a_cancelled_token_ends_the_loop_before_it_listens() {
        let token = CancellationToken::default();
        token.cancel();
        let listened = Listened::at_most(0);
        serve("cancelled", &token, || async {
            listened.note();
            Ok::<(), String>(())
        })
        .await;
        assert_eq!(listened.count(), 0);
    }

    /// On a paused clock that moves only when every task waits: the bound
    /// runs out only when the loop waits on after the cancel with nothing
    /// left to wake it.
    #[tokio::test(start_paused = true)]
    async fn a_cancel_ends_the_loop_while_it_waits_for_a_request() {
        let token = CancellationToken::default();
        let listened = Listened::at_most(1);
        let serving = serve("waiting", &token, || async {
            listened.note();
            std::future::pending::<Result<(), String>>().await
        });
        let cancel_once_listening = async {
            while listened.count() == 0 {
                tokio::task::yield_now().await;
            }
            token.cancel();
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(serving, cancel_once_listening)
        })
        .await
        .expect("the cancel ends the loop");
        assert_eq!(listened.count(), 1);
    }

    #[tokio::test]
    async fn requests_are_served_one_after_another_until_the_token_is_cancelled() {
        let token = CancellationToken::default();
        let listened = Listened::at_most(3);
        serve("served", &token, || async {
            if listened.note() == 3 {
                token.cancel();
            }
            Ok::<(), String>(())
        })
        .await;
        assert_eq!(listened.count(), 3);
    }

    /// On a paused clock that moves only when every task waits: the pause
    /// is the clock's, not the host's.
    #[tokio::test(start_paused = true)]
    async fn an_error_is_waited_out_before_the_loop_listens_again() {
        let token = CancellationToken::default();
        let listened = Listened::at_most(2);
        serve("failing", &token, || async {
            if listened.note() == 1 {
                return Err("the session is closed".to_owned());
            }
            token.cancel();
            Ok(())
        })
        .await;
        let listens = listened.at.borrow();
        assert_eq!(listens.len(), 2);
        assert!(listens[1] - listens[0] >= RETRY_BACKOFF);
    }

    #[tokio::test(start_paused = true)]
    async fn a_cancel_ends_the_loop_while_it_waits_out_an_error() {
        let token = CancellationToken::default();
        let listened = Listened::at_most(1);
        let start = Instant::now();
        serve("failing", &token, || async {
            listened.note();
            token.cancel();
            Err::<(), _>("the session is closed")
        })
        .await;
        assert_eq!(listened.count(), 1);
        assert!(Instant::now() - start < RETRY_BACKOFF);
    }
}
