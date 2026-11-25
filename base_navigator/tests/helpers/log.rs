//! The node's log as the tests read it. The node runs in the test's process
//! and logs through `tracing`, so one subscriber for the whole test binary,
//! installed by the first test that reads the log, keeps the message of every
//! event at info and above that any test's navigator logs from then on, the
//! events of every other crate left out. A test finds its own lines by a value
//! no other test's node logs.

use std::sync::{Mutex, OnceLock};

use control_core::sync::lock_ignoring_poison;
use tracing::Level;
use tracing::field::{Field, Visit};
use tracing_subscriber::filter::Targets;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

/// Every message logged since [`keep_the_log`] first ran.
fn kept() -> &'static Mutex<Vec<String>> {
    static KEPT: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    KEPT.get_or_init(|| Mutex::new(Vec::new()))
}

/// The layer that keeps each event's message.
struct Keeper;

impl<S: tracing::Subscriber> Layer<S> for Keeper {
    fn on_event(&self, event: &tracing::Event<'_>, _context: Context<'_, S>) {
        let mut message = MessageOf(String::new());
        event.record(&mut message);
        lock_ignoring_poison(kept()).push(message.0);
    }
}

/// The text of an event's `message` field.
struct MessageOf(String);

impl Visit for MessageOf {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
}

/// Keep every message the navigator logs at info and above in this test
/// binary from now on.
pub fn keep_the_log() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let navigator = Targets::new().with_target("base_navigator", Level::INFO);
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(Keeper.with_filter(navigator)),
        )
        .expect("the test binary installs one subscriber");
    });
}

/// The messages logged so far that hold every one of `parts`.
pub fn logged_with(parts: &[&str]) -> Vec<String> {
    lock_ignoring_poison(kept())
        .iter()
        .filter(|line| parts.iter().all(|part| line.contains(part)))
        .cloned()
        .collect()
}
