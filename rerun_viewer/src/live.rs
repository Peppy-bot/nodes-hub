//! The canvas that the node draws on: it keeps only the newest draw of each entity until
//! the canvas under it, the Rerun viewer, takes it. A thread of its own hands the draws
//! on in rounds: every draw that waits, then a flush, which waits until Rerun's data
//! server took them. So when the server waits on a page that reads slowly, only that
//! thread waits: the followers go on, each entity keeps its newest draw, and what reaches
//! the page is the stack as it is now, not a backlog of the past inside Rerun.
//!
//! The sender hands on only what changes the view: a draw that is the same as the last
//! one it handed on for its part of its entity (`Drawn::same_as`) is left out, until
//! the entity's data is `REFRESH_AFTER` newer than that one. So a stack at rest sends
//! little, and a page that opens late still gets each entity within `REFRESH_AFTER`.
//!
//! Each draw takes the time that `set_time` put on the thread that drew it, as Rerun
//! does: a drawer sets the time and draws in one synchronous call.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::mem::{Discriminant, discriminant};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, SystemTime};

use control_core::latched_log::LatchedLog;
use tracing::warn;

use crate::canvas::Canvas;
use crate::drawn::{Drawn, DrawnCanvas};

/// The most log lines that wait; past it, the oldest line that waits is dropped.
pub const MAX_WAITING_LINES: usize = 256;
/// How long a close waits for the canvas under it to take the draws that wait.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(2);
/// How much newer than the last draw handed on for its part a draw that is the same is
/// handed on again, by the time of its data: a page that opens late has each entity
/// within this, and the data server's history (`memory_limit`) holds it.
pub const REFRESH_AFTER: Duration = Duration::from_secs(2);
/// How many rounds in a row with no draw replaced or dropped count as caught up again:
/// about 5 s at the 10 snapshots of the arms a second that the node draws by default. A
/// data server that takes the draws in fits, as for a page that reads slowly, then
/// warns once, not once a fit.
const ROUNDS_TO_CATCH_UP: u32 = 50;

thread_local! {
    /// The time that `set_time` put on this thread, which each draw of the thread takes.
    static THREAD_TIME: Cell<Option<SystemTime>> = const { Cell::new(None) };
}

/// A draw that waits: the call, and the time of the thread that made it.
struct Waiting {
    at: Option<SystemTime>,
    call: Drawn,
}

/// What a draw sets: an entity, and the kind of call. Two draws of one kind at one
/// entity replace each other; draws of two kinds at one entity stand side by side, as
/// the values of a plot and the names of its lines do.
type Part = (String, Discriminant<Drawn>);

fn part_of(path: &str, call: &Drawn) -> Part {
    (path.to_string(), discriminant(call))
}

/// Whether the entity `path` is `root` or under it.
fn is_under(path: &str, root: &str) -> bool {
    path.strip_prefix(root)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// The draws that wait, in the order they came in.
#[derive(Default)]
struct Slots {
    /// Each draw that waits, by its number of arrival.
    by_arrival: BTreeMap<u64, Waiting>,
    /// The number of arrival of the draw that waits for each part: a newer draw of the
    /// part replaces it.
    by_part: HashMap<Part, u64>,
    /// The numbers of arrival of the log lines that wait, oldest first. Log lines are
    /// events: none replaces another.
    lines: VecDeque<u64>,
    next_arrival: u64,
    /// Whether a draw was replaced or dropped since the sender took the last round.
    dropped: bool,
    closing: bool,
}

impl Slots {
    fn arrival(&mut self) -> u64 {
        let arrival = self.next_arrival;
        self.next_arrival += 1;
        arrival
    }

    /// Puts `waiting` in the slot of its part of the entity `path`, after every draw that
    /// waits. Gives true when it replaced a draw that waited there.
    fn replace(&mut self, path: &str, waiting: Waiting) -> bool {
        let arrival = self.arrival();
        let replaced = self
            .by_part
            .insert(part_of(path, &waiting.call), arrival)
            .is_some_and(|older| self.by_arrival.remove(&older).is_some());
        self.by_arrival.insert(arrival, waiting);
        replaced
    }

    /// Drops the draws that wait at the entity `path` and under it, log lines aside.
    /// Gives true when it dropped any.
    fn drop_under(&mut self, path: &str) -> bool {
        let gone: Vec<Part> = self
            .by_part
            .keys()
            .filter(|(waiting, _)| is_under(waiting, path))
            .cloned()
            .collect();
        let mut dropped = false;
        for gone in gone {
            if let Some(arrival) = self.by_part.remove(&gone) {
                dropped |= self.by_arrival.remove(&arrival).is_some();
            }
        }
        dropped
    }

    /// Adds a log line after every draw that waits. Gives true when the oldest line that
    /// waited was dropped for it.
    fn push_line(&mut self, waiting: Waiting) -> bool {
        let arrival = self.arrival();
        self.by_arrival.insert(arrival, waiting);
        self.lines.push_back(arrival);
        if self.lines.len() <= MAX_WAITING_LINES {
            return false;
        }
        if let Some(oldest) = self.lines.pop_front() {
            self.by_arrival.remove(&oldest);
        }
        true
    }

    fn is_empty(&self) -> bool {
        self.by_arrival.is_empty()
    }

    /// Takes every draw that waits, in the order they came in.
    fn take_all(&mut self) -> Vec<Waiting> {
        self.by_part.clear();
        self.lines.clear();
        std::mem::take(&mut self.by_arrival).into_values().collect()
    }
}

/// The last draw the sender handed on for each part that the viewer shows now: no log
/// lines, which are events, and nothing at or under an entity that a clear removed. So
/// it holds one draw for each part of each entity in the view, and no more.
#[derive(Default)]
struct HandedOn {
    by_part: HashMap<Part, Waiting>,
}

impl HandedOn {
    /// Whether `waiting` changes nothing in the view: it is the same as the last draw
    /// handed on for its part, and either stands for the whole recording, or has no
    /// time, or is less than `REFRESH_AFTER` newer. A draw whose time is before the last
    /// one's is handed on: the clock stepped back. A clear and a log line are always
    /// handed on.
    fn covers(&self, waiting: &Waiting) -> bool {
        let last = waiting
            .call
            .path()
            .and_then(|path| self.by_part.get(&part_of(path, &waiting.call)));
        let Some(last) = last else {
            return false;
        };
        if !waiting.call.same_as(&last.call) {
            return false;
        }
        if waiting.call.is_static() {
            return true;
        }
        match (last.at, waiting.at) {
            (Some(last_at), Some(at)) => at
                .duration_since(last_at)
                .is_ok_and(|newer| newer < REFRESH_AFTER),
            _ => true,
        }
    }

    /// Keeps `waiting` as the last draw handed on for its part. A clear is not kept: it
    /// forgets every part of its entity and of the entities under it, since the viewer
    /// removed them and the next draw of each is new.
    fn keep(&mut self, waiting: Waiting) {
        let part = match &waiting.call {
            Drawn::LogLine { .. } => return,
            Drawn::Clear { path } => {
                self.by_part.retain(|(kept, _), _| !is_under(kept, path));
                return;
            }
            call => match call.path() {
                Some(path) => part_of(path, call),
                None => return,
            },
        };
        self.by_part.insert(part, waiting);
    }
}

/// What the canvas and its sender thread share.
struct Shared {
    slots: Mutex<Slots>,
    /// Wakes the sender when a draw comes in or the canvas closes.
    ready: Condvar,
    /// Warns once for each time the canvas under it falls behind: draws replaced or
    /// dropped before it took them, until it catches up for `ROUNDS_TO_CATCH_UP` rounds.
    behind: LatchedLog,
}

/// A canvas that keeps only the newest draw of each entity, and the log lines in order,
/// until the canvas under it takes them on a thread of its own.
pub struct LiveCanvas {
    shared: Arc<Shared>,
    /// The sender's word that it handed on the last draws.
    closed: Mutex<mpsc::Receiver<()>>,
}

impl LiveCanvas {
    /// Starts the thread that hands the draws on to `under`, oldest first.
    pub fn start(under: Arc<dyn Canvas>) -> Self {
        let shared = Arc::new(Shared {
            slots: Mutex::default(),
            ready: Condvar::new(),
            behind: LatchedLog::warn("keeping up with the viewer's data server"),
        });
        let (closed_sender, closed) = mpsc::channel();
        let sender = shared.clone();
        thread::Builder::new()
            .name("rerun_viewer_sender".to_string())
            .spawn(move || send(&sender, under, &closed_sender))
            .expect("the system starts a thread");
        Self {
            shared,
            closed: Mutex::new(closed),
        }
    }
}

impl DrawnCanvas for LiveCanvas {
    fn take(&self, call: Drawn) {
        match call {
            Drawn::Time(at) => {
                THREAD_TIME.set(Some(at));
                return;
            }
            // The sender flushes the canvas under this one after each round.
            Drawn::Flush => return,
            _ => {}
        }
        let waiting = Waiting {
            at: THREAD_TIME.get(),
            call,
        };
        let mut slots = lock(&self.shared.slots);
        let dropped = match &waiting.call {
            Drawn::LogLine { .. } => slots.push_line(waiting),
            Drawn::Clear { path } => {
                let path = path.clone();
                let dropped = slots.drop_under(&path);
                slots.replace(&path, waiting) || dropped
            }
            call => {
                let path = call
                    .path()
                    .expect("each call but a time and a flush draws at an entity")
                    .to_string();
                slots.replace(&path, waiting)
            }
        };
        slots.dropped |= dropped;
        drop(slots);
        self.shared.ready.notify_one();
    }
}

impl Drop for LiveCanvas {
    /// Lets the sender hand on the draws that wait, for at most `CLOSE_TIMEOUT`.
    fn drop(&mut self) {
        lock(&self.shared.slots).closing = true;
        self.shared.ready.notify_one();
        let closed = self
            .closed
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        if closed.recv_timeout(CLOSE_TIMEOUT).is_err() {
            warn!("the viewer's data server did not take the last draws within {CLOSE_TIMEOUT:?}");
        }
    }
}

/// The sender: hands the draws on to `under` in rounds, until the canvas closes and no
/// draw waits, then says so on `closed`. A round is every draw that waits and changes the
/// view (`HandedOn::covers`), oldest first, then a flush of `under`: until `under` took
/// the round, the draws that come in wait in their slots, so only the newest of each
/// entity goes in the next round. A round with draws replaced or dropped since the last
/// one warns, once until `ROUNDS_TO_CATCH_UP` rounds in a row have none.
fn send(shared: &Shared, under: Arc<dyn Canvas>, closed: &mpsc::Sender<()>) {
    let mut clean_rounds = 0;
    let mut handed_on = HandedOn::default();
    loop {
        let (round, dropped) = {
            let mut slots = lock(&shared.slots);
            while slots.is_empty() && !slots.closing {
                slots = shared
                    .ready
                    .wait(slots)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if slots.is_empty() {
                break;
            }
            (slots.take_all(), std::mem::take(&mut slots.dropped))
        };
        if dropped {
            clean_rounds = 0;
            shared.behind.failure(
                "the viewer's data server takes the draws slower than they come, as when a \
                 page stops reading: each entity keeps only its newest draw until it takes it",
            );
        } else {
            clean_rounds += 1;
            if clean_rounds >= ROUNDS_TO_CATCH_UP {
                shared.behind.success();
            }
        }
        let mut handed = false;
        for waiting in round {
            if handed_on.covers(&waiting) {
                continue;
            }
            if let Some(at) = waiting.at {
                under.set_time(at);
            }
            waiting.call.replay(under.as_ref());
            handed_on.keep(waiting);
            handed = true;
        }
        if handed {
            under.flush();
        }
    }
    // The close may have stopped waiting for this word.
    let _ = closed.send(());
    // The process ends after the close. Rerun's recording flushes with no time bound
    // when it drops, which waits for ever on a page that stopped reading, so the canvas
    // under this one is never dropped.
    std::mem::forget(under);
}

fn lock(slots: &Mutex<Slots>) -> MutexGuard<'_, Slots> {
    slots.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::FakeCanvas;

    /// Bounds every wait of a test that fails; never reached by one that passes.
    const TIMEOUT: Duration = Duration::from_secs(30);

    /// A fake canvas behind a gate: while the gate is shut, the first call that it gates
    /// says so, and each call that it gates waits until the test opens it.
    struct GatedCanvas {
        canvas: FakeCanvas,
        /// Which calls wait at the shut gate.
        gated: fn(&Drawn) -> bool,
        open: Mutex<bool>,
        opened: Condvar,
        /// Says that a call waits at the shut gate.
        waiting: Mutex<Option<mpsc::Sender<()>>>,
    }

    fn every_call(_: &Drawn) -> bool {
        true
    }

    fn the_flushes(call: &Drawn) -> bool {
        matches!(call, Drawn::Flush)
    }

    impl GatedCanvas {
        /// A gate shut for the calls `gated` picks, and the word that one waits at it.
        fn shut(gated: fn(&Drawn) -> bool) -> (Arc<Self>, mpsc::Receiver<()>) {
            let (sender, waits) = mpsc::channel();
            let gated = Self {
                canvas: FakeCanvas::default(),
                gated,
                open: Mutex::new(false),
                opened: Condvar::new(),
                waiting: Mutex::new(Some(sender)),
            };
            (Arc::new(gated), waits)
        }

        fn open(&self) {
            *self.open.lock().unwrap() = true;
            self.opened.notify_all();
        }
    }

    impl DrawnCanvas for GatedCanvas {
        fn take(&self, call: Drawn) {
            if (self.gated)(&call) {
                let mut open = self.open.lock().unwrap();
                if !*open && let Some(waiting) = self.waiting.lock().unwrap().take() {
                    waiting.send(()).unwrap();
                }
                while !*open {
                    open = self.opened.wait(open).unwrap();
                }
            }
            self.canvas.take(call);
        }
    }

    fn at(seconds: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)
    }

    /// The calls the canvas took at an entity.
    fn entity_calls(drawn: &[Drawn]) -> usize {
        drawn.iter().filter(|call| call.path().is_some()).count()
    }

    /// The texts the canvas drew, as (path, text).
    fn texts(canvas: &FakeCanvas) -> Vec<(String, String)> {
        canvas
            .drawn()
            .into_iter()
            .filter_map(|drawn| match drawn {
                Drawn::Text { path, markdown } => Some((path, markdown)),
                Drawn::Clear { path } => Some((path, "clear".to_string())),
                _ => None,
            })
            .collect()
    }

    fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
        expected
            .iter()
            .map(|(path, text)| (path.to_string(), text.to_string()))
            .collect()
    }

    /// Draws `text` at `path` at `seconds`, as a drawer does.
    fn draw(canvas: &LiveCanvas, seconds: u64, path: &str, text: &str) {
        canvas.set_time(at(seconds));
        canvas.text(path, text);
    }

    /// Holds the sender at the shut gate with a first draw at `held`.
    fn hold_the_sender(live: &LiveCanvas, waits: &mpsc::Receiver<()>) {
        draw(live, 1, "held", "first");
        waits
            .recv_timeout(TIMEOUT)
            .expect("the sender reaches the gate");
    }

    #[tokio::test]
    async fn each_draw_reaches_the_canvas_under_it_with_the_time_of_its_thread() {
        let under = Arc::new(FakeCanvas::default());
        let live = LiveCanvas::start(under.clone());
        draw(&live, 5, "a", "one");
        draw(&live, 7, "b", "two");
        under
            .wait_until(TIMEOUT, "both texts", |drawn| entity_calls(drawn) == 2)
            .await
            .unwrap();
        let drawn: Vec<Drawn> = under
            .drawn()
            .into_iter()
            .filter(|call| !matches!(call, Drawn::Flush))
            .collect();
        assert!(matches!(drawn[0], Drawn::Time(time) if time == at(5)));
        assert!(
            matches!(&drawn[1], Drawn::Text { path, markdown } if path == "a" && markdown == "one")
        );
        assert!(matches!(drawn[2], Drawn::Time(time) if time == at(7)));
        assert!(
            matches!(&drawn[3], Drawn::Text { path, markdown } if path == "b" && markdown == "two")
        );
    }

    #[tokio::test]
    async fn while_the_canvas_under_it_waits_each_entity_keeps_only_its_newest_draw() {
        let (gated, waits) = GatedCanvas::shut(every_call);
        let live = LiveCanvas::start(gated.clone());
        hold_the_sender(&live, &waits);
        draw(&live, 2, "pose", "older");
        draw(&live, 3, "map", "only");
        draw(&live, 4, "pose", "newer");
        gated.open();
        gated
            .canvas
            .wait_until(TIMEOUT, "the newest draws", |drawn| {
                entity_calls(drawn) == 3
            })
            .await
            .unwrap();
        // The newer pose replaced the older one and came in after the map.
        assert_eq!(
            texts(&gated.canvas),
            pairs(&[("held", "first"), ("map", "only"), ("pose", "newer")])
        );
        let times: Vec<SystemTime> = gated
            .canvas
            .drawn()
            .into_iter()
            .filter_map(|drawn| match drawn {
                Drawn::Time(time) => Some(time),
                _ => None,
            })
            .collect();
        assert_eq!(times, [at(1), at(3), at(4)], "each draw keeps its own time");
    }

    #[tokio::test]
    async fn until_the_canvas_under_it_took_a_round_the_next_keeps_only_the_newest_draws() {
        let (gated, waits) = GatedCanvas::shut(the_flushes);
        let live = LiveCanvas::start(gated.clone());
        draw(&live, 1, "pose", "first");
        waits
            .recv_timeout(TIMEOUT)
            .expect("the sender flushes its first round");
        draw(&live, 2, "pose", "older");
        draw(&live, 3, "pose", "newer");
        gated.open();
        gated
            .canvas
            .wait_until(TIMEOUT, "the newest pose", |drawn| entity_calls(drawn) == 2)
            .await
            .unwrap();
        assert_eq!(
            texts(&gated.canvas),
            pairs(&[("pose", "first"), ("pose", "newer")])
        );
        let drawn = gated.canvas.drawn();
        let position = |wanted: &dyn Fn(&Drawn) -> bool| drawn.iter().position(wanted);
        let first =
            position(&|call| matches!(call, Drawn::Text { markdown, .. } if markdown == "first"));
        let flush = position(&|call| matches!(call, Drawn::Flush));
        let newer =
            position(&|call| matches!(call, Drawn::Text { markdown, .. } if markdown == "newer"));
        assert!(
            first < flush && flush < newer,
            "a flush ends each round: {drawn:#?}"
        );
    }

    /// A live canvas over a fake one, for a test that hands its draws on one by one.
    struct Stage {
        under: Arc<FakeCanvas>,
        live: LiveCanvas,
        settled: Cell<u32>,
    }

    impl Stage {
        fn new() -> Self {
            let under = Arc::new(FakeCanvas::default());
            Self {
                live: LiveCanvas::start(under.clone()),
                under,
                settled: Cell::new(0),
            }
        }

        /// Waits until the sender went through every draw so far: it draws at an entity
        /// of its own, which no draw replaces, and waits for that draw.
        async fn settle(&self) {
            let count = self.settled.get() + 1;
            self.settled.set(count);
            let path = format!("settle/{count}");
            self.live.text(&path, "");
            self.under
                .wait_until(TIMEOUT, "the settle draw", |drawn| {
                    drawn.iter().any(|call| call.path() == Some(path.as_str()))
                })
                .await
                .unwrap();
        }

        /// Draws `text` at `path` at `seconds`, and waits until the sender went through it.
        async fn draw(&self, seconds: u64, path: &str, text: &str) {
            draw(&self.live, seconds, path, text);
            self.settle().await;
        }

        /// The texts and the clears that the canvas under it took, the settle draws aside.
        fn taken(&self) -> Vec<(String, String)> {
            texts(&self.under)
                .into_iter()
                .filter(|(path, _)| !path.starts_with("settle/"))
                .collect()
        }
    }

    #[tokio::test]
    async fn a_draw_that_changes_nothing_is_handed_on_again_only_after_the_refresh_time() {
        let stage = Stage::new();
        let refresh = REFRESH_AFTER.as_secs();
        stage.draw(10, "map", "same").await;
        stage.draw(10 + refresh - 1, "map", "same").await;
        stage.draw(10 + refresh, "map", "same").await;
        stage.draw(10 + refresh + 1, "map", "other").await;
        assert_eq!(
            stage.taken(),
            pairs(&[("map", "same"), ("map", "same"), ("map", "other")]),
            "the second draw, 1 s short of the refresh time, is left out"
        );
    }

    #[tokio::test]
    async fn the_values_of_a_plot_and_the_names_of_its_lines_stand_side_by_side() {
        let (gated, waits) = GatedCanvas::shut(every_call);
        let live = LiveCanvas::start(gated.clone());
        hold_the_sender(&live, &waits);
        let names = ["lift".to_string(), "reach".to_string()];
        // Two snapshots while the canvas under it waits: the names and the values of
        // one plot wait side by side, and the newer values replace the older ones.
        live.series_names("plots/joints/arm", &names);
        live.scalars("plots/joints/arm", &[0.1, 0.2]);
        live.series_names("plots/joints/arm", &names);
        live.scalars("plots/joints/arm", &[0.3, 0.4]);
        gated.open();
        gated
            .canvas
            .wait_until(TIMEOUT, "the plot", |drawn| entity_calls(drawn) == 3)
            .await
            .unwrap();
        assert!(matches!(
            &gated.canvas.at("plots/joints/arm")[..],
            [Drawn::SeriesNames { names: lines, .. }, Drawn::Scalars { values, .. }]
                if lines == &names && values == &[0.3, 0.4]
        ));
        // A third snapshot with the same values, once the first ones are handed on:
        // the names stand for the whole recording, and the values changed nothing.
        let stage = Stage::new();
        for _ in 0..2 {
            stage.live.set_time(at(1));
            stage.live.series_names("plots/joints/arm", &names);
            stage.live.scalars("plots/joints/arm", &[0.3, 0.4]);
            stage.settle().await;
        }
        assert_eq!(stage.under.at("plots/joints/arm").len(), 2);
    }

    #[tokio::test]
    async fn a_draw_before_the_last_one_handed_on_is_handed_on() {
        let stage = Stage::new();
        stage.draw(10, "map", "same").await;
        // The clock stepped back, as when a simulation starts again.
        stage.draw(3, "map", "same").await;
        assert_eq!(stage.taken(), pairs(&[("map", "same"), ("map", "same")]));
    }

    #[tokio::test]
    async fn a_draw_for_the_whole_recording_is_handed_on_once() {
        let stage = Stage::new();
        stage.live.set_time(at(1));
        stage.live.note("notes/cubes", "a note");
        stage.settle().await;
        stage.live.set_time(at(1 + 10 * REFRESH_AFTER.as_secs()));
        stage.live.note("notes/cubes", "a note");
        stage.settle().await;
        assert_eq!(stage.under.at("notes/cubes").len(), 1);
    }

    #[tokio::test]
    async fn after_a_clear_the_same_draw_under_it_is_handed_on_again() {
        let stage = Stage::new();
        stage.draw(1, "scene/lime/shape", "lime").await;
        stage.live.clear("scene/lime");
        stage.settle().await;
        stage.draw(1, "scene/lime/shape", "lime").await;
        assert_eq!(
            stage.taken(),
            pairs(&[
                ("scene/lime/shape", "lime"),
                ("scene/lime", "clear"),
                ("scene/lime/shape", "lime")
            ])
        );
    }

    #[test]
    fn a_clear_leaves_no_draw_kept_at_its_entity_or_under_it() {
        let text_at = |path: &str| Waiting {
            at: Some(at(1)),
            call: Drawn::Text {
                path: path.to_string(),
                markdown: "lime".to_string(),
            },
        };
        let mut handed_on = HandedOn::default();
        handed_on.keep(text_at("scene/lime"));
        handed_on.keep(text_at("scene/lime/shape"));
        handed_on.keep(text_at("scene/limestone"));
        handed_on.keep(Waiting {
            at: Some(at(2)),
            call: Drawn::Clear {
                path: "scene/lime".to_string(),
            },
        });
        let left: Vec<&str> = handed_on
            .by_part
            .keys()
            .map(|(path, _)| path.as_str())
            .collect();
        assert_eq!(
            left,
            ["scene/limestone"],
            "only the entity whose name starts the same stays"
        );
    }

    #[tokio::test]
    async fn a_log_line_is_always_handed_on() {
        let stage = Stage::new();
        stage.live.set_time(at(1));
        stage.live.log_line("scene/changes", "moved brain/lime_1");
        stage.settle().await;
        stage.live.log_line("scene/changes", "moved brain/lime_1");
        stage.settle().await;
        assert_eq!(stage.under.log_lines("scene/changes").len(), 2);
    }

    #[tokio::test]
    async fn a_clear_drops_the_draws_that_wait_under_it_and_a_later_draw_follows_it() {
        let (gated, waits) = GatedCanvas::shut(every_call);
        let live = LiveCanvas::start(gated.clone());
        hold_the_sender(&live, &waits);
        draw(&live, 2, "scene/lime/shape", "lime");
        draw(&live, 2, "scene/pear", "pear");
        live.clear("scene/lime");
        live.clear("scene/pear");
        draw(&live, 3, "scene/pear", "pear again");
        gated.open();
        gated
            .canvas
            .wait_until(TIMEOUT, "the clears and the new pear", |drawn| {
                entity_calls(drawn) == 4
            })
            .await
            .unwrap();
        // The clear of the pear still goes, before the new pear: it also removes what
        // the viewer holds under the pear.
        assert_eq!(
            texts(&gated.canvas),
            pairs(&[
                ("held", "first"),
                ("scene/lime", "clear"),
                ("scene/pear", "clear"),
                ("scene/pear", "pear again")
            ])
        );
    }

    #[tokio::test]
    async fn log_lines_keep_their_order_and_the_oldest_go_past_the_cap() {
        let (gated, waits) = GatedCanvas::shut(every_call);
        let live = LiveCanvas::start(gated.clone());
        hold_the_sender(&live, &waits);
        let extra = 10;
        for index in 0..MAX_WAITING_LINES + extra {
            live.log_line("scene/changes", &format!("line {index}"));
        }
        gated.open();
        gated
            .canvas
            .wait_until(TIMEOUT, "the lines", |drawn| {
                drawn
                    .iter()
                    .filter(|call| matches!(call, Drawn::LogLine { .. }))
                    .count()
                    == MAX_WAITING_LINES
            })
            .await
            .unwrap();
        let lines = gated.canvas.log_lines("scene/changes");
        let expected: Vec<String> = (extra..MAX_WAITING_LINES + extra)
            .map(|index| format!("line {index}"))
            .collect();
        assert_eq!(lines, expected);
    }

    #[tokio::test]
    async fn the_draws_that_wait_when_the_canvas_closes_still_reach_the_canvas_under_it() {
        let (gated, waits) = GatedCanvas::shut(every_call);
        let live = LiveCanvas::start(gated.clone());
        hold_the_sender(&live, &waits);
        draw(&live, 2, "map", "last");
        let shared = live.shared.clone();
        let close = thread::spawn(move || drop(live));
        while !lock(&shared.slots).closing {
            thread::yield_now();
        }
        gated.open();
        gated
            .canvas
            .wait_until(TIMEOUT, "the last map", |drawn| entity_calls(drawn) == 2)
            .await
            .unwrap();
        close.join().unwrap();
        assert_eq!(
            texts(&gated.canvas),
            pairs(&[("held", "first"), ("map", "last")])
        );
    }
}
