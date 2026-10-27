//! The journals of the runs: one JSON Lines file per run in the directory
//! of the director, each line one record with its sequence number and its
//! times. A journal holds the run's inputs, its events and its end; the
//! records of runs survive a restart of the director, and a run that has no
//! end when the director starts ends interrupted. The store reads each
//! journal once, when it opens, and keeps the size and the end of each.

use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

/// One line of a journal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub seq: u64,
    pub run_ns: u64,
    pub wall_ns: u64,
    #[serde(rename = "type")]
    pub kind: String,
    pub body: Json,
}

/// The kinds of records.
pub const OPENED: &str = "opened";
pub const CALL: &str = "call";
pub const EVENT: &str = "event";
pub const END: &str = "end";
/// The inputs of the plan of a run: its fleet, its documents and its roles.
pub const PLANNED: &str = "planned";
/// A topic read of a run.
pub const READ: &str = "read";
/// A command to a run: a confirm, a suspend, a resume, an abort, a stop or a
/// fault, and whether it came before the engine started.
pub const COMMAND: &str = "command";
/// An alert of severity 1 of a robot of the run, which the journal keeps for
/// the reader and a replay does not decide on.
pub const ALERT: &str = "alert";
/// The locations a skill instance of the run knows, listed before the
/// assumptions: `{ role, instance, locations }`.
pub const LOCATIONS: &str = "locations";
/// The answer of the daemon to an open or a close of a lease of the run:
/// `{ robot, open, refusal }`.
pub const LEASE: &str = "lease";
/// A step of a read of an observation that joined another reader's read in
/// flight: `{ name, nth, join }`, where `join` waits, took its answer, or
/// reads itself.
pub const SHARED: &str = "shared";

/// The journal store of the director.
pub struct Journals {
    dir: PathBuf,
    max_runs: u32,
    max_bytes: u64,
    /// Each journal of the directory, by run.
    kept: Mutex<BTreeMap<String, Kept>>,
}

/// What the store keeps of one journal: the sequence number of its next
/// record, its size and whether it has its end.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Kept {
    next_seq: u64,
    bytes: u64,
    ended: bool,
}

/// The file that keeps the next run number, so ids grow after pruning.
const NEXT_RUN_FILE: &str = "next_run";

/// The number of the run id `r<N>`.
pub fn run_number(run: &str) -> Option<u64> {
    run.strip_prefix('r')?.parse().ok()
}

/// The records of the text of a journal, and how many of its bytes hold
/// them. A last line with no newline is one the director did not finish
/// writing: it is kept when it parses, and left out when it does not.
fn records_of(text: &str) -> Result<(Vec<Record>, usize), String> {
    let mut records = Vec::new();
    let mut held = 0;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        let finished = line.ends_with('\n');
        match serde_json::from_str::<Record>(line.trim_end()) {
            Ok(record) => records.push(record),
            Err(_) if !finished => break,
            Err(error) => return Err(format!("line {}: {error}", index + 1)),
        }
        held += line.len();
    }
    Ok((records, held))
}

impl Journals {
    /// Opens the store under `root`, and reads each journal once. A
    /// journal whose last line the director did not finish writing is cut
    /// back to its records, with their last newline.
    pub fn open(root: &Path, max_runs: u32, max_bytes: u64) -> std::io::Result<Journals> {
        let dir = root.join("workflow_director").join("journals");
        fs::create_dir_all(&dir)?;
        let mut kept = BTreeMap::new();
        for entry in fs::read_dir(&dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            let Some(run) = name
                .strip_suffix(".jsonl")
                .filter(|run| run_number(run).is_some())
            else {
                continue;
            };
            let path = dir.join(&name);
            kept.insert(run.to_owned(), repaired(&path)?);
        }
        Ok(Journals {
            dir,
            max_runs,
            max_bytes,
            kept: Mutex::new(kept),
        })
    }

    fn path(&self, run: &str) -> PathBuf {
        self.dir.join(format!("{run}.jsonl"))
    }

    /// The next run id, `r<N>`, one more than every id given before.
    pub fn next_run(&self) -> std::io::Result<String> {
        let kept = self.kept.lock().expect("no journal writer panics");
        let file = self.dir.join(NEXT_RUN_FILE);
        let given = fs::read_to_string(&file)
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .unwrap_or(0);
        let on_disk = kept
            .keys()
            .filter_map(|run| run_number(run))
            .max()
            .unwrap_or(0);
        let next = given.max(on_disk) + 1;
        fs::write(&file, next.to_string())?;
        Ok(format!("r{next}"))
    }

    /// The runs that have a journal, oldest first.
    pub fn runs(&self) -> Vec<String> {
        let kept = self.kept.lock().expect("no journal writer panics");
        oldest_first(kept.keys())
    }

    /// Appends a record of `kind` to the journal of `run`, with its sequence
    /// number and the wall time of now.
    pub fn append(&self, run: &str, run_ns: u64, kind: &str, body: Json) -> std::io::Result<()> {
        let mut kept = self.kept.lock().expect("no journal writer panics");
        let journal = kept.entry(run.to_owned()).or_default();
        let record = Record {
            seq: journal.next_seq,
            run_ns,
            wall_ns: wall_ns(),
            kind: kind.to_owned(),
            body,
        };
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path(run))?;
        let mut line = serde_json::to_string(&record).map_err(std::io::Error::other)?;
        line.push('\n');
        file.write_all(line.as_bytes())?;
        journal.next_seq += 1;
        journal.bytes += line.len() as u64;
        journal.ended |= kind == END;
        Ok(())
    }

    /// The records of the journal of `run`.
    pub fn read(&self, run: &str) -> std::io::Result<Vec<Record>> {
        let path = self.path(run);
        let text = fs::read_to_string(&path)?;
        records_of(&text)
            .map(|(records, _)| records)
            .map_err(|error| std::io::Error::other(format!("{}: {error}", path.display())))
    }

    pub fn text(&self, run: &str) -> std::io::Result<String> {
        fs::read_to_string(self.path(run))
    }

    /// Removes the journals of ended runs, oldest first, beyond the bounds,
    /// and gives the runs it removed.
    pub fn prune(&self) -> std::io::Result<Vec<String>> {
        let mut kept = self.kept.lock().expect("no journal writer panics");
        let mut ended: VecDeque<String> = oldest_first(kept.keys())
            .into_iter()
            .filter(|run| kept[run].ended)
            .collect();
        let mut bytes: u64 = ended.iter().map(|run| kept[run].bytes).sum();
        let mut removed = Vec::new();
        while ended.len() > self.max_runs as usize || bytes > self.max_bytes {
            let Some(oldest) = ended.pop_front() else {
                break;
            };
            fs::remove_file(self.path(&oldest))?;
            bytes -= kept.remove(&oldest).map_or(0, |journal| journal.bytes);
            removed.push(oldest);
        }
        Ok(removed)
    }
}

/// The runs of `runs` by their number, oldest first.
fn oldest_first<'a>(runs: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut runs: Vec<String> = runs.cloned().collect();
    runs.sort_by_key(|run| run_number(run));
    runs
}

/// Reads the journal at `path`, cuts it back to its records, each with its
/// newline, and gives what the store keeps of it.
fn repaired(path: &Path) -> std::io::Result<Kept> {
    let text = fs::read_to_string(path)?;
    let (records, held) = records_of(&text).map_err(|error| {
        std::io::Error::other(format!(
            "{}: {error}; move this journal out of its directory, and the director starts without it",
            path.display()
        ))
    })?;
    if held < text.len() {
        OpenOptions::new()
            .write(true)
            .open(path)?
            .set_len(held as u64)?;
    }
    let mut bytes = held as u64;
    if !text[..held].is_empty() && !text[..held].ends_with('\n') {
        OpenOptions::new()
            .append(true)
            .open(path)?
            .write_all(b"\n")?;
        bytes += 1;
    }
    Ok(Kept {
        next_seq: records.len() as u64,
        bytes,
        ended: records.iter().any(|record| record.kind == END),
    })
}

pub fn wall_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(max_runs: u32) -> (tempfile::TempDir, Journals) {
        let dir = tempfile::tempdir().unwrap();
        let journals = Journals::open(dir.path(), max_runs, u64::MAX).unwrap();
        (dir, journals)
    }

    #[test]
    fn run_ids_grow_across_pruning_and_records_read_back() {
        let (_dir, journals) = store(1);
        let first = journals.next_run().unwrap();
        journals.append(&first, 0, OPENED, Json::Null).unwrap();
        journals.append(&first, 1, END, Json::Null).unwrap();
        let second = journals.next_run().unwrap();
        journals.append(&second, 0, OPENED, Json::Null).unwrap();
        journals.append(&second, 1, END, Json::Null).unwrap();
        assert_eq!(journals.prune().unwrap(), vec![first.clone()]);
        assert_eq!(journals.runs(), vec![second.clone()]);
        assert_eq!(journals.next_run().unwrap(), "r3");
        let seqs: Vec<u64> = journals
            .read(&second)
            .unwrap()
            .iter()
            .map(|r| r.seq)
            .collect();
        assert_eq!(seqs, vec![0, 1]);
    }

    #[test]
    fn a_journal_with_no_end_is_never_pruned() {
        let (_dir, journals) = store(0);
        let open = journals.next_run().unwrap();
        journals.append(&open, 0, OPENED, Json::Null).unwrap();
        assert_eq!(journals.prune().unwrap(), Vec::<String>::new());
        journals.append(&open, 0, END, Json::Null).unwrap();
        assert_eq!(journals.prune().unwrap(), vec![open]);
    }

    /// A store whose journal `r1` holds two records and then `tail`.
    fn store_with_tail(tail: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let journals = Journals::open(dir.path(), 10, u64::MAX).unwrap();
        let run = journals.next_run().unwrap();
        journals.append(&run, 0, OPENED, Json::Null).unwrap();
        journals.append(&run, 1, CALL, Json::Null).unwrap();
        let path = journals.path(&run);
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(tail.as_bytes()).unwrap();
        (dir, path)
    }

    #[test]
    fn a_last_line_the_director_did_not_finish_is_cut_at_open() {
        let (dir, path) = store_with_tail(r#"{"seq":2,"run_ns":2,"wall"#);
        let journals = Journals::open(dir.path(), 10, u64::MAX).unwrap();
        journals.append("r1", 3, END, Json::Null).unwrap();
        let records = journals.read("r1").unwrap();
        let seqs: Vec<u64> = records.iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![0, 1, 2]);
        assert_eq!(records[2].kind, END);
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(journals.kept.lock().unwrap()["r1"].bytes, text.len() as u64);
    }

    #[test]
    fn a_whole_last_record_without_its_newline_is_kept() {
        let record = Record {
            seq: 2,
            run_ns: 2,
            wall_ns: 2,
            kind: END.to_owned(),
            body: Json::Null,
        };
        let (dir, path) = store_with_tail(&serde_json::to_string(&record).unwrap());
        let journals = Journals::open(dir.path(), 0, u64::MAX).unwrap();
        assert_eq!(journals.read("r1").unwrap().len(), 3);
        assert!(fs::read_to_string(&path).unwrap().ends_with('\n'));
        assert_eq!(
            journals.prune().unwrap(),
            vec!["r1".to_owned()],
            "the run ended"
        );
    }

    #[test]
    fn a_finished_line_that_does_not_parse_names_its_journal_and_line() {
        let (dir, _path) = store_with_tail("not a record\n");
        let error = Journals::open(dir.path(), 10, u64::MAX)
            .err()
            .expect("refused");
        let message = error.to_string();
        assert!(
            message.contains("r1.jsonl") && message.contains("line 3"),
            "{message}"
        );
        assert!(
            message.contains("move this journal out of its directory"),
            "{message}"
        );
    }

    #[test]
    fn the_journals_past_the_size_bound_go_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let journals = Journals::open(dir.path(), 10, 1).unwrap();
        let runs: Vec<String> = (0..3).map(|_| journals.next_run().unwrap()).collect();
        for run in &runs {
            journals.append(run, 0, END, Json::Null).unwrap();
        }
        assert_eq!(journals.prune().unwrap(), runs);
        assert!(journals.runs().is_empty());
    }
}
