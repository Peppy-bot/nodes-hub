//! `session.json`: what this session recorded and where it came from. The
//! drain tasks only note producers into memory; a dedicated task rewrites the
//! file off the hot path.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use peppygen::Parameters;
use peppylib::runtime::CancellationToken;
use serde_json::json;
use tokio::sync::Notify;
use tracing::warn;

/// (core_node, instance_id) pairs seen per stream name.
type ProducersByStream = BTreeMap<&'static str, BTreeSet<(String, String)>>;

/// Producers observed per stream, updated by every drain task.
#[derive(Clone)]
pub struct ProducerLog {
    inner: Arc<Mutex<ProducersByStream>>,
    changed: Arc<Notify>,
}

impl ProducerLog {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(BTreeMap::new())),
            changed: Arc::new(Notify::new()),
        }
    }

    /// Cheap on the hot path: a lookup, and an insert + notify only the first
    /// time a producer is seen on a stream.
    pub fn observe(&self, stream: &'static str, core_node: &str, instance_id: &str) {
        let mut map = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let entry = map.entry(stream).or_default();
        let key = (core_node.to_string(), instance_id.to_string());
        if entry.insert(key) {
            self.changed.notify_one();
        }
    }

    fn snapshot(&self) -> ProducersByStream {
        self.inner.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

pub struct Session {
    path: PathBuf,
    params: Parameters,
}

impl Session {
    pub fn new(session_dir: &Path, params: Parameters) -> Self {
        Self {
            path: session_dir.join("session.json"),
            params,
        }
    }

    fn write(&self, log: &ProducerLog) {
        let producers: BTreeMap<&str, Vec<serde_json::Value>> = log
            .snapshot()
            .into_iter()
            .map(|(stream, set)| {
                let list = set
                    .into_iter()
                    .map(|(core_node, instance_id)| {
                        json!({"core_node": core_node, "instance_id": instance_id})
                    })
                    .collect();
                (stream, list)
            })
            .collect();
        let doc = json!({
            "recorder": {
                "version": env!("CARGO_PKG_VERSION"),
                "git_rev": env!("RECORDER_GIT_REV"),
            },
            "parameters": &self.params,
            "producers": producers,
        });
        let text = serde_json::to_string_pretty(&doc).expect("session doc serializes");
        let tmp = self.path.with_extension("json.tmp");
        let result = std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, &self.path));
        if let Err(e) = result {
            warn!("writing {}: {e}", self.path.display());
        }
    }

    /// Writes once at startup, then again whenever a new producer appears.
    pub async fn run(self, log: ProducerLog, token: CancellationToken) {
        self.write(&log);
        loop {
            tokio::select! {
                _ = token.cancelled() => return,
                _ = log.changed.notified() => self.write(&log),
            }
        }
    }
}
