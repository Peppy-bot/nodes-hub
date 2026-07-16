//! The episode manager: single owner of the recorder state machine and the
//! fps sampler. Service commands and pacer ticks meet in one select loop, so
//! start/stop and sampling cannot race. The dataset schema is captured from
//! the discovered sources at the first start.

use std::sync::Arc;
use std::sync::mpsc::SyncSender;

use control_core::Pacer;
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};

use crate::config::Config;
use crate::lerobot_sink::{CameraInit, SinkHandle};
use crate::plan::RecordingPlan;
use crate::snapshot::{CacheReader, CacheView, SourceSchema, sample};
use crate::status::{Status, StatusState};

pub struct StartReply {
    pub accepted: bool,
    pub episode_index: i64,
    pub message: String,
}

pub struct StopReply {
    pub accepted: bool,
    pub episode_index: i64,
    pub frames: u64,
    pub message: String,
}

pub enum Command {
    Start {
        task: Option<String>,
        reply: SyncSender<StartReply>,
    },
    Stop {
        save: bool,
        reply: SyncSender<StopReply>,
    },
}

const DISK_FLOOR_BYTES: u64 = 1 << 30;

enum State {
    Idle,
    Recording {
        episode_index: i64,
        task: String,
        frames: u64,
    },
    /// Terminal: a writer save failed. The node stays alive but refuses further
    /// starts and reports the failure on `recorder_status`.
    Error {
        episode_index: i64,
        message: String,
    },
}

pub struct Manager {
    config: Config,
    plan: Arc<RecordingPlan>,
    cache: CacheReader,
    sink: SinkHandle,
    status: watch::Sender<Status>,
    state: State,
    last_episode_index: i64,
    /// Captured on the first successful start; reused for later episodes.
    schema: Option<SourceSchema>,
}

impl Manager {
    pub fn new(
        config: Config,
        plan: Arc<RecordingPlan>,
        cache: CacheReader,
        sink: SinkHandle,
        status: watch::Sender<Status>,
    ) -> Self {
        Self {
            config,
            plan,
            cache,
            sink,
            status,
            state: State::Idle,
            last_episode_index: -1,
            schema: None,
        }
    }

    pub async fn run(mut self, mut commands: mpsc::Receiver<Command>, token: CancellationToken) {
        let period = std::time::Duration::from_secs_f64(1.0 / self.config.fps.get() as f64);
        let mut pacer = Pacer::new(period).expect("fps is non-zero");
        loop {
            let recording = matches!(self.state, State::Recording { .. });
            tokio::select! {
                biased;
                _ = token.cancelled() => {
                    if recording {
                        info!("shutdown while recording: saving the open episode");
                        self.stop(true).await;
                    }
                    return;
                }
                command = commands.recv() => {
                    match command {
                        Some(Command::Start { task, reply }) => {
                            let _ = reply.send(self.start(task).await);
                        }
                        Some(Command::Stop { save, reply }) => {
                            let _ = reply.send(self.stop(save).await);
                        }
                        None => return,
                    }
                }
                _ = pacer.pace(), if recording => self.tick().await,
            }
        }
    }

    fn publish_status(&self, message: &str) {
        let (state, episode_index, frames, task, message) = match &self.state {
            State::Idle => (
                StatusState::Idle,
                self.last_episode_index,
                0,
                String::new(),
                message.to_string(),
            ),
            State::Recording {
                episode_index,
                task,
                frames,
            } => (
                StatusState::Recording,
                *episode_index,
                *frames,
                task.clone(),
                message.to_string(),
            ),
            // The terminal error message is fixed; the caller's message is moot.
            State::Error {
                episode_index,
                message,
            } => (
                StatusState::Error,
                *episode_index,
                0,
                String::new(),
                message.clone(),
            ),
        };
        self.status.send_replace(Status {
            state,
            episode_index,
            frames,
            task,
            message,
        });
    }

    /// First failed precondition at the moment of start, if any. `now_ns` is
    /// the recorder's synchronized clock, compared against producer stamps.
    fn refuse_reason(&self, view: &CacheView, now_ns: u64) -> Option<String> {
        let free = fs2::available_space(&self.config.output_root).unwrap_or(0);
        if free < DISK_FLOOR_BYTES {
            return Some(format!(
                "only {} MB free under output_root",
                free / (1024 * 1024)
            ));
        }
        for (i, slot) in view.states.iter().enumerate() {
            if slot.is_none() {
                return Some(format!("state source {i} has not produced yet"));
            }
        }
        for slot in &view.actions {
            if slot.is_none() {
                return Some("an action source has not produced yet".to_string());
            }
        }
        let start_fresh_ns = self.config.camera_start_fresh.as_nanos() as u64;
        for (slot, entry) in view.cameras.iter().zip(&self.plan.cameras) {
            let fresh = slot
                .as_ref()
                .is_some_and(|f| f.age_ns(now_ns) <= start_fresh_ns);
            if !fresh {
                return Some(format!("camera {} has no fresh frame", entry.key));
            }
        }
        None
    }

    fn camera_inits(&self, view: &CacheView) -> Option<Vec<CameraInit>> {
        self.plan
            .cameras
            .iter()
            .zip(&view.cameras)
            .map(|(entry, slot)| {
                let frame = slot.as_ref()?;
                Some(CameraInit {
                    key: entry.key.clone(),
                    width: std::num::NonZeroU32::new(frame.value.width)?,
                    height: std::num::NonZeroU32::new(frame.value.height)?,
                    encoding: frame.value.encoding,
                })
            })
            .collect()
    }

    async fn start(&mut self, task: Option<String>) -> StartReply {
        let refuse = |message: String| StartReply {
            accepted: false,
            episode_index: -1,
            message,
        };
        if let State::Error { message, .. } = &self.state {
            return refuse(format!("recorder halted after a writer failure: {message}"));
        }
        if matches!(self.state, State::Recording { .. }) {
            return refuse("already recording".to_string());
        }
        // Staleness gates compare producer stamps against the recorder's
        // synchronized clock; without a ready clock no gate is meaningful.
        let now_ns = match peppygen::clock::now_ns() {
            Ok(now_ns) => now_ns,
            Err(e) => return refuse(format!("recorder clock not ready: {e}")),
        };
        let view = self.cache.view();
        if let Some(reason) = self.refuse_reason(&view, now_ns) {
            self.publish_status(&reason);
            return refuse(reason);
        }
        let schema = match &self.schema {
            Some(schema) => schema.clone(),
            None => match SourceSchema::from_view(&view, &self.plan) {
                Ok(schema) => {
                    self.schema = Some(schema.clone());
                    schema
                }
                Err(e) => return refuse(e),
            },
        };
        let Some(cameras) = self.camera_inits(&view) else {
            return refuse("a camera frame has zero dimensions".to_string());
        };

        let task = task.unwrap_or_else(|| self.config.default_task.clone());
        match self.sink.begin(task.clone(), schema, cameras).await {
            Ok(()) => {
                let episode_index = self.last_episode_index + 1;
                info!("episode {episode_index} started: {task:?}");
                self.state = State::Recording {
                    episode_index,
                    task,
                    frames: 0,
                };
                self.publish_status("");
                StartReply {
                    accepted: true,
                    episode_index,
                    message: String::new(),
                }
            }
            Err(e) => {
                warn!("start refused by writer: {e}");
                self.publish_status(&e);
                refuse(e)
            }
        }
    }

    async fn stop(&mut self, save: bool) -> StopReply {
        let State::Recording { episode_index, .. } = self.state else {
            return StopReply {
                accepted: false,
                episode_index: -1,
                frames: 0,
                message: "not recording".to_string(),
            };
        };
        match self.sink.end(save).await {
            Ok(summary) => {
                if save {
                    self.last_episode_index = episode_index;
                }
                self.state = State::Idle;
                let message = if save { "" } else { "episode discarded" };
                info!(
                    "episode {episode_index} {}: {} frames",
                    if save { "saved" } else { "discarded" },
                    summary.frames
                );
                self.publish_status(message);
                StopReply {
                    accepted: true,
                    episode_index,
                    frames: summary.frames,
                    message: message.to_string(),
                }
            }
            Err(e) => {
                warn!("episode {episode_index} save failed: {e}");
                self.state = State::Error {
                    episode_index,
                    message: e.clone(),
                };
                self.publish_status("");
                StopReply {
                    accepted: false,
                    episode_index,
                    frames: 0,
                    message: e,
                }
            }
        }
    }

    async fn tick(&mut self) {
        let State::Recording {
            episode_index,
            frames,
            ..
        } = &mut self.state
        else {
            return;
        };
        let episode_index = *episode_index;
        // The clock was ready at start; losing it mid-episode is a gap like any
        // other, ending the episode with a save.
        let now_ns = match peppygen::clock::now_ns() {
            Ok(now_ns) => now_ns,
            Err(e) => {
                warn!("episode {episode_index}: clock unavailable ({e}), stopping with save");
                self.stop_with_message("recorder clock unavailable").await;
                return;
            }
        };
        let view = self.cache.view();
        let schema = self.schema.as_ref().expect("schema set before recording");
        match sample(&view, schema, &self.plan, &self.config, now_ns) {
            Ok(row) => {
                if self.sink.try_frame(row).is_err() {
                    warn!("episode {episode_index}: encoder backpressure, stopping with save");
                    self.stop_with_message("encoder backpressure").await;
                    return;
                }
                *frames += 1;
                let frames = *frames;
                if frames.is_multiple_of(self.config.fps.get() as u64) {
                    self.publish_status("");
                }
                if frames >= self.config.max_episode_frames {
                    warn!("episode {episode_index}: max_episode_s reached, stopping with save");
                    self.stop_with_message("max_episode_s reached").await;
                }
            }
            Err(gap) => {
                let message = gap.to_string();
                warn!("episode {episode_index}: {message}, stopping with save");
                self.stop_with_message(&message).await;
            }
        }
    }

    async fn stop_with_message(&mut self, message: &str) {
        let _ = self.stop(true).await;
        self.publish_status(message);
    }
}
