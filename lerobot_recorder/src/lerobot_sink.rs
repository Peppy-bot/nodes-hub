//! The LeRobot writer on its own blocking thread. The dataset schema is built
//! on the first episode from the discovered [`SourceSchema`] and camera specs;
//! episodes then stream frames in. Encoder stalls surface to the manager as a
//! full frame queue, never as a blocked drain task. Immutable chunk files are
//! forwarded to the storage task as they roll over.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use lerobot_dataset::{
    CameraSpec, DatasetConfig, DatasetWriter, DepthSpec, Frame, PixelFrame, SourceEncoding,
    VideoSettings,
};
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

use crate::config::Config;
use crate::snapshot::{FrameRow, SourceSchema};
use crate::storage::StorageEvent;
use crate::types::CameraEncoding;

/// Geometry and encoding of one camera at its first frame.
#[derive(Debug, Clone)]
pub struct CameraInit {
    /// Dataset key without the `observation.images.` prefix.
    pub key: String,
    pub width: NonZeroU32,
    pub height: NonZeroU32,
    pub encoding: CameraEncoding,
}

pub enum Request {
    Begin {
        task: String,
        schema: Box<SourceSchema>,
        cameras: Vec<CameraInit>,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Frame(Box<FrameRow>),
    End {
        save: bool,
        reply: oneshot::Sender<Result<EndSummary, String>>,
    },
}

#[derive(Debug, Clone)]
pub struct EndSummary {
    pub frames: u64,
}

#[derive(Clone)]
pub struct SinkHandle {
    tx: mpsc::Sender<Request>,
}

pub fn channel(config: &Config) -> (SinkHandle, mpsc::Receiver<Request>) {
    let capacity = (config.fps.get() as usize * 2).max(8);
    let (tx, rx) = mpsc::channel(capacity);
    (SinkHandle { tx }, rx)
}

impl SinkHandle {
    pub async fn begin(
        &self,
        task: String,
        schema: SourceSchema,
        cameras: Vec<CameraInit>,
    ) -> Result<(), String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Request::Begin {
                task,
                schema: Box::new(schema),
                cameras,
                reply,
            })
            .await
            .map_err(|_| "recorder sink is gone".to_string())?;
        rx.await.map_err(|_| "recorder sink is gone".to_string())?
    }

    pub fn try_frame(&self, row: FrameRow) -> Result<(), FrameBackpressure> {
        self.tx
            .try_send(Request::Frame(Box::new(row)))
            .map_err(|_| FrameBackpressure)
    }

    pub async fn end(&self, save: bool) -> Result<EndSummary, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Request::End { save, reply })
            .await
            .map_err(|_| "recorder sink is gone".to_string())?;
        rx.await.map_err(|_| "recorder sink is gone".to_string())?
    }
}

#[derive(Debug)]
pub struct FrameBackpressure;

fn source_encoding(encoding: CameraEncoding) -> SourceEncoding {
    match encoding {
        CameraEncoding::Rgb8 => SourceEncoding::Rgb8,
        CameraEncoding::Bgr8 => SourceEncoding::Bgr8,
        CameraEncoding::Yuyv => SourceEncoding::Yuyv,
        CameraEncoding::Mjpeg => SourceEncoding::Mjpeg,
        CameraEncoding::Z16 => SourceEncoding::Z16,
    }
}

fn dataset_config(
    config: &Config,
    schema: &SourceSchema,
    cameras: &[CameraInit],
) -> Result<DatasetConfig, String> {
    let mut builder = DatasetConfig::builder(config.robot_type.clone(), config.fps)
        .state(schema.state_names.clone())
        .action(schema.action_names.clone())
        .video(VideoSettings {
            codec: config.codec,
            ..VideoSettings::default()
        });
    if schema.has_velocity {
        builder = builder.vector_feature("observation.velocity", schema.velocity_names.clone());
    }
    for camera in cameras {
        let key = format!("observation.images.{}", camera.key);
        if camera.encoding.is_depth() {
            builder = builder.depth_camera(
                key,
                DepthSpec {
                    width: camera.width,
                    height: camera.height,
                    depth_unit_m: config.depth_unit_m,
                    quantization: Default::default(),
                },
            );
        } else {
            builder = builder.camera(
                key,
                CameraSpec {
                    width: camera.width,
                    height: camera.height,
                    source: source_encoding(camera.encoding),
                },
            );
        }
    }
    builder.build().map_err(|e| e.to_string())
}

fn add_row(
    episode: &mut lerobot_dataset::EpisodeWriter<'_>,
    dataset_config: &DatasetConfig,
    row: &FrameRow,
) -> Result<(), String> {
    let state_id = dataset_config.vector_id("observation.state").unwrap();
    let action_id = dataset_config.vector_id("action").unwrap();
    let mut vectors = vec![
        (state_id, row.state.as_slice()),
        (action_id, row.action.as_slice()),
    ];
    if let Some(velocity_id) = dataset_config.vector_id("observation.velocity") {
        vectors.push((velocity_id, row.velocity.as_slice()));
    }

    let mut images = Vec::with_capacity(row.images.len());
    for (camera_id, frame) in dataset_config.camera_ids().zip(&row.images) {
        let spec = dataset_config.camera_spec(camera_id);
        let pixels = match frame.encoding {
            CameraEncoding::Rgb8 => PixelFrame::rgb8(spec.width, spec.height, &frame.bytes),
            CameraEncoding::Bgr8 => PixelFrame::bgr8(spec.width, spec.height, &frame.bytes),
            CameraEncoding::Yuyv => PixelFrame::yuyv(spec.width, spec.height, &frame.bytes),
            CameraEncoding::Mjpeg => PixelFrame::mjpeg(&frame.bytes),
            CameraEncoding::Z16 => PixelFrame::z16(spec.width, spec.height, &frame.bytes),
        }
        .map_err(|e| e.to_string())?;
        images.push((camera_id, pixels));
    }
    episode
        .add_frame(Frame {
            vectors: &vectors,
            images: &images,
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Blocking worker: owns the `DatasetWriter`, one episode at a time. Rolled-over
/// files are forwarded to `storage` for upload; the session is synced at the
/// end.
pub fn run(
    config: Config,
    dataset_dir: PathBuf,
    mut rx: mpsc::Receiver<Request>,
    storage: mpsc::Sender<StorageEvent>,
) {
    let mut writer: Option<DatasetWriter> = None;

    'session: while let Some(request) = rx.blocking_recv() {
        let (task, schema, cameras, reply) = match request {
            Request::Begin {
                task,
                schema,
                cameras,
                reply,
            } => (task, schema, cameras, reply),
            Request::Frame(_) => continue 'session,
            Request::End { reply, .. } => {
                let _ = reply.send(Err("no episode open".to_string()));
                continue 'session;
            }
        };

        if writer.is_none() {
            match create_writer(&config, &dataset_dir, &schema, &cameras) {
                Ok(w) => writer = Some(w),
                Err(e) => {
                    let _ = reply.send(Err(e));
                    continue 'session;
                }
            }
        }
        let dataset = writer.as_mut().expect("created above");
        let dataset_config = dataset.config().clone();
        let mut episode = match dataset.begin_episode(&task) {
            Ok(episode) => episode,
            Err(e) => {
                let _ = reply.send(Err(e.to_string()));
                continue 'session;
            }
        };
        let _ = reply.send(Ok(()));

        let mut poisoned: Option<String> = None;
        loop {
            match rx.blocking_recv() {
                Some(Request::Frame(row)) => {
                    if poisoned.is_none()
                        && let Err(e) = add_row(&mut episode, &dataset_config, &row)
                    {
                        error!("episode frame failed: {e}");
                        poisoned = Some(e);
                    }
                }
                Some(Request::End { save, reply }) => {
                    let result = match (&poisoned, save) {
                        (Some(reason), _) => {
                            episode.abort();
                            Err(reason.clone())
                        }
                        (None, false) => {
                            episode.abort();
                            Ok(EndSummary { frames: 0 })
                        }
                        (None, true) => match episode.end() {
                            Ok(meta) => {
                                for rel in meta.finalized_files {
                                    let _ = storage.blocking_send(StorageEvent::Upload(rel));
                                }
                                Ok(EndSummary {
                                    frames: meta.length,
                                })
                            }
                            Err(e) => Err(e.to_string()),
                        },
                    };
                    let _ = reply.send(result);
                    continue 'session;
                }
                Some(Request::Begin { reply, .. }) => {
                    let _ = reply.send(Err("episode already open".to_string()));
                }
                None => {
                    episode.abort();
                    break 'session;
                }
            }
        }
    }

    if let Some(dataset) = writer {
        match dataset.finalize() {
            Ok(summary) => info!(
                "dataset finalized: {} episodes, {} frames",
                summary.total_episodes, summary.total_frames
            ),
            Err(e) => error!("dataset finalize failed: {e}"),
        }
    }
    let _ = storage.blocking_send(StorageEvent::Finalize);
}

fn create_writer(
    config: &Config,
    dataset_dir: &Path,
    schema: &SourceSchema,
    cameras: &[CameraInit],
) -> Result<DatasetWriter, String> {
    let dataset_config = dataset_config(config, schema, cameras)?;
    DatasetWriter::create(dataset_dir, dataset_config).map_err(|e| e.to_string())
}
