//! Optional object-storage mirror. The LeRobot writer keeps the canonical
//! dataset on local disk; when a bucket backend is configured, this task
//! uploads immutable chunk files as they roll over and syncs the metadata plus
//! any still-open chunks at session end. Uploading only immutable files avoids
//! re-sending multi-GB parquet/mp4 that are still being appended.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use object_store::{
    ObjectStore, ObjectStoreExt, PutPayload, aws::AmazonS3Builder, path::Path as ObjPath,
};
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::{BucketConfig, StorageBackend};

pub enum StorageEvent {
    /// A root-relative file became immutable; upload it.
    Upload(PathBuf),
    /// Session finished; upload metadata and any not-yet-mirrored files.
    Finalize,
}

pub fn channel() -> (mpsc::Sender<StorageEvent>, mpsc::Receiver<StorageEvent>) {
    mpsc::channel(256)
}

/// Runs the mirror. For [`StorageBackend::Local`] it simply drains events (the
/// dataset already lives on disk).
pub async fn run(
    backend: StorageBackend,
    session_dir: PathBuf,
    dataset_name: String,
    mut rx: mpsc::Receiver<StorageEvent>,
    token: CancellationToken,
) {
    let mirror = match backend {
        StorageBackend::Local => None,
        StorageBackend::Bucket(cfg) => match build_store(&cfg) {
            Ok(store) => Some(Mirror {
                store: Arc::from(store),
                dataset_dir: session_dir.join(&dataset_name),
                prefix: cfg.prefix,
            }),
            Err(e) => {
                warn!("storage mirror disabled: {e}");
                None
            }
        },
    };

    // Immutable chunks already mirrored on rollover; skipped by the final sync.
    let mut uploaded: HashSet<PathBuf> = HashSet::new();
    loop {
        let event = tokio::select! {
            _ = token.cancelled() => break,
            event = rx.recv() => match event {
                Some(event) => event,
                None => break,
            },
        };
        let Some(mirror) = &mirror else { continue };
        match event {
            StorageEvent::Upload(rel) => {
                if mirror.upload_relative(&rel).await {
                    uploaded.insert(rel);
                }
            }
            StorageEvent::Finalize => mirror.sync_all(&uploaded).await,
        }
    }
}

struct Mirror {
    store: Arc<dyn ObjectStore>,
    dataset_dir: PathBuf,
    prefix: String,
}

impl Mirror {
    /// `rel` is relative to the dataset dir (as reported by the writer).
    /// Returns whether the upload succeeded.
    async fn upload_relative(&self, rel: &Path) -> bool {
        let local = self.dataset_dir.join(rel);
        self.upload_file(&local, &rel.to_string_lossy()).await
    }

    /// Returns whether the file was put successfully.
    async fn upload_file(&self, local: &Path, key_suffix: &str) -> bool {
        let bytes = match tokio::fs::read(local).await {
            Ok(b) => b,
            Err(e) => {
                warn!("mirror: reading {}: {e}", local.display());
                return false;
            }
        };
        let key = if self.prefix.is_empty() {
            key_suffix.to_string()
        } else {
            format!("{}/{key_suffix}", self.prefix)
        };
        let path = match ObjPath::parse(&key) {
            Ok(p) => p,
            Err(e) => {
                warn!("mirror: bad object key {key:?}: {e}");
                return false;
            }
        };
        if let Err(e) = self.store.put(&path, PutPayload::from(bytes)).await {
            warn!("mirror: uploading {key}: {e}");
            return false;
        }
        true
    }

    /// Uploads the files the rollover path did not already mirror: rewritten
    /// metadata, still-open chunks, and any immutable chunk whose earlier
    /// upload failed. Chunks already mirrored (in `uploaded`) are skipped to
    /// avoid re-sending multi-GB files.
    async fn sync_all(&self, uploaded: &HashSet<PathBuf>) {
        let mut count = 0usize;
        let mut skipped = 0usize;
        let mut stack = vec![self.dataset_dir.clone()];
        while let Some(dir) = stack.pop() {
            let mut entries = match tokio::fs::read_dir(&dir).await {
                Ok(e) => e,
                Err(e) => {
                    warn!("mirror: reading dir {}: {e}", dir.display());
                    continue;
                }
            };
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(rel) = path.strip_prefix(&self.dataset_dir) {
                    if uploaded.contains(rel) {
                        skipped += 1;
                        continue;
                    }
                    self.upload_file(&path, &rel.to_string_lossy()).await;
                    count += 1;
                }
            }
        }
        info!("mirror: session sync complete, {count} uploaded, {skipped} already mirrored");
    }
}

fn build_store(cfg: &BucketConfig) -> Result<Box<dyn ObjectStore>, String> {
    // Credentials come from the standard AWS env vars
    // (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
    let mut builder = AmazonS3Builder::from_env()
        .with_bucket_name(&cfg.bucket)
        .with_region(&cfg.region);
    if let Some(endpoint) = &cfg.endpoint {
        // R2 and other S3-compatibles use a custom endpoint with path-style
        // addressing.
        builder = builder
            .with_endpoint(endpoint)
            .with_virtual_hosted_style_request(false)
            .with_allow_http(false);
    }
    builder
        .build()
        .map(|s| Box::new(s) as Box<dyn ObjectStore>)
        .map_err(|e| e.to_string())
}
