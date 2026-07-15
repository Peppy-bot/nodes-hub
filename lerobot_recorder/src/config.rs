//! Launch parameters parsed once into a typed [`Config`]; an invalid launch
//! dies at bringup with a precise reason instead of misrecording.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

use lerobot_dataset::VideoCodec;
use peppygen::Parameters;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("fps must be non-zero")]
    ZeroFps,
    #[error("{0} must be non-zero")]
    ZeroField(&'static str),
    #[error("{0} must be a positive duration")]
    NonPositiveDuration(&'static str),
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("video_codec must be \"libx264\" or \"libsvtav1\", got {0:?}")]
    BadCodec(String),
    #[error("storage_backend must be \"local\", \"s3\", or \"r2\", got {0:?}")]
    BadStorageBackend(String),
    #[error("{backend} storage requires storage_bucket to be set")]
    MissingBucket { backend: &'static str },
    #[error("depth_unit_m must be a positive, finite value")]
    BadDepthUnit,
    #[error("camera_keys entry {0:?} is not of the form instance_id=key")]
    BadCameraKeyEntry(String),
    #[error("camera key {0:?} must match [a-z0-9_]+")]
    BadCameraKey(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageBackend {
    Local,
    /// Mirror to an S3-compatible bucket (`s3` or Cloudflare `r2`).
    Bucket(BucketConfig),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketConfig {
    pub kind: &'static str,
    pub bucket: String,
    pub endpoint: Option<String>,
    pub region: String,
    pub prefix: String,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub robot_type: String,
    pub fps: NonZeroU32,
    pub output_root: PathBuf,
    pub dataset_name: String,
    pub default_task: String,
    pub codec: VideoCodec,
    pub camera_keys: HashMap<String, String>,
    pub record_depth: bool,
    pub depth_unit_m: f64,
    pub storage: StorageBackend,
    pub state_staleness: Duration,
    pub camera_start_fresh: Duration,
    pub camera_timeout: Duration,
    pub max_episode_frames: u64,
    pub status_period: Duration,
}

fn positive_duration(name: &'static str, secs: f64) -> Result<Duration, ConfigError> {
    if secs.is_finite() && secs > 0.0 {
        Ok(Duration::from_secs_f64(secs))
    } else {
        Err(ConfigError::NonPositiveDuration(name))
    }
}

fn non_empty(name: &'static str, value: &str) -> Result<String, ConfigError> {
    if value.trim().is_empty() {
        Err(ConfigError::EmptyField(name))
    } else {
        Ok(value.to_string())
    }
}

pub fn valid_camera_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Sanitizes an instance id into a dataset camera key: lowercase, and any run
/// of non `[a-z0-9]` becomes a single `_`.
pub fn sanitize_key(instance_id: &str) -> String {
    let mut out = String::with_capacity(instance_id.len());
    let mut prev_us = false;
    for c in instance_id.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_us = false;
        } else if !prev_us {
            out.push('_');
            prev_us = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "camera".to_string()
    } else {
        trimmed
    }
}

fn parse_camera_keys(map: &str) -> Result<HashMap<String, String>, ConfigError> {
    let mut out = HashMap::new();
    for entry in map.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let (instance, key) = entry
            .split_once('=')
            .ok_or_else(|| ConfigError::BadCameraKeyEntry(entry.to_string()))?;
        let (instance, key) = (instance.trim(), key.trim());
        if instance.is_empty() || key.is_empty() {
            return Err(ConfigError::BadCameraKeyEntry(entry.to_string()));
        }
        if !valid_camera_key(key) {
            return Err(ConfigError::BadCameraKey(key.to_string()));
        }
        out.insert(instance.to_string(), key.to_string());
    }
    Ok(out)
}

fn parse_storage(params: &Parameters) -> Result<StorageBackend, ConfigError> {
    let kind = match params.storage_backend.as_str() {
        "local" => return Ok(StorageBackend::Local),
        "s3" => "s3",
        "r2" => "r2",
        other => return Err(ConfigError::BadStorageBackend(other.to_string())),
    };
    if params.storage_bucket.trim().is_empty() {
        return Err(ConfigError::MissingBucket { backend: kind });
    }
    let endpoint = Some(params.storage_endpoint.trim())
        .filter(|e| !e.is_empty())
        .map(str::to_string);
    Ok(StorageBackend::Bucket(BucketConfig {
        kind,
        bucket: params.storage_bucket.trim().to_string(),
        endpoint,
        region: params.storage_region.trim().to_string(),
        prefix: params.storage_prefix.trim().trim_matches('/').to_string(),
    }))
}

impl Config {
    pub fn parse(params: &Parameters) -> Result<Config, ConfigError> {
        let fps = NonZeroU32::new(params.fps).ok_or(ConfigError::ZeroFps)?;
        if params.max_episode_s == 0 {
            return Err(ConfigError::ZeroField("max_episode_s"));
        }
        if params.status_rate_hz == 0 {
            return Err(ConfigError::ZeroField("status_rate_hz"));
        }
        let codec = match params.video_codec.as_str() {
            "libx264" => VideoCodec::H264Libx264,
            "libsvtav1" => VideoCodec::Av1SvtAv1,
            other => return Err(ConfigError::BadCodec(other.to_string())),
        };
        if !(params.depth_unit_m.is_finite() && params.depth_unit_m > 0.0) {
            return Err(ConfigError::BadDepthUnit);
        }

        Ok(Config {
            robot_type: non_empty("robot_type", &params.robot_type)?,
            fps,
            output_root: PathBuf::from(non_empty("output_root", &params.output_root)?),
            dataset_name: non_empty("dataset_name", &params.dataset_name)?,
            default_task: non_empty("default_task", &params.default_task)?,
            codec,
            camera_keys: parse_camera_keys(&params.camera_keys)?,
            record_depth: params.record_depth,
            depth_unit_m: params.depth_unit_m,
            storage: parse_storage(params)?,
            state_staleness: positive_duration("state_staleness_s", params.state_staleness_s)?,
            camera_start_fresh: positive_duration(
                "camera_start_fresh_s",
                params.camera_start_fresh_s,
            )?,
            camera_timeout: positive_duration("camera_timeout_s", params.camera_timeout_s)?,
            max_episode_frames: params.max_episode_s as u64 * params.fps as u64,
            status_period: Duration::from_secs_f64(1.0 / params.status_rate_hz as f64),
        })
    }

    /// The dataset key for a camera instance: an explicit `camera_keys`
    /// override, else the sanitized instance id.
    pub fn camera_key(&self, instance_id: &str) -> String {
        self.camera_keys
            .get(instance_id)
            .cloned()
            .unwrap_or_else(|| sanitize_key(instance_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Parameters {
        Parameters {
            camera_keys: String::new(),
            camera_start_fresh_s: 0.5,
            camera_timeout_s: 2.0,
            dataset_name: "teleop".into(),
            default_task: "pick".into(),
            depth_unit_m: 0.001,
            fps: 30,
            launcher_id: "l".into(),
            max_episode_s: 1200,
            output_root: "/data".into(),
            record_depth: true,
            robot_type: "bot".into(),
            state_staleness_s: 0.25,
            status_rate_hz: 2,
            storage_backend: "local".into(),
            storage_bucket: String::new(),
            storage_endpoint: String::new(),
            storage_prefix: String::new(),
            storage_region: "auto".into(),
            video_codec: "libx264".into(),
        }
    }

    #[test]
    fn parses_defaults() {
        let c = Config::parse(&params()).unwrap();
        assert_eq!(c.max_episode_frames, 36000);
        assert_eq!(c.storage, StorageBackend::Local);
        assert_eq!(c.status_period, Duration::from_millis(500));
    }

    #[test]
    fn camera_key_override_then_sanitize() {
        let mut p = params();
        p.camera_keys = "front_inst=front".into();
        let c = Config::parse(&p).unwrap();
        assert_eq!(c.camera_key("front_inst"), "front");
        assert_eq!(c.camera_key("Wrist Cam-2"), "wrist_cam_2");
    }

    #[test]
    fn bucket_storage_requires_bucket() {
        let mut p = params();
        p.storage_backend = "r2".into();
        assert_eq!(
            Config::parse(&p).unwrap_err(),
            ConfigError::MissingBucket { backend: "r2" }
        );
        p.storage_bucket = "datasets".into();
        p.storage_endpoint = "https://acct.r2.cloudflarestorage.com".into();
        let c = Config::parse(&p).unwrap();
        assert!(matches!(c.storage, StorageBackend::Bucket(b) if b.kind == "r2"));
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut p = params();
        p.fps = 0;
        assert_eq!(Config::parse(&p).unwrap_err(), ConfigError::ZeroFps);
        let mut p = params();
        p.video_codec = "vp9".into();
        assert_eq!(
            Config::parse(&p).unwrap_err(),
            ConfigError::BadCodec("vp9".into())
        );
        let mut p = params();
        p.depth_unit_m = 0.0;
        assert_eq!(Config::parse(&p).unwrap_err(), ConfigError::BadDepthUnit);
        let mut p = params();
        p.camera_keys = "bad".into();
        assert!(matches!(
            Config::parse(&p).unwrap_err(),
            ConfigError::BadCameraKeyEntry(_)
        ));
    }

    #[test]
    fn sanitize_edge_cases() {
        assert_eq!(sanitize_key("____"), "camera");
        assert_eq!(sanitize_key("rs-front-01"), "rs_front_01");
    }
}
