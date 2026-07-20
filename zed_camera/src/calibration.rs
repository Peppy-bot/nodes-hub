//! Per-serial factory calibration, downloaded once from Stereolabs and cached
//! next to the node's working directory. The conf is the authoritative
//! geometry for this exact unit; rectification is only as good as this file.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const CALIBRATION_URL: &str = "https://calib.stereolabs.com/?SN=";

/// The cached calibration file for `serial`, downloading it on first use.
pub fn ensure_calibration(cache_dir: &Path, serial: i32) -> Result<PathBuf, String> {
    let path = cache_dir.join(format!("zed_SN{serial}.conf"));
    if is_valid_conf(&path) {
        return Ok(path);
    }
    fs::create_dir_all(cache_dir).map_err(|e| format!("create {}: {e}", cache_dir.display()))?;

    let url = format!("{CALIBRATION_URL}{serial}");
    let body = ureq::get(&url)
        .call()
        .map_err(|e| format!("download {url}: {e}"))?
        .into_string()
        .map_err(|e| format!("read calibration body: {e}"))?;
    if !looks_like_conf(&body) {
        return Err(format!(
            "calibration server returned no calibration for serial {serial}"
        ));
    }
    fs::write(&path, &body).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

fn is_valid_conf(path: &Path) -> bool {
    fs::read_to_string(path).is_ok_and(|s| looks_like_conf(&s))
}

fn looks_like_conf(body: &str) -> bool {
    body.contains("[LEFT_CAM_") && body.contains("[STEREO]")
}

/// One eye's pinhole intrinsics and OpenCV distortion (k1, k2, p1, p2, k3);
/// the ZED conf carries no tangential terms, so p1/p2 stay zero.
pub struct CamConf {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub p1: f64,
    pub p2: f64,
}

/// Stereo geometry for one resolution, read exactly as the Stereolabs
/// rectification recipe reads it: per-eye intrinsics, the baseline, the
/// resolution-suffixed translation offsets (absent in shipping confs, hence
/// zero), and the (rx, cv, rz) inter-eye rotation vector.
pub struct StereoConf {
    pub left: CamConf,
    pub right: CamConf,
    pub baseline: f64,
    pub ty: f64,
    pub tz: f64,
    pub rx: f64,
    pub cv: f64,
    pub rz: f64,
}

/// The conf section/key suffix for a full eye width (VGA/HD/FHD/2K).
pub fn resolution_key(eye_width: u32) -> &'static str {
    match eye_width {
        2208 => "2K",
        1920 => "FHD",
        672 => "VGA",
        _ => "HD",
    }
}

impl StereoConf {
    /// Parse the calibration for one resolution from a cached conf.
    pub fn load(path: &Path, res_key: &str) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let ini = parse_ini(&text);
        let res = res_key.to_ascii_lowercase();
        let get = |section: &str, key: &str| -> Option<f64> {
            ini.get(&(section.to_string(), key.to_string())).copied()
        };
        let cam = |side: &str| -> Result<CamConf, String> {
            let section = format!("{side}_cam_{res}");
            let required =
                |key: &str| get(&section, key).ok_or_else(|| format!("[{section}] missing {key}"));
            let optional = |key: &str| get(&section, key).unwrap_or(0.0);
            Ok(CamConf {
                fx: required("fx")?,
                fy: required("fy")?,
                cx: required("cx")?,
                cy: required("cy")?,
                k1: optional("k1"),
                k2: optional("k2"),
                k3: optional("k3"),
                p1: optional("p1"),
                p2: optional("p2"),
            })
        };
        let stereo = |key: &str| get("stereo", key).unwrap_or(0.0);
        Ok(Self {
            left: cam("left")?,
            right: cam("right")?,
            baseline: get("stereo", "baseline").ok_or("[STEREO] missing Baseline")?,
            ty: stereo(&format!("ty_{res}")),
            tz: stereo(&format!("tz_{res}")),
            rx: stereo(&format!("rx_{res}")),
            cv: stereo(&format!("cv_{res}")),
            rz: stereo(&format!("rz_{res}")),
        })
    }
}

/// Flatten an INI conf to lowercased (section, key) -> value for numeric keys.
fn parse_ini(text: &str) -> HashMap<(String, String), f64> {
    let mut out = HashMap::new();
    let mut section = String::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = name.trim().to_ascii_lowercase();
        } else if let Some((key, value)) = line.split_once('=')
            && let Ok(value) = value.trim().parse::<f64>()
        {
            out.insert((section.clone(), key.trim().to_ascii_lowercase()), value);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conf_sniffing_rejects_html_and_accepts_ini() {
        assert!(!looks_like_conf("<html>301 Moved Permanently</html>"));
        assert!(looks_like_conf(
            "[LEFT_CAM_HD]\nfx=1\n[STEREO]\nBaseline=63\n"
        ));
    }

    #[test]
    fn parses_resolution_block_and_zeroes_absent_keys() {
        let conf = "[LEFT_CAM_HD]\nfx=772.22\nfy=772.19\ncx=636.985\ncy=361.008\n\
                    k1=-0.03\nk2=-0.004\nk3=0.037\nk4=-0.02\n\
                    [RIGHT_CAM_HD]\nfx=770.8\nfy=770.6\ncx=644.6\ncy=345.6\nk1=-0.03\nk2=-0.004\nk3=0.02\n\
                    [STEREO]\nBaseline=62.902\nTY=0.062\nTZ=-0.0006\nRX_HD=-0.0023\nCV_HD=0.0046\nRZ_HD=-0.0005\n";
        let dir = std::env::temp_dir().join("zed_conf_parse_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("SN.conf");
        fs::write(&path, conf).unwrap();

        let s = StereoConf::load(&path, resolution_key(1280)).unwrap();
        assert_eq!(s.left.fx, 772.22);
        assert_eq!(s.left.cx, 636.985);
        assert_eq!(s.baseline, 62.902);
        // p1/p2 absent -> zero; k4 is ignored by the 5-term model.
        assert_eq!((s.left.p1, s.left.p2), (0.0, 0.0));
        // Rotation is per-resolution; translation suffix is absent -> zero,
        // matching the reference recipe (bare TY/TZ are not read).
        assert_eq!(s.rx, -0.0023);
        assert_eq!((s.ty, s.tz), (0.0, 0.0));
    }
}
