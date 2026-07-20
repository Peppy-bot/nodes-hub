//! Per-serial factory calibration parsing. The .conf is the authoritative
//! geometry for one exact unit (published at
//! `https://calib.stereolabs.com/?SN=<serial>`), provided to the node as a
//! file; rectification is only as good as this file.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// One eye's pinhole intrinsics and OpenCV distortion (k1, k2, p1, p2, k3);
/// the ZED conf carries no tangential terms, so p1/p2 stay zero.
#[derive(Debug)]
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
#[derive(Debug)]
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
    /// Parse the calibration for one resolution from a conf file.
    pub fn load(path: &Path, res_key: &str) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
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
        let stereo_required =
            |key: String| get("stereo", &key).ok_or(format!("[STEREO] missing {key}"));
        Ok(Self {
            left: cam("left")?,
            right: cam("right")?,
            baseline: stereo_required("baseline".to_string())?,
            // Suffixed translation offsets are absent in shipping confs and
            // default to zero, exactly as the reference recipe reads them.
            ty: get("stereo", &format!("ty_{res}")).unwrap_or(0.0),
            tz: get("stereo", &format!("tz_{res}")).unwrap_or(0.0),
            rx: stereo_required(format!("rx_{res}"))?,
            cv: stereo_required(format!("cv_{res}"))?,
            rz: stereo_required(format!("rz_{res}"))?,
        })
    }
}

/// Flatten an INI conf to lowercased (section, key) -> value for numeric keys.
fn parse_ini(text: &str) -> HashMap<(String, String), f64> {
    text.lines()
        .map(str::trim)
        .fold(
            (String::new(), HashMap::new()),
            |(section, mut values), line| {
                if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                    (name.trim().to_ascii_lowercase(), values)
                } else {
                    if let Some((key, value)) = line.split_once('=')
                        && let Ok(value) = value.trim().parse::<f64>()
                    {
                        values.insert((section.clone(), key.trim().to_ascii_lowercase()), value);
                    }
                    (section, values)
                }
            },
        )
        .1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_rejects_a_conf_missing_its_rotation() {
        let conf = "[LEFT_CAM_HD]\nfx=1\nfy=1\ncx=1\ncy=1\n\
                    [RIGHT_CAM_HD]\nfx=1\nfy=1\ncx=1\ncy=1\n\
                    [STEREO]\nBaseline=62.902\nCV_HD=0.004\nRZ_HD=-0.0005\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.conf");
        fs::write(&path, conf).unwrap();
        assert!(
            StereoConf::load(&path, "HD")
                .unwrap_err()
                .contains("missing rx_hd")
        );
    }

    #[test]
    fn load_rejects_a_conf_missing_its_geometry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.conf");
        fs::write(&path, "<html>301 Moved Permanently</html>").unwrap();
        assert!(
            StereoConf::load(&path, "HD")
                .unwrap_err()
                .contains("missing fx")
        );
    }

    #[test]
    fn parses_resolution_block_and_zeroes_absent_keys() {
        let conf = "[LEFT_CAM_HD]\nfx=772.22\nfy=772.19\ncx=636.985\ncy=361.008\n\
                    k1=-0.03\nk2=-0.004\nk3=0.037\nk4=-0.02\n\
                    [RIGHT_CAM_HD]\nfx=770.8\nfy=770.6\ncx=644.6\ncy=345.6\nk1=-0.03\nk2=-0.004\nk3=0.02\n\
                    [STEREO]\nBaseline=62.902\nTY=0.062\nTZ=-0.0006\nRX_HD=-0.0023\nCV_HD=0.0046\nRZ_HD=-0.0005\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("SN.conf");
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
