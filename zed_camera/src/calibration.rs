//! Per-serial factory calibration: fetch and parse. The .conf is the
//! authoritative geometry for one exact unit, published per serial at
//! `https://calib.stereolabs.com/?SN=<serial>` and fetched fresh at startup;
//! rectification is only as good as it.

use std::collections::HashMap;
use std::time::Duration;

use crate::resolution::Resolution;

/// Where Stereolabs publishes the per-serial factory calibration.
const CALIBRATION_HOST: &str = "https://calib.stereolabs.com/";
/// Time to establish the connection before giving up.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Ceiling on the whole request, including a server that accepts then stalls
/// on the body, so a startup fetch can never hang the node.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
/// Ceiling on the body. A conf is a few kilobytes; the cap only keeps a
/// misbehaving server from exhausting memory.
const MAX_CONF_BYTES: u64 = 10 * 1024 * 1024;

/// Fetch one unit's factory calibration text by serial, bounded so it cannot
/// hang. The conf carries every resolution; [`StereoConf::from_conf_str`]
/// selects one.
pub fn fetch_conf(serial: i32) -> Result<String, String> {
    fetch_conf_from(CALIBRATION_HOST, serial)
}

/// [`fetch_conf`] against any host. A body that is not valid UTF-8 is an
/// error rather than a lossy decode: the geometry comes from this text, so a
/// corrupted conf must fail the fetch, not reach the parser altered.
fn fetch_conf_from(host: &str, serial: i32) -> Result<String, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_global(Some(REQUEST_TIMEOUT))
        .build()
        .into();
    agent
        .get(host)
        .query("SN", serial.to_string())
        .call()
        .map_err(|e| format!("fetch calibration for serial {serial}: {e}"))?
        .body_mut()
        .with_config()
        .limit(MAX_CONF_BYTES)
        .read_to_string()
        .map_err(|e| format!("read calibration body for serial {serial}: {e}"))
}

/// One eye's pinhole intrinsics.
#[derive(Debug)]
pub struct CamConf {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// Both eyes' lens distortion under the one model the conf writes for the
/// pair, each eye's coefficients in OpenCV's order. A coefficient the conf
/// leaves out is zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lenses {
    /// OpenCV's 5-term radial-tangential model, (k1, k2, p1, p2, k3): the
    /// conf writes no k4.
    RadialTangential { left: [f64; 5], right: [f64; 5] },
    /// OpenCV's equidistant fisheye model, (k1, k2, k3, k4): the conf writes
    /// k4 and no tangential term.
    Fisheye { left: [f64; 4], right: [f64; 4] },
}

/// One eye's distortion as its section writes it, before the pair is
/// checked for one model.
enum Distortion {
    RadialTangential([f64; 5]),
    Fisheye([f64; 4]),
}

/// Stereo geometry for one resolution, read exactly as the Stereolabs
/// rectification recipe reads it: per-eye intrinsics, the lenses'
/// distortion, the baseline, the resolution-suffixed translation offsets
/// (absent in shipping confs, hence zero), and the (rx, cv, rz) inter-eye
/// rotation vector.
#[derive(Debug)]
pub struct StereoConf {
    pub left: CamConf,
    pub right: CamConf,
    pub lenses: Lenses,
    pub baseline: f64,
    pub ty: f64,
    pub tz: f64,
    pub rx: f64,
    pub cv: f64,
    pub rz: f64,
}

impl StereoConf {
    /// Parse the calibration for one resolution from conf text.
    pub fn from_conf_str(text: &str, resolution: Resolution) -> Result<Self, String> {
        let ini = parse_ini(text);
        let res = resolution.conf_key().to_ascii_lowercase();
        let get = |section: &str, key: &str| -> Result<Option<f64>, String> {
            match ini.get(&(section.to_string(), key.to_string())) {
                Some(v) if !v.is_finite() => {
                    Err(format!("[{section}] {key} must be finite, got {v}"))
                }
                v => Ok(v.copied()),
            }
        };
        let cam = |side: &str| -> Result<(CamConf, Distortion), String> {
            let section = format!("{side}_cam_{res}");
            let required = |key: &str| -> Result<f64, String> {
                get(&section, key)?.ok_or_else(|| format!("[{section}] missing {key}"))
            };
            let optional =
                |key: &str| -> Result<f64, String> { Ok(get(&section, key)?.unwrap_or(0.0)) };
            let written = |key: &str| -> Result<bool, String> { Ok(get(&section, key)?.is_some()) };
            let tangential = written("p1")? || written("p2")?;
            if written("k5")? || written("k6")? || (written("k4")? && tangential) {
                return Err(format!(
                    "[{section}] writes a rational lens model (k5 or k6, or k4 with p1 or p2); \
                     this node rectifies the radial-tangential and the fisheye models"
                ));
            }
            let distortion = if written("k4")? {
                Distortion::Fisheye([
                    optional("k1")?,
                    optional("k2")?,
                    optional("k3")?,
                    optional("k4")?,
                ])
            } else {
                Distortion::RadialTangential([
                    optional("k1")?,
                    optional("k2")?,
                    optional("p1")?,
                    optional("p2")?,
                    optional("k3")?,
                ])
            };
            let intrinsics = CamConf {
                fx: required("fx")?,
                fy: required("fy")?,
                cx: required("cx")?,
                cy: required("cy")?,
            };
            Ok((intrinsics, distortion))
        };
        let stereo_required = |key: String| -> Result<f64, String> {
            get("stereo", &key)?.ok_or_else(|| format!("[STEREO] missing {key}"))
        };
        let (left, left_distortion) = cam("left")?;
        let (right, right_distortion) = cam("right")?;
        let lenses = match (left_distortion, right_distortion) {
            (Distortion::RadialTangential(left), Distortion::RadialTangential(right)) => {
                Lenses::RadialTangential { left, right }
            }
            (Distortion::Fisheye(left), Distortion::Fisheye(right)) => {
                Lenses::Fisheye { left, right }
            }
            _ => {
                return Err(format!(
                    "[LEFT_CAM_{0}] and [RIGHT_CAM_{0}] write different lens models (k4 in one \
                     only); a stereo pair rectifies under one",
                    resolution.conf_key()
                ));
            }
        };
        let baseline = stereo_required("baseline".to_string())?;
        if baseline <= 0.0 {
            return Err(format!(
                "[STEREO] baseline must be positive, got {baseline}"
            ));
        }
        Ok(Self {
            left,
            right,
            lenses,
            baseline,
            // Suffixed translation offsets are absent in shipping confs and
            // default to zero, exactly as the reference recipe reads them.
            ty: get("stereo", &format!("ty_{res}"))?.unwrap_or(0.0),
            tz: get("stereo", &format!("tz_{res}"))?.unwrap_or(0.0),
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
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::thread::JoinHandle;

    /// Serves `response` to the first connection on a loopback port. Returns
    /// the host to fetch from and the server, which yields the request line
    /// it received.
    fn serve_once(response: Vec<u8>) -> (String, JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let host = format!("http://{}/", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            // The header block ends at the first bare "\r\n".
            let mut header = String::new();
            while reader.read_line(&mut header).unwrap() > "\r\n".len() {
                header.clear();
            }
            let mut stream = reader.into_inner();
            stream.write_all(&response).unwrap();
            request_line
        });
        (host, server)
    }

    fn http_response(status: &str, body: &[u8]) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(body);
        response
    }

    #[test]
    fn fetch_asks_for_the_serial_and_returns_the_conf_text() {
        let (host, server) = serve_once(http_response("200 OK", b"[STEREO]\nBaseline=62.902\n"));
        let conf = fetch_conf_from(&host, 12345).unwrap();
        assert_eq!(conf, "[STEREO]\nBaseline=62.902\n");
        assert_eq!(server.join().unwrap(), "GET /?SN=12345 HTTP/1.1\r\n");
    }

    #[test]
    fn fetch_fails_on_an_error_status() {
        let (host, server) = serve_once(http_response("404 Not Found", b"unknown serial"));
        let err = fetch_conf_from(&host, 12345).unwrap_err();
        assert_eq!(err, "fetch calibration for serial 12345: http status: 404");
        server.join().unwrap();
    }

    #[test]
    fn fetch_rejects_a_conf_that_is_not_utf8() {
        let (host, server) =
            serve_once(http_response("200 OK", b"[STEREO]\nBaseline=6\xff2.902\n"));
        let err = fetch_conf_from(&host, 12345).unwrap_err();
        assert!(
            err.starts_with("read calibration body for serial 12345: "),
            "got: {err}"
        );
        server.join().unwrap();
    }

    #[test]
    fn rejects_a_conf_missing_its_rotation() {
        let conf = "[LEFT_CAM_HD]\nfx=1\nfy=1\ncx=1\ncy=1\n\
                    [RIGHT_CAM_HD]\nfx=1\nfy=1\ncx=1\ncy=1\n\
                    [STEREO]\nBaseline=62.902\nCV_HD=0.004\nRZ_HD=-0.0005\n";
        assert!(
            StereoConf::from_conf_str(conf, Resolution::Hd720)
                .unwrap_err()
                .contains("missing rx_hd")
        );
    }

    #[test]
    fn rejects_a_body_without_calibration_geometry() {
        assert!(
            StereoConf::from_conf_str("<html>301 Moved Permanently</html>", Resolution::Hd720)
                .unwrap_err()
                .contains("missing fx")
        );
    }

    #[test]
    fn rejects_non_finite_and_non_positive_values() {
        let base = "[LEFT_CAM_HD]\nfx=772.22\nfy=772.19\ncx=636.985\ncy=361.008\n\
                    [RIGHT_CAM_HD]\nfx=770.8\nfy=770.6\ncx=644.6\ncy=345.6\n\
                    [STEREO]\nBaseline=62.902\nRX_HD=-0.0023\nCV_HD=0.0046\nRZ_HD=-0.0005\n";
        let cases = [
            ("Baseline=62.902", "Baseline=NaN", "baseline must be finite"),
            ("RX_HD=-0.0023", "RX_HD=inf", "rx_hd must be finite"),
            ("fx=772.22", "fx=-inf", "fx must be finite"),
            (
                "Baseline=62.902",
                "Baseline=-62.902",
                "baseline must be positive",
            ),
        ];
        for (valid, broken, expected_error) in cases {
            let poisoned = base.replace(valid, broken);
            let err = StereoConf::from_conf_str(&poisoned, Resolution::Hd720).unwrap_err();
            assert!(err.contains(expected_error), "got: {err}");
        }
    }

    /// The HD sections and the stereo block of the ZED Mini SN 10383163's
    /// factory calibration: each eye writes k1 to k4 and no tangential term.
    const FISHEYE_HD: &str = "[LEFT_CAM_HD]\nfx=772.22\nfy=772.19\ncx=636.985\ncy=361.008\n\
        k1=-0.029395\nk2=-0.00488804\nk3=0.0369709\nk4=-0.0227607\n\
        [RIGHT_CAM_HD]\nfx=770.885\nfy=770.635\ncx=644.655\ncy=345.6635\n\
        k1=-0.0325864\nk2=0.0139434\nk3=-0.00241467\nk4=0.00409011\n\
        [STEREO]\nBaseline=62.902\nTY=0.0623366\nTZ=-0.000617955\n\
        RX_HD=-0.00230597\nCV_HD=0.0046253\nRZ_HD=-0.000490389\n";

    #[test]
    fn parses_resolution_block_and_zeroes_absent_keys() {
        let s = StereoConf::from_conf_str(FISHEYE_HD, Resolution::Hd720).unwrap();
        assert_eq!(s.left.fx, 772.22);
        assert_eq!(s.left.cx, 636.985);
        assert_eq!(s.baseline, 62.902);
        // Rotation is per-resolution; translation suffix is absent -> zero,
        // matching the reference recipe (bare TY/TZ are not read).
        assert_eq!(s.rx, -0.00230597);
        assert_eq!((s.ty, s.tz), (0.0, 0.0));
    }

    #[test]
    fn k4_without_a_tangential_term_is_the_fisheye_model() {
        let s = StereoConf::from_conf_str(FISHEYE_HD, Resolution::Hd720).unwrap();
        assert_eq!(
            s.lenses,
            Lenses::Fisheye {
                left: [-0.029395, -0.00488804, 0.0369709, -0.0227607],
                right: [-0.0325864, 0.0139434, -0.00241467, 0.00409011],
            }
        );
    }

    #[test]
    fn no_k4_is_the_radial_tangential_model_in_opencv_order() {
        let conf = "[LEFT_CAM_HD]\nfx=700\nfy=700\ncx=640\ncy=360\nk1=-0.17\nk2=0.02\np1=0.001\np2=-0.002\nk3=0.003\n\
                    [RIGHT_CAM_HD]\nfx=700\nfy=700\ncx=640\ncy=360\nk1=-0.16\n\
                    [STEREO]\nBaseline=120\nRX_HD=0\nCV_HD=0\nRZ_HD=0\n";
        let s = StereoConf::from_conf_str(conf, Resolution::Hd720).unwrap();
        assert_eq!(
            s.lenses,
            Lenses::RadialTangential {
                left: [-0.17, 0.02, 0.001, -0.002, 0.003],
                right: [-0.16, 0.0, 0.0, 0.0, 0.0],
            }
        );
    }

    #[test]
    fn refuses_a_rational_model_and_a_pair_of_two_models() {
        let cases = [
            (
                "k4=-0.0227607\n",
                "k4=-0.0227607\np1=0.001\n",
                "writes a rational lens model",
            ),
            (
                "k4=-0.0227607\n",
                "k4=-0.0227607\nk5=0.01\n",
                "writes a rational lens model",
            ),
            ("k4=0.00409011\n", "", "write different lens models"),
        ];
        for (valid, broken, expected_error) in cases {
            let poisoned = FISHEYE_HD.replacen(valid, broken, 1);
            assert_ne!(poisoned, FISHEYE_HD, "{valid:?} is not in the conf");
            let err = StereoConf::from_conf_str(&poisoned, Resolution::Hd720).unwrap_err();
            assert!(err.contains(expected_error), "got: {err}");
        }
    }
}
