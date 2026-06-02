//! Thin peppy node wrapping the `srs_model` library: implements the
//! `forward_kinematics`, `srs_inverse_kinematics`, and `gravity_coriolis_compensation` interfaces
//! (interfaces_hub). All the math lives in the library; this file loads the model
//! from configuration and marshals requests/responses.
//!
//! The interfaces are DOF-generic (joint arrays are unspecified length on the
//! wire, i.e. `Vec<f64>`), so each handler converts to the library's fixed
//! [`JointVec`] at the boundary via [`joints`] and rejects a wrong-DOF request.
//!
//! Friction is deliberately not implemented here: it needs none of the rigid-body
//! model (a pure per-joint velocity formula), so it belongs in the consumer's
//! control layer.
//!
//! One task per service, mirroring the one-task-per-service idiom used across
//! nodes_hub. FK-using tasks each own a private [`ForwardKinematics`] behind a
//! `Mutex` (posing needs `&mut`, and the handler must be `Fn`); the guard is taken
//! and dropped inside the synchronous handler, so it never crosses an await.

use std::sync::{Arc, Mutex};

use peppygen::exposed_services::gravity_coriolis_compensation::v1::{
    get_compensation, get_coriolis, get_gravity,
};
use peppygen::exposed_services::forward_kinematics::v1::get_fk;
use peppygen::exposed_services::srs_inverse_kinematics::v1::get_ik;
use peppygen::{NodeBuilder, NodeRunner, Parameters, Result};
use tracing::{error, info};

use srs_model::dynamics::{coriolis, gravity};
use srs_model::fk::ForwardKinematics;
use srs_model::ik::{self, ArmAnglePolicy};
use srs_model::model::ArmModel;
use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use srs_model::{ARM_DOF, JointVec};

/// The robot description, shared by every service task. Each FK-using task builds
/// its own [`ForwardKinematics`] from `urdf`; the [`ArmModel`] is read-only.
struct Spec {
    urdf: String,
    base_link: String,
    tip_link: String,
    model: ArmModel,
}

impl Spec {
    fn forward_kinematics(&self) -> ForwardKinematics {
        ForwardKinematics::from_urdf(&self.urdf, &self.base_link, &self.tip_link)
            .expect("URDF already validated at startup")
    }
}

/// Convert a wire joint vector (unspecified length, per the DOF-generic
/// interfaces) into the fixed `[f64; ARM_DOF]` the library uses, rejecting a
/// request whose joint count does not match this model.
fn joints(v: &[f64]) -> std::result::Result<JointVec, String> {
    JointVec::try_from(v).map_err(|_| format!("expected {ARM_DOF} joint values, got {}", v.len()))
}

/// Map a request-validation message to a service error, for the dynamics services
/// whose responses carry no status field (the IK solve reports via `success`).
fn bad_request(msg: String) -> peppygen::Error {
    peppygen::Error::Io(std::io::Error::other(msg))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();

    NodeBuilder::new().run(|params: Parameters, node_runner| async move {
        let urdf = std::fs::read_to_string(&params.urdf_path)
            .unwrap_or_else(|e| panic!("read urdf_path '{}': {e}", params.urdf_path));

        // Build the model once up front: this validates the chain is a clean
        // 7-DOF SRS arm (else `Err`) and fixes the world<->base mount transform.
        let model = ArmModel::from_urdf(&urdf, &params.base_link, &params.tip_link)
            .unwrap_or_else(|e| {
                panic!("load SRS model ({} -> {}): {e}", params.base_link, params.tip_link)
            });

        // Log the resolved world->base mount so a "bare arm" URDF (one missing the
        // tree from the world root down to base_link) is caught here, where it
        // would otherwise silently mis-orient gravity. An identity translation on
        // a robot whose arm is mounted off the origin is the tell-tale sign.
        let mount = model.base_from_world.inverse();
        info!(
            "srs_model loaded {} -> {}: arm base at world {:?} (verify this matches the mounting)",
            params.base_link, params.tip_link, mount.translation.vector,
        );

        let spec = Arc::new(Spec {
            urdf,
            base_link: params.base_link,
            tip_link: params.tip_link,
            model,
        });

        spawn_get_compensation(node_runner.clone(), spec.clone());
        spawn_get_gravity(node_runner.clone(), spec.clone());
        spawn_get_coriolis(node_runner.clone(), spec.clone());
        spawn_get_fk(node_runner.clone(), spec.clone());
        spawn_get_ik(node_runner.clone(), spec);

        Ok(())
    })
}

/// gravity + coriolis (and their sum) in one round trip, for the per-tick control loop.
fn spawn_get_compensation(runner: Arc<NodeRunner>, spec: Arc<Spec>) {
    tokio::spawn(async move {
        let fk = Mutex::new(spec.forward_kinematics());
        loop {
            let result = get_compensation::handle_next_request(&runner, |req| {
                let q = joints(&req.data.joint_positions).map_err(bad_request)?;
                let qd = joints(&req.data.joint_velocities).map_err(bad_request)?;
                let mut fk = fk.lock().unwrap_or_else(|e| e.into_inner());
                let posed = fk.at(&q);
                let gravity = gravity::torques(&posed);
                let coriolis = coriolis::torques(&posed, &qd);
                let total: JointVec = std::array::from_fn(|i| gravity[i] + coriolis[i]);
                Ok(get_compensation::Response::new(
                    gravity.to_vec(),
                    coriolis.to_vec(),
                    total.to_vec(),
                ))
            })
            .await;
            if let Err(e) = result {
                error!("get_compensation: {e}");
            }
        }
    });
}

fn spawn_get_gravity(runner: Arc<NodeRunner>, spec: Arc<Spec>) {
    tokio::spawn(async move {
        let fk = Mutex::new(spec.forward_kinematics());
        loop {
            let result = get_gravity::handle_next_request(&runner, |req| {
                let q = joints(&req.data.joint_positions).map_err(bad_request)?;
                let mut fk = fk.lock().unwrap_or_else(|e| e.into_inner());
                let torques = gravity::torques(&fk.at(&q));
                Ok(get_gravity::Response::new(torques.to_vec()))
            })
            .await;
            if let Err(e) = result {
                error!("get_gravity: {e}");
            }
        }
    });
}

fn spawn_get_coriolis(runner: Arc<NodeRunner>, spec: Arc<Spec>) {
    tokio::spawn(async move {
        let fk = Mutex::new(spec.forward_kinematics());
        loop {
            let result = get_coriolis::handle_next_request(&runner, |req| {
                let q = joints(&req.data.joint_positions).map_err(bad_request)?;
                let qd = joints(&req.data.joint_velocities).map_err(bad_request)?;
                let mut fk = fk.lock().unwrap_or_else(|e| e.into_inner());
                let torques = coriolis::torques(&fk.at(&q), &qd);
                Ok(get_coriolis::Response::new(torques.to_vec()))
            })
            .await;
            if let Err(e) = result {
                error!("get_coriolis: {e}");
            }
        }
    });
}

/// FK in the world frame: pose the chain, take the EE pose (base frame), convert
/// to world via the mount transform.
fn spawn_get_fk(runner: Arc<NodeRunner>, spec: Arc<Spec>) {
    tokio::spawn(async move {
        let fk = Mutex::new(spec.forward_kinematics());
        loop {
            let result = get_fk::handle_next_request(&runner, |req| {
                let q = joints(&req.data.joint_positions).map_err(bad_request)?;
                let mut fk = fk.lock().unwrap_or_else(|e| e.into_inner());
                let ee_world = spec.model.world_pose(&fk.at(&q).ee_pose());
                let p = ee_world.translation.vector;
                let r = ee_world.rotation;
                Ok(get_fk::Response::new(
                    [p.x, p.y, p.z],
                    [r.i, r.j, r.k, r.w],
                ))
            })
            .await;
            if let Err(e) = result {
                error!("get_fk: {e}");
            }
        }
    });
}

/// IK from a world-frame target: convert the target into the arm base frame the
/// solver works in, solve, report the joint solution (or success=false).
fn spawn_get_ik(runner: Arc<NodeRunner>, spec: Arc<Spec>) {
    tokio::spawn(async move {
        loop {
            let result = get_ik::handle_next_request(&runner, |req| {
                let seed = match joints(&req.data.seed) {
                    Ok(s) => s,
                    Err(msg) => return Ok(get_ik::Response::new(false, Vec::new(), 0.0, msg)),
                };
                let policy = match parse_policy(&req.data.arm_angle_policy) {
                    Ok(make) => make(req.data.arm_angle),
                    Err(msg) => return Ok(get_ik::Response::new(false, Vec::new(), 0.0, msg)),
                };

                let [x, y, z] = req.data.target_position;
                let [qx, qy, qz, qw] = req.data.target_orientation;
                let world_target = Isometry3::from_parts(
                    Translation3::new(x, y, z),
                    UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
                );
                let base_target = spec.model.base_pose(&world_target);
                let r_d = base_target.rotation.to_rotation_matrix();
                let p_d: Vector3<f64> = base_target.translation.vector;

                let response = match ik::solve(&spec.model, &r_d, &p_d, policy, &seed) {
                    Some(sol) => {
                        get_ik::Response::new(true, sol.q.to_vec(), sol.arm_angle, "ok".to_string())
                    }
                    None => get_ik::Response::new(
                        false,
                        Vec::new(),
                        0.0,
                        "no in-limits IK solution for target".to_string(),
                    ),
                };
                Ok(response)
            })
            .await;
            if let Err(e) = result {
                error!("get_ik: {e}");
            }
        }
    });
}

/// Parse the `arm_angle_policy` request field into a constructor taking the
/// request's `arm_angle` (used only by `fixed`).
fn parse_policy(s: &str) -> std::result::Result<fn(f64) -> ArmAnglePolicy, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "from_seed" => Ok((|_| ArmAnglePolicy::FromSeed) as fn(f64) -> ArmAnglePolicy),
        "fixed" => Ok(ArmAnglePolicy::Fixed as fn(f64) -> ArmAnglePolicy),
        other => Err(format!(
            "arm_angle_policy must be 'from_seed' or 'fixed', got '{other}'"
        )),
    }
}
