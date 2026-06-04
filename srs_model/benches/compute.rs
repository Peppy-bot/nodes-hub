//! Microbenchmarks for the per-request compute cost of the service handlers
//! (pure math, excluding the peppy IPC round trip). Run with `cargo bench`.
//!
//! Each bench mirrors what the corresponding service handler does: the `get_fk`
//! / `get_*` dynamics benches re-pose the FK chain at the requested configuration
//! (as the handlers do, once per request) and evaluate the term; the `get_ik`
//! benches run the closed-form solver. The fixture is the bundled OpenArm V1.0
//! (left arm).
//!
//! IK has two cost regimes:
//!   - `FromSeed` always computes the exact feasible-arm-angle intervals (the
//!     "cold"/search path); the seed only selects which feasible angle is used,
//!     so a related vs decorrelated seed costs the same (both measured below).
//!   - `Fixed(psi)` skips the interval search (single placement), so it is the
//!     cheaper path.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use srs_model::{coriolis, gravity};
use srs_model::fk::ForwardKinematics;
use srs_model::ik::{self, ArmAnglePolicy};
use srs_model::model::ArmModel;
use srs_model::JointVec;

const URDF: &str = include_str!("../tests/fixtures/openarm_v10.urdf");
const BASE: &str = "openarm_left_link0";

fn fixture() -> (ForwardKinematics, ArmModel) {
    (
        ForwardKinematics::from_urdf(URDF, BASE).expect("fk"),
        ArmModel::from_urdf(URDF, BASE).expect("model"),
    )
}

fn benchmarks(c: &mut Criterion) {
    let (mut fk, model) = fixture();
    // A non-singular configuration and a non-trivial velocity.
    let q: JointVec = [0.3, -0.4, 0.2, 0.8, -0.5, 0.3, 0.6];
    let qd: JointVec = [0.4, -0.3, 0.5, -0.2, 0.6, -0.1, 0.3];

    // IK target: the FK of `q`, so it is reachable. `q` is a "warm" seed (it is
    // the solution); `cold_seed` is decorrelated, exercising the same interval
    // search from a different starting angle. `psi` is `q`'s (feasible) arm angle.
    let target = fk.at(&q).ee_pose();
    let r_d = target.rotation.to_rotation_matrix();
    let p_d = target.translation.vector;
    let cold_seed: JointVec = [0.0, 0.2, -0.2, 1.2, 0.3, -0.3, 0.4];
    let psi = ik::arm_angle_of(&model, &q);

    c.bench_function("get_fk", |b| {
        b.iter(|| black_box(fk.at(black_box(&q)).ee_pose()))
    });

    c.bench_function("get_gravity", |b| {
        b.iter(|| black_box(gravity::torques(&fk.at(black_box(&q)))))
    });

    c.bench_function("get_coriolis", |b| {
        b.iter(|| black_box(coriolis::torques(&fk.at(black_box(&q)), black_box(&qd))))
    });

    // The combined service the arm calls every control tick: gravity + Coriolis.
    c.bench_function("get_compensation", |b| {
        b.iter(|| {
            let posed = fk.at(black_box(&q));
            let g = gravity::torques(&posed);
            let co = coriolis::torques(&posed, black_box(&qd));
            black_box((g, co))
        })
    });

    // IK, FromSeed with a warm seed (the solution itself).
    c.bench_function("get_ik_from_seed_warm", |b| {
        b.iter(|| {
            black_box(ik::solve(
                &model,
                black_box(&r_d),
                black_box(&p_d),
                ArmAnglePolicy::FromSeed,
                black_box(&q),
            ))
        })
    });

    // IK, FromSeed with a decorrelated ("cold") seed: same interval search.
    c.bench_function("get_ik_from_seed_cold", |b| {
        b.iter(|| {
            black_box(ik::solve(
                &model,
                black_box(&r_d),
                black_box(&p_d),
                ArmAnglePolicy::FromSeed,
                black_box(&cold_seed),
            ))
        })
    });

    // IK, Fixed arm angle: skips the feasible-interval search (cheaper path).
    c.bench_function("get_ik_fixed", |b| {
        b.iter(|| {
            black_box(ik::solve(
                &model,
                black_box(&r_d),
                black_box(&p_d),
                ArmAnglePolicy::Fixed(black_box(psi)),
                black_box(&q),
            ))
        })
    });
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
