//! Feedforward dynamics for an SRS arm control loop: gravity and Coriolis +
//! centripetal torques. Both operate on the world-frame FK accessors of
//! [`crate::fk::ForwardKinematics`] (gravity points along world `-z`) and are
//! verified against KDL reference values.
//!
//! Friction is intentionally not modeled here: it needs none of the rigid-body
//! model (it is a per-joint actuator quantity, a pure function of joint
//! velocity), so it lives in the consumer's control layer rather than in this
//! model crate.

pub mod coriolis;
pub mod gravity;
