//! The independent restrictions that limit a step, one module per mechanism.
//!
//! Each limiter is constructed per tick with exactly the data it reads and is
//! a pure function of the proposed [`Step`], returning an [`Allowance`]. None
//! of them sees another's output and none of them touches the collision model,
//! so they may be evaluated in any order and combined by keeping the most
//! restrictive fraction per DOF.
//!
//! A restriction that cannot be written as a per-DOF fraction does not belong
//! here. The closing-velocity barrier is the one such case in the governor: it
//! is a directional projection, and it lives in [`super::barrier`].

pub(in crate::governor) mod allowance;
pub(in crate::governor) mod dof_speed;
pub(in crate::governor) mod ee_speed;
pub(in crate::governor) mod measured_tripwire;

pub(in crate::governor) use allowance::{Allowance, Limits};
pub(in crate::governor) use dof_speed::DofSpeed;
pub(in crate::governor) use ee_speed::EeSpeed;
pub(in crate::governor) use measured_tripwire::{MeasuredTripwire, Tripwire};

use super::Step;

/// One independent restriction on a step.
pub(in crate::governor) trait Limiter {
    /// Reported as the binding mechanism when this limiter is the tightest.
    fn name(&self) -> &'static str;

    fn allow(&self, step: &Step) -> Allowance;
}
