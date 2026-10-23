//! The launcher's limits the planner's unit tests share.

use crate::types::{ARM_DOF, JointVec};

/// The launcher-default per-joint acceleration limits (peppy.json5).
pub(crate) const A_MAX: JointVec = [5.0; ARM_DOF];
