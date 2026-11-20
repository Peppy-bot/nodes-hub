//! The node's records as room_map:v1 and transforms:v1 carry them, in B4.5's shape: a
//! frame, a timestamp, a position and a unit quaternion; a grid's extent as
//! `[x_min, x_max, y_min, y_max]`, and as navigation_core's map extent; and times between
//! the wire's `SystemTime` and the nanoseconds mapping_core stamps.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use control_core::planar_pose::PlanarPose;
use frame_tree::arithmetic::quaternion_of_yaw;
use mapping_core::GridExtent;
use navigation_core::saved_grid::MapExtent;

/// `duration` in nanoseconds, the longest duration a u64 holds past it.
pub fn duration_ns(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// `time` as nanoseconds since the epoch of its clock.
pub fn ns_of(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |since| {
        i64::try_from(since.as_nanos()).unwrap_or(i64::MAX)
    })
}

/// `ns` nanoseconds since the epoch of the clock, as the wire carries a time.
pub fn time_of(ns: i64) -> SystemTime {
    UNIX_EPOCH + Duration::from_nanos(u64::try_from(ns).unwrap_or(0))
}

/// A planar pose's position, its z 0.
pub fn position_of(pose: PlanarPose) -> [f64; 3] {
    [pose.x(), pose.y(), 0.0]
}

/// A planar pose's orientation, its turn about +Z, as tf2's `setRPY` makes it.
pub fn orientation_of(pose: PlanarPose) -> [f64; 4] {
    quaternion_of_yaw(pose.yaw())
}

/// A grid's extent as room_map gives it: `[x_min, x_max, y_min, y_max]` (m).
pub fn extent_of(extent: &GridExtent) -> [f64; 4] {
    [
        extent.min_x_m,
        extent.min_x_m + extent.width_m,
        extent.min_y_m,
        extent.min_y_m + extent.height_m,
    ]
}

/// A grid's extent as navigation_core reads a map's: the closed rectangle of
/// [`extent_of`].
pub fn map_extent(extent: &GridExtent) -> MapExtent {
    MapExtent::parse(extent_of(extent))
        .expect("a grid's extent: finite edges from its corner, its width and height at least 0")
}
