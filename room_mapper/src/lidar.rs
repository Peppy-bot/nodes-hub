//! The lidar's scans: lidar_2d's scan_stream read into camera_core's `LaserScan`, each ray
//! that ends within the padded footprint left out, as room_map:v1 states, and the scan
//! measured once the odometry encloses its sweep: where the lidar stood on the floor of
//! odom at its first and its last ray, and where the base stood at its timestamp.
//!
//! The lidar is fixed to the base, so its place in base_footprint, the mount, holds for the
//! node's life, and a ray's end in base_footprint is the mount applied to the ray's end in
//! the lidar's frame, wherever the base stood at the ray's time.

use camera_core::laser_scan::{LaserScan, LaserScanError, Sweep, UNMEASURED_MARK_M};
use control_core::planar_pose::PlanarPose;
use control_core::time::nanoseconds;
use frame_tree::Transform;
use navigation_core::PaddedFootprint;

use crate::geometry::on_floor;
use crate::odometry::{Odometry, OdometryStart, Span};

/// A scan's place in the order the lidar's scans arrive at the node, counted from 1, or, for
/// a call, the number of scans arrived when it was made: a scan that arrives after the call
/// holds a greater one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Arrival(pub u64);

/// A scan of lidar_2d's scan_stream as it arrived.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrivedScan {
    pub arrival: Arrival,
    /// The time (ns) of ray 0, on the clock of transforms.
    pub stamp_ns: i64,
    /// The lidar's frame, which the scan's header names.
    pub frame: String,
    pub min_angle_rad: f32,
    pub angle_increment_rad: f32,
    pub time_increment_s: f32,
    pub min_range_m: f32,
    pub max_range_m: f32,
    pub ranges: Vec<f32>,
}

/// A scan the odometry places: its rays, those that end within the padded footprint
/// marked unmeasured, where the lidar stood on the floor of odom at its first and its last
/// ray, and where the base stood at its timestamp, all in the odometry of `start`.
#[derive(Debug, Clone)]
pub struct MeasuredScan {
    pub arrival: Arrival,
    pub stamp_ns: i64,
    pub start: OdometryStart,
    pub laser: LaserScan,
    pub sensor: Sweep<PlanarPose>,
    pub base: PlanarPose,
}

/// What the odometry makes of a scan.
#[derive(Debug, Clone)]
pub enum Measurement {
    Measured(MeasuredScan),
    /// The odometry may yet enclose the scan's sweep.
    Waiting,
    /// No odometry encloses the scan's sweep, and none will.
    Never,
}

impl ArrivedScan {
    /// The scan with each ray that ends within `footprint`, placed in base_footprint by
    /// `mount`, marked unmeasured, so the mapping and the localization read nothing of it.
    pub fn laser_scan(
        &self,
        mount: &Transform,
        footprint: &PaddedFootprint,
    ) -> Result<LaserScan, LaserScanError> {
        let ranges: Vec<f32> = self
            .ranges
            .iter()
            .enumerate()
            .map(|(index, &range)| {
                let measured = range >= self.min_range_m && range <= self.max_range_m;
                if !measured {
                    return range;
                }
                let angle = f64::from(self.min_angle_rad)
                    + index as f64 * f64::from(self.angle_increment_rad);
                let end = [
                    f64::from(range) * angle.cos(),
                    f64::from(range) * angle.sin(),
                    0.0,
                ];
                let [x, y, _] = mount.apply(end);
                if footprint.holds([x, y]) {
                    UNMEASURED_MARK_M
                } else {
                    range
                }
            })
            .collect();
        LaserScan::parse(
            self.min_angle_rad,
            self.angle_increment_rad,
            self.time_increment_s,
            self.min_range_m,
            self.max_range_m,
            &ranges,
        )
    }

    /// The scan measured in `odometry`, its rays as [`ArrivedScan::laser_scan`] gives them:
    /// at the time of ray 0, the scan's timestamp, and of its last ray, `laser`'s sweep
    /// after it, before it for a lidar that sweeps clockwise.
    pub fn measure(&self, laser: LaserScan, mount: &Transform, odometry: &Odometry) -> Measurement {
        let last_ray = self.stamp_ns + nanoseconds(laser.sweep_s());
        match odometry.span(&[self.stamp_ns, last_ray]) {
            Span::Waiting => Measurement::Waiting,
            Span::Never => Measurement::Never,
            Span::Enclosed { start, poses } => {
                let [base_first, base_last] = [poses[0], poses[1]];
                Measurement::Measured(MeasuredScan {
                    arrival: self.arrival,
                    stamp_ns: self.stamp_ns,
                    start,
                    laser,
                    sensor: Sweep {
                        first_ray: on_floor(&base_first.compose(mount)),
                        last_ray: on_floor(&base_last.compose(mount)),
                    },
                    base: on_floor(&base_first),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odometry::Measurement as Measured;
    use control_core::footprint::Footprint;
    use control_core::non_negative_finite::NonNegativeFinite;
    use frame_tree::arithmetic::quaternion_of_yaw;

    /// A lidar 0.1 m ahead of base_footprint and 0.3 m up, level and facing forward.
    fn mount() -> Transform {
        Transform::new([0.1, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]).expect("a transform")
    }

    /// A base 0.6 m square about its center, padded by 0.05 m.
    fn footprint() -> PaddedFootprint {
        let outline = Footprint::parse(&[[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]])
            .expect("an outline");
        PaddedFootprint::new(&outline, NonNegativeFinite::parse(0.05).expect("a padding"))
            .expect("a padded footprint")
    }

    /// A scan of four rays a quarter turn apart from straight ahead, each of `ranges`.
    fn scan(time_increment_s: f32, ranges: [f32; 4]) -> ArrivedScan {
        ArrivedScan {
            arrival: Arrival(1),
            stamp_ns: 1_000_000_000,
            frame: "lidar_frame".into(),
            min_angle_rad: 0.0,
            angle_increment_rad: std::f32::consts::FRAC_PI_2,
            time_increment_s,
            min_range_m: 0.05,
            max_range_m: 40.0,
            ranges: ranges.to_vec(),
        }
    }

    // From the lidar 0.1 m ahead of the center, a ray straight ahead ends past the padded
    // footprint's front edge (0.35 m) at 0.3 m and within it at 0.2 m; a ray behind ends
    // within it at 0.3 m and past it at 0.5 m. The marks are kept as they are.
    #[test]
    fn a_ray_that_ends_within_the_padded_footprint_is_marked_unmeasured() {
        let still = Transform::new([0.0; 3], [0.0, 0.0, 0.0, 1.0]).expect("a transform");
        let hits = |arrived: &ArrivedScan| {
            arrived
                .laser_scan(&mount(), &footprint())
                .expect("a scan")
                .placed_hits(&Sweep {
                    first_ray: still,
                    last_ray: still,
                })
                .len()
        };
        assert_eq!(hits(&scan(0.0, [0.3, 1.0, 0.5, 10000.0])), 3, "three hits");
        assert_eq!(
            hits(&scan(0.0, [0.2, 1.0, 0.3, 0.0])),
            1,
            "the side ray alone hits"
        );
    }

    #[test]
    fn a_scan_is_measured_once_the_odometry_encloses_its_sweep() {
        let mut odometry = Odometry::default();
        let start = OdometryStart(7);
        let at = |at_ns: i64, x: f64, yaw: f64| Measured {
            at_ns,
            odom_from_base: Transform::new([x, 0.0, 0.0], quaternion_of_yaw(yaw))
                .expect("a transform"),
        };
        let arrived = scan(-0.01, [1.0; 4]);
        let laser = || arrived.laser_scan(&mount(), &footprint()).expect("a scan");
        odometry.take(start, at(960_000_000, 0.0, 0.0));
        assert!(matches!(
            arrived.measure(laser(), &mount(), &odometry),
            Measurement::Waiting
        ));
        odometry.take(start, at(1_010_000_000, 1.0, 0.0));
        let Measurement::Measured(measured) = arrived.measure(laser(), &mount(), &odometry) else {
            panic!("the odometry encloses 0.97 s to 1.0 s");
        };
        assert_eq!(measured.start, start);
        // The base moves 20 m/s; the f32 time increment holds the last ray's time to a
        // few nanoseconds, so its place to well under a micrometer.
        let ratio = |at_s: f64| (at_s - 0.96) / 0.05;
        assert!((measured.base.x() - ratio(1.0)).abs() < 1e-9);
        assert!((measured.sensor.first_ray.x() - (ratio(1.0) + 0.1)).abs() < 1e-9);
        assert!(
            (measured.sensor.last_ray.x() - (ratio(0.97) + 0.1)).abs() < 1e-6,
            "the last ray of the clockwise sweep comes 0.03 s before the timestamp: {:?}",
            measured.sensor.last_ray
        );
    }
}
