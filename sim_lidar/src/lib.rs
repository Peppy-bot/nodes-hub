//! Sim lidar: a relay from a simulation's sim_lidar_2d_link pairing to the
//! lidar_2d contract surface. Scans forward with their timestamp,
//! capture_index and frame untouched, so consumers place each ray on the
//! capture time; the relay holds each scan's frame to the lidar's frame, which
//! it reads from sensor_mounts' describe_sensors once at its start by its name
//! in its copy, and drops a scan in another frame, logging the first of a run
//! with both frames and what to fix. The relay relays nothing before it knows
//! the lidar's frame. A relay whose instance id names no lidar in its copy, or
//! whose name sensor_mounts gives no lidar's frame, fails its start.

#![forbid(unsafe_code)]

mod node;

pub use node::{leg_died, setup};
