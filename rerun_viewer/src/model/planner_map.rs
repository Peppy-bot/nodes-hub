//! The motion planner's voxel map as occupancy_map publishes it: the occupied cubes that
//! its plans, Cartesian paths and checks keep their clearance from.

use std::time::SystemTime;

use control_core::positive_finite::PositiveFinite;

/// Why an occupancy_map message gives no map to draw.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PlannerMapError {
    #[error("voxel_size_m is {0}; the side of a cube is a finite number above 0")]
    VoxelSize(f64),
    #[error("centers has {0} values, not 3 for each cube")]
    Centers(usize),
    #[error("the center of cube {0} has a coordinate that is not a finite number")]
    NotFinite(usize),
}

/// The planner's voxel map at one instant.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannerMap {
    /// When the planner read the map, on its clock.
    pub at: SystemTime,
    pub voxel_size: PositiveFinite,
    /// The center of each occupied cube in the robot frame.
    pub centers: Vec<[f32; 3]>,
    /// Why the map holds no depth frame; empty while it holds one or the planner reads
    /// no camera.
    pub waiting: String,
}

impl PlannerMap {
    /// The map that the fields of an occupancy_map message give: `centers` holds three
    /// coordinates for each cube.
    pub fn parse(
        at: SystemTime,
        voxel_size_m: f64,
        centers: &[f64],
        waiting: &str,
    ) -> Result<Self, PlannerMapError> {
        let voxel_size = PositiveFinite::parse(voxel_size_m)
            .map_err(|_| PlannerMapError::VoxelSize(voxel_size_m))?;
        let (cubes, rest) = centers.as_chunks::<3>();
        if !rest.is_empty() {
            return Err(PlannerMapError::Centers(centers.len()));
        }
        let centers = cubes
            .iter()
            .enumerate()
            .map(|(index, center)| {
                if center.iter().all(|c| c.is_finite()) {
                    Ok(center.map(|c| c as f32))
                } else {
                    Err(PlannerMapError::NotFinite(index))
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            at,
            voxel_size,
            centers,
            waiting: waiting.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_three_coordinates_are_the_center_of_one_cube() {
        let map = PlannerMap::parse(
            SystemTime::UNIX_EPOCH,
            0.02,
            &[0.31, 0.01, 0.51, -0.25, 0.5, 1.0],
            "",
        )
        .expect("a map");
        assert_eq!(map.voxel_size.get(), 0.02);
        assert_eq!(map.centers, [[0.31, 0.01, 0.51], [-0.25, 0.5, 1.0]]);
        assert_eq!(map.waiting, "");
    }

    #[test]
    fn a_message_that_gives_no_map_is_refused() {
        let at = SystemTime::UNIX_EPOCH;
        assert_eq!(
            PlannerMap::parse(at, 0.0, &[], ""),
            Err(PlannerMapError::VoxelSize(0.0))
        );
        assert_eq!(
            PlannerMap::parse(at, 0.02, &[0.1, 0.2, 0.3, 0.4], ""),
            Err(PlannerMapError::Centers(4))
        );
        assert_eq!(
            PlannerMap::parse(at, 0.02, &[0.1, 0.2, 0.3, 0.4, f64::INFINITY, 0.6], ""),
            Err(PlannerMapError::NotFinite(1))
        );
    }
}
