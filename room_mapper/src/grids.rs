//! map_grids: the loaded map's grid, published when a map is loaded or finished, and while
//! mapping runs at most once per grid period once the grid changes; none of a new map
//! before its first scan.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use control_core::sync::lock_ignoring_poison;
use mapping_core::OccupancyGrid;
use peppygen::NodeRunner;
use peppygen::emitted_topics::room_map::map_grids;
use tokio::sync::watch;
use tracing::warn;

use crate::inputs::Inputs;
use crate::messages::{extent_of, time_of};
use crate::rooms::members::MAP_FRAME;
use crate::rooms::state::RoomState;

/// Runs the grid's publisher until the changes of the loaded map end: one grid at each
/// change, and one each grid period while mapping runs and its scans changed.
pub async fn run(
    runner: Arc<NodeRunner>,
    room: Arc<Mutex<RoomState>>,
    inputs: Arc<Inputs>,
    mut changes: watch::Receiver<()>,
    period: Duration,
) -> peppygen::Result<()> {
    let publisher = map_grids::declare_publisher(&runner).await?;
    let period_ns = u64::try_from(period.as_nanos()).unwrap_or(u64::MAX);
    let mut published_scans: Option<u32> = None;
    loop {
        let next = inputs.clock.now_ns() + period_ns;
        let changed = tokio::select! {
            changed = changes.changed() => {
                if changed.is_err() {
                    return Ok(());
                }
                true
            }
            () = inputs.clock.until(next) => false,
        };
        let (grid, state) = tokio::task::block_in_place(|| {
            let room = lock_ignoring_poison(&room);
            (room.grid(), room.state())
        });
        let Some(map) = state.map else {
            published_scans = None;
            continue;
        };
        let building_changed = state.mapping && published_scans != Some(map.scans);
        if !(changed || building_changed) {
            continue;
        }
        let Some(grid) = grid else {
            continue;
        };
        published_scans = state.mapping.then_some(map.scans);
        let timestamp = time_of(i64::try_from(inputs.clock.now_ns()).unwrap_or(i64::MAX));
        let message = map_grids::build_message(
            MAP_FRAME.to_string(),
            timestamp,
            map.name,
            grid.extent().resolution_m,
            extent_of(&grid.extent()),
            grid.width(),
            grid.height(),
            cells_of(&grid),
        );
        let sent = match message {
            Ok(payload) => publisher.publish(payload).await,
            Err(error) => Err(error),
        };
        if let Err(error) = sent {
            warn!("map_grids is not published: {error}");
        }
    }
}

/// The cells of `grid`, row by row from y_min, each row from x_min: 0 free, 100 occupied
/// and 255 unknown, Nav2's map server's values for a trinary map, its -1 as 255.
fn cells_of(grid: &OccupancyGrid) -> Vec<u8> {
    (0..grid.height())
        .flat_map(|y| (0..grid.width()).map(move |x| (x, y)))
        .map(|(x, y)| {
            grid.cell(x, y)
                .expect("a cell of the grid's own extent")
                .occupancy()
                .cast_unsigned()
        })
        .collect()
}
