//! map_grids: the loaded map's grid, published at once when a map is loaded, mapping starts
//! or mapping finishes, and while mapping runs at most once per grid period once the grid
//! changes. Each carries the stamp of its map's load; a new map's grid before its first
//! scan has no extent and no cell.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use control_core::sync::lock_ignoring_poison;
use mapping_core::OccupancyGrid;
use peppygen::NodeRunner;
use peppygen::emitted_topics::room_map::map_grids;
use tokio::sync::mpsc;
use tracing::warn;

use crate::inputs::Inputs;
use crate::messages::{duration_ns, extent_of, time_of};
use crate::rooms::members::MAP_FRAME;
use crate::rooms::state::{GridSource, MapGrid, RoomState};

/// Runs the grid's publisher until the members' grids end: each grid a member hands over at
/// a change of the loaded map, and while mapping runs, at most one each grid period once the
/// scans in its pose graph changed. A grid is drawn only to be published.
pub async fn run(
    runner: Arc<NodeRunner>,
    room: Arc<Mutex<RoomState>>,
    inputs: Arc<Inputs>,
    mut handed: mpsc::UnboundedReceiver<MapGrid>,
    period: Duration,
) -> peppygen::Result<()> {
    let publisher = map_grids::declare_publisher(&runner).await?;
    let period_ns = duration_ns(period);
    let mut published: Option<GridSource> = None;
    loop {
        let next = inputs.clock.now_ns() + period_ns;
        // A member's grid first: the period draws none while one waits.
        let drawn = tokio::select! {
            biased;
            handed_over = handed.recv() => match handed_over {
                Some(grid) => Some(grid),
                None => return Ok(()),
            },
            () = inputs.clock.until(next) => tokio::task::block_in_place(|| {
                draw_building(&room, &handed, published)
            }),
        };
        let Some(grid) = drawn else {
            continue;
        };
        published = Some(grid.source);
        let (extent, columns, rows, cells) = match &grid.grid {
            Some(drawn) => (
                Some(extent_of(&drawn.extent())),
                drawn.width(),
                drawn.height(),
                cells_of(drawn),
            ),
            None => (None, 0, 0, Vec::new()),
        };
        let timestamp = time_of(i64::try_from(inputs.clock.now_ns()).unwrap_or(i64::MAX));
        let message = map_grids::build_message(
            MAP_FRAME.to_string(),
            timestamp,
            grid.map_name,
            time_of(grid.load_unix_ns),
            grid.cell_size_m,
            extent,
            columns,
            rows,
            cells,
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

/// The grid of the map being built, drawn at a grid period when the scans in its pose graph
/// changed since the grid `published` last. None while a member's grid waits in `handed`:
/// the member handed it over while it held the room, so it goes out first.
fn draw_building(
    room: &Mutex<RoomState>,
    handed: &mpsc::UnboundedReceiver<MapGrid>,
    published: Option<GridSource>,
) -> Option<MapGrid> {
    let room = lock_ignoring_poison(room);
    if !handed.is_empty() {
        return None;
    }
    let source @ GridSource::Building { .. } = room.grid_source()? else {
        return None;
    };
    if published == Some(source) {
        return None;
    }
    room.map_grid()
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
