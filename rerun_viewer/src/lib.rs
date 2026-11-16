//! rerun_viewer: the node's parameters, the canvas it draws on (the live canvas over the
//! Rerun viewer, or the fake of the tests), the values it draws and their arithmetic,
//! how it draws each value, the wiring that turns the messages of a stack into those
//! values, and the node's start. See README.md.

pub mod canvas;
pub mod config;
pub mod draw;
pub mod drawn;
pub mod live;
pub mod model;
pub mod node;
pub mod testing;
pub mod wire;
