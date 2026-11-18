//! transform_server: holds a robot's tree of frames, from the transforms and the
//! static_transforms of every publisher bound to its transforms link, and answers
//! transform_lookup's lookup_transform from it for the nodes that hold no tree, as
//! tf2_ros's buffer server answers LookupTransform. A lookup waits up to the request's
//! timeout_s on the node's clock, the simulation clock in a simulation, for the tree to
//! answer it. See README.md.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Duration;

use control_core::fault::Fault;
use control_core::robot_tree::RobotTree;
use control_core::time::duration_from_secs;
use peppygen::{NodeRunner, Parameters, Result};

mod clock;
mod follow;
mod lookup;

/// The cache window of the tree, cache_window_s: refused naming the parameter unless it
/// is finite and above 0.
fn cache_window(params: &Parameters) -> Result<Duration> {
    duration_from_secs(params.cache_window_s).map_err(|e| {
        peppygen::Error::Node(std::io::Error::other(format!("cache_window_s {e}")).into())
    })
}

/// Starts the node: the tree, whose cache window is cache_window_s, on the node's clock,
/// the followers that fill it, and the service that answers lookups in it. The end of a
/// simulation clock's ticks raises `fault`.
pub async fn setup(params: Parameters, runner: Arc<NodeRunner>, fault: Fault) -> Result<()> {
    let window = cache_window(&params)?;
    let clock = clock::node_clock(&runner, fault).await?;
    let tree =
        Arc::new(RobotTree::new(window, clock).expect("duration_from_secs gives a window above 0"));
    follow::follow_tree(&runner, tree.clone()).await?;
    lookup::serve(runner, tree);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cache_window_that_is_not_finite_and_above_0_stops_the_node_naming_it() {
        for (cache_window_s, rule) in [
            (0.0, "must be a positive finite number of seconds, got 0"),
            (-1.0, "must be a positive finite number of seconds, got -1"),
            (
                f64::NAN,
                "must be a positive finite number of seconds, got NaN",
            ),
            (1e-12, "1e-12 s rounds to no time at all"),
        ] {
            let refused = cache_window(&Parameters { cache_window_s })
                .expect_err("a window the tree does not take");
            assert!(
                refused
                    .to_string()
                    .contains(&format!("cache_window_s {rule}")),
                "{refused}"
            );
        }
        assert_eq!(
            cache_window(&Parameters {
                cache_window_s: 10.0
            })
            .ok(),
            Some(Duration::from_secs(10))
        );
    }
}
