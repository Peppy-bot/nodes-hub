//! The workspace contract's two services, describe_workspace and
//! check_positions, answered for the life of the node by [`Workspace`] from
//! the robot's design, so served from bringup, ahead of the readiness gate.
//! Each request parses its fields first, then reads the perception camera's
//! geometry from the camera linked as perception_geometry, when the robot
//! has a perception camera and a camera is linked: the view then follows that
//! camera's own intrinsics, depth model and depth range, and a camera that
//! cannot give them now makes the answer a refusal. The generated handlers are
//! synchronous, so each answers in place on the runtime: the camera's two
//! answers are awaited together, for at most [`GEOMETRY_TIMEOUT`].

use std::sync::Arc;
use std::time::Duration;

use peppygen::NodeRunner;
use peppygen::consumed_services::perception_geometry::{
    get_color_intrinsics, get_depth_intrinsics,
};
use peppygen::exposed_services::workspace::{check_positions, describe_workspace};
use peppylib::runtime::CancellationToken;
use tracing::info;
use workspace_core::Camera;
use workspace_core::design::{Positions, SurfaceHeight, ViewCheck};

use crate::serving;
use crate::workspace::{
    CameraGeometryError, PerceptionCamera, Workspace, depth_from_wire, intrinsics_from_wire,
};

/// How long a request waits for each of the camera's answers: short, so a
/// camera that does not answer leaves the caller's deadline room for the
/// refusal that says so.
const GEOMETRY_TIMEOUT: Duration = Duration::from_secs(2);

/// Serve both services for the life of the node, each on its own task, so a
/// surface measured for the first time does not hold a check back.
pub fn spawn(runner: Arc<NodeRunner>, token: CancellationToken, workspace: Arc<Workspace>) {
    info!(
        "workspace: perception camera {}",
        workspace
            .perception_camera()
            .map_or("none", |camera| camera.name)
    );
    tokio::spawn(serve_describe(
        runner.clone(),
        token.clone(),
        workspace.clone(),
    ));
    tokio::spawn(serve_check(runner, token, workspace));
}

async fn serve_describe(
    runner: Arc<NodeRunner>,
    token: CancellationToken,
    workspace: Arc<Workspace>,
) {
    serving::serve("describe_workspace", &token, || {
        describe_workspace::handle_next_request(&runner, |request| {
            Ok(tokio::task::block_in_place(|| {
                describe(&runner, &workspace, request.data.surface_height)
            }))
        })
    })
    .await;
}

async fn serve_check(runner: Arc<NodeRunner>, token: CancellationToken, workspace: Arc<Workspace>) {
    serving::serve("check_positions", &token, || {
        check_positions::handle_next_request(&runner, |request| {
            Ok(tokio::task::block_in_place(|| {
                check(&runner, &workspace, &request.data.positions)
            }))
        })
    })
    .await;
}

/// The answer to one describe_workspace request, or its refusal.
fn describe(
    runner: &NodeRunner,
    workspace: &Workspace,
    surface_height: f64,
) -> describe_workspace::Response {
    let refused = |reason: String| {
        describe_workspace::Response::new(
            false,
            reason,
            String::new(),
            false,
            0.0,
            None,
            None,
            None,
        )
    };
    let height = match SurfaceHeight::from_wire(surface_height) {
        Ok(height) => height,
        Err(e) => return refused(e.to_string()),
    };
    let view_check = match read_view_check(runner, workspace) {
        Ok(view_check) => view_check,
        Err(e) => return refused(e.to_string()),
    };
    let answer = workspace.describe(height, &view_check);
    describe_workspace::Response::new(
        true,
        answer.message,
        perception_camera_name(&view_check),
        answer.workable,
        answer.area,
        answer.rectangle,
        answer.reach,
        answer.view,
    )
}

/// The answer to one check_positions request, or its refusal.
fn check(
    runner: &NodeRunner,
    workspace: &Workspace,
    positions: &[f64],
) -> check_positions::Response {
    let refused = |reason: String| {
        check_positions::Response::new(false, reason, String::new(), false, Vec::new())
    };
    let positions = match Positions::from_wire(positions) {
        Ok(positions) => positions,
        Err(e) => return refused(e.to_string()),
    };
    let view_check = match read_view_check(runner, workspace) {
        Ok(view_check) => view_check,
        Err(e) => return refused(e.to_string()),
    };
    let answer = workspace.check(&positions, &view_check);
    let all_workable = answer.all_workable();
    let results = answer
        .points
        .into_iter()
        .map(|point| check_positions::ResponseResultsItem {
            position: point.position,
            workable: point.workable(),
            reachable: point.reach.reached(),
            arm: point.reach.arm().to_owned(),
            short_by: point.reach.short_by(),
            in_view: point.view.seen(),
            view: point.view.name().to_owned(),
            message: point.message,
        })
        .collect();
    check_positions::Response::new(
        true,
        answer.message,
        perception_camera_name(&view_check),
        all_workable,
        results,
    )
}

/// The perception camera's name as the answers carry it: "" for a robot
/// without one.
fn perception_camera_name(view_check: &ViewCheck<'_>) -> String {
    view_check
        .perception_camera()
        .unwrap_or_default()
        .to_owned()
}

/// How the answer checks the view: through the perception camera, placed by
/// the design and seen through the geometry the linked camera gives now; not
/// at all when the robot has no perception camera or no camera is linked
/// ([`Workspace::unlinked_view_check`]). Run from a handler: it blocks in
/// place on the runtime for the answers.
fn read_view_check(
    runner: &NodeRunner,
    workspace: &Workspace,
) -> Result<ViewCheck<'static>, CameraGeometryError> {
    let (Some(perception), Some(linked)) = (
        workspace.perception_camera(),
        get_color_intrinsics::bound_producer(runner),
    ) else {
        return Ok(workspace.unlinked_view_check());
    };
    let (colour, depth) = tokio::runtime::Handle::current().block_on(async {
        tokio::join!(
            get_color_intrinsics::poll(runner, linked, GEOMETRY_TIMEOUT),
            get_depth_intrinsics::poll(runner, linked, GEOMETRY_TIMEOUT),
        )
    });
    Ok(ViewCheck::Checked {
        camera: perception.name,
        geometry: camera_from_answers(perception, colour, depth)?,
    })
}

/// The perception camera seen through the linked camera's two answers, or
/// why it cannot be: either answer missing or refused, or its values not a
/// camera's.
fn camera_from_answers(
    perception: &PerceptionCamera,
    colour: peppygen::Result<get_color_intrinsics::Response>,
    depth: peppygen::Result<get_depth_intrinsics::Response>,
) -> Result<Camera, CameraGeometryError> {
    use CameraGeometryError::{Colour, Depth, NoDepth};
    let camera = perception.name;
    let colour = colour
        .map_err(|e| Colour {
            camera,
            reason: e.to_string(),
        })?
        .data;
    if !colour.success {
        return Err(Colour {
            camera,
            reason: colour.message,
        });
    }
    let intrinsics = intrinsics_from_wire(
        colour.width,
        colour.height,
        colour.fx,
        colour.fy,
        colour.cx,
        colour.cy,
    )
    .map_err(|reason| Colour {
        camera,
        reason: reason.to_owned(),
    })?;
    let depth = depth
        .map_err(|e| Depth {
            camera,
            reason: e.to_string(),
        })?
        .data;
    if !depth.success {
        return Err(NoDepth {
            camera,
            reason: depth.message,
        });
    }
    let depth =
        depth_from_wire(&depth.depth_model, depth.min_depth_m, depth.max_depth_m).map_err(|e| {
            Depth {
                camera,
                reason: e.to_string(),
            }
        })?;
    Ok(perception.camera(intrinsics, depth))
}
