//! The workspace contract's two services, describe_workspace and
//! check_positions, answered for the life of the node by [`Workspace`] from
//! the robot's design, so served from bringup, ahead of the readiness gate.
//! Each request parses its fields first, a request that does not parse
//! getting an error reply with `rejected`, then reads the perception
//! camera's geometry from the camera linked as perception_geometry, when the
//! robot has a perception camera and a camera is linked: the view then
//! follows that camera's own intrinsics, depth model and depth range, and a
//! camera that cannot give them now makes the answer an error reply with
//! `precondition_failed`. The generated handlers are synchronous, so each
//! answers in place on the runtime: the camera's two answers are awaited
//! together, for at most [`GEOMETRY_TIMEOUT`].

use std::sync::Arc;
use std::time::Duration;

use control_core::serving;
use control_core::sync::wait_for;
use peppygen::NodeRunner;
use peppygen::consumed_services::perception_geometry::{
    get_color_intrinsics, get_depth_intrinsics,
};
use peppygen::exposed_services::workspace::{check_positions, describe_workspace};
use peppylib::runtime::CancellationToken;
use tracing::info;
use workspace_core::Camera;
use workspace_core::design::{Positions, SurfaceHeight, ViewCheck};

use crate::workspace::{
    CameraGeometryError, PerceptionCamera, Workspace, depth_from_wire, intrinsics_from_wire,
};

/// How long a request waits for each of the camera's answers: short, so a
/// camera that does not answer leaves the caller's deadline room for the
/// error reply that says so.
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
    serving::serve("describe_workspace", token.cancelled(), || {
        describe_workspace::handle_next_request(&runner, |request| {
            tokio::task::block_in_place(|| {
                describe(&runner, &workspace, request.data.surface_height)
            })
        })
    })
    .await;
}

async fn serve_check(runner: Arc<NodeRunner>, token: CancellationToken, workspace: Arc<Workspace>) {
    serving::serve("check_positions", token.cancelled(), || {
        check_positions::handle_next_request(&runner, |request| {
            tokio::task::block_in_place(|| check(&runner, &workspace, &request.data.positions))
        })
    })
    .await;
}

/// The answer to one describe_workspace request, or its error reply.
fn describe(
    runner: &NodeRunner,
    workspace: &Workspace,
    surface_height: f64,
) -> Result<describe_workspace::Response, describe_workspace::Failure> {
    use describe_workspace::{Failure, FailureKind};
    let height = SurfaceHeight::from_wire(surface_height)
        .map_err(|e| Failure::new(FailureKind::Rejected, e.to_string()))?;
    let view_check = read_view_check(runner, workspace)
        .map_err(|e| Failure::new(FailureKind::PreconditionFailed, e.to_string()))?;
    let answer = workspace.describe(height, &view_check);
    Ok(describe_workspace::Response::new(
        perception_camera_name(&view_check),
        answer.workable,
        answer.area,
        answer.rectangle,
        answer.reach,
        answer.view,
    ))
}

/// The answer to one check_positions request, or its error reply.
fn check(
    runner: &NodeRunner,
    workspace: &Workspace,
    positions: &[f64],
) -> Result<check_positions::Response, check_positions::Failure> {
    use check_positions::{Failure, FailureKind};
    let positions = Positions::from_wire(positions)
        .map_err(|e| Failure::new(FailureKind::Rejected, e.to_string()))?;
    let view_check = read_view_check(runner, workspace)
        .map_err(|e| Failure::new(FailureKind::PreconditionFailed, e.to_string()))?;
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
    Ok(check_positions::Response::new(
        perception_camera_name(&view_check),
        all_workable,
        results,
    ))
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
    let (colour, depth) = wait_for(async {
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
/// why it cannot be: either answer missing or an error reply, whose message
/// is the reason, or its values not a camera's. A depth error reply with
/// `rejected` is a camera that gives no depth.
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
            reason: get_color_intrinsics::Failure::of(&e)
                .map_or_else(|| e.to_string(), |failure| failure.message),
        })?
        .data;
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
        .map_err(|e| match get_depth_intrinsics::Failure::of(&e) {
            Some(failure) if failure.kind == get_depth_intrinsics::FailureKind::Rejected => {
                NoDepth {
                    camera,
                    reason: failure.message,
                }
            }
            Some(failure) => Depth {
                camera,
                reason: failure.message,
            },
            None => Depth {
                camera,
                reason: e.to_string(),
            },
        })?
        .data;
    let depth =
        depth_from_wire(&depth.depth_model, depth.min_depth_m, depth.max_depth_m).map_err(|e| {
            Depth {
                camera,
                reason: e.to_string(),
            }
        })?;
    Ok(perception.camera(intrinsics, depth))
}
