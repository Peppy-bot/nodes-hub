//! The workspace contract's two services, describe_workspace and
//! check_positions, answered for the life of the node by [`Workspace`] from
//! the robot's design, so served from bringup, ahead of the readiness gate.
//! Each request parses its fields first, a request that does not parse
//! getting an error reply with `rejected`. check_positions then places its
//! points, in its frame at its timestamp, in the robot frame by the robot's
//! tree with one placement ([`Placer`]), points the tree cannot place getting
//! an error reply with the failure kind of the placement,
//! and refuses a placed point too far from the robot's base point with
//! `rejected`. Each then reads the perception camera's geometry from the
//! camera linked as perception_geometry, when the robot has a perception
//! camera and a camera is linked: the view then follows that camera's own
//! intrinsics, depth model and depth range, and a camera that cannot give
//! them now makes the answer an error reply with `precondition_failed`. Every
//! answer is in the robot frame, which it names. The generated handlers are
//! synchronous, so each answers in place on the runtime: the camera's two
//! answers are awaited together, for at most [`GEOMETRY_TIMEOUT`].

use std::sync::Arc;
use std::time::Duration;

use camera_core::pinhole::{Intrinsics, Pinhole};
use control_core::geometry::{Point, Stamped, report_timestamp};
use control_core::robot_tree::PlacedValue;
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
use workspace_core::design::{Positions, PositionsError, SurfaceHeight, ViewCheck};

use crate::failure::Unplaced;
use crate::placement::Placer;
use crate::workspace::{CameraGeometryError, PerceptionCamera, Workspace, depth_from_wire};

/// How long a request waits for each of the camera's answers: short, so a
/// camera that does not answer leaves the caller's deadline room for the
/// error reply that says so.
const GEOMETRY_TIMEOUT: Duration = Duration::from_secs(2);

/// Serve both services for the life of the node, each on its own task, so a
/// surface measured for the first time does not hold a check back. `placer`
/// places the points of check_positions in the robot frame.
pub fn spawn(
    runner: Arc<NodeRunner>,
    token: CancellationToken,
    workspace: Arc<Workspace>,
    placer: Arc<Placer>,
) {
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
        placer.robot_frame().to_string(),
    ));
    tokio::spawn(serve_check(runner, token, workspace, placer));
}

async fn serve_describe(
    runner: Arc<NodeRunner>,
    token: CancellationToken,
    workspace: Arc<Workspace>,
    robot_frame: String,
) {
    serving::serve("describe_workspace", token.cancelled(), || {
        describe_workspace::handle_requests(&runner, |request| {
            tokio::task::block_in_place(|| {
                describe(
                    &runner,
                    &workspace,
                    &robot_frame,
                    request.data.surface_height,
                )
            })
        })
    })
    .await;
}

async fn serve_check(
    runner: Arc<NodeRunner>,
    token: CancellationToken,
    workspace: Arc<Workspace>,
    placer: Arc<Placer>,
) {
    serving::serve("check_positions", token.cancelled(), || {
        check_positions::handle_requests(&runner, |request| {
            tokio::task::block_in_place(|| check(&runner, &workspace, &placer, &request.data))
        })
    })
    .await;
}

/// The answer to one describe_workspace request, or its error reply: its
/// bounds in `robot_frame`, which hold at every time, so the answer holds no
/// timestamp.
fn describe(
    runner: &NodeRunner,
    workspace: &Workspace,
    robot_frame: &str,
    surface_height: f64,
) -> Result<describe_workspace::Response, describe_workspace::Failure> {
    use describe_workspace::{Failure, FailureKind};
    let height = SurfaceHeight::from_wire(surface_height)
        .map_err(|e| Failure::new(FailureKind::Rejected, e.to_string()))?;
    let view_check = read_view_check(runner, workspace)
        .map_err(|e| Failure::new(FailureKind::PreconditionFailed, e.to_string()))?;
    let answer = workspace.describe(height, &view_check);
    Ok(describe_workspace::Response::new(
        robot_frame.to_owned(),
        perception_camera_name(&view_check),
        answer.workable,
        answer.area,
        answer.rectangle,
        answer.reach,
        answer.view,
    ))
}

/// The answer to one check_positions request, or its error reply: each
/// point as `placer` placed it, in the robot frame at the time of the robot
/// frame's place.
fn check(
    runner: &NodeRunner,
    workspace: &Workspace,
    placer: &Placer,
    request: &check_positions::RequestData,
) -> Result<check_positions::Response, check_positions::Failure> {
    use check_positions::{Failure, FailureKind};
    let placed = placed_points(placer, request)?;
    let positions = Positions::parse(placed.value)
        .map_err(|e| Failure::new(FailureKind::Rejected, e.to_string()))?;
    let view_check = read_view_check(runner, workspace)
        .map_err(|e| Failure::new(FailureKind::PreconditionFailed, e.to_string()))?;
    let answer = workspace.check(&positions, &view_check);
    let all_workable = answer.all_workable();
    let results = answer
        .points
        .into_iter()
        .map(|point| check_positions::ResponseResultsItem {
            frame: placer.robot_frame().to_string(),
            timestamp: report_timestamp(placed.time),
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

/// The points of a request, placed together in the robot frame by
/// `placer`, in the request's order, with the time of the robot frame's
/// place; refused with `rejected` for an empty list and for a value no
/// contract takes, a point named by its place in the request, from 1, and
/// with the failure kind of the placement when the tree cannot place them.
/// The list and every point are refused before the placement waits for the
/// tree.
fn placed_points(
    placer: &Placer,
    request: &check_positions::RequestData,
) -> Result<PlacedValue<Vec<Point>>, check_positions::Failure> {
    use check_positions::{Failure, FailureKind};
    let rejected = |message: String| Failure::new(FailureKind::Rejected, message);
    if request.points.is_empty() {
        return Err(rejected(PositionsError::Empty.to_string()));
    }
    let points = request
        .points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            Point::parse(point.position).map_err(|e| rejected(format!("point {}: {e}", index + 1)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let frame = placer.frame_of(request.frame.as_deref());
    let stamped =
        Stamped::parse(frame, request.timestamp, points).map_err(|e| rejected(e.to_string()))?;
    placer.place_now(&stamped).map_err(|e| {
        let unplaced = Unplaced::of("the points", &e);
        Failure::new(unplaced.kind.into(), unplaced.message)
    })
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
        geometry: Box::new(camera_from_answers(perception, colour, depth)?),
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
    let pinhole = Pinhole::parse(&Intrinsics {
        width: colour.width,
        height: colour.height,
        fx: colour.fx,
        fy: colour.fy,
        cx: colour.cx,
        cy: colour.cy,
        distortion_model: &colour.distortion_model,
        distortion: &colour.distortion,
    })
    .map_err(|e| Colour {
        camera,
        reason: e.to_string(),
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
    Ok(perception.camera(pinhole, depth))
}
