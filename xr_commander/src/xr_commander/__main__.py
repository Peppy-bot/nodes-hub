"""Wiring: parameters, camera discovery, the WebXR server, one task per stream.

The server is started, never awaited: the health probe registers only once
`setup` returns.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import peppygen.clock
from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_actions.postures import (
    move_to_home as postures_move_to_home,
)
from peppygen.consumed_actions.postures import (
    move_to_ready as postures_move_to_ready,
)
from peppygen.consumed_actions.recorder import (
    record_episode as recorder_record_episode,
)
from peppygen.consumed_services.recorder import (
    finish_session as recorder_finish_session,
)
from peppygen.consumed_topics.alerts import alerts as alerts_topic
from peppygen.consumed_topics.color_cameras import (
    video_stream as color_cameras_video_stream,
)
from peppygen.consumed_topics.motor_health import (
    motor_health as motor_health_topic,
)
from peppygen.consumed_topics.rgbd_cameras import (
    video_stream as rgbd_cameras_video_stream,
)
from peppygen.paired_topics.left_arm import pose_setpoints as left_arm_pose_setpoints
from peppygen.paired_topics.left_arm import pose_states as left_arm_pose_states
from peppygen.paired_topics.left_gripper import (
    gripper_setpoints as left_gripper_setpoints,
)
from peppygen.paired_topics.right_arm import pose_setpoints as right_arm_pose_setpoints
from peppygen.paired_topics.right_arm import pose_states as right_arm_pose_states
from peppygen.paired_topics.right_gripper import (
    gripper_setpoints as right_gripper_setpoints,
)
from peppygen.parameters import Parameters
from teleop_xr.video_stream import ExternalVideoSource

from xr_commander import (
    alerts,
    config,
    motor_health,
    panel,
    publish,
    record,
    task_page,
    tls,
    video,
)
from xr_commander.bus import log
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample
from xr_commander.publish import LatestPose
from xr_commander.xr import XrSession


@dataclass(frozen=True)
class _Recording:
    """The state only a bound recorder brings: what the panel reports, and the
    label the next episode records under. Held together so the task page and
    the REC row can never exist one without the other."""

    status: record.RecorderStatus
    label: task_page.TaskLabel


@dataclass(frozen=True)
class _HandWiring:
    """The three generated modules one operator hand drives."""

    pose_setpoints: ModuleType
    pose_states: ModuleType
    gripper_setpoints: ModuleType


# Keyed by WebXR handedness; the launcher decides which limb a hand drives.
_HANDS = {
    "left": _HandWiring(
        pose_setpoints=left_arm_pose_setpoints,
        pose_states=left_arm_pose_states,
        gripper_setpoints=left_gripper_setpoints,
    ),
    "right": _HandWiring(
        pose_setpoints=right_arm_pose_setpoints,
        pose_states=right_arm_pose_states,
        gripper_setpoints=right_gripper_setpoints,
    ),
}

# Only the colour half of an RGB-D camera is taken.
_CAMERA_SLOTS = (
    ("color camera", color_cameras_video_stream),
    ("rgbd camera", rgbd_cameras_video_stream),
)


def _primary_pressed(sample: HandSample) -> bool:
    return sample.primary_button


def _secondary_pressed(sample: HandSample) -> bool:
    return sample.secondary_button


# The right controller's face buttons: A (the lower button) goes home, B goes
# ready. Each entry is (consumed action module, its button).
_POSTURE_BUTTONS = (
    (postures_move_to_home, _primary_pressed),
    (postures_move_to_ready, _secondary_pressed),
)


def _headset_origin(settings: config.Settings, *, outbound_ip=tls.outbound_ip) -> str:
    """Where this node serves, as the operator would type it."""
    bound = settings.https_host
    host = outbound_ip() if bound.is_unspecified else str(bound)
    if host is not None and ":" in host:
        host = f"[{host}]"
    return f"https://{host or '<this machine>'}:{settings.https_port}"


def _headset_url_hint(settings: config.Settings, *, outbound_ip=tls.outbound_ip) -> str:
    """A URL the operator can type into the headset's browser as printed."""
    port = settings.https_port
    origin = _headset_origin(settings, outbound_ip=outbound_ip)
    return (
        f"open {origin} in the headset's browser "
        "(click through the self-signed warning) and enter VR. Over USB instead: "
        f"`adb reverse tcp:{port} tcp:{port}`, then https://localhost:{port}."
    )


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    settings = config.from_parameters(params)

    await peppygen.clock.init(node_runner)

    slot_tracks = [
        (
            label,
            topic_module,
            video.discover_tracks(node_runner, topic_module, ExternalVideoSource),
        )
        for label, topic_module in _CAMERA_SLOTS
    ]
    tracks = [track for _label, _module, found in slot_tracks for track in found]
    # The status panel rides the same WebRTC path as a camera, so it shares the
    # one headset-name space and is refused a collision like any other track.
    panel_sink = ExternalVideoSource() if settings.status_panel_enabled else None
    track_ids = [t.instance_id for t in tracks]
    if panel_sink is not None:
        track_ids.append(panel.TRACK_ID)
    video.assert_unique_track_ids(track_ids)
    described = ", ".join(t.instance_id for t in tracks)
    log(f"{len(tracks)} camera track(s): {described or 'none'}")

    # One certificate per machine, reused so the browser's acceptance sticks.
    try:
        tls_dir = Path.home() / ".xr_commander" / "tls"
        cert_path, key_path = await asyncio.to_thread(tls.ensure_certificate, tls_dir)
    except (OSError, RuntimeError) as e:
        raise RuntimeError(
            "cannot create the TLS identity under ~/.xr_commander/tls "
            f"(unwritable home?): {e}"
        ) from e
    video_sources = {t.instance_id: t.sink for t in tracks}
    if panel_sink is not None:
        video_sources[panel.TRACK_ID] = panel_sink
    # One gate for everything recording-related: teleoperation without a
    # recorder bound serves no task page and reads no label.
    recording = (
        _Recording(record.RecorderStatus(), task_page.TaskLabel())
        if recorder_record_episode.bound_producers(node_runner)
        else None
    )
    session = XrSession(
        host=str(settings.https_host),
        port=settings.https_port,
        tls_cert_path=cert_path,
        tls_key_path=key_path,
        video_sources=video_sources,
        camera_views=video.camera_views(track_ids),
        routers=() if recording is None else (task_page.build_router(recording.label),),
    )

    async def stop_session() -> None:
        log("shutdown signal received")
        # Joins the server thread, so keep it off the event loop.
        await asyncio.to_thread(session.stop)

    # Registered before start: a failed start must still be torn down.
    node_runner.on_shutdown(stop_session)

    # start() blocks until the server is up (or raises); keep it off the loop.
    await asyncio.to_thread(session.start)

    log(
        f"{_headset_url_hint(settings)} The WebSocket carries "
        "unauthenticated motion control: trusted networks only."
    )
    if recording is not None:
        log(
            f"episodes record as {task_page.UNNAMED_TASK!r} until the task is "
            f"named at {_headset_origin(settings)}{task_page.PATH}"
        )

    token = node_runner.cancellation_token()
    # Shared by the alert listener, the status panel, and every camera drain.
    alerts_bound = bool(alerts_topic.bound_producers(node_runner))
    active_alerts = alerts.ActiveAlerts(
        # A stack with nothing bound to the alert slot can never receive an
        # alert, so its quiet panel means "not wired" rather than "nothing
        # wrong". Silence must not be rendered as health.
        producers_bound=alerts_bound,
    )
    # Shared by the health listener and the status panel, on the same
    # unwired-is-not-healthy reasoning as the alerts.
    health_bound = bool(motor_health_topic.bound_producers(node_runner))
    motor_reports = motor_health.MotorHealthReports(producers_bound=health_bound)
    tasks = [
        asyncio.create_task(
            publish.run_posture_button(
                node_runner,
                action_module=action_module,
                pressed=pressed,
                session=session,
                settings=settings,
                token=token,
            )
        )
        for action_module, pressed in _POSTURE_BUTTONS
    ]
    # Skipped when nothing is bound, like the camera drains: there is
    # nothing to receive, and the panel already says "not wired".
    if alerts_bound:
        tasks.append(
            asyncio.create_task(
                alerts.drain_alerts(node_runner, alerts_topic, active_alerts, token)
            )
        )
    if health_bound:
        tasks.append(
            asyncio.create_task(
                motor_health.drain_motor_health(
                    node_runner, motor_health_topic, motor_reports, token
                )
            )
        )
    for label, topic_module, found in slot_tracks:
        if not found:
            continue
        # Stamped by the drain, read by the watchdog that blanks silent tracks.
        health: dict[str, float] = {}
        tasks.append(
            asyncio.create_task(
                video.drain_frames(
                    node_runner,
                    topic_module,
                    {t.instance_id: t.sink for t in found},
                    token,
                    label,
                    settings.view_max_width,
                    health,
                )
            )
        )
        tasks.append(
            asyncio.create_task(video.watch_camera_silence(found, health, token))
        )
    if recording is not None:
        tasks.append(
            asyncio.create_task(
                record.run_recorder_buttons(
                    node_runner,
                    action_module=recorder_record_episode,
                    finish_module=recorder_finish_session,
                    status=recording.status,
                    session=session,
                    settings=settings,
                    token=token,
                    read_task=recording.label.for_episode,
                )
            )
        )

    # Shared with the status panel, which reports the same state the pose
    # streams act on.
    def arm_paired(topic_module):
        return lambda: topic_module.paired(node_runner) is not None

    hands = tuple(
        panel.HandSource(
            handedness=handedness,
            clutch=HandClutch(settings.motion_scale),
            measured=LatestPose(),
            arm_paired=arm_paired(wiring.pose_setpoints),
        )
        for handedness, wiring in _HANDS.items()
    )
    clutches = {hand.handedness: hand.clutch for hand in hands}
    measured = {hand.handedness: hand.measured for hand in hands}
    if panel_sink is not None:
        tasks.append(
            asyncio.create_task(
                panel.stream_status(
                    panel_sink,
                    session=session,
                    hands=hands,
                    settings=settings,
                    token=token,
                    alerts=active_alerts,
                    health=motor_reports,
                    recorder=None if recording is None else recording.status,
                    read_task=None if recording is None else recording.label.current,
                )
            )
        )

    for handedness, wiring in _HANDS.items():
        tasks.append(
            asyncio.create_task(
                publish.drain_pose_states(
                    node_runner,
                    wiring.pose_states,
                    measured[handedness],
                    handedness,
                    token,
                )
            )
        )
        tasks.append(
            asyncio.create_task(
                publish.stream_pose(
                    node_runner,
                    topic_module=wiring.pose_setpoints,
                    handedness=handedness,
                    clutch=clutches[handedness],
                    session=session,
                    measured=measured[handedness],
                    settings=settings,
                    token=token,
                )
            )
        )
        tasks.append(
            asyncio.create_task(
                publish.stream_gripper(
                    node_runner,
                    topic_module=wiring.gripper_setpoints,
                    handedness=handedness,
                    session=session,
                    settings=settings,
                    token=token,
                )
            )
        )

    return tasks


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
