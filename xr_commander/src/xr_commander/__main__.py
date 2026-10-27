"""Wiring: parameters, the WebXR server, the paired robots, the fleet, one
task per stream.

The server is started, never awaited: the health probe registers only once
`setup` returns.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import peppygen.clock
from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_actions.fleet import changes as fleet_changes
from peppygen.consumed_actions.fleet import lease as fleet_lease
from peppygen.consumed_actions.postures import (
    move_to_home as postures_move_to_home,
)
from peppygen.consumed_actions.postures import (
    move_to_ready as postures_move_to_ready,
)
from peppygen.consumed_actions.recorder import (
    record_episode as recorder_record_episode,
)
from peppygen.consumed_services.fleet import check_move as fleet_check_move
from peppygen.consumed_services.fleet import release as fleet_release
from peppygen.consumed_services.fleet import take as fleet_take
from peppygen.consumed_services.limb_motion import stop as limb_motion_stop
from peppygen.consumed_services.recorder import (
    finish_session as recorder_finish_session,
)
from peppygen.consumed_services.sensor_streams import hold as sensor_streams_hold
from peppygen.consumed_services.sensor_streams import (
    release as sensor_streams_release,
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
from peppygen.emitted_topics.headset import page as headset_page_topic
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
    addresses,
    alerts,
    config,
    drive,
    fleet,
    motor_health,
    page,
    panel,
    publish,
    record,
    robot_choice,
    sensor_hold,
    task_page,
    tls,
    video,
)
from xr_commander.bus import log
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample
from xr_commander.publish import MeasuredPoses
from xr_commander.robots import PairedRobots
from xr_commander.xr import XrSession


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


def _headset_url_hint(page_urls: Sequence[str], port: int) -> str:
    """How the operator reaches the page: its URL on the network, or the USB
    route through `adb reverse`."""
    usb = f"`adb reverse tcp:{port} tcp:{port}`, then open https://localhost:{port}"
    start = (
        "click through the self-signed warning, and choose Start or Start Passthrough"
    )
    if page_urls:
        return (
            f"open {' or '.join(page_urls)} in the headset's browser, {start}. "
            f"Over USB: {usb}."
        )
    return (
        "the page has no network address (https_host is loopback, or this machine "
        "has none), so it is reachable over USB only: "
        f"{usb}, {start}."
    )


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    settings = config.from_parameters(params)

    await peppygen.clock.init(node_runner)

    # The status panel rides the same WebRTC path as a camera, under a name
    # no camera role may take.
    panel_sink = ExternalVideoSource() if settings.status_panel_enabled else None
    video_sources: dict[str, ExternalVideoSource] = {}
    if panel_sink is not None:
        video_sources[panel.TRACK_ID] = panel_sink

    # One certificate per machine, reused so the browser's acceptance sticks.
    try:
        tls_dir = Path.home() / ".xr_commander" / "tls"
        cert_path, key_path = await asyncio.to_thread(tls.ensure_certificate, tls_dir)
    except (OSError, RuntimeError) as e:
        raise RuntimeError(
            "cannot create the TLS identity under ~/.xr_commander/tls "
            f"(unwritable home?): {e}"
        ) from e
    # The label episodes record under, set on the page; the chosen robot's
    # recorder, when it has one, records under it.
    task_label = task_page.TaskLabel()
    session = XrSession(
        host=str(settings.https_host),
        port=settings.https_port,
        tls_cert_path=cert_path,
        tls_key_path=key_path,
        video_sources=video_sources,
    )

    # Every robot's backbone pairs into the four slots; the hands drive the
    # chosen one.
    robots = PairedRobots(
        node_runner,
        arms={hand: wiring.pose_setpoints for hand, wiring in _HANDS.items()},
        grippers={hand: wiring.gripper_setpoints for hand, wiring in _HANDS.items()},
    )
    # The robot leases: the headset holds the lease of the robot a person
    # drives, and follows the fleet listing.
    fleet_timing = fleet.Timing()
    the_fleet = fleet.Fleet(
        fleet.Holder(node_runner.bound_instance_id(), node_runner.bound_core_node())
    )
    # The page follows the robots, their states and the choice; the driver
    # sends each change it makes.
    robot_states = robot_choice.RobotStates(
        robots, state_of=lambda name: driver.state_of(name)
    )
    driver = drive.Driver(
        the_fleet,
        fleet.FleetCalls(
            node_runner,
            take_module=fleet_take,
            release_module=fleet_release,
            check_move_module=fleet_check_move,
            timing=fleet_timing,
        ),
        drive.LimbStops(node_runner, robots, limb_motion_stop, fleet_timing),
        timing=fleet_timing,
        on_change=robot_states.refresh,
    )
    # The routes behind the page's Robot card, and the task page beside the
    # page.
    session.serve(robot_choice.build_router(robots, robot_states))
    session.serve(task_page.build_router(task_label))

    async def stop_session() -> None:
        log("shutdown signal received")
        # The page's streams end, so the server stops without waiting on them.
        robot_states.close()
        # Joins the server thread, so keep it off the event loop.
        await asyncio.to_thread(session.stop)

    # Registered before start: a failed start must still be torn down.
    node_runner.on_shutdown(stop_session)

    # start() blocks until the server is up (or raises); keep it off the loop.
    await asyncio.to_thread(session.start)
    # The daemon renders the URLs an operator opens from this announcement,
    # one per address of the machine.
    bound_host, bound_port = session.bound_address()
    node_runner.announce_endpoint("page", "https", bound_host, bound_port)
    page_urls = addresses.page_urls(bound_host, bound_port)

    log(
        f"{_headset_url_hint(page_urls, bound_port)} The WebSocket carries "
        "unauthenticated motion control: trusted networks only."
    )
    log(
        f"episodes record as {task_page.UNNAMED_TASK!r} until a task label is "
        f"set at {task_page.PATH}"
    )
    # Whoever is not wearing the headset reads where its page is from the
    # wire; the topic retains it for them.
    await page.publish_page(node_runner, headset_page_topic, page_urls)

    token = node_runner.cancellation_token()
    # Shared by the alert listener and the status panel. A stack with nothing
    # bound to the alert slot can never receive an alert, so the panel draws
    # no alert section for it: silence must not be rendered as health. The
    # slot is read live, so a producer that joins after start-up brings the
    # section.
    active_alerts = alerts.ActiveAlerts(
        bound_now=lambda: (
            (p.core_node, p.instance_id)
            for p in alerts_topic.bound_producers(node_runner)
        ),
    )
    # Shared by the health listener and the status panel, on the same
    # unwired-is-not-healthy reasoning as the alerts.
    motor_reports = motor_health.MotorHealthReports(
        bound_now=lambda: tuple(
            p.instance_id for p in motor_health_topic.bound_producers(node_runner)
        ),
    )
    tasks = [
        asyncio.create_task(
            fleet.follow_fleet(
                node_runner,
                the_fleet,
                driver,
                changes_module=fleet_changes,
                lease_module=fleet_lease,
                timing=fleet_timing,
                token=token,
            )
        ),
        asyncio.create_task(robot_choice.follow_robot_states(robot_states, token)),
    ]
    posture_status = publish.PostureStatus()
    tasks.extend(
        asyncio.create_task(
            publish.run_posture_button(
                node_runner,
                action_module=action_module,
                pressed=pressed,
                session=session,
                robots=robots,
                driver=driver,
                status=posture_status,
                settings=settings,
                token=token,
            )
        )
        for action_module, pressed in _POSTURE_BUTTONS
    )
    # Always listening: every slot takes any number of producers and one that
    # joins after start-up has to reach the panel, so there is nothing to
    # gate on.
    tasks.append(
        asyncio.create_task(
            alerts.drain_alerts(node_runner, alerts_topic, active_alerts, token)
        )
    )
    tasks.append(
        asyncio.create_task(
            alerts.follow_producers(node_runner, alerts_topic, active_alerts, token)
        )
    )
    tasks.append(
        asyncio.create_task(
            motor_health.drain_motor_health(
                node_runner, motor_health_topic, motor_reports, token
            )
        )
    )
    # The chosen robot's cameras, one panel per camera role.
    panels = video.CameraPanels(
        make_sink=ExternalVideoSource,
        show_views=session.show_views,
        reserved=frozenset(video_sources),
    )
    camera_modules = [topic_module for _label, topic_module in _CAMERA_SLOTS]
    tasks.append(
        asyncio.create_task(
            video.follow_camera_panels(
                node_runner, camera_modules, panels, robots, token
            )
        )
    )
    # Stamped per panel by the drains, read by the watchdog that shows NO
    # SIGNAL on a silent panel.
    health: dict[str, video.LastFrame] = {}
    tasks.extend(
        asyncio.create_task(
            video.drain_frames(
                node_runner,
                topic_module,
                panels,
                robots,
                token,
                label,
                settings.view_max_width,
                health,
            )
        )
        for label, topic_module in _CAMERA_SLOTS
    )
    tasks.append(
        asyncio.create_task(
            video.watch_camera_silence(panels, health, robots.chosen, token)
        )
    )
    tasks.append(asyncio.create_task(video.feed_panels(panels, robots.chosen, token)))

    # In a simulation, the chosen robot's sensors stream at their own rate
    # while a headset page in session drives it. Registered after
    # `stop_session`, so the hold is released before the server stops.
    sensor_target = sensor_streams_hold.bound_producer(node_runner)
    if sensor_target is not None:
        timing = sensor_hold.Timing()
        hold = sensor_hold.ChosenRobotHold(
            sensor_hold.SensorStreams(
                node_runner,
                sensor_target,
                hold_module=sensor_streams_hold,
                release_module=sensor_streams_release,
                timeout_s=timing.call_timeout_s,
            ),
            timing,
        )
        node_runner.on_shutdown(hold.stop)
        tasks.append(
            asyncio.create_task(
                sensor_hold.follow_chosen_robot(
                    hold,
                    sensor_hold.driven_robot(
                        robots.chosen, session.latest, timing.session_timeout_s
                    ),
                    token,
                )
            )
        )

    recorder_status = record.RecorderStatus()

    def recorder_row() -> record.RecorderStatus | None:
        chosen_has_one = robots.producer_for(recorder_record_episode) is not None
        return record.shown_recorder(recorder_status, chosen_has_one)

    tasks.append(
        asyncio.create_task(
            record.run_recorder_buttons(
                node_runner,
                action_module=recorder_record_episode,
                finish_module=recorder_finish_session,
                status=recorder_status,
                session=session,
                robots=robots,
                settings=settings,
                token=token,
                read_task=task_label.for_episode,
            )
        )
    )

    # Shared with the status panel, which reports the same pairs the streams
    # command.
    def slot_peer(topic_module):
        return lambda: robots.peer_for(topic_module)

    hands = tuple(
        panel.HandSource(
            handedness=handedness,
            clutch=HandClutch(settings.motion_scale),
            state=driver.hands[handedness],
            measured=MeasuredPoses(),
            arm_peer=slot_peer(wiring.pose_setpoints),
            gripper_peer=slot_peer(wiring.gripper_setpoints),
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
                    recorder=recorder_row,
                    read_task=task_label.current,
                    posture=posture_status,
                    robot=lambda: driver.robot_line(robots.chosen()),
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
                    gripper_module=wiring.gripper_setpoints,
                    handedness=handedness,
                    hand=driver.hands[handedness],
                    driver=driver,
                    clutch=clutches[handedness],
                    session=session,
                    measured=measured[handedness],
                    robots=robots,
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
                    hand=driver.hands[handedness],
                    session=session,
                    robots=robots,
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
