"""Which robot, limb, base, camera and lidar a pair names, and what that
decides.

Every robot's limbs reach the engine on two slots, its base on a third, its
cameras on two more and its lidars on a sixth, each holding any number of
pairs. The copy a pair carries is the robot that attached under that name,
the link it comes from on the robot's side is its limb or its base, and its
relay's name in the copy is its camera or its lidar. These cover the ways
that is read: a setpoint arriving on a pair, a state, a frame or a scan
published back on one, and what a robot holds.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from robot_description_py import model as bundled
from sim_robot_core.cameras import DEPTH_TO_COLOR, pinhole
from sim_robot_core.models import shipped_entry
from sim_robot_core.object_state import RobotRecord
from sim_robot_core.pairs import Held
from sim_robot_core.registry import Caller, Registry
from sim_robot_core.setpoints import MAX_VALUE, GripperSetpoint, JointSetpoint

import runtime_fakes
from runtime_fakes import MessageDepthToColor, member, pair, queue, topic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import sim_topics  # noqa: E402  (needs the runtime fakes)

# One backbone leads every limb pair of its robot, each from the link named
# after the limb.
ALPHA_LEFT_ARM = member("alpha_backbone_inst", "left_arm", "alpha")
ALPHA_RIGHT_ARM = member("alpha_backbone_inst", "right_arm", "alpha")
ALPHA_LEFT_GRIPPER = member("alpha_backbone_inst", "left_gripper", "alpha")
ALPHA_RIGHT_GRIPPER = member("alpha_backbone_inst", "right_gripper", "alpha")
BRAVO_LEFT_ARM = member("bravo_backbone_inst", "left_arm", "bravo")
CHARLO_ARM = member("charlo_backbone_inst", "arm", "charlo")
CHARLO_GRIPPER = member("charlo_backbone_inst", "gripper", "charlo")
# A camera pair comes from its relay, named after the camera in the copy.
ALPHA_WRIST_LEFT = member("alpha_wrist_left", "simulation", "alpha")
ALPHA_WRIST_RIGHT = member("alpha_wrist_right", "simulation", "alpha")
ALPHA_CHEST = member("alpha_chest", "simulation", "alpha")
CHARLO_WRIST = member("charlo_wrist", "simulation", "charlo")
# A base node leads its robot's wheels from the link named after the base.
INDIA_BASE = member("india_base_inst", "base", "india")
# A lidar pair comes from its relay, named after the lidar in the copy.
INDIA_LIDAR = member("india_lidar", "simulation", "india")
JULIET_LIDAR = member("juliet_lidar", "simulation", "juliet")
# An OpenArm's arm moves seven joints.
OPENARM_ARM = 7


@pytest.fixture(name="scene")
def scene_fixture():
    runtime_fakes.reset()
    loop = asyncio.new_event_loop()
    robots = Registry()
    io = sim_topics.SimTopicIO(node_runner=runtime_fakes.NodeRunner(), loop=loop, robots=robots)
    yield SimpleNamespace(io=io, robots=robots, loop=loop)
    loop.close()


def _admit(scene, name: str, model: str) -> None:
    entry = shipped_entry(model, bundled(model).description())
    scene.robots.admit(name, entry, Caller("sim16", f"{name}_init"))


def _arm_setpoint(sender, positions, velocities=()):
    return sender.info, SimpleNamespace(positions=list(positions), velocities=list(velocities))


def _arm(positions, velocities=()) -> JointSetpoint:
    """The setpoint the engine takes from these positions and velocities."""
    return JointSetpoint(positions=tuple(positions), velocities=tuple(velocities))


def _gripper_setpoint(sender, opening, max_effort=0.0):
    return sender.info, SimpleNamespace(opening=opening, max_effort=max_effort)


async def _drain(loop_turns: int = 10) -> None:
    """Let scheduled callbacks and done-callbacks run: pure scheduling, no
    wall-clock dependence."""
    for _ in range(loop_turns):
        await asyncio.sleep(0)


def _serve(scene, arms=(), grippers=(), bases=()) -> None:
    """Starts the transport with these setpoints queued on the two limb
    slots and the base slot, waits until each consume task took every
    arrival of its subscription, and stops the service that runs for the
    node's life."""
    ended = [
        queue("arms", "joint_setpoints", arms),
        queue("grippers", "gripper_setpoints", grippers),
        queue("bases", "joint_setpoints", bases),
    ]

    async def serve() -> None:
        await scene.io.start()
        await asyncio.gather(*(event.wait() for event in ended))
        await scene.io.stop()

    scene.loop.run_until_complete(serve())


def _publishes(scene) -> None:
    """Runs what the physics and render threads handed to the node loop."""
    scene.loop.run_until_complete(_drain())


def _sent(slot: str, name: str) -> list:
    return topic(slot, name).publisher.sent


class TestWhichRobotAPairBelongsTo:
    def test_a_pair_belongs_to_the_copy_it_carries(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        _admit(scene, "bravo", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM, BRAVO_LEFT_ARM)

        _serve(
            scene,
            arms=[
                _arm_setpoint(BRAVO_LEFT_ARM, [0.9] * OPENARM_ARM),
                _arm_setpoint(ALPHA_LEFT_ARM, [0.5] * OPENARM_ARM),
            ],
        )

        assert scene.io.setpoints.arm("alpha", "left_arm") == _arm([0.5] * OPENARM_ARM)
        assert scene.io.setpoints.arm("bravo", "left_arm") == _arm([0.9] * OPENARM_ARM)

    def test_a_pair_with_no_copy_belongs_to_the_only_robot_in_the_scene(self, scene):
        _admit(scene, "solo", "so101")
        backbone = member("backbone_inst", "arm", None)
        pair("arms", backbone)

        _serve(scene, arms=[_arm_setpoint(backbone, [0.1] * 5)])

        assert scene.io.setpoints.arm("solo", "arm") == _arm([0.1] * 5)
        assert scene.io.held_by("solo") == Held(arms=frozenset({"arm"}))

    def test_a_pair_with_no_copy_names_no_robot_in_a_fleet(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        _admit(scene, "bravo", "openarm_v2")
        backbone = member("backbone_inst", "left_arm", None)
        pair("arms", backbone)

        _serve(scene, arms=[_arm_setpoint(backbone, [0.5] * OPENARM_ARM)])

        for robot in ("alpha", "bravo"):
            assert scene.io.setpoints.arm(robot, "left_arm") is None
            assert scene.io.held_by(robot).is_empty()

    def test_a_peer_the_slot_does_not_hold_belongs_to_no_robot(self, scene):
        _admit(scene, "alpha", "openarm_v2")

        _serve(scene, arms=[_arm_setpoint(ALPHA_LEFT_ARM, [0.5] * OPENARM_ARM)])

        assert scene.io.setpoints.arm("alpha", "left_arm") is None


class TestLimbsOfOneSlot:
    def test_each_arm_of_one_backbone_takes_its_own_setpoint(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM, ALPHA_RIGHT_ARM)

        _serve(
            scene,
            arms=[
                _arm_setpoint(ALPHA_LEFT_ARM, [0.1] * 7, [1.0] * 7),
                _arm_setpoint(ALPHA_RIGHT_ARM, [0.2] * 7, [2.0] * 7),
            ],
        )

        assert scene.io.setpoints.arm("alpha", "left_arm") == _arm([0.1] * 7, [1.0] * 7)
        assert scene.io.setpoints.arm("alpha", "right_arm") == _arm([0.2] * 7, [2.0] * 7)

    def test_each_arms_state_goes_back_to_the_peer_of_the_same_limb_only(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM, ALPHA_RIGHT_ARM)
        _serve(scene)

        scene.io.publish_arm_states("alpha", "left_arm", [0.1] * 7, [1.0] * 7)
        scene.io.publish_arm_states("alpha", "right_arm", [0.2] * 7, [2.0] * 7)
        _publishes(scene)

        # The engine measures no joint torques, so efforts ride empty.
        assert _sent("arms", "joint_states") == [
            (ALPHA_LEFT_ARM.info, (0.0, [0.1] * 7, [1.0] * 7, [])),
            (ALPHA_RIGHT_ARM.info, (0.0, [0.2] * 7, [2.0] * 7, [])),
        ]

    def test_each_gripper_of_one_backbone_is_driven_and_answered_on_its_own_pair(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        pair("grippers", ALPHA_LEFT_GRIPPER, ALPHA_RIGHT_GRIPPER)

        _serve(
            scene,
            grippers=[
                _gripper_setpoint(ALPHA_RIGHT_GRIPPER, 0.75, max_effort=4.0),
                _gripper_setpoint(ALPHA_LEFT_GRIPPER, 0.25),
            ],
        )
        scene.io.publish_gripper_states("alpha", "right_gripper", 0.7)
        _publishes(scene)

        assert scene.io.setpoints.gripper("alpha", "left_gripper") == GripperSetpoint(0.25, 0.0)
        assert scene.io.setpoints.gripper("alpha", "right_gripper") == GripperSetpoint(0.75, 4.0)
        assert _sent("grippers", "gripper_states") == [
            (ALPHA_RIGHT_GRIPPER.info, (0.0, 0.7, 0.0, 0.0))
        ]

    def test_a_limb_the_robot_holds_no_pair_for_publishes_nothing(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM)
        _serve(scene)

        scene.io.publish_arm_states("alpha", "right_arm", [0.2] * 7, [0.0] * 7)
        _publishes(scene)

        assert _sent("arms", "joint_states") == []

    def test_a_state_whose_pair_ended_before_its_publish_is_dropped(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM)
        _serve(scene)

        scene.io.publish_arm_states("alpha", "left_arm", [0.1] * 7, [0.0] * 7)
        pair("arms")
        _publishes(scene)

        assert _sent("arms", "joint_states") == []


class TestRobotsOfDifferentModels:
    @pytest.fixture(name="fleet")
    def fleet_fixture(self, scene):
        """An OpenArm and an SO-101 on the same slots."""
        _admit(scene, "alpha", "openarm_v2")
        _admit(scene, "charlo", "so101")
        pair("arms", ALPHA_LEFT_ARM, ALPHA_RIGHT_ARM, CHARLO_ARM)
        pair("grippers", ALPHA_LEFT_GRIPPER, ALPHA_RIGHT_GRIPPER, CHARLO_GRIPPER)
        return scene

    def test_each_robots_setpoints_reach_its_own_limbs(self, fleet):
        _serve(
            fleet,
            arms=[
                _arm_setpoint(CHARLO_ARM, [0.3] * 5),
                _arm_setpoint(ALPHA_LEFT_ARM, [0.1] * 7),
            ],
            grippers=[_gripper_setpoint(CHARLO_GRIPPER, 1.0)],
        )

        assert fleet.io.setpoints.arm("charlo", "arm") == _arm([0.3] * 5)
        assert fleet.io.setpoints.gripper("charlo", "gripper") == GripperSetpoint(1.0, 0.0)
        assert fleet.io.setpoints.arm("alpha", "left_arm") == _arm([0.1] * 7)
        # A limb belongs to the robot that names it: neither answers to the
        # other's.
        assert fleet.io.setpoints.arm("alpha", "arm") is None
        assert fleet.io.setpoints.arm("charlo", "left_arm") is None
        assert fleet.io.setpoints.gripper("alpha", "gripper") is None

    def test_each_robots_state_goes_back_on_its_own_pairs(self, fleet):
        _serve(fleet)

        fleet.io.publish_arm_states("charlo", "arm", [0.3] * 5, [0.0] * 5)
        fleet.io.publish_gripper_states("charlo", "gripper", 0.5)
        fleet.io.publish_arm_states("alpha", "right_arm", [0.2] * 7, [0.0] * 7)
        _publishes(fleet)

        assert _sent("arms", "joint_states") == [
            (CHARLO_ARM.info, (0.0, [0.3] * 5, [0.0] * 5, [])),
            (ALPHA_RIGHT_ARM.info, (0.0, [0.2] * 7, [0.0] * 7, [])),
        ]
        assert _sent("grippers", "gripper_states") == [(CHARLO_GRIPPER.info, (0.0, 0.5, 0.0, 0.0))]

    def test_each_robot_holds_its_own_pairs(self, fleet):
        pair("rgb_cameras", CHARLO_WRIST, ALPHA_WRIST_LEFT)
        pair("rgbd_cameras", ALPHA_CHEST)

        assert fleet.io.held_by("charlo") == Held(
            arms=frozenset({"arm"}),
            grippers=frozenset({"gripper"}),
            rgb_cameras=frozenset({"wrist"}),
        )
        assert fleet.io.held_by("alpha") == Held(
            arms=frozenset({"left_arm", "right_arm"}),
            grippers=frozenset({"left_gripper", "right_gripper"}),
            rgb_cameras=frozenset({"wrist_left"}),
            rgbd_cameras=frozenset({"chest"}),
        )
        assert fleet.io.held_by("bravo").is_empty()


class TestCameras:
    @pytest.fixture(name="rig")
    def rig_fixture(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        _admit(scene, "charlo", "so101")
        pair("rgb_cameras", CHARLO_WRIST, ALPHA_WRIST_LEFT, ALPHA_WRIST_RIGHT)
        pair("rgbd_cameras", ALPHA_CHEST)
        _serve(scene)
        return scene

    def test_a_color_frame_goes_to_the_relay_named_after_the_camera(self, rig):
        assert rig.io.publish_color_frame("charlo", "wrist", 1.5, "wrist_optical_frame", 7, "rgb8", 4, 2, b"wrist")
        _publishes(rig)

        ((peer, payload),) = _sent("rgb_cameras", "video_stream")
        header, *frame = payload
        assert peer == CHARLO_WRIST.info
        assert (header.timestamp, header.frame, header.capture_index) == (
            1.5,
            "wrist_optical_frame",
            7,
        )
        assert frame == ["rgb8", 4, 2, b"wrist"]

    def test_a_color_stream_is_described_to_its_own_relay(self, rig):
        rig.io.publish_color_stream_info("alpha", "wrist_right", 960, 600, 15, "rgb8")
        _publishes(rig)

        assert _sent("rgb_cameras", "stream_info") == [
            (ALPHA_WRIST_RIGHT.info, (960, 600, 15, "rgb8"))
        ]

    def test_an_rgbd_capture_goes_whole_to_its_relay_on_the_rgbd_slot(self, rig):
        assert rig.io.publish_rgbd_frames(
            "alpha",
            "chest",
            2.5,
            "chest_optical_frame",
            3,
            "depth_to_color",
            ("rgb8", 4, 2, b"color"),
            ("z16", 2, 1, b"depth"),
        )
        rig.io.publish_rgbd_stream_info("alpha", "chest", 4, 2, 15, "rgb8", 2, 1, "z16", 0.001)
        _publishes(rig)

        ((color_peer, color),) = _sent("rgbd_cameras", "video_stream")
        ((depth_peer, depth),) = _sent("rgbd_cameras", "depth_stream")
        assert color_peer == depth_peer == ALPHA_CHEST.info
        for header in (color[0], depth[0]):
            assert (header.timestamp, header.frame, header.capture_index, header.align_mode) == (
                2.5,
                "chest_optical_frame",
                3,
                "depth_to_color",
            )
        assert list(color[1:]) == ["rgb8", 4, 2, b"color"]
        assert list(depth[1:]) == ["z16", 2, 1, b"depth"]
        assert _sent("rgbd_cameras", "stream_info") == [
            (ALPHA_CHEST.info, (4, 2, 15, "rgb8", 2, 1, "z16", 0.001))
        ]
        assert _sent("rgb_cameras", "video_stream") == []

    def test_an_rgbd_geometry_gives_its_relay_the_depth_to_color_pose_with_no_timestamp(self, rig):
        rig.io.publish_rgbd_geometry(
            "alpha",
            "chest",
            pinhole(58.0, 4, 2),
            pinhole(58.0, 2, 1),
            "z",
            0.1,
            10.0,
            "depth_to_color",
            "chest_optical_frame",
            DEPTH_TO_COLOR,
        )
        _publishes(rig)

        ((peer, geometry),) = _sent("rgbd_cameras", "geometry")
        assert peer == ALPHA_CHEST.info
        assert geometry.depth_to_color == MessageDepthToColor(
            frame="chest_optical_frame",
            position=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
        )

    def test_a_camera_is_its_robots_on_the_slot_of_its_kind(self, rig):
        """A camera nobody paired is delivered to nobody: another robot's
        relay of the same slot does not stand in, and neither does the same
        robot's relay on the other slot."""
        assert not rig.io.publish_color_frame("charlo", "wrist_left", 1.0, "wrist_left_optical_frame", 0, "rgb8", 4, 2, b"")
        assert not rig.io.publish_color_frame("alpha", "wrist", 1.0, "wrist_optical_frame", 0, "rgb8", 4, 2, b"")
        assert not rig.io.publish_color_frame("alpha", "chest", 1.0, "chest_optical_frame", 0, "rgb8", 4, 2, b"")
        _publishes(rig)

        assert _sent("rgb_cameras", "video_stream") == []

    def test_a_frame_behind_one_still_in_flight_is_dropped_for_that_camera_alone(self, rig):
        assert rig.io.publish_color_frame("alpha", "wrist_left", 1.0, "wrist_left_optical_frame", 0, "rgb8", 4, 2, b"first")
        # The loop has not run, so the first frame is still in flight.
        assert not rig.io.publish_color_frame("alpha", "wrist_left", 1.1, "wrist_left_optical_frame", 1, "rgb8", 4, 2, b"second")
        assert rig.io.publish_color_frame("alpha", "wrist_right", 1.1, "wrist_right_optical_frame", 0, "rgb8", 4, 2, b"other")
        _publishes(rig)

        assert rig.io.publish_color_frame("alpha", "wrist_left", 1.2, "wrist_left_optical_frame", 2, "rgb8", 4, 2, b"third")
        _publishes(rig)
        delivered = [(peer, payload[-1]) for peer, payload in _sent("rgb_cameras", "video_stream")]
        assert delivered == [
            (ALPHA_WRIST_LEFT.info, b"first"),
            (ALPHA_WRIST_RIGHT.info, b"other"),
            (ALPHA_WRIST_LEFT.info, b"third"),
        ]


class TestLidars:
    @pytest.fixture(name="bases")
    def bases_fixture(self, scene):
        """Two botshare bases, india's lidar paired and juliet's not."""
        _admit(scene, "india", "botshare_base")
        _admit(scene, "juliet", "botshare_base")
        pair("lidars", INDIA_LIDAR)
        _serve(scene)
        return scene

    @staticmethod
    def _lidar(scene, robot: str):
        (lidar,) = scene.robots.of_name(robot).entry.lidars
        return lidar

    def test_a_scan_goes_to_the_relay_named_after_the_lidar_naming_its_frame(self, bases):
        lidar = self._lidar(bases, "india")
        assert bases.io.publish_scan("india", lidar, 1.5, 7, [2.0, 10000.0, 0.0])
        _publishes(bases)

        ((peer, payload),) = _sent("lidars", "scan_stream")
        header, *scan = payload
        assert peer == INDIA_LIDAR.info
        assert (header.timestamp, header.frame, header.capture_index) == (1.5, "lidar_frame", 7)
        assert scan == [
            lidar.sweep.min_angle_rad,
            lidar.sweep.angle_increment_rad,
            lidar.time_increment_s,
            lidar.min_range_m,
            lidar.max_range_m,
            [2.0, 10000.0, 0.0],
            [],
        ]

    def test_a_lidar_is_its_robots_on_the_lidar_slot(self, bases):
        """The lidar of a robot that holds no pair for it is viewed by
        nobody: another robot's relay does not stand in."""
        assert bases.io.held_by("india") == Held(lidars=frozenset({"lidar"}))
        assert bases.io.held_by("juliet").is_empty()
        assert bases.io.lidar_viewed("india", "lidar")
        assert not bases.io.lidar_viewed("juliet", "lidar")
        assert not bases.io.publish_scan("juliet", self._lidar(bases, "juliet"), 1.5, 0, [2.0])
        _publishes(bases)

        assert [peer for peer, _ in _sent("lidars", "scan_stream")] == []

    def test_a_scan_behind_one_still_in_flight_is_dropped(self, bases):
        lidar = self._lidar(bases, "india")
        assert bases.io.publish_scan("india", lidar, 1.0, 0, [1.0])
        # The loop has not run, so the first scan is still in flight.
        assert not bases.io.publish_scan("india", lidar, 1.1, 1, [1.1])
        _publishes(bases)
        assert bases.io.publish_scan("india", lidar, 1.2, 2, [1.2])
        _publishes(bases)

        assert [payload[0].capture_index for _, payload in _sent("lidars", "scan_stream")] == [0, 2]


class TestSetpoints:
    def test_the_latest_setpoint_of_a_limb_wins(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM)

        _serve(
            scene,
            arms=[
                _arm_setpoint(ALPHA_LEFT_ARM, [0.1] * OPENARM_ARM),
                _arm_setpoint(ALPHA_LEFT_ARM, [0.4] * OPENARM_ARM),
            ],
        )

        assert scene.io.setpoints.arm("alpha", "left_arm") == _arm([0.4] * OPENARM_ARM)

    @pytest.mark.parametrize(
        ("positions", "velocities"),
        [
            ([float("nan")] * OPENARM_ARM, [0.0] * OPENARM_ARM),
            ([0.0] * OPENARM_ARM, [float("inf")] * OPENARM_ARM),
            ([10 * MAX_VALUE] * OPENARM_ARM, []),
            ([0.0] * OPENARM_ARM, [-10 * MAX_VALUE] * OPENARM_ARM),
            ([0.0] * 5, []),
            ([0.0] * OPENARM_ARM, [0.0] * 5),
        ],
    )
    def test_an_arm_setpoint_the_simulation_cannot_hold_never_reaches_the_scene(
        self, scene, positions, velocities
    ):
        """One that is not a number, one past what a simulation holds, and one
        of another joint count than the arm's are dropped as they arrive."""
        _admit(scene, "alpha", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM)

        _serve(scene, arms=[_arm_setpoint(ALPHA_LEFT_ARM, positions, velocities)])

        assert scene.io.setpoints.arm("alpha", "left_arm") is None

    @pytest.mark.parametrize(
        ("opening", "max_effort"),
        [(float("nan"), 0.0), (0.5, float("inf")), (0.5, -1.0)],
    )
    def test_an_unusable_gripper_setpoint_never_reaches_the_scene(self, scene, opening, max_effort):
        _admit(scene, "charlo", "so101")
        pair("grippers", CHARLO_GRIPPER)

        _serve(scene, grippers=[_gripper_setpoint(CHARLO_GRIPPER, opening, max_effort)])

        assert scene.io.setpoints.gripper("charlo", "gripper") is None

    def test_a_robot_that_left_takes_its_setpoints_with_it(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        _admit(scene, "bravo", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM, BRAVO_LEFT_ARM)
        pair("grippers", ALPHA_LEFT_GRIPPER)
        _serve(
            scene,
            arms=[
                _arm_setpoint(ALPHA_LEFT_ARM, [0.5] * OPENARM_ARM),
                _arm_setpoint(BRAVO_LEFT_ARM, [0.9] * OPENARM_ARM),
            ],
            grippers=[_gripper_setpoint(ALPHA_LEFT_GRIPPER, 0.25)],
        )

        scene.io.setpoints.forget("alpha")

        assert scene.io.setpoints.arm("alpha", "left_arm") is None
        assert scene.io.setpoints.gripper("alpha", "left_gripper") is None
        assert scene.io.setpoints.arm("bravo", "left_arm") == _arm([0.9] * OPENARM_ARM)


class TestTheBase:
    def test_a_base_setpoint_reaches_the_base_of_the_copy_it_carries(self, scene):
        _admit(scene, "india", "botshare_base")
        pair("bases", INDIA_BASE)

        _serve(scene, bases=[_arm_setpoint(INDIA_BASE, [0.0, 0.0], [1.0, 1.5])])

        assert scene.io.setpoints.base("india", 0.0) == JointSetpoint((0.0, 0.0), (1.0, 1.5))
        assert scene.io.held_by("india") == Held(bases=frozenset({"base"}))

    @pytest.mark.parametrize(
        ("positions", "velocities"),
        [
            ([0.0], [1.0]),
            ([0.0, 0.0], [1.0]),
            ([0.0, 0.0], []),
            ([0.0, 0.0], [float("nan"), 0.0]),
            ([0.0, 0.0], [0.0, 10 * MAX_VALUE]),
        ],
    )
    def test_a_base_setpoint_the_simulation_cannot_hold_never_reaches_the_wheels(
        self, scene, positions, velocities
    ):
        _admit(scene, "india", "botshare_base")
        pair("bases", INDIA_BASE)

        _serve(scene, bases=[_arm_setpoint(INDIA_BASE, positions, velocities)])

        assert scene.io.setpoints.base("india", 0.0) is None

    def test_a_base_pair_of_a_robot_with_no_base_drives_nothing(self, scene):
        _admit(scene, "charlo", "so101")
        stray = member("charlo_base_inst", "base", "charlo")
        pair("bases", stray)

        _serve(scene, bases=[_arm_setpoint(stray, [0.0, 0.0], [1.0, 1.0])])

        assert scene.io.setpoints.base("charlo", 0.0) is None

    def test_the_wheels_hold_at_rate_0_once_a_reading_finds_their_pair_gone(self, scene):
        _admit(scene, "india", "botshare_base")
        pair("bases", INDIA_BASE)
        _serve(scene, bases=[_arm_setpoint(INDIA_BASE, [0.0, 0.0], [1.0, 1.5])])

        scene.io.setpoints.keep_paired_bases()
        assert scene.io.setpoints.base("india", 0.0) is not None
        pair("bases")
        scene.io.setpoints.keep_paired_bases()

        assert scene.io.setpoints.base("india", 0.0) is None
        assert scene.io.setpoints.take_base_release("india")

    def test_a_base_pair_that_comes_back_drives_nothing_until_its_leader_sends(self, scene):
        _admit(scene, "india", "botshare_base")
        pair("bases", INDIA_BASE)
        _serve(scene, bases=[_arm_setpoint(INDIA_BASE, [0.0, 0.0], [1.0, 1.5])])

        # The base node came back on a pair of its own between two readings.
        back = member("india_base_inst", "base", "india", core_node="sim17")
        pair("bases", back)
        scene.io.setpoints.keep_paired_bases()
        assert scene.io.setpoints.base("india", 0.0) is None
        assert scene.io.setpoints.take_base_release("india"), "its wheels are held where they stand"

        _serve(scene, bases=[_arm_setpoint(back, [0.0, 0.0], [0.5, 0.5])])
        scene.io.setpoints.keep_paired_bases()
        assert scene.io.setpoints.base("india", 0.0) == JointSetpoint((0.0, 0.0), (0.5, 0.5))

    def test_a_robot_that_left_takes_its_base_setpoint_with_it(self, scene):
        _admit(scene, "india", "botshare_base")
        pair("bases", INDIA_BASE)
        _serve(scene, bases=[_arm_setpoint(INDIA_BASE, [0.0, 0.0], [1.0, 1.5])])

        scene.io.setpoints.forget("india")

        assert scene.io.setpoints.base("india", 0.0) is None

    def test_the_wheel_state_goes_back_on_the_base_pair_of_its_robot(self, scene):
        _admit(scene, "india", "botshare_base")
        pair("bases", INDIA_BASE)
        _serve(scene)

        scene.io.publish_base_states("india", [0.1, 0.2], [1.0, 1.5])
        _publishes(scene)

        assert _sent("bases", "joint_states") == [(INDIA_BASE.info, (0.0, [0.1, 0.2], [1.0, 1.5], []))]


class TestObjectState:
    """The robots' records, as the bridge captures them on each telemetry
    tick: the stream publishes each capture in world, and get_object_states
    answers the latest one."""

    RECORD = RobotRecord.fixed("alpha", "openarm_v2", (0.0, -1.5, 0.0), (0.0, 0.0, 0.0, 1.0))

    @pytest.fixture(name="standing")
    def standing_fixture(self, scene):
        """alpha, stood: its attach goal reports it standing."""
        _admit(scene, "alpha", "openarm_v2")
        scene.robots.stand("alpha")
        return scene

    def test_a_capture_is_published_in_world_under_its_stamp_with_no_object(self, standing):
        scene = standing
        _serve(scene)

        scene.io.publish_object_states([self.RECORD])
        _publishes(scene)

        stream = sys.modules["peppygen.emitted_topics.objects.object_states"]
        assert stream.publisher.sent == [(None, ("world", 0.0, [], [self.RECORD.fields()]))]

    def test_get_object_states_answers_the_latest_capture(self, standing):
        scene = standing
        _serve(scene)
        scene.io.publish_object_states([])
        scene.io.publish_object_states([self.RECORD])

        answer = scene.io._answer_object_states(None)  # pylint: disable=W0212

        assert (answer.frame, answer.timestamp, answer.objects) == ("world", 0.0, [])
        assert answer.robots == [self.RECORD.fields()]

    def test_a_robot_lists_from_the_feedback_that_reports_it_standing(self, scene):
        """The scene composes a joining robot in before its attach goal
        reports it standing."""
        _admit(scene, "alpha", "openarm_v2")
        _serve(scene)

        scene.io.publish_object_states([self.RECORD])
        assert scene.io._answer_object_states(None).robots == []  # pylint: disable=W0212
        scene.robots.stand("alpha")
        scene.io.publish_object_states([self.RECORD])
        assert scene.io._answer_object_states(None).robots == [self.RECORD.fields()]  # pylint: disable=W0212

    def test_no_capture_yet_is_an_error_reply_and_not_an_empty_scene(self, scene):
        with pytest.raises(runtime_fakes.ServiceFailure) as refusal:
            scene.io._answer_object_states(None)  # pylint: disable=W0212
        assert refusal.value.kind == "precondition_failed"
        assert "MuJoCo has not captured its object state yet" in refusal.value.message
