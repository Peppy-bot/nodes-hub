"""Which robot, limb and camera a pair names, and what that decides.

Every robot's limbs reach the engine on two slots and its cameras on two more,
each holding any number of pairs. The copy a pair carries is the robot that
attached under that name, the link it comes from on the robot's side is its
limb, and its relay's name in the copy is its camera. These cover the ways
that is read: a setpoint arriving on a pair, a state or a frame published
back on one, what a robot holds, and which robots a rig is rendered for.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pyjson5
import pytest
from sim_robot_core.models import shipped_entry
from sim_robot_core.pairs import ARMS, GRIPPERS, RGB_CAMERAS, RGBD_CAMERAS, Held
from sim_robot_core.registry import Caller, Registry

import runtime_fakes
from runtime_fakes import member, pair, topic

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
CHARLO_FRONT = member("charlo_front", "simulation", "charlo")


@pytest.fixture(name="scene")
def scene_fixture():
    runtime_fakes.reset()
    loop = asyncio.new_event_loop()
    robots = Registry()
    io = sim_topics.SimTopicIO(node_runner=object(), loop=loop, robots=robots)
    yield SimpleNamespace(io=io, robots=robots, loop=loop)
    loop.close()


def _admit(scene, name: str, model: str) -> None:
    scene.robots.admit(name, shipped_entry(model), Caller("sim16", f"{name}_init"), 0.0)


def _arm_setpoint(sender, positions, velocities=()):
    return sender.info, SimpleNamespace(positions=list(positions), velocities=list(velocities))


def _gripper_setpoint(sender, opening, max_effort=0.0):
    return sender.info, SimpleNamespace(opening=opening, max_effort=max_effort)


async def _drain(loop_turns: int = 10) -> None:
    """Let scheduled callbacks and done-callbacks run: pure scheduling, no
    wall-clock dependence."""
    for _ in range(loop_turns):
        await asyncio.sleep(0)


def _serve(scene, arms=(), grippers=()) -> None:
    """Starts the transport with these setpoints queued on the two limb
    slots, and runs each consume task to the end of its subscription."""
    topic("arms", "joint_setpoints").arrivals = list(arms)
    topic("grippers", "gripper_setpoints").arrivals = list(grippers)

    async def serve() -> None:
        await scene.io.start()
        await asyncio.gather(*scene.io._tasks)  # pylint: disable=W0212

    scene.loop.run_until_complete(serve())


def _publishes(scene) -> None:
    """Runs what the physics and render threads handed to the node loop."""
    scene.loop.run_until_complete(_drain())


def _sent(slot: str, name: str) -> list:
    return topic(slot, name).publisher.sent


class TestTheNodesSlots:
    @pytest.fixture(name="node")
    def node_fixture(self):
        return pyjson5.loads((Path(__file__).resolve().parents[1] / "peppy.json5").read_text())

    def test_the_node_declares_the_four_slots_every_pair_is_read_from(self, node):
        assert node["manifest"]["name"] == "sim_isaac"
        pairings = {
            pairing["link_id"]: (pairing["name"], pairing["role"], pairing["cardinality"])
            for pairing in node["manifest"]["depends_on"]["pairings"]
        }
        assert pairings == {
            ARMS: ("joint_link", "follower", "zero_or_more"),
            GRIPPERS: ("gripper_link", "follower", "zero_or_more"),
            RGB_CAMERAS: ("sim_rgb_camera_link", "camera", "zero_or_more"),
            RGBD_CAMERAS: ("sim_rgbd_camera_link", "camera", "zero_or_more"),
        }

    def test_the_faked_runtime_carries_the_topics_the_node_declares_on_them(self, node):
        topics = node["interfaces"]["topics"]
        declared: dict[str, set] = {}
        for entry in (*topics["emits"], *topics["consumes"]):
            if entry["link_id"] in runtime_fakes.SLOT_TOPICS:
                declared.setdefault(entry["link_id"], set()).add(entry["name"])
        assert declared == {
            slot: set(names) for slot, names in runtime_fakes.SLOT_TOPICS.items()
        }


class TestWhichRobotAPairBelongsTo:
    def test_a_pair_belongs_to_the_copy_it_carries(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        _admit(scene, "bravo", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM, BRAVO_LEFT_ARM)

        _serve(scene, arms=[_arm_setpoint(BRAVO_LEFT_ARM, [0.9]), _arm_setpoint(ALPHA_LEFT_ARM, [0.5])])

        assert scene.io.latest_arm_command("alpha", "left_arm") == ([0.5], [])
        assert scene.io.latest_arm_command("bravo", "left_arm") == ([0.9], [])

    def test_a_pair_with_no_copy_belongs_to_the_only_robot_in_the_scene(self, scene):
        _admit(scene, "solo", "so101")
        backbone = member("backbone_inst", "arm", None)
        pair("arms", backbone)

        _serve(scene, arms=[_arm_setpoint(backbone, [0.1] * 5)])

        assert scene.io.latest_arm_command("solo", "arm") == ([0.1] * 5, [])
        assert scene.io.held_by("solo") == Held(arms=frozenset({"arm"}))

    def test_a_pair_with_no_copy_names_no_robot_in_a_fleet(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        _admit(scene, "bravo", "openarm_v2")
        backbone = member("backbone_inst", "left_arm", None)
        pair("arms", backbone)

        _serve(scene, arms=[_arm_setpoint(backbone, [0.5])])

        for robot in ("alpha", "bravo"):
            assert scene.io.latest_arm_command(robot, "left_arm") is None
            assert scene.io.held_by(robot).is_empty()

    def test_a_peer_the_slot_does_not_hold_belongs_to_no_robot(self, scene):
        _admit(scene, "alpha", "openarm_v2")

        _serve(scene, arms=[_arm_setpoint(ALPHA_LEFT_ARM, [0.5])])

        assert scene.io.latest_arm_command("alpha", "left_arm") is None


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

        assert scene.io.latest_arm_command("alpha", "left_arm") == ([0.1] * 7, [1.0] * 7)
        assert scene.io.latest_arm_command("alpha", "right_arm") == ([0.2] * 7, [2.0] * 7)

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

        assert scene.io.latest_gripper_command("alpha", "left_gripper") == (0.25, 0.0)
        assert scene.io.latest_gripper_command("alpha", "right_gripper") == (0.75, 4.0)
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

        assert fleet.io.latest_arm_command("charlo", "arm") == ([0.3] * 5, [])
        assert fleet.io.latest_gripper_command("charlo", "gripper") == (1.0, 0.0)
        assert fleet.io.latest_arm_command("alpha", "left_arm") == ([0.1] * 7, [])
        # A limb belongs to the robot that names it: neither answers to the
        # other's.
        assert fleet.io.latest_arm_command("alpha", "arm") is None
        assert fleet.io.latest_arm_command("charlo", "left_arm") is None
        assert fleet.io.latest_gripper_command("alpha", "gripper") is None

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

    def test_a_rig_is_rendered_for_each_robot_holding_a_camera_pair(self, fleet):
        """Whatever its model and whichever slot its camera pairs on, and for
        no robot that pairs limbs alone."""
        assert fleet.io.camera_robots() == set()

        pair("rgb_cameras", CHARLO_FRONT)
        assert fleet.io.camera_robots() == {"charlo"}

        pair("rgbd_cameras", ALPHA_CHEST)
        assert fleet.io.camera_robots() == {"alpha", "charlo"}

        pair("rgb_cameras")
        assert fleet.io.camera_robots() == {"alpha"}

    def test_a_camera_pair_with_no_copy_is_the_only_robots(self, scene):
        _admit(scene, "solo", "so101")
        pair("rgb_cameras", member("front", "simulation", None))

        assert scene.io.camera_robots() == {"solo"}
        assert scene.io.held_by("solo") == Held(rgb_cameras=frozenset({"front"}))

    def test_each_robot_holds_its_own_pairs(self, fleet):
        pair("rgb_cameras", CHARLO_FRONT, ALPHA_WRIST_LEFT)
        pair("rgbd_cameras", ALPHA_CHEST)

        assert fleet.io.held_by("charlo") == Held(
            arms=frozenset({"arm"}),
            grippers=frozenset({"gripper"}),
            rgb_cameras=frozenset({"front"}),
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
        pair("rgb_cameras", CHARLO_FRONT, ALPHA_WRIST_LEFT, ALPHA_WRIST_RIGHT)
        pair("rgbd_cameras", ALPHA_CHEST)
        _serve(scene)
        return scene

    def test_a_color_frame_goes_to_the_relay_named_after_the_camera(self, rig):
        assert rig.io.publish_color_frame("charlo", "front", 1.5, 7, "rgb8", 4, 2, b"front")
        _publishes(rig)

        ((peer, payload),) = _sent("rgb_cameras", "video_stream")
        header, *frame = payload
        assert peer == CHARLO_FRONT.info
        assert (header.timestamp, header.frame_id) == (1.5, 7)
        assert frame == ["rgb8", 4, 2, b"front"]

    def test_a_color_stream_is_described_to_its_own_relay(self, rig):
        rig.io.publish_color_stream_info("alpha", "wrist_right", 960, 600, 15, "rgb8")
        _publishes(rig)

        assert _sent("rgb_cameras", "stream_info") == [
            (ALPHA_WRIST_RIGHT.info, (960, 600, 15, "rgb8"))
        ]

    def test_an_rgbd_capture_goes_whole_to_its_relay_on_the_rgbd_slot(self, rig):
        assert rig.io.publish_rgbd_frames(
            "alpha", "chest", 2.5, 3, "depth_to_color", ("rgb8", 4, 2, b"color"), ("z16", 2, 1, b"depth")
        )
        rig.io.publish_rgbd_stream_info("alpha", "chest", 4, 2, 15, "rgb8", 2, 1, "z16", 0.001)
        _publishes(rig)

        ((color_peer, color),) = _sent("rgbd_cameras", "video_stream")
        ((depth_peer, depth),) = _sent("rgbd_cameras", "depth_stream")
        assert color_peer == depth_peer == ALPHA_CHEST.info
        for header in (color[0], depth[0]):
            assert (header.timestamp, header.frame_id, header.align_mode) == (2.5, 3, "depth_to_color")
        assert list(color[1:]) == ["rgb8", 4, 2, b"color"]
        assert list(depth[1:]) == ["z16", 2, 1, b"depth"]
        assert _sent("rgbd_cameras", "stream_info") == [
            (ALPHA_CHEST.info, (4, 2, 15, "rgb8", 2, 1, "z16", 0.001))
        ]
        assert _sent("rgb_cameras", "video_stream") == []

    def test_a_camera_is_its_robots_on_the_slot_of_its_kind(self, rig):
        """A camera nobody paired is delivered to nobody: another robot's
        relay of the same slot does not stand in, and neither does the same
        robot's relay on the other slot."""
        assert not rig.io.publish_color_frame("charlo", "wrist_left", 1.0, 0, "rgb8", 4, 2, b"")
        assert not rig.io.publish_color_frame("alpha", "front", 1.0, 0, "rgb8", 4, 2, b"")
        assert not rig.io.publish_color_frame("alpha", "chest", 1.0, 0, "rgb8", 4, 2, b"")
        _publishes(rig)

        assert _sent("rgb_cameras", "video_stream") == []

    def test_a_frame_behind_one_still_in_flight_is_dropped_for_that_camera_alone(self, rig):
        assert rig.io.publish_color_frame("alpha", "wrist_left", 1.0, 0, "rgb8", 4, 2, b"first")
        # The loop has not run, so the first frame is still in flight.
        assert not rig.io.publish_color_frame("alpha", "wrist_left", 1.1, 1, "rgb8", 4, 2, b"second")
        assert rig.io.publish_color_frame("alpha", "wrist_right", 1.1, 0, "rgb8", 4, 2, b"other")
        _publishes(rig)

        assert rig.io.publish_color_frame("alpha", "wrist_left", 1.2, 2, "rgb8", 4, 2, b"third")
        _publishes(rig)
        delivered = [(peer, payload[-1]) for peer, payload in _sent("rgb_cameras", "video_stream")]
        assert delivered == [
            (ALPHA_WRIST_LEFT.info, b"first"),
            (ALPHA_WRIST_RIGHT.info, b"other"),
            (ALPHA_WRIST_LEFT.info, b"third"),
        ]


class TestSetpoints:
    def test_the_latest_setpoint_of_a_limb_wins(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM)

        _serve(scene, arms=[_arm_setpoint(ALPHA_LEFT_ARM, [0.1]), _arm_setpoint(ALPHA_LEFT_ARM, [0.4])])

        assert scene.io.latest_arm_command("alpha", "left_arm") == ([0.4], [])

    @pytest.mark.parametrize(
        ("positions", "velocities"),
        [([float("nan")], [0.0]), ([0.0], [float("inf")])],
    )
    def test_a_non_finite_arm_setpoint_never_reaches_the_scene(self, scene, positions, velocities):
        _admit(scene, "alpha", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM)

        _serve(scene, arms=[_arm_setpoint(ALPHA_LEFT_ARM, positions, velocities)])

        assert scene.io.latest_arm_command("alpha", "left_arm") is None

    @pytest.mark.parametrize(
        ("opening", "max_effort"),
        [(float("nan"), 0.0), (0.5, float("inf")), (0.5, -1.0)],
    )
    def test_an_unusable_gripper_setpoint_never_reaches_the_scene(self, scene, opening, max_effort):
        _admit(scene, "charlo", "so101")
        pair("grippers", CHARLO_GRIPPER)

        _serve(scene, grippers=[_gripper_setpoint(CHARLO_GRIPPER, opening, max_effort)])

        assert scene.io.latest_gripper_command("charlo", "gripper") is None

    def test_a_robot_that_left_takes_its_setpoints_with_it(self, scene):
        _admit(scene, "alpha", "openarm_v2")
        _admit(scene, "bravo", "openarm_v2")
        pair("arms", ALPHA_LEFT_ARM, BRAVO_LEFT_ARM)
        pair("grippers", ALPHA_LEFT_GRIPPER)
        _serve(
            scene,
            arms=[_arm_setpoint(ALPHA_LEFT_ARM, [0.5]), _arm_setpoint(BRAVO_LEFT_ARM, [0.9])],
            grippers=[_gripper_setpoint(ALPHA_LEFT_GRIPPER, 0.25)],
        )

        scene.io.forget("alpha")

        assert scene.io.latest_arm_command("alpha", "left_arm") is None
        assert scene.io.latest_gripper_command("alpha", "left_gripper") is None
        assert scene.io.latest_arm_command("bravo", "left_arm") == ([0.9], [])
