"""Discovery: the launcher's bindings decide which limbs get recorded, how
they are named, and which commanded source belongs to which measured one."""

from types import SimpleNamespace

import pytest

from lerobot_recorder import plan as plan_mod
from lerobot_recorder.plan import LinkKind, discover, limb_name

CORE = "cn"


@pytest.fixture(autouse=True)
def no_cameras(monkeypatch):
    """These tests discover with node_runner=None; the real generated camera
    modules would ask that runner for the bound producer sets, so stand in
    the empty sets a camera-less launch binds."""
    from peppygen.consumed_topics.color_cameras import (
        video_stream as color_cameras_video_stream,
    )
    from peppygen.consumed_topics.rgbd_cameras import (
        video_stream as rgbd_cameras_video_stream,
    )

    monkeypatch.setattr(color_cameras_video_stream, "bound_producers", lambda _r: [])
    monkeypatch.setattr(rgbd_cameras_video_stream, "bound_producers", lambda _r: [])


def source(instance_id: str, link_id: str = "link"):
    return SimpleNamespace(
        producer=SimpleNamespace(core_node=CORE, instance_id=instance_id),
        source_link_id=link_id,
    )


def discover_bound():
    """Discover from one read of the bound slots, as setup does; the boot
    config seeds them, so the read answers the launch's membership."""
    return discover(node_runner=None, limb_sources=plan_mod.read_limb_sources(None))


def slot(sources):
    return SimpleNamespace(sources=lambda _runner: list(sources))


def bind(monkeypatch, *, joints=((), ()), grippers=((), ())):
    """Stand in for the generated slot modules with the sources a launcher
    would have bound: (measured, commanded) per pairing kind."""
    monkeypatch.setattr(
        plan_mod,
        "limb_slots",
        lambda: (
            (LinkKind.JOINT, slot(joints[0]), slot(joints[1])),
            (LinkKind.GRIPPER, slot(grippers[0]), slot(grippers[1])),
        ),
    )


def bimanual(monkeypatch):
    bind(
        monkeypatch,
        joints=(
            [source("left_arm_inst"), source("right_arm_inst")],
            [source("backbone_inst", "left_arm_link"), source("backbone_inst", "right_arm_link")],
        ),
        grippers=(
            [source("left_grip_inst"), source("right_grip_inst")],
            [
                source("backbone_inst", "left_gripper_link"),
                source("backbone_inst", "right_gripper_link"),
            ],
        ),
    )


def test_limbs_are_named_after_their_follower_joints_first(monkeypatch):
    bimanual(monkeypatch)
    plan = discover_bound()
    assert [e.feature_key for e in plan.state] == [
        "left_arm_inst",
        "right_arm_inst",
        "left_grip_inst",
        "right_grip_inst",
    ]
    assert [e.kind for e in plan.state] == [
        LinkKind.JOINT,
        LinkKind.JOINT,
        LinkKind.GRIPPER,
        LinkKind.GRIPPER,
    ]
    # The action records the same limbs under the same names, so a dataset's
    # state and action dimensions line up.
    assert [e.feature_key for e in plan.action] == [
        "left_arm_inst",
        "right_arm_inst",
        "left_grip_inst",
        "right_grip_inst",
    ]


def test_commanded_sources_sharing_an_instance_stay_distinct(monkeypatch):
    """Every openarm pairing is led by the one backbone instance, so the
    observed link is the only thing telling the two arms apart."""
    bimanual(monkeypatch)
    plan = discover_bound()
    assert [e.key for e in plan.action] == [
        (CORE, "backbone_inst", "left_arm_link"),
        (CORE, "backbone_inst", "right_arm_link"),
        (CORE, "backbone_inst", "left_gripper_link"),
        (CORE, "backbone_inst", "right_gripper_link"),
    ]


def test_action_falls_back_to_the_measured_source_of_its_own_limb(monkeypatch):
    bimanual(monkeypatch)
    plan = discover_bound()
    assert plan.action_fallback == {
        (CORE, "backbone_inst", "left_arm_link"): (CORE, "left_arm_inst", "link"),
        (CORE, "backbone_inst", "right_arm_link"): (CORE, "right_arm_inst", "link"),
        (CORE, "backbone_inst", "left_gripper_link"): (CORE, "left_grip_inst", "link"),
        (CORE, "backbone_inst", "right_gripper_link"): (CORE, "right_grip_inst", "link"),
    }


def test_a_robot_the_launcher_bound_differently_records_what_it_has(monkeypatch):
    """One arm, no gripper: the manifest fixes nothing about the robot."""
    bind(
        monkeypatch,
        joints=([source("arm_inst")], [source("leader_inst", "arm_link")]),
    )
    plan = discover_bound()
    assert [e.feature_key for e in plan.state] == ["arm_inst"]
    assert [e.feature_key for e in plan.action] == ["arm_inst"]


def test_unpairable_bindings_are_refused(monkeypatch):
    """A measured source with no commanded source of its own has no action to
    record, and nothing here can guess which one it should have been."""
    bind(
        monkeypatch,
        joints=(
            [source("left_arm_inst"), source("right_arm_inst")],
            [source("backbone_inst", "left_arm_link")],
        ),
    )
    with pytest.raises(ValueError, match="joint limbs are bound 2 measured to 1 commanded"):
        discover_bound()


def test_one_instance_following_two_pairings_names_both(monkeypatch):
    used: set[str] = set()
    first = limb_name(used, source("arm_inst", "left"))
    second = limb_name(used, source("arm_inst", "right"))
    assert (first, second) == ("arm_inst", "arm_inst_right")


def test_a_name_that_cannot_be_made_unique_is_refused():
    used = {"arm_inst", "arm_inst_left"}
    with pytest.raises(ValueError, match="both name themselves"):
        limb_name(used, source("arm_inst", "left"))


def test_a_launch_with_no_limbs_is_refused(monkeypatch):
    """Cameras alone are not a robot dataset: nothing would fill
    observation.state or action."""
    bind(monkeypatch)
    with pytest.raises(ValueError, match="no limbs are bound"):
        discover_bound()


def test_one_source_bound_to_two_limbs_is_refused(monkeypatch):
    """Two limbs sharing an identity would share one cache slot, so both would
    record whichever message landed last."""
    twice = source("arm_inst", "arm_link")
    bind(monkeypatch, joints=([twice, twice], [source("lead", "a"), source("lead", "b")]))
    with pytest.raises(ValueError, match="bound to two limbs"):
        discover_bound()


def test_discovery_consumes_the_snapshot_not_the_live_sets(monkeypatch):
    """The sets stay live after boot, so the plan must come from the one
    snapshot read, or a replan mid-discovery could pair sources from two
    plan revisions."""
    deliveries = {
        "joints": [source("arm_inst")],
        "joints_cmd": [source("lead", "arm_link")],
        "grippers": [],
        "grippers_cmd": [],
    }

    def live_slot(bucket):
        return SimpleNamespace(sources=lambda _runner: list(deliveries[bucket]))

    monkeypatch.setattr(
        plan_mod,
        "limb_slots",
        lambda: (
            (LinkKind.JOINT, live_slot("joints"), live_slot("joints_cmd")),
            (LinkKind.GRIPPER, live_slot("grippers"), live_slot("grippers_cmd")),
        ),
    )
    snapshot = plan_mod.read_limb_sources(None)
    deliveries["joints"] = []
    deliveries["joints_cmd"] = []

    plan = discover(node_runner=None, limb_sources=snapshot)
    assert [e.feature_key for e in plan.state] == ["arm_inst"]
