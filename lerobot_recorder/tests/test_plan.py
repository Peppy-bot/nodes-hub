"""Discovery: the launcher's bindings decide which limbs get recorded, how
they are named, and which commanded source belongs to which measured one."""

from types import SimpleNamespace

import pytest

from lerobot_recorder.plan import (
    BoundSources,
    LinkKind,
    discover,
    limb_name,
    snapshot_sources,
)

CORE = "cn"


def source(instance_id: str, link_id: str = "link"):
    return SimpleNamespace(
        producer=SimpleNamespace(core_node=CORE, instance_id=instance_id),
        source_link_id=link_id,
    )


def bind(*, joints=((), ()), grippers=((), ())) -> BoundSources:
    """The snapshot a launcher's bindings would read as: (measured,
    commanded) per pairing kind, no cameras."""
    return BoundSources(
        limbs=(
            (LinkKind.JOINT, tuple(joints[0]), tuple(joints[1])),
            (LinkKind.GRIPPER, tuple(grippers[0]), tuple(grippers[1])),
        ),
        color_producers=(),
        rgbd_producers=(),
    )


def bimanual() -> BoundSources:
    return bind(
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


def test_limbs_are_named_after_their_follower_joints_first():
    plan = discover(bimanual())
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


def test_commanded_sources_sharing_an_instance_stay_distinct():
    """Every openarm pairing is led by the one backbone instance, so the
    observed link is the only thing telling the two arms apart."""
    plan = discover(bimanual())
    assert [e.key for e in plan.action] == [
        (CORE, "backbone_inst", "left_arm_link"),
        (CORE, "backbone_inst", "right_arm_link"),
        (CORE, "backbone_inst", "left_gripper_link"),
        (CORE, "backbone_inst", "right_gripper_link"),
    ]


def test_action_falls_back_to_the_measured_source_of_its_own_limb():
    plan = discover(bimanual())
    assert plan.action_fallback == {
        (CORE, "backbone_inst", "left_arm_link"): (CORE, "left_arm_inst", "link"),
        (CORE, "backbone_inst", "right_arm_link"): (CORE, "right_arm_inst", "link"),
        (CORE, "backbone_inst", "left_gripper_link"): (CORE, "left_grip_inst", "link"),
        (CORE, "backbone_inst", "right_gripper_link"): (CORE, "right_grip_inst", "link"),
    }


def test_a_robot_the_launcher_bound_differently_records_what_it_has():
    """One arm, no gripper: the manifest fixes nothing about the robot."""
    plan = discover(
        bind(joints=([source("arm_inst")], [source("leader_inst", "arm_link")]))
    )
    assert [e.feature_key for e in plan.state] == ["arm_inst"]
    assert [e.feature_key for e in plan.action] == ["arm_inst"]


def test_unpairable_bindings_are_refused():
    """A measured source with no commanded source of its own has no action to
    record, and nothing here can guess which one it should have been."""
    bound = bind(
        joints=(
            [source("left_arm_inst"), source("right_arm_inst")],
            [source("backbone_inst", "left_arm_link")],
        ),
    )
    with pytest.raises(ValueError, match="joint limbs are bound 2 measured to 1 commanded"):
        discover(bound)


def test_one_instance_following_two_pairings_names_both():
    used: set[str] = set()
    first = limb_name(used, source("arm_inst", "left"))
    second = limb_name(used, source("arm_inst", "right"))
    assert (first, second) == ("arm_inst", "arm_inst_right")


def test_a_name_that_cannot_be_made_unique_is_refused():
    used = {"arm_inst", "arm_inst_left"}
    with pytest.raises(ValueError, match="both name themselves"):
        limb_name(used, source("arm_inst", "left"))


def test_a_launch_with_no_limbs_is_refused():
    """Cameras alone are not a robot dataset: nothing would fill
    observation.state or action."""
    with pytest.raises(ValueError, match="no limbs are bound"):
        discover(bind())


def test_one_source_bound_to_two_limbs_is_refused():
    """Two limbs sharing an identity would share one cache slot, so both would
    record whichever message landed last."""
    twice = source("arm_inst", "arm_link")
    bound = bind(joints=([twice, twice], [source("lead", "a"), source("lead", "b")]))
    with pytest.raises(ValueError, match="bound to two limbs"):
        discover(bound)


def test_the_snapshot_is_immune_to_the_live_sets_moving_on():
    """The sets stay live after boot, so the snapshot must materialize its
    one read, or a replan mid-discovery could pair sources from two plan
    revisions."""
    deliveries = {
        "joints": [source("arm_inst")],
        "joints_cmd": [source("lead", "arm_link")],
        "grippers": [],
        "grippers_cmd": [],
    }

    def live_slot(bucket):
        return SimpleNamespace(sources=lambda _runner: deliveries[bucket])

    no_cameras = SimpleNamespace(bound_producers=lambda _runner: [])
    snapshot = snapshot_sources(
        None,
        (
            (LinkKind.JOINT, live_slot("joints"), live_slot("joints_cmd")),
            (LinkKind.GRIPPER, live_slot("grippers"), live_slot("grippers_cmd")),
        ),
        no_cameras,
        no_cameras,
    )
    deliveries["joints"].clear()
    deliveries["joints_cmd"].clear()

    plan = discover(snapshot)
    assert [e.feature_key for e in plan.state] == ["arm_inst"]
