"""Discovery: the launcher's bindings decide which limbs get recorded, how
they are named, and which commanded source belongs to which measured one."""

from types import SimpleNamespace

import pytest

from lerobot_recorder.plan import (
    BoundSources,
    LinkKind,
    PairingEnd,
    SourceKey,
    discover,
    limb_name,
    snapshot_sources,
)

CORE = "cn"


def peer(instance_id: str, link_id: str):
    """The far end a launcher names a pair by, as the runtime tags it."""
    return SimpleNamespace(
        producer=SimpleNamespace(core_node=CORE, instance_id=instance_id),
        peer_link_id=link_id,
    )


def source(instance_id: str, link_id: str = "link", peer=None):
    return SimpleNamespace(
        producer=SimpleNamespace(core_node=CORE, instance_id=instance_id),
        source_link_id=link_id,
        peer=peer,
    )


def key(instance_id: str, link_id: str = "link", peer: PairingEnd | None = None) -> SourceKey:
    return SourceKey(source=PairingEnd(CORE, instance_id, link_id), peer=peer)


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
            [source("backbone_inst", "left_arm"), source("backbone_inst", "right_arm")],
        ),
        grippers=(
            [source("left_gripper_inst"), source("right_gripper_inst")],
            [
                source("backbone_inst", "left_gripper"),
                source("backbone_inst", "right_gripper"),
            ],
        ),
    )


def test_limbs_are_named_after_the_pairing_their_command_travels_on():
    plan = discover(bimanual())
    assert [e.feature_key for e in plan.state] == [
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
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
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
    ]


def simulated_bimanual() -> BoundSources:
    """The engines' shape: one simulation answers for every limb it stands
    on a single slot per kind of limb, `arms` and `grippers`, each pair
    named by the backbone link that opened it."""
    return bind(
        joints=(
            [
                source("simulation_inst", "arms", peer("backbone_inst", "left_arm")),
                source("simulation_inst", "arms", peer("backbone_inst", "right_arm")),
            ],
            [source("backbone_inst", "left_arm"), source("backbone_inst", "right_arm")],
        ),
        grippers=(
            [
                source("simulation_inst", "grippers", peer("backbone_inst", "left_gripper")),
                source("simulation_inst", "grippers", peer("backbone_inst", "right_gripper")),
            ],
            [source("backbone_inst", "left_gripper"), source("backbone_inst", "right_gripper")],
        ),
    )


def test_a_simulated_robot_records_the_columns_a_physical_one_does():
    """A simulated robot's four limbs all answer from one simulation instance
    on one slot per kind; a physical robot's answer from a driver each. Both
    are commanded on the backbone's four limb links, so both datasets carry
    the same columns, which is what lets a policy trained on one run on the
    other."""
    simulated = discover(simulated_bimanual())
    physical = discover(bimanual())
    assert [e.feature_key for e in simulated.state] == [e.feature_key for e in physical.state]
    assert [e.feature_key for e in simulated.action] == [e.feature_key for e in physical.action]


def test_measured_sources_sharing_a_slot_are_told_apart_by_their_pair():
    """A simulation stands both arms on its one `arms` slot, so the pair's
    far end, the backbone link it was opened from, is the only thing telling
    the two measured sources apart; without it both arms would share one
    cache slot. Each command still falls back to the measured source of its
    own limb."""
    plan = discover(simulated_bimanual())
    left = key("simulation_inst", "arms", PairingEnd(CORE, "backbone_inst", "left_arm"))
    right = key("simulation_inst", "arms", PairingEnd(CORE, "backbone_inst", "right_arm"))
    assert [e.key for e in plan.state[:2]] == [left, right]
    assert plan.action_fallback[key("backbone_inst", "left_arm")] == left
    assert plan.action_fallback[key("backbone_inst", "right_arm")] == right


def test_commanded_sources_sharing_an_instance_stay_distinct():
    """Every openarm pairing is led by the one backbone instance, so the
    observed link is the only thing telling the two arms apart."""
    plan = discover(bimanual())
    assert [e.key for e in plan.action] == [
        key("backbone_inst", "left_arm"),
        key("backbone_inst", "right_arm"),
        key("backbone_inst", "left_gripper"),
        key("backbone_inst", "right_gripper"),
    ]


def test_action_falls_back_to_the_measured_source_of_its_own_limb():
    plan = discover(bimanual())
    assert plan.action_fallback == {
        key("backbone_inst", "left_arm"): key("left_arm_inst"),
        key("backbone_inst", "right_arm"): key("right_arm_inst"),
        key("backbone_inst", "left_gripper"): key("left_gripper_inst"),
        key("backbone_inst", "right_gripper"): key("right_gripper_inst"),
    }


def test_a_robot_the_launcher_bound_differently_records_what_it_has():
    """One arm, no gripper: the manifest fixes nothing about the robot."""
    plan = discover(
        bind(joints=([source("arm_inst")], [source("leader_inst", "arm")]))
    )
    assert [e.feature_key for e in plan.state] == ["arm"]
    assert [e.feature_key for e in plan.action] == ["arm"]


def test_copies_keep_separate_dataset_dimensions_and_fallbacks():
    for name in ["alpha", "bravo"]:
        follower = f"{name}_left_gripper_inst"
        backbone = f"{name}_backbone_inst"
        plan = discover(bind(grippers=(
            [source(follower)], [source(backbone, "left_gripper")],
        )))
        # A copy records its own dataset, so the columns carry no copy name.
        assert [entry.feature_key for entry in plan.state] == ["left_gripper"]
        assert [entry.feature_key for entry in plan.action] == ["left_gripper"]
        assert plan.action_fallback == {
            key(backbone, "left_gripper"): key(follower),
        }


def test_unpairable_bindings_are_refused():
    """A measured source with no commanded source of its own has no action to
    record, and nothing here can guess which one it should have been."""
    bound = bind(
        joints=(
            [source("left_arm_inst"), source("right_arm_inst")],
            [source("backbone_inst", "left_arm")],
        ),
    )
    with pytest.raises(ValueError, match="joint limbs are bound 2 measured to 1 commanded"):
        discover(bound)


def test_limbs_commanded_on_one_backbone_are_told_apart_by_link():
    used: set[str] = set()
    first = limb_name(used, source("backbone_inst", "left_arm"))
    second = limb_name(used, source("backbone_inst", "right_arm"))
    assert (first, second) == ("left_arm", "right_arm")


def test_two_limbs_commanded_on_one_link_are_refused():
    used = {"left_arm"}
    with pytest.raises(ValueError, match="both name themselves 'left_arm'"):
        limb_name(used, source("backbone_inst", "left_arm"))


def test_a_launch_with_no_limbs_is_refused():
    """Cameras alone are not a robot dataset: nothing would fill
    observation.state or action."""
    with pytest.raises(ValueError, match="no limbs are bound"):
        discover(bind())


def test_one_source_bound_to_two_limbs_is_refused():
    """Two limbs sharing an identity would share one cache slot, so both would
    record whichever message landed last."""
    twice = source("arm_inst", "arm")
    bound = bind(joints=([twice, twice], [source("lead", "a"), source("lead", "b")]))
    with pytest.raises(ValueError, match="bound to two limbs"):
        discover(bound)


def test_the_snapshot_is_immune_to_the_live_sets_moving_on():
    """The sets stay live after boot, so the snapshot must materialize its
    one read, or a replan mid-discovery could pair sources from two plan
    revisions."""
    deliveries = {
        "joints": [source("arm_inst")],
        "joints_cmd": [source("lead", "arm")],
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
    assert [e.feature_key for e in plan.state] == ["arm"]
