from types import SimpleNamespace

from peppygen.paired_topics.right_arm import pose_setpoints as right_arm_pose_setpoints
from peppygen.paired_topics.right_gripper import (
    gripper_setpoints as right_gripper_setpoints,
)

from tests.helpers import boot, robots_of
from xr_commander.robots import (
    PairedRobots,
    robot_of_peer,
    robot_of_producer,
)


def peer(instance_id, copy=None):
    """A pair as the generated slot answers it."""
    return SimpleNamespace(
        copy=copy,
        info=SimpleNamespace(producer=SimpleNamespace(instance_id=instance_id)),
    )


def slot(*peers):
    """A paired topic module standing in for one slot, answering the pairs
    it was given."""
    return live_slot(list(peers))


def live_slot(peers):
    """A slot whose pairs follow the list it is given."""
    return SimpleNamespace(peers=lambda _runner: list(peers))


def one_slot(*peers):
    """The robots of one right-arm slot holding `peers`."""
    return PairedRobots(None, arms={"right": slot(*peers)}, grippers={})


def test_a_pair_is_named_by_its_copy_else_its_instance():
    assert robot_of_peer(peer("alpha_backbone_inst", "alpha")) == "alpha"
    assert robot_of_peer(peer("backbone_inst")) == "backbone_inst"
    assert (
        robot_of_producer(
            SimpleNamespace(copy=SimpleNamespace(name="bravo"), producer=None)
        )
        == "bravo"
    )
    assert (
        robot_of_producer(
            SimpleNamespace(
                copy=None, producer=SimpleNamespace(instance_id="recorder_inst")
            )
        )
        == "recorder_inst"
    )


def test_the_robots_are_listed_once_in_name_order_and_the_first_that_paired_is_driven():
    left: list = []
    # An SO-101 pairs on the right side alone, and pairs first.
    right = [peer("charlie_backbone_inst", "charlie")]
    robots = PairedRobots(
        None, arms={"left": live_slot(left), "right": live_slot(right)}, grippers={}
    )
    assert robots.names() == ["charlie"]
    # An OpenArm joins on both sides: the list is in name order, and
    # charlie, the first that paired, stays driven.
    left.append(peer("alpha_backbone_inst", "alpha"))
    right.append(peer("alpha_backbone_inst", "alpha"))
    assert robots.names() == ["alpha", "charlie"]
    assert robots.chosen() == "charlie"
    # Charlie leaves: the hands go to alpha. Charlie rejoins, and the hands
    # stay on alpha.
    right.remove(right[0])
    assert robots.chosen() == "alpha"
    right.insert(0, peer("charlie_backbone_inst", "charlie"))
    assert robots.names() == ["alpha", "charlie"]
    assert robots.chosen() == "alpha"


def test_the_first_robot_is_driven_until_one_is_chosen_and_while_it_stands():
    paired = [
        peer("alpha_backbone_inst", "alpha"),
        peer("bravo_backbone_inst", "bravo"),
    ]
    robots = PairedRobots(None, arms={"right": live_slot(paired)}, grippers={})
    assert robots.chosen() == "alpha"
    assert robots.choose("bravo")
    assert robots.chosen() == "bravo"
    # Bravo leaves: the first standing is driven; bravo back, the choice stands.
    paired.pop()
    assert robots.chosen() == "alpha"
    paired.append(peer("bravo_backbone_inst", "bravo"))
    assert robots.chosen() == "bravo"


def test_a_choice_of_a_robot_not_paired_is_refused_and_changes_nothing():
    paired = [peer("alpha_backbone_inst", "alpha")]
    robots = PairedRobots(None, arms={"right": live_slot(paired)}, grippers={})
    assert not robots.choose("delta")
    assert not robots.choose("")
    assert robots.chosen() == "alpha"
    paired.append(peer("delta_backbone_inst", "delta"))
    assert robots.chosen() == "alpha"
    assert one_slot().chosen() is None


def test_the_chosen_robots_pair_is_found_per_slot_and_absent_where_it_has_none():
    left = slot(peer("alpha_backbone_inst", "alpha"))
    right = slot(
        peer("charlie_backbone_inst", "charlie"), peer("alpha_backbone_inst", "alpha")
    )
    robots = PairedRobots(None, arms={"left": left, "right": right}, grippers={})
    assert robots.peer_for(right).producer.instance_id == "alpha_backbone_inst"
    robots.choose("charlie")
    assert robots.peer_for(left) is None
    assert robots.peer_for(right).producer.instance_id == "charlie_backbone_inst"


def test_the_chosen_robots_producer_is_found_on_a_contract_slot():
    robots = one_slot(peer("alpha_backbone_inst", "alpha"))
    members = [
        SimpleNamespace(copy=SimpleNamespace(name="bravo"), producer="bravo's"),
        SimpleNamespace(copy=SimpleNamespace(name="alpha"), producer="alpha's"),
    ]
    module = SimpleNamespace(bound_members=lambda _runner: members)
    assert robots.producer_for(module) == "alpha's"
    assert one_slot().producer_for(module) is None


def test_a_producer_outside_any_copy_serves_every_robot_unless_it_is_one():
    # The robot outside any copy pairs first, so the hands drive it.
    paired = [peer("backbone_inst")]
    robots = PairedRobots(None, arms={"right": live_slot(paired)}, grippers={})
    assert robots.chosen() == "backbone_inst"
    paired.append(peer("alpha_backbone_inst", "alpha"))
    recorder = SimpleNamespace(
        copy=None, producer=SimpleNamespace(instance_id="recorder_inst")
    )
    backbone = SimpleNamespace(
        copy=None, producer=SimpleNamespace(instance_id="backbone_inst")
    )
    recorders = SimpleNamespace(bound_members=lambda _runner: [recorder])
    postures = SimpleNamespace(bound_members=lambda _runner: [backbone])
    # Its own backbone serves it.
    assert robots.chosen() == "backbone_inst"
    assert robots.producer_for(postures) is backbone.producer
    robots.choose("alpha")
    # A lone recorder serves alpha too; another robot's backbone does not.
    assert robots.producer_for(recorders) is recorder.producer
    assert robots.producer_for(postures) is None


def test_a_robot_drives_the_hands_whose_arm_slot_it_pairs():
    left = slot(peer("alpha_backbone_inst", "alpha"))
    right = slot(
        peer("alpha_backbone_inst", "alpha"), peer("bravo_backbone_inst", "bravo")
    )
    # A gripper pair names no hand an arm: the arm slots alone do.
    gripper = slot(peer("charlie_backbone_inst", "charlie"))
    robots = PairedRobots(
        None, arms={"left": left, "right": right}, grippers={"left": gripper}
    )
    assert robots.arm_sides("alpha") == {"left", "right"}
    assert robots.arm_sides("bravo") == {"right"}
    assert robots.arm_sides("charlie") == frozenset()
    assert robots.arm_sides("delta") == frozenset()


async def test_the_pairs_the_harness_seeds_are_grouped_by_copy_over_the_wire():
    async with boot(
        right_arm_members={
            "alpha_backbone_inst": "alpha",
            "bravo_backbone_inst": "bravo",
        },
        right_gripper_members={"bravo_backbone_inst": "bravo"},
    ) as h:
        robots = robots_of(h)
        assert robots.names() == ["alpha", "bravo"]
        assert robots.peer_for(right_gripper_setpoints) is None
        assert robots.choose("bravo")
        assert robots.peer_for(right_arm_pose_setpoints).producer.instance_id == (
            "bravo_backbone_inst"
        )
        assert robots.peer_for(right_gripper_setpoints) is not None


def test_a_producer_is_found_for_any_robot_by_name():
    robots = one_slot(
        peer("alpha_backbone_inst", "alpha"), peer("bravo_backbone_inst", "bravo")
    )
    members = [
        SimpleNamespace(copy=SimpleNamespace(name="bravo"), producer="bravo's"),
        SimpleNamespace(copy=SimpleNamespace(name="alpha"), producer="alpha's"),
    ]
    module = SimpleNamespace(bound_members=lambda _runner: members)
    assert robots.producer_of(module, "bravo") == "bravo's"
    assert robots.producer_of(module, "charlie") is None
    assert robots.producer_of(module, None) is None


def test_a_robot_that_left_is_still_no_shared_producer():
    # A copy-less robot's backbone stays bound on postures while its pairs
    # are down; it serves no other robot's posture buttons.
    paired = [peer("backbone_inst")]
    robots = PairedRobots(None, arms={"right": live_slot(paired)}, grippers={})
    assert robots.names() == ["backbone_inst"]
    paired.append(peer("alpha_backbone_inst", "alpha"))
    assert robots.names() == ["alpha", "backbone_inst"]
    backbone = SimpleNamespace(
        copy=None, producer=SimpleNamespace(instance_id="backbone_inst")
    )
    postures = SimpleNamespace(bound_members=lambda _runner: [backbone])
    robots.choose("alpha")
    paired.pop(0)
    assert robots.producer_for(postures) is None
