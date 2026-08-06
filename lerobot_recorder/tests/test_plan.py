"""Discovery: the launcher's bindings decide which limbs get recorded, how
they are named, and which commanded source belongs to which measured one."""

import asyncio
from types import SimpleNamespace

import pytest

from lerobot_recorder import plan as plan_mod
from lerobot_recorder.plan import LinkKind, discover, limb_name

CORE = "cn"
# wait_for_sources checks the runtime's token so a stop ends the start.
RUNNER = SimpleNamespace(cancellation_token=lambda: SimpleNamespace(is_cancelled=lambda: False))


def source(instance_id: str, link_id: str = "link"):
    return SimpleNamespace(
        producer=SimpleNamespace(core_node=CORE, instance_id=instance_id),
        source_link_id=link_id,
    )


def discover_bound():
    """Discover from a fresh read of the bound slots, as _start does via the
    settled snapshot wait_for_sources hands it."""
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


def test_discovery_waits_for_the_daemon_to_deliver_the_sources(monkeypatch):
    """Observer sets arrive asynchronously, so an empty first read means "not
    yet", not "no robot"."""
    deliveries = {"joints": [], "joints_cmd": [], "grippers": [], "grippers_cmd": []}

    def late_slot(bucket):
        return SimpleNamespace(sources=lambda _runner: list(deliveries[bucket]))

    monkeypatch.setattr(
        plan_mod,
        "limb_slots",
        lambda: (
            (LinkKind.JOINT, late_slot("joints"), late_slot("joints_cmd")),
            (LinkKind.GRIPPER, late_slot("grippers"), late_slot("grippers_cmd")),
        ),
    )

    async def run():
        async def deliver():
            await asyncio.sleep(0.05)
            deliveries["joints"] = [source("arm_inst")]
            deliveries["joints_cmd"] = [source("leader_inst", "arm_link")]

        asyncio.create_task(deliver())
        snapshot = await plan_mod.wait_for_sources(RUNNER, settle_s=0.05, timeout_s=5.0)
        # The membership can change between settling and discovery; the plan
        # must come from the settled snapshot, not a re-read of the live sets.
        deliveries["joints"] = []
        deliveries["joints_cmd"] = []
        return discover(node_runner=None, limb_sources=snapshot)

    plan = asyncio.run(run())
    assert [e.feature_key for e in plan.state] == ["arm_inst"]


def test_waiting_for_sources_gives_up_so_an_empty_launch_is_reported(monkeypatch):
    bind(monkeypatch)

    async def run():
        return await plan_mod.wait_for_sources(RUNNER, settle_s=0.05, timeout_s=0.2)

    snapshot = asyncio.run(run())
    with pytest.raises(ValueError, match="no limbs are bound"):
        discover(node_runner=None, limb_sources=snapshot)


def test_waiting_for_sources_ends_the_start_when_the_node_stops(monkeypatch):
    """The wait runs before the node has a shutdown hook, so a stop during it
    must not fall through and build a session inside the teardown window."""
    bind(monkeypatch)
    token = SimpleNamespace(is_cancelled=lambda: True)
    runner = SimpleNamespace(cancellation_token=lambda: token)

    async def run():
        await plan_mod.wait_for_sources(runner, settle_s=0.01, timeout_s=5.0)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())


def test_a_stop_beats_an_expired_deadline(monkeypatch):
    """A token that fired during the final poll sleep must end the start, not
    fall through the timeout exit into building a session mid-teardown."""
    bind(monkeypatch)
    token = SimpleNamespace(is_cancelled=lambda: True)
    runner = SimpleNamespace(cancellation_token=lambda: token)

    async def run():
        await plan_mod.wait_for_sources(runner, settle_s=0.01, timeout_s=0.0)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())


def test_a_slot_delivered_late_still_joins_the_plan(monkeypatch):
    """Delivery is per slot, so returning at the first non-empty read would
    discover a bimanual robot as a gripperless one, permanently. Delivery is
    driven by read count rather than the clock so the gap cannot race the
    poll."""
    reads = {"n": 0}
    GRIPPERS_ARRIVE_ON_READ = 4

    def slot_for(bucket):
        def sources(_runner):
            if bucket.startswith("joints"):
                return [source("arm_inst")] if bucket == "joints" else [source("lead", "arm")]
            if reads["n"] < GRIPPERS_ARRIVE_ON_READ:
                return []
            return [source("grip_inst")] if bucket == "grippers" else [source("lead", "grip")]

        return SimpleNamespace(sources=sources)

    def limb_slots():
        reads["n"] += 1
        return (
            (LinkKind.JOINT, slot_for("joints"), slot_for("joints_cmd")),
            (LinkKind.GRIPPER, slot_for("grippers"), slot_for("grippers_cmd")),
        )

    monkeypatch.setattr(plan_mod, "limb_slots", limb_slots)

    async def run():
        snapshot = await plan_mod.wait_for_sources(RUNNER, settle_s=0.3, timeout_s=5.0)
        return discover(node_runner=None, limb_sources=snapshot)

    plan = asyncio.run(run())
    assert [e.feature_key for e in plan.state] == ["arm_inst", "grip_inst"]


def test_settled_membership_returns_well_before_the_timeout(monkeypatch):
    """A wait that never settles would add its whole timeout to every start."""
    bind(monkeypatch, joints=([source("arm_inst")], [source("leader_inst", "arm_link")]))

    async def run():
        loop = asyncio.get_running_loop()
        started = loop.time()
        await plan_mod.wait_for_sources(RUNNER, settle_s=0.05, timeout_s=4.0)
        return loop.time() - started

    assert asyncio.run(run()) < 1.0


def test_one_source_bound_to_two_limbs_is_refused(monkeypatch):
    """Two limbs sharing an identity would share one cache slot, so both would
    record whichever message landed last."""
    twice = source("arm_inst", "arm_link")
    bind(monkeypatch, joints=([twice, twice], [source("lead", "a"), source("lead", "b")]))
    with pytest.raises(ValueError, match="bound to two limbs"):
        discover_bound()
