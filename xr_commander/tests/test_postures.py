import asyncio
import contextlib

import pytest
from peppygen.consumed_actions.postures import move_to_home, move_to_ready

from tests.helpers import (
    FakeToken,
    ManualClock,
    ScriptedHeadset,
    boot,
    default_settings,
    eventually,
    press_until_goal,
    robots_of,
    scripted_driver,
)
from xr_commander import postures
from xr_commander.devices import primary_pressed, secondary_pressed
from xr_commander.drive import POSTURE


def done(module):
    """A posture result as a robot sends it. The commander reads only how the
    goal ended, thus the arm pose in it is a fixed value."""
    return module.ResultResponseData(
        arm_names=["right_arm"],
        positions=[0.3, -0.2, 0.3],
        orientations=[0.0, 0.0, 0.0, 1.0],
    )


def boot_postures():
    """A stack whose one paired robot, outside any copy, offers the
    postures: the same instance id names the pair and the producer."""
    return boot(
        postures_instance_ids=["backbone_inst"],
        right_arm_members={"backbone_inst": None},
    )


@contextlib.asynccontextmanager
async def posture_button(h, *, action_module, pressed, status=None, driver=None):
    """`run_posture_button` running against the real runner, with a fleet
    that lists the robot free; yields the scripted headset session."""
    session = ScriptedHeadset()
    token = FakeToken()
    task = asyncio.create_task(
        postures.run_posture_button(
            h.node_runner,
            action_module=action_module,
            pressed=pressed,
            session=session,
            robots=robots_of(h),
            driver=driver or scripted_driver("backbone_inst"),
            status=status or postures.PostureStatus(),
            settings=default_settings(),
            token=token,
        )
    )
    try:
        yield session
    finally:
        token.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_a_posture_move_shows_in_flight_then_its_outcome():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        status = postures.PostureStatus()
        async with posture_button(
            h, action_module=move_to_home, pressed=primary_pressed, status=status
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            active = await pending.accept()
            await eventually(lambda: status.state() == "moving to home...")
            await active.complete(done(move_to_home))
            await eventually(lambda: status.state() == "move to home completed")
            assert status.moving is None


async def test_a_failed_move_flashes_the_robots_reason():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        status = postures.PostureStatus()
        async with posture_button(
            h, action_module=move_to_home, pressed=primary_pressed, status=status
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            active = await pending.accept()
            await active.fail(
                move_to_home.FailureKind.RESOURCE_UNAVAILABLE,
                "the follower stopped reporting; check the arm's bus",
            )
            await eventually(
                lambda: (
                    status.state()
                    == "move to home failed: the follower stopped reporting"
                )
            )


async def test_a_refused_posture_flashes_its_cause():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        status = postures.PostureStatus()
        async with posture_button(
            h, action_module=move_to_home, pressed=primary_pressed, status=status
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            await pending.refuse(
                move_to_home.FailureKind.LEADER_STREAMING,
                "a leader is streaming arm commands; release the headset's grip, then retry",
            )
            await eventually(
                lambda: (
                    status.state()
                    == "move to home refused: a leader is streaming arm commands"
                )
            )
            assert status.moving is None


def test_the_result_wait_scales_with_long_posture_moves():
    assert postures._result_timeout_s(2.0) == 120.0
    assert postures._result_timeout_s(100.0) == 300.0


async def test_the_button_fires_once_per_rising_edge():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        async with posture_button(
            h, action_module=move_to_home, pressed=primary_pressed
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            # The goal reached the real move_to_home producer, carrying the
            # configured posture duration.
            assert pending.request.duration_s == 2.0
            active = await pending.accept()
            # Still held: one edge, one goal.
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            await active.complete(done(move_to_home))


async def test_a_button_already_held_at_first_tracking_does_not_fire():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        async with posture_button(
            h, action_module=move_to_home, pressed=primary_pressed
        ) as session:
            # A thumb already down on the first tracked frame is not a press.
            session.press(right=(True, False, False))
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            # An observed release and then a press is the first real edge.
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            await (await pending.accept()).complete(done(move_to_home))


async def test_a_stale_gap_does_not_refire_a_held_button():
    """A dropout while the thumb stays down must not read as a fresh press:
    the gap makes the input unknown, not released."""
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        async with posture_button(
            h, action_module=move_to_home, pressed=primary_pressed
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            pending = await mock.next_goal(10.0)
            await (await pending.accept()).complete(done(move_to_home))
            # The headset drops out while the thumb stays down...
            session.hands = {}
            await asyncio.sleep(0.05)
            # ...and comes back with the button still held: not a press.
            session.press(right=(True, False, False))
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            # A real release and press after the gap re-arms the button.
            pending = await press_until_goal(
                session,
                mock,
                {"right": (False, False, False)},
                {"right": (True, False, False)},
            )
            await (await pending.accept()).complete(done(move_to_home))


async def test_the_selector_decides_which_button_fires_the_action():
    # Under a secondary-button selector (the real move_to_ready wiring),
    # primary presses stay quiet and a secondary press fires.
    async with boot_postures() as h:
        ready = h.mocks.deps.postures[0].move_to_ready
        async with posture_button(
            h, action_module=move_to_ready, pressed=secondary_pressed
        ) as session:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            with pytest.raises(TimeoutError):
                await ready.next_goal(0.4)
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(False, True, False))
            pending = await ready.next_goal(10.0)
            await (await pending.accept()).complete(done(move_to_ready))


async def test_a_posture_press_is_inert_while_the_chosen_robot_offers_none(capsys):
    async with boot(
        postures_instance_ids=["backbone_inst"],
        right_arm_members={"backbone_inst": None, "bravo_backbone_inst": "bravo"},
    ) as h:
        mock = h.mocks.deps.postures[0].move_to_home
        robots = robots_of(h)
        assert robots.choose("bravo")
        session = ScriptedHeadset()
        token = FakeToken()
        status = postures.PostureStatus()
        task = asyncio.create_task(
            postures.run_posture_button(
                h.node_runner,
                action_module=move_to_home,
                pressed=primary_pressed,
                session=session,
                robots=robots,
                driver=scripted_driver("backbone_inst", "bravo"),
                status=status,
                settings=default_settings(),
                token=token,
            )
        )
        try:
            session.press(right=(False, False, False))
            await asyncio.sleep(0.05)
            session.press(right=(True, False, False))
            with pytest.raises(TimeoutError):
                await mock.next_goal(0.4)
            assert status.state() == "move to home: bravo has no postures"
        finally:
            token.cancel()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        assert capsys.readouterr().out.count("bravo has no postures") == 1


class LostResult:
    """A posture goal whose result never comes."""

    async def get_result(self, _timeout_s):
        raise TimeoutError("no result")


async def test_a_posture_move_whose_result_is_lost_is_stopped_before_its_lease_is_released():
    clock = ManualClock()
    log = []
    driver = scripted_driver("alpha", clock=clock, log=log)

    async def started():
        return "goal"

    await driver.start_posture_move("alpha", started)
    assert log == [("take", "alpha"), ("stop", "alpha")]
    status = postures.PostureStatus(clock)
    status.moving = "home"
    await postures._watch_result("home", LostResult(), 1.0, status, driver, "alpha")
    await clock.idle()
    assert log[2:] == [("stop", "alpha"), ("release", "alpha")]
    assert not driver.holds_for("alpha", POSTURE)
    assert status.state() == "move to home: result unknown"
