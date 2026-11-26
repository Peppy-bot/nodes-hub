import asyncio
import contextlib

from peppygen.consumed_actions.postures import move_to_home, move_to_ready

from tests.helpers import (
    CALL_S,
    HELD_TICKS,
    HOME_REACHED,
    FakeGoals,
    ManualClock,
    ScriptedHeadset,
    boot,
    default_settings,
    eventually,
    parked,
    posture_done,
    postures_of,
    robots_of,
    scripted_driver,
)
from xr_commander import postures
from xr_commander.devices import primary_pressed, secondary_pressed
from xr_commander.grip_world import Press, PressInputs

SETTINGS = default_settings()


def boot_postures():
    """A stack whose one paired robot, outside any copy, offers the
    postures: the same instance id names the pair and the producer."""
    return boot(
        postures_instance_ids=["backbone_inst"],
        right_arm_members={"backbone_inst": None},
    )


def harness_goals(h) -> postures.PostureGoals:
    """The posture goals on the harness's mocks of the robot's postures."""
    return postures.PostureGoals(
        h.node_runner,
        {"home": move_to_home, "ready": move_to_ready},
        SETTINGS.posture_move_duration_s,
    )


class RecordingDriver:
    """A driver that records each event it is handed and steps nothing."""

    def __init__(self) -> None:
        self.events: list = []

    def step(self, event) -> None:
        self.events.append(event)


def presses(driver: RecordingDriver) -> list[tuple[str, str | None]]:
    """The label and the chosen robot of each press the driver got."""
    return [
        (event.label, event.inputs.chosen)
        for event in driver.events
        if isinstance(event, Press)
    ]


@contextlib.asynccontextmanager
async def posture_button(h, clock, driver, *, action_module, pressed):
    """`run_posture_button` on `clock` over the real runner's robots: yields
    the scripted headset session and a step of one tick of the button."""
    session = ScriptedHeadset(clock)

    def loop(token):
        return postures.run_posture_button(
            action_module=action_module,
            pressed=pressed,
            session=session,
            robots=robots_of(h),
            driver=driver,
            settings=SETTINGS,
            token=token,
            clock=clock,
        )

    async with parked(clock, loop, period=SETTINGS.tick_period_s) as step:
        yield session, step


async def press(session, step, buttons=(True, False, False)) -> None:
    """The right hand's buttons read released at one tick, then `buttons` at
    the next: one rising edge."""
    session.press(right=(False, False, False))
    await step()
    session.press(right=buttons)
    await step()


async def test_a_posture_move_shows_in_flight_then_its_outcome():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        clock = ManualClock()
        driver = scripted_driver("backbone_inst", clock=clock, goals=harness_goals(h))
        async with posture_button(
            h, clock, driver, action_module=move_to_home, pressed=primary_pressed
        ) as (session, step):
            await press(session, step)
            assert driver.posture_line() == "moving to home..."
            pending = await mock.next_goal(CALL_S)
            active = await pending.accept()
            await active.complete(posture_done(move_to_home))
            await eventually(lambda: driver.posture_line() == HOME_REACHED)
            assert driver.world.posture_moving is None


async def test_a_failed_move_flashes_the_robots_reason():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        clock = ManualClock()
        driver = scripted_driver("backbone_inst", clock=clock, goals=harness_goals(h))
        async with posture_button(
            h, clock, driver, action_module=move_to_home, pressed=primary_pressed
        ) as (session, step):
            await press(session, step)
            pending = await mock.next_goal(CALL_S)
            active = await pending.accept()
            await active.fail(
                move_to_home.FailureKind.RESOURCE_UNAVAILABLE,
                "the follower stopped reporting; check the arm's bus",
            )
            await eventually(
                lambda: (
                    driver.posture_line()
                    == "move to home failed: the follower stopped reporting"
                )
            )


async def test_a_refused_posture_flashes_its_cause():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        clock = ManualClock()
        driver = scripted_driver("backbone_inst", clock=clock, goals=harness_goals(h))
        async with posture_button(
            h, clock, driver, action_module=move_to_home, pressed=primary_pressed
        ) as (session, step):
            await press(session, step)
            pending = await mock.next_goal(CALL_S)
            await pending.refuse(
                move_to_home.FailureKind.LEADER_STREAMING,
                "a leader is streaming arm commands; release the headset's grip, then retry",
            )
            await eventually(
                lambda: (
                    driver.posture_line()
                    == "move to home refused: a leader is streaming arm commands"
                )
            )
            assert driver.world.posture_moving is None


def test_the_result_wait_scales_with_long_posture_moves():
    assert postures._result_timeout_s(2.0) == 120.0
    assert postures._result_timeout_s(100.0) == 300.0


async def test_a_posture_goal_carries_the_configured_duration():
    async with boot_postures() as h:
        mock = h.mocks.deps.postures[0].move_to_home
        target = postures.posture_target(robots_of(h), move_to_home, "backbone_inst")
        fired = asyncio.create_task(harness_goals(h).fire("home", target))
        pending = await mock.next_goal(CALL_S)
        # The goal reached the real move_to_home producer, carrying the
        # configured posture duration.
        assert pending.request.duration_s == SETTINGS.posture_move_duration_s
        active = await pending.accept()
        assert (await asyncio.wait_for(fired, CALL_S)).accepted
        await active.complete(posture_done(move_to_home))


async def test_the_button_presses_once_per_rising_edge():
    async with boot_postures() as h:
        driver = RecordingDriver()
        async with posture_button(
            h,
            ManualClock(),
            driver,
            action_module=move_to_home,
            pressed=primary_pressed,
        ) as (session, step):
            await press(session, step)
            assert presses(driver) == [("home", "backbone_inst")]
            (pressed,) = driver.events
            assert not pressed.inputs.squeezing
            assert pressed.inputs.target.instance == "backbone_inst"
            # Still held: one edge, one press.
            await step(HELD_TICKS)
            assert len(driver.events) == 1


async def test_a_button_already_held_at_first_tracking_does_not_press():
    async with boot_postures() as h:
        driver = RecordingDriver()
        async with posture_button(
            h,
            ManualClock(),
            driver,
            action_module=move_to_home,
            pressed=primary_pressed,
        ) as (session, step):
            # A thumb already down on the first tracked frame is not a press.
            session.press(right=(True, False, False))
            await step(HELD_TICKS)
            assert driver.events == []
            # An observed release and then a press is the first real edge.
            await press(session, step)
            assert presses(driver) == [("home", "backbone_inst")]


async def test_a_stale_gap_does_not_press_a_held_button_again():
    """A dropout while the thumb stays down does not read as a fresh press:
    the gap makes the input unknown, not released."""
    async with boot_postures() as h:
        driver = RecordingDriver()
        async with posture_button(
            h,
            ManualClock(),
            driver,
            action_module=move_to_home,
            pressed=primary_pressed,
        ) as (session, step):
            await press(session, step)
            assert len(driver.events) == 1
            # The headset drops out while the thumb stays down...
            session.hands = {}
            await step(HELD_TICKS)
            # ...and comes back with the button still held: not a press.
            session.press(right=(True, False, False))
            await step(HELD_TICKS)
            assert len(driver.events) == 1
            # A real release and press after the gap re-arms the button.
            await press(session, step)
            assert len(driver.events) == 2


async def test_the_selector_decides_which_button_presses_the_posture():
    # Under a secondary-button selector (the real move_to_ready wiring),
    # primary presses stay quiet and a secondary press presses.
    async with boot_postures() as h:
        driver = RecordingDriver()
        async with posture_button(
            h,
            ManualClock(),
            driver,
            action_module=move_to_ready,
            pressed=secondary_pressed,
        ) as (session, step):
            await press(session, step, (True, False, False))
            assert driver.events == []
            await press(session, step, (False, True, False))
            assert presses(driver) == [("ready", "backbone_inst")]


async def test_a_posture_press_is_inert_while_the_chosen_robot_offers_none(capsys):
    async with boot(
        postures_instance_ids=["backbone_inst"],
        right_arm_members={"backbone_inst": None, "bravo_backbone_inst": "bravo"},
    ) as h:
        clock = ManualClock()
        log = []
        driver = scripted_driver("backbone_inst", "bravo", clock=clock, log=log)
        robots = robots_of(h)
        assert robots.choose("bravo")
        session = ScriptedHeadset(clock)

        def loop(token):
            return postures.run_posture_button(
                action_module=move_to_home,
                pressed=primary_pressed,
                session=session,
                robots=robots,
                driver=driver,
                settings=SETTINGS,
                token=token,
                clock=clock,
            )

        async with parked(clock, loop, period=SETTINGS.tick_period_s) as step:
            await press(session, step)
            assert driver.posture_line() == "move to home: bravo has no postures"
            assert log == []
        assert capsys.readouterr().out.count("bravo has no postures") == 1


async def test_a_posture_move_whose_result_is_lost_is_stopped_before_its_lease_is_released():
    clock = ManualClock()
    log = []
    goals = FakeGoals(log)
    goals.results.put_nowait(TimeoutError("no result"))
    driver = scripted_driver("alpha", clock=clock, log=log, goals=goals)
    target = postures_of("alpha")
    driver.step(Press("home", PressInputs(False, "alpha", target)))
    await clock.idle()
    assert log == [
        ("take", "alpha"),
        ("stop", "alpha"),
        ("check_move", "alpha_inst"),
        ("goal", "home"),
        ("stop", "alpha"),
        ("release", "alpha"),
    ]
    assert driver.world.robot("alpha").hold is None
    assert driver.world.posture_moving is None
    assert driver.posture_line() == "move to home: result unknown"
