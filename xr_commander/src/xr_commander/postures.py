"""The posture buttons: the right controller's face buttons move the chosen
robot to its postures, and the panel's right-hand row shows the move in
flight and how it ended.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from xr_commander.bus import (
    GOAL_CANCEL_TIMEOUT_S,
    REAL_CLOCK,
    CancellationToken,
    Clock,
    FlashNote,
    Latch,
    cause,
    fire_goal,
    log,
    spawn,
    ticks,
)
from xr_commander.config import Settings
from xr_commander.devices import (
    POSTURE_HAND,
    FrameSource,
    Hand,
    HandSample,
    RisingEdge,
    fresh_sample,
)
from xr_commander.drive import GRIP_HELD, POSTURE, Driver, Lost, Refusal
from xr_commander.robots import PairedRobots

_GOAL_RESULT_FLOOR_S = 120.0
# A follower may stretch a short requested duration to its own limits, so the
# result wait scales the request and keeps a generous floor.
_RESULT_GRACE_FACTOR = 3.0


def _result_timeout_s(posture_move_duration_s: float) -> float:
    """How long to wait for a posture move's result before giving up on it."""
    return max(_GOAL_RESULT_FLOOR_S, _RESULT_GRACE_FACTOR * posture_move_duration_s)


class PostureStatus:
    """The posture buttons as the panel shows them, on the right hand's row:
    the move in flight, then its outcome flashed for a few seconds. Every
    writer runs on the event loop."""

    def __init__(self, clock: Clock = REAL_CLOCK) -> None:
        self.moving: str | None = None
        self._note = FlashNote(clock)

    def set_note(self, text: str) -> None:
        self._note.set(text)

    def state(self, now_monotonic: float | None = None) -> str | None:
        """What the row says: a fresh note, else the move in flight, else
        None."""
        note = self._note.text(now_monotonic)
        if note:
            return note
        if self.moving is not None:
            return f"moving to {self.moving}..."
        return None

    def end_move(self, label: str) -> None:
        """The move to `label` is over; another button's move stays."""
        if self.moving == label:
            self.moving = None


async def run_posture_button(
    node_runner,
    *,
    action_module,
    pressed: Callable[[HandSample], bool],
    session: FrameSource,
    robots: PairedRobots,
    driver: Driver,
    status: PostureStatus,
    settings: Settings,
    token: CancellationToken,
    clock: Clock = REAL_CLOCK,
) -> None:
    """Fire the module's posture move at the chosen robot on the button's
    rising edge.

    `pressed` reads the right hand's button for this posture. A press takes
    the robot as a squeeze does, asks the daemon whether the headset may
    move it, and keeps it until the move ends. A press while a grip is held,
    while a hand still keeps the chosen robot after its release, or while a
    posture move runs does nothing, and the row says why. A squeeze during
    the move stops it: the squeeze stops the moves in flight on the robot.
    A move whose send or result is lost ends as
    `Driver.end_lost_posture_move` says. The move in flight and its outcome
    go to `status`.
    """
    name = action_module.TARGET_ACTION_NAME
    label = name.removeprefix("move_to_")
    unoffered = Latch()

    button = RisingEdge()
    waiter: asyncio.Task | None = None

    def ignored(note: str) -> None:
        status.set_note(note)
        log(note)

    async def press() -> asyncio.Task | None:
        """The press on the chosen robot: the task that watches the move it
        started, or None with the reason on the row. The move counts as in
        flight from the press on, so the other posture button waits for it."""
        target = robots.producer_for(action_module)
        robot = robots.chosen()
        if target is None:
            note = (
                f"move to {label}: no robot paired"
                if robot is None
                else f"move to {label}: {robot} has no postures"
            )
            status.set_note(note)
            unoffered.trip(note)
            return None
        unoffered.clear()
        status.moving = label

        async def start_move():
            """The accepted goal of the move, else why it did not start."""
            refused = await driver.check_move(target.instance_id)
            if refused is not None:
                return Refusal(refused)
            fired = await fire_goal(
                node_runner,
                action_module,
                target,
                name,
                duration_s=settings.posture_move_duration_s,
            )
            if fired is None:
                return Lost("the goal could not be sent (see log)")
            if not fired.accepted:
                return Refusal(cause(fired.refusal.message))
            driver.on_posture_cancel(robot, lambda: _cancel_soon(fired, label))
            return fired

        started = await driver.start_posture_move(robot, start_move)
        if isinstance(started, Refusal):
            status.end_move(label)
            ignored(f"move to {label} refused: {started.cause}")
            return None
        timeout = _result_timeout_s(settings.posture_move_duration_s)
        return asyncio.create_task(
            _watch_result(label, started, timeout, status, driver, robot)
        )

    try:
        async for _ in ticks(settings.tick_period_s, token, clock):
            stale_s = settings.stale_timeout_s
            right = fresh_sample(session, POSTURE_HAND, stale_s, clock)
            left = fresh_sample(session, Hand.LEFT, stale_s, clock)
            squeeze = bool(right and right.squeezing) or bool(left and left.squeezing)
            rising = button.step(pressed(right) if right is not None else None)
            if waiter is not None and waiter.done():
                waiter = None
            if not rising:
                continue
            held = driver.press_refusal(robots.chosen())
            if squeeze:
                ignored(f"move to {label} ignored: {GRIP_HELD}")
            elif held is not None:
                ignored(f"move to {label} ignored: {held}")
            elif status.moving is not None:
                ignored(f"move to {label} ignored: still moving to {status.moving}")
            elif waiter is None:
                waiter = await press()
    finally:
        if waiter is not None:
            waiter.cancel()


def _cancel_soon(handle, label: str) -> None:
    """Cancel the posture move to `label` without waiting on it: its result
    watcher reports how it ended."""

    async def cancel() -> None:
        try:
            outcome = await handle.cancel_goal(GOAL_CANCEL_TIMEOUT_S)
        except Exception as e:
            log(f"move to {label}: cancel failed: {e!r}")
            return
        log(f"move to {label}: cancel {outcome.state.name.lower()}")

    spawn(cancel())


async def _watch_result(
    label: str,
    handle,
    timeout_s: float,
    status: PostureStatus,
    driver: Driver,
    robot: str,
) -> None:
    """Log the move's result and flash it on the row. A move whose result
    does not come ends as `Driver.end_lost_posture_move` says. The move is
    over either way, so its hold of `robot` drops."""
    try:
        result = await handle.get_result(timeout_s)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        note = f"move to {label}: result unknown"
        log(f"{note}: {e!r}")
        status.set_note(note)
        await driver.end_lost_posture_move(robot)
        return
    finally:
        status.end_move(label)
        driver.drop(robot, POSTURE)
    note = _outcome_note(label, result)
    log(note if result.cancel_reason is None else f"{note}: {result.cancel_reason}")
    status.set_note(note)


def _outcome_note(label: str, result) -> str:
    """What the row says of how the move to `label` ended: completed, failed
    with the cause of its failure, or cancelled, abandoned or expired."""
    status = result.status.name
    if status == "FAILED":
        return f"move to {label} failed: {cause(result.failure.message)}"
    return f"move to {label} {status.lower()}"
