"""The posture buttons: the right controller's face buttons move the chosen
robot to its postures, and the panel's right-hand row shows the move in
flight and how it ended.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType

from xr_commander.bus import (
    GOAL_CANCEL_TIMEOUT_S,
    REAL_CLOCK,
    CancellationToken,
    Clock,
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
from xr_commander.drive import Driver
from xr_commander.grip_world import PostureTarget, Press, PressInputs
from xr_commander.robots import PairedRobots

_GOAL_RESULT_FLOOR_S = 120.0
# A follower may stretch a short requested duration to its own limits, so the
# result wait scales the request and keeps a generous floor.
_RESULT_GRACE_FACTOR = 3.0


def _result_timeout_s(posture_move_duration_s: float) -> float:
    """How long to wait for a posture move's result before giving up on it."""
    return max(_GOAL_RESULT_FLOOR_S, _RESULT_GRACE_FACTOR * posture_move_duration_s)


def label_of(action_module) -> str:
    """The posture a module moves to, as the row and its notes name it."""
    return action_module.TARGET_ACTION_NAME.removeprefix("move_to_")


def posture_target(
    robots: PairedRobots, action_module, robot: str | None
) -> PostureTarget | None:
    """`robot`'s producer of the module's postures, None while it has none."""
    producer = robots.producer_of(action_module, robot)
    return None if producer is None else PostureTarget(producer, producer.instance_id)


class PostureGoals:
    """The headset's posture goals on the robots' backbones (`drive.Goals`)."""

    def __init__(
        self, node_runner, modules: Mapping[str, ModuleType], duration_s: float
    ) -> None:
        self._node_runner = node_runner
        self._modules = dict(modules)
        self._duration_s = duration_s

    async def fire(self, label: str, target: PostureTarget):
        """The goal of the move to `label` at `target`, None when the send
        failed."""
        module = self._modules[label]
        return await fire_goal(
            self._node_runner,
            module,
            target.producer,
            module.TARGET_ACTION_NAME,
            duration_s=self._duration_s,
        )

    async def result(self, label: str, goal) -> str:
        """The row's note of how the move ended; raises when the result does
        not come within the result timeout."""
        result = await goal.get_result(_result_timeout_s(self._duration_s))
        note = _outcome_note(label, result)
        if result.cancel_reason is not None:
            log(f"{note}: {result.cancel_reason}")
        return note

    def cancel(self, label: str, goal) -> None:
        """Cancel the goal without waiting on it."""
        _cancel_soon(goal, label)


async def run_posture_button(
    *,
    action_module,
    pressed: Callable[[HandSample], bool],
    session: FrameSource,
    robots: PairedRobots,
    driver: Driver,
    settings: Settings,
    token: CancellationToken,
    clock: Clock = REAL_CLOCK,
) -> None:
    """Press the module's posture at each rising edge of its button: the
    world decides whether the press starts a move, and the posture row says
    why it does not. The loop never waits on a start."""
    label = label_of(action_module)
    button = RisingEdge()
    async for _ in ticks(settings.tick_period_s, token, clock):
        stale_s = settings.stale_timeout_s
        right = fresh_sample(session, POSTURE_HAND, stale_s, clock)
        left = fresh_sample(session, Hand.LEFT, stale_s, clock)
        squeezing = bool(right and right.squeezing) or bool(left and left.squeezing)
        if not button.step(pressed(right) if right is not None else None):
            continue
        _names, chosen = robots.standing()
        target = posture_target(robots, action_module, chosen)
        driver.step(Press(label, PressInputs(squeezing, chosen, target)))


def _cancel_soon(handle, label: str) -> None:
    """Cancel the posture move to `label` without waiting on it."""

    async def cancel() -> None:
        try:
            outcome = await handle.cancel_goal(GOAL_CANCEL_TIMEOUT_S)
        except Exception as e:
            log(f"move to {label}: cancel failed: {e!r}")
            return
        log(f"move to {label}: cancel {outcome.state.name.lower()}")

    spawn(cancel())


def _outcome_note(label: str, result) -> str:
    """What the row says of how the move to `label` ended: completed, failed
    with the cause of its failure, or cancelled, abandoned or expired."""
    status = result.status.name
    if status == "FAILED":
        return f"move to {label} failed: {cause(result.failure.message)}"
    return f"move to {label} {status.lower()}"
