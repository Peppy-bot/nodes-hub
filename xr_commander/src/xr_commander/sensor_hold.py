"""The chosen robot's sensors, held at their own rate in a simulation.

A simulation streams each sensor of a robot at 1 Hz until a holder holds it
over sim_sensor_streams, and at the sensor's own rate while one does. The
headset holds every sensor of the robot it drives, under the holder name
`headset`, so that the operator sees that robot's cameras at their own rate.
It drives the chosen robot while a headset page is in session: the page
sent an XR frame within `Timing.session_timeout_s`. A launcher deploys the
headset beside a simulation whether or not someone wears it, and the
chosen robot falls back to the first robot paired, so without a session
the headset holds nothing and the robots' cameras stay idle. The headset
renews the hold a third of the way through each lease the simulation
answers with. When the robot it drives changes, or it drives none, it
releases its hold on the robot it held before it holds the next one, and
it releases its hold when the node stops. On a stack without a simulation
the slot is vacant, and the headset holds nothing.

A refused or failed hold is logged once and tried again after
`Timing.retry_s`; until then the robot's sensors stream at 1 Hz, which the
camera panels still show.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from xr_commander.bus import CancellationToken, Latch, log, ticks
from xr_commander.devices import XrFrame

# The name the simulation lists the headset's holds under.
HOLDER = "headset"


@dataclass(frozen=True)
class Timing:
    """The hold's cadence and deadlines, injectable so that tests drive a
    clock of their own."""

    # How often the chosen robot is read: a new choice reaches the
    # simulation within this.
    follow_period_s: float = 0.2
    # A hold or a release is one round trip to the simulation.
    call_timeout_s: float = 2.0
    # The wait before a refused or failed hold is tried again.
    retry_s: float = 1.0
    # How long after its last XR frame a headset page still counts as in
    # session: a page reload or a short gap in tracking keeps the hold.
    session_timeout_s: float = 5.0


@dataclass(frozen=True)
class Answer:
    """What the simulation answered to a hold or a release."""

    success: bool
    message: str
    # How long a hold lasts from the call that made or renewed it (ms).
    lease_ms: int


class SensorStreams:
    """The hold and release services of the simulation bound to the slot."""

    def __init__(
        self,
        node_runner,
        target,
        hold_module: ModuleType,
        release_module: ModuleType,
        timeout_s: float,
    ) -> None:
        self._node_runner = node_runner
        self._target = target
        self._hold = hold_module
        self._release = release_module
        self._timeout_s = timeout_s

    async def hold(self, robot: str) -> Answer:
        """Holds every sensor of `robot`, or renews that hold."""
        request = self._hold.Request(robot=robot, holder=HOLDER, sensors=[])
        response = await self._hold.poll(
            self._node_runner, self._target, request, self._timeout_s
        )
        return _answer(response.data)

    async def release(self, robot: str) -> Answer:
        """Ends the headset's hold on `robot`."""
        request = self._release.Request(robot=robot, holder=HOLDER)
        response = await self._release.poll(
            self._node_runner, self._target, request, self._timeout_s
        )
        return _answer(response.data)


def _answer(data) -> Answer:
    return Answer(success=data.success, message=data.message, lease_ms=data.lease_ms)


class ChosenRobotHold:
    """The headset's hold: the robot whose sensors it holds, when the hold
    is renewed, and its release when the choice changes or the node stops.

    Calls run one at a time on the node's loop: the shutdown hook waits for
    a call in flight, so that no hold lands after the release.
    """

    def __init__(
        self,
        streams: SensorStreams,
        timing: Timing = Timing(),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.timing = timing
        self._streams = streams
        self._monotonic = monotonic
        # The robot whose sensors the headset holds or tries to hold. A
        # change releases it even when its hold was refused, because a hold
        # whose answer was lost can still have landed.
        self._robot: str | None = None
        # Whether the simulation accepted a hold of `_robot` since it
        # became the robot.
        self._is_held = False
        self._renew_at = 0.0
        self._calls = asyncio.Lock()
        self._stopped = False
        self._failing = Latch()

    async def follow(self, chosen: str | None) -> None:
        """One step towards a hold of `chosen`'s sensors, and of no other
        robot's: a changed choice releases the robot held before, and the
        hold of `chosen` is made or renewed when it is due."""
        async with self._calls:
            if self._stopped:
                return
            if chosen != self._robot:
                await self._release_held()
                self._robot = chosen
                self._renew_at = self._monotonic()
            if self._robot is not None and self._monotonic() >= self._renew_at:
                await self._hold(self._robot)

    async def stop(self) -> None:
        """Releases the hold for good: the node's shutdown hook."""
        async with self._calls:
            self._stopped = True
            await self._release_held()

    async def _hold(self, robot: str) -> None:
        # Read before the call: the lease runs from the simulation's answer,
        # so the renewal lands early rather than late.
        now = self._monotonic()
        try:
            answer = await self._streams.hold(robot)
        except Exception as e:
            self._failing.trip(f"sensor hold of {robot} failed: {e!r}")
            self._renew_at = now + self.timing.retry_s
            return
        if not answer.success:
            self._failing.trip(f"sensor hold of {robot} refused: {answer.message}")
            self._renew_at = now + self.timing.retry_s
            return
        self._failing.clear()
        if not self._is_held:
            log(f"holding the sensors of {robot} at their own rate")
            self._is_held = True
        renew_after_s = max(answer.lease_ms / 3000.0, self.timing.follow_period_s)
        self._renew_at = now + renew_after_s

    async def _release_held(self) -> None:
        robot, was_held = self._robot, self._is_held
        self._robot, self._is_held = None, False
        self._failing.clear()
        if robot is None:
            return
        # Not tried again: an unreleased hold lapses with its lease.
        try:
            answer = await self._streams.release(robot)
        except Exception as e:
            log(f"sensor release of {robot} failed: {e!r}; the hold lapses")
            return
        if not answer.success:
            log(f"sensor release of {robot} refused: {answer.message}; the hold lapses")
            return
        if was_held:
            log(f"released the sensors of {robot}")


def driven_robot(
    chosen: Callable[[], str | None],
    latest_frame: Callable[[], XrFrame | None],
    session_timeout_s: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> Callable[[], str | None]:
    """The robot the headset drives, read live: the chosen robot while a
    headset page is in session, its latest XR frame at most
    `session_timeout_s` old on `monotonic`, the clock the frames are stamped
    on; None otherwise."""

    def robot() -> str | None:
        frame = latest_frame()
        if frame is None:
            return None
        if monotonic() - frame.received_monotonic_s > session_timeout_s:
            return None
        return chosen()

    return robot


async def follow_chosen_robot(
    hold: ChosenRobotHold,
    driven: Callable[[], str | None],
    token: CancellationToken,
) -> None:
    """Keep the hold on the robot the headset drives until the token
    cancels; the shutdown hook then releases it."""
    async for _ in ticks(hold.timing.follow_period_s, token):
        await hold.follow(driven())
