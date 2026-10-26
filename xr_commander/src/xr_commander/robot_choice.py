"""The robot choice behind the page's Robot card: the robots paired into
the headset, who holds each, and the one the hands drive.

teleop_xr's page carries the card. As it loads, the card reads `GET /robot`
and lists the paired robots in name order, each with its state in the words
of `fleet.state_text`, the halt, or why the headset cannot ask the fleet,
with the chosen one selected, or says `NO_ROBOT_TEXT` while there is none. It then follows
`GET /robot/events`, a stream of server-sent events, each the same JSON as
`GET /robot`, sent whenever a robot, its state or the choice changes. A pick
posts to `POST /robot` as the form field `robot`, and the card shows the
text of a refusal.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TypedDict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from xr_commander.bus import REAL_CLOCK, CancellationToken, Clock, log, ticks
from xr_commander.forms import BAD_REQUEST, NO_CONTENT, form_field, read_capped_body
from xr_commander.robots import JOIN_HINT, PairedRobots

ROBOT_PATH = "/robot"
EVENTS_PATH = "/robot/events"
ROBOT_FIELD = "robot"
NO_ROBOT_TEXT = f"No robot is paired with the headset. {JOIN_HINT}"

# How often the node reads the robots and the choice for a change to send;
# the driver sends each change of a hold or of the listing at once.
STATES_PERIOD_S = 0.2


class RobotOffer(TypedDict):
    """One paired robot on the card: its name and its state."""

    name: str
    state: str


class RobotChoice(TypedDict):
    """What the card reads: the paired robots, the one driven, and what to
    say while there is none."""

    robots: list[RobotOffer]
    chosen: str | None
    empty: str


def robot_choice(
    names: Sequence[str], chosen: str | None, states: Mapping[str, str]
) -> RobotChoice:
    return {
        "robots": [{"name": name, "state": states.get(name, "")} for name in names],
        "chosen": chosen,
        "empty": NO_ROBOT_TEXT,
    }


def _offer_latest(queue: asyncio.Queue, choice: RobotChoice | None) -> None:
    """Put `choice` in a one-place queue, in place of a choice not read
    yet: a page that falls behind gets the latest only. None ends the
    page's stream, and stays in the queue until the stream reads it."""
    if queue.full() and queue.get_nowait() is None:
        choice = None
    queue.put_nowait(choice)


class RobotStates:
    """The robots, their states and the choice as the card shows them, and
    the pages that follow them.

    `refresh` runs on the node's loop, at each change the driver makes and
    once a period; it sends what changed to each page's stream, which reads
    it on the page server's loop from a queue of its own."""

    def __init__(self, robots: PairedRobots, state_of: Callable[[str], str]) -> None:
        self._robots = robots
        self._state_of = state_of
        self._states: Mapping[str, str] = MappingProxyType({})
        self._shown: RobotChoice | None = None
        self._lock = threading.Lock()
        self._followers: dict[asyncio.Queue, asyncio.AbstractEventLoop] = {}

    def refresh(self) -> None:
        """Send the robots, their states and the choice to each page when
        they changed."""
        names, chosen = self._robots.standing()
        states = {name: self._state_of(name) for name in names}
        choice = robot_choice(names, chosen, states)
        if choice == self._shown:
            return
        self._shown = choice
        self._states = MappingProxyType(states)
        self._send(choice)

    def current(self) -> RobotChoice:
        """The robots and the choice as they stand, with the states last
        sent: for a page's request, on the page server's thread."""
        return robot_choice(*self._robots.standing(), self._states)

    def close(self) -> None:
        """End every page's stream, as the node stops."""
        self._send(None)

    def _send(self, choice: RobotChoice | None) -> None:
        with self._lock:
            followers = list(self._followers.items())
        for queue, loop in followers:
            loop.call_soon_threadsafe(_offer_latest, queue, choice)

    def follow(self) -> asyncio.Queue:
        """A queue that each sent choice reaches, on the running loop."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        with self._lock:
            self._followers[queue] = asyncio.get_running_loop()
        return queue

    def unfollow(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._followers.pop(queue, None)


async def follow_robot_states(
    states: RobotStates, token: CancellationToken, clock: Clock = REAL_CLOCK
) -> None:
    """Send the robots and the choice whenever they change: a pair that
    joins or leaves, or a robot chosen on the page."""
    async for _ in ticks(STATES_PERIOD_S, token, clock):
        states.refresh()


def _event(choice: RobotChoice) -> str:
    return f"data: {json.dumps(choice)}\n\n"


def build_router(robots: PairedRobots, states: RobotStates) -> APIRouter:
    """The card's routes, over the robots as they stand at each request."""
    router = APIRouter()

    @router.get(ROBOT_PATH)
    async def show_robots() -> JSONResponse:
        # Read at each load: a browser never shows an offer from before.
        return JSONResponse(states.current(), headers={"Cache-Control": "no-store"})

    @router.get(EVENTS_PATH)
    async def follow_robots() -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            queue = states.follow()
            try:
                yield _event(states.current())
                while (choice := await queue.get()) is not None:
                    yield _event(choice)
            finally:
                states.unfollow(queue)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    @router.post(ROBOT_PATH)
    async def choose_robot(request: Request) -> Response:
        try:
            name = form_field(
                request.headers.get("content-type", ""),
                await read_capped_body(request),
                ROBOT_FIELD,
            )
        except ValueError as e:
            return Response(str(e), status_code=BAD_REQUEST, media_type="text/plain")
        if not robots.choose(name):
            names = robots.names()
            paired = (
                f"paired robots: {', '.join(names)}"
                if names
                else f"none is. {JOIN_HINT}"
            )
            return Response(
                f"robot {name!r} is not paired with the headset; {paired}",
                status_code=BAD_REQUEST,
                media_type="text/plain",
            )
        log(f"robot chosen: {name}")
        return Response(status_code=NO_CONTENT)

    return router
