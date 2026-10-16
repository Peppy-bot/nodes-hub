"""The robot choice behind the page's Robot card: the robots paired into
the headset and the one the hands drive.

teleop_xr's page carries the card. As it loads, the card reads `GET /robot`
and lists the paired robots in name order with the chosen one selected,
or says `NO_ROBOT_TEXT` while there is none; a pick posts to `POST /robot`
as the form field `robot`, and the card shows the text of a refusal.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from xr_commander.bus import log
from xr_commander.forms import BAD_REQUEST, NO_CONTENT, form_field, read_capped_body
from xr_commander.robots import PairedRobots

ROBOT_PATH = "/robot"
ROBOT_FIELD = "robot"
JOIN_HINT = (
    "Join one with robot_commander=none:"
    " peppy stack join OPTION:NAME --with robot_commander=none"
)
NO_ROBOT_TEXT = f"No robot is paired with the headset. {JOIN_HINT}"


class RobotChoice(TypedDict):
    """What the card reads: the paired robots, the one driven, and what to
    say while there is none."""

    robots: list[str]
    chosen: str | None
    empty: str


def robot_choice(names: Sequence[str], chosen: str | None) -> RobotChoice:
    return {"robots": list(names), "chosen": chosen, "empty": NO_ROBOT_TEXT}


def build_router(robots: PairedRobots) -> APIRouter:
    """The card's routes, over the robots as they stand at each request."""
    router = APIRouter()

    @router.get(ROBOT_PATH)
    async def show_robots() -> JSONResponse:
        # Read at each load: a browser never shows an offer from before.
        return JSONResponse(
            robot_choice(*robots.standing()), headers={"Cache-Control": "no-store"}
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
