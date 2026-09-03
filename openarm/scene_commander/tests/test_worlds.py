"""The robot menu's wire: the panel boots under the generated harness
against mocked engines and its HTTP API is driven the way the browser
drives it. With the worlds slot vacant (an engine that keeps one world for
its life) the menu is reported unsupported and a switch refused before
anything reaches the wire; with the slot bound, the worlds come from the
engine's get_worlds_list and a switch is one load_world goal, answered as
the engine answers it."""

import asyncio
import json
import socket

import aiohttp

from openarm_scene_commander.__main__ import setup
from peppygen.consumed_actions.worlds import load_world
from peppygen.consumed_services.simulation import get_assets_list, get_objects_list
from peppygen.consumed_services.worlds import get_worlds_list
from peppygen.fixtures import harness
from peppygen.parameters import Parameters

WAIT_S = 30.0

WORLDS = [
    {"world_id": "openarm_v2", "display_name": "OpenArm v2 on its pedestal", "robot": "openarm_v2", "current": True},
    {"world_id": "aloha", "display_name": "Aloha 2 on its table", "robot": "aloha", "current": False},
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def params(port: int) -> Parameters:
    return Parameters(http_host="127.0.0.1", http_port=port)


async def answer_scene_boot(h) -> None:
    """The panel's setup polls the scene catalogue and the objects before
    it serves; the engine mock answers both."""
    responder = await h.mocks.deps.simulation.get_assets_list.next_request(WAIT_S)
    await responder.respond(
        get_assets_list.ResponseData(success=True, message="1 assets available", assets_json="[]")
    )
    responder = await h.mocks.deps.simulation.get_objects_list.next_request(WAIT_S)
    await responder.respond(
        get_objects_list.ResponseData(success=True, message="0 runtime objects", objects_json="[]")
    )


async def answer_worlds(h, worlds: list[dict]) -> None:
    responder = await h.mocks.deps.worlds.get_worlds_list.next_request(WAIT_S)
    await responder.respond(
        get_worlds_list.ResponseData(
            success=True,
            message=f"{len(worlds)} worlds",
            worlds_json=json.dumps(worlds),
        )
    )


async def wait_for_server(port: int) -> None:
    """The panel's setup returns before its server task has bound the
    port: retry the health endpoint until it answers (bounded)."""
    deadline = asyncio.get_running_loop().time() + WAIT_S
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(f"http://127.0.0.1:{port}/api/health") as response:
                    assert response.status == 200
                    return
            except aiohttp.ClientConnectorError:
                assert asyncio.get_running_loop().time() < deadline, "the panel never served"
                await asyncio.sleep(0.05)


async def get(port: int, path: str) -> tuple[int, dict]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{port}{path}") as response:
            return response.status, await response.json()


async def post(port: int, path: str, body: dict) -> tuple[int, dict]:
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://127.0.0.1:{port}{path}", json=body) as response:
            return response.status, await response.json()


async def test_a_vacant_worlds_slot_reports_no_robot_menu():
    port = free_port()
    async with harness.start(setup, parameters=params(port), worlds_vacant=True) as h:
        await answer_scene_boot(h)
        assert h.mocks.deps.worlds is None
        await wait_for_server(port)

        status, worlds = await get(port, "/api/worlds")
        assert status == 200
        assert worlds == {"success": True, "supported": False, "worlds": [], "count": 0}

        # A switch never reaches the wire: there is no producer to send it to.
        status, refused = await post(port, "/api/world/load", {"world_id": "aloha"})
        assert status == 400
        assert refused["success"] is False
        assert "does not implement world_control" in refused["message"]


async def test_a_bound_worlds_slot_lists_the_engine_worlds_and_switches():
    port = free_port()
    async with harness.start(setup, parameters=params(port)) as h:
        await answer_scene_boot(h)
        # The setup also reads the worlds once, to log them.
        await answer_worlds(h, WORLDS)
        await wait_for_server(port)

        listing = asyncio.create_task(get(port, "/api/worlds"))
        await answer_worlds(h, WORLDS)
        status, worlds = await listing
        assert status == 200
        assert worlds["supported"] is True
        assert worlds["count"] == 2
        assert [w["world_id"] for w in worlds["worlds"]] == ["openarm_v2", "aloha"]
        assert [w["current"] for w in worlds["worlds"]] == [True, False]

        # Switching is one load_world goal, completed as the engine completes it.
        switch = asyncio.create_task(post(port, "/api/world/load", {"world_id": "aloha"}))
        pending = await h.mocks.deps.worlds.load_world.next_goal(WAIT_S)
        assert pending.request.world_id == "aloha"
        active = await pending.accept()
        await active.complete(
            load_world.ResultResponseData(success=True, message="Loaded world 'aloha': robot 'aloha'")
        )
        status, switched = await switch
        assert status == 200
        assert switched == {"success": True, "message": "Loaded world 'aloha': robot 'aloha'"}

        # A refusal is the engine's reason, as a failed request.
        refusal = asyncio.create_task(post(port, "/api/world/load", {"world_id": "nope"}))
        pending = await h.mocks.deps.worlds.load_world.next_goal(WAIT_S)
        active = await pending.accept()
        await active.complete(
            load_world.ResultResponseData(success=False, message="unknown world 'nope'")
        )
        status, refused = await refusal
        assert status == 400
        assert refused == {"success": False, "message": "unknown world 'nope'"}
