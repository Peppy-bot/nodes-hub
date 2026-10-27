import asyncio
import contextlib
import json
from pathlib import Path

import teleop_xr
from fastapi import FastAPI
from teleop_xr.config import TeleopSettings
from xr_commander.robot_choice import (
    EVENTS_PATH,
    NO_ROBOT_TEXT,
    ROBOT_PATH,
    STATES_PERIOD_S,
    RobotStates,
    build_router,
    follow_robot_states,
    robot_choice,
)
from xr_commander.robots import JOIN_HINT
from xr_commander.xr import include_router_first

from tests.helpers import (
    FORM,
    FakeRobots,
    FakeToken,
    ManualClock,
    asgi_request,
)


def states_of(robots, states=None) -> RobotStates:
    """The page's publisher over `robots`, with each robot's state looked up
    in `states`."""
    states = {} if states is None else states
    return RobotStates(robots, state_of=lambda name: states.get(name, ""))


def app_over(robots, states=None) -> FastAPI:
    app = FastAPI()
    include_router_first(app, build_router(robots, states or states_of(robots)))
    return app


def names_of(served) -> list[str]:
    return [robot["name"] for robot in json.loads(served.body)["robots"]]


def test_the_offer_lists_the_paired_robots_their_states_and_the_chosen_one():
    robots = FakeRobots(["alpha", "bravo"], "bravo")
    states = states_of(
        robots, {"alpha": "free", "bravo": "leased by director_inst (tidy the table)"}
    )
    states.refresh()
    served = asgi_request(app_over(robots, states), "GET", ROBOT_PATH)
    assert served.status == 200
    assert json.loads(served.body) == {
        "robots": [
            {"name": "alpha", "state": "free"},
            {"name": "bravo", "state": "leased by director_inst (tidy the table)"},
        ],
        "chosen": "bravo",
        "empty": NO_ROBOT_TEXT,
    }
    # A browser never shows an offer from before.
    assert served.headers["cache-control"] == "no-store"


def test_the_offer_says_what_to_type_while_nothing_is_paired():
    served = json.loads(asgi_request(app_over(FakeRobots()), "GET", ROBOT_PATH).body)
    assert served["robots"] == [] and served["chosen"] is None
    assert served["empty"] == NO_ROBOT_TEXT
    assert "peppy stack join OPTION:NAME --with robot_commander=none" in NO_ROBOT_TEXT


def test_the_offer_is_the_robots_as_they_stand_at_each_request():
    robots = FakeRobots(["alpha"], "alpha")
    app = app_over(robots)
    assert names_of(asgi_request(app, "GET", ROBOT_PATH)) == ["alpha"]
    robots._names.append("bravo")
    assert names_of(asgi_request(app, "GET", ROBOT_PATH)) == ["alpha", "bravo"]


def test_the_routes_outrank_the_page_teleop_xr_serves():
    # The installed teleop_xr mounts its page at '/', which matches every
    # path; the card's routes come first, and the page keeps the rest.
    app = teleop_xr.Teleop(TeleopSettings()).app
    robots = FakeRobots(["alpha"], "alpha")
    include_router_first(app, build_router(robots, states_of(robots)))
    assert names_of(asgi_request(app, "GET", ROBOT_PATH)) == ["alpha"]
    assert "<title>TeleopXR</title>" in asgi_request(app, "GET", "/").body
    assert asgi_request(app, "GET", "/config").status == 200


def test_the_installed_page_reads_the_cards_route():
    # The page's bundle, as the installed teleop_xr ships it, reads and
    # posts the route served here; a fork that moves it fails here, ahead
    # of node boot.
    chunks = Path(teleop_xr.__file__).with_name("dist") / "_next" / "static" / "chunks"
    bundle = [path.read_text(encoding="utf-8") for path in chunks.glob("*.js")]
    assert any(f'"{ROBOT_PATH}"' in chunk for chunk in bundle)
    assert any(f'"{EVENTS_PATH}"' in chunk for chunk in bundle)


def test_choosing_a_paired_robot_drives_it():
    robots = FakeRobots(["alpha", "bravo"], "alpha")
    served = asgi_request(
        app_over(robots), "POST", ROBOT_PATH, body=b"robot=bravo", headers=FORM
    )
    assert served.status == 204 and served.body == ""
    assert robots.chosen() == "bravo"


def test_choosing_an_unpaired_robot_is_refused_naming_the_paired_ones():
    robots = FakeRobots(["alpha"], "alpha")
    served = asgi_request(
        app_over(robots), "POST", ROBOT_PATH, body=b"robot=delta", headers=FORM
    )
    assert served.status == 400
    assert (
        "'delta' is not paired" in served.body and "paired robots: alpha" in served.body
    )
    assert robots.chosen() == "alpha"


def test_a_pick_while_nothing_is_paired_says_what_to_type():
    served = asgi_request(
        app_over(FakeRobots()), "POST", ROBOT_PATH, body=b"robot=delta", headers=FORM
    )
    assert served.status == 400
    assert (
        served.body
        == f"robot 'delta' is not paired with the headset; none is. {JOIN_HINT}"
    )


def test_a_malformed_choice_is_refused():
    robots = FakeRobots(["alpha"], "alpha")
    served = asgi_request(
        app_over(robots), "POST", ROBOT_PATH, body=b"nothing=1", headers=FORM
    )
    assert served.status == 400 and "one robot field" in served.body
    assert robots.choices == []


async def stream_events(app, path: str, count: int, between=None) -> list[dict]:
    """The first `count` events of the stream route at `path`, as the page
    reads them; `between()` runs after each event but the last, then the
    page goes away."""
    events, _ = await stream_events_and_headers(app, path, count, between)
    return events


async def stream_events_and_headers(app, path: str, count: int, between=None):
    """[`stream_events`], with the headers the stream starts with."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 4443),
    }
    events: list[dict] = []
    headers: dict[str, str] = {}
    gone = asyncio.Event()
    asked = False

    async def receive():
        nonlocal asked
        if not asked:
            asked = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            headers.update(
                (key.decode(), value.decode()) for key, value in message["headers"]
            )
        if message["type"] != "http.response.body" or not message.get("body"):
            return
        text = message["body"].decode()
        assert text.startswith("data: ") and text.endswith("\n\n")
        events.append(json.loads(text.removeprefix("data: ")))
        if len(events) == count:
            gone.set()
        elif between is not None:
            between()

    await asyncio.wait_for(app(scope, receive, send), 10.0)
    return events, headers


async def test_the_events_start_with_the_robots_as_they_stand_then_follow_each_change():
    robots = FakeRobots(["alpha"], "alpha")
    held = {"alpha": "free"}
    states = states_of(robots, held)
    states.refresh()

    def change():
        held["alpha"] = "driven from this headset"
        states.refresh()

    events = await stream_events(app_over(robots, states), EVENTS_PATH, 2, change)
    assert events[0]["robots"] == [{"name": "alpha", "state": "free"}]
    assert events[1] == robot_choice(
        ["alpha"], "alpha", {"alpha": "driven from this headset"}
    )


async def test_a_page_that_falls_behind_gets_the_latest_robots_only():
    robots = FakeRobots(["alpha"], "alpha")
    held = {"alpha": "free"}
    states = states_of(robots, held)
    queue = states.follow()
    for state in ("leased by director_inst (tidy the table)", "free"):
        held["alpha"] = state
        states.refresh()
    await asyncio.sleep(0)
    assert queue.qsize() == 1
    assert (await queue.get())["robots"][0]["state"] == "free"
    states.unfollow(queue)


async def test_the_events_are_server_sent_and_a_page_that_leaves_is_followed_no_more():
    robots = FakeRobots(["alpha"], "alpha")
    states = states_of(robots)
    states.refresh()
    _, headers = await stream_events_and_headers(
        app_over(robots, states), EVENTS_PATH, 1
    )
    assert headers["content-type"].startswith("text/event-stream")
    assert not states._followers


async def test_a_stream_ends_as_the_node_stops_whatever_changes_after():
    robots = FakeRobots(["alpha"], "alpha")
    held = {"alpha": "free"}
    states = states_of(robots, held)
    queue = states.follow()
    states.close()
    # A change lands in the same batch of the page server's loop.
    held["alpha"] = "driven from this headset"
    states.refresh()
    await asyncio.sleep(0)
    assert await queue.get() is None


async def test_the_stream_route_ends_as_the_node_stops():
    robots = FakeRobots(["alpha"], "alpha")
    states = states_of(robots)
    states.refresh()
    events = await stream_events(
        app_over(robots, states), EVENTS_PATH, count=2, between=states.close
    )
    assert len(events) == 1


async def test_the_node_sends_a_pair_that_joins_and_a_choice_within_a_period():
    clock = ManualClock()
    robots = FakeRobots(["alpha"], "alpha")
    states = states_of(robots)
    queue = states.follow()
    token = FakeToken()
    task = asyncio.create_task(follow_robot_states(states, token, clock))
    try:
        await clock.advance(STATES_PERIOD_S, parked=[task])
        first = await asyncio.wait_for(queue.get(), 10.0)
        assert [robot["name"] for robot in first["robots"]] == ["alpha"]
        # Nothing changed: nothing is sent.
        await clock.advance(STATES_PERIOD_S, parked=[task])
        assert queue.empty()
        robots._names.append("bravo")
        robots.choose("bravo")
        await clock.advance(STATES_PERIOD_S, parked=[task])
        second = await asyncio.wait_for(queue.get(), 10.0)
        assert second["chosen"] == "bravo"
        assert [robot["name"] for robot in second["robots"]] == ["alpha", "bravo"]
    finally:
        token.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
