import json
from pathlib import Path

import teleop_xr
from fastapi import FastAPI
from teleop_xr.config import TeleopSettings
from xr_commander.robot_choice import JOIN_HINT, NO_ROBOT_TEXT, ROBOT_PATH, build_router
from xr_commander.xr import include_router_first

from tests.helpers import FORM, FakeRobots, asgi_request


def app_over(robots) -> FastAPI:
    app = FastAPI()
    include_router_first(app, build_router(robots))
    return app


def test_the_offer_lists_the_paired_robots_and_the_chosen_one():
    served = asgi_request(
        app_over(FakeRobots(["alpha", "bravo"], "bravo")), "GET", ROBOT_PATH
    )
    assert served.status == 200
    assert json.loads(served.body) == {
        "robots": ["alpha", "bravo"],
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
    assert json.loads(asgi_request(app, "GET", ROBOT_PATH).body)["robots"] == ["alpha"]
    robots._names.append("bravo")
    assert json.loads(asgi_request(app, "GET", ROBOT_PATH).body)["robots"] == [
        "alpha",
        "bravo",
    ]


def test_the_routes_outrank_the_page_teleop_xr_serves():
    # The installed teleop_xr mounts its page at '/', which matches every
    # path; the card's routes come first, and the page keeps the rest.
    app = teleop_xr.Teleop(TeleopSettings()).app
    include_router_first(app, build_router(FakeRobots(["alpha"], "alpha")))
    assert json.loads(asgi_request(app, "GET", ROBOT_PATH).body)["robots"] == ["alpha"]
    assert "<title>TeleopXR</title>" in asgi_request(app, "GET", "/").body
    assert asgi_request(app, "GET", "/config").status == 200


def test_the_installed_page_reads_the_cards_route():
    # The page's bundle, as the installed teleop_xr ships it, reads and
    # posts the route served here; a fork that moves it fails here, ahead
    # of node boot.
    chunks = Path(teleop_xr.__file__).with_name("dist") / "_next" / "static" / "chunks"
    assert any(
        f'"{ROBOT_PATH}"' in path.read_text(encoding="utf-8")
        for path in chunks.glob("*.js")
    )


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
