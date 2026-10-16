import asyncio
import json
import threading

import teleop_xr
from teleop_xr.config import TeleopSettings, ViewConfig

from tests.helpers import FORM, asgi_request, frontend_app
from xr_commander.task_page import PATH, TaskLabel, build_router
from xr_commander.xr import XrSession, include_router_first, wait_until_started


def test_a_router_added_first_outranks_the_frontend_mount(tmp_path):
    label = TaskLabel()
    app = frontend_app(tmp_path)
    include_router_first(app, build_router(label))

    assert asgi_request(app, "GET", PATH).status == 200
    posted = asgi_request(
        app, "POST", PATH, body=b"task=stack+the+blocks", headers=FORM
    )
    assert posted.status == 303
    assert label.current() == "stack the blocks"


def test_appending_the_router_instead_would_be_shadowed(tmp_path):
    # Guards the fix above: without the ordering the mount answers first and
    # the page is unreachable.
    app = frontend_app(tmp_path)
    app.include_router(build_router(TaskLabel()))

    assert asgi_request(app, "GET", PATH).status == 404


def test_the_frontend_still_answers_everything_else(tmp_path):
    # Only the page's own path is taken; the mount keeps the rest.
    (tmp_path / "index.html").write_text("<!doctype html>frontend")
    app = frontend_app(tmp_path)
    include_router_first(app, build_router(TaskLabel()))

    served = asgi_request(app, "GET", "/index.html")
    assert served.status == 200
    assert "frontend" in served.body


def test_the_wait_returns_once_the_server_reports_started():
    reports = iter([False, False, True])
    assert wait_until_started(lambda: next(reports), lambda: True, 1.0, tick_s=0.0)


def test_a_dead_server_thread_ends_the_wait_early():
    assert not wait_until_started(lambda: False, lambda: False, 30.0, tick_s=0.0)


def test_the_deadline_bounds_a_server_that_never_starts():
    assert not wait_until_started(lambda: False, lambda: True, 0.05, tick_s=0.0)


def test_a_panel_change_is_shown_on_the_running_servers_loop(tmp_path):
    # The server handles its pages on its own loop in its own thread; the
    # change runs there, between the page messages, never on the caller's,
    # through the fork's own method, with the session's own panel raised
    # above the cameras' row.
    from xr_commander import tls

    cert, key = tls.ensure_certificate(tmp_path)
    status = object()
    session = XrSession(
        host="127.0.0.1",
        port=0,
        tls_cert_path=cert,
        tls_key_path=key,
        video_sources={"status": status},
    )
    shown: list[tuple[asyncio.AbstractEventLoop, int, dict, dict]] = []
    done = threading.Event()
    replace = session._teleop.replace_video_sources

    async def record(sources, views):
        shown.append(
            (asyncio.get_running_loop(), threading.get_ident(), sources, views)
        )
        await replace(sources, views)
        done.set()

    session._teleop.replace_video_sources = record
    session.start()
    try:
        chest = object()
        session.show_views({"chest": chest})
        assert done.wait(5.0)
        ((loop, thread, sources, views),) = shown
        assert loop is session._server_loop()
        assert thread != threading.get_ident()
        assert sources == {"status": status, "chest": chest}
        assert views == {"status": ViewConfig(raised=True), "chest": ViewConfig()}
        # The next page to load is told the views, the session's own raised.
        told = json.loads(asgi_request(session._teleop.app, "GET", "/config").body)
        assert {
            name: view["raised"] for name, view in told["camera_views"].items()
        } == {
            "status": True,
            "chest": False,
        }
    finally:
        session.stop()
    # Stopped, there is no page to show a change on.
    session.show_views({})
    assert len(shown) == 1


def test_the_installed_teleop_xr_offers_what_the_session_uses():
    # The fork's surface the session is built on; a drift fails here, ahead
    # of node boot.
    assert ViewConfig(raised=True).raised and ViewConfig().device is None
    teleop = teleop_xr.Teleop(
        TeleopSettings(camera_views={"status": ViewConfig(raised=True)})
    )
    served = asgi_request(teleop.app, "GET", "/config")
    assert served.status == 200
    assert json.loads(served.body)["camera_views"]["status"]["raised"] is True
