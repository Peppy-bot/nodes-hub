from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from tests.helpers import asgi_request
from xr_commander.task_page import PATH, TaskLabel, build_router
from xr_commander.xr import include_router_first, wait_until_started

FORM = {"content-type": "application/x-www-form-urlencoded"}


def frontend_app(tmp_path) -> FastAPI:
    """A stand-in for teleop_xr's app: a frontend mounted at '/', which
    matches every path and shadows anything routed after it."""
    app = FastAPI()
    app.mount("/", StaticFiles(directory=tmp_path), name="frontend")
    return app


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
