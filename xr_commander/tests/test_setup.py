import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers import default_parameters
from xr_commander import __main__ as wiring


@pytest.mark.asyncio
async def test_setup_announces_the_bound_page_and_starts_every_loop(monkeypatch):
    runner = MagicMock()
    session = MagicMock()
    session.bound_address.return_value = ("0.0.0.0", 4443)
    monkeypatch.setattr(wiring.peppygen.clock, "init", AsyncMock())
    monkeypatch.setattr(wiring.tls, "ensure_certificate", lambda _: ("cert", "key"))
    monkeypatch.setattr(wiring, "XrSession", MagicMock(return_value=session))
    monkeypatch.setattr(wiring.alerts_topic, "bound_producers", lambda _: [])
    monkeypatch.setattr(wiring.motor_health_topic, "bound_producers", lambda _: [])
    for operation in (
        "run_posture_button",
        "drain_pose_states",
        "stream_pose",
        "stream_gripper",
    ):
        monkeypatch.setattr(wiring.publish, operation, AsyncMock())
    for operation in (
        "follow_camera_panels",
        "drain_frames",
        "watch_camera_silence",
        "feed_panels",
    ):
        monkeypatch.setattr(wiring.video, operation, AsyncMock())
    monkeypatch.setattr(wiring.alerts, "drain_alerts", AsyncMock())
    monkeypatch.setattr(wiring.alerts, "follow_producers", AsyncMock())
    monkeypatch.setattr(wiring.record, "run_recorder_buttons", AsyncMock())
    health = AsyncMock()
    monkeypatch.setattr(wiring.motor_health, "drain_motor_health", health)
    monkeypatch.setattr(
        wiring.addresses,
        "page_urls",
        lambda host, port: [f"https://198.51.100.7:{port}"],
    )
    page = AsyncMock()
    monkeypatch.setattr(wiring.page, "publish_page", page)

    tasks = await wiring.setup(default_parameters(status_panel_enabled=False), runner)
    try:
        await asyncio.gather(*tasks)
        session.start.assert_called_once()
        # The page is announced with the address the server bound, after the
        # server is up, so the daemon reports a socket that answers.
        runner.announce_endpoint.assert_called_once_with(
            "page", "https", "0.0.0.0", 4443
        )
        # Every slot is drained from the start: a producer that joins later
        # reaches the panel through the subscription following the set.
        health.assert_awaited_once()
        assert not health.call_args.args[2].producers_bound
        # The page's URLs go out once, read from the address the server
        # bound.
        page.assert_awaited_once()
        assert page.await_args.args[2] == ["https://198.51.100.7:4443"]
        # Without the status panel no video source is reserved; the session
        # starts with none and the cameras' panels are added as they bind.
        assert wiring.XrSession.call_args.kwargs["video_sources"] == {}
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
