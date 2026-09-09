import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers import default_parameters
from xr_commander import __main__ as wiring


@pytest.mark.asyncio
@pytest.mark.parametrize("producers", [(), ("alpha_left_arm_inst", "alpha_right_gripper_inst")])
async def test_setup_starts_health_drain_only_for_its_configured_producers(monkeypatch, producers):
    runner = MagicMock()
    session = MagicMock()
    monkeypatch.setattr(wiring.peppygen.clock, "init", AsyncMock())
    monkeypatch.setattr(wiring.video, "discover_tracks", lambda *_: [])
    monkeypatch.setattr(wiring.tls, "ensure_certificate", lambda _: ("cert", "key"))
    monkeypatch.setattr(wiring, "XrSession", MagicMock(return_value=session))
    monkeypatch.setattr(wiring.alerts_topic, "bound_producers", lambda _: [])
    monkeypatch.setattr(wiring.recorder_record_episode, "bound_producers", lambda _: [])
    monkeypatch.setattr(wiring.motor_health_topic, "bound_producers",
                        lambda _: [SimpleNamespace(instance_id=name) for name in producers])
    for operation in ("run_posture_button", "drain_pose_states", "stream_pose", "stream_gripper"):
        monkeypatch.setattr(wiring.publish, operation, AsyncMock())
    health = AsyncMock()
    monkeypatch.setattr(wiring.motor_health, "drain_motor_health", health)

    tasks = await wiring.setup(default_parameters(status_panel_enabled=False), runner)
    try:
        await asyncio.gather(*tasks)
        session.start.assert_called_once()
        assert health.await_count == bool(producers)
        if producers:
            assert health.call_args.args[2].producers_bound
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
