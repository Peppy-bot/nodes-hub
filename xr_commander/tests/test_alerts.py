import asyncio
from types import SimpleNamespace
from unittest import mock

import cv2
import pytest

from tests.helpers import ClosingSubscription, FakeToken, FakeTopic
from xr_commander.alerts import (
    ALERT_STALE_AFTER_MS,
    ActiveAlerts,
    drain_alerts,
    fit_text,
)

STALE_AFTER_S = ALERT_STALE_AFTER_MS / 1000.0


def _wire(source, severity, kind="motor_condition", message="holding 96%"):
    return SimpleNamespace(
        source=source,
        kind=kind,
        severity=severity,
        message=message,
    )


def raise_alert(
    active,
    source="left arm j2",
    kind="motor_overload",
    severity=2,
    message="holding 96% of rated torque",
):
    active.update(source, kind, severity, message)


def test_an_alert_names_its_source_severity_and_message():
    active = ActiveAlerts()
    raise_alert(active)
    (alert,) = active.active()
    assert alert.text == "LEFT ARM J2 CRITICAL: holding 96% of rated torque"
    assert alert.severity == 2


def test_an_undefined_severity_is_refused():
    # The contract tops out at 3; accepting more would let a malformed
    # producer outrank a genuine fault.
    with pytest.raises(ValueError):
        ActiveAlerts().update(
            "left arm j2", "motor_overload", 4, "boom"
        )


def test_an_alert_without_an_identity_is_refused():
    # The commander rejects these too; a panel line " CRITICAL: ..." names
    # nothing the operator can act on.
    with pytest.raises(ValueError):
        ActiveAlerts().update("", "motor_overload", 2, "boom")
    with pytest.raises(ValueError):
        ActiveAlerts().update("left arm j2", "", 2, "boom")


def test_a_severity_zero_clear_removes_the_alert():
    active = ActiveAlerts()
    raise_alert(active)
    assert active.active()
    active.update("left arm j2", "motor_overload", 0, "recovered")
    assert active.active() == ()


def test_a_clear_only_removes_its_own_identity():
    active = ActiveAlerts()
    raise_alert(active, source="left arm j2")
    raise_alert(active, source="right arm j1", severity=1, message="warm")
    active.update("left arm j2", "motor_overload", 0, "recovered")
    (alert,) = active.active()
    assert alert.text.startswith("RIGHT ARM J1")
    assert alert.severity == 1


def test_a_re_emit_refreshes_the_staleness_clock(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr("xr_commander.alerts.time.monotonic", lambda: clock["now"])
    active = ActiveAlerts()
    raise_alert(active)
    clock["now"] += STALE_AFTER_S - 0.1
    raise_alert(active)  # a contract-cadence re-emit
    clock["now"] += STALE_AFTER_S - 0.1
    assert active.active()


def test_every_active_alert_is_listed_worst_first():
    active = ActiveAlerts()
    raise_alert(active, source="left arm j2", severity=1, message="warm")
    raise_alert(active, source="right arm j1", severity=3, message="limp")
    raise_alert(active, source="right arm j4", severity=2, message="hot")
    assert [a.severity for a in active.active()] == [3, 2, 1]


def test_equal_severities_keep_a_stable_order():
    # Re-drawn a few times a second: an unstable order would make the list
    # flicker between identical states.
    def listed():
        active = ActiveAlerts()
        for source in ("right arm j4", "left arm j2", "middle j9"):
            raise_alert(active, source=source, severity=1, message="warm")
        return [a.text for a in active.active()]

    assert listed() == sorted(listed())
    assert listed() == listed()


def test_a_stale_alert_leaves_the_list_too(monkeypatch):
    # The panel purges dead producers' entries, so a quiet producer cannot
    # leave a line the operator reads as current.
    clock = {"now": 100.0}
    monkeypatch.setattr("xr_commander.alerts.time.monotonic", lambda: clock["now"])
    active = ActiveAlerts()
    raise_alert(active)
    clock["now"] += STALE_AFTER_S + 0.1
    assert active.active() == ()


def test_drain_alerts_survives_a_slot_it_cannot_subscribe_to():
    # Every camera drain already fails this way. An alert drain that raises
    # instead takes its own failure out of the task and leaves no surface.
    class Refuses:
        async def subscribe(self, _runner):
            raise RuntimeError("no such slot")

    active = ActiveAlerts()
    asyncio.run(drain_alerts(object(), Refuses(), active, FakeToken()))
    assert active.active() == ()


def test_drain_alerts_keeps_every_producers_alerts():
    stream = [
        (SimpleNamespace(instance_id="left_arm"), _wire("left arm j2", 2)),
        (SimpleNamespace(instance_id="right_arm_inst"), _wire("right arm j5", 3)),
    ]
    active = ActiveAlerts()
    asyncio.run(
        drain_alerts(
            object(), FakeTopic(ClosingSubscription(stream)), active, FakeToken()
        )
    )
    assert [a.severity for a in active.active()] == [3, 2]


def test_a_malformed_producer_is_reported_once_not_every_re_emit():
    # A shared latch is cleared by any other producer's good message, so one
    # bad arm would log on every re-emit forever on a two-arm stack.
    bad = SimpleNamespace(instance_id="left_arm")
    good = SimpleNamespace(instance_id="right_arm_inst")
    stream = []
    for _ in range(5):
        stream.append((bad, _wire("left arm j2", 9)))
        stream.append((good, _wire("right arm j5", 1)))

    logged: list[str] = []
    active = ActiveAlerts()
    with mock.patch("xr_commander.bus.log", side_effect=lambda m: logged.append(m)):
        asyncio.run(
            drain_alerts(
                object(), FakeTopic(ClosingSubscription(stream)), active, FakeToken()
            )
        )
    unusable = [m for m in logged if "unusable" in m]
    assert len(unusable) == 1, f"logged {len(unusable)} times: {unusable}"


def test_a_long_message_is_truncated_rather_than_shrunk_to_a_smear():
    # Shrinking without a floor turns a long message into a one-pixel smear
    # carrying no information.
    font = cv2.FONT_HERSHEY_SIMPLEX
    long_text = "LEFT ARM J2 CRITICAL: " + "x" * 800
    body, scale = fit_text(long_text, font, 1.0, 2, 600)
    assert scale >= 0.5, "the glyphs stay legible"
    assert len(body) < len(long_text), "the message is truncated"
    assert body.startswith("LEFT ARM J2 CRITICAL:"), "the joint survives"
    assert cv2.getTextSize(body, font, scale, 2)[0][0] <= 600


def test_a_message_that_fits_is_left_exactly_as_it_is():
    font = cv2.FONT_HERSHEY_SIMPLEX
    body, scale = fit_text("J2 HOT", font, 1.0, 2, 600)
    assert (body, scale) == ("J2 HOT", 1.0)


def test_truncation_only_when_the_whole_text_cannot_fit_at_the_floor():
    # A single proportional estimate under-shrinks when glyph widths round
    # up, truncating messages that would fit whole a hair smaller. Sweep
    # the widths: any truncation must be forced, not an estimation artifact.
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "RIGHT ARM J4 CRITICAL: winding 91 C"
    floor = 0.9 * 0.55
    for available in range(60, 620, 4):
        body, _scale = fit_text(text, font, 0.9, 2, available)
        if body != text:
            width_at_floor = cv2.getTextSize(text, font, floor, 2)[0][0]
            assert width_at_floor > available, (
                f"truncated at {available}px though the whole text fits at the floor"
            )


def test_equal_severity_alerts_keep_their_order_when_a_reading_ticks():
    # Sorting on the rendered text swaps two rows whenever a measurement
    # changes, because the producer re-emits the live number.
    active = ActiveAlerts()
    active.update("left arm j2", "a", 1, "80 C")
    active.update("left arm j2", "b", 1, "79 C")
    before = [a.text for a in active.active()]
    active.update("left arm j2", "b", 1, "78 C")
    after = [a.text for a in active.active()]
    assert [t.rsplit(": ", 1)[-1] for t in before] == ["80 C", "79 C"]
    assert [t.rsplit(": ", 1)[-1] for t in after] == [
        "80 C",
        "78 C",
    ], "identity holds the position, not the text"


def test_an_unwired_alert_slot_is_distinguishable_from_a_quiet_robot():
    unwired = ActiveAlerts(producers_bound=False)
    wired = ActiveAlerts(producers_bound=True)
    assert unwired.active() == wired.active() == ()
    assert not unwired.producers_bound
    assert wired.producers_bound


def test_an_alert_lives_through_the_whole_stale_window(monkeypatch):
    # The window is inclusive: an alert dies after it, not at it, so an
    # arrival landing exactly on the boundary tick never flickers.
    clock = {"now": 100.0}
    monkeypatch.setattr("xr_commander.alerts.time.monotonic", lambda: clock["now"])
    active = ActiveAlerts()
    active.update("left arm j2", "motor_condition", 2, "hot")
    clock["now"] += STALE_AFTER_S
    assert len(active.active()) == 1
    clock["now"] += 0.001
    assert active.active() == ()
