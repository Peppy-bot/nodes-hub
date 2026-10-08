import asyncio

import cv2
import pytest

from peppygen.consumed_topics.alerts import alerts as alerts_topic

from tests.helpers import FakeToken, boot, eventually, running_drain
from xr_commander.alerts import (
    ALERT_STALE_AFTER_MS,
    ActiveAlerts,
    drain_alerts,
    fit_text,
    follow_producers,
)

STALE_AFTER_S = ALERT_STALE_AFTER_MS / 1000.0
LEFT_ARM = ("core", "left_arm_inst")
RIGHT_ARM = ("core", "right_arm_inst")


def _item(source, severity, kind="motor_condition", message="holding 96%"):
    return alerts_topic.MessageActiveItem(
        source=source, kind=kind, severity=severity, message=message
    )


def _wire(*active):
    """One real wire alert set, as a producer publishes it."""
    return alerts_topic.Message(timestamp=0.0, active=list(active))


def active_alerts(*, bound=(LEFT_ARM, RIGHT_ARM), monotonic=None):
    """An ActiveAlerts over a fixed membership."""
    held = list(bound)
    if monotonic is None:
        return ActiveAlerts(bound_now=lambda: held)
    return ActiveAlerts(bound_now=lambda: held, monotonic=monotonic)


def publish_set(active, producer=LEFT_ARM, *items):
    """`producer`'s set, as wire entries."""
    return active.replace(producer, items)


def raise_alert(
    active,
    source="left arm j2",
    kind="motor_overload",
    severity=2,
    message="holding 96% of rated torque",
    producer=LEFT_ARM,
):
    """A set holding this one alert."""
    return publish_set(active, producer, _item(source, severity, kind, message))


def test_an_alert_names_its_source_severity_and_message():
    active = active_alerts()
    assert raise_alert(active) == []
    (alert,) = active.active()
    assert alert.text == "LEFT ARM J2 CRITICAL: holding 96% of rated torque"
    assert alert.severity == 2


def test_an_entry_with_an_undefined_severity_is_dropped():
    # The contract runs from 1 to 3, and a listed alert is active, so 0 is
    # undefined too.
    for severity in (0, 4):
        active = active_alerts()
        refused = publish_set(
            active,
            LEFT_ARM,
            _item("left arm j2", severity),
            _item("left arm j5", 3),
        )
        assert refused, f"severity {severity} is reported"
        (alert,) = active.active()
        assert alert.severity == 3, "the good entry stands"


def test_an_entry_without_an_identity_is_dropped():
    # A panel line " CRITICAL: ..." names nothing an operator can act on.
    for bad in (_item("", 2), _item("left arm j2", 2, kind="")):
        active = active_alerts()
        refused = publish_set(active, LEFT_ARM, bad, _item("left arm j5", 3))
        assert refused
        (alert,) = active.active()
        assert alert.text.startswith("LEFT ARM J5")


def test_an_identity_listed_twice_keeps_the_first_entry():
    active = active_alerts()
    refused = publish_set(
        active,
        LEFT_ARM,
        _item("left arm j2", 1, kind="motor_condition"),
        _item("left arm j2", 3, kind="motor_condition"),
    )
    assert refused
    (alert,) = active.active()
    assert alert.severity == 1, "the first entry stands"


def test_two_kinds_on_one_source_both_stand():
    active = active_alerts()
    refused = publish_set(
        active,
        LEFT_ARM,
        _item("left arm j2", 1, kind="motor_condition"),
        _item("left arm j2", 2, kind="encoder"),
    )
    assert refused == [], "identity is (source, kind)"
    assert len(active.active()) == 2


def test_a_producer_cannot_replace_or_clear_anothers_alert():
    active = active_alerts()
    raise_alert(active, producer=LEFT_ARM)
    raise_alert(active, producer=RIGHT_ARM, severity=1, message="warm")
    assert len(active.active()) == 2, "same wire strings, distinct producers"
    publish_set(active, RIGHT_ARM)
    (alert,) = active.active()
    assert alert.severity == 2, "the empty set removed only its own entry"


def test_a_set_that_omits_an_alert_clears_it():
    active = active_alerts()
    raise_alert(active)
    assert active.active()
    publish_set(active, LEFT_ARM)
    assert active.active() == ()


def test_a_set_replaces_the_whole_of_its_producers_entries():
    active = active_alerts()
    publish_set(
        active,
        LEFT_ARM,
        _item("left arm j2", 2),
        _item("left arm j5", 1),
    )
    assert len(active.active()) == 2
    publish_set(active, LEFT_ARM, _item("left arm j5", 1))
    (alert,) = active.active()
    assert alert.text.startswith("LEFT ARM J5")


def test_every_active_alert_is_listed_worst_first_across_producers():
    active = active_alerts()
    publish_set(
        active,
        LEFT_ARM,
        _item("left arm j2", 1),
        _item("left arm j7", 3),
    )
    publish_set(active, RIGHT_ARM, _item("right arm j1", 2))
    assert [a.severity for a in active.active()] == [3, 2, 1]


def test_equal_severities_keep_a_stable_order_across_producers():
    # Re-drawn a few times a second: an unstable order would make the list
    # flicker between identical states.
    def listed():
        active = active_alerts()
        publish_set(active, RIGHT_ARM, _item("right arm j4", 1))
        publish_set(
            active,
            LEFT_ARM,
            _item("left arm j9", 1),
            _item("left arm j2", 1),
        )
        return [a.text for a in active.active()]

    first = listed()
    assert first == listed()
    assert [t.split()[1] for t in first[:2]] == ["ARM", "ARM"]
    assert first[0].startswith("LEFT ARM J9"), "producer order, then listed order"
    assert first[1].startswith("LEFT ARM J2")
    assert first[2].startswith("RIGHT ARM J4")


def test_a_producer_that_leaves_the_slot_takes_its_alerts_with_it():
    bound = {LEFT_ARM, RIGHT_ARM}
    active = ActiveAlerts(bound_now=lambda: bound)
    raise_alert(active, producer=LEFT_ARM)
    raise_alert(active, producer=RIGHT_ARM, source="right arm j1")
    assert len(active.active()) == 2
    bound.discard(LEFT_ARM)
    active.retain(bound)
    (alert,) = active.active()
    assert alert.text.startswith("RIGHT ARM J1")


def test_an_empty_slot_clears_every_alert_and_reads_as_unwired():
    # The start-up state and the last-producer-leaves state. A robot nobody
    # is watching must not render as a healthy one.
    active = active_alerts()
    raise_alert(active)
    assert active.producers_bound
    active.retain([])
    assert active.active() == ()
    assert not active.producers_bound


def test_a_producer_that_joins_after_start_up_moves_the_panel_off_unwired():
    bound: set[tuple[str, str]] = set()
    active = ActiveAlerts(bound_now=lambda: bound)
    assert not active.producers_bound
    bound.add(LEFT_ARM)
    active.retain(bound)
    assert active.producers_bound


def test_a_producer_that_stops_republishing_ages_out():
    # An aged-out set means the producer's conditions are unknown, so it must
    # not keep rendering as current.
    clock = {"now": 100.0}
    active = active_alerts(monotonic=lambda: clock["now"])
    raise_alert(active)
    clock["now"] += STALE_AFTER_S
    assert len(active.active()) == 1, "live through the whole window"
    clock["now"] += 0.001
    assert active.active() == ()


def test_a_republish_refreshes_the_window():
    clock = {"now": 100.0}
    active = active_alerts(monotonic=lambda: clock["now"])
    raise_alert(active)
    clock["now"] += STALE_AFTER_S - 0.1
    raise_alert(active)  # a cadence re-publish
    clock["now"] += STALE_AFTER_S - 0.1
    assert active.active()


def test_drain_alerts_survives_a_slot_it_cannot_subscribe_to():
    # Every camera drain already fails this way. An alert drain that raises
    # instead takes its own failure out of the task and leaves no surface.
    class Refuses:
        async def subscribe(self, _runner):
            raise RuntimeError("no such slot")

    active = active_alerts()
    asyncio.run(drain_alerts(object(), Refuses(), active, FakeToken()))
    assert active.active() == ()


async def test_drain_alerts_keeps_every_producers_alerts():
    async with boot(alerts_instances=2) as h:
        left, right = h.mocks.deps.alerts
        active = active_alerts()
        async with running_drain(
            lambda token: drain_alerts(h.node_runner, alerts_topic, active, token)
        ) as drain:
            await left.alerts.publish(_wire(_item("left arm j2", 2)))
            await right.alerts.publish(_wire(_item("right arm j5", 3)))
            await eventually(
                lambda: [a.severity for a in active.active()] == [3, 2],
                message="both producers' alerts, worst first",
            )


async def test_a_malformed_entry_is_reported_once_and_its_set_still_lands(capsys):
    # The latch is per producer: the good producer's set lands between the
    # bad one's sets, and the bad one still logs once. The bad entry is
    # dropped and the rest of that producer's set renders, so one unusable
    # entry cannot blank the motors it reported correctly. The refusals go to
    # the captured log, read incrementally.
    async with boot(alerts_instances=2) as h:
        bad, good = h.mocks.deps.alerts
        logged: list[str] = []
        active = active_alerts()

        def dropped():
            logged.extend(
                line
                for line in capsys.readouterr().out.splitlines()
                if "dropped" in line
            )
            return logged

        async with running_drain(
            lambda token: drain_alerts(h.node_runner, alerts_topic, active, token)
        ):
            for _ in range(3):
                await bad.alerts.publish(
                    _wire(_item("left arm j2", 9), _item("left arm j5", 3))
                )
            await eventually(
                lambda: len(dropped()) == 1, message="the first refusal"
            )
            await eventually(
                lambda: [a.severity for a in active.active()] == [3],
                message="the good entry of the bad set",
            )
            await good.alerts.publish(_wire(_item("right arm j5", 1)))
            await eventually(
                lambda: len(active.active()) == 2, message="the good producer"
            )
            for _ in range(2):
                await bad.alerts.publish(
                    _wire(_item("left arm j2", 9), _item("left arm j5", 3))
                )
            # A clean set from the bad producer re-arms its latch.
            await bad.alerts.publish(_wire(_item("left arm j2", 2)))
            await eventually(
                lambda: [a.severity for a in active.active()] == [2, 1],
                message="the recovery",
            )
        assert len(dropped()) == 1, f"logged {len(dropped())} times: {dropped()}"


async def test_follow_producers_drops_a_departed_producers_alerts():
    async with boot(alerts_instances=1) as h:
        (left,) = h.mocks.deps.alerts
        active = active_alerts(bound=[])
        async with running_drain(
            lambda token: drain_alerts(h.node_runner, alerts_topic, active, token)
        ):
            await left.alerts.publish(_wire(_item("left arm j2", 2)))
            await eventually(
                lambda: len(active.active()) == 1, message="the alert arrived"
            )
            # The real slot holds the mock, so the poll keeps the set.
            async with running_drain(
                lambda token: follow_producers(
                    h.node_runner, alerts_topic, active, token, period_s=0.01
                )
            ):
                await eventually(
                    lambda: active.producers_bound,
                    message="the poll found the bound producer",
                )
                assert len(active.active()) == 1, "a held producer keeps its set"
            # With the poll stopped, a membership that no longer holds it
            # takes its alerts.
            active.retain([])
            assert active.active() == ()


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
    # changes, because every cadence re-publish carries the live number.
    active = active_alerts()
    publish_set(
        active,
        LEFT_ARM,
        _item("left arm j2", 1, kind="a", message="80 C"),
        _item("left arm j2", 1, kind="b", message="79 C"),
    )
    before = [a.text for a in active.active()]
    publish_set(
        active,
        LEFT_ARM,
        _item("left arm j2", 1, kind="a", message="80 C"),
        _item("left arm j2", 1, kind="b", message="78 C"),
    )
    after = [a.text for a in active.active()]
    assert [t.rsplit(": ", 1)[-1] for t in before] == ["80 C", "79 C"]
    assert [t.rsplit(": ", 1)[-1] for t in after] == [
        "80 C",
        "78 C",
    ], "identity holds the position, not the text"


def test_an_unwired_alert_slot_is_distinguishable_from_a_quiet_robot():
    # Both render no alerts, and only one of them means the robot is fine.
    unwired = active_alerts(bound=[])
    wired = active_alerts()
    assert unwired.active() == wired.active() == ()
    assert not unwired.producers_bound
    assert wired.producers_bound
