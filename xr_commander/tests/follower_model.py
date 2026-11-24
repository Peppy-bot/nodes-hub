"""drive_link's follower as botshare_backbone runs it, for the trace tests of
the base rule: the follow_state of each drive_states, when the follower
publishes one, and when each reaches the headset. Every time is on the
instance's clock in ns, the clock that the headset and the follower read in a
simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

# drive_link's window: the leader streams from each setpoint until more than
# 0.5 s pass with no setpoint (control_core::drive_link's `WINDOW_NS`).
WINDOW_NS = 500_000_000
# The longest the follower goes without a drive_states while the wheels
# report: base_state's 0.1 s (botshare_backbone's `REPORT_PERIOD_NS`).
REPORT_PERIOD_NS = 100_000_000
# The longest a message takes to arrive in a trace: 0.5 s, past which a leader
# counts the base as not reporting.
MAX_DELIVERY_DELAY_NS = 500_000_000

# The limits a drive_states carries, as the test builds them.
Limits = TypeVar("Limits")


@dataclass(frozen=True)
class Published(Generic[Limits]):
    """One drive_states as the follower publishes it: when, its follow_state
    as drive_link names it, and the limits it carries."""

    published_ns: int
    follow_state: str
    limits: Limits


class FollowerModel(Generic[Limits]):
    """The follower of one pair. Events happen at the time of the last
    `advance_to`, which gives the drive_states published since the time
    before, each with the follow_state at its own time."""

    def __init__(self, limits: Limits, report_period_ns: int) -> None:
        """The follower at its start: before its clock's first reading, the
        wheels silent (`Follower::default`), and once they report, a
        drive_states every `report_period_ns`, each giving `limits`
        (`Base::publish`)."""
        if not 0 < report_period_ns <= REPORT_PERIOD_NS:
            raise ValueError(
                f"a report period of {report_period_ns} ns lies outside "
                f"(0, {REPORT_PERIOD_NS}] ns: base_state reports at least every 0.1 s"
            )
        self._limits = limits
        self._report_period_ns = report_period_ns
        self._now_ns: int | None = None
        self._last_setpoint_ns: int | None = None
        self._silent = True
        self._silence_lingers = False
        self._stop_ns: int | None = None
        self._goal_runs = False
        self._next_publish_ns: int | None = None

    def advance_to(self, now_ns: int) -> list[Published[Limits]]:
        """The clock reads `now_ns`, its first reading at the first call:
        each drive_states published after the reading before and up to it."""
        if self._now_ns is not None and now_ns < self._now_ns:
            raise ValueError(
                f"the clock reads {now_ns} ns, before its last reading of "
                f"{self._now_ns} ns: give the step back to clock_went_back"
            )
        if self._now_ns is None and not self._silent:
            self._next_publish_ns = now_ns + self._report_period_ns
        self._now_ns = now_ns
        published = []
        while self._next_publish_ns is not None and self._next_publish_ns <= now_ns:
            at_ns = self._next_publish_ns
            published.append(Published(at_ns, self._state_at(at_ns), self._limits))
            self._next_publish_ns = at_ns + self._report_period_ns
        return published

    def clock_went_back(self, now_ns: int) -> None:
        """The clock stepped back to `now_ns`: the newest setpoint and the
        robot stop take `now_ns` when they came after it
        (`Follower::clock_went_back`), the goal fails, and the wheels fall
        silent (`Base::start_timing_over`)."""
        if self._now_ns is None or now_ns >= self._now_ns:
            raise ValueError(f"{now_ns} ns is no step back from {self._now_ns} ns")
        self._now_ns = now_ns
        if self._last_setpoint_ns is not None:
            self._last_setpoint_ns = min(self._last_setpoint_ns, now_ns)
        if self._stop_ns is not None:
            self._stop_ns = min(self._stop_ns, now_ns)
        self._goal_runs = False
        self.silence_starts()

    def setpoint(self) -> bool:
        """A setpoint arrives now: whether the base follows it. One that
        arrives before the clock's first reading is dropped uncounted
        (`run_setpoint_listener`); one during a hold is dropped and counts
        (`Follower::setpoint`)."""
        if self._now_ns is None:
            return False
        self._release(self._now_ns)
        self._last_setpoint_ns = self._now_ns
        return self._state_at(self._now_ns) == "following"

    def stop(self) -> bool:
        """The follower takes a robot stop now: whether it holds the base,
        which it does once the clock has a reading (`Base::stop`)."""
        if self._now_ns is None:
            return False
        self._release(self._now_ns)
        self._stop_ns = self._now_ns
        return True

    def silence_starts(self) -> None:
        """The wheels fall silent now: the hold lingers when the leader
        streams (`Follower::silence_starts`), the goal fails (`Base::step`),
        and no drive_states goes out (`Base::publish_reports`)."""
        if self._silent:
            return
        if self._now_ns is not None:
            self._release(self._now_ns)
            self._silence_lingers = self._streams_at(self._now_ns)
        self._silent = True
        self._goal_runs = False
        self._next_publish_ns = None

    def silence_ends(self) -> None:
        """The wheels report again now (`Follower::silence_ends`); the first
        drive_states goes out one report period on."""
        if not self._silent:
            return
        self._silent = False
        if self._now_ns is None:
            return
        self._release(self._now_ns)
        self._next_publish_ns = self._now_ns + self._report_period_ns

    def start_goal(self) -> bool:
        """A goal of the follower's node asks to start now: admitted while
        the wheels report, the leader does not stream and no goal runs
        (`BaseStatus::admit`)."""
        now_ns = self._now_ns
        if (
            now_ns is None
            or self._silent
            or self._streams_at(now_ns)
            or self._goal_runs
        ):
            return False
        self._goal_runs = True
        return True

    def end_goal(self) -> None:
        """The goal ends now."""
        self._goal_runs = False

    def _streams_at(self, now_ns: int) -> bool:
        """Whether the leader streams at `now_ns` (`Follower::streams_at`)."""
        return (
            self._last_setpoint_ns is not None
            and now_ns - self._last_setpoint_ns <= WINDOW_NS
        )

    def _release(self, now_ns: int) -> None:
        """Ends each hold whose pause has passed by `now_ns`
        (`Follower::release`)."""
        if self._stop_ns is not None:
            quiet_from = (
                self._stop_ns
                if self._last_setpoint_ns is None
                else max(self._last_setpoint_ns, self._stop_ns)
            )
            if now_ns - quiet_from > WINDOW_NS:
                self._stop_ns = None
        if self._silence_lingers and not self._silent and not self._streams_at(now_ns):
            self._silence_lingers = False

    def _state_at(self, now_ns: int) -> str:
        """drive_states' follow_state at `now_ns`: the first hold that holds
        (`Follower::state`)."""
        self._release(now_ns)
        if self._silent or self._silence_lingers:
            return "base_silence"
        if self._stop_ns is not None:
            return "robot_stop"
        if self._goal_runs:
            return "follower_goal"
        return "following"


class InOrderDelivery:
    """How one stream's messages arrive: each after a delay of at most
    `MAX_DELIVERY_DELAY_NS`, never before one sent earlier. A message the
    generator drops gets no arrival."""

    def __init__(self) -> None:
        self._last_arrival_ns: int | None = None

    def arrival_ns(self, sent_ns: int, delay_ns: int) -> int:
        """When the message sent at `sent_ns` arrives, `delay_ns` on."""
        if not 0 <= delay_ns <= MAX_DELIVERY_DELAY_NS:
            raise ValueError(
                f"a delay of {delay_ns} ns lies outside [0, {MAX_DELIVERY_DELAY_NS}] ns"
            )
        arrival_ns = sent_ns + delay_ns
        if self._last_arrival_ns is not None:
            arrival_ns = max(arrival_ns, self._last_arrival_ns)
        self._last_arrival_ns = arrival_ns
        return arrival_ns

    def clock_went_back(self) -> None:
        """The clock stepped back: the messages in flight are dropped, and
        the order starts over."""
        self._last_arrival_ns = None
