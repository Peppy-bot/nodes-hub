"""Fast reactive policy: owns the control loop, escalates what it cannot solve.

A mock in the shape of a small onboard model. It does no inference; it reports
which subgoal it is currently servoing on, which is exactly what a test (or a
reader) needs to see in order to tell whether the cross-daemon pairing is
actually delivering.

The one behaviour here that is not scaffolding is the staleness watchdog. A
subgoal is authoritative for `subgoal_ttl_ms` after this node adopts it, and
then this node falls back to its own local behaviour. peppy notifies a peer
when an instance stops or dies, but an unreachable daemon cannot send that
notification, so a node whose correctness depends on freshness owns the
watchdog itself. That is why the uplink is an enhancement path rather than a
dependency, and why a severed link degrades this node instead of stopping it.
"""

import asyncio
import time

from peppygen import NodeBuilder, NodeRunner, QoSProfile
from peppygen.parameters import Parameters
from peppygen.consumed_topics.camera import video_stream
from peppygen.consumed_topics.deliberation import subgoal as subgoal_topic
from peppygen.emitted_topics.deliberation import situation as situation_topic

# What this node does when no subgoal is authoritative. Named rather than
# inlined so the logs distinguish "planner said so" from "planner is not
# reachable and this is the floor".
LOCAL_FALLBACK = "hold_position"


class AuthoritativeSubgoal:
    """The subgoal currently in force, and whether it still is.

    Freshness is decided from the ADOPTION time on this node's own clock, never
    from the planner's `stamp`. The two machines do not share a clock, and a
    watchdog that trusted a remote timestamp would misjudge exactly when it
    matters most.
    """

    def __init__(self, ttl_ms: int) -> None:
        self._ttl_s = ttl_ms / 1000.0
        self._subgoal_id = ""
        self._directive = LOCAL_FALLBACK
        self._adopted_at = 0.0
        self._expired_logged = False

    def adopt(self, subgoal_id: str, directive: str) -> None:
        self._subgoal_id = subgoal_id
        self._directive = directive
        self._adopted_at = time.monotonic()
        self._expired_logged = False
        print(f"[policy] adopted subgoal {subgoal_id}: {directive}", flush=True)

    def current(self) -> tuple[str, str]:
        """The `(subgoal_id, directive)` to servo on right now."""
        if not self._subgoal_id:
            return "", LOCAL_FALLBACK
        if time.monotonic() - self._adopted_at <= self._ttl_s:
            return self._subgoal_id, self._directive
        # Log the transition once, not once per control cycle: at 200 Hz the
        # latter would bury everything else in the log.
        if not self._expired_logged:
            print(
                f"[policy] subgoal {self._subgoal_id} expired after "
                f"{self._ttl_s:.3f}s; falling back to {LOCAL_FALLBACK}",
                flush=True,
            )
            self._expired_logged = True
        return "", LOCAL_FALLBACK


async def receive_subgoals(runner: NodeRunner, authoritative: AuthoritativeSubgoal) -> None:
    """Adopt each subgoal the planner sends, across the daemon boundary."""
    token = runner.cancellation_token()
    subscription = await subgoal_topic.subscribe(runner)
    while not token.is_cancelled():
        received = await subscription.next()
        if received is None:
            break
        _producer, message = received
        authoritative.adopt(message.subgoal_id, message.directive)


async def watch_camera(runner: NodeRunner, seen: dict) -> None:
    """Drain the camera, keeping only a count.

    The frames themselves are not interesting to a wiring test; that the local
    producer link delivers at all is.
    """
    token = runner.cancellation_token()
    subscription = await video_stream.subscribe(runner)
    while not token.is_cancelled():
        received = await subscription.next()
        if received is None:
            break
        seen["frames"] = seen.get("frames", 0) + 1


async def control_loop(
    runner: NodeRunner,
    authoritative: AuthoritativeSubgoal,
    seen: dict,
    control_rate_hz: int,
) -> None:
    """The servo loop, and the situation reports that ride alongside it.

    This loop never awaits anything on the far side of the WAN. That is the
    guarantee placement buys: the camera, the arm, and this node are on one
    daemon, so a slow or absent planner cannot stall the robot.
    """
    token = runner.cancellation_token()
    period = 1.0 / control_rate_hz
    tick = 0

    while not token.is_cancelled():
        subgoal_id, directive = authoritative.current()

        # Report the situation at a small fraction of the control rate: the
        # planner runs orders of magnitude slower, so flooding it would only
        # cost bandwidth.
        if tick % control_rate_hz == 0:
            print(
                f"[policy] servoing on {directive!r} "
                f"(subgoal_id={subgoal_id or 'none'}, frames={seen.get('frames', 0)})",
                flush=True,
            )
            await situation_topic.emit(
                runner,
                situation_topic.Situation(
                    stamp=runner.now(),
                    active_subgoal_id=subgoal_id,
                    observation=f"frames={seen.get('frames', 0)}",
                    # Ask for help whenever nothing is authoritative. A real
                    # policy would escalate on its own uncertainty; the shape of
                    # the conversation is the same.
                    escalating=not subgoal_id,
                ),
            )

        tick += 1
        await asyncio.sleep(period)


async def run(runner: NodeRunner) -> None:
    parameters = Parameters(runner)
    control_rate_hz = parameters.control_rate_hz
    authoritative = AuthoritativeSubgoal(parameters.subgoal_ttl_ms)
    seen: dict = {}

    print(
        f"[policy] up at {control_rate_hz} Hz, "
        f"subgoal TTL {parameters.subgoal_ttl_ms} ms",
        flush=True,
    )

    await asyncio.gather(
        watch_camera(runner, seen),
        receive_subgoals(runner, authoritative),
        control_loop(runner, authoritative, seen, control_rate_hz),
    )


def main() -> None:
    NodeBuilder().build().run(run)


if __name__ == "__main__":
    main()
