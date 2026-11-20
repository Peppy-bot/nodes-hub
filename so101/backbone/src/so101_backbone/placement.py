"""The placement of each command's point or pose in the robot frame, by the
robot's tree through transform_lookup, at the command's timestamp, as the
contracts place a command:

- a move_arm goal, a check_arm_move request and the points of
  check_positions, together by one lookup, where their frame stood at their
  timestamp, in the robot frame as it stands at the latest time: the lookup
  gives the request's frame and timestamp as the source and the robot frame
  with no timestamp, at the latest time, as the target, so a robot-frame pose
  from before a base move names the same place on the floor after it;
- each pose setpoint of pose_link by the same lookup, as pose_link places a
  setpoint: a setpoint in the robot frame at a time names where the robot
  frame stood then. One in the robot frame with no timestamp names where the
  robot frame stands at the latest time, so it is taken as it comes.

Each lookup waits up to `lookup_timeout_s` in the server for the tree to
answer, and the node waits for the server's answer that long plus
ANSWER_WAIT_S, so a lookup the server refuses once its wait has passed
reaches the command with its kind. A lookup that fails gives a Refusal with
the failure kind of the lookup, its message naming the command, then the
lookup's cause; a lookup that gets no answer, which peppylib raises as one of
_NO_ANSWER_ERRORS, gives one with `resource_unavailable`; a call that gives
nothing the node can use, which the generated client and peppylib raise as
ValueError, gives one with `attempt_failed`; a placement that carries a value
past the finite numbers gives one with `rejected`. A placed value carries the time of the robot
frame's place, as the lookup's answer gives it, or none when that place holds
at every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from control_core_py.geometry import GeometryError, Placement, Point, Pose, PoseTarget, Stamped
from peppygen import NodeRunner
from peppygen.consumed_services.transform_lookup import lookup_transform

from so101_backbone.failures import FailureKind, Refusal
from so101_backbone.wire import ANSWER_WAIT_S

# The longest wait (s) a request of transform_lookup's lookup_transform may
# ask for; the implementer refuses a longer one with `rejected`.
MAX_LOOKUP_TIMEOUT_S = 1.0

# What a call of peppylib raises when it gets no answer: TimeoutError when the
# call's wait passes, ConnectionError for a provider it cannot reach, and
# RuntimeError for the other errors of its messaging.
_NO_ANSWER_ERRORS = (TimeoutError, ConnectionError, RuntimeError)

T = TypeVar("T", Point, Pose, PoseTarget, tuple[Point, ...])


@dataclass(frozen=True)
class Placed(Generic[T]):
    """A value placed in the robot frame, and `timestamp`, the time of the
    robot frame's place it stands in, None for a place that holds at every
    time."""

    value: T
    timestamp: float | None


class Placer:
    """The robot's tree as transform_lookup answers it, the robot frame every
    command is placed in, and how long a lookup waits."""

    def __init__(self, node_runner: NodeRunner, robot_frame: str, timeout_s: float) -> None:
        self._node_runner = node_runner
        self.robot_frame = robot_frame
        self._timeout_s = timeout_s

    async def place(self, stamped: Stamped[T], what: str) -> Placed[T]:
        """`stamped` in the robot frame at the latest time, where its frame
        stood at its timestamp; a Refusal naming `what` when the tree cannot
        place it."""
        answer = await self._look_up(stamped.frame, stamped.timestamp, what)
        return Placed(_placed_by(answer, stamped.value, what), answer.timestamp)

    async def place_setpoint(self, stamped: Stamped[Pose]) -> Pose:
        """The pose setpoint `stamped` in the robot frame at the latest time,
        as `place` places it; a Refusal when the tree cannot place it. One in
        the robot frame with no timestamp names where the robot frame stands
        then, and is taken as it comes."""
        if stamped.frame == self.robot_frame and stamped.timestamp is None:
            return stamped.value
        return (await self.place(stamped, "the setpoint")).value

    async def _look_up(
        self, frame: str, timestamp: float | None, what: str
    ) -> lookup_transform.ResponseData:
        """Where `frame` stood at `timestamp` in the robot frame at the latest
        time, or the Refusal naming `what`."""
        request = lookup_transform.Request(
            target_frame=self.robot_frame,
            source_frame=frame,
            source_timestamp=timestamp,
            target_timestamp=None,
            timeout_s=self._timeout_s,
        )
        try:
            answer = await lookup_transform.poll(
                self._node_runner,
                lookup_transform.bound_producer(self._node_runner),
                request,
                self._timeout_s + ANSWER_WAIT_S,
            )
        except lookup_transform.Failure as failure:
            raise Refusal(FailureKind(failure.kind.value), f"{what}: {failure.message}") from None
        except _NO_ANSWER_ERRORS as error:
            raise Refusal(
                FailureKind.RESOURCE_UNAVAILABLE,
                f"{what}: the robot's transform lookup is not answered ({error!r})",
            ) from error
        except ValueError as error:
            # The generated client raises ValueError for an answer it cannot
            # read: a failure kind that lookup_transform does not declare, or
            # values of another length than the contract's; peppylib raises it
            # for a call that the node's links do not allow.
            raise Refusal(
                FailureKind.ATTEMPT_FAILED,
                f"{what}: the robot's transform lookup gave nothing this node can use ({error!r})",
            ) from error
        return answer.data


def _placed_by(answer: lookup_transform.ResponseData, value: T, what: str) -> T:
    """`value` carried into the answer's frame by the answer's placement; a
    Refusal with `rejected` naming `what` when the placement carries a
    coordinate past the finite numbers."""
    try:
        return _placement_of(answer).place(value)
    except GeometryError as e:
        raise Refusal(FailureKind.REJECTED, f"{what}: {e}") from None


def _placement_of(answer: lookup_transform.ResponseData) -> Placement:
    """The place of the answer's child frame in its frame."""
    x, y, z = answer.position
    qx, qy, qz, qw = answer.orientation
    return Placement((float(x), float(y), float(z)), (float(qx), float(qy), float(qz), float(qw)))
