"""The failure kinds this node gives, and the refusal that carries one.

The kinds are the kinds of peppy and the kinds that the limb_motion,
postures and workspace contracts declare. Each generated member has a
FailureKind of its own; its Failure takes each kind of this one by its name,
and refuses a kind that the member does not declare."""

from __future__ import annotations

from enum import StrEnum


class FailureKind(StrEnum):
    # A limb name or a value the backbone does not take.
    REJECTED = "rejected"
    # A valid goal that the state of the limb does not allow now: the limb
    # executes another move.
    PRECONDITION_FAILED = "precondition_failed"
    # The follower of a limb delivers no fresh state, or the node stops.
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    # The goal or the request cannot go on, and no other kind says why.
    ATTEMPT_FAILED = "attempt_failed"
    # An arm move did not settle by its timeout.
    TIMEOUT = "timeout"
    # No plan reaches the target.
    UNREACHABLE = "unreachable"
    # A leader streams a limb of the goal.
    LEADER_STREAMING = "leader_streaming"


class Refusal(Exception):
    """A goal that must not run, or a request that gets an error reply: the
    failure kind and the operator-facing reason."""

    def __init__(self, kind: FailureKind, reason: str):
        super().__init__(reason)
        self.kind = kind
