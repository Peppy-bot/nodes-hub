"""Sim posture handling: the policy-achievable subset of the G1 FSM.

The sim honors the same posture names as the real node but only distinguishes
what a standing controller can express: whether the joints are actively held
(pd_enabled) and which fsm id to report. Full posture fidelity arrives with the
locomotion policy.
"""

from __future__ import annotations


class UnknownPosture(ValueError):
    """Raised when a wire string does not name a known posture."""


# posture -> (reported fsm id, joints actively held). damp / zero_torque go limp.
_SIM_POSTURE: dict[str, tuple[int, bool]] = {
    "damp": (1, False),
    "zero_torque": (0, False),
    "squat_to_stand": (4, True),
    "lie_to_stand": (4, True),
    "start": (200, True),
    "sit": (3, True),
    "stand_to_squat": (2, True),
    "high_stand": (4, True),
    "low_stand": (4, True),
}


def sim_posture(raw: str) -> tuple[int, bool]:
    """Return (fsm_id, pd_enabled) for a posture name, or raise UnknownPosture."""
    key = raw.strip().lower()
    try:
        return _SIM_POSTURE[key]
    except KeyError as exc:
        known = ", ".join(_SIM_POSTURE)
        raise UnknownPosture(f"unknown posture {raw!r}; expected one of: {known}") from exc
