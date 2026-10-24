"""Why a goal of this engine ends cancelled, in words.

A goal receives its cancel signal for one of two causes that peppy gives: a
cancel request of its caller, or a sender that is gone for longer than the
sender grace of its action. The goal ends cancelled with that cause as its
reason, in the words peppy gives each cause.
"""

# The reason a goal ends cancelled with, by the cause of its cancel signal.
_REASONS = {
    "caller": "cancelled by its caller",
    "sender_gone": "its sender is gone",
}


def cancel_reason(context) -> str:
    """Why the goal of `context` received its cancel signal."""
    return _REASONS.get(context.cancel_cause(), "cancelled")
