"""abort: stop the running manipulation sequence and report which one it
was. The one goal admitted while a sequence runs; it never enters a lane
itself, so it never appears in get_state."""

from __future__ import annotations

from ..sequencer import MANIPULATION


async def run(brain, ctx) -> None:
    goal = ctx.request().data
    started = brain.now()
    reason = goal.reason.strip() or "aborted"
    try:
        stopped = await brain.sequencer.stop(MANIPULATION, f"aborted: {reason}")
    except Exception as error:
        await ctx.complete(success=False, message=f"abort failed: {error!r}", aborted_action="", action_time=0.0)
        return
    elapsed = max(0.0, (brain.now() - started) / 1e9)
    await ctx.complete(
        success=True,
        message="" if stopped else "no sequence was running",
        aborted_action=stopped or "",
        action_time=elapsed,
    )
