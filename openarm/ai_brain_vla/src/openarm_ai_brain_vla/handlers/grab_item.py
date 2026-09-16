"""grab_item: close a gripper on an item named by id or by pose."""

from __future__ import annotations

from ..ports import Refusal
from ..sequencer import MANIPULATION

ZERO = dict(gripper_name="", item_id="")


async def run(brain, ctx) -> None:
    goal = ctx.request().data

    async def body(job) -> dict:
        state = brain.state
        # The item first: its position decides which free gripper suits it.
        known = state.item(goal.item_id) if goal.item_id else None
        near = known.position if known is not None else tuple(goal.position)
        gripper = state.gripper_for_grab(goal.gripper_name, near)
        if not brain.manipulator.available:
            raise Refusal(f"no manipulation backend: manipulation_backend is '{brain.manipulator.name}'")
        item = known
        if item is None:
            orientation = tuple(goal.orientation) if goal.orientation is not None else None
            item = state.mint_from_pose(tuple(goal.position), orientation, brain.now())
        outcome = await brain.manipulator.grab(item, gripper, goal.max_effort, job.cancel)
        job.cancel.check()
        if not outcome.success:
            raise Refusal(outcome.message or "the grasp failed")
        state.set_held(gripper, item.item_id)
        return dict(gripper_name=gripper.name, item_id=item.item_id)

    await brain.run_guarded(ctx, MANIPULATION, "grab_item", body, ZERO)
