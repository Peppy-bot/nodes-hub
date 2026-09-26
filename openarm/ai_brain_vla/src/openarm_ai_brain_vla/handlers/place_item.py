"""place_item: carry the held item to a pose and open there."""

from __future__ import annotations

from ..ports import Pose, Refusal
from ..sequencer import MANIPULATION

ZERO = dict(gripper_name="", item_id="", final_position=[0.0, 0.0, 0.0])


async def run(brain, ctx) -> None:
    goal = ctx.request().data

    async def body(job) -> dict:
        state = brain.state
        gripper = state.holder(goal.gripper_name)
        if not brain.manipulator.available:
            raise Refusal(f"no manipulation backend: manipulation_backend is '{brain.manipulator.name}'")
        orientation = tuple(goal.orientation) if goal.orientation is not None else None
        pose = Pose(tuple(goal.position), orientation)
        outcome = await brain.manipulator.place(gripper, pose, job.cancel)
        job.cancel.check()
        if not outcome.success:
            raise Refusal(outcome.message or "the placement failed")
        item_id = state.clear_held(gripper)
        final = outcome.final_position if outcome.final_position is not None else pose.position
        return dict(gripper_name=gripper.name, item_id=item_id, final_position=list(final))

    await brain.run_guarded(ctx, MANIPULATION, "place_item", body, ZERO)
