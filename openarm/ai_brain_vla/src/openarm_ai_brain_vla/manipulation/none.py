"""The manipulator that is not there: `manipulation_backend: "none"`.

It moves nothing. Every sequence is refused with a message naming the
missing backend, after the gripper and item rules have had their say, so
a caller still learns "gripper already holds an item" before "no
backend". abort and get_state work as on any brain.
"""

from __future__ import annotations

from ..ports import CancelToken, Gripper, Item, Outcome, Pose


class NoneManipulator:
    name = "none"

    @property
    def available(self) -> bool:
        return False

    async def start(self, robot) -> None:
        return None

    async def grab(self, item: Item, gripper: Gripper, max_effort: float, cancel: CancelToken) -> Outcome:
        return Outcome(False, self.reason())

    async def drop(self, gripper: Gripper, cancel: CancelToken) -> Outcome:
        return Outcome(False, self.reason())

    async def place(self, gripper: Gripper, pose: Pose, cancel: CancelToken) -> Outcome:
        return Outcome(False, self.reason())

    async def stop(self) -> None:
        return None

    def reason(self) -> str:
        return "no manipulation backend: manipulation_backend is 'none'"
