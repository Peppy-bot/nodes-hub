"""ai_brain_vla: the entry point. Wiring only: the node's generated modules
for brain_core, the models the parameters name, and brain_core's shared
setup."""

from __future__ import annotations

import asyncio
from typing import Callable, Mapping

from peppygen import NodeBuilder, NodeRunner, QoSProfile, clock
from peppygen.consumed_actions.limb_motion import move_arm, move_gripper
from peppygen.consumed_services.camera import depth_stream_info
from peppygen.consumed_services.camera_mounts import get_camera_poses
from peppygen.consumed_services.geometry import get_color_intrinsics, get_depth_intrinsics
from peppygen.consumed_topics.camera import depth_stream, video_stream
from peppygen.exposed_actions.brain_model import switch_model
from peppygen.exposed_actions.item_manipulation import abort, drop_item, grab_item, place_item
from peppygen.exposed_actions.item_perception import identify_item, scan_items
from peppygen.exposed_services.brain_model import get_model, list_models
from peppygen.exposed_services.item_manipulation import get_state
from peppygen.parameters import Parameters

from brain_core import node
from brain_core.bindings import Bindings
from brain_core.ports import Detector, Manipulator

from .models import MANIPULATORS, VlaModels
from .perception import DETECTORS

BINDINGS = Bindings(
    clock=clock,
    standard_qos=QoSProfile.Standard,
    move_arm=move_arm,
    move_gripper=move_gripper,
    video_stream=video_stream,
    depth_stream=depth_stream,
    depth_stream_info=depth_stream_info,
    get_color_intrinsics=get_color_intrinsics,
    get_depth_intrinsics=get_depth_intrinsics,
    get_camera_poses=get_camera_poses,
    scan_items=scan_items,
    identify_item=identify_item,
    grab_item=grab_item,
    drop_item=drop_item,
    place_item=place_item,
    abort=abort,
    get_state=get_state,
    switch_model=switch_model,
    get_model=get_model,
    list_models=list_models,
)


async def setup(
    params: Parameters,
    node_runner: NodeRunner,
    *,
    detectors: Mapping[str, Callable[[str], Detector]] = DETECTORS,
    manipulators: Mapping[str, Callable[[], Manipulator]] = MANIPULATORS,
) -> list[asyncio.Task]:
    """The node's entry point. `detectors` and `manipulators` stand in for
    the models the brain runs when given, which is how the whole node runs
    over the wire against fakes."""
    source = VlaModels(
        params.perception_model, params.manipulation_model, params.perception_gallery,
        detectors=detectors, manipulators=manipulators,
    )
    return await node.setup(params, node_runner, BINDINGS, source)


def main() -> None:
    node.configure_logging("ai_brain_vla")
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
