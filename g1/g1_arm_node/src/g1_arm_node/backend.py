"""The G1 arm backend: an abstract seam plus the real arm-action wrapper.

The node logic depends only on `ArmBackend`, so it is exercised in tests with an
in-memory fake and, on hardware, with `ArmActionClientBackend`. The Unitree SDK
is imported lazily and guarded, so the module loads (and the node builds) without
the SDK or its CycloneDDS backend present.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod


class ArmBackend(ABC):
    @abstractmethod
    def execute_action(self, action_id: int) -> int:
        """Run a preset arm action by id; return the SDK return code."""

    @abstractmethod
    def get_action_list(self) -> str:
        """Return the robot's available arm actions as a JSON string."""


class SdkUnavailable(RuntimeError):
    """Raised when the real backend is requested but the Unitree SDK is absent."""


class ArmActionClientBackend(ArmBackend):
    """Wraps the Unitree SDK G1ArmActionClient over CycloneDDS."""

    def __init__(self, network_interface: str, domain_id: int) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
        except ImportError as exc:
            raise SdkUnavailable(
                "unitree_sdk2py is not installed; build the node with the "
                "`hardware` extra (`uv sync --extra hardware`) to reach a robot"
            ) from exc

        ChannelFactoryInitialize(domain_id, network_interface)
        self._arm = G1ArmActionClient()
        self._arm.SetTimeout(10.0)
        self._arm.Init()

    def execute_action(self, action_id: int) -> int:
        code = self._arm.ExecuteAction(action_id)
        # Some SDK builds return (code, data); normalize to the code.
        if isinstance(code, tuple):
            code = code[0]
        return int(code)

    def get_action_list(self) -> str:
        result = self._arm.GetActionList()
        # GetActionList returns (code, data); serialize the data as JSON.
        data = result[1] if isinstance(result, tuple) else result
        try:
            return json.dumps(data)
        except TypeError:
            return json.dumps(str(data))
