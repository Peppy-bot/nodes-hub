"""The G1 control backend: an abstract seam plus the real LocoClient wrapper.

The node logic depends only on `G1Backend`, so it is exercised in tests with an
in-memory fake and, on hardware, with `LocoClientBackend`. The Unitree SDK is
imported lazily and guarded: the module loads (and the node builds) without the
SDK or its CycloneDDS C backend present, which are only needed to reach a robot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .postures import POSTURE_FSM_ID, POSTURE_METHOD, Posture


class G1Backend(ABC):
    """Everything the node needs from the underlying G1, real or simulated."""

    @abstractmethod
    def transition(self, posture: Posture) -> int:
        """Command a discrete FSM transition; return the resulting fsm id."""

    @abstractmethod
    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Command a planar base velocity (body frame)."""

    @abstractmethod
    def read_state(self) -> tuple[int, int, int]:
        """Return (fsm_id, mode, battery_soc) for the g1_state telemetry."""

    def shutdown(self) -> None:
        """Bring the robot to a safe state. Default: damp."""
        self.transition(Posture.DAMP)


class SdkUnavailable(RuntimeError):
    """Raised when the real backend is requested but the Unitree SDK is absent."""


class LocoClientBackend(G1Backend):
    """Wraps the Unitree SDK LocoClient over CycloneDDS.

    Constructing this initializes the DDS transport, so it is only built when the
    node is actually pointed at a robot (or a DDS peer). Method names are resolved
    through POSTURE_METHOD, keeping the SDK surface in one place.
    """

    def __init__(self, network_interface: str, domain_id: int) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        except ImportError as exc:
            raise SdkUnavailable(
                "unitree_sdk2py is not installed; build the node with the "
                "`hardware` extra (`uv sync --extra hardware`) to reach a robot"
            ) from exc

        ChannelFactoryInitialize(domain_id, network_interface)
        self._loco = LocoClient()
        self._loco.SetTimeout(10.0)
        self._loco.Init()
        self._last_fsm_id = POSTURE_FSM_ID[Posture.DAMP]

    def transition(self, posture: Posture) -> int:
        method = getattr(self._loco, POSTURE_METHOD[posture])
        method()
        self._last_fsm_id = POSTURE_FSM_ID[posture]
        return self._last_fsm_id

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        self._loco.Move(vx, vy, vyaw)

    def read_state(self) -> tuple[int, int, int]:
        # fsm_id tracks the last commanded transition. A live SportModeState /
        # LowState subscriber replaces the tracked id and supplies real mode and
        # battery at hardware bring-up; until then battery reports full.
        mode = 1 if self._last_fsm_id >= POSTURE_FSM_ID[Posture.START] else 0
        return self._last_fsm_id, mode, 100
