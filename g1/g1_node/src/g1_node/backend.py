"""The G1 control backend: an abstract seam plus the real SDK wrapper.

The node logic depends only on `G1Backend`, so it is exercised in tests with an
in-memory fake and, on hardware, with `LocoClientBackend`. The Unitree SDK is
imported lazily and guarded: the module loads (and the node builds) without the
SDK or its CycloneDDS C backend present, which are only needed to reach a robot.

`LocoClientBackend` owns the three high-level DDS clients the locomotion node
uses: LocoClient (motion + FSM), MotionSwitcherClient (control-mode switch), and
a LowState subscriber for telemetry. Audio and arm gestures live in their own
nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .postures import POSTURE_FSM_ID, POSTURE_METHOD, Posture

# hg LowState.motor_state capacity; the G1 populates the joints it uses.
MOTOR_COUNT = 35


@dataclass
class ImuState:
    quaternion: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    gyroscope: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    accelerometer: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    temperature: int = 0


@dataclass
class JointStates:
    q: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)
    dq: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)
    tau_est: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)
    temperature: list[int] = field(default_factory=lambda: [0] * MOTOR_COUNT)
    voltage: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)
    motor_mode: list[int] = field(default_factory=lambda: [0] * MOTOR_COUNT)


@dataclass
class Telemetry:
    fsm_id: int = 0
    mode_pr: int = 0
    mode_machine: int = 0
    tick: int = 0
    imu: ImuState = field(default_factory=ImuState)
    joints: JointStates = field(default_factory=JointStates)


class G1Backend(ABC):
    """Everything the locomotion node needs from the G1, real or simulated."""

    # --- Motion ---------------------------------------------------------------
    @abstractmethod
    def transition(self, posture: Posture) -> int:
        """Command a discrete FSM transition; return the resulting fsm id."""

    @abstractmethod
    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Command a continuous planar base velocity (body frame)."""

    @abstractmethod
    def stop_move(self) -> None:
        """Immediately stop base motion."""

    @abstractmethod
    def move_timed(self, vx: float, vy: float, omega: float, duration: float) -> None:
        """Command a base velocity held for a fixed duration."""

    # --- Stance / mode setters ------------------------------------------------
    @abstractmethod
    def balance_stand(self, balance_mode: int) -> None: ...

    @abstractmethod
    def set_balance_mode(self, balance_mode: int) -> None: ...

    @abstractmethod
    def set_stand_height(self, stand_height: float) -> None: ...

    @abstractmethod
    def set_speed_mode(self, speed_mode: int) -> None: ...

    @abstractmethod
    def set_fsm_id(self, fsm_id: int) -> None: ...

    @abstractmethod
    def get_fsm_id(self) -> int: ...

    @abstractmethod
    def set_task_id(self, task_id: int) -> None: ...

    @abstractmethod
    def switch_to_user_ctrl(self) -> None: ...

    @abstractmethod
    def switch_to_internal_ctrl(self, mode: int) -> None: ...

    # --- Motion-mode switch ---------------------------------------------------
    @abstractmethod
    def check_mode(self) -> str: ...

    @abstractmethod
    def select_mode(self, name: str) -> None: ...

    @abstractmethod
    def release_mode(self) -> None: ...

    # --- Telemetry ------------------------------------------------------------
    @abstractmethod
    def read_telemetry(self) -> Telemetry:
        """Snapshot the latest measured state for the telemetry topics."""

    def shutdown(self) -> None:
        """Bring the robot to a safe state. Default: damp."""
        self.transition(Posture.DAMP)


class SdkUnavailable(RuntimeError):
    """Raised when the real backend is requested but the Unitree SDK is absent."""


class LocoClientBackend(G1Backend):
    """Wraps the Unitree SDK loco + mode-switch clients and a LowState feed.

    Constructing this initializes the DDS transport, so it is only built when the
    node is actually pointed at a robot (or a DDS peer). Posture methods are
    resolved through POSTURE_METHOD, keeping the SDK surface in one place.
    """

    def __init__(self, network_interface: str, domain_id: int) -> None:
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        except ImportError as exc:
            raise SdkUnavailable(
                "unitree_sdk2py is not installed; build the node with the "
                "`hardware` extra (`uv sync --extra hardware`) to reach a robot"
            ) from exc

        ChannelFactoryInitialize(domain_id, network_interface)

        self._loco = LocoClient()
        self._loco.SetTimeout(10.0)
        self._loco.Init()

        self._switcher = MotionSwitcherClient()
        self._switcher.SetTimeout(10.0)
        self._switcher.Init()

        self._last_fsm_id = POSTURE_FSM_ID[Posture.DAMP]

        # The DDS callback swaps this reference from a subscriber thread; the GIL
        # makes the assignment atomic and read_telemetry only reads the latest.
        self._latest_lowstate = None
        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init(self._on_lowstate, 10)

    def _on_lowstate(self, msg) -> None:
        self._latest_lowstate = msg

    # --- Motion ---------------------------------------------------------------
    def transition(self, posture: Posture) -> int:
        getattr(self._loco, POSTURE_METHOD[posture])()
        self._last_fsm_id = POSTURE_FSM_ID[posture]
        return self._last_fsm_id

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        self._loco.Move(vx, vy, vyaw)

    def stop_move(self) -> None:
        self._loco.StopMove()

    def move_timed(self, vx: float, vy: float, omega: float, duration: float) -> None:
        self._loco.SetVelocity(vx, vy, omega, duration)

    # --- Stance / mode setters ------------------------------------------------
    def balance_stand(self, balance_mode: int) -> None:
        self._loco.BalanceStand(balance_mode)

    def set_balance_mode(self, balance_mode: int) -> None:
        self._loco.SetBalanceMode(balance_mode)

    def set_stand_height(self, stand_height: float) -> None:
        self._loco.SetStandHeight(stand_height)

    def set_speed_mode(self, speed_mode: int) -> None:
        self._loco.SetSpeedMode(speed_mode)

    def set_fsm_id(self, fsm_id: int) -> None:
        self._loco.SetFsmId(fsm_id)
        self._last_fsm_id = fsm_id

    def get_fsm_id(self) -> int:
        fsm_id = self._loco.GetFsmId()
        # Some SDK builds return (code, data); normalize to the id.
        if isinstance(fsm_id, tuple):
            fsm_id = fsm_id[1]
        self._last_fsm_id = int(fsm_id)
        return self._last_fsm_id

    def set_task_id(self, task_id: int) -> None:
        self._loco.SetTaskId(task_id)

    def switch_to_user_ctrl(self) -> None:
        self._loco.SwitchToUserCtrl()

    def switch_to_internal_ctrl(self, mode: int) -> None:
        self._loco.SwitchToInternalCtrl(mode)

    # --- Motion-mode switch ---------------------------------------------------
    def check_mode(self) -> str:
        result = self._switcher.CheckMode()
        return str(result)

    def select_mode(self, name: str) -> None:
        self._switcher.SelectMode(name)

    def release_mode(self) -> None:
        self._switcher.ReleaseMode()

    # --- Telemetry ------------------------------------------------------------
    def read_telemetry(self) -> Telemetry:
        low = self._latest_lowstate
        if low is None:
            return Telemetry(fsm_id=self._last_fsm_id)

        imu = low.imu_state
        motors = low.motor_state
        joints = JointStates(
            q=[float(motors[i].q) for i in range(MOTOR_COUNT)],
            dq=[float(motors[i].dq) for i in range(MOTOR_COUNT)],
            tau_est=[float(motors[i].tau_est) for i in range(MOTOR_COUNT)],
            temperature=[int(motors[i].temperature[0]) for i in range(MOTOR_COUNT)],
            voltage=[float(motors[i].vol) for i in range(MOTOR_COUNT)],
            motor_mode=[int(motors[i].mode) for i in range(MOTOR_COUNT)],
        )
        return Telemetry(
            fsm_id=self._last_fsm_id,
            mode_pr=int(low.mode_pr),
            mode_machine=int(low.mode_machine),
            tick=int(low.tick),
            imu=ImuState(
                quaternion=[float(v) for v in imu.quaternion],
                gyroscope=[float(v) for v in imu.gyroscope],
                accelerometer=[float(v) for v in imu.accelerometer],
                rpy=[float(v) for v in imu.rpy],
                temperature=int(imu.temperature),
            ),
            joints=joints,
        )
