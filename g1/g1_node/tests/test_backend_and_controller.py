import asyncio
import types

from g1_node.backend import (
    MOTOR_COUNT,
    G1Backend,
    LocoClientBackend,
    Telemetry,
)
from g1_node.postures import POSTURE_FSM_ID, POSTURE_METHOD, Posture
from g1_node.__main__ import G1Controller


class FakeBackend(G1Backend):
    """Records every backend call so the controller can be exercised in-memory."""

    def __init__(self):
        self.postures: list[Posture] = []
        self.velocities: list[tuple[float, float, float]] = []
        self.calls: list[tuple] = []
        self.telemetry = Telemetry(fsm_id=1)

    def transition(self, posture):
        self.postures.append(posture)
        return POSTURE_FSM_ID[posture]

    def set_velocity(self, vx, vy, vyaw):
        self.velocities.append((vx, vy, vyaw))

    def stop_move(self):
        self.calls.append(("stop_move",))

    def move_timed(self, vx, vy, omega, duration):
        self.calls.append(("move_timed", vx, vy, omega, duration))

    def balance_stand(self, balance_mode):
        self.calls.append(("balance_stand", balance_mode))

    def set_balance_mode(self, balance_mode):
        self.calls.append(("set_balance_mode", balance_mode))

    def set_stand_height(self, stand_height):
        self.calls.append(("set_stand_height", stand_height))

    def set_speed_mode(self, speed_mode):
        self.calls.append(("set_speed_mode", speed_mode))

    def set_fsm_id(self, fsm_id):
        self.calls.append(("set_fsm_id", fsm_id))

    def get_fsm_id(self):
        return 42

    def set_task_id(self, task_id):
        self.calls.append(("set_task_id", task_id))

    def switch_to_user_ctrl(self):
        self.calls.append(("switch_to_user_ctrl",))

    def switch_to_internal_ctrl(self, mode):
        self.calls.append(("switch_to_internal_ctrl", mode))

    def check_mode(self):
        return "ai"

    def select_mode(self, name):
        self.calls.append(("select_mode", name))

    def release_mode(self):
        self.calls.append(("release_mode",))

    def read_telemetry(self):
        return self.telemetry


class FakeLoco:
    """Stands in for the SDK LocoClient: records every call."""

    def __init__(self):
        self.calls: list[tuple] = []
        for name in set(POSTURE_METHOD.values()) | {
            "StopMove", "SwitchToUserCtrl",
        }:
            setattr(self, name, self._nullary(name))

    def _nullary(self, name):
        return lambda: self.calls.append((name,))

    def Move(self, vx, vy, vyaw):
        self.calls.append(("Move", vx, vy, vyaw))

    def SetVelocity(self, vx, vy, omega, duration):
        self.calls.append(("SetVelocity", vx, vy, omega, duration))

    def SetStandHeight(self, h):
        self.calls.append(("SetStandHeight", h))

    def SetFsmId(self, fsm_id):
        self.calls.append(("SetFsmId", fsm_id))

    def GetFsmId(self):
        return 7


class FakeSwitcher:
    def __init__(self):
        self.calls: list[tuple] = []

    def CheckMode(self):
        return {"name": "ai"}

    def SelectMode(self, name):
        self.calls.append(("SelectMode", name))

    def ReleaseMode(self):
        self.calls.append(("ReleaseMode",))


def _loco_backend(loco, switcher=None):
    """Build a LocoClientBackend without its SDK/DDS __init__."""
    backend = object.__new__(LocoClientBackend)
    backend._loco = loco
    backend._switcher = switcher or FakeSwitcher()
    backend._last_fsm_id = POSTURE_FSM_ID[Posture.DAMP]
    backend._latest_lowstate = None
    return backend


# --- controller ---------------------------------------------------------------


def test_controller_full_bringup_sequence():
    backend = FakeBackend()
    controller = G1Controller(backend)
    for posture in (Posture.DAMP, Posture.SQUAT_TO_STAND, Posture.START):
        asyncio.run(controller.apply_posture(posture))
    assert backend.postures == [Posture.DAMP, Posture.SQUAT_TO_STAND, Posture.START]
    assert controller.current is Posture.START


def test_controller_velocity_and_deadman():
    backend = FakeBackend()
    controller = G1Controller(backend)
    asyncio.run(controller.apply_velocity(0.4, 0.0, 0.2))
    asyncio.run(controller.stop_base())
    assert backend.velocities == [(0.4, 0.0, 0.2), (0.0, 0.0, 0.0)]


def test_controller_call_sync_dispatches_service():
    backend = FakeBackend()
    controller = G1Controller(backend)
    controller.call_sync(backend.set_stand_height, 0.75)
    assert ("set_stand_height", 0.75) in backend.calls


def test_controller_reads_telemetry():
    backend = FakeBackend()
    backend.telemetry = Telemetry(fsm_id=200, mode_pr=1, mode_machine=2, tick=99)
    controller = G1Controller(backend)
    t = asyncio.run(controller.read_telemetry())
    assert (t.fsm_id, t.mode_pr, t.mode_machine, t.tick) == (200, 1, 2, 99)


# --- LocoClientBackend SDK dispatch ------------------------------------------


def test_loco_backend_transition_dispatch_all_postures():
    loco = FakeLoco()
    backend = _loco_backend(loco)
    for posture in Posture:
        loco.calls.clear()
        fsm = backend.transition(posture)
        assert loco.calls == [(POSTURE_METHOD[posture],)]
        assert fsm == POSTURE_FSM_ID[posture]


def test_loco_backend_service_methods_dispatch():
    loco = FakeLoco()
    backend = _loco_backend(loco)
    backend.set_velocity(0.5, -0.1, 0.3)
    backend.stop_move()
    backend.move_timed(0.2, 0.0, 0.1, 1.5)
    backend.set_stand_height(0.7)
    backend.set_fsm_id(4)
    assert ("Move", 0.5, -0.1, 0.3) in loco.calls
    assert ("StopMove",) in loco.calls
    assert ("SetVelocity", 0.2, 0.0, 0.1, 1.5) in loco.calls
    assert ("SetStandHeight", 0.7) in loco.calls
    assert ("SetFsmId", 4) in loco.calls


def test_loco_backend_get_fsm_id_normalizes_and_caches():
    backend = _loco_backend(FakeLoco())
    assert backend.get_fsm_id() == 7
    assert backend._last_fsm_id == 7


def test_loco_backend_mode_switch_dispatch():
    switcher = FakeSwitcher()
    backend = _loco_backend(FakeLoco(), switcher)
    assert "ai" in backend.check_mode()
    backend.select_mode("normal")
    backend.release_mode()
    assert ("SelectMode", "normal") in switcher.calls
    assert ("ReleaseMode",) in switcher.calls


# --- telemetry decode from a synthetic LowState ------------------------------


def _fake_lowstate():
    imu = types.SimpleNamespace(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[0.1, 0.2, 0.3],
        accelerometer=[0.0, 0.0, 9.8],
        rpy=[0.01, 0.02, 0.03],
        temperature=35,
    )
    motors = [
        types.SimpleNamespace(
            q=float(i), dq=float(i) * 0.1, tau_est=float(i) * 0.5,
            temperature=[40 + i, 41 + i], vol=48.0, mode=1,
        )
        for i in range(MOTOR_COUNT)
    ]
    return types.SimpleNamespace(
        mode_pr=3, mode_machine=5, tick=1234, imu_state=imu, motor_state=motors
    )


def test_read_telemetry_decodes_lowstate():
    backend = _loco_backend(FakeLoco())
    backend._last_fsm_id = 200
    backend._latest_lowstate = _fake_lowstate()

    t = backend.read_telemetry()
    assert (t.fsm_id, t.mode_pr, t.mode_machine, t.tick) == (200, 3, 5, 1234)
    assert t.imu.quaternion == [1.0, 0.0, 0.0, 0.0]
    assert t.imu.gyroscope == [0.1, 0.2, 0.3]
    assert t.imu.temperature == 35
    assert len(t.joints.q) == MOTOR_COUNT
    assert t.joints.q[3] == 3.0
    assert t.joints.tau_est[4] == 2.0
    assert t.joints.temperature[2] == 42  # first of the two sensors, motor idx 2
    assert t.joints.motor_mode[0] == 1


def test_read_telemetry_without_lowstate_returns_defaults():
    backend = _loco_backend(FakeLoco())
    backend._last_fsm_id = 1
    t = backend.read_telemetry()
    assert t.fsm_id == 1
    assert len(t.joints.q) == MOTOR_COUNT
    assert t.imu.quaternion == [1.0, 0.0, 0.0, 0.0]
