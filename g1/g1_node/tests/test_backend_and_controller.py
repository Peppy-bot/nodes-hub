import asyncio

from g1_node.backend import G1Backend, LocoClientBackend
from g1_node.postures import POSTURE_FSM_ID, POSTURE_METHOD, Posture
from g1_node.__main__ import G1Controller


class FakeBackend(G1Backend):
    """Records every backend call so the controller can be exercised in-memory."""

    def __init__(self):
        self.postures: list[Posture] = []
        self.velocities: list[tuple[float, float, float]] = []
        self.state = (1, 0, 100)

    def transition(self, posture: Posture) -> int:
        self.postures.append(posture)
        return POSTURE_FSM_ID[posture]

    def set_velocity(self, vx, vy, vyaw):
        self.velocities.append((vx, vy, vyaw))

    def read_state(self):
        return self.state


class FakeLoco:
    """Stands in for the SDK LocoClient: every mapped method records its call."""

    def __init__(self):
        self.calls: list[str] = []
        for name in POSTURE_METHOD.values():
            setattr(self, name, self._record(name))

    def _record(self, name):
        def method():
            self.calls.append(name)

        return method


def test_controller_tracks_current_and_forwards_transition():
    backend = FakeBackend()
    controller = G1Controller(backend)
    assert controller.current is None

    fsm = asyncio.run(controller.apply_posture(Posture.DAMP))
    assert fsm == POSTURE_FSM_ID[Posture.DAMP]
    assert controller.current is Posture.DAMP
    assert backend.postures == [Posture.DAMP]


def test_controller_full_bringup_sequence():
    backend = FakeBackend()
    controller = G1Controller(backend)
    for posture in (Posture.DAMP, Posture.STAND_UP, Posture.START):
        asyncio.run(controller.apply_posture(posture))
    assert backend.postures == [Posture.DAMP, Posture.STAND_UP, Posture.START]
    assert controller.current is Posture.START


def test_controller_forwards_velocity_and_deadman_zero():
    backend = FakeBackend()
    controller = G1Controller(backend)
    asyncio.run(controller.apply_velocity(0.4, 0.0, 0.2))
    asyncio.run(controller.stop_base())
    assert backend.velocities == [(0.4, 0.0, 0.2), (0.0, 0.0, 0.0)]


def test_controller_reads_state():
    backend = FakeBackend()
    backend.state = (200, 1, 87)
    controller = G1Controller(backend)
    assert asyncio.run(controller.read_state()) == (200, 1, 87)


def test_loco_backend_dispatches_mapped_sdk_method():
    # Bypass __init__ (which needs the SDK + DDS) and inject a fake client to
    # prove POSTURE_METHOD dispatches to the right LocoClient call.
    backend = object.__new__(LocoClientBackend)
    fake = FakeLoco()
    backend._loco = fake
    backend._last_fsm_id = POSTURE_FSM_ID[Posture.DAMP]

    for posture in Posture:
        fake.calls.clear()
        fsm = backend.transition(posture)
        assert fake.calls == [POSTURE_METHOD[posture]]
        assert fsm == POSTURE_FSM_ID[posture]


def test_loco_backend_set_velocity_calls_move():
    backend = object.__new__(LocoClientBackend)
    moves: list[tuple] = []

    class MoveLoco:
        def Move(self, vx, vy, vyaw):
            moves.append((vx, vy, vyaw))

    backend._loco = MoveLoco()
    backend.set_velocity(0.5, -0.1, 0.3)
    assert moves == [(0.5, -0.1, 0.3)]
