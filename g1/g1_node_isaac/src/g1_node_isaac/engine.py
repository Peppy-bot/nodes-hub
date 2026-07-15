"""Isaac Sim G1 engine: the structural twin of the MuJoCo engine.

Same interface as `g1_node_mujoco.engine` (step / snapshot / set_velocity /
set_posture) so the node wiring is identical; the physics backend is Isaac Sim.
Isaac is provided by the Isaac Sim runtime (not a pip wheel), so the import is
guarded: the module loads and the node builds without Isaac present, and
constructing the engine off an Isaac runtime raises `IsaacUnavailable`.

The step body is the integration seam. On an Isaac Sim machine it loads the G1
USD, steps the stage, holds the robot with a joint PD, and (once a pretrained
`unitree_rl_gym` policy is wired) walks it. The seam is marked below; the
telemetry snapshot shape matches the shared contracts exactly like the MuJoCo
engine, so the node's publishers are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MOTOR_COUNT = 35


@dataclass
class Snapshot:
    fsm_id: int = 1
    tick: int = 0
    quaternion: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    gyroscope: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    accelerometer: list[float] = field(default_factory=lambda: [0.0, 0.0, 9.81])
    rpy: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    linear_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    q: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)
    dq: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)
    tau_est: list[float] = field(default_factory=lambda: [0.0] * MOTOR_COUNT)


class IsaacUnavailable(RuntimeError):
    """Raised when the engine is constructed off an Isaac Sim runtime."""


class G1IsaacEngine:
    """Owns the Isaac Sim stage and the standing controller (see module doc)."""

    def __init__(self, usd_path: str | None = None) -> None:
        try:
            # Provided by the Isaac Sim runtime, not a pip package.
            from isaacsim.simulation_app import SimulationApp  # noqa: F401
        except ImportError as exc:
            raise IsaacUnavailable(
                "Isaac Sim is not available; run g1_node_isaac inside the Isaac "
                "Sim runtime (container packaging mirrors openarm's isaac node)"
            ) from exc

        # Integration seam: boot the SimulationApp, add the G1 USD to the stage,
        # create the ArticulationView, reset to the home pose. The USD is imported
        # from a g1_description lib (URDF -> USD), mirroring openarm's isaac node.
        raise IsaacUnavailable(
            "g1_node_isaac engine is scaffolded; wire the Isaac stage + G1 USD "
            "load and the articulation step to enable it (see module docstring)"
        )

    # The following mirror the MuJoCo engine's interface so __main__ is identical.
    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        raise NotImplementedError

    def set_posture(self, fsm_id: int, pd_enabled: bool) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> Snapshot:
        raise NotImplementedError
