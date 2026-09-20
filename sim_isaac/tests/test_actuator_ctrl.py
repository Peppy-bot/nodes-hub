"""One limb's actuator controller against a stand-in articulation view: the
gains and effort ceilings of the robot's own model reach the drives of that
limb's joints and no others, and a limb whose model carries none keeps the
drives its stage authors."""

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(_ENGINE_DIR))
sys.path.insert(0, str(_ENGINE_DIR / "exts"))

import actuator_ctrl  # noqa: E402  pylint: disable=C0413
import isaac_models  # noqa: E402  pylint: disable=C0413


class _View:
    """An articulation view: the dofs it reports, and what was written to
    its drives."""

    dof_names: list = []
    inertia: list = []

    def __init__(self, prim_paths_expr, name):
        self.prim, self.name = prim_paths_expr, name
        self.gains = None
        self.max_efforts = [100.0] * len(self.dof_names)

    def initialize(self):
        pass

    def get_mass_matrices(self):
        return [np.diag(self.inertia)]

    def set_gains(self, kps, kds, joint_indices):
        self.gains = (kps[0].tolist(), kds[0].tolist(), joint_indices.tolist())

    def set_max_efforts(self, efforts, joint_indices):
        for index, effort in zip(joint_indices.tolist(), efforts[0].tolist()):
            self.max_efforts[index] = effort

    def get_max_efforts(self):
        return np.array([self.max_efforts])


@pytest.fixture(name="views")
def _views(monkeypatch):
    built = []

    class Articulation(_View):
        def __init__(self, prim_paths_expr, name):
            super().__init__(prim_paths_expr, name)
            built.append(self)

    prims = ModuleType("isaacsim.core.prims")
    prims.Articulation = Articulation
    for name in ("isaacsim", "isaacsim.core"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", prims)
    return Articulation, built


def _so101():
    return isaac_models.IsaacModels.read().of("so101")


def test_an_so101s_arm_takes_its_own_gains_on_its_own_joints(views):
    view_type, built = views
    so101 = _so101()
    (arm,) = so101.entry.arms
    # The jaw sits first on the articulation, and is no joint of the arm.
    view_type.dof_names = ["gripper", *arm.joints]
    view_type.inertia = [0.0] * 6

    ctrl = actuator_ctrl.IsaacActuatorCtrl(
        "/World/charlo", list(arm.joints), name="arm_charlo", params=so101.arm_params(arm)
    )

    assert ctrl.setup()
    (view,) = built
    assert (view.prim, view.name) == ("/World/charlo", "arm_charlo")
    assert view.gains == ([998.22] * 5, [2.731] * 5, [1, 2, 3, 4, 5])
    assert view.max_efforts == [100.0, 2.94, 2.94, 2.94, 2.94, 2.94]


def test_a_drives_damping_is_raised_to_critical_where_the_models_kd_falls_short(views):
    view_type, built = views
    so101 = _so101()
    (jaw,) = so101.entry.grippers
    view_type.dof_names = ["gripper"]
    view_type.inertia = [0.01]

    ctrl = actuator_ctrl.IsaacActuatorCtrl(
        "/World/charlo", list(jaw.joints), name="gripper_charlo", params=so101.gripper_params(jaw)
    )

    assert ctrl.setup()
    kps, kds, _ = built[0].gains
    assert kps == [998.22]
    assert kds == [pytest.approx(2.0 * (998.22 * 0.01) ** 0.5)]


def test_a_limb_whose_model_carries_no_gains_keeps_the_drives_its_stage_authors(views):
    view_type, built = views
    view_type.dof_names = ["gripper"]
    view_type.inertia = [0.01]
    params = {"joint_names": ["gripper"], "kp": [], "kd": [], "max_efforts": []}

    ctrl = actuator_ctrl.IsaacActuatorCtrl("/World/charlo", ["gripper"], name="g", params=params)

    assert ctrl.setup()
    assert built[0].gains is None
    assert built[0].max_efforts == [100.0]


def test_an_unset_wire_effort_restores_the_models_own_ceiling(views):
    view_type, built = views
    so101 = _so101()
    (jaw,) = so101.entry.grippers
    view_type.dof_names = ["gripper"]
    view_type.inertia = [0.0]
    ctrl = actuator_ctrl.IsaacActuatorCtrl(
        "/World/charlo", list(jaw.joints), name="gripper_charlo", params=so101.gripper_params(jaw)
    )
    assert ctrl.setup()

    assert ctrl.set_force_limit(["gripper"], 1.5)
    assert built[0].max_efforts == [1.5]
    assert ctrl.set_force_limit(["gripper"], 0.0)
    assert built[0].max_efforts == [2.94]


def test_an_openarms_arm_takes_the_servo_gains_and_torque_caps_of_its_model(views):
    view_type, built = views
    openarm = isaac_models.IsaacModels.read().of("openarm_v2")
    left = openarm.entry.arms[0]
    view_type.dof_names = list(left.joints)
    view_type.inertia = [0.0] * 7

    ctrl = actuator_ctrl.IsaacActuatorCtrl(
        "/World/alpha", list(left.joints), name="left_arm_alpha", params=openarm.arm_params(left)
    )

    assert ctrl.setup()
    assert built[0].gains[0] == [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0]
    assert built[0].max_efforts == [40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0]
