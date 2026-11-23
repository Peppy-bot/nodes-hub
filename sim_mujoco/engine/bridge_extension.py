#!/usr/bin/env python3
# pylint: disable=C0413
"""MujocoBridgeExtension owns the physics tick of the scene. Each step it
applies the latest setpoint of every limb and base of every robot standing,
clamps the integral of each wheel's rate loop, advances physics, and
(throttled to state_rate_hz) publishes MuJoCo's own clock followed by the
measured joint, gripper and wheel state it stamps and each robot's record on
object_state under the same stamp, and records where each robot that rolls
stands.

Every robot is driven through its own limb and base pairs and publishes its
state back on them. A base's wheels run at the rates of its pair's latest
setpoint, and at rate 0 while it holds none; once a reading of the pairs
finds that pair gone, or the setpoint outlives its leader's window on the
engine's clock, they are held where they stand, the integral of each wheel's
rate loop cleared (sim_robot_core.setpoints). A robot's joints, actuators and
cameras carry its own prefix in the scene, which is what tells one robot's
limbs from another's when two robots of the same model stand together.
Transport is typed peppygen via SimTopicIO; there is no JSON and no raw
peppylib on the path.
"""
from __future__ import annotations

import logging
import time

from control_core_py.geometry import normalized, rotate, xyzw_of_wxyz
from sim_robot_core.cameras import FramePacer
from sim_robot_core.object_state import RobotRecord
from sim_robot_core.setpoints import MAX_VALUE
from sim_robot_core.world import rolled_placement
from robot_description_py import FingerSpan

from exts import (
    MujocoActuatorCtrl,
    MujocoArticulation,
    MujocoCameraSensor,
    MujocoGripperSensor,
    MujocoWheelDrive,
    home_inertia,
)
from exts.camera_sensor import Rig
from sim_topics import SimTopicIO
from world import Robot, World

logger = logging.getLogger(__name__)


class _RobotLimbs:
    """One standing robot's own views of the scene: the actuators its limbs
    are driven through, the sensor each gripper is read from, the span each
    finger travels, the rate loops of its base's wheels, where its joints sit
    in the scene's state, and where the free joint of a robot that rolls
    does."""

    def __init__(self, robot: Robot, model, data) -> None:
        self.robot = robot
        self.entry = robot.known.entry
        self._model = model
        self._data = data
        # One actuator controller per robot: it resolves that robot's
        # actuators by their scene names, applies the model's MIT gains to
        # its arm joints when its entry carries some, and leaves every other
        # actuator on its MJCF defaults.
        self.actuator = MujocoActuatorCtrl(model, data, params=self._gains())
        self.gripper_sensors: dict[str, MujocoGripperSensor] = {}
        # Where the gripper's closed and open states put each finger joint,
        # as the robot's description gives them; commanded opening fractions
        # place the joint on that span, and measured positions read back as
        # one.
        self.finger_spans: dict[str, list[FingerSpan]] = {}
        # The fingers of each gripper an actuator drives, by their scene
        # names, with their spans. A finger an equality couples to a driven
        # one follows it and is measured alone.
        self.driven_fingers: dict[str, list[tuple[str, FingerSpan]]] = {}
        # Last force limit written per gripper, so the cap is not re-sent per
        # tick.
        self.applied_effort: dict[str, float] = {}
        # The arms whose setpoint's control lies past what MuJoCo holds, each
        # warned of once until a setpoint of it fits.
        self.refusing: set[str] = set()
        # Where each joint of this robot sits in the scene's joint order.
        self.joint_index: dict[str, int] = {}
        # The wheels of its base, by their scene names, left then right; none
        # for a robot without a base.
        wheels = [self.scene_name(joint) for joint in self.entry.wheel_joints()]
        self.wheels = MujocoWheelDrive(model, data, wheels) if wheels else None
        # Where the free joint a robot that rolls hangs on sits in qpos and
        # in qvel.
        self.free_qpos: int | None = None
        self.free_dof: int | None = None

    def scene_name(self, joint: str) -> str:
        return self.robot.scene_name(joint)

    def _gains(self) -> dict:
        params = self.robot.known.actuator_params()
        return {
            **params,
            "joint_names": [self.scene_name(joint) for joint in params["joint_names"]],
        }

    def setup(self, joint_index: dict[str, int], inertia: "list[float]") -> None:
        """Resolves this robot in the scene just compiled. A joint its entry
        names that the scene lacks is refused here, with its name."""
        missing = sorted(
            {
                joint
                for joint in self.entry.joints()
                if self.scene_name(joint) not in joint_index
            }
        )
        if missing:
            raise RuntimeError(
                f"the {self.entry.model} entry names joints not in the scene standing "
                f"'{self.robot.instance}': {missing}"
            )
        self.joint_index = {
            joint: joint_index[self.scene_name(joint)] for joint in self.entry.joints()
        }
        if not self.actuator.setup(inertia):
            raise RuntimeError(f"MujocoActuatorCtrl setup failed for '{self.robot.instance}'")
        for gripper in self.entry.grippers:
            driven = [
                (self.scene_name(joint), span)
                for joint, span in zip(gripper.joints, gripper.spans)
                if self.actuator.drives(self.scene_name(joint))
            ]
            if not driven:
                raise RuntimeError(
                    f"no actuator of the scene standing '{self.robot.instance}' drives a finger "
                    f"of gripper '{gripper.name}': {list(gripper.joints)}"
                )
            self.actuator.require_force_limited([name for name, _ in driven])
            self.driven_fingers[gripper.name] = driven
        for gripper in self.entry.grippers:
            fingers = [self.scene_name(joint) for joint in gripper.joints]
            sensor = MujocoGripperSensor(self._model, self._data, finger_joints=fingers)
            if not sensor.setup():
                raise RuntimeError(
                    f"MujocoGripperSensor setup failed for gripper '{gripper.name}' of "
                    f"'{self.robot.instance}'"
                )
            self.gripper_sensors[gripper.name] = sensor
            self.finger_spans[gripper.name] = list(gripper.spans)
        if self.wheels is not None:
            self.wheels.setup()
        if self.robot.known.rolls:
            import mujoco  # pylint: disable=C0415

            free = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, self.robot.instance)
            if free < 0:
                raise RuntimeError(
                    f"the scene standing '{self.robot.instance}' has no free joint of that name, "
                    "which a robot that rolls hangs from the world on"
                )
            self._hold_mount_to_robot_frame()
            self.free_qpos = int(self._model.jnt_qposadr[free])
            self.free_dof = int(self._model.jnt_dofadr[free])

    def _hold_mount_to_robot_frame(self) -> None:
        """Refuses a robot that rolls whose mount body the compiled scene
        places off the free body it hangs on, or turns in it, whatever its
        MJCF writes the pose with (pos, quat, euler, axisangle, xyaxes,
        zaxis, a frame around it): the free joint is its robot frame, which
        the engine reports and places it by, and stands where the mount body
        does only when the body sits at its origin unturned."""
        mount = self.robot.known.robot.mount_body
        body = self._model.body(self.scene_name(mount)).id
        position = [float(value) for value in self._model.body_pos[body]]
        turn = [float(value) for value in self._model.body_quat[body]]
        if position == [0.0, 0.0, 0.0] and turn == [1.0, 0.0, 0.0, 0.0]:
            return
        raise RuntimeError(
            f"the MJCF of {self.entry.model} places its mount body '{mount}' at {position}, "
            f"turned by {turn} (w, x, y, z): a robot that rolls stands on its robot frame, so give "
            "the mount body no pose of its own and no frame around it"
        )

    def fits(self, arm: str, controls: dict[int, float]) -> bool:
        """Whether the controls a setpoint of `arm` gives its actuators, with
        the feedforward of its velocities, are each a number from -MAX_VALUE
        to MAX_VALUE, which MuJoCo holds. One past it drops the setpoint, as
        one that is not a number is dropped, and the arm holds the setpoint
        before it; the log names the arm once, until a setpoint of it fits."""
        beyond = [ctrl for ctrl in controls.values() if not abs(ctrl) <= MAX_VALUE]
        if not beyond:
            self.refusing.discard(arm)
            return True
        if arm not in self.refusing:
            self.refusing.add(arm)
            logger.warning(
                "dropping the setpoints for '%s' %s until one fits: with the feedforward of "
                "its velocities it asks a control of %g, and a control is a number from %g to %g",
                self.robot.instance,
                arm,
                beyond[0],
                -MAX_VALUE,
                MAX_VALUE,
            )
        return False

    def measured(self, states, joints) -> tuple[list[float], list[float]]:
        """The positions and velocities of `joints` of this robot, out of
        `states`, the scene's joint positions and velocities in its joint
        order."""
        positions, velocities = states
        indices = [self.joint_index[joint] for joint in joints]
        return [positions[index] for index in indices], [velocities[index] for index in indices]

    def robot_record(self) -> RobotRecord:
        """This robot's record on object_state as the scene stands now: a
        robot that rolls where its free joint stands, its robot frame, with
        the free joint's velocities in world axes (MuJoCo gives its angular
        velocity in the body's own axes), and any other at its placement, at
        rest."""
        robot = self.robot
        if self.free_qpos is None:
            placement = robot.placement
            return RobotRecord.fixed(
                robot.instance,
                robot.model,
                placement.position.position,
                placement.orientation.quaternion,
            )
        at, dof = self.free_qpos, self.free_dof
        qpos, qvel = self._data.qpos, self._data.qvel
        orientation = normalized(xyzw_of_wxyz([float(q) for q in qpos[at + 3 : at + 7]]))
        own_axes = tuple(float(w) for w in qvel[dof + 3 : dof + 6])
        return RobotRecord(
            robot=robot.instance,
            model=robot.model,
            position=tuple(float(p) for p in qpos[at : at + 3]),
            orientation=orientation,
            linear_velocity=tuple(float(v) for v in qvel[dof : dof + 3]),
            angular_velocity=rotate(orientation, own_axes),
        )

    def rolled_placement(self):
        """Where the wheels of a robot that rolls carried it: the origin of
        its free joint, facing the heading its body turned to."""
        at = self.free_qpos
        qpos = self._data.qpos
        return rolled_placement(qpos[at : at + 3], xyzw_of_wxyz(qpos[at + 3 : at + 7]))

    def start_in_posture(self) -> None:
        """Puts this robot in the posture its model starts in, and holds it
        there: the joints are placed and the actuator of each joint one drives
        targets the same position, so the arm stands still until its first
        setpoint. A finger an equality couples to another has no actuator."""
        import mujoco  # pylint: disable=C0415

        posture = self.entry.start_posture
        placed = {self.scene_name(joint): position for joint, position in posture.items()}
        for joint, position in placed.items():
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            self._data.qpos[self._model.jnt_qposadr[jid]] = position
        self.actuator.write_targets(
            {joint: position for joint, position in placed.items() if self.actuator.drives(joint)}
        )

    def teardown(self) -> None:
        self.actuator.teardown()
        for sensor in self.gripper_sensors.values():
            sensor.teardown()
        if self.wheels is not None:
            self.wheels.teardown()


class MujocoBridgeExtension:
    """Drives every robot standing in the scene from its own command streams,
    publishes its state on its own pairs, and records in the world where each
    robot that rolls stands."""

    def __init__(
        self,
        model,
        data,
        io: SimTopicIO,
        world: World,
        state_rate_hz: int,
        renders: bool,
        time_base_s: float,
        camera_counters=None,
    ) -> None:
        self._model = model
        self._data = data
        self._io = io
        self._world = world
        self._robots: list[Robot] = world.robots()
        # State publishes ride an absolute grid at state_rate_hz: serializing
        # every reader at the ~500 Hz physics tick saturates the single sim
        # thread. Writers and the physics step still run every tick.
        if state_rate_hz <= 0:
            raise ValueError(f"state_rate_hz must be positive, got {state_rate_hz}")
        self._state_pacer = FramePacer(state_rate_hz)
        # The engine clock runs on from where the previous scene left it, so
        # the fleet's time never goes back when a robot joins or leaves.
        self._time_base_s = time_base_s
        self._articulation = MujocoArticulation(model, data)
        self._limbs: dict[str, _RobotLimbs] = {
            robot.instance: _RobotLimbs(robot, model, data) for robot in self._robots
        }
        # The cameras of every robot whose model carries a rig, rendered off
        # this thread when the engine renders at all.
        self._camera_sensor = (
            MujocoCameraSensor(model, self._rigs(), io, camera_counters)
            if renders and self._rigs()
            else None
        )

    def _rigs(self) -> "list[Rig]":
        return [
            Rig(
                robot=robot.instance,
                prefix=robot.prefix,
                cameras=tuple(robot.known.entry.cameras),
            )
            for robot in self._robots
            if robot.known.entry.cameras
        ]

    def startup(self, posture_for: "list[Robot]" = ()) -> None:
        """Resolves every robot in the scene and starts rendering. The robots
        of `posture_for` are put in the posture their model starts in; the
        others carried their state onto this scene and keep it."""
        if not self._articulation.setup():
            raise RuntimeError("MujocoArticulation setup failed")
        joint_index = {
            name: index for index, name in enumerate(self._articulation.get_joint_names())
        }
        inertia = home_inertia(self._model)
        for limbs in self._limbs.values():
            limbs.setup(joint_index, inertia)
        for robot in posture_for:
            self._limbs[robot.instance].start_in_posture()
        if posture_for:
            import mujoco  # pylint: disable=C0415

            mujoco.mj_forward(self._model, self._data)
        if self._camera_sensor is not None:
            self._camera_sensor.start()
        logger.info(
            "MujocoBridgeExtension ready for %s",
            ", ".join(
                f"'{limbs.robot.instance}' ({limbs.entry.model}) with "
                f"{len(limbs.entry.arms)} arm(s), {len(limbs.entry.grippers)} gripper(s)"
                for limbs in self._limbs.values()
            )
            or "an empty scene",
        )

    def engine_time_s(self) -> float:
        """The engine clock: the scenes before this one, plus MuJoCo's own
        time in this one, which mj_step advances."""
        return self._time_base_s + float(self._data.time)

    def step(self) -> None:
        import mujoco  # pylint: disable=C0415

        self._apply_commands()
        mujoco.mj_step(self._model, self._data)
        # `data.time` is MuJoCo's own clock, advanced by mj_step above, so a
        # paused engine stops advancing it. Recorded ahead of every stamp of
        # this step, the camera snapshot included.
        self._io.record_engine_time(self.engine_time_s())
        if self._camera_sensor is not None:
            self._camera_sensor.snapshot(self._data.qpos)
            self._camera_sensor.raise_if_failed()

        if not self._state_pacer.take_if_due(time.monotonic()):
            return
        # Published before the state it stamps; a stopped engine stops the
        # domain's clock with it.
        self._io.publish_clock_tick()
        self._publish_state()

    def _apply_commands(self) -> None:
        setpoints = self._io.setpoints
        now_s = self.engine_time_s()
        for limbs in self._limbs.values():
            robot = limbs.robot.instance
            for arm in limbs.entry.arms:
                setpoint = setpoints.arm(robot, arm.name)
                if setpoint is None:
                    continue
                joints = [limbs.scene_name(joint) for joint in arm.joints]
                velocities = dict(zip(joints, setpoint.velocities)) if setpoint.velocities else None
                positions = dict(zip(joints, setpoint.positions))
                controls = limbs.actuator.controls(positions, velocities)
                if limbs.fits(arm.name, controls):
                    limbs.actuator.write_controls(controls)

            if limbs.wheels is not None:
                setpoint = setpoints.base(robot, now_s)
                if setpoints.take_base_release(robot):
                    limbs.wheels.hold()
                stopped = (0.0,) * len(limbs.entry.wheel_joints())
                limbs.wheels.drive(stopped if setpoint is None else setpoint.velocities)
                limbs.wheels.clamp_integrals()

            for gripper in limbs.entry.grippers:
                setpoint = setpoints.gripper(robot, gripper.name)
                if setpoint is None:
                    continue
                driven = limbs.driven_fingers[gripper.name]
                joints = [name for name, _ in driven]
                # Re-applied only on change: the cap is a model write, not a
                # per-tick target.
                if limbs.applied_effort.get(gripper.name) != setpoint.max_effort:
                    # Recorded only once written, so a not-ready tick retries.
                    if limbs.actuator.set_force_limit(joints, setpoint.max_effort):
                        limbs.applied_effort[gripper.name] = setpoint.max_effort
                # Map the opening fraction onto each driven finger's own span,
                # so the same command drives prismatic fingers, revolute ones
                # and a single jaw.
                limbs.actuator.write_targets(
                    {name: span.position_at(setpoint.opening) for name, span in driven}
                )

    def _publish_state(self) -> None:
        states = self._articulation.get_joint_states()
        for limbs in self._limbs.values():
            robot = limbs.robot.instance
            if states is not None:
                for arm in limbs.entry.arms:
                    self._io.publish_arm_states(
                        robot, arm.name, *limbs.measured(states, arm.joints)
                    )
                if limbs.wheels is not None:
                    self._io.publish_base_states(
                        robot, *limbs.measured(states, limbs.entry.wheel_joints())
                    )

            for name, sensor in limbs.gripper_sensors.items():
                data = sensor.get_gripper_state()
                spans = limbs.finger_spans[name]
                if data and len(data["positions"]) == len(spans):
                    # Opening = mean per-finger travel fraction, the inverse of
                    # the command mapping above.
                    fractions = [
                        span.opening_at(position)
                        for position, span in zip(data["positions"], spans)
                    ]
                    self._io.publish_gripper_states(robot, name, sum(fractions) / len(fractions))

            if limbs.free_qpos is not None:
                self._world.follow(robot, limbs.rolled_placement())
        self._io.publish_object_states([limbs.robot_record() for limbs in self._limbs.values()])

    def shutdown(self) -> dict:
        """Stops every view this scene reads the model through, and says what
        each standing camera counted, so the scene composed next carries on
        from there."""
        logger.info(
            "MujocoBridgeExtension shutting down: %s",
            ", ".join(f"'{name}'" for name in self._limbs) or "an empty scene",
        )
        counters: dict = {}

        def stop_rendering() -> None:
            nonlocal counters
            if self._camera_sensor is None:
                return
            self._camera_sensor.stop()
            counters = self._camera_sensor.counters()

        for close in (
            stop_rendering,
            self._articulation.teardown,
            *(limbs.teardown for limbs in self._limbs.values()),
        ):
            try:
                close()
            except Exception:  # pylint: disable=W0718
                logger.exception("a view of this scene did not close")
        return counters
