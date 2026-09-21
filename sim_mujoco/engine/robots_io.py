#!/usr/bin/env python3
"""The simulation_robot contract, on the node loop.

A robot joins by attaching with the copy it runs as, the model it is and
where it stands, and its goal runs for as long as it is in the scene. Its
limbs reach it through its own pairs, so nothing here carries motion: what
this serves is who may join, where, how long they stay, and whether the
robot that joined is ready to be driven.

The scene belongs to the thread that steps it, so standing a robot and
taking one out are handed to that thread and waited on: both compose and
compile the scene again, which no other thread may do while physics reads
it.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from typing import Optional

from peppygen.exposed_actions.robots import attach
from peppygen.exposed_services.robots import is_ready

from sim_robot_core.registry import Caller, Registry, Robot

from edits import Edits
from mujoco_models import MujocoModel, MujocoModels
from world import SPOT_PITCH_M, Placement, World, name_in_the_scene, within_one_spot

# How long a robot's stay waits between checks of its own lease, as a share
# of the lease: a robot whose limbs are gone leaves within a quarter of a
# lease of the lease running out. The same tick records whether the robot
# holds the pairs its model asks for, which is what renews its lease.
_LEASE_CHECK_SHARE = 0.25
# The shortest lease check, for a lease too short to share.
_LEASE_CHECK_FLOOR_S = 0.05
# Pause after a runtime error before serving again, so a broken transport
# cannot hot-spin a loop or flood the log.
_RETRY_BACKOFF_S = 1.0
# The same, for the readiness service, which is back well inside a
# readiness poll.
_READY_RETRY_BACKOFF_S = 0.05
logger = logging.getLogger(__name__)


class Ending(enum.Enum):
    """How a robot's stay ended."""

    #: Its caller cancelled the goal.
    LEFT = "left"
    #: Its pairs were not the ones its model asks for, for the lease.
    LAPSED = "lapsed"
    #: Another goal of the same caller hosts the robot now.
    HANDED_OVER = "handed over"
    #: The engine is going down and takes the robot with it.
    STOPPED = "stopped"
    #: Its holder could not be told it stands, so it is taken out.
    FAILED = "failed"


class RobotsIO:
    """Serves how a robot joins the scene and whether it is ready: who may
    attach, where they stand, how long they stay, and whether each holds
    every limb pair its model asks for."""

    def __init__(
        self,
        node_runner,
        loop: asyncio.AbstractEventLoop,
        models: MujocoModels,
        robots: Registry,
        world: World,
        edits: Edits,
        io,
        lease_s: float,
    ) -> None:
        self._node_runner = node_runner
        self._loop = loop
        # The models this engine stands, each with the limbs and cameras a
        # robot of it pairs.
        self._models = models
        self._robots = robots
        # The robots standing in the scene, and the scene they compose.
        self._world = world
        # The changes waiting for the thread that steps the scene.
        self._edits = edits
        # The limb pairs, which are what a robot is driven through and what
        # keeps it in the scene.
        self._io = io
        if lease_s <= 0.0:
            raise ValueError(f"robot_lease_ms must be positive, got {lease_s * 1000:g}")
        self._lease_s = lease_s
        # Set by the launch once the engine exists: a scene that changed has
        # to be composed again before it is stepped, and the engine reading
        # the scene lets go of it first.
        self._rebind = lambda: None
        self._unbind = lambda: None
        # Where the robots admitted but not standing yet were promised to
        # stand, beside the name the admission in flight reserved. The node
        # loop admits, stands and forgets, so one admission finishes with
        # both before the next reads either.
        self._placements: dict[str, Placement] = {}
        self._tasks: list[asyncio.Task] = []
        self._stays: set[asyncio.Task] = set()
        # The signal that ends each robot's stay without taking the robot out
        # of the scene, which is how a re-registering copy takes its own
        # robot over. Node loop only.
        self._handovers: dict[str, asyncio.Event] = {}
        self._stopping = asyncio.Event()
        # The name an admission reserved, until the goal it was reserved for
        # reaches its stay. Node loop only.
        self._admitted: Optional[str] = None

    def binds_with(self, rebind, unbind=None) -> None:
        """How the engine composes the scene again once it changed, and how
        it lets go of the scene first. Standing a robot and taking one out
        both replace the model the engine reads."""
        self._rebind = rebind
        self._unbind = unbind or (lambda: None)

    def stand(self, name: str, known: MujocoModel, placement: Placement) -> None:
        """Stands a robot of the model `known` and composes the scene around
        it, on the thread that steps the scene. A robot the scene cannot take
        (a model asking for other simulation settings, a camera whose body
        its MJCF lacks) is taken back out, so one robot that cannot join
        never takes the scene down."""
        try:
            self._unbind()
            self._world.add(name, known, placement)
            try:
                self._rebind()
            except Exception:
                self._world.remove(name)
                raise
        except Exception:
            # Standing let go of the scene before it changed it. Whatever
            # went wrong, the scene is taken up again here, so it keeps
            # running after a robot fails to join.
            self._rebind()
            raise
        finally:
            self._robots.renew(self._loop.time())

    def unstand(self, name: str) -> None:
        """Takes a robot out and composes the scene around the robots that
        remain, on the thread that steps the scene. What the robot's joining
        held went back before this was asked for, so the scene is all that is
        left to change."""
        try:
            self._unbind()
            self._world.remove(name)
            self._io.forget(name)
        finally:
            self._rebind()
        self._robots.renew(self._loop.time())

    async def start(self) -> None:
        """Exposes the contract and spawns its loops. Runs on the node loop
        before the sim thread starts, so a robot that attaches early waits
        while the scene compiles and then stands."""
        handle = await attach.ActionHandle.expose(self._node_runner)
        self._tasks = [
            asyncio.create_task(self._serve_attach(handle)),
            asyncio.create_task(self._serve_ready()),
            asyncio.create_task(self._watch_pairs()),
        ]
        logger.info(
            "the scene is open: robots attach with a model of %s, lease %.1fs",
            ", ".join(self._models.names()),
            self._lease_s,
        )

    async def stop(self) -> None:
        """Ends every stay and stops serving. A robot whose engine is going
        down is told so on its own goal."""
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        for task in list(self._stays):
            task.cancel()
        await asyncio.gather(*self._tasks, *self._stays, return_exceptions=True)

    async def _watch_pairs(self) -> None:
        """Records which robots hold the pairs their model asks for, on the
        same tick their stay checks its lease. A robot's pairs dissolve when
        its nodes stop, so this is how a robot that is gone stops renewing,
        and a robot paired as another model never renews at all."""
        period = self._lease_check_period()
        while True:
            try:
                matched = {
                    robot.name for robot in self._robots.robots() if self._mismatch(robot) is None
                }
                self._robots.note_paired(matched, self._loop.time())
            except asyncio.CancelledError:
                raise
            except Exception:  # pylint: disable=W0718
                logger.exception("reading the limb pairs failed")
            await asyncio.sleep(period)

    async def _serve_attach(self, handle: attach.ActionHandle) -> None:
        while True:
            try:
                context = await handle.handle_goal_next_request(self._admit)
                if context is None:
                    self._withdraw()
                    return
                self._admitted = None
                stay = asyncio.create_task(self._stay(context))
                self._stays.add(stay)
                stay.add_done_callback(self._stays.discard)
            except asyncio.CancelledError:
                raise
            except Exception:  # pylint: disable=W0718
                self._withdraw()
                logger.exception("attach failed")
                await asyncio.sleep(_RETRY_BACKOFF_S)

    def _withdraw(self) -> None:
        """Gives back what an admission reserved for a goal that never
        reached its stay. A goal is accepted on a hop of its own after this
        engine has answered for it, and a robot with no stay has nothing
        watching its lease, so its name and its spot are held by nobody."""
        name, self._admitted = self._admitted, None
        if name is None:
            return
        self._forget(name)
        self._robots.release(name)
        logger.warning(
            "the robot admitted as '%s' never reached its stay: its name and its spot "
            "are free again",
            name,
        )

    def _lease_check_period(self) -> float:
        """How often a stay checks its lease: a share of the lease, floored."""
        return max(self._lease_s * _LEASE_CHECK_SHARE, _LEASE_CHECK_FLOOR_S)

    def _admit(self, request: attach.GoalRequest) -> attach.GoalDecision:
        """Whether a robot may join, and the limbs it will drive, under the
        names its own model gives them: the model must be one this engine
        stands, the placement must be free, and the name must be free or
        already this caller's own robot. Admitting reserves the name, or
        hands this caller's standing robot over to the goal being admitted."""
        caller = Caller(core_node=request.core_node, instance_id=request.instance_id)
        self._admitted = None
        try:
            robot_name = name_in_the_scene(request.data.robot)
            entry = self._models.of(request.data.model).entry
        except ValueError as error:
            return attach.GoalDecision.reject(str(error))
        limbs = attach.GoalResponse(
            arm_names=entry.arm_names(),
            arm_joints=entry.arm_joint_counts(),
            gripper_names=entry.gripper_names(),
        )
        try:
            adopted = self._robots.admit(robot_name, entry, caller, self._loop.time())
        except ValueError as error:
            return attach.GoalDecision.reject(str(error))
        if adopted:
            # The robot stands where it stands; only its host changes,
            # and the stay taking it over is what ends the one hosting
            # it now.
            return attach.GoalDecision.accept(limbs)
        try:
            placement = self._placement(request.data.placement, robot_name)
        except ValueError as error:
            self._robots.release(robot_name)
            return attach.GoalDecision.reject(str(error))
        self._placements[robot_name] = placement
        self._admitted = robot_name
        return attach.GoalDecision.accept(limbs)

    def _placement(self, asked, name: str) -> Placement:
        """Where the robot stands: what it asked for, or a spot of the
        engine's own. A spot another robot stands on is refused, because two
        robots in one place resolve their overlap by throwing each other.
        Standing a robot happens later, on the thread that steps the scene,
        so the spots already promised count as taken too."""
        promised = tuple(self._placements.values())
        if asked is None:
            return self._world.free_spot(promised, name)
        placement = Placement.of(asked.position, asked.yaw)
        holder = self._whoever_holds(placement, name)
        if holder is not None:
            raise ValueError(
                f"{holder} within {SPOT_PITCH_M:g} m of {list(placement.position)}: stand "
                "this robot further off, or turn its `placement.auto` on and take the spot "
                "this scene gives it"
            )
        return placement

    def _whoever_holds(self, placement: Placement, name: str) -> Optional[str]:
        """Who is in the way of a robot joining as `name`: the robot standing
        there, or the copy promised the spot while it waits to stand. None
        while the spot is free. A robot standing under `name` itself is the
        one coming back: its name went back when its stay ended, and the
        scene has yet to let it go."""
        standing = self._world.standing_within(placement, name)
        if standing is not None:
            return f"'{standing.instance}' stands"
        promised = next(
            (
                held
                for held, spot in self._placements.items()
                if within_one_spot(placement, spot)
            ),
            None,
        )
        return None if promised is None else f"'{promised}' is about to stand"

    def _handover(self, name: str) -> asyncio.Event:
        """The signal that ends the goal hosting `name` while the robot stays
        in the scene."""
        return self._handovers.setdefault(name, asyncio.Event())

    def _hand_over(self, name: str) -> None:
        """Ends the goal hosting `name` and arms the signal for the goal that
        takes it over."""
        self._handover(name).set()
        self._handovers[name] = asyncio.Event()

    async def _stay(self, context: attach.GoalContext) -> None:
        """One robot's stay in the scene: it joins, it is driven through its
        limb pairs, and it leaves when the goal is cancelled, when its pairs
        have not been its model's for the lease, or when the engine stops.

        A goal admitted for the robot its caller already stands takes that
        stay over instead: the scene is untouched, and the goal that hosted
        the robot ends without taking it out."""
        request = context.request()
        name = request.data.robot
        model = request.data.model
        robot = self._robots.of_name(name)
        if robot is None:
            await context.complete(False, "the name was given back before the robot stood")
            return
        adopted = robot.standing()
        if adopted:
            # The goal that hosted this robot ends here, where the goal
            # taking it over exists and is about to watch the robot's lease.
            # Ending it takes the signal this stay then waits on, so the two
            # happen in this order.
            self._hand_over(name)
            logger.info("robot '%s' (%s) is hosted by its new goal", name, model)
        handover = self._handover(name)
        if not adopted and not await self._stand(context, name, model):
            return
        if not await self._standing(context, name):
            return
        ending, why = await self._watch(context, name, handover)
        await self._take_out(context, name, ending, why)

    async def _stand(self, context: attach.GoalContext, name: str, model: str) -> bool:
        """Puts the robot in the scene, answering the goal itself when the
        scene will not take it, or when the robot leaves while its stand
        still waits for the thread that steps the scene: that stand is
        withdrawn and the name given back at once. False when the robot never
        stood."""
        placement = self._placements.get(name)
        if placement is None:
            # Whatever took the spot back took the name with it, and this
            # goal is the one holding it now.
            self._robots.release(name)
            await context.complete(False, "the name was given back before the robot stood")
            return False
        known = self._models.of(model)
        edit = self._edits.submit(lambda: self.stand(name, known, placement))
        standing = asyncio.wrap_future(edit)
        leaving = asyncio.ensure_future(context.cancel_signal())
        try:
            await asyncio.wait({standing, leaving}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            leaving.cancel()
        if edit.cancel():
            self._forget(name)
            self._robots.release(name)
            logger.info("robot '%s' left before it stood", name)
            await context.complete_cancelled(True, "the robot left before it stood")
            return False
        try:
            await standing
        except Exception as error:  # pylint: disable=W0718
            logger.warning("robot '%s' could not join: %s", name, error)
            self._forget(name)
            self._robots.release(name)
            await context.complete(False, f"the scene could not stand the robot: {error}")
            return False
        self._robots.stand(name, self._loop.time())
        logger.info("robot '%s' (%s) is in the scene", name, model)
        return True

    async def _standing(self, context: attach.GoalContext, name: str) -> bool:
        """Tells the goal's holder the robot stands. False when it could not
        be told: a robot whose holder cannot follow its stay is taken out."""
        try:
            await context.publish_feedback(standing=True)
        except Exception as error:  # pylint: disable=W0718
            await self._take_out(
                context,
                name,
                Ending.FAILED,
                f"the robot's holder could not be told it stands: {error}",
            )
            return False
        return True

    async def _take_out(
        self, context: attach.GoalContext, name: str, ending: Ending, why: str
    ) -> None:
        """Ends a robot's stay the way it ended: a robot handed over stays
        standing for its new goal, one whose engine is stopping goes down
        with the scene, and any other is taken out of the scene first."""
        if ending is Ending.HANDED_OVER:
            logger.info("robot '%s' is hosted by another goal of its copy", name)
            await context.complete(False, why)
            return
        # Everything this robot's joining held goes back before the scene is
        # asked to let it go. Taking a robot out waits for the thread that
        # steps the scene, which is a whole scene composed and compiled, and
        # a copy that comes straight back is admitted inside that window. It
        # finds its name free, so it is stood afresh, under the name and on
        # the spot it is promised then.
        self._forget(name)
        self._robots.release(name)
        if ending is not Ending.STOPPED:
            try:
                await asyncio.wrap_future(self._edits.submit(lambda: self.unstand(name)))
            except Exception as error:  # pylint: disable=W0718
                logger.error(
                    "robot '%s' could not be taken out: %s. It stands with no host "
                    "until this simulation restarts",
                    name,
                    error,
                )
        if ending is Ending.LEFT:
            await context.complete_cancelled(True, why)
        else:
            await context.complete(False, why)

    async def _watch(
        self, context: attach.GoalContext, name: str, handover: asyncio.Event
    ) -> tuple[Ending, str]:
        """Waits for the robot's stay to end, and says how it ended."""
        cancel = asyncio.create_task(context.cancel_signal())
        handed = asyncio.create_task(handover.wait())
        period = self._lease_check_period()
        try:
            while True:
                done, _ = await asyncio.wait(
                    {cancel, handed}, timeout=period, return_when=asyncio.FIRST_COMPLETED
                )
                if handed in done:
                    return (
                        Ending.HANDED_OVER,
                        "this robot is hosted by another goal of this copy",
                    )
                if cancel in done:
                    return Ending.LEFT, "the robot left the scene"
                if self._stopping.is_set():
                    return Ending.STOPPED, "the engine stopped"
                robot = self._robots.of_name(name)
                if robot is None:
                    return Ending.STOPPED, "the name was given back"
                lapse = self._lapse(robot)
                if lapse is not None:
                    return Ending.LAPSED, lapse
        except asyncio.CancelledError:
            return Ending.STOPPED, "the engine stopped"
        finally:
            cancel.cancel()
            handed.cancel()

    def _mismatch(self, robot: Robot) -> Optional[str]:
        """Why the robot's pairs are not the ones its model asks for, naming
        both lists, or None when they are."""
        return robot.entry.mismatch(self._io.held_by(robot.name))

    def _lapse(self, robot: Robot) -> Optional[str]:
        """Why the robot's lease ran out, or None while it runs: its pairs
        are not its model's now, and were not for the lease. They are read
        here as well as on the watcher's tick, so a robot still holding them
        keeps its place however long the node loop went without renewing its
        lease. A robot holding no pair is told its limbs were gone, and any
        other which of its pairs and its model's limbs and cameras differ."""
        now = self._loop.time()
        held = self._io.held_by(robot.name)
        mismatch = robot.entry.mismatch(held)
        if mismatch is None:
            self._robots.note_paired({robot.name}, now)
            return None
        with robot.lock:
            last = robot.last_paired_s
        if now - last <= self._lease_s:
            return None
        if held.is_empty():
            return (
                f"none of this robot's limbs were paired for {self._lease_s:.1f}s, "
                "the lease this scene gives"
            )
        return (
            f"this robot's pairs were not its model's for {self._lease_s:.1f}s, the lease "
            f"this scene gives: {mismatch}"
        )

    def _forget(self, name: str) -> None:
        """Drops what this robot's joining held, leaving the scene alone."""
        self._handovers.pop(name, None)
        self._placements.pop(name, None)

    async def _serve_ready(self) -> None:
        while True:
            try:
                await is_ready.handle_next_request(self._node_runner, self._ready)
            except asyncio.CancelledError:
                raise
            except Exception:  # pylint: disable=W0718
                logger.exception("readiness service failed")
                await asyncio.sleep(_READY_RETRY_BACKOFF_S)

    def _ready(self, request: is_ready.Request) -> is_ready.Response:
        """Whether the asking instance's robot is ready to be driven: it
        stands in the scene and holds a pair for every limb of its model, so
        a setpoint reaches each one and its state comes back. A robot's limbs
        go ready together, so one answer covers them all."""
        caller = Caller(core_node=request.core_node, instance_id=request.instance_id)
        robot = self._robots.of_caller(caller)
        ready = (
            robot is not None
            and robot.standing()
            and robot.entry.holds_every_limb(self._io.held_by(robot.name))
        )
        return is_ready.Response(ready=ready)
