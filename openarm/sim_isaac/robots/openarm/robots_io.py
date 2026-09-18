#!/usr/bin/env python3
"""The simulation_robot contract, on the node loop.

A robot joins by attaching with the copy it runs as, the model it is and
where it stands, and its goal runs for as long as it is in the scene. Its
limbs reach it through its own pairs, so nothing here carries motion: what
this serves is who may join, how long they stay, and whether a robot that
joined is ready to be driven.

The scene belongs to the thread that steps it, so standing a robot and taking
one out are handed to that thread as edits and waited on here.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import threading

from peppygen.exposed_actions.robots import attach
from peppygen.exposed_services.robots import is_ready

from edits import Edits
from robots import Caller, Limbs, Registry
from world import Placement, World

# How long a robot's stay waits between checks of its own lease, as a share
# of the lease: a robot whose limbs are gone leaves within a quarter of a
# lease of the lease running out. The same tick records which robots are
# holding a pair, which is what renews a lease.
_LEASE_CHECK_SHARE = 0.25
# The shortest lease check, for a lease too short to share.
_LEASE_CHECK_FLOOR_S = 0.05
# Pause after a runtime error before serving again, so a broken transport
# cannot hot-spin a loop or flood the log.
_RETRY_BACKOFF_S = 1.0
# The same, for the readiness service: every robot's readiness runs through
# that one loop, so it is back well inside a readiness poll.
_READY_RETRY_BACKOFF_S = 0.05

logger = logging.getLogger(__name__)


class Ending(enum.Enum):
    """How a robot's stay ended."""

    #: Its caller cancelled the goal.
    LEFT = "left"
    #: Its limbs held no pair for the lease.
    LAPSED = "lapsed"
    #: Another goal of the same caller hosts the robot now.
    HANDED_OVER = "handed over"
    #: The engine is going down and takes every robot with it.
    STOPPED = "stopped"
    #: Its holder could not be told it stands, so it is taken out.
    FAILED = "failed"



class RobotsIO:
    """Serves how a robot joins the scene and whether it is ready: who may
    attach, how long a robot stays, and which robots hold every limb pair."""

    def __init__(
        self,
        node_runner,
        loop: asyncio.AbstractEventLoop,
        world: World,
        robots: Registry,
        edits: Edits,
        io,
        limbs: Limbs,
        lease_s: float,
    ) -> None:
        self._node_runner = node_runner
        self._loop = loop
        self._world = world
        self._robots = robots
        self._edits = edits
        # The limb pairs, which are what a robot is driven through and what
        # keeps it in the scene.
        self._io = io
        self._limbs = limbs
        if lease_s <= 0.0:
            raise ValueError(f"robot_lease_ms must be positive, got {lease_s * 1000:g}")
        self._lease_s = lease_s
        # Set by the launch once the engine exists: a scene that changed has
        # to be resolved again before it is stepped, and an engine that reads
        # it through handles it cannot keep lets go of them first.
        self._rebind = lambda: None
        self._unbind = lambda: None
        self._placements: dict[str, Placement] = {}
        # Guards the placements promised to robots that have not stood yet.
        self._admit_lock = threading.Lock()
        self._tasks: list[asyncio.Task] = []
        self._stays: set[asyncio.Task] = set()
        # The signal that ends each robot's stay without taking the robot
        # out of the scene, which is how a re-registering copy takes its own
        # robot over. Node loop only.
        self._handovers: dict[str, asyncio.Event] = {}
        self._stopping = threading.Event()

    def _lease_check_period(self) -> float:
        """How often a stay checks its lease: a share of the lease, floored."""
        return max(self._lease_s * _LEASE_CHECK_SHARE, _LEASE_CHECK_FLOOR_S)

    def binds_with(self, rebind, unbind=None) -> None:
        """How the engine resolves the scene again once it changed, and how
        it lets go of the scene first. Standing a robot and taking one out
        both replace what the engine reads."""
        self._rebind = rebind
        self._unbind = unbind or (lambda: None)

    def stand(self, name: str, model: str, placement: Placement) -> None:
        """Stands a robot and resolves the scene around it, on the thread
        that steps the scene. A robot the engine cannot resolve (a model
        whose joints are not the ones this engine drives) is taken back out,
        so one robot that cannot join never takes the scene down."""
        self._unbind()
        try:
            self._world.add(name, model, placement)
            try:
                self._rebind()
            except Exception:
                self._world.remove(name)
                raise
        except Exception:
            # Standing let go of the stage before it changed it. Whatever went
            # wrong, the stage is taken up again here, so the scene keeps
            # running after a robot fails to join.
            self._rebind()
            raise
        finally:
            self._robots.renew(self._loop.time())

    def unstand(self, name: str) -> None:
        """Takes a robot out, gives its name back, and resolves the scene
        around the robots that remain, on the thread that steps the scene.
        The name goes back the moment nothing stands under it: the resolve
        that follows takes most of a second, and a robot that comes straight
        back is asking for its own name inside it. A removal that raises
        keeps the name, because the robot is still standing."""
        self._unbind()
        try:
            self._world.remove(name)
            self._robots.release(name)
            self._io.forget(name)
        finally:
            self._rebind()
            self._robots.renew(self._loop.time())

    async def start(self) -> None:
        """Exposes both contracts and spawns their loops. Runs on the node
        loop before the sim thread starts, so a robot that attaches early
        waits while the scene compiles and then stands."""
        handle = await attach.ActionHandle.expose(self._node_runner)
        self._tasks = [
            asyncio.create_task(self._serve_attach(handle)),
            asyncio.create_task(self._serve_ready()),
            asyncio.create_task(self._watch_pairs()),
        ]
        logger.info(
            "the scene is open: robots attach with a model of %s, lease %.1fs",
            ", ".join(self._world.catalogue().models()),
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
        """Records which robots are holding a limb pair, on the same tick a
        stay checks its lease. A robot's pairs dissolve when its nodes stop,
        so this is how a robot that is gone stops renewing."""
        period = self._lease_check_period()
        while True:
            try:
                self._robots.note_paired(
                    self._io.robots_with_any_limb(), self._loop.time()
                )
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
                    return
                stay = asyncio.create_task(self._stay(context))
                self._stays.add(stay)
                stay.add_done_callback(self._stays.discard)
            except asyncio.CancelledError:
                raise
            except Exception:  # pylint: disable=W0718
                logger.exception("attach failed")
                await asyncio.sleep(_RETRY_BACKOFF_S)

    def _admit(self, request: attach.GoalRequest) -> attach.GoalDecision:
        """Whether a robot may join, and the limbs it will drive: the model
        must be one this engine carries, the placement must be free, and the
        name must be free or already this caller's own robot. Admitting
        reserves the name, or hands this caller's standing robot over to the
        goal being admitted."""
        caller = Caller(core_node=request.core_node, instance_id=request.instance_id)
        try:
            self._world.catalogue().scene(request.data.model)
        except (ValueError, FileNotFoundError) as error:
            return attach.GoalDecision.reject(str(error))
        limbs = attach.GoalResponse(
            arm_names=list(self._limbs.arm_names),
            arm_joints=list(self._limbs.arm_joints),
            gripper_names=list(self._limbs.gripper_names),
        )
        with self._admit_lock:
            try:
                adopted = self._robots.admit(
                    request.data.robot, request.data.model, caller, self._loop.time()
                )
            except ValueError as error:
                return attach.GoalDecision.reject(str(error))
            if adopted:
                # The robot stands where it stands; only its host changes.
                self._hand_over(request.data.robot)
                return attach.GoalDecision.accept(limbs)
            try:
                placement = self._placement(request.data.placement)
            except ValueError as error:
                self._robots.release(request.data.robot)
                return attach.GoalDecision.reject(str(error))
            self._placements[request.data.robot] = placement
        return attach.GoalDecision.accept(limbs)

    def _handover(self, name: str) -> asyncio.Event:
        """The signal that ends the goal hosting `name` while the robot
        stays in the scene."""
        return self._handovers.setdefault(name, asyncio.Event())

    def _hand_over(self, name: str) -> None:
        """Ends the goal hosting `name` and arms the signal for the goal
        that takes it over."""
        self._handover(name).set()
        self._handovers[name] = asyncio.Event()

    def _placement(self, asked) -> Placement:
        """Where the robot stands: what it asked for, or a spot of the
        engine's own. A spot another robot stands on is refused, because two
        robots in one place resolve their overlap by throwing each other.
        Standing a robot happens later, on the thread that steps the scene,
        so the spots already promised count as taken too."""
        promised = tuple(self._placements.values())
        if asked is None:
            return self._world.free_spot(promised)
        placement = Placement.of(asked.position, asked.yaw)
        if self._world.occupied(placement, promised):
            raise ValueError(
                f"a robot already stands within a spot of {list(placement.position)}; join "
                "with placement { auto: true } to take a free spot"
            )
        return placement

    async def _stay(self, context: attach.GoalContext) -> None:
        """One robot's stay in the scene: it joins, it is driven through its
        limb pairs, and it leaves when the goal is cancelled, when its pairs
        are gone for the lease, or when the engine takes it out.

        A goal admitted for the robot its caller already stands takes that
        stay over instead: the scene is untouched, and the goal that hosted
        the robot ends without taking it out."""
        request = context.request()
        name = request.data.robot
        model = request.data.model
        handover = self._handover(name)
        robot = self._robots.of_name(name)
        if robot is None:
            await context.complete(False, "the name was given back before the robot stood")
            return
        if robot.standing():
            logger.info("robot '%s' (%s) is hosted by its new goal", name, model)
        elif not await self._stand(context, name, model):
            return
        if not await self._standing(context, name):
            return
        ending, why = await self._watch(context, name, handover)
        await self._take_out(context, name, ending, why)

    async def _standing(self, context: attach.GoalContext, name: str) -> bool:
        """Tells the goal's holder the robot stands. False when it could not
        be told: a robot whose holder cannot follow its stay is taken out."""
        try:
            await context.publish_feedback(standing=True)
        except Exception as error:  # pylint: disable=W0718
            await self._take_out(
                context, name, Ending.FAILED, f"the robot's holder could not be told it stands: {error}"
            )
            return False
        return True

    async def _take_out(
        self, context: attach.GoalContext, name: str, ending: Ending, why: str
    ) -> None:
        """Ends a robot's stay the way it ended: a robot handed over stays
        standing for its new goal, one whose engine is stopping goes down
        with the stage, and any other is taken off the stage first."""
        if ending is Ending.HANDED_OVER:
            logger.info("robot '%s' is hosted by another goal of its copy", name)
            await context.complete(False, why)
            return
        if ending is Ending.STOPPED:
            self._robots.release(name)
        else:
            try:
                await asyncio.wrap_future(self._edits.submit(lambda: self.unstand(name)))
            except Exception as error:  # pylint: disable=W0718
                # The name stays taken, because the robot is still standing:
                # the stage kept it when taking it out failed.
                logger.error(
                    "robot '%s' could not be taken out: %s. It stands with no host, "
                    "and its name stays taken until this simulation restarts",
                    name,
                    error,
                )
        self._forget(name)
        if ending is Ending.LEFT:
            await context.complete_cancelled(True, why)
        else:
            await context.complete(False, why)

    async def _stand(self, context: attach.GoalContext, name: str, model: str) -> bool:
        """Puts the robot on the stage, answering the goal itself when the
        scene will not take it, or when the robot leaves while its stand
        still waits for the thread that steps the scene: that stand is
        withdrawn and the name given back at once. False when the robot
        never stood."""
        placement = self._placements.get(name)
        if placement is None:
            await context.complete(False, "the name was given back before the robot stood")
            return False
        edit = self._edits.submit(lambda: self.stand(name, model, placement))
        standing = asyncio.wrap_future(edit)
        leaving = asyncio.ensure_future(context.cancel_signal())
        try:
            await asyncio.wait({standing, leaving}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            leaving.cancel()
        if not standing.done() and edit.cancel():
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
        self._robots.stand(name, self._limbs, self._loop.time())
        logger.info("robot '%s' (%s) is in the scene", name, model)
        return True

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
                if self._lapsed(robot):
                    return Ending.LAPSED, (
                        f"none of this robot's limbs were paired for {self._lease_s:.1f}s, "
                        "the lease this scene gives"
                    )
        except asyncio.CancelledError:
            return Ending.STOPPED, "the engine stopped"
        finally:
            cancel.cancel()
            handed.cancel()

    def _lapsed(self, robot) -> bool:
        """Whether the robot's lease ran out: it holds no limb pair now, and
        held none for the lease. Its pairs are read here as well as on the
        watcher's tick, so a robot still holding a pair keeps its place
        however long the node loop went without renewing its lease."""
        now = self._loop.time()
        if robot.name in self._io.robots_with_any_limb():
            self._robots.note_paired({robot.name}, now)
            return False
        with robot.lock:
            last = robot.last_paired_s
        return now - last > self._lease_s

    def _forget(self, name: str) -> None:
        """Drops what this robot's joining held, leaving the scene alone."""
        self._handovers.pop(name, None)
        with self._admit_lock:
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
        stands in the scene and holds every one of its limb pairs, so a
        setpoint reaches it and its state comes back. A robot's limbs are
        this engine's and go ready together, so one answer covers them all."""
        caller = Caller(core_node=request.core_node, instance_id=request.instance_id)
        robot = self._robots.of_caller(caller)
        ready = (
            robot is not None
            and robot.standing()
            and robot.name in self._io.robots_with_every_limb()
        )
        return is_ready.Response(ready=ready)
