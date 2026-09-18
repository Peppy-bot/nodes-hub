#!/usr/bin/env python3
"""The simulation_robot contract, on the node loop.

This engine stands one robot. It joins by attaching with the copy it runs as
and the model it is, the engine loads that model's scene on the thread that
steps it, and the goal runs for as long as the robot is in the scene. Its
limbs reach it through its own pairs, so nothing here carries motion: what
this serves is who may join, how long they stay, and whether the robot that
joined is ready to be driven. A second robot is refused while one stands,
and the robot stands where its scene puts it.
"""

from __future__ import annotations

import asyncio
import enum
import logging

from peppygen.exposed_actions.robots import attach
from peppygen.exposed_services.robots import is_ready

from robots import Caller, Limbs, Registry
from scenes import Catalogue
from stands import Stands

# How long a robot's stay waits between checks of its own lease, as a share
# of the lease: a robot whose limbs are gone leaves within a quarter of a
# lease of the lease running out. The same tick records whether the robot
# is holding a pair, which is what renews its lease.
_LEASE_CHECK_SHARE = 0.25
# The shortest lease check, for a lease too short to share.
_LEASE_CHECK_FLOOR_S = 0.05
# Pause after a runtime error before serving again, so a broken transport
# cannot hot-spin a loop or flood the log.
_RETRY_BACKOFF_S = 1.0
# The same, for the readiness service, which is back well inside a
# readiness poll.
_READY_RETRY_BACKOFF_S = 0.05
# The model whose links config/cameras.json5 mounts the rig on.
RENDERED_MODEL = "openarm_v2"

logger = logging.getLogger(__name__)


class Ending(enum.Enum):
    """How a robot's stay ended."""

    #: Its caller cancelled the goal.
    LEFT = "left"
    #: Its limbs held no pair for the lease.
    LAPSED = "lapsed"
    #: Another goal of the same caller hosts the robot now.
    HANDED_OVER = "handed over"
    #: The engine is going down and takes the robot with it.
    STOPPED = "stopped"
    #: Its holder could not be told it stands, so it is taken out.
    FAILED = "failed"


class RobotsIO:
    """Serves how a robot joins the scene and whether it is ready: who may
    attach, how long the robot stays, and whether it holds every limb pair."""

    def __init__(
        self,
        node_runner,
        loop: asyncio.AbstractEventLoop,
        catalogue: Catalogue,
        robots: Registry,
        stands: Stands,
        io,
        limbs: Limbs,
        lease_s: float,
        renders: bool,
    ) -> None:
        self._node_runner = node_runner
        self._loop = loop
        self._catalogue = catalogue
        self._robots = robots
        # The thread that steps the scene, which loads and lets go of it.
        self._stands = stands
        # The limb pairs, which are what a robot is driven through and what
        # keeps it in the scene.
        self._io = io
        self._limbs = limbs
        if lease_s <= 0.0:
            raise ValueError(f"robot_lease_ms must be positive, got {lease_s * 1000:g}")
        self._lease_s = lease_s
        # Whether this engine renders the camera rig, which mounts on one
        # model's links.
        self._renders = renders
        self._tasks: list[asyncio.Task] = []
        self._stays: set[asyncio.Task] = set()
        # The signal that ends the robot's stay without taking it out of the
        # scene, which is how a re-registering copy takes its own robot
        # over. Node loop only.
        self._handovers: dict[str, asyncio.Event] = {}
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        """Exposes the contract and spawns its loops. Runs on the node loop
        before the sim thread starts, which waits for the first robot."""
        handle = await attach.ActionHandle.expose(self._node_runner)
        self._tasks = [
            asyncio.create_task(self._serve_attach(handle)),
            asyncio.create_task(self._serve_ready()),
            asyncio.create_task(self._watch_pairs()),
        ]
        logger.info(
            "the scene is open: one robot attaches with a model of %s, lease %.1fs",
            ", ".join(self._catalogue.models()),
            self._lease_s,
        )

    async def stop(self) -> None:
        """Ends the stay and stops serving. A robot whose engine is going
        down is told so on its own goal."""
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        for task in list(self._stays):
            task.cancel()
        await asyncio.gather(*self._tasks, *self._stays, return_exceptions=True)

    async def _watch_pairs(self) -> None:
        """Records whether the robot is holding a limb pair, on the same tick
        its stay checks its lease. Its pairs dissolve when its nodes stop, so
        this is how a robot that is gone stops renewing."""
        period = self._lease_check_period()
        while True:
            try:
                self._robots.note_paired(self._io.robots_with_any_limb(), self._loop.time())
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

    def _lease_check_period(self) -> float:
        """How often a stay checks its lease: a share of the lease, floored."""
        return max(self._lease_s * _LEASE_CHECK_SHARE, _LEASE_CHECK_FLOOR_S)

    def _admit(self, request: attach.GoalRequest) -> attach.GoalDecision:
        """Whether the robot may join, and the limbs it will drive: the model
        must be one this engine carries and, while the rig is rendered, the
        one the rig mounts on; the spot is the scene's own; and the only
        robot standing may be this caller's own. Admitting reserves the
        name, or hands this caller's standing robot over to the goal being
        admitted."""
        caller = Caller(core_node=request.core_node, instance_id=request.instance_id)
        try:
            self._catalogue.scene(request.data.model)
        except (ValueError, FileNotFoundError) as error:
            return attach.GoalDecision.reject(str(error))
        if request.data.placement is not None:
            return attach.GoalDecision.reject(
                "this engine stands its robot where the scene puts it: join with "
                "placement { auto: true }"
            )
        if self._renders and request.data.model != RENDERED_MODEL:
            return attach.GoalDecision.reject(
                f"this engine renders the camera rig of {RENDERED_MODEL}, so with "
                f"cameras_enabled it stands no other model"
            )
        limbs = attach.GoalResponse(
            arm_names=list(self._limbs.arm_names),
            arm_joints=list(self._limbs.arm_joints),
            gripper_names=list(self._limbs.gripper_names),
        )
        held = self._robots.of_name(request.data.robot)
        standing = self._robots.robots()
        if standing and held is None:
            other = standing[0]
            return attach.GoalDecision.reject(
                f"this engine stands one robot, and '{other.name}' of {other.caller} stands "
                "already: remove it first, or run a second simulation"
            )
        try:
            adopted = self._robots.admit(
                request.data.robot, request.data.model, caller, self._loop.time()
            )
        except ValueError as error:
            return attach.GoalDecision.reject(str(error))
        if adopted:
            # The robot stands in the scene it stands in; only its host changes.
            self._hand_over(request.data.robot)
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

    async def _stay(self, context: attach.GoalContext) -> None:
        """The robot's stay in the scene: its scene is loaded, it is driven
        through its limb pairs, and it leaves when the goal is cancelled,
        when its pairs are gone for the lease, or when the engine stops.

        A goal admitted for the robot its caller already stands takes that
        stay over instead: the scene keeps running and the goal that hosted
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
        else:
            try:
                await asyncio.wrap_future(
                    self._stands.stand(
                        self._catalogue.scene(model),
                        name,
                        head_camera_pack=self._catalogue.head_camera_pack(model),
                    )
                )
            except Exception as error:  # pylint: disable=W0718
                logger.warning("robot '%s' could not join: %s", name, error)
                self._robots.release(name)
                self._handovers.pop(name, None)
                await context.complete(False, f"the scene could not stand the robot: {error}")
                return
            self._robots.stand(name, self._limbs, self._loop.time())
            logger.info("robot '%s' (%s) is in the scene", name, model)
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
        with the scene, and any other is taken out of the scene first."""
        if ending is Ending.HANDED_OVER:
            logger.info("robot '%s' is hosted by another goal of its copy", name)
            await context.complete(False, why)
            return
        if ending is not Ending.STOPPED:
            try:
                await asyncio.wrap_future(self._stands.unstand())
            except Exception as error:  # pylint: disable=W0718
                logger.error("robot '%s' could not be taken out: %s", name, error)
        self._io.forget(name)
        self._handovers.pop(name, None)
        # The name goes back once nothing stands under it, so a robot that
        # comes straight back finds it free.
        self._robots.release(name)
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
        setpoint reaches it and its state comes back."""
        caller = Caller(core_node=request.core_node, instance_id=request.instance_id)
        robot = self._robots.of_caller(caller)
        ready = (
            robot is not None
            and robot.standing()
            and robot.name in self._io.robots_with_every_limb()
        )
        return is_ready.Response(ready=ready)
