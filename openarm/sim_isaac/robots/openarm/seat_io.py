#!/usr/bin/env python3
"""The simulation_robot contract, on the node loop.

A robot attaches with the model it is and where it stands, and the goal runs
for as long as the robot is in the scene: its measured state goes back as
that goal's feedback, and the goal ends when the robot leaves. Setpoints
arrive on the command service, which is also the robot's heartbeat.

The scene belongs to the thread that steps it, so standing a robot and
taking one out are handed to that thread as edits and waited on here.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from peppygen.exposed_actions.robots import attach
from peppygen.exposed_services.robots import command

from edits import Edits
from seats import Caller, Limbs, Registry
from world import Placement, World

logger = logging.getLogger(__name__)

# How long a robot's stay waits between checks of its own lease, as a share
# of the lease: a robot whose commands stopped leaves within a quarter of a
# lease of the lease running out.
_LEASE_CHECK_SHARE = 0.25
# Pause after a runtime error before serving again, so a broken transport
# cannot hot-spin a loop or flood the log.
_RETRY_BACKOFF_S = 1.0
# The same, for the command service: every robot's heartbeat runs through
# that one loop, so it is back well inside the shortest usable lease.
_COMMAND_RETRY_BACKOFF_S = 0.05


class SeatIO:
    """Serves the seat contract: who may attach, what each robot's stay
    looks like, and where its setpoints and state go."""

    def __init__(
        self,
        node_runner,
        loop: asyncio.AbstractEventLoop,
        world: World,
        seats: Registry,
        edits: Edits,
        limbs: Limbs,
        lease_s: float,
    ) -> None:
        self._node_runner = node_runner
        self._loop = loop
        self._world = world
        self._seats = seats
        self._edits = edits
        self._limbs = limbs
        self._lease_s = lease_s
        # Set by the launch once the engine exists: a scene that changed has
        # to be resolved again before it is stepped, and an engine that reads
        # it through handles it cannot keep lets go of them first.
        self._rebind = lambda: None
        self._unbind = lambda: None
        self._contexts: dict[Caller, attach.GoalContext] = {}
        self._placements: dict[Caller, Placement] = {}
        # Guards the two above against the physics thread, which reads them to
        # send a robot's state back.
        self._feedback_lock = threading.Lock()
        self._in_flight: set[Caller] = set()
        self._tasks: list[asyncio.Task] = []
        self._stays: set[asyncio.Task] = set()
        self._stopping = threading.Event()

    def binds_with(self, rebind, unbind=None) -> None:
        """How the engine resolves the scene again once it changed, and how
        it lets go of the scene first. Standing a robot and taking one out
        both replace what the engine reads."""
        self._rebind = rebind
        self._unbind = unbind or (lambda: None)

    def stand(self, instance: str, model: str, placement: Placement) -> None:
        """Stands a robot and resolves the scene around it, on the thread
        that steps the scene. A robot the engine cannot resolve (a model
        whose joints are not the ones this engine drives) is taken back out,
        so one robot that cannot join never takes the scene down."""
        self._unbind()
        try:
            self._world.add(instance, model, placement)
            try:
                self._rebind()
            except Exception:
                self._world.remove(instance)
                raise
        except Exception:
            # Standing let go of the stage before it changed it. Whatever went
            # wrong, the stage is taken up again here, so a robot that cannot
            # join leaves the scene running rather than stopped.
            self._rebind()
            raise
        finally:
            self._seats.renew(self._loop.time())

    def unstand(self, instance: str) -> None:
        """Takes a robot out and resolves the scene around it."""
        self._unbind()
        try:
            self._world.remove(instance)
        finally:
            self._rebind()
            self._seats.renew(self._loop.time())

    async def start(self) -> None:
        """Exposes the contract and spawns its loops. Runs on the node loop
        before the sim thread starts, so a robot that attaches early waits
        while the scene compiles and then stands."""
        handle = await attach.ActionHandle.expose(self._node_runner)
        self._tasks = [
            asyncio.create_task(self._serve_attach(handle)),
            asyncio.create_task(self._serve_command()),
        ]
        logger.info(
            "seats open: robots attach with a model of %s, lease %.1fs",
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

    def publish_state(self, seat, timestamp_s: float, arms, grippers) -> None:
        """Sends one robot's measured state back on its own goal. Called from
        the physics thread, which never blocks on messaging: one feedback per
        seat is in flight at a time, and a state that arrives while one is
        outstanding is dropped."""
        with self._feedback_lock:
            context = self._contexts.get(seat.caller)
            if context is None or seat.caller in self._in_flight:
                return
            self._in_flight.add(seat.caller)
        feedback_arms = [
            attach.FeedbackArmsItem(positions=list(positions), velocities=list(velocities))
            for positions, velocities in arms
        ]
        feedback_grippers = [
            attach.FeedbackGrippersItem(
                opening=0.0 if reading is None else reading[0],
                effort=0.0 if reading is None else reading[1],
            )
            for reading in grippers
        ]
        caller = seat.caller

        def _publish() -> None:
            task = asyncio.ensure_future(
                context.publish_feedback(timestamp_s, feedback_arms, feedback_grippers)
            )
            task.add_done_callback(lambda done: self._feedback_done(caller, done))

        try:
            self._loop.call_soon_threadsafe(_publish)
        except RuntimeError:
            # The loop is closed: the node is shutting down and this state has
            # nowhere to go.
            with self._feedback_lock:
                self._in_flight.discard(caller)

    def _feedback_done(self, caller: Caller, task) -> None:
        with self._feedback_lock:
            self._in_flight.discard(caller)
        error = task.exception() if not task.cancelled() else None
        if error is not None:
            logger.warning("robot '%s': state feedback failed: %s", caller.instance, error)

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
        """Whether a robot may take a seat, and the limbs it will drive: the
        model must be one this engine carries, the placement must be free, and
        no robot may already hold the caller's seat. Admitting reserves it."""
        caller = Caller(core_node=request.core_node, instance_id=request.instance_id)
        try:
            self._world.catalogue().scene(request.data.model)
        except (ValueError, FileNotFoundError) as error:
            return attach.GoalDecision.reject(str(error))
        try:
            placement = self._placement(request.data.placement)
        except ValueError as error:
            return attach.GoalDecision.reject(str(error))
        try:
            seat = self._seats.reserve(caller, request.data.model, self._loop.time())
        except ValueError as error:
            return attach.GoalDecision.reject(str(error))
        self._placements[caller] = placement
        return attach.GoalDecision.accept(
            attach.GoalResponse(
                arm_names=list(self._limbs.arm_names),
                arm_joints=list(self._limbs.arm_joints),
                gripper_names=list(self._limbs.gripper_names),
            )
        )

    def _placement(self, asked) -> Placement:
        """Where the robot stands: what it asked for, or a spot of the
        engine's own. A spot another robot stands on is refused, because two
        robots in one place resolve their overlap by throwing each other."""
        if asked is None:
            return self._world.free_spot()
        placement = Placement.of(asked.position, asked.yaw)
        if self._world.occupied(placement):
            raise ValueError(
                f"a robot already stands at {list(placement.position)}; attach with no "
                "placement to take a free spot"
            )
        return placement

    async def _stay(self, context: attach.GoalContext) -> None:
        """One robot's stay in the scene: it joins, its state streams back as
        this goal's feedback, and it leaves when the goal is cancelled, when
        its commands stop for the lease, or when the engine takes it out."""
        request = context.request()
        caller = Caller(core_node=request.core_node, instance_id=request.instance_id)
        model = request.data.model
        placement = self._placements.get(caller)
        if self._seats.seat_of(caller) is None or placement is None:
            await context.complete(False, "the seat was given back before the robot stood")
            return
        with self._feedback_lock:
            self._contexts[caller] = context
        try:
            await asyncio.wrap_future(
                self._edits.submit(lambda: self.stand(caller.instance, model, placement))
            )
        except Exception as error:  # pylint: disable=W0718
            logger.warning("robot '%s' could not join: %s", caller.instance, error)
            self._release(caller)
            await context.complete(False, f"the scene could not stand the robot: {error}")
            return

        self._seats.stand(caller, self._limbs, self._loop.time())
        logger.info("robot '%s' (%s) is in the scene", caller.instance, model)
        cancelled, why = await self._watch(context, caller)
        self._release(caller)
        try:
            await asyncio.wrap_future(
                self._edits.submit(lambda: self.unstand(caller.instance))
            )
        except Exception as error:  # pylint: disable=W0718
            logger.warning("robot '%s' could not be taken out: %s", caller.instance, error)
        if cancelled:
            await context.complete_cancelled(True, why)
        else:
            await context.complete(False, why)

    async def _watch(self, context: attach.GoalContext, caller: Caller) -> tuple[bool, str]:
        """Waits for the robot's stay to end, and says how it ended."""
        cancel = asyncio.create_task(context.cancel_signal())
        period = max(self._lease_s * _LEASE_CHECK_SHARE, 0.05)
        try:
            while True:
                done, _ = await asyncio.wait({cancel}, timeout=period)
                if done:
                    return True, "the robot left the scene"
                if self._stopping.is_set():
                    return False, "the engine stopped"
                seat = self._seats.seat_of(caller)
                if seat is None:
                    return False, "the seat was given back"
                if self._lapsed(seat):
                    return False, (
                        f"no command for {self._lease_s:.1f}s, the lease this scene gives"
                    )
        except asyncio.CancelledError:
            return False, "the engine stopped"
        finally:
            cancel.cancel()

    def _lapsed(self, seat) -> bool:
        with seat.lock:
            last = seat.last_command_s
        return self._loop.time() - last > self._lease_s

    def _release(self, caller: Caller) -> None:
        with self._feedback_lock:
            self._contexts.pop(caller, None)
            self._in_flight.discard(caller)
        self._placements.pop(caller, None)
        self._seats.release(caller)

    async def _serve_command(self) -> None:
        while True:
            try:
                await command.handle_next_request(self._node_runner, self._command)
            except asyncio.CancelledError:
                raise
            except Exception:  # pylint: disable=W0718
                logger.exception("command service failed")
                await asyncio.sleep(_COMMAND_RETRY_BACKOFF_S)

    def _command(self, request: command.Request) -> command.Response:
        caller = Caller(core_node=request.core_node, instance_id=request.instance_id)
        answer = self._seats.command(
            caller, request.data.arms, request.data.grippers, self._loop.time()
        )
        return command.Response(
            success=answer.taken, joining=answer.joining, message=answer.message
        )
