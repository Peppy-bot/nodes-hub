import asyncio
import contextlib
import heapq
import itertools
import math
import time
from types import MappingProxyType, SimpleNamespace

import numpy as np
import peppygen.clock
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from peppygen.consumed_actions.fleet import changes, lease
from peppygen.consumed_topics.color_cameras import video_stream as color_video_stream
from peppygen.fixtures import harness
from peppygen.parameters import Parameters

from xr_commander import config
from xr_commander.bus import REAL_CLOCK, Answered, Clock, FlashNote, Outcome
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample, XrFrame
from xr_commander.drive import Driver, HandState
from xr_commander.fleet import Fleet, FleetView, Following, Holder, Timing
from xr_commander import frames
from xr_commander.frames import Pose
from xr_commander.panel import HandSource
from xr_commander.publish import MeasuredPoses
from xr_commander.robots import PairedRobots

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation of `angle` about `axis`; degenerate axis or zero angle is identity."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < frames._EPS or angle == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    half = 0.5 * angle
    return np.concatenate([(axis / norm) * math.sin(half), [math.cos(half)]])


def frame_message(width=1, height=1, encoding="bgr8", data=None):
    if data is None:
        data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
    return color_video_stream.Message(
        header=color_video_stream.MessageHeader(timestamp=0.0, frame_id=0),
        encoding=encoding,
        width=width,
        height=height,
        frame=data,
    )


X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
POSE = Pose(np.zeros(3), IDENTITY)


def squeezing_hand():
    return HandSample(pose=POSE, squeezing=True, trigger=0.0)


class ScriptedHeadset:
    """A headset whose hands the test sets, its frames stamped on `clock`
    `age_s` in the past: the FrameSource seam is a device surface, which a
    test drives itself."""

    def __init__(
        self, clock: Clock = REAL_CLOCK, *, hands=None, age_s: float = 0.0
    ) -> None:
        self._clock = clock
        self._age_s = age_s
        self.hands = hands or {}
        # The trigger of every hand that `press` sets.
        self.trigger = 0.0

    def press(self, *, pose=POSE, **hands):
        """Each hand as (primary button, secondary button, squeeze), at
        `pose`."""
        self.hands = {
            side: HandSample(
                pose=pose,
                squeezing=squeeze,
                trigger=self.trigger,
                primary_button=primary,
                secondary_button=secondary,
            )
            for side, (primary, secondary, squeeze) in hands.items()
        }

    def latest(self):
        return XrFrame(
            received_monotonic_s=self._clock.monotonic() - self._age_s,
            hands=self.hands,
        )


def left_buttons(*, x=False, y=False) -> dict:
    """`ScriptedHeadset.press`'s left hand with X and Y as given, no squeeze."""
    return {"left": (x, y, False)}


# A stand-in for the chosen robot's pair on a slot, for tests that read the
# panel and the streams without a wire.
PEER = "peer"


def source(
    handedness, *, arm_bound=True, gripper_bound=None, fresh=True, engaged=False
):
    measured = MeasuredPoses()
    if fresh:
        measured.set(PEER, POSE)
    clutch = HandClutch(1.0)
    if engaged:
        clutch.step(squeezing=True, hand=POSE, measured_ee=POSE)
    gripper = arm_bound if gripper_bound is None else gripper_bound
    return HandSource(
        handedness=handedness,
        clutch=clutch,
        state=HandState(FlashNote()),
        measured=measured,
        arm_peer=lambda: PEER if arm_bound else None,
        gripper_peer=lambda: PEER if gripper else None,
    )


def default_parameters(**overrides) -> SimpleNamespace:
    """The full raw launch-parameter set, valid by default."""
    values = {
        "command_rate_hz": 1000,
        "https_port": 4443,
        "https_host": "0.0.0.0",
        "motion_scale": 1.0,
        "gripper_open_fraction": 1.0,
        "posture_move_duration_s": 2.0,
        "stale_timeout_s": 10.0,
        "view_max_width": 640,
        "status_panel_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def default_settings(**overrides) -> config.Settings:
    """The settings `default_parameters(**overrides)` parse into."""
    return config.from_parameters(default_parameters(**overrides))


async def idle_setup(params, node_runner):
    """A setup that starts nothing: the real `setup` raises TLS, an HTTPS
    server, and a WebXR session, none of which belongs in a test. Harness
    tests boot the node's runtime (ephemeral router, generated mocks, seeded
    slots) under this no-op and drive the production functions directly
    against the real runner and real generated modules. The manifest's
    `page` endpoint is announced with the launch address so the runtime's
    seal, which the harness runs when setup returns, finds it."""
    node_runner.announce_endpoint("page", "https", params.https_host, params.https_port)
    return []


def asgi_request(
    app, method: str, path: str, *, body=b"", headers: dict | None = None
) -> SimpleNamespace:
    """One request through an ASGI app, without a web client library.

    The node ships no HTTP client, and these routes are worth testing where
    they are mounted: their ordering against the frontend mount is the hazard.
    `body` takes a list of chunks to arrive as a streamed body rather than one
    delivery.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (name.lower().encode(), value.encode())
            for name, value in (headers or {}).items()
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 4443),
    }
    sent = []
    pending = list(body) if isinstance(body, list) else [body]

    async def receive():
        if not pending:
            return {"type": "http.disconnect"}
        chunk = pending.pop(0)
        return {"type": "http.request", "body": chunk, "more_body": bool(pending)}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    payload = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return SimpleNamespace(
        status=start["status"],
        headers={key.decode(): value.decode() for key, value in start["headers"]},
        body=payload.decode("utf-8", "replace"),
    )


def boot(**kwargs):
    """The generated harness under `idle_setup`: awaitable or an async
    context manager. Keyword arguments are `harness.start`'s: per-slot mock
    counts (`<link_id>_instances`) and the usual overrides."""
    kwargs.setdefault("parameters", Parameters(**vars(default_parameters())))
    return harness.start(idle_setup, **kwargs)


@contextlib.asynccontextmanager
async def boot_clocked(**kwargs):
    """[`boot`] with the node's clock initialized: for tests that await
    time-based node behavior. Shuts down on exit, like `boot`'s context
    form."""
    h = await boot(**kwargs)
    try:
        await peppygen.clock.init(h.node_runner)
        yield h
    finally:
        await h.shutdown()


async def eventually(predicate, message="condition"):
    """Poll until `predicate()` is truthy; AssertionError past 10s.

    The bounded-wait primitive for harness tests: wire delivery is ordered
    but not instant, so state assertions converge instead of sleeping blind.
    """
    timeout = 10.0
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"{message}: not met within {timeout}s")
        await asyncio.sleep(0.005)


async def press_until_goal(session, mock_action, release, press):
    """Release-then-press cycles until the node fires: buttons re-arm only
    once the previous goal's watcher finished, so the retry absorbs that tail
    without sleeping blind. `release`/`press` are the two `session.press`
    kwargs dicts."""
    deadline = time.monotonic() + 10.0
    while True:
        session.press(**release)
        await asyncio.sleep(0.03)
        session.press(**press)
        try:
            return await mock_action.next_goal(0.5)
        except TimeoutError:
            if time.monotonic() >= deadline:
                raise


async def tap_x(session, *, settle=0.05):
    """One observed X press edge: release, let the tracker see it, press.
    The sleep spans the tracker's frame cadence, not a node wait."""
    session.press(**left_buttons(x=False))
    await asyncio.sleep(settle)
    session.press(**left_buttons(x=True))


# The reason of a goal that its caller cancelled.
CANCELLED_BY_CALLER = "cancelled by its caller"


async def stop_and_save(session, active, cancelled_result, *, settle=0.05):
    """The second X press of an episode: stop-and-save, not a fresh goal. It
    cancels the active goal, and the recorder ends the goal cancelled, for
    the reason of the caller's cancel, with its result."""
    session.press(**left_buttons(x=False))
    await asyncio.sleep(settle)
    session.press(**left_buttons(x=True))
    await asyncio.wait_for(active.cancel_signal(), 10.0)
    await active.complete_cancelled(CANCELLED_BY_CALLER, cancelled_result)


class RecordingSink:
    """A FrameSink that keeps every frame it was handed."""

    def __init__(self):
        self.frames = []

    def put_frame(self, frame):
        self.frames.append(frame)


class FakeSubscription:
    """Yields queued items, then pends forever like the generated one.

    For the pure bus-plumbing tests (`messages`/`next_message`), which
    exercise receive-loop edge cases no real subscription can be scripted
    into (mid-stream exceptions, self-cancelling receives), and under
    `StubTopic`."""

    def __init__(self, items):
        self._items = list(items)

    async def next(self):
        if self._items:
            return self._items.pop(0)
        await asyncio.Event().wait()


class StubTopic(FakeSubscription):
    """A topic module whose subscription yields queued (producer, message)
    items, for a drain fed what the harness cannot publish: a message the
    generated modules cannot build, or producers on more than one core node
    (every harness mock shares the harness's)."""

    async def subscribe(self, _runner):
        return self


@contextlib.asynccontextmanager
async def running(*loops):
    """Each of `loops`, a function of a cancellation token, as a task for
    the body's duration: yields the tasks, and cancels and awaits them on
    exit."""
    token = FakeToken()
    tasks = [asyncio.create_task(loop(token)) for loop in loops]
    try:
        yield tasks
    finally:
        token.cancel()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task


@contextlib.asynccontextmanager
async def running_drain(make_drain):
    """One drain task for the body's duration: a hand-fired token cancels it
    on exit (never the runner's real token, which would converge the whole
    node), and the task is awaited bounded so a hung drain fails the test
    instead of leaking."""
    token = FakeToken()
    drain = asyncio.create_task(make_drain(token))
    try:
        yield drain
    finally:
        token.cancel()
        await asyncio.wait_for(drain, 5.0)


class FakeToken:
    """A cancellation token the test fires by hand.

    Kept deliberately alongside the harness: the production tasks accept any
    token with the CancellationToken shape, and a test must be able to end
    one drain without cancelling the runner's real token (which would
    converge the whole node mid-test). The pure loop tests have no runner at
    all."""

    def __init__(self):
        self._event = asyncio.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self):
        return self._event.is_set()

    async def cancelled(self):
        await self._event.wait()


def robots_of(h):
    """The paired robots over the harness's real runner and slot modules."""
    from peppygen.paired_topics.left_arm import pose_setpoints as left_arm
    from peppygen.paired_topics.left_gripper import gripper_setpoints as left_gripper
    from peppygen.paired_topics.right_arm import pose_setpoints as right_arm
    from peppygen.paired_topics.right_gripper import gripper_setpoints as right_gripper

    return PairedRobots(
        h.node_runner,
        arms={"left": left_arm, "right": right_arm},
        grippers={"left": left_gripper, "right": right_gripper},
    )


def peer_of(h, module):
    """The one mock peer the harness paired into `module`'s slot."""
    (member,) = module.peers(h.node_runner)
    return member.info


FORM = {"content-type": "application/x-www-form-urlencoded"}


def frontend_app(tmp_path) -> FastAPI:
    """A stand-in for teleop_xr's app: a frontend mounted at '/', which
    matches every path and shadows anything routed after it."""
    app = FastAPI()
    app.mount("/", StaticFiles(directory=tmp_path), name="frontend")
    return app


class FakeRobots:
    """The paired robots as a pure test sets them, and the choices a page
    makes on them."""

    def __init__(self, names=(), chosen=None):
        self._names = list(names)
        self._chosen = chosen
        self.choices = []

    def names(self):
        return list(self._names)

    def chosen(self):
        return self._chosen

    def standing(self):
        return self.names(), self._chosen

    def choose(self, name):
        self.choices.append(name)
        if name not in self._names:
            return False
        self._chosen = name
        return True


class ManualClock:
    """A clock that moves only when a test advances it: each sleeper wakes,
    in deadline order, once the time reaches its deadline."""

    def __init__(self, start_s: float = 1000.0) -> None:
        self._now = start_s
        self._sleepers: list[tuple[float, int, asyncio.Future]] = []
        self._order = itertools.count()
        # The tasks asleep on this clock now.
        self._asleep: set[asyncio.Task] = set()

    def monotonic(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        wake = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self._now + seconds, next(self._order), wake))
        task = asyncio.current_task()
        self._asleep.add(task)
        try:
            await wake
        finally:
            self._asleep.discard(task)

    async def advance(self, seconds: float, *, parked=()) -> None:
        """Wait until each task of `parked` sleeps on this clock or ended,
        move the time on by `seconds`, wake each sleeper due by then, and
        let the woken tasks run up to their next wait. With every loop of a
        test parked, each advance by one period is one step of each loop."""
        await settle()
        await self._parked(lambda: parked)
        self._now += seconds
        while self._sleepers and self._sleepers[0][0] <= self._now:
            _, _, wake = heapq.heappop(self._sleepers)
            if not wake.done():
                wake.set_result(None)
        await settle()

    async def idle(self) -> None:
        """Wait until every other task of the loop sleeps on this clock or
        ended: for a test with no wire, the tasks that the code under test
        spawned have then done all they can before the time moves."""
        me = asyncio.current_task()
        await self._parked(lambda: asyncio.all_tasks() - {me})

    async def _parked(self, tasks) -> None:
        deadline = time.monotonic() + PARK_TIMEOUT_S
        while not all(task.done() or task in self._asleep for task in tasks()):
            assert time.monotonic() < deadline, "a task never came back to the clock"
            await asyncio.sleep(PARK_POLL_S)


# How long a loop may take to come back to the clock, on the wire, and how
# often the test looks.
PARK_TIMEOUT_S = 10.0
PARK_POLL_S = 0.001
# Each yield runs every task ready at that moment once; this many let a
# woken task reach its next wait through a few awaits.
SETTLE_YIELDS = 20


async def settle() -> None:
    """Let every ready task run up to its next wait through a few awaits."""
    for _ in range(SETTLE_YIELDS):
        await asyncio.sleep(0)


# The headset as the fleet listing names it, in tests that script the fleet,
# and a director that holds leases beside it.
HEADSET = Holder("headset_inst", "fixture-core")
DIRECTOR = Holder("director_inst", "fixture-core")
# The halt of the tests, by `peppy-cli`, as a refusal and the card give it.
HALTED = "the stack is halted by peppy-cli until peppy fleet clear-halt"
# The bound of a call that the test answers, and the retry of the headset.
CALL_S = 10.0
RETRY_S = 1.0


# The revision of a listing that a test gives no revision.
FIRST_REVISION = 1


def fleet_view(
    robots=(), *, leases=None, halted_by=None, revision=FIRST_REVISION
) -> FleetView:
    return FleetView(
        revision=revision,
        robots=frozenset(robots),
        leases=MappingProxyType(dict(leases or {})),
        halted_by=halted_by,
    )


def following_fleet(*robots, **view) -> Fleet:
    """A fleet whose goals run and whose listing holds `robots`."""
    fleet = Fleet(HEADSET)
    fleet.view = fleet_view(robots, **view)
    fleet.status = Following()
    return fleet


AGREED = Answered(None)


class FakeCalls:
    """The daemon's answers to a lease holder, scripted per test: each call
    goes to `log` in order, yields once as a call on the wire does, and
    gets the outcome set for its member. A yes to a move check reads the
    revision of the first listing of `fleet_view`."""

    def __init__(self, log: list) -> None:
        self._log = log
        self.take_outcome: Outcome = AGREED
        self.check_outcome: Outcome = Answered(FIRST_REVISION)
        self.release_outcome: Outcome = AGREED

    async def take(self, robot):
        self._log.append(("take", robot))
        await asyncio.sleep(0)
        return self.take_outcome

    async def release(self, robot):
        self._log.append(("release", robot))
        await asyncio.sleep(0)
        return self.release_outcome

    async def check_move(self, *, robot=None, instance=None):
        self._log.append(("check_move", robot or instance))
        await asyncio.sleep(0)
        return self.check_outcome


class FakeStops:
    """A backbone's answers to `stop`, each call in `log`."""

    def __init__(self, log: list) -> None:
        self._log = log
        self.refusal: str | None = None

    async def stop(self, robot):
        self._log.append(("stop", robot))
        await asyncio.sleep(0)
        return self.refusal


def scripted_driver(*robots, clock=REAL_CLOCK, log=None) -> Driver:
    """A driver whose fleet lists `robots` free, whose daemon agrees to
    every call and whose backbones stop on demand; `log` records the
    calls."""
    log = log if log is not None else []
    return Driver(
        following_fleet(*robots),
        FakeCalls(log),
        FakeStops(log),
        timing=Timing(),
        on_change=lambda: None,
        clock=clock,
    )


# The fleet listing and the lease events as the daemon sends them.


def listing(robots=(), leases=(), halted_by=None):
    """A fleet listing as the daemon sends it in a `changes` feedback."""
    return changes.FeedbackMessageListing(
        version="unknown",
        core_node="fixture-core",
        launcher="test",
        halted=halted_by is not None,
        halted_by=halted_by or "",
        robots=[
            changes.FeedbackMessageListingRobotsItem(
                name=name,
                option="openarm_sim",
                core_node="fixture-core",
                kind="simulated",
                clock="simulation",
                software="running",
                robot_commander="",
                instances=[],
                contracts=[],
            )
            for name in robots
        ],
        leases=[
            changes.FeedbackMessageListingLeasesItem(
                robot=robot,
                holder=holder.instance,
                holder_core_node=holder.core_node,
                name=name,
                open=is_open,
                giver="",
                giver_core_node="",
                receiver="",
                receiver_core_node="",
            )
            for robot, holder, name, is_open in leases
        ],
        options=[],
        documents=[],
        nodes=[],
    )


def change(kind, robot="", revision=1, **listed):
    return changes.FeedbackMessage(
        change=kind, robot=robot, revision=revision, listing=listing(**listed)
    )


def lease_event(event, robot="alpha", cause=""):
    return lease.FeedbackMessage(
        event=event, robot=robot, giver="", giver_core_node="", cause=cause
    )


async def accept_both(mock):
    """The two goals the headset starts, accepted by the mock daemon: the
    changes goal, then the lease goal."""
    follow = await (await mock.changes.next_goal(CALL_S)).accept()
    hold = await (await mock.lease.next_goal(CALL_S)).accept()
    return follow, hold
