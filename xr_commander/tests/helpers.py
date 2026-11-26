import asyncio
import contextlib
import heapq
import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import numpy as np
import peppygen.clock
from control_core_py.geometry import Pose, Quaternion, Stamped
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from peppygen.consumed_actions.fleet import changes, lease
from peppygen.consumed_services.fleet import check_move, release, take
from peppygen.consumed_services.robot_stop import stop
from peppygen.consumed_topics.color_cameras import video_stream as color_video_stream
from peppygen.paired_topics.base import drive_states
from peppygen.fixtures import harness
from peppygen.parameters import Parameters

from xr_commander import config
from xr_commander.bus import REAL_CLOCK, Answered, Clock, Outcome
from xr_commander.clutch import HandClutch
from xr_commander.devices import HandSample, XrFrame
from xr_commander.drive import (
    Driver,
    Goals,
    GripStatus,
    LeaseCalls,
    RobotStops,
    Stops,
)
from xr_commander.fleet import (
    LEASE_NAME,
    FleetCalls,
    FleetView,
    Following,
    Holder,
    Timing,
)
from xr_commander.grip_world import Idle, PostureTarget
from xr_commander.panel import HandSource
from xr_commander.publish import MeasuredPoses
from xr_commander.robots import PairedRobots

IDENTITY = (0.0, 0.0, 0.0, 1.0)
X = (1.0, 0.0, 0.0)
Y = (0.0, 1.0, 0.0)
Z = (0.0, 0.0, 1.0)


def quat_from_axis_angle(axis, angle: float) -> Quaternion:
    """The turn of `angle` (rad) about the unit vector `axis`, [x, y, z, w]."""
    s = math.sin(0.5 * angle)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle))


def pose_at(position, orientation=IDENTITY) -> Pose:
    """The pose at `position` (m) turned by `orientation`, [x, y, z, w]."""
    return Pose.parse(position, orientation)


# The camera frame a picture of the tests names.
CAMERA_FRAME = "camera_color_optical_frame"


def frame_message(width=1, height=1, encoding="bgr8", data=None):
    if data is None:
        data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
    return color_video_stream.Message(
        header=color_video_stream.MessageHeader(
            timestamp=0.0, frame=CAMERA_FRAME, capture_index=0
        ),
        encoding=encoding,
        width=width,
        height=height,
        data=data,
    )


POSE = pose_at((0.0, 0.0, 0.0))
# The follower's robot frame, an arm's base_link, as its pose_states name it,
# and the time of its reports.
ROBOT_FRAME = "base_link"
REPORTED_AT_S = 1.0
# Another frame a follower can report its end-effector in: the robot frame of
# a mobile base.
OTHER_FRAME = "base_footprint"


def reported(pose: Pose, frame: str = ROBOT_FRAME) -> Stamped[Pose]:
    """`pose` as the follower reports it on pose_states."""
    return Stamped(frame, REPORTED_AT_S, pose)


def posture_done(module):
    """A posture result as a robot sends it. The commander reads only how the
    goal ended, thus the arm pose in it is a fixed value."""
    return module.ResultResponseData(
        frame=ROBOT_FRAME,
        timestamp=REPORTED_AT_S,
        arm_names=["right_arm"],
        grasp_points=[
            module.ResultResponseDataGraspPointsItem(
                position=[0.3, -0.2, 0.3], orientation=[0.0, 0.0, 0.0, 1.0]
            )
        ],
    )


def controller(
    handedness,
    *,
    position=(0.0, 0.0, 0.0),
    squeezing=False,
    trigger=0.0,
    primary=False,
    secondary=False,
    axes=(0.0, 0.0, 0.0, 0.0),
):
    """A controller as the page sends it in an `xr_state` frame."""
    buttons = [
        {"pressed": trigger >= 1.0, "touched": True, "value": trigger},
        {
            "pressed": squeezing,
            "touched": True,
            "value": 1.0 if squeezing else 0.0,
        },
    ]
    if primary or secondary:
        buttons += [
            {"pressed": False, "touched": False, "value": 0.0},
            {"pressed": False, "touched": False, "value": 0.0},
            {"pressed": primary, "touched": primary, "value": 0.0},
        ]
    if secondary:
        buttons.append({"pressed": True, "touched": True, "value": 0.0})
    return {
        "role": "controller",
        "handedness": handedness,
        "gripPose": {
            "position": dict(zip("xyz", position, strict=True)),
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "gamepad": {
            "buttons": buttons,
            "axes": list(axes),
        },
    }


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
        # How old each frame is when the stream reads it (s).
        self.age_s = age_s
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
            received_monotonic_s=self._clock.monotonic() - self.age_s,
            hands=self.hands,
        )


def left_buttons(*, x=False, y=False) -> dict:
    """`ScriptedHeadset.press`'s left hand with X and Y as given, no squeeze."""
    return {"left": (x, y, False)}


# A stand-in for the chosen robot's pair on a slot, for tests that read the
# panel and the streams without a wire.
PEER = "peer"


@dataclass(frozen=True)
class Peer:
    """A pair's `PeerInfo` in a test: hashable and equal by name, as two
    infos of one restarted pair are."""

    name: str


class GatedPublisher:
    """A publisher whose sends wait while the test holds `gate` shut; each
    send first yields, as a send on the wire does."""

    def __init__(self) -> None:
        self.sent: list = []
        self.gate = asyncio.Event()
        self.gate.set()

    async def publish_to(self, peer, payload) -> None:
        await asyncio.sleep(0)
        self.sent.append((peer, payload))
        await self.gate.wait()


class NoWireSlot:
    """A pairing slot of a test with no wire: its pairs in delivered order,
    each with its robot, which `set_pairs` changes, a message of the fields
    it is built from, and one gated publisher."""

    def __init__(self, *pairs: tuple[str, Peer]) -> None:
        self.set_pairs(*pairs)
        self.publisher = GatedPublisher()

    def set_pairs(self, *pairs: tuple[str, Peer]) -> None:
        """The slot holds `pairs` from its next read on, in delivered
        order."""
        self._members = tuple(
            SimpleNamespace(copy=robot, info=info) for robot, info in pairs
        )

    def peers(self, _runner):
        return list(self._members)

    def build_message(self, *fields):
        return fields

    async def declare_publisher(self, _runner):
        return self.publisher


def source(
    handedness, *, arm_bound=True, gripper_bound=None, fresh=True, engaged=False
):
    measured = MeasuredPoses()
    if fresh:
        measured.set(PEER, reported(POSE))
    clutch = HandClutch(1.0)
    if engaged:
        clutch.step(squeezing=True, hand=POSE, measured_ee=reported(POSE))
    gripper = arm_bound if gripper_bound is None else gripper_bound
    return HandSource(
        handedness=handedness,
        clutch=clutch,
        grip=lambda: GripStatus(Idle(), frozenset(), ""),
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
        # The longest the node accepts, for the tests on the real clock.
        "stale_timeout_s": 0.5,
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
    delivery. The client stays connected until the app sends the last body
    message of its response, as behind a server: a receive past the request
    body waits for that message, then reports the disconnect. A response
    that watches for the disconnect, as a file response does, sends its
    whole body.
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

    async def exchange():
        answered = asyncio.Event()

        async def receive():
            if pending:
                chunk = pending.pop(0)
                return {"type": "http.request", "body": chunk, "more_body": bool(pending)}
            await answered.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                answered.set()

        await app(scope, receive, send)

    asyncio.run(exchange())
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
    from peppygen.paired_topics.base import drive_setpoints as base
    from peppygen.paired_topics.left_arm import pose_setpoints as left_arm
    from peppygen.paired_topics.left_gripper import gripper_setpoints as left_gripper
    from peppygen.paired_topics.right_arm import pose_setpoints as right_arm
    from peppygen.paired_topics.right_gripper import gripper_setpoints as right_gripper

    return PairedRobots(
        h.node_runner,
        arms={"left": left_arm, "right": right_arm},
        grippers={"left": left_gripper, "right": right_gripper},
        base=base,
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


@contextlib.asynccontextmanager
async def parked(clock: ManualClock, *streams, period: float):
    """`streams` running on `clock`, each ticking every `period`: yields a
    step that runs `count` ticks of every stream, each to its end."""
    async with running(*streams) as tasks:

        async def step(count=1):
            for _ in range(count):
                await clock.advance(period, parked=tasks)

        yield step


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
# A third client, beside the director.
OTHER = Holder("other_inst", "fixture-core")
# The name of the director's lease of a robot.
DIRECTOR_LEASE = "tidy the table"
# Who halts the stack in the tests, and the halt as a refusal and the card
# give it.
HALTER = "peppy-cli"
HALTED = f"the stack is halted by {HALTER} until peppy fleet clear-halt"
# Why the headset lets go when the changes goal of the fleet ends.
CHANGES_ENDED = "the changes goal of the fleet ended"
# The bound of a call that the test answers, and the retry of the headset.
CALL_S = 10.0
RETRY_S = 1.0


# The revision of a listing that a test gives no revision.
FIRST_REVISION = 1
# A measured pose away from `POSE`, so a message at it is told apart.
ELSEWHERE = pose_at((0.4, 0.0, 0.0))
# The pairs a robot holds on a slot at most, in a test whose pairs join,
# leave and restart: one that restarts, and the one it replaces.
MOST_PAIRS = 2


def fleet_view(
    robots=(), *, leases=None, halted_by=None, revision=FIRST_REVISION
) -> FleetView:
    return FleetView(
        revision=revision,
        robots=frozenset(robots),
        leases=MappingProxyType(dict(leases or {})),
        halted_by=halted_by,
    )


def listed(driver: Driver, *robots, **view) -> Driver:
    """The driver once its goals run and a listing of `robots` came."""
    driver.goals_started(1)
    driver.goals_status(Following())
    driver.fleet_changed(fleet_view(robots, **view))
    return driver


class OnCue:
    """Calls, each waiting on a future of its own until the test answers it:
    `calls` keeps what each call names and its future, in call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, asyncio.Future]] = []

    async def wait_for_answer(self, named):
        """Wait for the test's answer to a call that names `named`."""
        answer = asyncio.get_running_loop().create_future()
        self.calls.append((named, answer))
        return await answer

    def unanswered(self) -> list[int]:
        """The index in `calls` of each call the test has not answered."""
        return [
            index
            for index, (_named, answer) in enumerate(self.calls)
            if not answer.done()
        ]

    def answer(self, index: int, value) -> None:
        """Call `index` of `calls` returns `value`."""
        self.calls[index][1].set_result(value)

    def throw(self, index: int, error: Exception) -> None:
        """Call `index` of `calls` raises `error`."""
        self.calls[index][1].set_exception(error)


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
        # The takes that wait for the test's answer, from `takes_on_cue` on.
        self._takes_on_cue: OnCue | None = None

    async def take(self, robot):
        self._log.append(("take", robot))
        await asyncio.sleep(0)
        if self._takes_on_cue is not None:
            return await self._takes_on_cue.wait_for_answer(robot)
        return self.take_outcome

    def takes_on_cue(self) -> OnCue:
        """Each take from now on waits until the test answers it in the
        returned calls."""
        self._takes_on_cue = OnCue()
        return self._takes_on_cue

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


class StopsOnCue(OnCue):
    """A backbone that answers each stop once the test answers it, each stop
    kept in call order with its robot."""

    async def stop(self, robot):
        return await self.wait_for_answer(robot)

    def answer(self, index: int, refusal: str | None = None) -> None:
        """Answer stop `index` of `calls`: with `refusal`, None for an
        answer."""
        super().answer(index, refusal)


class FakeGoals:
    """The posture goals of a test: each fire and cancel goes to `log`; a
    fire yields once and gives `handle`, a result waits on `results`."""

    def __init__(self, log: list) -> None:
        self._log = log
        # The goal a fire gives: accepted by default, a refused stub or None
        # (a send that failed) as a test sets it.
        self.handle = SimpleNamespace(accepted=True)
        # The notes, or the exceptions raised, of the results to come.
        self.results: asyncio.Queue = asyncio.Queue()

    async def fire(self, label, target):
        self._log.append(("goal", label))
        await asyncio.sleep(0)
        return self.handle

    async def result(self, label, goal) -> str:
        outcome = await self.results.get()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def cancel(self, label, goal) -> None:
        self._log.append(("cancel", label))


# The note of a move to home whose goal finished.
HOME_REACHED = "move to home completed"
# The posture goal a backbone refuses, with the remedy after the cause.
REFUSED_GOAL = SimpleNamespace(
    accepted=False, refusal=SimpleNamespace(message="busy; try again later")
)


def postures_of(robot: str) -> PostureTarget:
    """The postures producer of `robot` in a test, as a press reads it."""
    return PostureTarget(f"{robot} postures", f"{robot}_inst")


def _no_refresh() -> None:
    """A page that a change of a hold or of the fleet refreshes nothing on."""


def unlisted_driver(
    *,
    clock: Clock = REAL_CLOCK,
    log: list | None = None,
    calls: LeaseCalls | None = None,
    stops: Stops | None = None,
    goals: Goals | None = None,
    timing: Timing = Timing(),
    on_change: Callable[[], None] = _no_refresh,
) -> Driver:
    """A driver whose goals have not started: its daemon agrees to every
    call, its backbones stop on demand and its posture goals are accepted,
    each a fake that records its calls in `log`, unless given."""
    log = log if log is not None else []
    return Driver(
        HEADSET,
        calls if calls is not None else FakeCalls(log),
        stops if stops is not None else FakeStops(log),
        goals if goals is not None else FakeGoals(log),
        timing=timing,
        on_change=on_change,
        clock=clock,
    )


def scripted_driver(*robots, **fakes) -> Driver:
    """`unlisted_driver(**fakes)` once its goals run and its fleet lists
    `robots` free."""
    return listed(unlisted_driver(**fakes), *robots)


# The bound of each call to a mock that the test answers, and the retry.
WIRE_TIMING = Timing(call_timeout_s=CALL_S, retry_s=RETRY_S)

# The robot of a wire rig, outside any copy: its backbone's instance id.
ROBOT = "backbone_inst"
SETTINGS = config.from_parameters(
    default_parameters(command_rate_hz=100, stale_timeout_s=0.25)
)
TICK_S = SETTINGS.tick_period_s
# The most ticks a test waits for a state, and the ticks a held grip gets to
# stream where it must stream nothing.
WIRE_TICKS = 50
HELD_TICKS = 5


async def ticks_until(tick, predicate, limit=WIRE_TICKS) -> None:
    """Call `tick()` until `predicate()`, at most `limit` times."""
    for _ in range(limit):
        if predicate():
            return
        await tick()
    raise AssertionError("the condition was never met")


class TickedRig:
    """A rig whose loops run on `clock`: a tick is one step of each loop of
    `loops`."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        # The loops that each tick waits for.
        self.loops: list[asyncio.Task] = []

    async def tick(self, count=1):
        for _ in range(count):
            await self.clock.advance(TICK_S, parked=self.loops)

    async def ticks_until(self, predicate, limit=WIRE_TICKS):
        """Tick until `predicate()`."""
        await ticks_until(self.tick, predicate, limit)


def wire_stops(h, robots: PairedRobots) -> RobotStops:
    """The stops of the robots through the harness's mocks of `robot_stop`."""
    return RobotStops(h.node_runner, robots, stop, WIRE_TIMING)


def driver_on_the_wire(
    h, stops: Stops, clock: Clock, goals: Goals | None = None
) -> Driver:
    """A driver whose daemon is the harness's mock of the target `fleet`,
    whose backbones stop through `stops`, and whose posture goals are
    `goals`, `FakeGoals` unless given. The caller lists it (`listed`)."""
    return Driver(
        HEADSET,
        FleetCalls(
            h.node_runner,
            take_module=take,
            release_module=release,
            check_move_module=check_move,
            timing=WIRE_TIMING,
        ),
        stops,
        goals if goals is not None else FakeGoals([]),
        timing=WIRE_TIMING,
        on_change=lambda: None,
        clock=clock,
    )


async def daemon_answers_take(daemon, robot: str, refusal: str | None = None):
    """The mock daemon's answer to the next take, of `robot` under the
    headset's lease name: a yes, or `refusal`."""
    request, responder = await daemon.take.next_request(CALL_S)
    assert (request.robot, request.name) == (robot, LEASE_NAME)
    if refusal is None:
        await responder.respond()
    else:
        await responder.respond_failure(take.FailureKind.LEASED, refusal)


async def backbone_answers_stop(stops, stopped: list[str]) -> None:
    """A mock backbone's answer to the next stop, whose reason is the
    headset's lease name: the parts it stopped."""
    request, responder = await stops.next_request(CALL_S)
    assert request.reason == LEASE_NAME
    await responder.respond(stop.ResponseData(stopped=stopped))


async def daemon_answers_check_move(daemon, instance: str, refusal=None) -> None:
    """The mock daemon's answer to the next check whether the headset may
    call a member that moves the robot of `instance`: a yes, or `refusal`."""
    request, responder = await daemon.check_move.next_request(CALL_S)
    assert (request.instance, request.robot) == (instance, None)
    denied = (
        None
        if refusal is None
        else check_move.ResponseDataRefusal(failure_kind="leased", message=refusal)
    )
    await responder.respond(check_move.ResponseData(refusal=denied, revision=1))


async def daemon_answers_release(daemon, robot: str) -> None:
    """The mock daemon's yes to the next release, of `robot`."""
    request, responder = await daemon.release.next_request(CALL_S)
    assert request.robot == robot
    await responder.respond()


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


# The instance's clock when each base test starts (ns).
START_NS = 1_000_000_000_000


def limits_given(*, x=(-0.4, 0.8), y=(0.0, 0.0), turn=(-1.5, 1.0)):
    """The limits a drive_states gives for a base with these ranges (m/s,
    rad/s): a differential base by default, whose backward and clockwise
    limits differ from its forward and counterclockwise ones."""
    drives_y = y != (0.0, 0.0)
    return SimpleNamespace(
        linear_limits=drive_states.MessageLinearLimits(
            min_velocity=[x[0], y[0]],
            max_velocity=[x[1], y[1]],
            max_acceleration=[1.0, 1.0 if drives_y else 0.0],
            max_deceleration=[2.0, 2.0 if drives_y else 0.0],
            deadband=[0.0, 0.0],
            rest_velocity=[0.01, 0.01 if drives_y else 0.0],
        ),
        angular_limits=drive_states.MessageAngularLimits(
            min_rate=turn[0],
            max_rate=turn[1],
            max_acceleration=2.0,
            max_deceleration=3.0,
            deadband=0.0,
            rest_rate=0.02,
        ),
    )


DIFFERENTIAL = limits_given()
HOLONOMIC = limits_given(y=(-0.3, 0.6))


def stick_axes(forward=0.0, left=0.0) -> list[float]:
    """The xr-standard axes of a controller whose thumbstick is deflected
    `forward` and `left`: the Gamepad API gives -1 forward and left."""
    return [0.0, 0.0, -left, -forward]


def states(follow_state="following", limits=DIFFERENTIAL):
    """A drive_states as the headset reads it."""
    return SimpleNamespace(
        follow_state=follow_state,
        mode="standing",
        linear_limits=limits.linear_limits,
        angular_limits=limits.angular_limits,
    )


def setpoint_sent(send) -> tuple[Peer, tuple[float, float, float]] | None:
    """A base tick's send as its pair and its setpoint's x, y and turn, None
    for silence; the setpoint carries no frame and no timestamp."""
    if send is None:
        return None
    pair, (frame, timestamp, linear, angular) = send
    assert (frame, timestamp) == (None, None)
    return pair, (linear[0], linear[1], angular[2])
