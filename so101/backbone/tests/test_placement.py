"""The placement of a command through transform_lookup when the server
refuses the lookup or the lookup gets no answer: a refusal the server gives
once its wait has passed reaches the command with its kind, each error
peppylib raises for a call that gets no answer refuses the command with
resource_unavailable, a call that gives nothing the node can use with
attempt_failed, and an error of any other kind is no refusal. The points of
one frame and one time are placed together, by one lookup, and a placement
that carries a value past the finite numbers refuses the command with
rejected. A pose setpoint is placed by the same lookup, but one in the robot
frame with no timestamp, which is taken as it comes."""

from types import SimpleNamespace

import pytest
from control_core_py.geometry import Point, Pose, Stamped
from peppygen.consumed_services.transform_lookup import lookup_transform

from so101_backbone import placement
from so101_backbone.failures import FailureKind, Refusal
from so101_backbone.model import ROBOT_FRAME
from so101_backbone.wire import ANSWER_WAIT_S

TARGET = Stamped("odom", None, Point((0.1, 0.0, 0.2)))


def placer_with_lookup(monkeypatch, poll, timeout_s: float = 0.1) -> placement.Placer:
    """A placer whose calls of lookup_transform go to `poll`, each asking the
    server to wait `timeout_s` for the tree."""
    monkeypatch.setattr(placement.lookup_transform, "poll", poll)
    monkeypatch.setattr(placement.lookup_transform, "bound_producer", lambda _runner: None)
    return placement.Placer(object(), ROBOT_FRAME, timeout_s)


def placer_whose_lookup_raises(monkeypatch, error: Exception) -> placement.Placer:
    """A placer whose every call of lookup_transform raises `error`."""

    async def poll(*_arguments):
        raise error

    return placer_with_lookup(monkeypatch, poll)


@pytest.mark.parametrize(
    ("kind", "refused_kind"),
    [
        (lookup_transform.FailureKind.UNKNOWN_FRAME, FailureKind.UNKNOWN_FRAME),
        (lookup_transform.FailureKind.EXTRAPOLATION, FailureKind.EXTRAPOLATION),
    ],
    ids=["unknown_frame", "extrapolation"],
)
async def test_a_refusal_the_server_gives_once_its_wait_has_passed_reaches_the_command(
    monkeypatch, kind, refused_kind
):
    # The server holds a lookup it cannot answer for the request's timeout_s,
    # here above 5 s, then refuses it. The node waits that long plus its
    # answer wait on the wire, so the refusal reaches the command with its
    # kind.
    waits = []

    async def poll(_runner, _producer, request, timeout):
        waits.append(timeout)
        if timeout <= request.timeout_s:
            raise TimeoutError("the node stopped waiting while the server held the lookup")
        raise lookup_transform.Failure(kind, "the tree cannot place odom")

    placer = placer_with_lookup(monkeypatch, poll, timeout_s=6.0)

    with pytest.raises(Refusal) as refused:
        await placer.place(TARGET, "the target")

    assert refused.value.kind is refused_kind
    assert str(refused.value) == "the target: the tree cannot place odom"
    assert waits == [6.0 + ANSWER_WAIT_S]


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("no answer within the wait"),
        ConnectionError("transform_lookup is unreachable"),
        RuntimeError("the messaging failed"),
    ],
    ids=["timeout", "unreachable", "messaging"],
)
async def test_a_lookup_without_an_answer_refuses_the_command_with_resource_unavailable(
    monkeypatch, error
):
    placer = placer_whose_lookup_raises(monkeypatch, error)

    with pytest.raises(Refusal) as refused:
        await placer.place(TARGET, "the target")

    assert refused.value.kind is FailureKind.RESOURCE_UNAVAILABLE
    assert str(refused.value) == (
        f"the target: the robot's transform lookup is not answered ({error!r})"
    )


async def test_a_lookup_that_gives_nothing_the_node_can_use_refuses_the_command_with_attempt_failed(
    monkeypatch,
):
    # The generated client raises ValueError for a failure kind that
    # lookup_transform does not declare, as its Failure does for one.
    async def poll(*_arguments):
        raise lookup_transform.Failure("lost", "the tree is gone")

    placer = placer_with_lookup(monkeypatch, poll)

    with pytest.raises(Refusal) as refused:
        await placer.place(TARGET, "the target")

    assert refused.value.kind is FailureKind.ATTEMPT_FAILED
    assert str(refused.value).startswith(
        "the target: the robot's transform lookup gave nothing this node can use (ValueError("
        "\"'lost' is not a failure kind of lookup_transform"
    ), str(refused.value)


@pytest.mark.parametrize(
    "position, orientation, reason",
    [
        (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            "the orientation [0, 0, 0, 2] has a squared norm of 4, 0.01 or more from 1: send a unit "
            "quaternion [x, y, z, w], such as [0, 0, 0, 1]",
        ),
        (
            [float("nan"), 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            "a position must be finite: got x NaN, y 0, z 0; give a finite number of meters on each axis",
        ),
    ],
)
async def test_an_answer_that_holds_no_pose_refuses_the_command_with_attempt_failed(
    monkeypatch, position, orientation, reason
):
    # The answer is parsed as the contracts parse a pose; one the parse
    # refuses places nothing.
    async def poll(*_arguments):
        return SimpleNamespace(data=SimpleNamespace(position=position, orientation=orientation, timestamp=None))

    placer = placer_with_lookup(monkeypatch, poll)

    with pytest.raises(Refusal) as refused:
        await placer.place(TARGET, "the target")

    assert refused.value.kind is FailureKind.ATTEMPT_FAILED
    assert str(refused.value) == f"the target: the robot's transform lookup gave no pose: {reason}"


async def test_an_error_of_another_kind_is_no_refusal(monkeypatch):
    placer = placer_whose_lookup_raises(monkeypatch, TypeError("a fault of the node"))

    with pytest.raises(TypeError, match="a fault of the node"):
        await placer.place(TARGET, "the target")


async def test_the_points_of_one_frame_and_time_are_placed_together_by_one_lookup(monkeypatch):
    asked = []

    async def poll(_runner, _producer, request, _timeout_s):
        asked.append(request)
        return SimpleNamespace(
            data=SimpleNamespace(
                position=[1.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0], timestamp=None
            )
        )

    monkeypatch.setattr(placement.lookup_transform, "poll", poll)
    monkeypatch.setattr(placement.lookup_transform, "bound_producer", lambda _runner: None)
    placer = placement.Placer(object(), ROBOT_FRAME, 0.1)
    points = (Point((0.1, 0.0, 0.2)), Point((0.0, 0.3, 0.0)))

    placed = await placer.place(Stamped("odom", None, points), "the points")

    assert len(asked) == 1
    assert [point.position for point in placed.value] == [
        pytest.approx((1.1, 0.0, 0.2)),
        pytest.approx((1.0, 0.3, 0.0)),
    ]
    assert placed.timestamp is None


async def test_a_placement_that_carries_a_point_past_the_finite_numbers_refuses_it(monkeypatch):
    # A finite point placed by a finite placement can overflow: the points
    # are refused whole with rejected, naming them and what to give.
    async def poll(_runner, _producer, _request, _timeout_s):
        return SimpleNamespace(
            data=SimpleNamespace(
                position=[1.7e308, 0.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0], timestamp=None
            )
        )

    placer = placer_with_lookup(monkeypatch, poll)
    points = (Point((0.1, 0.0, 0.2)), Point((1.7e308, 0.0, 0.0)))

    with pytest.raises(Refusal) as refused:
        await placer.place(Stamped("odom", None, points), "the points")

    assert refused.value.kind is FailureKind.REJECTED
    assert str(refused.value) == (
        "the points: a position must be finite: got x inf, y 0, z 0; give a finite number of "
        "meters on each axis"
    )


def answering_identity(monkeypatch, asked: list) -> placement.Placer:
    """A placer whose every lookup is answered with the identity, each
    request appended to `asked`."""

    async def poll(_runner, _producer, request, _timeout_s):
        asked.append(request)
        return SimpleNamespace(
            data=SimpleNamespace(
                position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0], timestamp=None
            )
        )

    return placer_with_lookup(monkeypatch, poll)


async def test_a_pose_setpoint_asks_for_its_frame_at_its_time_in_the_robot_frame_at_the_latest_time(
    monkeypatch,
):
    # As pose_link places a setpoint: one in the robot frame at a time names
    # where the robot frame stood then, and one in another frame with no
    # timestamp where that frame stands at the latest time; one in the robot
    # frame with no timestamp names where the robot frame stands then, and is
    # taken as it comes, with no lookup.
    asked = []
    placer = answering_identity(monkeypatch, asked)
    pose = Pose.parse((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))

    for stamped in (
        Stamped(ROBOT_FRAME, 1_700_000_000.5, pose),
        Stamped("odom", None, pose),
        Stamped(ROBOT_FRAME, None, pose),
    ):
        assert await placer.place_setpoint(stamped) == pose

    assert [
        (r.source_frame, r.source_timestamp, r.target_frame, r.target_timestamp) for r in asked
    ] == [
        (ROBOT_FRAME, 1_700_000_000.5, ROBOT_FRAME, None),
        ("odom", None, ROBOT_FRAME, None),
    ]
