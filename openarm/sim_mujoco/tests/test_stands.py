"""The handoff between a robot attaching on the node loop and the thread that
steps the scene: a stand is answered once the thread has made it, an unstand
once nothing stands, and both fail when the engine stops first."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robots" / "openarm"))

from head_camera import Pack  # noqa: E402  pylint: disable=C0413
from stands import Stands  # noqa: E402  pylint: disable=C0413

SCENE = Path("/opt/robot_assets/openarm_bimanual_v2.xml")
HEAD_CAMERA_PACK = Pack(directory=Path("/staged/head_camera"), body_position=(0.0315, 0.0, 0.743))


def test_a_stand_reaches_the_thread_and_is_answered_when_made():
    stands = Stands()
    asked = stands.stand(SCENE, "alpha", head_camera_pack=HEAD_CAMERA_PACK)

    request = stands.next_stand(timeout_s=0.0)
    assert (request.scene, request.robot, request.head_camera_pack) == (
        SCENE,
        "alpha",
        HEAD_CAMERA_PACK,
    )
    assert not asked.done()

    request.future.set_result(None)
    assert asked.result(timeout=0) is None


def test_a_stand_the_thread_cannot_make_carries_the_reason_back():
    stands = Stands()
    asked = stands.stand(SCENE, "alpha", head_camera_pack=None)
    stands.next_stand(timeout_s=0.0).future.set_exception(FileNotFoundError("no such scene"))
    with pytest.raises(FileNotFoundError, match="no such scene"):
        asked.result(timeout=0)


def test_nothing_waits_when_no_stand_was_asked():
    assert Stands().next_stand(timeout_s=0.0) is None


def test_an_unstand_is_pending_until_the_thread_takes_it():
    stands = Stands()
    assert not stands.unstand_pending()
    asked = stands.unstand()
    assert stands.unstand_pending()

    # A second asker shares the pending answer.
    assert stands.unstand() is asked

    taken = stands.take_unstand()
    assert taken is asked
    assert not stands.unstand_pending()
    assert stands.take_unstand() is None
    taken.set_result(None)
    assert asked.result(timeout=0) is None


def test_a_stopping_engine_fails_everything_still_waiting():
    stands = Stands()
    standing = stands.stand(SCENE, "alpha", head_camera_pack=None)
    leaving = stands.unstand()

    stands.cancel_all("the engine stopped")

    for waiting in (standing, leaving):
        with pytest.raises(RuntimeError, match="the engine stopped"):
            waiting.result(timeout=0)
    assert stands.next_stand(timeout_s=0.0) is None
    assert not stands.unstand_pending()
