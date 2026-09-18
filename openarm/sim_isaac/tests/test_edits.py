"""The queue of scene changes the stepping thread makes, and what happens to
the ones still waiting when the simulation stops."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"


def _edits_module():
    spec = importlib.util.spec_from_file_location(
        "_edits_under_test", _ROBOT_DIR / "edits.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="edits")
def _edits():
    return _edits_module().Edits()


def test_a_drained_edit_answers_the_caller_waiting_on_it(edits):
    made = edits.submit(lambda: "stood")
    assert edits.drain() == 1
    assert made.result(timeout=1) == "stood"


def test_an_edit_that_raises_carries_the_reason_back(edits):
    made = edits.submit(lambda: (_ for _ in ()).throw(ValueError("no such model")))
    edits.drain()
    with pytest.raises(ValueError, match="no such model"):
        made.result(timeout=1)


def test_a_shutdown_fails_every_edit_still_waiting(edits):
    """Nothing drains the queue once the simulation loop is over, and a robot
    joining or leaving waits on its edit with no deadline, so a shutdown
    fails every edit still queued and no handler waits past it."""
    joining = edits.submit(lambda: "stood")
    leaving = edits.submit(lambda: "taken out")

    edits.cancel_all("the simulation is shutting down")

    for waiting in (joining, leaving):
        with pytest.raises(RuntimeError, match="shutting down"):
            waiting.result(timeout=1)
    # The queue is empty afterwards, so a later drain has nothing to run.
    assert edits.drain() == 0
