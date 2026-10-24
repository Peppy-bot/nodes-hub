"""The models this brain runs, for brain_core: each role runs the model its
parameter names for the node's whole life, and the parameters are read
once, at the start."""

import re
from pathlib import Path

import pytest
from brain_core.manipulation.none import NoneManipulator
from brain_core.models import Models
from brain_core.perception.none import NoneDetector
from brain_core.ports import FailureKind, Refusal, Role
from brain_core.sequencer import Sequencer
from conftest import FakeDetector, FakeManipulator

from ai_brain_vla.models import MANIPULATORS, VlaEntry, VlaModels
from ai_brain_vla.perception import DETECTORS, TAKES_A_GALLERY

MANIFEST = Path(__file__).resolve().parents[1] / "peppy.json5"


def test_the_default_perception_model_is_sam3_siglip():
    # The manifest names the default; the brain must still run it.
    [default] = re.findall(r'perception_model: \{ \$type: "string", \$default: "([^"]*)" \}', MANIFEST.read_text())
    assert default == "sam3_siglip" and default in DETECTORS


class Built:
    """A registry of one fake detector that records the gallery each build
    was given."""

    def __init__(self) -> None:
        self.galleries: list[str] = []
        self.detector = FakeDetector()

    def __call__(self, gallery: str):
        self.galleries.append(gallery)
        return self.detector


def test_the_brain_runs_none_or_sam3_siglip_for_perception_and_none_for_manipulation():
    assert sorted(DETECTORS) == ["none", "sam3_siglip"] and TAKES_A_GALLERY == {"sam3_siglip"}
    assert sorted(MANIPULATORS) == ["none"]
    assert isinstance(DETECTORS["none"](""), NoneDetector)
    assert isinstance(MANIPULATORS["none"](), NoneManipulator)


@pytest.mark.parametrize("perception, manipulation, gallery, message", [
    ("sam9", "none", "", "^perception_model 'sam9' is not a model this brain runs; it runs none, sam3_siglip$"),
    ("gemini_er", "none", "", "^perception_model 'gemini_er' is not a model this brain runs"),
    ("none", "policy", "", "^manipulation_model 'policy' is not a model this brain runs; it runs none$"),
    ("sam3_siglip", "none", "/data/gallery", r"^perception_gallery '/data/gallery' is not an enrolment this node ships \(waldo_catalogue\)$"),
    ("none", "none", "waldo_catalogue", "^perception_gallery 'waldo_catalogue' is for a model that reads one; perception_model is 'none'$"),
])
def test_a_parameter_the_brain_cannot_run_stops_it_at_its_start(perception, manipulation, gallery, message):
    with pytest.raises(ValueError, match=message):
        VlaModels(perception, manipulation, gallery)


def test_each_role_offers_its_running_model_alone_in_the_present_tense():
    models = VlaModels("sam3_siglip", "none", "waldo_catalogue", detectors={"sam3_siglip": Built()})
    assert models.roles() == (Role.PERCEPTION, Role.MANIPULATION)
    assert models.names(Role.PERCEPTION) == ["sam3_siglip"] and models.names(Role.MANIPULATION) == ["none"]
    assert models.offered(Role.PERCEPTION, "sam3_siglip") == VlaEntry(Role.PERCEPTION, "sam3_siglip")
    with pytest.raises(Refusal, match="^this brain runs only 'sam3_siglip' for perception$") as raised:
        models.offered(Role.PERCEPTION, "none")
    assert raised.value.kind is FailureKind.REJECTED
    with pytest.raises(Refusal, match="^this brain runs only 'none' for manipulation$") as raised:
        models.offered(Role.MANIPULATION, "scripted")
    assert raised.value.kind is FailureKind.REJECTED


async def test_the_start_builds_each_model_once_and_a_switch_to_it_hands_back_the_same():
    built = Built()
    manipulator = FakeManipulator()
    models = VlaModels(
        "sam3_siglip", "scripted", " waldo_catalogue ",
        detectors={"sam3_siglip": built}, manipulators={"scripted": lambda: manipulator},
    )
    core = Models(models, Sequencer(), robot=None)
    await core.start({Role.PERCEPTION: "sam3_siglip", Role.MANIPULATION: "scripted"})
    assert core.current(Role.PERCEPTION) is built.detector and core.current(Role.MANIPULATION) is manipulator
    assert built.galleries == [" waldo_catalogue "]
    assert await models.prepare(VlaEntry(Role.PERCEPTION, "sam3_siglip")) is built.detector
    assert built.galleries == [" waldo_catalogue "]
