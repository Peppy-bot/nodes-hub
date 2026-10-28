"""The vlm manipulator: brain_core's scripted grab and place with the
alignment step a catalog model answers, through `ModelAligner` over the
`RouterStepModel` of the model's route. `vlm_manipulator` builds it;
`probe_alignment` is the switch probe of a manipulation model."""

from __future__ import annotations

from brain_core.manipulators.scripted import CentreProposer, ScriptedManipulator

from ...catalog import CatalogEntry, Route
from ...provider import Provider
from .aligner import ModelAligner
from .step_model import RouterStepModel, probe_alignment

__all__ = ["ModelAligner", "RouterStepModel", "probe_alignment", "vlm_manipulator"]


def vlm_manipulator(entry: CatalogEntry, route: Route, provider: Provider) -> ScriptedManipulator:
    """The scripted grab and place whose alignment step the catalog model
    `entry` answers through `route`."""
    return ScriptedManipulator(CentreProposer(), aligner=ModelAligner(RouterStepModel(entry, route, provider)))
