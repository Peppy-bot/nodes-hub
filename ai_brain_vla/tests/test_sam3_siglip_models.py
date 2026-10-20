"""The sam3_siglip backend with its models: needs torch, transformers and
the two models' weights on the machine (perception/weights.py), so it skips
where they are absent (CI) and downloads nothing. Run it where they are: it
proves the detector embeds its vocabulary and the shipped enrolment, and
says nothing for an empty scene."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import NEVER  # noqa: E402
from ai_brain_vla.perception.sam3_siglip import BACKGROUND_PHRASES, Sam3SiglipDetector, load_vocabulary  # noqa: E402
from ai_brain_vla.perception.weights import SOURCES, node_directory  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the models need a GPU to answer in time")
@pytest.mark.skipif(
    not all((node_directory() / source.directory_name).is_dir() for source in SOURCES),
    reason=f"the models' weights are not in {node_directory()}",
)
def test_the_models_load_the_shipped_enrolment_and_find_nothing_in_a_blank_frame():
    detector = Sam3SiglipDetector()
    detector.load("waldo_catalogue")
    assert detector.why_unavailable() == ""
    assert detector._prototypes.shape[0] == 120
    assert detector._vocabulary_table.shape[0] == len(load_vocabulary()) + len(BACKGROUND_PHRASES)
    # A flat grey frame holds no item; every proposal, if any, is named as
    # background or kept under the confidence.
    assert detector.find(np.full((240, 320, 3), 110, dtype=np.uint8), [], NEVER) == []
