"""The sam3_siglip backend with its models: needs torch, transformers and
the two models' weights staged (perception/weights.py), so it skips where
they are absent (CI). Run it where they are: it proves the port embeds its
vocabulary and an enrolment gallery, and says nothing for an empty scene."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import NEVER  # noqa: E402
from openarm_ai_brain_vla.perception.sam3_siglip import BACKGROUND_PHRASES, Sam3SiglipDetector, load_vocabulary  # noqa: E402
from openarm_ai_brain_vla.perception.weights import WEIGHTS_DIRECTORY_VARIABLE  # noqa: E402
from test_gallery import write_harvest  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the models need a GPU to answer in time")
@pytest.mark.skipif(WEIGHTS_DIRECTORY_VARIABLE not in os.environ, reason=f"the models' weights are not staged: {WEIGHTS_DIRECTORY_VARIABLE} is not set")
def test_the_models_load_a_gallery_and_find_nothing_in_a_blank_frame(tmp_path):
    detector = Sam3SiglipDetector()
    detector.load("", str(write_harvest(tmp_path / "g")))
    assert detector.available
    assert detector._prototypes.shape[0] == 3
    assert detector._vocabulary_table.shape[0] == len(load_vocabulary()) + len(BACKGROUND_PHRASES)
    detector.set_vocabulary([])
    # A flat grey frame holds no item; every proposal, if any, is named as
    # background or kept under the confidence.
    assert detector.detect(np.full((240, 320, 3), 110, dtype=np.uint8), NEVER) == []
