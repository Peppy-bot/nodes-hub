"""The sam3_siglip backend with its models: needs torch, transformers and
the two models' weights, so it skips where they are absent (CI). Run it
where they are: it proves the port embeds its vocabulary and an enrolment
gallery, and says nothing for an empty scene."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import NEVER  # noqa: E402
from openarm_ai_brain_vla.perception.sam3_siglip import BACKGROUND_PHRASES, Sam3SiglipDetector, load_vocabulary  # noqa: E402
from test_gallery import write_harvest  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the models need a GPU to answer in time")
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
