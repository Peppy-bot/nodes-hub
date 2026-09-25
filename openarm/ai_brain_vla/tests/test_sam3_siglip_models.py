"""The sam3_siglip backend with its models: needs torch, transformers and
the two models' weights, so it skips where they are absent (CI). Run it
where they are: it proves the port names a gallery item on a frame and
says nothing for an empty scene."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from openarm_ai_brain_vla.perception.sam3_siglip import Sam3SiglipDetector  # noqa: E402
from test_sam3_siglip import write_gallery  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the models need a GPU to answer in time")
def test_the_models_load_a_gallery_and_find_nothing_in_a_blank_frame(tmp_path):
    detector = Sam3SiglipDetector()
    detector.load(str(write_gallery(tmp_path / "g")))
    assert detector.available
    detector.set_vocabulary([])
    # A flat grey frame holds none of the items; every proposal, if any,
    # names as no item above the study's confidence.
    assert detector.detect(np.full((240, 320, 3), 110, dtype=np.uint8)) == []
