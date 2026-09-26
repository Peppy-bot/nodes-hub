"""The yoloe_vp backend with its model: needs torch, ultralytics and the
weights (fetched on first use), so it skips where they are absent (CI).
Run it where they are: it proves the port loads a gallery and says
nothing for an empty scene."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from openarm_ai_brain_vla.perception.yoloe_vp import YoloeVpDetector  # noqa: E402
from test_sam3_siglip import write_gallery  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the model needs a GPU to answer in time")
def test_the_model_loads_a_gallery_and_finds_nothing_in_a_blank_frame(tmp_path):
    detector = YoloeVpDetector()
    detector.load(str(write_gallery(tmp_path / "g")))
    assert detector.available
    detector.set_vocabulary([])
    assert detector.detect(np.full((240, 320, 3), 110, dtype=np.uint8)) == []
