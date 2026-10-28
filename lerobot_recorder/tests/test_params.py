"""The launch parameters the node refuses before it starts, in the order it
checks them."""

import pytest

from lerobot_recorder import params as params_mod
from tests.support import make_params


def test_good_parameters_parse_into_the_config():
    config = params_mod.parse(make_params(fps=15, image_writer_threads=2))
    assert (config.fps, config.image_writer_threads) == (15, 2)
    assert str(config.storage.root) == "/tmp/unused"
    assert config.storage.s3 is None


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"fps": 0}, "fps must be positive"),
        ({"image_writer_threads": 0}, "image_writer_threads must be positive"),
        ({"storage_root": "relative"}, "storage_root must be an absolute path"),
        # Every parameter refused at once: the first checked names the fault.
        (
            {"fps": -1, "image_writer_threads": 0, "storage_root": "relative"},
            "fps must be positive",
        ),
        ({"image_writer_threads": 0, "storage_root": "relative"}, "image_writer_threads"),
    ],
)
def test_a_refused_parameter_is_named(overrides, refusal):
    with pytest.raises(ValueError, match=refusal):
        params_mod.parse(make_params(**overrides))
