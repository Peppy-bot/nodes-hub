"""The scenes this engine stands, by the model a robot attaches with."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robots" / "openarm"))

from head_camera import Pack  # noqa: E402  pylint: disable=C0413
from scenes import BAKED_SCENES, Catalogue  # noqa: E402  pylint: disable=C0413

# The head camera pack every catalogue carries; finding scenes never reads it.
HEAD_CAMERA_PACK = Pack(directory=Path("/staged/head_camera"), body_position=(0.0315, 0.0, 0.743))


def test_the_baked_catalogue_carries_both_generations():
    assert Catalogue.baked(head_camera_pack=HEAD_CAMERA_PACK).models() == ["openarm_v1", "openarm_v2"]
    assert set(BAKED_SCENES) == {"openarm_v1", "openarm_v2"}


def test_a_model_names_its_scene_file(tmp_path):
    scene = tmp_path / "v2.xml"
    scene.write_text("<mujoco/>")
    catalogue = Catalogue({"openarm_v2": scene}, head_camera_pack=HEAD_CAMERA_PACK)
    assert catalogue.scene("openarm_v2") == scene


def test_an_unknown_model_says_what_the_engine_stands(tmp_path):
    catalogue = Catalogue(
        {"openarm_v2": tmp_path / "v2.xml", "openarm_v1": tmp_path / "v1.xml"},
        head_camera_pack=HEAD_CAMERA_PACK,
    )
    with pytest.raises(ValueError, match="unknown model 'openarm_v3': this engine stands openarm_v1, openarm_v2"):
        catalogue.scene("openarm_v3")


def test_a_scene_missing_from_the_image_is_reported(tmp_path):
    catalogue = Catalogue({"openarm_v2": tmp_path / "missing.xml"}, head_camera_pack=HEAD_CAMERA_PACK)
    with pytest.raises(FileNotFoundError, match="baked into the container image"):
        catalogue.scene("openarm_v2")
