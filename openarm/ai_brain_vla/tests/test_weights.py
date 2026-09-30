"""The staging of the sam3_siglip backend's weights: what is fetched, where
it is staged, and what an interrupted fetch leaves. The Hub is a fake: the
suite downloads nothing."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import fake_download
from openarm_ai_brain_vla.perception import weights
from openarm_ai_brain_vla.perception.weights import FETCHING_SUFFIX, SAM3, SIGLIP, SOURCES, WEIGHTS_DIRECTORY_VARIABLE

NODE = Path(__file__).resolve().parents[1]


def test_every_model_is_pinned_by_a_full_commit_hash():
    assert SOURCES == (SAM3, SIGLIP)
    assert len({source.name for source in SOURCES}) == len(SOURCES)
    for source in SOURCES:
        assert re.fullmatch(r"[0-9a-f]{40}", source.revision), source
        assert source.directory_name == f"{source.name}-{source.revision}"
    # The checkpoint of SAM 3's original code base is not one transformers reads.
    assert SAM3.left_out == ("sam3.pt",) and SIGLIP.left_out == ()


def test_a_fetch_stages_every_model_under_its_name_and_revision(tmp_path):
    fetched_into = []

    def download(source, directory):
        fetched_into.append((source, directory.name))
        fake_download(source, directory)

    weights.fetch(tmp_path / "weights", download=download)
    # Each model is downloaded beside its staged name and renamed once whole.
    assert fetched_into == [(source, f"{source.directory_name}{FETCHING_SUFFIX}") for source in SOURCES]
    assert sorted(path.name for path in (tmp_path / "weights").iterdir()) == sorted(source.directory_name for source in SOURCES)
    for source in SOURCES:
        staged_file = tmp_path / "weights" / source.directory_name / "config.json"
        assert staged_file.read_text() == f"{source.repository}@{source.revision}"


def test_a_staged_model_is_not_fetched_again(tmp_path):
    weights.fetch(tmp_path, download=fake_download)

    def refuse(source, directory):
        raise AssertionError(f"fetched {source.repository} again")

    weights.fetch(tmp_path, download=refuse)
    assert weights.stage(SAM3, tmp_path, refuse) == tmp_path / SAM3.directory_name


def test_an_interrupted_fetch_stages_nothing_and_the_next_one_starts_again(tmp_path, monkeypatch):
    def interrupted(source, directory):
        (directory / "model.safetensors.incomplete").write_text("half of it")
        raise ConnectionError("the link went down")

    with pytest.raises(ConnectionError):
        weights.fetch(tmp_path, download=interrupted)
    # What the fetch left is not under the staged name, so nothing reads it.
    assert [path.name for path in tmp_path.iterdir()] == [f"{SAM3.directory_name}{FETCHING_SUFFIX}"]
    monkeypatch.setenv(WEIGHTS_DIRECTORY_VARIABLE, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="are not staged at"):
        weights.staged(SAM3)
    weights.fetch(tmp_path, download=fake_download)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(source.directory_name for source in SOURCES)
    assert [path.name for path in weights.staged(SAM3).iterdir()] == ["config.json"]


def test_staged_weights_are_found_through_the_variable(staged_weights):
    for source in SOURCES:
        assert weights.staged(source) == staged_weights / source.directory_name


def test_weights_that_are_not_staged_are_refused_saying_how_they_get_there(tmp_path, monkeypatch):
    how = r"the node image stages them at build \(apptainer.def\); a native run needs `python -m openarm_ai_brain_vla.perception.weights fetch <directory>` and OPENARM_AI_BRAIN_VLA_WEIGHTS naming that directory"
    monkeypatch.delenv(WEIGHTS_DIRECTORY_VARIABLE, raising=False)
    with pytest.raises(FileNotFoundError, match=f"the weights of jetjodh/sam3 are not staged, OPENARM_AI_BRAIN_VLA_WEIGHTS is not set: {how}"):
        weights.staged(SAM3)
    monkeypatch.setenv(WEIGHTS_DIRECTORY_VARIABLE, "  ")
    with pytest.raises(FileNotFoundError, match="OPENARM_AI_BRAIN_VLA_WEIGHTS is not set"):
        weights.staged(SAM3)
    monkeypatch.setenv(WEIGHTS_DIRECTORY_VARIABLE, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=f"the weights of google/siglip-so400m-patch14-384 are not staged at {tmp_path / SIGLIP.directory_name}: {how}"):
        weights.staged(SIGLIP)


@pytest.fixture
def hub(monkeypatch):
    """Stands in for huggingface_hub, which the suite does not install:
    records every snapshot asked for and writes one file into its
    directory."""
    asked = []

    def snapshot_download(repository, *, revision, local_dir, ignore_patterns):
        asked.append({"repository": repository, "revision": revision, "local_dir": local_dir, "ignore_patterns": ignore_patterns})
        (local_dir / "model.safetensors").write_text("weights")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    return asked


def test_the_hub_is_asked_for_the_pinned_revision_less_what_is_left_out(tmp_path, hub):
    weights.download_from_hub(SAM3, tmp_path)
    assert hub == [{"repository": "jetjodh/sam3", "revision": SAM3.revision, "local_dir": tmp_path, "ignore_patterns": ["sam3.pt"]}]


def test_the_command_stages_every_model_in_the_directory_it_is_given(tmp_path, hub, no_network):
    weights.main(["fetch", str(tmp_path / "weights")])
    assert [(asked["repository"], asked["revision"]) for asked in hub] == [(source.repository, source.revision) for source in SOURCES]
    for source in SOURCES:
        assert (tmp_path / "weights" / source.directory_name / "model.safetensors").read_text() == "weights"


def test_the_image_stages_the_weights_where_it_tells_the_node_to_read_them():
    definition = (NODE / "apptainer.def").read_text()
    exported = re.search(rf"export {WEIGHTS_DIRECTORY_VARIABLE}=(\S+)", definition)
    fetched = re.search(r"-m openarm_ai_brain_vla\.perception\.weights fetch \\\n\s+(\S+)", definition)
    assert exported and fetched and exported.group(1) == fetched.group(1)
