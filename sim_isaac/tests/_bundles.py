"""What the suites check of the bundle a builder writes beside its stage:
the files it holds, MODIFICATIONS.md, the manifest, the date and the release
it takes, and that a build of the same release on the same date packs to the
same bytes."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mujoco_usd  # noqa: E402  pylint: disable=C0413

# The date the suites build a bundle on.
BUILD_DATE = "2026-11-21"


def staged_release(tmp_path_factory, variable: str, robot_id: str) -> Path:
    """The release waldo_catalogue.py pins, staged with `robot_id`'s model in
    the directory the environment variable `variable` names, or in a
    directory of the suite's own without it, which needs the network. A
    release already staged there is checked against its digest."""
    staged = os.environ.get(variable)
    release = Path(staged) if staged else tmp_path_factory.mktemp(f"{robot_id}_release") / "release"
    mujoco_usd.waldo_catalogue.fetch(release, robots=[robot_id])
    return release


def _build(builder, release: Path, directory: Path) -> tuple[Path, str]:
    """The bundle `builder` writes from `release` on BUILD_DATE into a
    directory under `directory`, and the SHA-256 of the archive it packs."""
    output, archive = directory / "bundle", directory / "bundle.tar.gz"
    arguments = ["--release", str(release), "--output", str(output), "--date", BUILD_DATE]
    assert builder.main([*arguments, "--archive", str(archive)]) == 0
    return output, mujoco_usd.sha256(archive.read_bytes())


def check_bundle(builder, release: Path, directory: Path) -> tuple[Path, dict, str]:
    """Builds the bundle of `builder` and checks what every bundle holds: its
    stage, its upstream notices as the release stages them,
    MODIFICATIONS.md, which says who modified which files and when, and the
    manifest of the date and of each script's and each file's digest; and
    that a second build of the same release on the same date packs to the
    same bytes. Returns the bundle's directory, its manifest, and the text
    of its MODIFICATIONS.md on one line."""
    bundle = builder.BUNDLE
    output, packed = _build(builder, release, directory / "first")
    files = [bundle.stage, *bundle.notices, mujoco_usd.MODIFICATIONS_FILENAME]
    assert sorted(path.name for path in output.iterdir()) == sorted([*files, mujoco_usd.MANIFEST_FILENAME])
    for name, path in bundle.notices.items():
        assert (output / name).read_bytes() == (release / bundle.robot_id / path).read_bytes(), name
    modifications = " ".join((output / mujoco_usd.MODIFICATIONS_FILENAME).read_text().split())
    assert modifications.startswith(f"# Modifications {bundle.stage} is a modified form of the files at ")
    assert (
        f"Peppy Inc made it on {BUILD_DATE} with sim_isaac/scripts/{bundle.builder.name} and "
        "sim_isaac/scripts/mujoco_usd.py of Peppy-bot/nodes-hub"
    ) in modifications
    manifest = json.loads((output / mujoco_usd.MANIFEST_FILENAME).read_text())
    assert manifest["modified"] == {"by": "Peppy Inc", "date": BUILD_DATE}
    assert manifest["builder"] == {
        script.name: mujoco_usd.sha256(script.read_bytes())
        for script in (Path(builder.__file__), Path(mujoco_usd.__file__))
    }
    assert manifest["files"] == {name: mujoco_usd.sha256((output / name).read_bytes()) for name in files}
    assert _build(builder, release, directory / "second")[1] == packed
    return output, manifest, modifications


def check_the_date_is_required(builder, release: Path, directory: Path, capsys) -> None:
    """A build with no `--date`, or with one that is no date, is refused
    before it writes anything, naming what to give."""
    output = directory / "bundle"
    for date in ([], ["--date", "2026-13-01"]):
        with pytest.raises(SystemExit) as refused:
            builder.main(["--release", str(release), "--output", str(output), *date])
        assert refused.value.code == 2
    refusals = capsys.readouterr().err
    assert "the following arguments are required: --date" in refusals
    assert "argument --date: '2026-13-01' is no date: give the date of the build as YYYY-MM-DD" in refusals
    assert not output.exists()


def check_the_release_is_the_pinned_one(builder, release: Path, directory: Path, capsys) -> None:
    """A build from a directory that holds no release, one that holds another
    release, or the pinned release staged without the builder's robot, is
    refused before it writes anything, naming the command that stages the
    pinned release; the pinned release staged with the robot, `release`, is
    read as it is."""
    robot_id = builder.BUNDLE.robot_id
    pinned = mujoco_usd.waldo_catalogue.RELEASE_DIGEST
    other_digest = "0" * 64
    other = directory / "other"
    other.mkdir()
    index = json.loads((release / "index.json").read_text())
    (other / "index.json").write_text(json.dumps(index | {"prefix": f"catalogues/{other_digest}"}))
    without = directory / "without"
    without.mkdir()
    for name in ("index.json", "robots.json"):
        shutil.copy(release / name, without / name)
    empty = directory / "empty"
    empty.mkdir()
    output = directory / "bundle"
    for staged in (empty, other, without):
        with pytest.raises(SystemExit) as refused:
            builder.main(["--release", str(staged), "--output", str(output), "--date", BUILD_DATE])
        assert refused.value.code == 2
    refusals = capsys.readouterr().err
    staging = (
        "stage the release waldo_catalogue.py pins in a new directory, a path that does not exist yet, with "
        f"`python3 sim_base_images/waldo_catalogue.py fetch <new directory> --robot {robot_id}` "
        "and give that directory as --release"
    )
    assert (
        f"argument --release: {empty} does not hold the release waldo_catalogue.py pins ([Errno 2] No such file "
        f"or directory: '{empty / 'index.json'}'): {staging}"
    ) in refusals
    assert (
        f"argument --release: {other} does not hold the release waldo_catalogue.py pins (the release index "
        f"names 'catalogues/{other_digest}', not catalogues/{pinned}): {staging}"
    ) in refusals
    assert (
        f"argument --release: the release staged at {without} holds no model of '{robot_id}': {staging}"
    ) in refusals
    assert not output.exists()
    assert mujoco_usd.pinned_release(str(release), robot_id) == release
