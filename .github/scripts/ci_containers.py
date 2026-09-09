"""Build cached test images and construct Apptainer commands for CI."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


@contextmanager
def log_group(title: str, *, stream=sys.stdout):
    print(f"::group::{title}", file=stream, flush=True)
    try:
        yield
    finally:
        print("::endgroup::", file=stream, flush=True)


def test_image_definition(project: Path, fallback_base: str) -> str:
    """Keep image preparation, stopping before the node builds copied sources."""
    definition = project / "apptainer.def"
    if not definition.is_file():
        return f"Bootstrap: docker\nFrom: {fallback_base}\n"

    lines = definition.read_text().splitlines()
    base = next((line.removeprefix("From:").lstrip() for line in lines if line.startswith("From:")), "")
    if not base:
        raise ValueError(f"{definition} names no image on a From: line")

    source_dir = None
    in_files = False
    for line in lines:
        if line.startswith("%files"):
            in_files = True
        elif re.match(r"%[a-z]", line):
            in_files = False
        elif in_files and len(fields := line.split()) == 2:
            source_dir = fields[1]
            break

    preparation = []
    in_post = False
    for line in lines:
        if line.startswith("%post"):
            in_post = True
        elif re.match(r"%[a-z]", line):
            in_post = False
        elif in_post:
            if source_dir and re.fullmatch(r"\s*cd\s+" + re.escape(source_dir) + r"\s*", line):
                break
            preparation.append(line + "\n")

    return f"Bootstrap: docker\nFrom: {base}\n\n%post\n" + "".join(preparation)


def derive_test_image(project: str, fallback_base: str) -> Path:
    """Cache whole images by definition hash; failed builds leave no image behind."""
    cache = Path(os.environ["CI_CACHE_DIR"])
    apptainer = Path(os.environ["PEPPY_APPTAINER_DIR"]) / "bin/apptainer"
    definition = test_image_definition(Path(project), fallback_base)
    digest = hashlib.sha256(definition.encode()).hexdigest()
    images = cache / "test-images"
    images.mkdir(parents=True, exist_ok=True)
    image = images / f"{digest}.sif"
    if image.is_file():
        return image

    # Rename only a completed build, so other runs never execute a partial SIF.
    partial = image.with_name(f"{image.name}.building.{os.getpid()}")
    try:
        with tempfile.TemporaryDirectory(prefix="nodes-hub-test-image-") as directory:
            source = Path(directory) / "apptainer.def"
            source.write_text(definition)
            with log_group(f"build the test image for {project}", stream=sys.stderr):
                print(definition, file=sys.stderr, end="", flush=True)
                subprocess.run(
                    [str(apptainer), "build", str(partial), str(source)],
                    env={**os.environ, "APPTAINER_CACHEDIR": str(cache / "apptainer")},
                    stdout=sys.stderr,
                    check=True,
                )
            partial.replace(image)
    finally:
        partial.unlink(missing_ok=True)
    return image


def container_command(
    project: str, image: Path, *, binds: list[str], variables: dict[str, str]
) -> list[str]:
    """Bind the checkout and explicitly pass each suite's paths and environment."""
    apptainer = Path(os.environ["PEPPY_APPTAINER_DIR"]) / "bin/apptainer"
    command = [
        str(apptainer), "exec", "--cleanenv", "--no-home", "--pwd", f"/work/{project}",
        "--bind", f"{os.environ['GITHUB_WORKSPACE']}:/work",
    ]
    for bind in binds:
        command.extend(["--bind", bind])
    for name, value in variables.items():
        command.extend(["--env", f"{name}={value}"])
    return [*command, str(image)]
