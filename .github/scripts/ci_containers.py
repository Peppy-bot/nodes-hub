"""Build cached test images and construct Apptainer commands for CI."""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request


@contextmanager
def log_group(title: str, *, stream=sys.stdout):
    print(f"::group::{title}", file=stream, flush=True)
    try:
        yield
    finally:
        print("::endgroup::", file=stream, flush=True)


# The media types of an image manifest and of a multi-platform index. A
# registry answers a tag with the digest of what the tag names: the index of a
# multi-platform image, the manifest of a single-platform one.
MANIFEST_MEDIA_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
DOCKER_HUB_NAMES = ("docker.io", "index.docker.io", "registry-1.docker.io")
DOCKER_HUB_API_HOST = "registry-1.docker.io"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
REGISTRY_TIMEOUT_SECONDS = 30


class RegistryError(Exception):
    """A registry answered with something other than a digest or a token."""


@dataclass(frozen=True)
class ImageReference:
    """The image a `From:` line names: `[registry/]repository[:tag][@digest]`.

    `name` is the reference as the line spells it, without its tag and digest.
    `api_host` and `repository` locate the image in the registry API.
    """

    name: str
    api_host: str
    repository: str
    tag: str
    digest: str | None

    @classmethod
    def parse(cls, text: str) -> "ImageReference":
        name, at, digest = text.partition("@")
        if at and not DIGEST.fullmatch(digest):
            raise ValueError(f"{text!r} names a digest that is not sha256:<64 hex digits>")
        tag = "latest"
        if ":" in name.rsplit("/", 1)[-1]:
            name, _, tag = name.rpartition(":")
        if not name or not tag or any(character.isspace() for character in text):
            raise ValueError(f"{text!r} is not an image reference")

        first, slash, rest = name.partition("/")
        names_registry = bool(slash) and ("." in first or ":" in first or first == "localhost")
        api_host, repository = (first, rest) if names_registry else (DOCKER_HUB_NAMES[0], name)
        if api_host in DOCKER_HUB_NAMES:
            api_host = DOCKER_HUB_API_HOST
            if "/" not in repository:
                repository = f"library/{repository}"
        return cls(name=name, api_host=api_host, repository=repository, tag=tag, digest=digest or None)

    def pinned(self, digest: str) -> str:
        return f"{self.name}@{digest}"


def anonymous_token(challenge: str) -> str:
    """The token that a `WWW-Authenticate: Bearer ...` challenge gives anyone."""
    scheme, _, parameters = challenge.partition(" ")
    fields = dict(re.findall(r'(\w+)="([^"]*)"', parameters))
    realm = fields.pop("realm", "")
    if scheme.lower() != "bearer" or not realm:
        raise RegistryError(f"the registry asks for an authentication other than a bearer token: {challenge!r}")
    url = f"{realm}?{urllib.parse.urlencode(fields)}"
    with urllib.request.urlopen(url, timeout=REGISTRY_TIMEOUT_SECONDS) as response:
        body = json.load(response)
    token = (body.get("token") or body.get("access_token")) if isinstance(body, dict) else None
    if not isinstance(token, str) or not token:
        raise RegistryError(f"{realm} gives no token")
    return token


def manifest_headers(url: str, headers: dict[str, str]):
    """The headers of the answer to a HEAD request of a manifest.

    An HTTPError holds the answer open: it is closed before it is raised, and
    its headers stay readable.
    """
    request = urllib.request.Request(url, headers=headers, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=REGISTRY_TIMEOUT_SECONDS) as response:
            return response.headers
    except urllib.error.HTTPError as error:
        error.close()
        raise


def registry_digest(reference: ImageReference) -> str:
    """The digest that the registry of *reference* gives its tag now.

    It asks with a HEAD request of the manifest, which Docker Hub does not
    count against its pull limits. A registry that asks for a token gets an
    anonymous one.
    """
    url = f"https://{reference.api_host}/v2/{reference.repository}/manifests/{reference.tag}"
    headers = {"Accept": MANIFEST_MEDIA_TYPES}
    try:
        answer = manifest_headers(url, headers)
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise
        token = anonymous_token(error.headers.get("WWW-Authenticate", ""))
        answer = manifest_headers(url, {**headers, "Authorization": f"Bearer {token}"})
    digest = answer.get("Docker-Content-Digest", "")
    if not DIGEST.fullmatch(digest):
        raise RegistryError(f"{url} gives no digest of its manifest")
    return digest


def pinned_base_image(base: str) -> str:
    """*base* pinned to the digest that its registry gives it now.

    A test image is cached under the hash of its definition, so a definition
    that names a tag keeps its image when the tag moves to another image.
    Pinned, the definition changes with the image. When the registry cannot be
    read, *base* stays as it is, with a warning, and the image cached for it
    is used.
    """
    reference = ImageReference.parse(base)
    if reference.digest:
        return base
    try:
        return reference.pinned(registry_digest(reference))
    except (OSError, ValueError, RegistryError) as error:
        print(f"::warning::cannot read the digest of {base}, so the test image is cached under its tag: {error}", file=sys.stderr, flush=True)
        return base


def test_image_definition(project: Path, fallback_base: str, pin_base: Callable[[str], str]) -> str:
    """Keep image preparation, stopping before the node builds copied sources.

    The definition names its base image as *pin_base* gives it.
    """
    definition = project / "apptainer.def"
    if not definition.is_file():
        return f"Bootstrap: docker\nFrom: {pin_base(fallback_base)}\n"

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

    return f"Bootstrap: docker\nFrom: {pin_base(base)}\n\n%post\n" + "".join(preparation)


# Where a suite's cargo keeps the crates it downloads, in the container.
# Cargo's default home there lies under the runner's home, which --no-home
# leaves read-only.
CARGO_HOME = "/cargo"


def cargo_home_bind(cache: Path) -> str:
    """The bind of the CI cache's cargo home at CARGO_HOME, made on first
    use: the crates every suite's cargo downloads, kept between runs."""
    home = cache / "cargo-home"
    home.mkdir(parents=True, exist_ok=True)
    return f"{home}:{CARGO_HOME}"


def derive_test_image(project: str, fallback_base: str) -> Path:
    """Cache whole images by definition hash; failed builds leave no image behind.

    The definition pins its base image by digest, so an image is built again
    when the tag of its base image moves.
    """
    cache = Path(os.environ["CI_CACHE_DIR"])
    apptainer = Path(os.environ["PEPPY_APPTAINER_DIR"]) / "bin/apptainer"
    definition = test_image_definition(Path(project), fallback_base, pinned_base_image)
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
