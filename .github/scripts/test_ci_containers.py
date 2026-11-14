#!/usr/bin/env python3
"""How a test image names its base image, and when it is built again."""

import contextlib
from email.message import Message
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

import ci_containers
from ci_containers import ImageReference, RegistryError

DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64
TOKEN_REALM = "https://auth.example.com/token"
CHALLENGE = (
    f'Bearer realm="{TOKEN_REALM}",service="registry.example.com",'
    'scope="repository:peppybot/python-uv-base:pull"'
)


def headers(**fields):
    message = Message()
    for name, value in fields.items():
        message[name.replace("_", "-")] = value
    return message


class Answer:
    """What urlopen returns: a context manager with headers and a body."""

    def __init__(self, header_fields=None, body=b""):
        self.headers = headers(**(header_fields or {}))
        self._body = io.BytesIO(body)

    def read(self, *args):
        return self._body.read(*args)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeRegistry:
    """Answers the manifest requests and the token requests of a registry
    that, like Docker Hub, asks for a bearer token first."""

    def __init__(self, digest=DIGEST, token_body=None, challenge=CHALLENGE):
        self.digest = digest
        self.token_body = token_body if token_body is not None else json.dumps({"token": "t0k"}).encode()
        self.challenge = challenge
        self.requests = []

    def urlopen(self, request, timeout):
        if isinstance(request, str):
            self.requests.append(("GET", request, {}))
            return Answer(body=self.token_body)
        sent = {name.lower(): value for name, value in request.header_items()}
        self.requests.append((request.get_method(), request.full_url, sent))
        if self.challenge and "authorization" not in sent:
            raise urllib.error.HTTPError(
                request.full_url, 401, "Unauthorized", headers(WWW_Authenticate=self.challenge), None
            )
        if self.digest is None:
            return Answer()
        return Answer({"Docker-Content-Digest": self.digest})


@contextlib.contextmanager
def registry(fake):
    with mock.patch.object(ci_containers.urllib.request, "urlopen", fake.urlopen):
        yield fake


class ParseImageReference(unittest.TestCase):
    def test_a_docker_hub_reference_is_located_in_the_registry_api(self):
        cases = {
            "peppybot/python-uv-base:latest": (
                "peppybot/python-uv-base", "registry-1.docker.io", "peppybot/python-uv-base", "latest", None,
            ),
            # No tag names `latest`.
            "peppybot/rust-cargo-base": (
                "peppybot/rust-cargo-base", "registry-1.docker.io", "peppybot/rust-cargo-base", "latest", None,
            ),
            # An official image lives under `library/`.
            "ubuntu:24.04": ("ubuntu", "registry-1.docker.io", "library/ubuntu", "24.04", None),
            "docker.io/peppybot/sim-mujoco:3.17.0-1": (
                "docker.io/peppybot/sim-mujoco", "registry-1.docker.io", "peppybot/sim-mujoco", "3.17.0-1", None,
            ),
            "index.docker.io/ubuntu": ("index.docker.io/ubuntu", "registry-1.docker.io", "library/ubuntu", "latest", None),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                reference = ImageReference.parse(text)
                self.assertEqual(
                    (reference.name, reference.api_host, reference.repository, reference.tag, reference.digest),
                    expected,
                )

    def test_a_first_component_with_a_dot_a_port_or_localhost_names_the_registry(self):
        cases = {
            "nvcr.io/nvidia/cuda:12.6.0-devel-ubuntu24.04": ("nvcr.io", "nvidia/cuda", "12.6.0-devel-ubuntu24.04"),
            "localhost:5000/base": ("localhost:5000", "base", "latest"),
            "localhost:5000/base:1": ("localhost:5000", "base", "1"),
            "localhost/nodes/base:2": ("localhost", "nodes/base", "2"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                reference = ImageReference.parse(text)
                self.assertEqual((reference.api_host, reference.repository, reference.tag), expected)

    def test_a_reference_keeps_the_digest_it_names(self):
        reference = ImageReference.parse(f"peppybot/python-uv-base:latest@{DIGEST}")

        self.assertEqual(reference.digest, DIGEST)
        self.assertEqual(reference.name, "peppybot/python-uv-base")

    def test_a_pinned_reference_names_the_digest_and_no_tag(self):
        reference = ImageReference.parse("docker.io/peppybot/python-uv-base:latest")

        self.assertEqual(reference.pinned(DIGEST), f"docker.io/peppybot/python-uv-base@{DIGEST}")

    def test_text_in_another_form_is_refused(self):
        for text in ["", ":latest", "peppybot/base:", "peppybot/base@sha256:abc", "peppybot/base@md5:" + "a" * 32,
                     "peppybot/base latest"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                ImageReference.parse(text)


class RegistryDigest(unittest.TestCase):
    def test_the_digest_is_asked_with_a_head_request_and_an_anonymous_token(self):
        with registry(FakeRegistry()) as fake:
            digest = ci_containers.registry_digest(ImageReference.parse("peppybot/python-uv-base:latest"))

        self.assertEqual(digest, DIGEST)
        manifest_url = "https://registry-1.docker.io/v2/peppybot/python-uv-base/manifests/latest"
        (first_method, first_url, first_headers), (token_method, token_url, _), (second_method, second_url, second_headers) = fake.requests
        self.assertEqual((first_method, first_url), ("HEAD", manifest_url))
        self.assertNotIn("authorization", first_headers)
        # The token is asked with the service and the scope of the challenge.
        self.assertEqual(token_method, "GET")
        self.assertEqual(
            token_url,
            f"{TOKEN_REALM}?service=registry.example.com&scope=repository%3Apeppybot%2Fpython-uv-base%3Apull",
        )
        self.assertEqual((second_method, second_url), ("HEAD", manifest_url))
        self.assertEqual(second_headers["authorization"], "Bearer t0k")
        # The index of a multi-platform image is asked for along with a manifest.
        for sent in (first_headers, second_headers):
            self.assertIn("application/vnd.oci.image.index.v1+json", sent["accept"])
            self.assertIn("application/vnd.docker.distribution.manifest.list.v2+json", sent["accept"])

    def test_a_registry_that_asks_for_no_token_answers_the_first_request(self):
        with registry(FakeRegistry(challenge=None)) as fake:
            digest = ci_containers.registry_digest(ImageReference.parse("localhost:5000/base:1"))

        self.assertEqual(digest, DIGEST)
        self.assertEqual([(method, url) for method, url, _ in fake.requests], [
            ("HEAD", "https://localhost:5000/v2/base/manifests/1"),
        ])

    def test_an_answer_without_a_digest_is_refused(self):
        for digest in [None, "", "sha256:abc", "md5:" + "a" * 32]:
            with self.subTest(digest=digest), registry(FakeRegistry(digest=digest)):
                with self.assertRaises(RegistryError):
                    ci_containers.registry_digest(ImageReference.parse("peppybot/python-uv-base"))

    def test_an_error_other_than_a_challenge_is_raised(self):
        def not_found(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", headers(), None)

        with mock.patch.object(ci_containers.urllib.request, "urlopen", not_found):
            with self.assertRaises(urllib.error.HTTPError):
                ci_containers.registry_digest(ImageReference.parse("peppybot/python-uv-base:nope"))


class AnonymousToken(unittest.TestCase):
    def test_access_token_is_read_when_there_is_no_token(self):
        body = json.dumps({"access_token": "acc355"}).encode()
        with registry(FakeRegistry(token_body=body)):
            self.assertEqual(ci_containers.anonymous_token(CHALLENGE), "acc355")

    def test_a_challenge_other_than_a_bearer_token_is_refused(self):
        for challenge in ["", 'Basic realm="registry"', "Bearer service=\"x\""]:
            with self.subTest(challenge=challenge), registry(FakeRegistry()) as fake:
                with self.assertRaises(RegistryError):
                    ci_containers.anonymous_token(challenge)
                self.assertEqual(fake.requests, [])

    def test_an_answer_without_a_token_is_refused(self):
        for body in [b"{}", b'{"token": ""}', b'{"token": 7}', b"[]"]:
            with self.subTest(body=body), registry(FakeRegistry(token_body=body)):
                with self.assertRaises(RegistryError):
                    ci_containers.anonymous_token(CHALLENGE)


class PinnedBaseImage(unittest.TestCase):
    def test_a_tag_is_pinned_to_the_digest_the_registry_gives_it(self):
        with registry(FakeRegistry()):
            self.assertEqual(
                ci_containers.pinned_base_image("peppybot/python-uv-base:latest"),
                f"peppybot/python-uv-base@{DIGEST}",
            )

    def test_a_reference_with_a_digest_is_kept_without_a_request(self):
        pinned = f"peppybot/python-uv-base@{DIGEST}"
        with registry(FakeRegistry()) as fake:
            self.assertEqual(ci_containers.pinned_base_image(pinned), pinned)
        self.assertEqual(fake.requests, [])

    def test_a_registry_that_cannot_be_read_keeps_the_tag_with_a_warning(self):
        failures = [
            urllib.error.URLError("no route"),
            TimeoutError("timed out"),
            urllib.error.HTTPError("https://x", 503, "Service Unavailable", headers(), None),
        ]
        for failure in failures:
            def fail(request, timeout, failure=failure):
                raise failure

            with self.subTest(failure=failure), mock.patch.object(ci_containers.urllib.request, "urlopen", fail):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    base = ci_containers.pinned_base_image("peppybot/python-uv-base:latest")

                self.assertEqual(base, "peppybot/python-uv-base:latest")
                self.assertTrue(stderr.getvalue().startswith(
                    "::warning::cannot read the digest of peppybot/python-uv-base:latest"
                ))

    def test_a_from_line_in_another_form_is_an_error(self):
        with registry(FakeRegistry()), self.assertRaises(ValueError):
            ci_containers.pinned_base_image("peppybot/base:")


DEF_FILE = """\
Bootstrap: docker
From: peppybot/rust-cargo-base:latest

%files
    . /opt/camera

%post
    set -eux
    apt-get update
    apt-get install -y --no-install-recommends libclang-dev

    cd /opt/camera
    cargo build --release
"""


class TestImageDefinition(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.project = Path(directory.name)

    def test_the_base_image_is_named_as_pin_base_gives_it(self):
        (self.project / "apptainer.def").write_text(DEF_FILE)

        definition = ci_containers.test_image_definition(self.project, "unused:latest", lambda base: f"<{base}>")

        self.assertEqual(
            definition,
            "Bootstrap: docker\n"
            "From: <peppybot/rust-cargo-base:latest>\n\n"
            "%post\n"
            "    set -eux\n"
            "    apt-get update\n"
            "    apt-get install -y --no-install-recommends libclang-dev\n"
            "\n",
        )

    def test_a_project_without_a_definition_gets_the_fallback_base(self):
        definition = ci_containers.test_image_definition(
            self.project, "peppybot/python-uv-base:latest", lambda base: f"<{base}>"
        )

        self.assertEqual(definition, "Bootstrap: docker\nFrom: <peppybot/python-uv-base:latest>\n")


class DeriveTestImage(unittest.TestCase):
    """A cached test image serves its definition until the digest of its base
    image moves."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.cache = root / "cache"
        self.project = root / "camera"
        self.project.mkdir()
        (self.project / "apptainer.def").write_text(DEF_FILE)
        environment = mock.patch.dict(
            os.environ, {"CI_CACHE_DIR": str(self.cache), "PEPPY_APPTAINER_DIR": str(root / "apptainer")}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.builds = []

    def build(self, command, **kwargs):
        """Stands in for `apptainer build <image> <definition>`."""
        self.builds.append(Path(command[3]).read_text())
        Path(command[2]).write_text("image")
        return mock.Mock(returncode=0)

    def derive(self, digest):
        def pin(base):
            return ImageReference.parse(base).pinned(digest)

        with mock.patch.object(ci_containers, "pinned_base_image", pin), \
                mock.patch.object(ci_containers.subprocess, "run", self.build), \
                contextlib.redirect_stderr(io.StringIO()):
            return ci_containers.derive_test_image(str(self.project), "unused:latest")

    def test_the_same_digest_reuses_the_cached_image(self):
        first = self.derive(DIGEST)
        second = self.derive(DIGEST)

        self.assertEqual(first, second)
        self.assertEqual(len(self.builds), 1)
        self.assertIn(f"From: peppybot/rust-cargo-base@{DIGEST}\n", self.builds[0])

    def test_a_moved_tag_builds_the_image_again(self):
        first = self.derive(DIGEST)
        second = self.derive(OTHER_DIGEST)

        self.assertNotEqual(first, second)
        self.assertEqual(len(self.builds), 2)
        self.assertIn(f"From: peppybot/rust-cargo-base@{OTHER_DIGEST}\n", self.builds[1])
        self.assertTrue(first.is_file() and second.is_file())


if __name__ == "__main__":
    unittest.main()
