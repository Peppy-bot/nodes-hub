"""The node's TLS identity: a per-machine self-signed certificate.

WebXR demands a secure context and public CAs do not issue for LAN addresses,
so a self-signed certificate is the only general option; the operator accepts
it once past the browser interstitial. Generated on first boot and reused so
that acceptance sticks across restarts.
"""

from __future__ import annotations

import datetime
import fcntl
import ipaddress
import os
import socket
from pathlib import Path

from cryptography import x509
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from xr_commander import addresses

# The certificate and key files under the TLS directory.
_CERT_FILE = "cert.pem"
_KEY_FILE = "key.pem"
_VALID_FOR = datetime.timedelta(days=3650)
# Regenerate ahead of expiry so a long-lived install never serves a cert the
# browser rejects outright (an expired cert cannot be clicked through).
_RENEW_MARGIN = datetime.timedelta(days=30)


def _lan_addresses() -> list[addresses.Address]:
    """Every address this host answers on, for the SAN: the page's addresses
    (the ones the startup log prints) and loopback.

    A later address change is harmless: the interstitial covers any mismatch.
    """
    names = {ipaddress.ip_address("127.0.0.1"), *addresses.host_addresses()}
    return sorted(names, key=str)


def _write_private(path: Path, data: bytes) -> None:
    """Owner-only from birth, atomically replaced: no torn or readable window."""
    scratch = path.with_suffix(".tmp")
    # A leftover scratch file would keep its old mode; never reuse one.
    scratch.unlink(missing_ok=True)
    fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise
    os.replace(scratch, path)


def _generate(directory: Path, valid_for: datetime.timedelta) -> tuple[Path, Path]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "xr_commander teleop server")]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    san = x509.SubjectAlternativeName(
        [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())]
        + [x509.IPAddress(a) for a in _lan_addresses()]
    )
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + valid_for)
        .add_extension(san, critical=False)
        # A TLS server leaf, said outright for any validator that checks.
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / _CERT_FILE
    key_path = directory / _KEY_FILE
    # Key first: a die between the writes leaves a mismatched pair, which the
    # reuse check detects and regenerates rather than serving.
    _write_private(
        key_path,
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )
    _write_private(cert_path, certificate.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def _reusable(cert_path: Path, key_path: Path) -> bool:
    """A matched, parseable pair that is nowhere near expiry."""
    certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    public = key.public_key()
    if certificate.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    ) != public.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    ):
        return False
    now = datetime.datetime.now(datetime.timezone.utc)
    return certificate.not_valid_after_utc - now >= _RENEW_MARGIN


def ensure_certificate(
    directory: Path, *, valid_for: datetime.timedelta = _VALID_FOR
) -> tuple[Path, Path]:
    """The (cert, key) pair under `directory`, regenerated when absent,
    unreadable, mismatched, or near expiry. Reuse keeps the browser's
    acceptance valid. Locked: two instances sharing the directory must not
    interleave their writes into a mismatched pair."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)  # exist_ok skips the mode on a pre-existing dir
    cert_path = directory / _CERT_FILE
    key_path = directory / _KEY_FILE
    with open(directory / ".lock", "w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        try:
            if _reusable(cert_path, key_path):
                return cert_path, key_path
        except (ValueError, TypeError, OSError, UnsupportedAlgorithm):
            pass  # missing, unreadable, or unparseable material: replace it
        return _generate(directory, valid_for)
