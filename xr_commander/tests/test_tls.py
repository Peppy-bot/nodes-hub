import datetime

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import ExtendedKeyUsageOID

from xr_commander.tls import ensure_certificate


def test_a_certificate_is_generated_and_then_reused(tmp_path):
    cert, key = ensure_certificate(tmp_path)
    first = cert.read_bytes()
    assert key.stat().st_mode & 0o777 == 0o600
    again, _ = ensure_certificate(tmp_path)
    assert again.read_bytes() == first


def test_the_san_covers_localhost_for_the_adb_reverse_path(tmp_path):
    cert, _ = ensure_certificate(tmp_path)
    parsed = x509.load_pem_x509_certificate(cert.read_bytes())
    san = parsed.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    assert "localhost" in san.value.get_values_for_type(x509.DNSName)


def test_garbage_material_is_replaced_not_served(tmp_path):
    cert, _ = ensure_certificate(tmp_path)
    cert.write_bytes(b"not a certificate")
    replaced, _ = ensure_certificate(tmp_path)
    parsed = x509.load_pem_x509_certificate(replaced.read_bytes())
    assert parsed.not_valid_after_utc > datetime.datetime.now(datetime.timezone.utc)


def test_a_near_expiry_certificate_is_renewed(tmp_path):
    short_cert, _ = ensure_certificate(tmp_path, valid_for=datetime.timedelta(days=1))
    short_bytes = short_cert.read_bytes()
    renewed, _ = ensure_certificate(tmp_path, valid_for=datetime.timedelta(days=3650))
    assert renewed.read_bytes() != short_bytes


def test_the_certificate_is_marked_as_a_server_leaf(tmp_path):
    cert, _ = ensure_certificate(tmp_path)
    parsed = x509.load_pem_x509_certificate(cert.read_bytes())
    constraints = parsed.extensions.get_extension_for_class(x509.BasicConstraints)
    assert constraints.value.ca is False
    usage = parsed.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
    assert ExtendedKeyUsageOID.SERVER_AUTH in usage.value
    # Both key identifiers, so even a strict chain verifier accepts the leaf.
    subject = parsed.extensions.get_extension_for_class(x509.SubjectKeyIdentifier)
    authority = parsed.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier)
    assert authority.value.key_identifier == subject.value.digest


def test_material_lands_owner_only_even_over_a_loose_stale_scratch(tmp_path):
    stale = tmp_path / "key.tmp"
    stale.touch()
    stale.chmod(0o644)
    _, key = ensure_certificate(tmp_path)
    assert key.stat().st_mode & 0o777 == 0o600


def test_the_directory_is_created_owner_only(tmp_path):
    ensure_certificate(tmp_path / "tls")
    assert (tmp_path / "tls").stat().st_mode & 0o777 == 0o700


def test_a_mismatched_key_and_cert_pair_is_regenerated(tmp_path):
    cert, key = ensure_certificate(tmp_path)
    old_cert = cert.read_bytes()
    # Simulate a torn write: the key is replaced, the cert is not.
    other = tmp_path / "other"
    _, other_key = ensure_certificate(other)
    key.write_bytes(other_key.read_bytes())
    ensure_certificate(tmp_path)
    parsed = x509.load_pem_x509_certificate(cert.read_bytes())
    loaded = serialization.load_pem_private_key(key.read_bytes(), password=None)
    assert parsed.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    ) == loaded.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    assert cert.read_bytes() != old_cert
