"""A packaged build points OpenSSL at the certifi bundle only when OpenSSL has no
certificates of its own. See learning.md "A packaged build has no CA certificates of its own"."""
import os
import ssl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import certifi  # noqa: E402
import mesa_shared  # noqa: E402

_NO_PATHS = ssl.DefaultVerifyPaths(None, None, "SSL_CERT_FILE", "/nowhere/cert.pem", "SSL_CERT_DIR", "/nowhere/certs")


def test_sets_certifi_when_openssl_has_no_certificates(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: _NO_PATHS)
    monkeypatch.setattr(sys, "platform", "darwin")
    mesa_shared.use_bundled_ca_certificates()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()
    assert os.path.isfile(os.environ["SSL_CERT_FILE"])


def test_leaves_openssl_alone_when_it_has_certificates(monkeypatch, tmp_path):
    cafile = tmp_path / "cert.pem"
    cafile.write_text("")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(ssl, "get_default_verify_paths",
                        lambda: ssl.DefaultVerifyPaths(str(cafile), None, "SSL_CERT_FILE", str(cafile), "SSL_CERT_DIR", None))
    monkeypatch.setattr(sys, "platform", "darwin")
    mesa_shared.use_bundled_ca_certificates()
    assert "SSL_CERT_FILE" not in os.environ


def test_respects_an_operator_setting(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/operator/ca.pem")
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: _NO_PATHS)
    monkeypatch.setattr(sys, "platform", "darwin")
    mesa_shared.use_bundled_ca_certificates()
    assert os.environ["SSL_CERT_FILE"] == "/operator/ca.pem"


def test_windows_is_left_alone(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: _NO_PATHS)
    monkeypatch.setattr(sys, "platform", "win32")
    mesa_shared.use_bundled_ca_certificates()
    assert "SSL_CERT_FILE" not in os.environ
