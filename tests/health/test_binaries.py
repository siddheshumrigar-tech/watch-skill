"""Binary resolution: managed bin dir shadows PATH; space-safe paths."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from watch_skill.config import reset_settings
from watch_skill.errors import DependencyError
from watch_skill.health import binaries

_EXE = ".exe" if sys.platform == "win32" else ""


def _fake_exe(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}{_EXE}"
    path.write_bytes(b"fake")
    # shutil.which on POSIX requires the executable bit (first CI run on
    # Linux caught this — Windows doesn't care)
    path.chmod(0o755)
    return path


def test_find_binary_prefers_managed_dir_over_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed = tmp_path / "managed bin with spaces"
    monkeypatch.setenv("WATCHSKILL_BIN_DIR", str(managed))
    reset_settings()

    managed_copy = _fake_exe(managed, "toolx")
    path_dir = tmp_path / "on path"
    _fake_exe(path_dir, "toolx")
    monkeypatch.setenv("PATH", str(path_dir))

    assert binaries.find_binary("toolx") == managed_copy


def test_find_binary_falls_back_to_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WATCHSKILL_BIN_DIR", str(tmp_path / "empty managed"))
    reset_settings()
    path_dir = tmp_path / "system dir with spaces"
    expected = _fake_exe(path_dir, "tooly")
    monkeypatch.setenv("PATH", str(path_dir))

    found = binaries.find_binary("tooly")
    assert found is not None
    assert found.resolve() == expected.resolve()


def test_find_binary_missing_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WATCHSKILL_BIN_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("PATH", str(tmp_path))
    reset_settings()
    assert binaries.find_binary("definitely-not-a-real-binary") is None


def test_require_binary_raises_structured_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WATCHSKILL_BIN_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("PATH", str(tmp_path))
    reset_settings()
    with pytest.raises(DependencyError) as excinfo:
        binaries.require_binary("ghost-tool")
    assert excinfo.value.code == "health.binary_missing"
    assert "doctor" in (excinfo.value.fix or "")


def test_managed_bin_dir_created_on_demand(isolated_settings: Path) -> None:
    bin_dir = binaries.managed_bin_dir(create=True)
    assert bin_dir.is_dir()
    assert bin_dir == isolated_settings / "bin"


def test_prepend_bin_dir_to_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    managed = tmp_path / "managed bin"
    monkeypatch.setenv("WATCHSKILL_BIN_DIR", str(managed))
    monkeypatch.setenv("PATH", "C:\\other" if sys.platform == "win32" else "/usr/bin")
    reset_settings()
    binaries.prepend_bin_dir_to_path()
    import os

    assert os.environ["PATH"].startswith(str(managed))
    before = os.environ["PATH"]
    binaries.prepend_bin_dir_to_path()
    assert os.environ["PATH"] == before  # idempotent


# --- SHA-256 verification of doctor's downloads (D9) -------------------------

import hashlib  # noqa: E402
import json  # noqa: E402

import httpx  # noqa: E402

PAYLOAD = b"#!/bin/sh\necho yt-dlp\n" * 100
GOOD = hashlib.sha256(PAYLOAD).hexdigest()


class _Stream:
    def __init__(self, url: str, status: int, body: bytes) -> None:
        self.status_code, self._body = status, body
        self.request = httpx.Request("GET", url)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("bad", request=self.request,
                                        response=httpx.Response(self.status_code))

    def iter_bytes(self, size: int):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]


def _serve(monkeypatch: pytest.MonkeyPatch, routes: dict[str, tuple[int, bytes]]) -> list[str]:
    fetched: list[str] = []

    def stream(method, url, **kwargs):
        fetched.append(url)
        status, body = next((v for k, v in routes.items() if url.endswith(k)), (404, b""))
        return _Stream(url, status, body)

    monkeypatch.setattr(httpx, "stream", stream)
    return fetched


def _asset() -> str:
    return "yt-dlp.exe" if sys.platform == "win32" else "yt-dlp"


def test_a_matching_checksum_installs_and_records_a_receipt(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Path
) -> None:
    sums = f"{'0' * 64}  yt-dlp_macos\n{GOOD}  {_asset()}\n".encode()
    fetched = _serve(monkeypatch, {"SHA2-256SUMS": (200, sums), f"/{_asset()}": (200, PAYLOAD)})
    dest = binaries.bootstrap_yt_dlp()
    assert dest.read_bytes() == PAYLOAD
    assert fetched[0].endswith("SHA2-256SUMS"), "the checksum is fetched before the binary"
    [receipt] = binaries.download_receipts()
    assert receipt["verified"] is True and receipt["sha256"] == GOOD
    assert receipt["asset"] == _asset() and receipt["checksum_source"].endswith("SHA2-256SUMS")


def test_a_mismatch_refuses_and_leaves_nothing(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Path
) -> None:
    """Regression: the download was installed without any integrity check."""
    sums = f"{'a' * 64}  {_asset()}\n".encode()
    _serve(monkeypatch, {"SHA2-256SUMS": (200, sums), f"/{_asset()}": (200, PAYLOAD)})
    with pytest.raises(DependencyError) as info:
        binaries.bootstrap_yt_dlp()
    assert info.value.code == "health.checksum_mismatch"
    assert info.value.fix
    assert info.value.details["actual"] == GOOD
    bin_dir = binaries.managed_bin_dir()
    assert not (bin_dir / _asset()).exists()
    assert not list(bin_dir.glob("*.part"))
    assert binaries.download_receipts() == []


def test_no_published_checksum_installs_unverified(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Path
) -> None:
    _serve(monkeypatch, {f"/{_asset()}": (200, PAYLOAD)})
    binaries.bootstrap_yt_dlp()
    [receipt] = binaries.download_receipts()
    assert receipt["verified"] is False and receipt["checksum_source"] is None


def test_a_checksum_file_that_does_not_list_the_asset_refuses(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Path
) -> None:
    sums = f"{GOOD}  something-else\n{'b' * 64}  another\n".encode()
    _serve(monkeypatch, {"SHA2-256SUMS": (200, sums), f"/{_asset()}": (200, PAYLOAD)})
    with pytest.raises(DependencyError) as info:
        binaries.bootstrap_yt_dlp()
    assert info.value.code == "health.checksum_unlisted"
    assert not (binaries.managed_bin_dir() / _asset()).exists()


def test_a_checksum_server_error_refuses(
    monkeypatch: pytest.MonkeyPatch, isolated_settings: Path
) -> None:
    _serve(monkeypatch, {"SHA2-256SUMS": (500, b""), f"/{_asset()}": (200, PAYLOAD)})
    with pytest.raises(DependencyError) as info:
        binaries.bootstrap_yt_dlp()
    assert info.value.code == "health.download_failed"


@pytest.mark.parametrize(("text", "asset"), [
    (f"{GOOD}  deno.zip\n", "deno.zip"),
    (f"{GOOD} *ffmpeg-master-latest-linux64-gpl.tar.xz\n{'c' * 64} *other.tar.xz\n",
     "ffmpeg-master-latest-linux64-gpl.tar.xz"),
    (f"{GOOD.upper()}\n", "ffmpeg-release-essentials.zip"),
    ("\nAlgorithm : SHA256\nHash      : " + GOOD.upper()
     + "\nPath      : D:\\a\\deno\\target\\release\\deno-x86_64-pc-windows-msvc.zip\n",
     "deno-x86_64-pc-windows-msvc.zip"),
    ("Algorithm       Hash                                                             Path\n"
     "---------       ----                                                             ----\n"
     f"SHA256          {GOOD.upper()}       D:\\a\\deno\\deno-x86_64-pc-wi...\n",
     "deno-x86_64-pc-windows-msvc.zip"),
], ids=["sha256sum", "star-name", "bare", "filehash-list", "filehash-table"])
def test_the_checksum_parser_reads_each_publisher_format(text: str, asset: str) -> None:
    assert binaries.parse_published_sha256(text, asset) == GOOD


def test_the_parser_does_not_guess_for_an_unnamed_asset() -> None:
    assert binaries.parse_published_sha256(f"{GOOD}  other.zip\n", "deno.zip") is None


def test_doctor_surfaces_unverified_downloads(isolated_settings: Path) -> None:
    from watch_skill.health import doctor

    assert doctor.check_download_integrity().status == "ok"
    receipts = [{"asset": "ffmpeg.zip", "verified": False}, {"asset": "yt-dlp", "verified": True}]
    path = binaries.managed_bin_dir(create=True) / "downloads.json"
    path.write_text(json.dumps(receipts), encoding="utf-8")
    result = doctor.check_download_integrity()
    assert result.status == "warn" and "ffmpeg.zip" in result.message
