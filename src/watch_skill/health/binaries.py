"""Locate and bootstrap the external binaries Watch Skill depends on.

Lookup order is managed-bin-dir first, then PATH: when the self-healing
updater refreshes a binary, the refreshed copy must win over a stale system
install. All paths are handled via :class:`pathlib.Path` and quoted-safe
(both this repo and its reference live in directories with spaces).
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

from watch_skill.config import get_settings
from watch_skill.errors import DependencyError

YT_DLP_RELEASE_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/"
FFMPEG_PORTABLE_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

# Static builds for the platforms gyan.dev does not cover. BtbN publishes
# self-contained Linux tarballs on a stable "latest" tag; evermeet.cx is the
# long-standing source for notarized macOS binaries and ships ffmpeg and
# ffprobe as separate downloads.
FFMPEG_LINUX_TARBALL_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-{arch}-gpl.tar.xz"
)
FFMPEG_LINUX_CHECKSUMS_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/checksums.sha256"
)
FFMPEG_MACOS_ZIP_URL = "https://evermeet.cx/ffmpeg/getrelease/{tool}/zip"

_WINDOWS = sys.platform == "win32"
_MACOS = sys.platform == "darwin"


def managed_bin_dir(create: bool = False) -> Path:
    """The directory where Watch Skill keeps self-downloaded binaries."""
    bin_dir = get_settings().resolved_bin_dir
    if create:
        bin_dir.mkdir(parents=True, exist_ok=True)
    return bin_dir


def _exe_name(name: str) -> str:
    if _WINDOWS and not name.lower().endswith(".exe"):
        return f"{name}.exe"
    return name


def find_binary(name: str) -> Path | None:
    """Resolve ``name`` to an executable path, or ``None`` if absent.

    Checks the managed bin dir first so self-healed copies shadow stale
    system installs, then falls back to PATH.
    """
    managed = managed_bin_dir() / _exe_name(name)
    if managed.is_file():
        return managed
    found = shutil.which(name)
    return Path(found) if found else None


def require_binary(name: str) -> Path:
    """Like :func:`find_binary` but raises a structured error when missing."""
    path = find_binary(name)
    if path is None:
        raise DependencyError(
            f"required binary '{name}' was not found",
            code="health.binary_missing",
            fix="run `watch-skill doctor` to bootstrap missing dependencies",
            details={"binary": name},
        )
    return path


def _download_failed(url: str, exc: Exception) -> DependencyError:
    return DependencyError(
        f"download failed for {url}: {exc}",
        code="health.download_failed",
        fix="check network connectivity and retry `watch-skill doctor`",
        details={"url": url},
    )


def _stream_to(url: str, dest: Path, timeout: float, expected_sha256: str | None) -> str:
    """Stream ``url`` to ``dest`` atomically and return its SHA-256.

    The hash is taken while streaming, so a mismatching file never reaches
    ``dest``: the ``.part`` file is deleted and nothing is installed.
    """
    import hashlib  # noqa: PLC0415

    import httpx  # lazy: keeps `watch-skill --help` under the cold-start budget

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(1024 * 256):
                    digest.update(chunk)
                    fh.write(chunk)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise _download_failed(url, exc) from exc
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise DependencyError(
            f"{url} does not match the SHA-256 its publisher lists",
            code="health.checksum_mismatch",
            fix="nothing was installed; retry `watch-skill doctor` later, and if "
            "it keeps failing install the tool with your package manager and "
            "report the mismatch",
            details={"url": url, "expected": expected_sha256.lower(), "actual": actual},
        )
    tmp.replace(dest)
    return actual


def _download_file(url: str, dest: Path, timeout: float = 600.0,
                   expected_sha256: str | None = None) -> Path:
    """Stream ``url`` to ``dest`` (atomic), refusing a SHA-256 mismatch."""
    _stream_to(url, dest, timeout, expected_sha256)
    return dest


_SHA256_RE = re.compile(r"\b[0-9a-fA-F]{64}\b")
_CHECKSUM_MAX_BYTES = 1024 * 1024


def parse_published_sha256(text: str, asset_name: str) -> str | None:
    """The hash for ``asset_name`` in a publisher's checksum file, or None.

    Accepts ``sha256sum`` output (``<hash>  name`` / ``<hash> *name``), the
    table PowerShell's ``Get-FileHash`` prints, and a file that holds one
    bare hash for a single asset.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    wanted = asset_name.lower()
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and _SHA256_RE.fullmatch(parts[0]):
            name = parts[-1].lstrip("*").replace("\\", "/").rsplit("/", 1)[-1]
            if name.lower() == wanted:
                return parts[0].lower()
    hashes = _SHA256_RE.findall(text)
    if len(hashes) != 1:
        return None
    # One hash left. It is this asset's when the file says so (Get-FileHash
    # prints the path beside or below it), when the file is nothing but the
    # hash, or when it is a Get-FileHash record whose path column was cut.
    bare = len(lines) == 1 and lines[0].split() == [hashes[0]]
    filehash = lines[0].lower().startswith("algorithm")
    if wanted in text.lower() or bare or filehash:
        return hashes[0].lower()
    return None


def _published_sha256(checksum_url: str, asset_name: str) -> str | None:
    """Fetch the publisher's checksum for ``asset_name``.

    None only when the checksum file does not exist (404): that download is
    recorded as unverified. Any other failure, or a file that does not list
    the asset, refuses the download.
    """
    import httpx  # noqa: PLC0415

    try:
        with httpx.stream("GET", checksum_url, follow_redirects=True, timeout=60.0) as resp:
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            body = bytearray()
            for chunk in resp.iter_bytes(64 * 1024):
                body.extend(chunk)
                if len(body) > _CHECKSUM_MAX_BYTES:
                    raise _download_failed(checksum_url, ValueError("checksum file over 1 MB"))
    except httpx.HTTPError as exc:
        raise _download_failed(checksum_url, exc) from exc
    found = parse_published_sha256(body.decode("utf-8", "replace"), asset_name)
    if found is None:
        raise DependencyError(
            f"{checksum_url} does not list {asset_name}",
            code="health.checksum_unlisted",
            fix="the publisher's checksum file changed shape; nothing was "
            "installed. Install the tool with your package manager instead",
            details={"checksum_url": checksum_url, "asset": asset_name},
        )
    return found


def _record_download(asset: str, url: str, sha256: str, verified: bool,
                     checksum_source: str | None) -> None:
    """Append a receipt to ``<bin_dir>/downloads.json`` (last 100 kept)."""
    import json  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    path = managed_bin_dir(create=True) / "downloads.json"
    try:
        receipts = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(receipts, list):
            receipts = []
    except (OSError, ValueError):
        receipts = []
    receipts.append({"asset": asset, "url": url, "sha256": sha256, "verified": verified,
                     "checksum_source": checksum_source,
                     "at": datetime.now(UTC).isoformat(timespec="seconds")})
    path.write_text(json.dumps(receipts[-100:], indent=2), encoding="utf-8")


def download_receipts() -> list[dict]:
    """Every recorded download, oldest first; empty when none were made."""
    import json  # noqa: PLC0415

    path = managed_bin_dir() / "downloads.json"
    try:
        receipts = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return receipts if isinstance(receipts, list) else []


def _fetch_verified(url: str, dest: Path, asset: str, checksum_url: str | None) -> Path:
    """Download ``url`` checked against its publisher's SHA-256, with a receipt.

    ``checksum_url=None`` means the source publishes no checksum; that is
    recorded as ``verified: false`` and surfaced by doctor rather than
    refused, because refusing would leave that platform with no bootstrap.
    """
    import hashlib  # noqa: PLC0415

    expected = _published_sha256(checksum_url, asset) if checksum_url else None
    _download_file(url, dest, expected_sha256=expected)
    digest = hashlib.sha256()
    with dest.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    _record_download(asset, url, digest.hexdigest(), expected is not None,
                     checksum_url if expected is not None else None)
    return dest


def _make_executable(path: Path) -> None:
    if not _WINDOWS:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def bootstrap_yt_dlp() -> Path:
    """Download the standalone yt-dlp binary into the managed bin dir.

    The standalone binary supports ``yt-dlp -U`` self-update, which the
    self-healing acquisition path relies on.
    """
    asset = "yt-dlp.exe" if _WINDOWS else "yt-dlp"
    dest = managed_bin_dir(create=True) / asset
    _fetch_verified(YT_DLP_RELEASE_URL + asset, dest, asset,
                    YT_DLP_RELEASE_URL + "SHA2-256SUMS")
    _make_executable(dest)
    return dest


def _linux_arch_tag() -> str:
    """BtbN's asset suffix for this CPU."""
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "linuxarm64"
    if machine in ("x86_64", "amd64"):
        return "linux64"
    raise DependencyError(
        f"no portable ffmpeg build for {machine}",
        code="health.unsupported_platform",
        fix="install ffmpeg with your package manager: apt install ffmpeg",
    )


def _bootstrap_ffmpeg_linux(bin_dir: Path) -> tuple[Path, Path]:
    """Extract ffmpeg + ffprobe from BtbN's static tarball."""
    url = FFMPEG_LINUX_TARBALL_URL.format(arch=_linux_arch_tag())
    archive = bin_dir / "ffmpeg-static.tar.xz"
    _fetch_verified(url, archive, url.rsplit("/", 1)[-1], FFMPEG_LINUX_CHECKSUMS_URL)
    found: dict[str, Path | None] = {"ffmpeg": None, "ffprobe": None}
    try:
        with tarfile.open(archive, mode="r:xz") as tf:
            for member in tf.getmembers():
                leaf = member.name.rsplit("/", 1)[-1]
                if leaf in found and found[leaf] is None and member.isfile():
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    target = bin_dir / leaf
                    with src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    target.chmod(0o755)
                    found[leaf] = target
    finally:
        archive.unlink(missing_ok=True)
    missing = [name for name, path in found.items() if path is None]
    if missing:
        raise DependencyError(
            f"portable ffmpeg tarball did not contain: {', '.join(missing)}",
            code="health.bootstrap_failed",
            fix="install ffmpeg with your package manager: apt install ffmpeg",
        )
    return found["ffmpeg"], found["ffprobe"]  # type: ignore[return-value]


def _bootstrap_ffmpeg_macos(bin_dir: Path) -> tuple[Path, Path]:
    """Fetch ffmpeg and ffprobe from evermeet.cx (one zip each)."""
    paths: dict[str, Path] = {}
    for tool in ("ffmpeg", "ffprobe"):
        zip_path = bin_dir / f"{tool}.zip"
        # evermeet.cx publishes no checksum file: recorded as unverified.
        _fetch_verified(FFMPEG_MACOS_ZIP_URL.format(tool=tool), zip_path,
                        f"{tool}.zip", None)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                member = next(
                    (m for m in zf.namelist() if m.rsplit("/", 1)[-1] == tool), None
                )
                if member is None:
                    raise DependencyError(
                        f"{tool} zip did not contain a {tool} binary",
                        code="health.bootstrap_failed",
                        fix="install ffmpeg with Homebrew: brew install ffmpeg",
                    )
                target = bin_dir / tool
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                target.chmod(0o755)
                paths[tool] = target
        finally:
            zip_path.unlink(missing_ok=True)
    return paths["ffmpeg"], paths["ffprobe"]


def bootstrap_ffmpeg_portable() -> tuple[Path, Path]:
    """Download a self-contained ffmpeg + ffprobe into the managed bin dir.

    Returns (ffmpeg_path, ffprobe_path). Each platform needs its own source:
    gyan.dev for Windows, BtbN's static tarballs for Linux, evermeet.cx for
    macOS. This used to be Windows-only, which meant `doctor` failed outright
    on the two platforms most contributors are on.
    """
    bin_dir = managed_bin_dir(create=True)
    if _MACOS:
        return _bootstrap_ffmpeg_macos(bin_dir)
    if not _WINDOWS:
        return _bootstrap_ffmpeg_linux(bin_dir)
    zip_path = bin_dir / "ffmpeg-release-essentials.zip"
    _fetch_verified(FFMPEG_PORTABLE_ZIP_URL, zip_path, zip_path.name,
                    FFMPEG_PORTABLE_ZIP_URL + ".sha256")
    wanted = {"ffmpeg.exe": None, "ffprobe.exe": None}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            leaf = member.rsplit("/", 1)[-1].lower()
            if leaf in wanted and wanted[leaf] is None:
                target = bin_dir / leaf
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                wanted[leaf] = target
    zip_path.unlink(missing_ok=True)
    missing = [name for name, path in wanted.items() if path is None]
    if missing:
        raise DependencyError(
            f"portable ffmpeg zip did not contain: {', '.join(missing)}",
            code="health.bootstrap_failed",
            fix="install ffmpeg manually: winget install Gyan.FFmpeg",
        )
    return wanted["ffmpeg.exe"], wanted["ffprobe.exe"]  # type: ignore[return-value]


# tesseract's own language files, which are data rather than a binary and so
# can be fetched anywhere. `tessdata_fast` is the maintained integer model
# set — a few MB per script against ~15 MB for `tessdata_best`, and the gap
# it closes here is 0%, not a percentage point.
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang}.traineddata"


def tessdata_dir(create: bool = False) -> Path:
    """Where Watch Skill keeps traineddata it downloaded itself."""
    path = managed_bin_dir(create=create) / "tessdata"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def bootstrap_traineddata(lang: str) -> Path:
    """Fetch one tesseract language file into the managed tessdata dir.

    The binary cannot be installed without a package manager on Linux or
    macOS, but this half always can — and it is the half that is usually
    missing. `apt install tesseract-ocr` ships English and leaves Lao,
    Khmer, and Myanmar in separate packages most people never learn about,
    so a machine with tesseract present still reads those at 0%.
    """
    dest = tessdata_dir(create=True) / f"{lang}.traineddata"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    _download_file(TESSDATA_URL.format(lang=lang), dest)
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise DependencyError(
            f"downloaded {lang}.traineddata was empty",
            code="health.bootstrap_failed",
            fix=f"check that '{lang}' exists in "
            "https://github.com/tesseract-ocr/tessdata_fast",
        )
    return dest


def bootstrap_deno() -> Path:
    """Get a deno binary into the managed bin dir (copy or portable zip).

    yt-dlp needs a JavaScript runtime for YouTube n-sig decryption; without
    one YouTube throttles downloads to a crawl. Prefers copying an existing
    system deno (e.g. winget's, which lands off-PATH for running processes);
    falls back to the official release zip.
    """
    bin_dir = managed_bin_dir(create=True)
    dest = bin_dir / _exe_name("deno")
    existing = shutil.which("deno")
    if existing is None and _WINDOWS:
        winget_root = (
            Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        )
        for candidate in winget_root.glob("DenoLand.Deno*/deno.exe"):
            existing = str(candidate)
            break
    if existing:
        shutil.copy2(existing, dest)
        return dest
    platform_tag = "x86_64-pc-windows-msvc" if _WINDOWS else (
        "aarch64-apple-darwin" if sys.platform == "darwin" else "x86_64-unknown-linux-gnu"
    )
    url = f"https://github.com/denoland/deno/releases/latest/download/deno-{platform_tag}.zip"
    zip_path = bin_dir / "deno.zip"
    _fetch_verified(url, zip_path, url.rsplit("/", 1)[-1], url + ".sha256sum")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.rsplit("/", 1)[-1].lower().startswith("deno"):
                with zf.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                break
    zip_path.unlink(missing_ok=True)
    if not dest.is_file():
        raise DependencyError(
            "deno bootstrap failed",
            code="health.bootstrap_failed",
            fix="install deno manually: winget install DenoLand.Deno",
        )
    _make_executable(dest)
    return dest


def prepend_bin_dir_to_path() -> None:
    """Make managed binaries visible to subprocesses spawned via bare names."""
    bin_dir = str(managed_bin_dir())
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in parts:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
