"""The `failed_request` check: a page loads without requests failing.

Unit tests drive a scripted fake `playwright.sync_api`; one real test runs
only when a Chromium is already installed, against a loopback server whose
page fetches a path that 404s.
"""
from __future__ import annotations

import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from watch_skill.verify import Check, CheckContext, CheckStatus
from watch_skill.verify.checks import run_check

ORIGIN = "http://127.0.0.1:8765"


class _Request:
    def __init__(self, url: str, failure: str) -> None:
        self.url, self.failure = url, failure


class _Response:
    def __init__(self, url: str, status: int) -> None:
        self.url, self.status = url, status


def _fake_playwright(monkeypatch: pytest.MonkeyPatch, events: list, *,
                     goto_error: str | None = None) -> dict:
    seen: dict = {"launched": 0, "closed": 0}

    class Page:
        def __init__(self) -> None:
            self.handlers: dict = {}

        def on(self, name, handler) -> None:
            self.handlers[name] = handler

        def goto(self, url, **kwargs) -> None:
            seen["url"] = url
            if goto_error:
                raise RuntimeError(goto_error)
            for event in events:
                name = "requestfailed" if isinstance(event, _Request) else "response"
                self.handlers[name](event)

        def wait_for_timeout(self, ms) -> None:
            seen["waited"] = ms

    class Browser:
        def new_page(self):
            return Page()

        def close(self) -> None:
            seen["closed"] += 1

    class Chromium:
        def launch(self, **kwargs):
            seen["launched"] += 1
            return Browser()

    class Manager:
        def __enter__(self):
            return types.SimpleNamespace(chromium=Chromium())

        def __exit__(self, *exc) -> None:
            return None

    package = types.ModuleType("playwright")
    api = types.ModuleType("playwright.sync_api")
    api.sync_playwright = Manager
    package.sync_api = api
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", api)
    return seen


def _run(tmp_path: Path, origin: str = ORIGIN, **params):
    return run_check(
        Check(id="n", type="failed_request", params={"url": f"{origin}/", **params}),
        CheckContext(working_dir=str(tmp_path), allowed_origins=[origin]),
    )


def test_a_clean_load_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _fake_playwright(monkeypatch, [_Response(f"{ORIGIN}/", 200),
                                          _Response(f"{ORIGIN}/app.js", 304)])
    result = _run(tmp_path, wait_ms=10)
    assert result.status is CheckStatus.PASS, result.summary
    assert result.observed == {"failures": [], "failure_count": 0}
    assert seen["waited"] == 10 and seen["closed"] == 1


def test_status_and_network_failures_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_playwright(monkeypatch, [
        _Response(f"{ORIGIN}/api/cart", 500),
        _Request(f"{ORIGIN}/api/price", "net::ERR_CONNECTION_REFUSED"),
    ])
    result = _run(tmp_path)
    assert result.status is CheckStatus.FAIL
    assert result.observed["failures"] == [
        {"url": f"{ORIGIN}/api/cart", "status": 500, "failure": None},
        {"url": f"{ORIGIN}/api/price", "status": None, "failure": "net::ERR_CONNECTION_REFUSED"},
    ]
    assert _run(tmp_path, max_failures=2).status is CheckStatus.PASS
    assert _run(tmp_path, min_status=501).observed["failure_count"] == 1


def test_include_and_exclude_filter_urls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_playwright(monkeypatch, [
        _Response(f"{ORIGIN}/api/cart", 500),
        _Response(f"{ORIGIN}/favicon.ico", 404),
    ])
    assert _run(tmp_path, exclude=[r"favicon"]).observed["failure_count"] == 1
    only_api = _run(tmp_path, include=[r"/api/"], exclude=[r"/cart$"])
    assert only_api.status is CheckStatus.PASS


def test_a_page_that_will_not_load_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_playwright(monkeypatch, [], goto_error="net::ERR_NAME_NOT_RESOLVED")
    result = _run(tmp_path, include=[r"nothing-matches-this"])
    assert result.status is CheckStatus.FAIL
    assert "navigation failed" in result.observed["failures"][0]["failure"]


def test_without_playwright_it_is_inconclusive(monkeypatch: pytest.MonkeyPatch,
                                               tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    assert _run(tmp_path).status is CheckStatus.INCONCLUSIVE


def test_an_origin_outside_the_contract_is_refused(tmp_path: Path) -> None:
    result = run_check(
        Check(id="n", type="failed_request", params={"url": "https://example.com/"}),
        CheckContext(working_dir=str(tmp_path), allowed_origins=[ORIGIN]),
    )
    assert result.status is CheckStatus.ERROR
    assert result.error["code"] == "verify.check_refused"


def test_a_bad_regex_is_malformed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_playwright(monkeypatch, [])
    result = _run(tmp_path, include=["("])
    assert result.error["code"] == "verify.check_malformed"


class _Site(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        if self.path == "/":
            body = b"<html><body><script>fetch('/missing')</script>ok</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args) -> None:
        return


def test_a_real_browser_reports_the_404(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    try:
        with sync_api.sync_playwright() as p:
            p.chromium.launch(headless=True).close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no Chromium is installed for Playwright: {str(exc)[:120]}")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        result = _run(tmp_path, origin=origin, include=[r"/missing$"], wait_ms=1000)
    finally:
        server.shutdown()
    assert result.status is CheckStatus.FAIL, result.summary
    assert result.observed["failures"][0]["status"] == 404
