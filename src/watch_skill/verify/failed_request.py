"""The ``failed_request`` check: loading a page makes no request that fails.

A page can render while the API call behind it returned 500, and nothing on
screen says so. This opens the verifier's own browser on the contract's URL,
records every request that failed at the network level and every response
at or above ``min_status``, and fails when there are more than
``max_failures``. It never clicks, fills or evaluates; loading is the whole
interaction. The same origin allowlist and address checks as ``browser_dom``
apply.
"""
from __future__ import annotations

import re
from typing import Any

_MAX_REPORTED = 20


def _patterns(params: dict[str, Any], key: str) -> list[re.Pattern[str]]:
    try:
        return [re.compile(str(p)) for p in params.get(key, []) or []]
    except re.error as exc:
        raise ValueError(f"{key} has an invalid regex: {exc}") from exc


def _observe(page: Any, url: str, keep: Any, min_status: int,
             timeout_ms: int, wait_ms: int) -> list[dict[str, Any]]:
    """Load ``url`` and return the failures seen while it settled."""
    failures: list[dict[str, Any]] = []

    def on_failed(request: Any) -> None:
        if keep(request.url):
            failures.append({"url": request.url[:200], "status": None,
                             "failure": str(getattr(request, "failure", None) or "failed")})

    def on_response(response: Any) -> None:
        if response.status >= min_status and keep(response.url):
            failures.append({"url": response.url[:200], "status": response.status,
                             "failure": None})

    page.on("requestfailed", on_failed)
    page.on("response", on_response)
    try:
        page.goto(url, wait_until="load", timeout=timeout_ms)
    except Exception as exc:  # noqa: BLE001 - a page that will not load is a failure
        failures.append({"url": url[:200], "status": None,
                         "failure": f"navigation failed: {str(exc)[:160]}"})
        return failures
    page.wait_for_timeout(wait_ms)
    return failures


def check_failed_request(check: Any, ctx: Any) -> tuple[Any, Any, Any, str]:
    """PASS when at most ``max_failures`` requests failed while the page loaded.

    Without Playwright this is INCONCLUSIVE: no browser ran, so nothing was
    observed to succeed.
    """
    from watch_skill.verify.checks import (  # noqa: PLC0415
        PLAYWRIGHT_INSTALL_HINT,
        CheckStatus,
        _assert_public_origin,
        launch_verifier_browser,
    )

    params = check.params
    url = str(params["url"])
    _assert_public_origin(url, ctx)
    include, exclude = _patterns(params, "include"), _patterns(params, "exclude")
    min_status, max_failures = int(params.get("min_status", 400)), int(params.get("max_failures", 0))
    wait_ms, timeout_ms = int(params.get("wait_ms", 2000)), int(params.get("timeout_ms", 15_000))
    expected = {"max_failures": max_failures, "min_status": min_status}

    def keep(request_url: str) -> bool:
        if include and not any(p.search(request_url) for p in include):
            return False
        return not any(p.search(request_url) for p in exclude)

    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except ImportError:
        return (CheckStatus.INCONCLUSIVE, expected, None,
                f"playwright is not installed, so requests cannot be observed; "
                f"{PLAYWRIGHT_INSTALL_HINT}")
    with sync_playwright() as p:
        browser = launch_verifier_browser(p)
        try:
            failures = _observe(browser.new_page(), url, keep, min_status, timeout_ms, wait_ms)
        finally:
            browser.close()
    observed = {"failures": failures[:_MAX_REPORTED], "failure_count": len(failures)}
    ok = len(failures) <= max_failures
    first = f"; first: {failures[0]['url']}" if failures else ""
    return (CheckStatus.PASS if ok else CheckStatus.FAIL, expected, observed,
            f"{len(failures)} failed request(s) while loading the page{first}")


RUNNERS = {"failed_request": check_failed_request}
