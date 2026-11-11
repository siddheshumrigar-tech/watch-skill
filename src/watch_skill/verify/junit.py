"""The ``test_report`` check: a JUnit XML report says the tests passed.

Counts come from the ``<testcase>`` elements, not the suite's ``tests=`` /
``failures=`` attributes, because those attributes are written by whatever
produced the file and are the easiest thing in it to get wrong. A report
with a DOCTYPE or an entity declaration is refused outright: there is no
legitimate reason for one in a test report, and entity expansion is how XML
parsers get attacked.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405 - DOCTYPE/ENTITY refused before parsing
from typing import Any

_DEFAULT_MAX_BYTES = 20 * 1024 * 1024
_REFUSED_MARKUP = (b"<!doctype", b"<!entity")


def _outcome(case: ET.Element) -> str:
    """``failed``, ``errored``, ``skipped`` or ``passed`` for one test case."""
    tags = {child.tag for child in case}
    if "failure" in tags:
        return "failed"
    if "error" in tags:
        return "errored"
    if "skipped" in tags:
        return "skipped"
    return "passed"


def _names(case: ET.Element) -> set[str]:
    """Every spelling a contract may use for this case."""
    name, cls = case.get("name", ""), case.get("classname", "")
    names = {name}
    if cls:
        names |= {f"{cls}.{name}", f"{cls}::{name}"}
    return names


def _parse(raw: bytes) -> list[ET.Element]:
    lowered = raw.lower()
    if any(marker in lowered for marker in _REFUSED_MARKUP):
        raise PermissionError("the report declares a DOCTYPE or an ENTITY, which is refused")
    try:
        root = ET.fromstring(raw)  # noqa: S314 - see _REFUSED_MARKUP
    except ET.ParseError as exc:
        raise ValueError(f"the report is not well-formed XML: {exc}") from exc
    if root.tag not in ("testsuites", "testsuite"):
        raise ValueError(f"expected a <testsuites> or <testsuite> root, got <{root.tag}>")
    return list(root.iter("testcase"))


def _required(cases: list[tuple[set[str], str]], required: list[str]) -> tuple[list[str],
                                                                                list[str]]:
    """(missing, failed_or_skipped) among the required test names."""
    missing, bad = [], []
    for wanted in required:
        outcomes = [outcome for names, outcome in cases if wanted in names]
        if not outcomes:
            missing.append(wanted)
        elif any(outcome != "passed" for outcome in outcomes):
            bad.append(wanted)
    return missing, bad


def check_test_report(check: Any, ctx: Any) -> tuple[Any, Any, Any, str]:
    """PASS when the report's counts meet the thresholds and every required
    test passed. No report, or a report with no test cases, is INCONCLUSIVE."""
    from watch_skill.verify.checks import CheckStatus, _resolve_within, _shown  # noqa: PLC0415

    params = check.params
    path = _resolve_within(str(params["path"]), ctx)
    max_failed, min_passed = int(params.get("max_failed", 0)), int(params.get("min_passed", 1))
    required = [str(name) for name in params.get("required_tests", [])]
    expected = {"max_failed": max_failed, "min_passed": min_passed, "required_tests": required}
    if not path.is_file():
        return (CheckStatus.INCONCLUSIVE, expected, None,
                f"no test report at {_shown(path, ctx)}")
    max_bytes = int(params.get("max_bytes", _DEFAULT_MAX_BYTES))
    if path.stat().st_size > max_bytes:
        raise ValueError(f"the report is larger than max_bytes ({max_bytes})")
    elements = _parse(path.read_bytes())
    if not elements:
        return (CheckStatus.INCONCLUSIVE, expected, {"total": 0},
                "the report contains no test cases")
    cases = [(_names(case), _outcome(case)) for case in elements]
    counts = {kind: sum(1 for _, o in cases if o == kind)
              for kind in ("passed", "failed", "errored", "skipped")}
    missing, bad = _required(cases, required)
    observed = {**counts, "total": len(cases),
                "missing_required": missing, "failed_required": bad}
    broken = counts["failed"] + counts["errored"]
    ok = broken <= max_failed and counts["passed"] >= min_passed and not missing and not bad
    return (
        CheckStatus.PASS if ok else CheckStatus.FAIL, expected, observed,
        f"{counts['passed']} passed, {broken} failed, {counts['skipped']} skipped"
        + (f"; required missing: {missing}" if missing else "")
        + (f"; required not passing: {bad}" if bad else ""),
    )


RUNNERS = {"test_report": check_test_report}
