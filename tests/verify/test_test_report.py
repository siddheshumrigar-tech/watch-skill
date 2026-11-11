"""The `test_report` check reads JUnit XML the way pytest writes it.

Counts come from the test cases, not the suite attributes; required tests
are matched by `classname.name`, `classname::name` or bare name; a report
that declares a DOCTYPE or an ENTITY is refused.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from watch_skill.verify import Check, CheckContext, CheckStatus
from watch_skill.verify.checks import run_check

GREEN = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="3" failures="0" errors="0" skipped="1">
  <testcase classname="tests.test_cart" name="test_total" time="0.01"/>
  <testcase classname="tests.test_cart" name="test_tax" time="0.01"/>
  <testcase classname="tests.test_cart" name="test_slow" time="0.0">
    <skipped message="slow"/>
  </testcase>
</testsuite></testsuites>
"""

# The suite attributes claim success; the cases say otherwise.
LYING = """<testsuite name="pytest" tests="3" failures="0" errors="0">
  <testcase classname="tests.test_cart" name="test_total"/>
  <testcase classname="tests.test_cart" name="test_tax"><failure message="1 != 2"/></testcase>
  <testcase classname="tests.test_cart" name="test_boot"><error message="boom"/></testcase>
</testsuite>
"""


def _run(tmp_path: Path, xml: str | None, **params):
    if xml is not None:
        (tmp_path / "report.xml").write_text(xml, encoding="utf-8")
    return run_check(
        Check(id="r", type="test_report", params={"path": "report.xml", **params}),
        CheckContext(working_dir=str(tmp_path), allowed_roots=[str(tmp_path)]),
    )


def test_a_green_report_passes_with_counts(tmp_path: Path) -> None:
    result = _run(tmp_path, GREEN)
    assert result.status is CheckStatus.PASS, result.summary
    assert result.observed == {"passed": 2, "failed": 0, "errored": 0, "skipped": 1,
                               "total": 3, "missing_required": [], "failed_required": []}


def test_counts_come_from_the_cases_not_the_attributes(tmp_path: Path) -> None:
    result = _run(tmp_path, LYING)
    assert result.status is CheckStatus.FAIL
    assert (result.observed["failed"], result.observed["errored"]) == (1, 1)
    assert _run(tmp_path, LYING, max_failed=2).status is CheckStatus.PASS


def test_min_passed(tmp_path: Path) -> None:
    assert _run(tmp_path, GREEN, min_passed=3).status is CheckStatus.FAIL


@pytest.mark.parametrize("name", [
    "tests.test_cart.test_total", "tests.test_cart::test_total", "test_total",
])
def test_required_tests_match_every_spelling(tmp_path: Path, name: str) -> None:
    assert _run(tmp_path, GREEN, required_tests=[name]).status is CheckStatus.PASS


def test_a_required_test_that_is_missing_failed_or_skipped_fails(tmp_path: Path) -> None:
    missing = _run(tmp_path, GREEN, required_tests=["test_refund"])
    assert missing.status is CheckStatus.FAIL
    assert missing.observed["missing_required"] == ["test_refund"]
    skipped = _run(tmp_path, GREEN, required_tests=["test_slow"])
    assert skipped.status is CheckStatus.FAIL
    assert skipped.observed["failed_required"] == ["test_slow"]
    failed = _run(tmp_path, LYING, max_failed=5, required_tests=["test_tax"])
    assert failed.status is CheckStatus.FAIL


def test_no_report_or_no_cases_is_inconclusive(tmp_path: Path) -> None:
    assert _run(tmp_path, None).status is CheckStatus.INCONCLUSIVE
    empty = _run(tmp_path, '<testsuite name="pytest" tests="0"/>')
    assert empty.status is CheckStatus.INCONCLUSIVE


def test_doctype_and_entities_are_refused(tmp_path: Path) -> None:
    bomb = ('<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]>'
            '<testsuite><testcase name="t"/></testsuite>')
    result = _run(tmp_path, bomb)
    assert result.status is CheckStatus.ERROR
    assert result.error["code"] == "verify.check_refused"


def test_malformed_xml_and_wrong_roots_are_malformed(tmp_path: Path) -> None:
    for xml in ("<testsuite><testcase", "<html><body/></html>"):
        result = _run(tmp_path, xml)
        assert result.status is CheckStatus.ERROR
        assert result.error["code"] == "verify.check_malformed"


def test_an_oversized_report_is_refused(tmp_path: Path) -> None:
    result = _run(tmp_path, GREEN, max_bytes=10)
    assert result.status is CheckStatus.ERROR


def test_a_report_outside_the_roots_is_refused(tmp_path: Path) -> None:
    outside = tmp_path.parent / "elsewhere.xml"
    result = run_check(
        Check(id="r", type="test_report", params={"path": str(outside)}),
        CheckContext(working_dir=str(tmp_path), allowed_roots=[str(tmp_path)]),
    )
    assert result.status is CheckStatus.ERROR
    assert result.error["code"] == "verify.check_refused"
