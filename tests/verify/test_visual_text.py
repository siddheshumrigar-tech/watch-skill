"""OCR text checks match the way search does, optionally fuzzily.

`visual_absent` used a plain lowercase substring, so an Arabic word with a
hamza on its alef escaped a ban on the same word written without one, and an
OCR misread such as `NaH` for `NaN` passed a check written to catch it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from watch_skill.verify import Check, CheckContext, CheckStatus
from watch_skill.verify.checks import run_check


def _ctx(tmp_path: Path, lines: list[str] | None) -> CheckContext:
    evidence = {"ocr_text": lines} if lines is not None else {}
    return CheckContext(working_dir=str(tmp_path), allowed_roots=[str(tmp_path)],
                        evidence=evidence)


def _run(kind: str, tmp_path: Path, lines: list[str] | None, **params):
    return run_check(Check(id="v", type=kind, params=params), _ctx(tmp_path, lines))


def test_absent_catches_a_hamza_variant(tmp_path: Path) -> None:
    """Regression: 'أحمد' on screen escaped a ban on 'احمد'."""
    result = _run("visual_absent", tmp_path, ["مرحبا أحمد"], term="احمد")
    assert result.status is CheckStatus.FAIL


def test_absent_catches_a_case_and_width_variant(tmp_path: Path) -> None:
    result = _run("visual_absent", tmp_path, ["Total: ＮａＮ"], term="nan")
    assert result.status is CheckStatus.FAIL


def test_fuzzy_absent_catches_an_ocr_misread(tmp_path: Path) -> None:
    lines = ["Total: $NaH"]
    assert _run("visual_absent", tmp_path, lines, term="NaN").status is CheckStatus.PASS
    fuzzy = _run("visual_absent", tmp_path, lines, term="NaN", min_similarity=0.6)
    assert fuzzy.status is CheckStatus.FAIL


def test_fuzzy_matching_spans_several_words(tmp_path: Path) -> None:
    lines = ["Error: paymcnt failed, retry"]
    result = _run("visual_present", tmp_path, lines, term="payment failed", min_similarity=0.85)
    assert result.status is CheckStatus.PASS


def test_present_passes_fails_and_is_inconclusive(tmp_path: Path) -> None:
    assert _run("visual_present", tmp_path, ["Order confirmed"],
                term="confirmed").status is CheckStatus.PASS
    assert _run("visual_present", tmp_path, ["Order pending"],
                term="confirmed").status is CheckStatus.FAIL
    assert _run("visual_present", tmp_path, None,
                term="confirmed").status is CheckStatus.INCONCLUSIVE


@pytest.mark.parametrize("value", [0, 1.5, -0.2])
def test_an_out_of_range_threshold_is_malformed(tmp_path: Path, value: float) -> None:
    for kind in ("visual_absent", "visual_present"):
        result = _run(kind, tmp_path, ["x"], term="x", min_similarity=value)
        assert result.status is CheckStatus.ERROR
        assert result.error["code"] == "verify.check_malformed"


def test_visual_present_is_a_supported_type() -> None:
    from watch_skill.verify import SUPPORTED_CHECK_TYPES

    assert "visual_present" in SUPPORTED_CHECK_TYPES
