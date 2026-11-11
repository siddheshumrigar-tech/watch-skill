"""`watch-skill models pull asr`: the command the not-cached error names.

A live session refuses to download its speech model, which is right, but the
old error told the agent to paste a Python one-liner. An agent needs a
command it can run, and it needs that command named in the error.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from watch_skill.errors import WatchSkillError


def _fake_whisper(monkeypatch: pytest.MonkeyPatch, *, cached: bool) -> dict:
    state = {"cached": cached, "downloads": [], "loads": []}

    def download_model(size, local_files_only=False, **kwargs):
        if local_files_only and not state["cached"]:
            raise RuntimeError("not in the local cache")
        if not local_files_only:
            state["downloads"].append(size)
            state["cached"] = True
        return f"C:/hf cache/models--Systran--faster-whisper-{size}"

    class WhisperModel:
        def __init__(self, size, **kwargs):
            if kwargs.get("local_files_only") and not state["cached"]:
                raise RuntimeError("not in the local cache")
            state["loads"].append((size, kwargs.get("local_files_only")))

    module = types.ModuleType("faster_whisper")
    module.download_model, module.WhisperModel = download_model, WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return state


def test_pull_downloads_once_and_proves_an_offline_load(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.models.asr_cache import pull_asr_model

    state = _fake_whisper(monkeypatch, cached=False)
    result = pull_asr_model("base")
    assert result["status"] == "downloaded" and result["model"] == "base"
    assert set(result) == {"status", "model", "path", "seconds"}
    assert state["downloads"] == ["base"]
    assert ("base", True) in state["loads"], "the pull must confirm a local_files_only load"

    again = pull_asr_model("base")
    assert again["status"] == "already_cached"
    assert state["downloads"] == ["base"]


def test_dry_run_downloads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.models.asr_cache import pull_asr_model

    state = _fake_whisper(monkeypatch, cached=False)
    assert pull_asr_model("tiny", dry_run=True)["status"] == "would_download"
    assert state["downloads"] == []


def test_offline_mode_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.config import reset_settings
    from watch_skill.models.asr_cache import pull_asr_model

    state = _fake_whisper(monkeypatch, cached=False)
    monkeypatch.setenv("WATCHSKILL_OFFLINE", "true")
    reset_settings()
    with pytest.raises(WatchSkillError) as info:
        pull_asr_model("tiny")
    assert info.value.code == "models.offline_refused"
    assert info.value.fix
    assert state["downloads"] == []


def test_without_faster_whisper_the_fix_is_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.models.asr_cache import pull_asr_model

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(WatchSkillError) as info:
        pull_asr_model("tiny")
    assert info.value.code == "models.missing_dependency"
    assert 'pip install "watch-skill[whisper]"' in info.value.fix


def test_the_not_cached_error_names_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the fix was a Python one-liner, not a command."""
    from watch_skill.live.asr import LocalWhisperASR

    _fake_whisper(monkeypatch, cached=False)
    with pytest.raises(WatchSkillError) as info:
        LocalWhisperASR(model_size="tiny")._load()
    assert info.value.code == "models.not_cached"
    assert "watch-skill models pull asr --size tiny" in info.value.fix
    assert info.value.details["command"] == "watch-skill models pull asr --size tiny"


def test_live_asr_uses_the_shared_size_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.live.asr import LocalWhisperASR
    from watch_skill.models.asr_cache import live_asr_model_size

    assert LocalWhisperASR._auto_size() == live_asr_model_size() == "tiny"  # conftest pins tiny


def test_cli_models_pull_asr(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from watch_skill.surfaces.cli.main import app

    _fake_whisper(monkeypatch, cached=False)
    result = CliRunner().invoke(app, ["models", "pull", "asr", "--size", "base", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "status": "would_download", "model": "base", "path": None, "seconds": 0.0,
    }


def test_cli_models_pull_rejects_other_kinds() -> None:
    from typer.testing import CliRunner

    from watch_skill.surfaces.cli.main import app

    result = CliRunner().invoke(app, ["models", "pull", "vlm"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"] == "models.unknown_kind"
