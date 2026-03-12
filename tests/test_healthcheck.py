from pathlib import Path

from chemcrow_lite.config import Settings
from chemcrow_lite.healthcheck import run_healthcheck


def _make_settings(tmp_path: Path) -> Settings:
    chromophore = tmp_path / "chromophore.xlsx"
    controlled = tmp_path / "controlled.csv"
    tasks = tmp_path / "tasks.json"
    chromophore.write_text("placeholder", encoding="utf-8")
    controlled.write_text("placeholder", encoding="utf-8")
    tasks.write_text("[]", encoding="utf-8")
    return Settings(
        provider_name="test-provider",
        llm_api_key="test-key",
        llm_base_url="https://api.test-provider.invalid/v1",
        model="test-model",
        temperature=0.1,
        max_steps=4,
        timeout_seconds=10,
        semantic_scholar_api_key=None,
        chromophore_data_path=chromophore,
        controlled_chemicals_path=controlled,
        tasks_path=tasks,
        audit_dir=tmp_path / "audit_logs",
    )


def test_run_healthcheck_stays_ok_when_semantic_scholar_rate_limited(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(
        "chemcrow_lite.healthcheck._probe_pubchem",
        lambda: {"status": "ok"},
    )
    monkeypatch.setattr(
        "chemcrow_lite.healthcheck._probe_semantic_scholar",
        lambda api_key: {"status": "rate_limited", "query": "q", "results": []},
    )
    monkeypatch.setattr(
        "chemcrow_lite.healthcheck._probe_llm_provider",
        lambda current_settings: {"status": "ok", "provider": current_settings.provider_name},
    )

    payload = run_healthcheck(settings)

    assert payload["overall_status"] == "ok"
    assert payload["semantic_scholar"]["status"] == "rate_limited"


def test_run_healthcheck_warns_on_probe_error(tmp_path: Path, monkeypatch) -> None:
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(
        "chemcrow_lite.healthcheck._probe_pubchem",
        lambda: {"status": "ok"},
    )
    monkeypatch.setattr(
        "chemcrow_lite.healthcheck._probe_semantic_scholar",
        lambda api_key: {"status": "error", "error": "upstream failure"},
    )
    monkeypatch.setattr(
        "chemcrow_lite.healthcheck._probe_llm_provider",
        lambda current_settings: {"status": "ok", "provider": current_settings.provider_name},
    )

    payload = run_healthcheck(settings)

    assert payload["overall_status"] == "warning"
    assert payload["semantic_scholar"]["status"] == "error"


def test_run_healthcheck_warns_on_missing_local_file(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _make_settings(tmp_path)
    settings = settings.with_overrides(tasks_path=tmp_path / "missing_tasks.json")
    monkeypatch.setattr(
        "chemcrow_lite.healthcheck._probe_pubchem",
        lambda: {"status": "ok"},
    )
    monkeypatch.setattr(
        "chemcrow_lite.healthcheck._probe_semantic_scholar",
        lambda api_key: {"status": "ok", "query": "q", "results": []},
    )
    monkeypatch.setattr(
        "chemcrow_lite.healthcheck._probe_llm_provider",
        lambda current_settings: {"status": "ok", "provider": current_settings.provider_name},
    )

    payload = run_healthcheck(settings)

    assert payload["overall_status"] == "warning"
    assert payload["local_files"]["tasks_json"]["exists"] is False
