import json
from pathlib import Path

from chemcrow_lite.agent import AgentRunResult
from chemcrow_lite.benchmarking import run_benchmark
from chemcrow_lite.config import Settings


class FakeAgent:
    def __init__(
        self,
        settings: Settings,
        use_tools: bool = True,
        system_prompt: str | None = None,
    ):
        self.use_tools = use_tools

    def run(self, prompt: str) -> AgentRunResult:
        label = "agent" if self.use_tools else "no_tools"
        return AgentRunResult(
            answer=f"{label}:{prompt}",
            audit_path=f"{label}.json",
            steps=1,
        )


def _make_settings(tmp_path: Path, tasks_path: Path) -> Settings:
    return Settings(
        provider_name="test-provider",
        llm_api_key="test-key",
        llm_base_url="https://api.test-provider.invalid/v1",
        model="test-model",
        temperature=0.1,
        max_steps=4,
        timeout_seconds=10,
        semantic_scholar_api_key=None,
        chromophore_data_path=tmp_path / "chromophore.xlsx",
        controlled_chemicals_path=tmp_path / "controlled.csv",
        tasks_path=tasks_path,
        audit_dir=tmp_path / "audit_logs",
    )


def test_run_benchmark_compare_mode(tmp_path: Path, monkeypatch) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "id": "task_01_test",
                    "source": "unit",
                    "prompt": "prompt one",
                    "evaluation": {"expected_keywords_any": ["prompt one"]},
                },
                {
                    "id": "task_04_chromophore",
                    "source": "unit",
                    "prompt": "prompt two",
                    "evaluation": {
                        "requires_candidate": True,
                        "requires_rmse": True,
                        "requires_cv_metrics": True,
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    settings = _make_settings(tmp_path, tasks_path)

    monkeypatch.setattr("chemcrow_lite.benchmarking.ChemCrowLiteAgent", FakeAgent)
    monkeypatch.setattr(
        "chemcrow_lite.benchmarking.chromophore_rf_screen",
        lambda data_path, target_nm, top_k: {
            "rmse_nm": 12.3,
            "cv_rmse_mean_nm": 11.1,
            "model_advantage_over_dummy_nm": 5.5,
            "best_candidate": {"smiles": "CCO"},
        },
    )

    payload = run_benchmark(settings, limit=2, mode="compare")

    assert payload["mode"] == "compare"
    assert payload["summary"]["compare_tasks"] == 1
    assert payload["summary"]["offline_workflows"] == 1
    assert payload["summary"]["both_ok"] == 1
    assert Path(payload["benchmark_path"]).exists()
    assert Path(payload["report_path"]).exists()
    assert payload["evaluation_summary"]["evaluated_runs"] == 3
    assert payload["results"][0]["runs"]["agent"]["status"] == "ok"
    assert payload["results"][0]["runs"]["no_tools"]["status"] == "ok"


def test_run_benchmark_no_tools_mode(tmp_path: Path, monkeypatch) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "id": "task_01_test",
                    "source": "unit",
                    "prompt": "prompt one",
                    "evaluation": {"expected_keywords_any": ["prompt one"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    settings = _make_settings(tmp_path, tasks_path)

    monkeypatch.setattr("chemcrow_lite.benchmarking.ChemCrowLiteAgent", FakeAgent)

    payload = run_benchmark(settings, limit=1, mode="no_tools")

    assert payload["mode"] == "no_tools"
    assert payload["summary"]["no_tools_ok"] == 1
    assert payload["results"][0]["mode"] == "no_tools"
    assert payload["results"][0]["evaluation"]["overall_pass"] is True
