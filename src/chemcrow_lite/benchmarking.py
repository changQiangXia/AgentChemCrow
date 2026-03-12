from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .agent import ChemCrowLiteAgent
from .config import Settings
from .evaluation import (
    evaluate_offline_workflow,
    evaluate_run,
    render_benchmark_report,
    summarize_evaluations,
)
from .prompting import BASELINE_SYSTEM_PROMPT, SYSTEM_PROMPT
from .tools.chromophore import chromophore_rf_screen


def _run_agent_prompt(
    agent: ChemCrowLiteAgent, prompt: str, mode: str
) -> dict[str, Any]:
    try:
        run = agent.run(prompt)
        return {
            "mode": mode,
            "status": "ok",
            "answer": run.answer,
            "audit_path": run.audit_path,
            "steps": run.steps,
        }
    except Exception as exc:
        return {
            "mode": mode,
            "status": "error",
            "error": str(exc),
        }


def _attach_run_evaluation(
    run_payload: dict[str, Any], task_spec: dict[str, Any], run_mode: str
) -> dict[str, Any]:
    if run_payload["status"] != "ok":
        run_payload["evaluation"] = {
            "overall_pass": False,
            "failed_checks": ["runtime_error"],
            "checks": {},
            "tool_event_count": 0,
        }
        return run_payload

    run_payload["evaluation"] = evaluate_run(
        answer=run_payload.get("answer", ""),
        audit_path=run_payload.get("audit_path"),
        task_spec=task_spec,
        run_mode=run_mode,
    )
    return run_payload


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "ok": 0,
        "error": 0,
        "compare_tasks": 0,
        "agent_ok": 0,
        "agent_error": 0,
        "no_tools_ok": 0,
        "no_tools_error": 0,
        "both_ok": 0,
        "offline_workflows": 0,
    }

    for item in results:
        if item.get("mode") == "offline_workflow":
            summary["offline_workflows"] += 1
            summary[item["status"]] += 1
            continue

        if item.get("mode") == "compare":
            summary["compare_tasks"] += 1
            agent_run = item["runs"]["agent"]
            baseline_run = item["runs"]["no_tools"]
            summary["agent_ok" if agent_run["status"] == "ok" else "agent_error"] += 1
            summary[
                "no_tools_ok"
                if baseline_run["status"] == "ok"
                else "no_tools_error"
            ] += 1
            if agent_run["status"] == "ok" and baseline_run["status"] == "ok":
                summary["both_ok"] += 1
                summary["ok"] += 1
            else:
                summary["error"] += 1
            continue

        summary[item["status"]] += 1
        if item.get("mode") == "agent":
            summary["agent_ok" if item["status"] == "ok" else "agent_error"] += 1
        if item.get("mode") == "no_tools":
            summary[
                "no_tools_ok" if item["status"] == "ok" else "no_tools_error"
            ] += 1

    return summary


def run_benchmark(
    settings: Settings, limit: int = 5, mode: str = "agent"
) -> dict[str, Any]:
    mode = mode.strip().lower()
    if mode not in {"agent", "no_tools", "compare"}:
        raise ValueError("mode must be one of: agent, no_tools, compare")

    tasks = json.loads(settings.tasks_path.read_text(encoding="utf-8"))
    tasks_by_id = {task["id"]: task for task in tasks}
    results: list[dict[str, Any]] = []
    agent = None
    baseline_agent = None

    for item in tasks[:limit]:
        task_id = item["id"]
        prompt = item["prompt"]
        source = item.get("source")

        if task_id == "task_04_chromophore":
            workflow = chromophore_rf_screen(
                data_path=settings.chromophore_data_path,
                target_nm=369.0,
                top_k=5,
            )
            evaluation = evaluate_offline_workflow(tasks_by_id[task_id], workflow)
            results.append(
                {
                    "id": task_id,
                    "source": source,
                    "mode": "offline_workflow",
                    "status": "ok",
                    "result": workflow,
                    "evaluation": evaluation,
                }
            )
            continue

        if mode == "compare":
            if agent is None:
                agent = ChemCrowLiteAgent(
                    settings, use_tools=True, system_prompt=SYSTEM_PROMPT
                )
            if baseline_agent is None:
                baseline_agent = ChemCrowLiteAgent(
                    settings,
                    use_tools=False,
                    system_prompt=BASELINE_SYSTEM_PROMPT,
                )
            results.append(
                {
                    "id": task_id,
                    "source": source,
                    "mode": "compare",
                    "runs": {
                        "agent": _attach_run_evaluation(
                            _run_agent_prompt(agent, prompt, "agent"),
                            tasks_by_id[task_id],
                            "agent",
                        ),
                        "no_tools": _attach_run_evaluation(
                            _run_agent_prompt(baseline_agent, prompt, "no_tools"),
                            tasks_by_id[task_id],
                            "no_tools",
                        ),
                    },
                }
            )
            continue

        if mode == "agent":
            if agent is None:
                agent = ChemCrowLiteAgent(
                    settings, use_tools=True, system_prompt=SYSTEM_PROMPT
                )
            run_result = _run_agent_prompt(agent, prompt, "agent")
        else:
            if baseline_agent is None:
                baseline_agent = ChemCrowLiteAgent(
                    settings,
                    use_tools=False,
                    system_prompt=BASELINE_SYSTEM_PROMPT,
                )
            run_result = _run_agent_prompt(baseline_agent, prompt, "no_tools")

        results.append(
            {
                "id": task_id,
                "source": source,
                **_attach_run_evaluation(run_result, tasks_by_id[task_id], mode),
            }
        )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": settings.model,
        "mode": mode,
        "task_count": len(results),
        "summary": _summarize_results(results),
        "evaluation_summary": summarize_evaluations(results),
        "results": results,
    }
    settings.audit_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = (
        settings.audit_dir
        / f"benchmark_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    benchmark_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload["benchmark_path"] = str(benchmark_path)
    report_path = benchmark_path.with_suffix(".md")
    report_path.write_text(render_benchmark_report(payload), encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload
