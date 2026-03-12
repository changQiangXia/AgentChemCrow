from __future__ import annotations

import json
from dataclasses import asdict

import typer
from openai import APIStatusError
from rich import print
from rich.console import Console

from .agent import ChemCrowLiteAgent
from .benchmarking import run_benchmark
from .config import Settings
from .healthcheck import run_healthcheck
from .prompting import BASELINE_SYSTEM_PROMPT, SYSTEM_PROMPT
from .tools.chromophore import chromophore_rf_screen
from .tools.reaction import reaction_outcome_heuristic, retrosynthesis_overview
from .tools.safety import control_chem_check, explosive_check, safety_summary

app = typer.Typer(help="ChemCrow-Lite CLI")
console = Console()


@app.command()
def ask(
    prompt: str,
    model: str = typer.Option(
        None, help="Override model name, e.g. moonshot-v1-8k"
    ),
    max_steps: int = typer.Option(None, help="Override max tool-using steps."),
    no_tools: bool = typer.Option(
        False, help="Run the same model without tools as a paper-style baseline."
    ),
) -> None:
    settings = Settings.from_env()
    if model:
        settings = settings.with_overrides(model=model)
    if max_steps:
        settings = settings.with_overrides(max_steps=max_steps)

    agent = ChemCrowLiteAgent(
        settings,
        use_tools=not no_tools,
        system_prompt=BASELINE_SYSTEM_PROMPT if no_tools else SYSTEM_PROMPT,
    )
    try:
        result = agent.run(prompt)
    except APIStatusError as exc:
        detail = ""
        try:
            detail = str(exc.response.json())
        except Exception:
            detail = str(exc)
        print("[bold red]API call failed[/bold red]")
        print(detail)
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            print(
                f"\n[yellow]Hint:[/yellow] 当前更像是 {settings.provider_name} API key 无效或权限不足。"
            )
        elif status_code == 429:
            print(
                f"\n[yellow]Hint:[/yellow] 当前更像是 {settings.provider_name} 额度、频率限制或资源配额问题。"
            )
        else:
            print(
                f"\n[yellow]Hint:[/yellow] 当前更像是上游 {settings.provider_name} 接口错误，而不是本地连通性问题。"
            )
        raise typer.Exit(code=1)
    except Exception as exc:
        print("[bold red]Run failed[/bold red]")
        print(str(exc))
        raise typer.Exit(code=1)
    print(f"[bold green]Answer[/bold green]\n{result.answer}")
    print(f"\n[bold cyan]Audit[/bold cyan] {result.audit_path}")
    print(f"[bold cyan]Steps[/bold cyan] {result.steps}")


@app.command("chromophore-demo")
def chromophore_demo(
    target_nm: float = typer.Option(369.0, help="Target absorption wavelength."),
    top_k: int = typer.Option(5, help="How many top candidates to report."),
) -> None:
    settings = Settings.from_env()
    result = chromophore_rf_screen(
        data_path=settings.chromophore_data_path,
        target_nm=target_nm,
        top_k=top_k,
    )
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("safety-check")
def safety_check(query: str) -> None:
    settings = Settings.from_env()
    result = {
        "control_check": control_chem_check(query, settings.controlled_chemicals_path),
        "explosive_check": explosive_check(query),
        "safety_summary": safety_summary(query, settings.controlled_chemicals_path),
    }
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("reaction-check")
def reaction_check(
    substrate: list[str] = typer.Option(..., "--substrate", help="One or more substrates."),
    reagent: list[str] = typer.Option([], "--reagent", help="Optional reagents/catalysts."),
    conditions: str = typer.Option("", help="Optional high-level conditions text."),
) -> None:
    result = reaction_outcome_heuristic(
        substrates=list(substrate),
        reagents=list(reagent),
        conditions=conditions or None,
    )
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("retrosynthesis-demo")
def retrosynthesis_demo(target: str) -> None:
    settings = Settings.from_env()
    result = retrosynthesis_overview(
        target=target,
        controlled_chemicals_path=str(settings.controlled_chemicals_path),
    )
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command()
def benchmark(
    limit: int = typer.Option(5, help="How many benchmark prompts to run."),
    mode: str = typer.Option(
        "compare", help="Benchmark mode: agent, no_tools, or compare."
    ),
) -> None:
    settings = Settings.from_env()
    payload = run_benchmark(settings, limit=limit, mode=mode)
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command()
def healthcheck() -> None:
    settings = Settings.from_env()
    payload = run_healthcheck(settings)
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command("show-config")
def show_config() -> None:
    settings = Settings.from_env()
    payload = asdict(settings)
    if payload.get("llm_api_key"):
        payload["llm_api_key"] = "***redacted***"
    if payload.get("semantic_scholar_api_key"):
        payload["semantic_scholar_api_key"] = "***redacted***"
    console.print_json(json.dumps(payload, ensure_ascii=False, default=str))


if __name__ == "__main__":
    app()
