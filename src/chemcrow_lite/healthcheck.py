from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import APIStatusError, OpenAI

from .config import Settings
from .tools.literature import literature_search
from .tools.pubchem import name_to_smiles


def _coerce_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _file_probe(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _probe_pubchem() -> dict[str, Any]:
    try:
        result = name_to_smiles("acetaminophen")
        return {
            "status": "ok",
            "query": "acetaminophen",
            "smiles": result["smiles"],
            "source": result["source"],
        }
    except Exception as exc:
        return {
            "status": "error",
            "query": "acetaminophen",
            "error": str(exc),
        }


def _probe_semantic_scholar(api_key: str | None) -> dict[str, Any]:
    try:
        result = literature_search(
            query="acetaminophen safety", limit=1, api_key=api_key
        )
        top_hit = result["results"][0] if result.get("results") else None
        return {
            "status": result.get("status", "unknown"),
            "query": "acetaminophen safety",
            "top_title": top_hit.get("title") if top_hit else None,
            "total": result.get("total"),
            "rate_limit_note": result.get("message"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "query": "acetaminophen safety",
            "error": str(exc),
        }


def _probe_llm_provider(settings: Settings) -> dict[str, Any]:
    if not settings.llm_api_key:
        return {"status": "missing_api_key"}

    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.timeout_seconds,
    )

    try:
        response = client.chat.completions.create(
            model=settings.model,
            temperature=0.0,
            max_tokens=8,
            messages=[{"role": "user", "content": "Reply with OK only."}],
        )
        message = response.choices[0].message
        return {
            "status": "ok",
            "provider": settings.provider_name,
            "model": settings.model,
            "reply": _coerce_content(message.content).strip(),
        }
    except APIStatusError as exc:
        detail: Any
        try:
            detail = exc.response.json()
        except Exception:
            detail = str(exc)
        return {
            "status": "error",
            "provider": settings.provider_name,
            "model": settings.model,
            "status_code": getattr(exc, "status_code", None),
            "detail": detail,
        }
    except Exception as exc:
        return {
            "status": "error",
            "provider": settings.provider_name,
            "model": settings.model,
            "error": str(exc),
        }


def run_healthcheck(settings: Settings) -> dict[str, Any]:
    local_files = {
        "chromophore_data": _file_probe(settings.chromophore_data_path),
        "controlled_registry": _file_probe(settings.controlled_chemicals_path),
        "tasks_json": _file_probe(settings.tasks_path),
    }
    pubchem = _probe_pubchem()
    semantic_scholar = _probe_semantic_scholar(settings.semantic_scholar_api_key)
    llm_provider = _probe_llm_provider(settings)

    overall_status = "ok"
    if not all(item["exists"] for item in local_files.values()):
        overall_status = "warning"
    if pubchem["status"] != "ok":
        overall_status = "warning"
    if semantic_scholar["status"] not in {"ok", "rate_limited"}:
        overall_status = "warning"
    if llm_provider["status"] != "ok":
        overall_status = "warning"

    return {
        "overall_status": overall_status,
        "local_files": local_files,
        "pubchem": pubchem,
        "semantic_scholar": semantic_scholar,
        "llm_provider": llm_provider,
    }
