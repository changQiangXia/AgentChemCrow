from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _slugify(text: str, limit: int = 48) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return (collapsed or "run")[:limit]


@dataclass
class ToolEvent:
    step: int
    tool_name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class AuditSession:
    question: str
    model: str
    started_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_events: list[ToolEvent] = field(default_factory=list)
    final_answer: str | None = None
    raw_messages: list[dict[str, Any]] = field(default_factory=list)

    def add_tool_event(
        self, step: int, tool_name: str, arguments: dict[str, Any], result: Any
    ) -> None:
        self.tool_events.append(
            ToolEvent(step=step, tool_name=tool_name, arguments=arguments, result=result)
        )

    def add_message(self, payload: dict[str, Any]) -> None:
        self.raw_messages.append(payload)

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def save(self, audit_dir: Path) -> Path:
        audit_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = audit_dir / f"{timestamp}_{_slugify(self.question)}.json"
        serializable = {
            "question": self.question,
            "model": self.model,
            "started_at": self.started_at,
            "metadata": self.metadata,
            "final_answer": self.final_answer,
            "tool_events": [asdict(event) for event in self.tool_events],
            "raw_messages": self.raw_messages,
        }
        path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path
