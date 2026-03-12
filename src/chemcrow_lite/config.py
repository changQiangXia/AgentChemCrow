from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    provider_name: str
    llm_api_key: str | None
    llm_base_url: str
    model: str
    temperature: float
    max_steps: int
    timeout_seconds: int
    semantic_scholar_api_key: str | None
    chromophore_data_path: Path
    controlled_chemicals_path: Path
    tasks_path: Path
    audit_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        llm_api_key = (
            os.getenv("CHEMCROW_API_KEY")
            or os.getenv("MOONSHOT_API_KEY")
            or os.getenv("KIMI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ZHIPU_API_KEY")
        )
        llm_base_url = (
            os.getenv("CHEMCROW_BASE_URL")
            or os.getenv("MOONSHOT_BASE_URL")
            or os.getenv("KIMI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("ZHIPU_BASE_URL")
            or "https://api.moonshot.cn/v1"
        )
        return cls(
            provider_name=os.getenv("CHEMCROW_PROVIDER", "kimi"),
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            model=os.getenv("CHEMCROW_MODEL", "moonshot-v1-8k"),
            temperature=float(os.getenv("CHEMCROW_TEMPERATURE", "0.1")),
            max_steps=int(os.getenv("CHEMCROW_MAX_STEPS", "8")),
            timeout_seconds=int(os.getenv("CHEMCROW_TIMEOUT_SECONDS", "45")),
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            chromophore_data_path=Path(
                os.getenv(
                    "CHEMCROW_CHROMOPHORE_DATA",
                    str(ROOT / "DB for chromophore_Sci_Data.xlsx"),
                )
            ),
            controlled_chemicals_path=Path(
                os.getenv(
                    "CHEMCROW_CONTROLLED_CHEMICALS",
                    str(ROOT / "data" / "controlled_chemicals_min.csv"),
                )
            ),
            tasks_path=Path(
                os.getenv(
                    "CHEMCROW_TASKS_JSON",
                    str(ROOT / "data" / "tasks_paper_min.json"),
                )
            ),
            audit_dir=Path(
                os.getenv("CHEMCROW_AUDIT_DIR", str(ROOT / "audit_logs"))
            ),
        )

    def with_overrides(self, **kwargs: object) -> "Settings":
        return replace(self, **kwargs)
