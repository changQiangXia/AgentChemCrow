from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .audit import AuditSession
from .config import Settings
from .guardrails import (
    HIGH_LEVEL_ONLY_SYSTEM_NOTE,
    REACTION_CONSERVATISM_SYSTEM_NOTE,
    assess_prompt_guardrails,
    build_guardrail_refusal,
    is_reaction_analysis_prompt,
    sanitize_reaction_response,
    sanitize_high_level_response,
)
from .prompting import SYSTEM_PROMPT
from .tools import ToolRegistry


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


@dataclass
class AgentRunResult:
    answer: str
    audit_path: str
    steps: int


class ChemCrowLiteAgent:
    def __init__(
        self,
        settings: Settings,
        use_tools: bool = True,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if not settings.llm_api_key:
            raise ValueError(
                "LLM API key is missing. Create a local .env file from .env.example."
            )
        self.settings = settings
        self.use_tools = use_tools
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.timeout_seconds,
        )
        self.registry = ToolRegistry(settings)

    def run(self, prompt: str) -> AgentRunResult:
        audit = AuditSession(question=prompt, model=self.settings.model)
        # 在任何模型调用之前先做安全判定，便于从首步开始完整记录
        # 拒绝、高层摘要等分流结果。
        assessment = assess_prompt_guardrails(
            prompt, self.settings.controlled_chemicals_path
        )
        audit.add_metadata("guardrail_assessment", assessment.as_dict())

        if assessment.response_mode == "hard_refusal":
            refusal = build_guardrail_refusal(assessment)
            audit.add_message({"role": "system", "content": self.system_prompt})
            audit.add_message({"role": "user", "content": prompt})
            audit.add_message({"role": "assistant", "content": refusal})
            audit.final_answer = refusal
            audit_path = audit.save(self.settings.audit_dir)
            return AgentRunResult(answer=refusal, audit_path=str(audit_path), steps=0)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        # 仅在反应分析或高层摘要模式下追加更严格的系统提示。
        if is_reaction_analysis_prompt(prompt):
            messages.append(
                {"role": "system", "content": REACTION_CONSERVATISM_SYSTEM_NOTE}
            )
        if assessment.response_mode == "high_level_only":
            messages.append({"role": "system", "content": HIGH_LEVEL_ONLY_SYSTEM_NOTE})
        messages.append({"role": "user", "content": prompt})

        for item in messages:
            audit.add_message(item)

        # 每一轮循环对应一个 agent step：模型推理、可选工具调用、
        # 工具结果回填，以及下一轮决策。
        for step in range(1, self.settings.max_steps + 1):
            request_payload: dict[str, Any] = dict(
                model=self.settings.model,
                temperature=self.settings.temperature,
                messages=messages,
            )
            if self.use_tools:
                request_payload["tools"] = self.registry.openai_tools
            response = self.client.chat.completions.create(**request_payload)
            message = response.choices[0].message
            assistant_payload = message.model_dump(exclude_none=True)
            audit.add_message(assistant_payload)

            if self.use_tools and message.tool_calls:
                messages.append(assistant_payload)
                # 工具在本地执行，结果会作为结构化观察写回消息流。
                for tool_call in message.tool_calls:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                    result = self.registry.execute(tool_call.function.name, arguments)
                    audit.add_tool_event(
                        step=step,
                        tool_name=tool_call.function.name,
                        arguments=arguments,
                        result=result,
                    )
                    tool_payload = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": self.registry.pretty_tool_result(result),
                    }
                    messages.append(tool_payload)
                    audit.add_message(tool_payload)
                continue

            # 最终自然语言输出仍需经过安全相关清洗，再落盘并返回。
            final_answer = _coerce_content(message.content).strip()
            if assessment.response_mode == "high_level_only":
                final_answer = sanitize_high_level_response(final_answer)
            if is_reaction_analysis_prompt(prompt):
                final_answer = sanitize_reaction_response(prompt, final_answer)
            audit.final_answer = final_answer
            audit_path = audit.save(self.settings.audit_dir)
            return AgentRunResult(
                answer=final_answer,
                audit_path=str(audit_path),
                steps=step,
            )

        # 达到最大步数时执行受控停止，并保留完整审计记录。
        audit.final_answer = (
            "Stopped because the maximum number of tool-using steps was reached."
        )
        audit_path = audit.save(self.settings.audit_dir)
        return AgentRunResult(
            answer=audit.final_answer,
            audit_path=str(audit_path),
            steps=self.settings.max_steps,
        )
