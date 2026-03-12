from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .tools.safety import load_controlled_registry


SYNTHESIS_KEYWORDS = (
    "synth",
    "synthesis",
    "make ",
    "prepare ",
    "route",
    "retrosynth",
    "protocol",
    "procedure",
    "reaction conditions",
    "lab-scale",
    "manufacture",
)

OPERATIONAL_DETAIL_KEYWORDS = (
    "step-by-step",
    "stoichiometry",
    "equiv",
    "grams",
    "gram ",
    "moles",
    "mol ",
    "temperature",
    "pressure",
    "hours",
    "minutes",
    "workup",
    "purification",
    "reflux",
    "distill",
)

REACTION_ANALYSIS_KEYWORDS = (
    "predict the product",
    "reaction mechanism",
    "mechanism",
    "lindlar",
    "catalyst",
    "retrosynth",
    "retrosynthesis",
)


HIGH_LEVEL_ONLY_SYSTEM_NOTE = """This request touches synthesis or experimental chemistry.

You may provide only a high-level educational overview.
Do not provide:
- step-by-step procedures
- reagent quantities or stoichiometry
- exact temperatures, pressures, durations, or concentrations
- workup, purification, or optimization instructions

Keep the answer safety-first, high-level, and non-operational.
"""


REACTION_CONSERVATISM_SYSTEM_NOTE = """This request involves reaction prediction or mechanism analysis.

Rules:
1. Do not assume omitted reagents, gases, solvents, catalysts, or conditions.
2. A catalyst mention does not imply hydrogen, oxidant, base, pressure, temperature, or atmosphere unless explicitly stated.
3. If key conditions are underspecified, say you cannot predict the outcome with high confidence.
4. Prefer conservative, uncertainty-aware reasoning over speculative product claims.
"""


@dataclass(frozen=True)
class PromptGuardrailAssessment:
    prompt: str
    is_synthesis_request: bool
    asks_operational_details: bool
    matched_controlled_terms: list[str]
    matched_risk_tiers: list[str]
    response_mode: str
    refusal_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _find_controlled_terms(
    prompt: str, controlled_chemicals_path: Path
) -> list[tuple[str, str]]:
    dataframe = load_controlled_registry(controlled_chemicals_path)
    lowered = f" {prompt.lower()} "
    matches: list[tuple[str, str]] = []
    seen_terms: set[str] = set()
    for _, row in dataframe.iterrows():
        terms = [row["name_norm"], *row["aliases_norm"]]
        for term in sorted(set(terms), key=len, reverse=True):
            if term in seen_terms:
                continue
            pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            if re.search(pattern, lowered):
                matches.append((term, str(row["risk_tier"])))
                seen_terms.add(term)
    return matches


def assess_prompt_guardrails(
    prompt: str, controlled_chemicals_path: Path
) -> PromptGuardrailAssessment:
    lowered = prompt.lower()
    is_synthesis_request = any(keyword in lowered for keyword in SYNTHESIS_KEYWORDS)
    asks_operational_details = any(
        keyword in lowered for keyword in OPERATIONAL_DETAIL_KEYWORDS
    )
    matched = _find_controlled_terms(prompt, controlled_chemicals_path)
    matched_controlled_terms = [term for term, _ in matched]
    matched_risk_tiers = sorted({risk_tier for _, risk_tier in matched})

    if (
        "hard_refusal" in matched_risk_tiers
        and (is_synthesis_request or asks_operational_details)
    ):
        return PromptGuardrailAssessment(
            prompt=prompt,
            is_synthesis_request=is_synthesis_request,
            asks_operational_details=asks_operational_details,
            matched_controlled_terms=matched_controlled_terms,
            matched_risk_tiers=matched_risk_tiers,
            response_mode="hard_refusal",
            refusal_reason="controlled_or_explosive_synthesis_request",
        )

    if is_synthesis_request or asks_operational_details or matched_controlled_terms:
        return PromptGuardrailAssessment(
            prompt=prompt,
            is_synthesis_request=is_synthesis_request,
            asks_operational_details=asks_operational_details,
            matched_controlled_terms=matched_controlled_terms,
            matched_risk_tiers=matched_risk_tiers,
            response_mode="high_level_only",
        )

    return PromptGuardrailAssessment(
        prompt=prompt,
        is_synthesis_request=is_synthesis_request,
        asks_operational_details=asks_operational_details,
        matched_controlled_terms=matched_controlled_terms,
        matched_risk_tiers=matched_risk_tiers,
        response_mode="normal",
    )


def build_guardrail_refusal(assessment: PromptGuardrailAssessment) -> str:
    matched = ", ".join(assessment.matched_controlled_terms) or "the requested substance"
    return (
        "I cannot assist with procedural or synthesis guidance for "
        f"{matched}. The request triggers a hard safety guardrail because it "
        "appears to involve a controlled or explosive substance. I can still "
        "help with high-level safety, regulatory context, or non-operational "
        "background information."
    )


def is_reaction_analysis_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(keyword in lowered for keyword in REACTION_ANALYSIS_KEYWORDS)


def sanitize_high_level_response(text: str) -> str:
    cleaned_blocks: list[str] = []
    removed_any = False

    for block in text.splitlines():
        stripped = block.strip()
        if not stripped:
            cleaned_blocks.append(block)
            continue
        if stripped.startswith("###"):
            cleaned_blocks.append(block)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        safe_sentences = [
            sentence
            for sentence in sentences
            if not any(keyword in sentence.lower() for keyword in OPERATIONAL_DETAIL_KEYWORDS)
        ]
        if safe_sentences:
            cleaned_blocks.append(" ".join(safe_sentences))
        else:
            removed_any = True
            continue
        if len(safe_sentences) != len(sentences):
            removed_any = True

    cleaned = "\n".join(block for block in cleaned_blocks if block.strip() or block == "")
    if removed_any:
        cleaned = (
            cleaned.rstrip()
            + "\n\nProcedural details have been omitted by the safety guardrail."
        )
    return cleaned.strip()


def sanitize_reaction_response(prompt: str, text: str) -> str:
    lowered_prompt = prompt.lower()
    lowered_text = text.lower()

    missing_hydrogen_context = (
        "h2" not in lowered_prompt
        and "hydrogen gas" not in lowered_prompt
        and (
            "assuming h₂" in lowered_text
            or "assuming h2" in lowered_text
            or "if h₂ were present" in lowered_text
            or "if h2 were present" in lowered_text
        )
    )

    lacks_uncertainty = not any(
        phrase in lowered_text
        for phrase in (
            "cannot predict with high confidence",
            "uncertain",
            "insufficient",
            "underspecified",
            "no reaction",
        )
    )

    if is_reaction_analysis_prompt(prompt) and (missing_hydrogen_context or lacks_uncertainty):
        caveat = (
            "Because key conditions are underspecified in the prompt, I cannot "
            "predict the reaction outcome with high confidence; any product "
            "statement above should be treated as hypothetical rather than as a "
            "definitive prediction under the stated conditions."
        )
        if caveat.lower() not in lowered_text:
            return text.rstrip() + "\n\n" + caveat

    return text.strip()
