from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .pubchem import _get_json, query_to_cas, query_to_cid


@lru_cache(maxsize=8)
def _load_controlled_registry(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    dataframe["name_norm"] = dataframe["name"].str.strip().str.lower()
    dataframe["cas_norm"] = dataframe["cas"].astype(str).str.strip()
    if "aliases" not in dataframe.columns:
        dataframe["aliases"] = ""
    dataframe["aliases_norm"] = dataframe["aliases"].fillna("").map(
        lambda value: [
            item.strip().lower()
            for item in str(value).split("|")
            if item and item.strip()
        ]
    )
    if "risk_tier" not in dataframe.columns:
        dataframe["risk_tier"] = dataframe["category"].map(
            lambda value: "hard_refusal"
            if str(value).strip().lower() in {"explosive", "controlled"}
            else "watchlist"
        )
    return dataframe


def load_controlled_registry(controlled_chemicals_path: Path) -> pd.DataFrame:
    return _load_controlled_registry(str(controlled_chemicals_path))


def _extract_text(node: Any) -> list[str]:
    snippets: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"String", "Extra", "Name"} and isinstance(value, str):
                snippets.append(value.strip())
            else:
                snippets.extend(_extract_text(value))
    elif isinstance(node, list):
        for item in node:
            snippets.extend(_extract_text(item))
    return [item for item in snippets if item]


def _find_sections_by_heading(node: Any, heading: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("TOCHeading") == heading:
            matches.append(node)
        for value in node.values():
            matches.extend(_find_sections_by_heading(value, heading))
    elif isinstance(node, list):
        for item in node:
            matches.extend(_find_sections_by_heading(item, heading))
    return matches


def control_chem_check(query: str, controlled_chemicals_path: Path) -> dict[str, Any]:
    registry = load_controlled_registry(controlled_chemicals_path)
    query_norm = query.strip().lower()

    exact_name = registry[registry["name_norm"] == query_norm]
    if not exact_name.empty:
        row = exact_name.iloc[0]
        return {
            "query": query,
            "is_controlled": True,
            "match_type": "name",
            "matched_name": row["name"],
            "matched_cas": row["cas"],
            "category": row["category"],
            "risk_tier": row["risk_tier"],
            "matched_alias": None,
        }

    alias_match = registry[registry["aliases_norm"].map(lambda items: query_norm in items)]
    if not alias_match.empty:
        row = alias_match.iloc[0]
        return {
            "query": query,
            "is_controlled": True,
            "match_type": "alias",
            "matched_name": row["name"],
            "matched_cas": row["cas"],
            "category": row["category"],
            "risk_tier": row["risk_tier"],
            "matched_alias": query.strip(),
        }

    try:
        cas = query_to_cas(query)["cas"]
    except Exception:
        cas = None

    if cas:
        exact_cas = registry[registry["cas_norm"] == cas]
        if not exact_cas.empty:
            row = exact_cas.iloc[0]
            return {
                "query": query,
                "is_controlled": True,
                "match_type": "cas",
                "matched_name": row["name"],
                "matched_cas": row["cas"],
                "category": row["category"],
                "risk_tier": row["risk_tier"],
                "matched_alias": None,
            }

    return {
        "query": query,
        "is_controlled": False,
        "match_type": None,
        "matched_name": None,
        "matched_cas": cas,
        "category": None,
        "risk_tier": None,
        "matched_alias": None,
        "note": "Starter registry performs exact name/alias/CAS matching. Similarity-based controlled-chemical screening can be added later with a vetted reference set.",
    }


def explosive_check(query: str) -> dict[str, Any]:
    cid = query_to_cid(query)
    payload = _get_json(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    )

    ghs_sections = _find_sections_by_heading(payload, "GHS Classification")
    text = " ".join(_extract_text(ghs_sections)).lower()
    is_explosive = "explos" in text
    return {
        "query": query,
        "cid": cid,
        "is_explosive": is_explosive,
        "evidence": text[:1200] if text else "No GHS Classification text found.",
    }


def safety_summary(query: str, controlled_chemicals_path: Path) -> dict[str, Any]:
    cid = query_to_cid(query)
    payload = _get_json(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    )
    control_result = control_chem_check(query, controlled_chemicals_path)

    section_map = {
        "ghs_information": "GHS Classification",
        "preventive_measures": "Preventive Measures",
        "ppe": "Personal Protective Equipment (PPE)",
        "toxicity": "Toxicity Summary",
        "hazards": "Hazards Summary",
    }

    extracted: dict[str, str] = {}
    for key, heading in section_map.items():
        sections = _find_sections_by_heading(payload, heading)
        text = " ".join(dict.fromkeys(_extract_text(sections)))
        extracted[key] = text[:1200] if text else "No explicit PubChem entry found."

    operator_safety = extracted["ppe"]
    if operator_safety == "No explicit PubChem entry found.":
        operator_safety = extracted["hazards"]

    environmental_risks = "No explicit environmental risk snippet found in the extracted PubChem sections."
    societal_impact = (
        f"Controlled chemical registry match: {control_result['matched_name']} ({control_result['category']})."
        if control_result["is_controlled"]
        else "Not found in the local controlled-chemical starter registry."
    )

    return {
        "query": query,
        "cid": cid,
        "operator_safety": operator_safety,
        "ghs_information": extracted["ghs_information"],
        "environmental_risks": environmental_risks,
        "societal_impact": societal_impact,
        "control_check": control_result,
    }
