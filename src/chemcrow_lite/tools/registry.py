from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..config import Settings
from .chromophore import chromophore_dataset_summary, chromophore_rf_screen
from .literature import literature_search
from .pubchem import (
    functional_groups,
    name_to_smiles,
    query_to_cas,
    smiles_to_name,
    smiles_to_weight,
)
from .reaction import reaction_outcome_heuristic, retrosynthesis_overview
from .safety import control_chem_check, explosive_check, safety_summary


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tools = self._build_tools()

    def _build_tools(self) -> dict[str, ToolDefinition]:
        settings = self.settings

        return {
            "name_to_smiles": ToolDefinition(
                name="name_to_smiles",
                description="Convert a molecule name or identifier into canonical SMILES using PubChem.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lambda args: name_to_smiles(args["query"]),
            ),
            "mol_to_cas": ToolDefinition(
                name="mol_to_cas",
                description="Find the CAS number of a molecule using PubChem.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lambda args: query_to_cas(args["query"]),
            ),
            "smiles_to_name": ToolDefinition(
                name="smiles_to_name",
                description="Convert a SMILES string into a likely common name using PubChem synonyms.",
                parameters={
                    "type": "object",
                    "properties": {"smiles": {"type": "string"}},
                    "required": ["smiles"],
                },
                handler=lambda args: smiles_to_name(args["smiles"]),
            ),
            "smiles_to_weight": ToolDefinition(
                name="smiles_to_weight",
                description="Compute exact molecular weight from SMILES using RDKit.",
                parameters={
                    "type": "object",
                    "properties": {"smiles": {"type": "string"}},
                    "required": ["smiles"],
                },
                handler=lambda args: smiles_to_weight(args["smiles"]),
            ),
            "functional_groups": ToolDefinition(
                name="functional_groups",
                description="Identify major functional groups in a molecule from SMILES.",
                parameters={
                    "type": "object",
                    "properties": {"smiles": {"type": "string"}},
                    "required": ["smiles"],
                },
                handler=lambda args: functional_groups(args["smiles"]),
            ),
            "control_chem_check": ToolDefinition(
                name="control_chem_check",
                description="Check whether a molecule appears in the local controlled/explosive starter registry.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lambda args: control_chem_check(
                    args["query"], settings.controlled_chemicals_path
                ),
            ),
            "explosive_check": ToolDefinition(
                name="explosive_check",
                description="Query PubChem GHS-related information and detect explosive risk.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lambda args: explosive_check(args["query"]),
            ),
            "safety_summary": ToolDefinition(
                name="safety_summary",
                description="Return a structured safety summary with operator safety, GHS information, and registry checks.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lambda args: safety_summary(
                    args["query"], settings.controlled_chemicals_path
                ),
            ),
            "literature_search": ToolDefinition(
                name="literature_search",
                description="Search Semantic Scholar for relevant scientific papers.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
                handler=lambda args: literature_search(
                    query=args["query"],
                    limit=int(args.get("limit", 3)),
                    api_key=settings.semantic_scholar_api_key,
                ),
            ),
            "reaction_outcome_heuristic": ToolDefinition(
                name="reaction_outcome_heuristic",
                description="Run a lightweight, non-operational reaction analysis heuristic to assess whether the stated substrates/reagents support a plausible transformation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "substrates": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "reagents": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "conditions": {"type": "string"},
                    },
                    "required": ["substrates"],
                },
                handler=lambda args: reaction_outcome_heuristic(
                    substrates=list(args["substrates"]),
                    reagents=list(args.get("reagents", [])),
                    conditions=args.get("conditions"),
                ),
            ),
            "retrosynthesis_overview": ToolDefinition(
                name="retrosynthesis_overview",
                description="Provide a high-level retrosynthetic family overview for a target molecule without procedural details.",
                parameters={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
                handler=lambda args: retrosynthesis_overview(
                    target=args["target"],
                    controlled_chemicals_path=str(settings.controlled_chemicals_path),
                ),
            ),
            "chromophore_dataset_summary": ToolDefinition(
                name="chromophore_dataset_summary",
                description="Summarize the local chromophore dataset used in the paper-inspired workflow.",
                parameters={"type": "object", "properties": {}},
                handler=lambda args: chromophore_dataset_summary(
                    settings.chromophore_data_path
                ),
            ),
            "chromophore_rf_screen": ToolDefinition(
                name="chromophore_rf_screen",
                description="Run the paper-inspired chromophore workflow: acetonitrile filtering, Morgan fingerprints, random forest training, RMSE, and candidate screening.",
                parameters={
                    "type": "object",
                    "properties": {
                        "target_nm": {"type": "number"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["target_nm"],
                },
                handler=lambda args: chromophore_rf_screen(
                    data_path=settings.chromophore_data_path,
                    target_nm=float(args["target_nm"]),
                    top_k=int(args.get("top_k", 5)),
                ),
            ),
        }

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool.as_openai_tool() for tool in self.tools.values()]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            return self.tools[tool_name].handler(arguments)
        except Exception as exc:
            return {
                "tool_name": tool_name,
                "error": str(exc),
                "arguments": arguments,
            }

    def pretty_tool_result(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)
