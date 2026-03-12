from __future__ import annotations

import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem

from .pubchem import canonicalize_smiles, functional_groups, is_smiles, name_to_smiles
from .safety import control_chem_check


TERMINAL_ALKYNE = Chem.MolFromSmarts("[CX2]#[CX2H1]")
ALKYNE = Chem.MolFromSmarts("[CX2]#[CX2]")
CARBOXYLIC_ACID = Chem.MolFromSmarts("C(=O)[OH]")
ALCOHOL = Chem.MolFromSmarts("[CX4][OX2H]")
AMINE = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
ARYL_HALIDE = Chem.MolFromSmarts("c[F,Cl,Br,I]")
ACID_ANHYDRIDE = Chem.MolFromSmarts("C(=O)OC(=O)")
ACID_CHLORIDE = Chem.MolFromSmarts("C(=O)Cl")

SEMI_HYDROGENATION = AllChem.ReactionFromSmarts("[C:1]#[C:2]>>[C:1]=[C:2]")
FULL_HYDROGENATION = AllChem.ReactionFromSmarts("[C:1]#[C:2]>>[C:1]-[C:2]")


def _resolve_query(query: str) -> dict[str, Any]:
    if is_smiles(query):
        smiles = canonicalize_smiles(query)
        source = "input"
    else:
        payload = name_to_smiles(query)
        smiles = payload["smiles"]
        source = payload["source"]
    return {
        "query": query,
        "smiles": smiles,
        "source": source,
    }


def _molecule_flags(smiles: str) -> dict[str, bool]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {
        "has_terminal_alkyne": molecule.HasSubstructMatch(TERMINAL_ALKYNE),
        "has_alkyne": molecule.HasSubstructMatch(ALKYNE),
        "has_carboxylic_acid": molecule.HasSubstructMatch(CARBOXYLIC_ACID),
        "has_alcohol": molecule.HasSubstructMatch(ALCOHOL),
        "has_amine": molecule.HasSubstructMatch(AMINE),
        "has_aryl_halide": molecule.HasSubstructMatch(ARYL_HALIDE),
        "has_acid_anhydride": molecule.HasSubstructMatch(ACID_ANHYDRIDE),
        "has_acid_chloride": molecule.HasSubstructMatch(ACID_CHLORIDE),
    }


def _keyword_present(text: str, keywords: tuple[str, ...]) -> bool:
    for keyword in keywords:
        pattern = re.escape(keyword)
        negation_patterns = (
            rf"\bno\s+{pattern}\b",
            rf"\bwithout\s+{pattern}\b",
            rf"\babsence of\s+{pattern}\b",
            rf"\b{pattern}\s+free\b",
            rf"\bno\s+[^.(),;]{{0,24}}{pattern}\b",
            rf"\bwithout\s+[^.(),;]{{0,24}}{pattern}\b",
        )
        if any(re.search(expr, text) for expr in negation_patterns):
            continue
        if re.search(rf"\b{pattern}\b", text):
            return True
    return False


def _reagent_flags(reagents: list[str], conditions: str | None) -> dict[str, bool]:
    lowered = " ".join(reagents + ([conditions] if conditions else [])).lower()
    return {
        "has_lindlar": _keyword_present(lowered, ("lindlar", "[pd].[pb+2]")),
        "has_palladium": _keyword_present(lowered, ("pd", "palladium")),
        "has_hydrogen": _keyword_present(lowered, ("h2", "hydrogen gas", "hydrogen")),
        "has_base": _keyword_present(
            lowered, ("base", "triethylamine", "amine base", "k2co3", "na2co3")
        ),
        "has_coupling_partner": _keyword_present(
            lowered,
            (
                "boronic",
                "organoboron",
                "terminal alkyne",
                "alkyne partner",
                "stannane",
                "zinc reagent",
            ),
        ),
        "has_acid_catalyst": _keyword_present(
            lowered, ("sulfuric acid", "phosphoric acid", "acid catalyst")
        ),
        "has_acylating_agent": _keyword_present(
            lowered,
            ("acetic anhydride", "anhydride", "acid chloride", "acetyl chloride"),
        ),
    }


def _predict_transformation(smiles: str, reaction: AllChem.ChemicalReaction) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    products = reaction.RunReactants((molecule,))
    if not products:
        return None
    product = products[0][0]
    return Chem.MolToSmiles(product, canonical=True)


def reaction_outcome_heuristic(
    substrates: list[str], reagents: list[str] | None = None, conditions: str | None = None
) -> dict[str, Any]:
    reagents = reagents or []
    resolved = [_resolve_query(item) for item in substrates]
    enriched_substrates: list[dict[str, Any]] = []
    for item in resolved:
        item["flags"] = _molecule_flags(item["smiles"])
        item["functional_groups"] = functional_groups(item["smiles"])["functional_groups"]
        enriched_substrates.append(item)

    flags = _reagent_flags(reagents, conditions)
    rationale: list[str] = []
    plausible_outcomes: list[dict[str, Any]] = []
    missing_components: list[str] = []
    detected_family = "no_clear_reaction_family"
    confidence = "low"

    has_alkyne = any(item["flags"]["has_alkyne"] for item in enriched_substrates)
    has_aryl_halide = any(item["flags"]["has_aryl_halide"] for item in enriched_substrates)
    has_acid = any(item["flags"]["has_carboxylic_acid"] for item in enriched_substrates)
    has_alcohol = any(item["flags"]["has_alcohol"] for item in enriched_substrates)
    has_amine = any(item["flags"]["has_amine"] for item in enriched_substrates)

    if flags["has_lindlar"] and has_alkyne:
        detected_family = "alkyne_semi_hydrogenation_candidate"
        if not flags["has_hydrogen"]:
            confidence = "high"
            missing_components.append("hydrogen source (H2)")
            rationale.append(
                "Lindlar catalyst is a semihydrogenation catalyst, but no hydrogen source was provided."
            )
            plausible_outcomes.append(
                {
                    "label": "no_reaction_expected_under_stated_conditions",
                    "product_smiles": None,
                    "reason": "Catalyst alone is insufficient without hydrogen.",
                }
            )
            for item in enriched_substrates:
                if item["flags"]["has_alkyne"]:
                    product_smiles = _predict_transformation(
                        item["smiles"], SEMI_HYDROGENATION
                    )
                    plausible_outcomes.append(
                        {
                            "label": "counterfactual_if_h2_were_added",
                            "product_smiles": product_smiles,
                            "reason": "Under standard Lindlar + H2 conditions, alkyne semihydrogenation would be plausible.",
                        }
                    )
        else:
            confidence = "medium"
            rationale.append(
                "Lindlar catalyst with hydrogen supports selective semihydrogenation of alkynes."
            )
            for item in enriched_substrates:
                if item["flags"]["has_alkyne"]:
                    plausible_outcomes.append(
                        {
                            "label": "semi_hydrogenation_product",
                            "product_smiles": _predict_transformation(
                                item["smiles"], SEMI_HYDROGENATION
                            ),
                            "reason": "Selective reduction of the alkyne to an alkene is plausible.",
                        }
                    )

    elif flags["has_palladium"] and has_alkyne:
        detected_family = "palladium_mediated_alkyne_transformation_candidate"
        if flags["has_hydrogen"]:
            confidence = "medium"
            rationale.append(
                "Palladium with hydrogen often drives deeper hydrogenation than Lindlar conditions."
            )
            for item in enriched_substrates:
                if item["flags"]["has_alkyne"]:
                    plausible_outcomes.append(
                        {
                            "label": "full_hydrogenation_product",
                            "product_smiles": _predict_transformation(
                                item["smiles"], FULL_HYDROGENATION
                            ),
                            "reason": "Unpoisoned palladium typically favors more complete reduction of an alkyne.",
                        }
                    )
        else:
            confidence = "high"
            missing_components.append("hydrogen source or explicit coupling reagents")
            rationale.append(
                "Bare palladium without hydrogen or coupling reagents is unlikely to transform the alkyne on its own."
            )
            plausible_outcomes.append(
                {
                    "label": "no_reaction_expected_under_stated_conditions",
                    "product_smiles": None,
                    "reason": "No hydrogen source, base, or coupling partner was specified.",
                }
            )

    if has_aryl_halide and flags["has_palladium"] and not flags["has_coupling_partner"]:
        rationale.append(
            "An aryl halide is present, but no coupling partner was supplied, so cross-coupling is not well supported."
        )
        missing_components.append("explicit coupling partner")
        if not flags["has_base"]:
            missing_components.append("base for palladium-catalyzed coupling")

    if has_acid and has_alcohol and flags["has_acid_catalyst"]:
        detected_family = "esterification_candidate"
        confidence = "medium"
        plausible_outcomes.append(
            {
                "label": "esterification_family",
                "product_smiles": None,
                "reason": "A carboxylic acid and alcohol under acid catalysis are consistent with esterification at a high level.",
            }
        )

    if has_amine and flags["has_acylating_agent"]:
        detected_family = "amide_or_acylation_candidate"
        confidence = "medium"
        plausible_outcomes.append(
            {
                "label": "amine_acylation_family",
                "product_smiles": None,
                "reason": "An amine with an anhydride or acid chloride is consistent with amide/acetylation chemistry at a high level.",
            }
        )

    if not plausible_outcomes:
        rationale.append(
            "No strongly supported reaction family was inferred from the stated substrates, reagents, and conditions."
        )
        plausible_outcomes.append(
            {
                "label": "no_clear_prediction",
                "product_smiles": None,
                "reason": "The lightweight heuristic module does not support a high-confidence transformation here.",
            }
        )

    return {
        "substrates": enriched_substrates,
        "reagents": reagents,
        "conditions": conditions,
        "reagent_flags": flags,
        "detected_reaction_family": detected_family,
        "confidence": confidence,
        "missing_components": sorted(dict.fromkeys(item for item in missing_components if item)),
        "plausible_outcomes": plausible_outcomes,
        "rationale": rationale,
        "safety_mode": "non_operational_high_level_only",
    }


def retrosynthesis_overview(
    target: str, controlled_chemicals_path: str | None = None
) -> dict[str, Any]:
    resolved = _resolve_query(target)
    fg_payload = functional_groups(resolved["smiles"])
    groups = fg_payload["functional_groups"]
    route_families: list[dict[str, str]] = []
    caution = None

    if controlled_chemicals_path:
        control_result = control_chem_check(target, controlled_chemicals_path)
        if control_result["is_controlled"] and control_result["risk_tier"] == "hard_refusal":
            return {
                "target": target,
                "smiles": resolved["smiles"],
                "status": "restricted",
                "control_check": control_result,
                "message": "Only non-operational background is appropriate for this target because it matches a hard-refusal safety category.",
            }
        if control_result["is_controlled"]:
            caution = control_result
    else:
        control_result = None

    if "ester" in groups:
        route_families.append(
            {
                "family": "ester_formation",
                "overview": "Disconnect the ester into an alcohol/phenol partner and an activated carboxylic-acid-derived fragment.",
            }
        )
    if "amide" in groups:
        route_families.append(
            {
                "family": "amide_formation",
                "overview": "Consider disconnection to an amine plus an activated carboxylic acid derivative or coupling partner.",
            }
        )
    if "carboxylic acid" in groups:
        route_families.append(
            {
                "family": "carboxylic_acid_installation",
                "overview": "Common high-level families include oxidation of a primary alcohol/aldehyde or carboxylation-oriented disconnections.",
            }
        )
    if "alcohol" in groups or "phenol" in groups:
        route_families.append(
            {
                "family": "alcohol_or_phenol_precursor",
                "overview": "Hydroxyl-bearing motifs often trace back to reductions, substitutions, or deprotection-oriented route families.",
            }
        )
    if "amine" in groups:
        route_families.append(
            {
                "family": "amine_installation",
                "overview": "Amines often arise from nitro reduction, reductive amination, or substitution on a prefunctionalized scaffold.",
            }
        )
    if "nitrile" in groups:
        route_families.append(
            {
                "family": "nitrile_installation",
                "overview": "High-level options include cyanation of a leaving-group-bearing precursor or dehydration of an amide precursor.",
            }
        )

    if not route_families:
        route_families.append(
            {
                "family": "scaffold_level_disconnection",
                "overview": "No strong functional-group-specific disconnection was inferred; begin from the core scaffold and add substituents iteratively.",
            }
        )

    return {
        "target": target,
        "smiles": resolved["smiles"],
        "functional_groups": groups,
        "route_families": route_families,
        "control_check": control_result,
        "caution": caution,
        "non_operational_notice": "This retrosynthesis module is non-operational and provides route families only, not step-by-step procedures or conditions.",
    }
