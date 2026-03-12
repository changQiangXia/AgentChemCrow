from pathlib import Path

from chemcrow_lite.tools.reaction import (
    reaction_outcome_heuristic,
    retrosynthesis_overview,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROLLED = ROOT / "data" / "controlled_chemicals_min.csv"


def test_lindlar_without_hydrogen_predicts_no_reaction() -> None:
    result = reaction_outcome_heuristic(
        substrates=["1-Chloro-4-ethynylbenzene"],
        reagents=["Lindlar catalyst"],
    )
    assert result["detected_reaction_family"] == "alkyne_semi_hydrogenation_candidate"
    assert "hydrogen source (H2)" in result["missing_components"]
    assert result["plausible_outcomes"][0]["label"] == "no_reaction_expected_under_stated_conditions"


def test_negated_conditions_do_not_count_as_hydrogen_or_base() -> None:
    result = reaction_outcome_heuristic(
        substrates=["1-Chloro-4-ethynylbenzene"],
        reagents=["[Pd]"],
        conditions="no hydrogen gas, no base, no coupling partner",
    )
    assert result["reagent_flags"]["has_hydrogen"] is False
    assert result["reagent_flags"]["has_base"] is False


def test_lindlar_with_hydrogen_has_counterfactual_product() -> None:
    result = reaction_outcome_heuristic(
        substrates=["1-Chloro-4-ethynylbenzene"],
        reagents=["Lindlar catalyst", "H2"],
    )
    labels = [item["label"] for item in result["plausible_outcomes"]]
    assert "semi_hydrogenation_product" in labels


def test_retrosynthesis_overview_for_aspirin_is_non_operational() -> None:
    result = retrosynthesis_overview("aspirin", controlled_chemicals_path=str(CONTROLLED))
    assert result["route_families"]
    assert "non-operational" in result["non_operational_notice"].lower()
