from pathlib import Path

from chemcrow_lite.tools.safety import control_chem_check

ROOT = Path(__file__).resolve().parents[1]
CONTROLLED = ROOT / "data" / "controlled_chemicals_min.csv"


def test_control_chem_check_name_match() -> None:
    result = control_chem_check("nitroglycerin", CONTROLLED)
    assert result["is_controlled"] is True
    assert result["match_type"] == "name"
    assert result["risk_tier"] == "hard_refusal"


def test_control_chem_check_alias_match() -> None:
    result = control_chem_check("gtn", CONTROLLED)
    assert result["is_controlled"] is True
    assert result["match_type"] == "alias"
    assert result["matched_name"] == "nitroglycerin"


def test_control_chem_check_watchlist_precursor() -> None:
    result = control_chem_check("acetic anhydride", CONTROLLED)
    assert result["is_controlled"] is True
    assert result["risk_tier"] == "watchlist"


def test_control_chem_check_unknown_is_not_controlled() -> None:
    result = control_chem_check("acetaminophen", CONTROLLED)
    assert result["is_controlled"] is False
