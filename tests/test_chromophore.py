from pathlib import Path

from chemcrow_lite.tools.chromophore import (
    chromophore_dataset_summary,
    chromophore_rf_screen,
    prepare_acetonitrile_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "DB for chromophore_Sci_Data.xlsx"


def test_prepare_acetonitrile_dataset_nonempty() -> None:
    dataframe = prepare_acetonitrile_dataset(DATA_PATH)
    assert len(dataframe) > 1500
    assert set(dataframe.columns) == {"smiles", "absorption_nm", "solvent"}
    assert dataframe["solvent"].eq("CC#N").all()


def test_dataset_summary_matches_expectation() -> None:
    summary = chromophore_dataset_summary(DATA_PATH)
    assert summary["raw_rows"] > 20000
    assert summary["acetonitrile_rows_after_cleaning"] > 1500


def test_rf_workflow_returns_ranked_candidates() -> None:
    result = chromophore_rf_screen(DATA_PATH, target_nm=369.0, top_k=3)
    assert result["rmse_nm"] > 0
    assert result["holdout_rmse_nm"] > 0
    assert result["cv_rmse_mean_nm"] > 0
    assert result["dummy_cv_rmse_mean_nm"] > 0
    assert result["model_advantage_over_dummy_nm"] > 0
    assert result["candidate_pool_size"] > 1000
    assert len(result["top_candidates"]) == 3
    assert result["best_candidate"]["smiles"]
