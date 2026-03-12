from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 1024
HOLDOUT_TEST_SIZE = 0.2
MODEL_RANDOM_STATE = 42
CV_FOLDS = 5


def _load_raw_dataframe(data_path: Path) -> pd.DataFrame:
    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_path, sheet_name="Published_version")
    return pd.read_csv(data_path)


def prepare_acetonitrile_dataset(data_path: Path) -> pd.DataFrame:
    dataframe = _load_raw_dataframe(data_path).copy()
    dataframe = dataframe.rename(
        columns={
            "Chromophore": "smiles",
            "Solvent": "solvent",
            "Absorption max": "absorption_nm",
            "Emission max": "emission_nm",
            "lifetime": "lifetime_ns",
        }
    )
    filtered = dataframe.loc[
        dataframe["solvent"] == "CC#N", ["smiles", "absorption_nm", "solvent"]
    ].dropna(subset=["smiles", "absorption_nm"])
    filtered = filtered.drop_duplicates(subset=["smiles", "absorption_nm"]).reset_index(
        drop=True
    )
    return filtered


def chromophore_dataset_summary(data_path: Path) -> dict[str, Any]:
    raw = _load_raw_dataframe(data_path)
    prepared = prepare_acetonitrile_dataset(data_path)
    return {
        "data_path": str(data_path),
        "raw_rows": int(len(raw)),
        "raw_columns": list(raw.columns),
        "acetonitrile_rows_after_cleaning": int(len(prepared)),
        "unique_acetonitrile_smiles": int(prepared["smiles"].nunique()),
        "missing_counts": raw.isna().sum().to_dict(),
    }


def _build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, random_state=MODEL_RANDOM_STATE)


def _smiles_to_array(smiles: str, n_bits: int = FINGERPRINT_BITS) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES in dataset: {smiles}")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=n_bits
    )
    bit_vector = generator.GetFingerprint(molecule)
    array = np.zeros((n_bits,), dtype=float)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def _build_proxy_pool(raw: pd.DataFrame, training_smiles: set[str]) -> list[str]:
    unique_smiles = (
        raw["Chromophore"].dropna().astype(str).drop_duplicates().tolist()
        if "Chromophore" in raw.columns
        else raw["smiles"].dropna().astype(str).drop_duplicates().tolist()
    )
    return [smiles for smiles in unique_smiles if smiles not in training_smiles]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate_holdout(features: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
    )

    model = _build_random_forest()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(x_train, y_train)
    dummy_predictions = dummy.predict(x_test)

    return {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "holdout_rmse_nm": round(_rmse(y_test, predictions), 4),
        "dummy_holdout_rmse_nm": round(_rmse(y_test, dummy_predictions), 4),
    }


def _evaluate_cross_validation(
    features: np.ndarray, labels: np.ndarray
) -> dict[str, float | list[float]]:
    splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=MODEL_RANDOM_STATE)
    rf_scores: list[float] = []
    dummy_scores: list[float] = []

    for train_idx, test_idx in splitter.split(features):
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = _build_random_forest()
        model.fit(x_train, y_train)
        rf_scores.append(_rmse(y_test, model.predict(x_test)))

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(x_train, y_train)
        dummy_scores.append(_rmse(y_test, dummy.predict(x_test)))

    return {
        "cv_folds": CV_FOLDS,
        "cv_rmse_mean_nm": round(float(np.mean(rf_scores)), 4),
        "cv_rmse_std_nm": round(float(np.std(rf_scores)), 4),
        "dummy_cv_rmse_mean_nm": round(float(np.mean(dummy_scores)), 4),
        "dummy_cv_rmse_std_nm": round(float(np.std(dummy_scores)), 4),
        "fold_rmse_nm": [round(value, 4) for value in rf_scores],
        "dummy_fold_rmse_nm": [round(value, 4) for value in dummy_scores],
    }


def chromophore_rf_screen(
    data_path: Path,
    target_nm: float = 369.0,
    top_k: int = 5,
    candidate_pool_path: Path | None = None,
) -> dict[str, Any]:
    prepared = prepare_acetonitrile_dataset(data_path)
    features = np.vstack(prepared["smiles"].map(_smiles_to_array).tolist())
    labels = prepared["absorption_nm"].to_numpy(dtype=float)

    holdout_metrics = _evaluate_holdout(features, labels)
    cv_metrics = _evaluate_cross_validation(features, labels)

    final_model = _build_random_forest()
    final_model.fit(features, labels)

    if candidate_pool_path and candidate_pool_path.exists():
        pool = [
            line.strip()
            for line in candidate_pool_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pool_strategy = "external_pool_file"
    else:
        raw = _load_raw_dataframe(data_path)
        pool = _build_proxy_pool(raw, set(prepared["smiles"]))
        pool_strategy = (
            "proxy_pool_from_unique_dataset_smiles_excluding_acetonitrile_training_smiles"
        )

    ranked: list[dict[str, float | str]] = []
    if pool:
        pool_features = np.vstack([_smiles_to_array(smiles) for smiles in pool])
        pool_predictions = final_model.predict(pool_features)
        ranked = sorted(
            [
                {
                    "smiles": smiles,
                    "predicted_absorption_nm": float(prediction),
                    "distance_to_target_nm": float(abs(prediction - target_nm)),
                }
                for smiles, prediction in zip(pool, pool_predictions)
            ],
            key=lambda item: item["distance_to_target_nm"],
        )

    model_advantage = round(
        float(cv_metrics["dummy_cv_rmse_mean_nm"]) - float(cv_metrics["cv_rmse_mean_nm"]),
        4,
    )

    return {
        "data_path": str(data_path),
        "target_nm": target_nm,
        "target_property": "absorption_nm",
        "solvent_filter": "CC#N",
        "prepared_rows": int(len(prepared)),
        "unique_training_smiles": int(prepared["smiles"].nunique()),
        "rmse_nm": holdout_metrics["holdout_rmse_nm"],
        "holdout_rmse_nm": holdout_metrics["holdout_rmse_nm"],
        "dummy_holdout_rmse_nm": holdout_metrics["dummy_holdout_rmse_nm"],
        "holdout_train_rows": holdout_metrics["train_rows"],
        "holdout_test_rows": holdout_metrics["test_rows"],
        "cv_folds": cv_metrics["cv_folds"],
        "cv_rmse_mean_nm": cv_metrics["cv_rmse_mean_nm"],
        "cv_rmse_std_nm": cv_metrics["cv_rmse_std_nm"],
        "dummy_cv_rmse_mean_nm": cv_metrics["dummy_cv_rmse_mean_nm"],
        "dummy_cv_rmse_std_nm": cv_metrics["dummy_cv_rmse_std_nm"],
        "fold_rmse_nm": cv_metrics["fold_rmse_nm"],
        "dummy_fold_rmse_nm": cv_metrics["dummy_fold_rmse_nm"],
        "model_advantage_over_dummy_nm": model_advantage,
        "fingerprint": {
            "type": "Morgan",
            "radius": FINGERPRINT_RADIUS,
            "n_bits": FINGERPRINT_BITS,
        },
        "model": {
            "name": "RandomForestRegressor",
            "n_estimators": 100,
            "random_state": MODEL_RANDOM_STATE,
        },
        "screening_model_fit": "fit_on_full_filtered_dataset_after_holdout_and_cv_evaluation",
        "candidate_pool_size": len(pool),
        "candidate_pool_strategy": pool_strategy,
        "best_candidate": ranked[0] if ranked else None,
        "top_candidates": ranked[:top_k],
    }
