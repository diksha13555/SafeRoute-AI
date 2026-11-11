"""Threshold sweep for the saved V3 model on 2014–2015 only.

Does not retrain, does not score 2016–2017, and does not change the
dashboard, simulator, or V3 checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train_saferoute_v3 import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    build_ego_vehicle_features,
    engineer_features,
)

ROOT = Path(__file__).resolve().parent
ACCIDENT_PATH = ROOT.parent / "Accident_Information.csv.zip"
VEHICLE_PATH = ROOT.parent / "Vehicle_Information.csv.zip"
MODEL_PATH = ROOT / "models" / "saferoute_v3_precrash_2026.pkl"
OUT_JSON = ROOT / "outputs" / "saferoute_v3_threshold_comparison.json"
OUT_CSV = ROOT / "outputs" / "saferoute_v3_threshold_comparison.csv"
DEPLOYED_THRESHOLD = 0.175


def load_validation_frame() -> pd.DataFrame:
    accidents = pd.read_csv(ACCIDENT_PATH, compression="zip", low_memory=False)
    accidents["Year"] = pd.to_numeric(accidents["Year"], errors="coerce")
    # Drop every other year before feature construction. 2016–2017 are not scored.
    accidents = accidents.loc[accidents["Year"].between(2014, 2015)].copy()
    accidents["Accident_Index"] = accidents["Accident_Index"].astype("string").str.strip()
    print(f"Validation accidents 2014–2015: {len(accidents):,}", flush=True)

    vehicles = pd.read_csv(VEHICLE_PATH, compression="zip", encoding="cp1252", low_memory=False)
    if "Engine_Capacity_.CC." in vehicles.columns and "Engine_Capacity" not in vehicles.columns:
        vehicles = vehicles.rename(columns={"Engine_Capacity_.CC.": "Engine_Capacity"})
    vehicles["Accident_Index"] = vehicles["Accident_Index"].astype("string").str.strip()
    vehicles = vehicles.loc[vehicles["Accident_Index"].isin(set(accidents["Accident_Index"]))].copy()
    ego = build_ego_vehicle_features(vehicles)
    del vehicles
    data = engineer_features(accidents, ego)
    data = data.loc[data["Year"].between(2014, 2015)].copy()
    if data["Year"].max() > 2015 or data["Year"].min() < 2014:
        raise RuntimeError("Validation frame contains years outside 2014–2015.")
    print(f"Validation rows after labels: {len(data):,}  KSI rate={data['KSI'].mean():.4f}", flush=True)
    return data


def encode_with_checkpoint(frame: pd.DataFrame, checkpoint: dict) -> pd.DataFrame:
    levels = checkpoint["category_levels"]
    matrix = frame[FEATURE_NAMES].copy()
    for col in CATEGORICAL_FEATURES:
        dtype = pd.CategoricalDtype(categories=list(levels[col]), ordered=False)
        matrix[col] = matrix[col].astype("string").str.strip().astype(dtype)
    for col in FEATURE_NAMES:
        if col not in CATEGORICAL_FEATURES:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce").astype("float64")
    if list(matrix.columns) != list(checkpoint["feature_names"]):
        raise RuntimeError("Feature order does not match the saved V3 checkpoint.")
    return matrix


def metrics_at(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "ksi_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "ksi_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "ksi_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    checkpoint = joblib.load(MODEL_PATH)
    model = checkpoint["model"]
    if list(model.feature_name_) != list(FEATURE_NAMES):
        raise RuntimeError("Loaded V3 feature order does not match the training script.")

    data = load_validation_frame()
    matrix = encode_with_checkpoint(data, checkpoint)
    y_true = data["KSI"].to_numpy(dtype=int)
    probability = model.predict_proba(matrix)[:, 1]
    print(f"Scored {len(y_true):,} validation rows.", flush=True)

    thresholds = np.round(np.arange(0.05, 0.3001, 0.005), 3)
    rows = [metrics_at(y_true, probability, float(threshold)) for threshold in thresholds]
    table = pd.DataFrame(rows)

    best_f1_row = table.loc[table["ksi_f1"].idxmax()]
    eligible = table.loc[table["ksi_recall"] >= 0.60]
    recall_target = None if eligible.empty else eligible.loc[eligible["ksi_precision"].idxmax()]
    deployed = metrics_at(y_true, probability, DEPLOYED_THRESHOLD)

    def pack(row) -> dict:
        if row is None:
            return None
        if isinstance(row, pd.Series):
            threshold = float(row["threshold"])
            return metrics_at(y_true, probability, threshold)
        return row

    best_f1 = pack(best_f1_row)
    recall_choice = pack(recall_target)

    result = {
        "model": "SafeRoute-AI V3",
        "checkpoint": str(MODEL_PATH),
        "validation_years": "2014-2015",
        "holdout_2016_2017": "Not loaded and not scored.",
        "rows": int(len(y_true)),
        "ksi_count": int(y_true.sum()),
        "ksi_prevalence": float(y_true.mean()),
        "deployed_threshold": DEPLOYED_THRESHOLD,
        "threshold_grid": "0.05 to 0.30 inclusive, step 0.005",
        "thresholds": rows,
        "max_ksi_f1": best_f1,
        "recall_at_least_0.60_best_precision": recall_choice,
        "deployed_threshold_remeasured": deployed,
    }
    if recall_choice is None:
        result["recall_target_note"] = "No threshold from 0.05 to 0.30 reaches 60% KSI recall."
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    table.drop(columns=["confusion_matrix"]).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_JSON}", flush=True)
    print(f"Saved {OUT_CSV}", flush=True)
    print("\nDeployed 0.175", deployed, flush=True)
    print("Max F1", best_f1, flush=True)
    print("Recall>=0.60", recall_choice, flush=True)


if __name__ == "__main__":
    main()
