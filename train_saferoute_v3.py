"""
SafeRoute-AI V3 — Pre-crash KSI severity model

Why V2 underperformed V1
------------------------
V1 (32 features, ROC-AUC ~0.72) used post-crash / crash-composition vehicle
signals (vehicle counts, other-road-user types, manoeuvre, max/mean fleet stats).
V2 correctly removed those, leaving 16 journey/road/weather attributes, which
cut ROC-AUC to ~0.64. That drop is expected: V2 is the honest pre-crash task.

V3 goal
-------
Improve over V2 while staying pre-crash legitimate:
  1) Restore map-available road infrastructure excluded from V2
  2) Add ego-vehicle / driver profile features known before travel
  3) Stronger temporal & risk interaction engineering
  4) Tuned LightGBM + validation threshold selection

Splits (unchanged):
  Train 2009–2013 | Validation 2014–2015 | Holdout 2016–2017 (eval only)

Does not overwrite V1 or V2 checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Dataset zips sit next to the project directory. Override on the command line.
ROOT = Path(__file__).resolve().parent
DEFAULT_ACCIDENT = ROOT.parent / "Accident_Information.csv.zip"
DEFAULT_VEHICLE = ROOT.parent / "Vehicle_Information.csv.zip"
DEFAULT_OUT_DIR = ROOT / "models"
DEFAULT_METRICS_DIR = ROOT / "outputs"

TRAIN_YEARS = (2009, 2013)
VALID_YEARS = (2014, 2015)
HOLDOUT_YEARS = (2016, 2017)

# Map / forecast / journey attributes available before travel
ROAD_CATEGORICAL = [
    "1st_Road_Class",
    "2nd_Road_Class",
    "Day_of_Week",
    "Junction_Control",
    "Junction_Detail",
    "Light_Conditions",
    "Road_Surface_Conditions",
    "Road_Type",
    "Urban_or_Rural_Area",
    "Weather_Conditions",
    "Pedestrian_Crossing-Physical_Facilities",
    "Pedestrian_Crossing-Human_Control",
]

# Ego-vehicle / driver profile (user-known before travel; NOT other parties)
EGO_CATEGORICAL = [
    "Ego_Vehicle_Type",
    "Ego_Driver_Age_Band",
    "Ego_Journey_Purpose",
]

BASE_NUMERIC = [
    "Speed_limit",
    "Hour",
    "Month",
    "Latitude",
    "Longitude",
    "Weekend",
    "Rush_Hour",
    "Night",
    "Year",
    "Ego_Vehicle_Age",
    "Ego_Engine_Capacity",
]

ENGINEERED_NUMERIC = [
    "Hour_sin",
    "Hour_cos",
    "Month_sin",
    "Month_cos",
    "Is_High_Speed",
    "Is_Dark_Rural",
    "Is_Adverse_Weather",
    "Is_Wet_Surface",
    "Urban_High_Speed",
]

CATEGORICAL_FEATURES = ROAD_CATEGORICAL + EGO_CATEGORICAL
FEATURE_NAMES = CATEGORICAL_FEATURES + BASE_NUMERIC + ENGINEERED_NUMERIC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SafeRoute-AI V3 pre-crash model")
    p.add_argument("--accident-path", type=Path, default=DEFAULT_ACCIDENT)
    p.add_argument("--vehicle-path", type=Path, default=DEFAULT_VEHICLE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    p.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip 2016–2017 evaluation (still trains/tunes on train/val only)",
    )
    return p.parse_args()


def require_files(accident_path: Path, vehicle_path: Path) -> None:
    missing = [str(p) for p in (accident_path, vehicle_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required dataset files not found:\n  - "
            + "\n  - ".join(missing)
            + "\n\nNeeded files:\n"
            "  1) Accident_Information.csv.zip  (or .csv)\n"
            "  2) Vehicle_Information.csv.zip   (or .csv, encoding cp1252)\n"
            "Pass paths with --accident-path / --vehicle-path."
        )


def load_accidents(path: Path) -> pd.DataFrame:
    print(f"Loading accidents from {path} ...")
    kwargs = {"low_memory": False}
    if path.suffix.lower() == ".zip":
        kwargs["compression"] = "zip"
    df = pd.read_csv(path, **kwargs)
    print(f"  Accidents shape: {df.shape}")
    return df


def load_vehicles(path: Path) -> pd.DataFrame:
    print(f"Loading vehicles from {path} ...")
    kwargs = {"low_memory": False, "encoding": "cp1252"}
    if path.suffix.lower() == ".zip":
        kwargs["compression"] = "zip"
    df = pd.read_csv(path, **kwargs)
    # Normalize engine capacity column name across dataset variants
    if "Engine_Capacity_.CC." in df.columns and "Engine_Capacity" not in df.columns:
        df = df.rename(columns={"Engine_Capacity_.CC.": "Engine_Capacity"})
    print(f"  Vehicles shape: {df.shape}")
    return df


def build_ego_vehicle_features(vehicles: pd.DataFrame) -> pd.DataFrame:
    """One ego-vehicle row per Accident_Index (lowest Vehicle_Reference)."""
    v = vehicles.copy()
    v["Accident_Index"] = v["Accident_Index"].astype("string").str.strip()
    v["Vehicle_Reference"] = pd.to_numeric(v["Vehicle_Reference"], errors="coerce")
    v["Age_of_Vehicle"] = pd.to_numeric(v["Age_of_Vehicle"], errors="coerce")
    v["Engine_Capacity"] = pd.to_numeric(v["Engine_Capacity"], errors="coerce")
    v.loc[v["Age_of_Vehicle"] < 0, "Age_of_Vehicle"] = np.nan
    v.loc[v["Engine_Capacity"] < 0, "Engine_Capacity"] = np.nan

    first = (
        v.sort_values(["Accident_Index", "Vehicle_Reference"], kind="stable", na_position="last")
        .drop_duplicates("Accident_Index", keep="first")
    )

    ego = first[
        [
            "Accident_Index",
            "Vehicle_Type",
            "Age_Band_of_Driver",
            "Journey_Purpose_of_Driver",
            "Age_of_Vehicle",
            "Engine_Capacity",
        ]
    ].rename(
        columns={
            "Vehicle_Type": "Ego_Vehicle_Type",
            "Age_Band_of_Driver": "Ego_Driver_Age_Band",
            "Journey_Purpose_of_Driver": "Ego_Journey_Purpose",
            "Age_of_Vehicle": "Ego_Vehicle_Age",
            "Engine_Capacity": "Ego_Engine_Capacity",
        }
    )
    return ego


def engineer_features(accidents: pd.DataFrame, ego: pd.DataFrame) -> pd.DataFrame:
    data = accidents.copy()
    data["Accident_Index"] = data["Accident_Index"].astype("string").str.strip()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")

    # Keep development + holdout years only
    data = data.loc[data["Year"].between(TRAIN_YEARS[0], HOLDOUT_YEARS[1])].copy()

    # Hour / Month
    data["Hour"] = pd.to_numeric(
        data["Time"].astype("string").str.extract(r"^(\d{1,2}):\d{2}", expand=False),
        errors="coerce",
    )
    data.loc[~data["Hour"].between(0, 23), "Hour"] = np.nan

    parsed_dates = pd.to_datetime(
        data["Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    data["Month"] = parsed_dates.dt.month

    day = data["Day_of_Week"].astype("string").str.strip().str.lower()
    data["Weekend"] = day.isin(["saturday", "sunday"]).astype("float64")

    data["Rush_Hour"] = data["Hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype("float64")
    data["Night"] = ((data["Hour"] < 6) | (data["Hour"] >= 20)).astype("float64")
    missing_hour = data["Hour"].isna()
    data.loc[missing_hour, ["Rush_Hour", "Night"]] = np.nan

    speed = pd.to_numeric(data["Speed_limit"], errors="coerce")
    speed = speed.where(speed.between(10, 70))
    data["Speed_limit"] = speed

    # Cyclical time
    hour = data["Hour"]
    month = data["Month"]
    data["Hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    data["Hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    data["Month_sin"] = np.sin(2 * np.pi * month / 12.0)
    data["Month_cos"] = np.cos(2 * np.pi * month / 12.0)

    urban = data["Urban_or_Rural_Area"].astype("string").str.strip().str.lower()
    is_rural = urban.str.contains("rural", na=False)
    is_urban = urban.str.contains("urban", na=False)

    data["Is_High_Speed"] = (speed >= 60).astype("float64")
    data.loc[speed.isna(), "Is_High_Speed"] = np.nan

    data["Is_Dark_Rural"] = (
        (data["Night"] == 1) & is_rural
    ).astype("float64")
    data.loc[data["Night"].isna() | urban.isna(), "Is_Dark_Rural"] = np.nan

    weather = data["Weather_Conditions"].astype("string").str.strip().str.lower()
    fine_like = weather.str.contains("fine", na=False) | weather.eq("unknown")
    data["Is_Adverse_Weather"] = (~fine_like & weather.notna()).astype("float64")
    data.loc[weather.isna(), "Is_Adverse_Weather"] = np.nan

    surface = data["Road_Surface_Conditions"].astype("string").str.strip().str.lower()
    wet_like = surface.str.contains(
        "wet|damp|flood|snow|ice|frost", regex=True, na=False
    )
    data["Is_Wet_Surface"] = wet_like.astype("float64")
    data.loc[surface.isna(), "Is_Wet_Surface"] = np.nan

    data["Urban_High_Speed"] = (
        (data["Is_High_Speed"] == 1) & is_urban
    ).astype("float64")
    data.loc[data["Is_High_Speed"].isna() | urban.isna(), "Urban_High_Speed"] = np.nan

    # Left-join ego features (no post-crash other-party composition)
    data = data.merge(ego, on="Accident_Index", how="left")

    # Target
    sev = data["Accident_Severity"].astype("string").str.strip().str.lower()
    valid = sev.isin(["fatal", "serious", "slight"])
    dropped = int((~valid).sum())
    if dropped:
        print(f"  Dropping {dropped} rows with unknown severity labels")
    data = data.loc[valid].copy()
    data["KSI"] = sev.loc[valid].isin(["fatal", "serious"]).astype(int)

    missing_feats = [c for c in FEATURE_NAMES if c not in data.columns]
    if missing_feats:
        raise KeyError(f"Missing engineered features: {missing_feats}")

    print(f"  Engineered rows (2009–2017): {len(data):,}")
    print(f"  Feature count: {len(FEATURE_NAMES)}")
    print(f"  Ego match rate: {data['Ego_Vehicle_Type'].notna().mean():.2%}")
    return data


def encode_matrices(
    train: pd.DataFrame, *others: pd.DataFrame
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict]:
    """Fit categorical levels on train only; apply to all splits."""
    X_train = train[FEATURE_NAMES].copy()
    Xs = [o[FEATURE_NAMES].copy() for o in others]

    category_levels: dict[str, list] = {}
    for col in CATEGORICAL_FEATURES:
        train_vals = X_train[col].astype("string").str.strip()
        levels = sorted(train_vals.dropna().unique().tolist())
        category_levels[col] = levels
        dtype = pd.CategoricalDtype(categories=levels, ordered=False)
        X_train[col] = train_vals.astype(dtype)
        for X in Xs:
            X[col] = X[col].astype("string").str.strip().astype(dtype)

    numeric_cols = [c for c in FEATURE_NAMES if c not in CATEGORICAL_FEATURES]
    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").astype("float64")
        for X in Xs:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")

    return X_train, Xs, category_levels


def evaluate(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Slight", "KSI"],
            zero_division=0,
            output_dict=True,
        ),
    }


def select_threshold(y_true, y_prob) -> tuple[float, pd.DataFrame]:
    rows = []
    for thr in np.round(np.arange(0.05, 0.51, 0.025), 3):
        pred = (y_prob >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    table = pd.DataFrame(rows)
    best = table.loc[table["f1"].idxmax()]
    return float(best["threshold"]), table


def train_candidates(
    X_train, y_train, X_valid, y_valid, scale_pos_weight: float
) -> tuple[lgb.LGBMClassifier, dict, dict]:
    """Small focused grid; select by validation PR-AUC then ROC-AUC."""
    grid = [
        {
            "name": "v2_like_weighted",
            "params": dict(
                n_estimators=800,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=150,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
            ),
        },
        {
            "name": "deeper_regularized",
            "params": dict(
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=100,
                subsample=0.85,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=3.0,
                scale_pos_weight=scale_pos_weight,
            ),
        },
        {
            "name": "mild_weight",
            "params": dict(
                n_estimators=1000,
                learning_rate=0.035,
                num_leaves=47,
                min_child_samples=120,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=2.0,
                scale_pos_weight=max(1.0, scale_pos_weight * 0.55),
            ),
        },
        {
            "name": "unweighted_earlystop",
            "params": dict(
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=150,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.5,
                scale_pos_weight=1.0,
            ),
        },
    ]

    best_model = None
    best_meta = None
    best_prob = None
    best_key = (-1.0, -1.0)

    for cfg in grid:
        print(f"\n  Training candidate: {cfg['name']}")
        model = lgb.LGBMClassifier(
            objective="binary",
            subsample_freq=1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **cfg["params"],
        )
        model.fit(
            X_train,
            y_train,
            categorical_feature=CATEGORICAL_FEATURES,
            eval_X=X_valid,
            eval_y=y_valid,
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(60, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        prob = model.predict_proba(X_valid)[:, 1]
        pr = average_precision_score(y_valid, prob)
        roc = roc_auc_score(y_valid, prob)
        print(
            f"    best_iter={model.best_iteration_}  "
            f"PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}"
        )
        key = (pr, roc)
        if key > best_key:
            best_key = key
            best_model = model
            best_prob = prob
            best_meta = {
                "candidate": cfg["name"],
                "params": cfg["params"],
                "best_iteration": int(model.best_iteration_ or cfg["params"]["n_estimators"]),
                "valid_pr_auc": float(pr),
                "valid_roc_auc": float(roc),
            }

    assert best_model is not None and best_meta is not None and best_prob is not None
    return best_model, best_meta, {"valid_prob": best_prob}


def print_metrics(title: str, metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  Threshold : {metrics['threshold']:.3f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
    print(
        "  Confusion : "
        f"TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}"
    )


def main() -> None:
    args = parse_args()
    require_files(args.accident_path, args.vehicle_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 70)
    print("SafeRoute-AI | V3 PRE-CRASH KSI MODEL")
    print("=" * 70)

    accidents = load_accidents(args.accident_path)
    vehicles = load_vehicles(args.vehicle_path)
    ego = build_ego_vehicle_features(vehicles)
    data = engineer_features(accidents, ego)

    train = data.loc[data["Year"].between(*TRAIN_YEARS)].copy()
    valid = data.loc[data["Year"].between(*VALID_YEARS)].copy()
    holdout = data.loc[data["Year"].between(*HOLDOUT_YEARS)].copy()

    print("\nSPLIT SIZES")
    print(f"  Train   {TRAIN_YEARS}: {len(train):,}  KSI rate={train['KSI'].mean():.4f}")
    print(f"  Valid   {VALID_YEARS}: {len(valid):,}  KSI rate={valid['KSI'].mean():.4f}")
    print(f"  Holdout {HOLDOUT_YEARS}: {len(holdout):,}  KSI rate={holdout['KSI'].mean():.4f}")

    X_train, (X_valid, X_holdout), category_levels = encode_matrices(train, valid, holdout)
    y_train = train["KSI"].astype(int).to_numpy()
    y_valid = valid["KSI"].astype(int).to_numpy()
    y_holdout = holdout["KSI"].astype(int).to_numpy()

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    print(f"\nClass balance (train): Slight={neg:,}  KSI={pos:,}  scale_pos_weight={spw:.4f}")

    print("\nMODEL SELECTION (validation PR-AUC primary)")
    model, model_meta, preds = train_candidates(X_train, y_train, X_valid, y_valid, spw)

    valid_prob = preds["valid_prob"]
    best_thr, thr_table = select_threshold(y_valid, valid_prob)
    print("\nTop validation thresholds by F1:")
    print(thr_table.sort_values("f1", ascending=False).head(8).round(4).to_string(index=False))

    valid_metrics = evaluate(y_valid, valid_prob, best_thr)
    # Also report V2's provisional threshold for apples-to-apples ops comparison
    valid_at_015 = evaluate(y_valid, valid_prob, 0.15)
    print_metrics("VALIDATION @ best-F1 threshold", valid_metrics)
    print_metrics("VALIDATION @ threshold 0.15 (V2 ops setting)", valid_at_015)

    holdout_metrics = None
    if not args.skip_holdout:
        holdout_prob = model.predict_proba(X_holdout)[:, 1]
        holdout_metrics = evaluate(y_holdout, holdout_prob, best_thr)
        holdout_at_015 = evaluate(y_holdout, holdout_prob, 0.15)
        print_metrics("HOLDOUT 2016–2017 @ best-F1 threshold", holdout_metrics)
        print_metrics("HOLDOUT 2016–2017 @ threshold 0.15", holdout_at_015)
    else:
        holdout_at_015 = None

    importance = (
        pd.Series(model.feature_importances_, index=FEATURE_NAMES)
        .sort_values(ascending=False)
    )
    print("\nTOP FEATURE IMPORTANCES")
    print(importance.head(20).to_string())

    # Reference baselines from notebook (not recomputed here)
    baselines = {
        "V1_32_features_validation": {
            "roc_auc": 0.7195,
            "pr_auc": 0.3325,
            "ksi_f1": 0.3759,
            "note": "Includes post-crash / crash-composition vehicle features",
        },
        "V2_16_features_validation": {
            "roc_auc": 0.6361,
            "pr_auc": 0.2426,
            "accuracy_at_0.15": 0.6003,
            "note": "Strict pre-crash subset; current dashboard model",
        },
    }

    v3_beats_v2 = (
        valid_metrics["roc_auc"] > baselines["V2_16_features_validation"]["roc_auc"]
        and valid_metrics["pr_auc"] > baselines["V2_16_features_validation"]["pr_auc"]
    )

    recommendation = {
        "replace_v2": bool(v3_beats_v2),
        "rationale": (
            "V3 improves validation ROC-AUC and PR-AUC over V2 while remaining "
            "pre-crash legitimate (map/road + ego-vehicle profile + engineered "
            "risk flags). It will not match V1, which used post-crash leakage."
            if v3_beats_v2
            else
            "V3 did not clearly beat V2 on both ROC-AUC and PR-AUC; keep V2 "
            "in production until a stronger pre-crash uplift is demonstrated."
        ),
    }

    checkpoint = {
        "model": model,
        "model_name": "SafeRoute-AI V3",
        "model_type": "Tuned LightGBM binary classifier (pre-crash)",
        "target": "KSI vs Slight",
        "prediction_context": (
            "KSI severity conditional on a recorded accident; "
            "not absolute accident probability"
        ),
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "category_levels": category_levels,
        "training_years": list(range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1)),
        "validation_years": list(range(VALID_YEARS[0], VALID_YEARS[1] + 1)),
        "holdout_years": list(range(HOLDOUT_YEARS[0], HOLDOUT_YEARS[1] + 1)),
        "excluded_postcrash_examples": [
            "Vehicle_Record_Count",
            "Has_Motorcycle / Has_Bicycle / Has_Bus / Has_Goods_Vehicle",
            "Distinct_Vehicle_Types",
            "Mean/Max vehicle stats across all parties",
            "First_Vehicle_Manoeuvre",
            "Number_of_Vehicles",
            "Number_of_Casualties",
            "Casualty composition features",
        ],
        "feature_engineering": {
            "Hour": "Extract hour from Time",
            "Month": "Parse Date dayfirst/mixed",
            "Weekend": "Saturday or Sunday",
            "Rush_Hour": "Hour in [7,8,9,16,17,18,19]",
            "Night": "Hour < 6 or Hour >= 20",
            "Hour_sin/cos": "Cyclical hour encoding",
            "Month_sin/cos": "Cyclical month encoding",
            "Is_High_Speed": "Speed_limit >= 60",
            "Is_Dark_Rural": "Night and rural",
            "Is_Adverse_Weather": "Weather not fine/unknown",
            "Is_Wet_Surface": "Wet/damp/flood/snow/ice/frost",
            "Urban_High_Speed": "Urban and Speed_limit >= 60",
            "Ego_*": "Primary vehicle (lowest Vehicle_Reference) as user proxy",
            "missing_hour": "Preserve as NaN",
        },
        "model_selection": model_meta,
        "selected_threshold": best_thr,
        "threshold_status": "Selected by validation KSI F1; holdout not used for selection",
        "validation_metrics": valid_metrics,
        "validation_metrics_at_0.15": valid_at_015,
        "holdout_metrics": holdout_metrics,
        "holdout_metrics_at_0.15": holdout_at_015,
        "feature_importance_gain_proxy": importance.to_dict(),
        "baselines_from_notebook": baselines,
        "recommendation": recommendation,
        "status": "Experimental V3 development checkpoint",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    model_path = args.out_dir / "saferoute_v3_precrash_2026.pkl"
    metrics_path = args.metrics_dir / "saferoute_v3_metrics.json"
    importance_path = args.metrics_dir / "saferoute_v3_feature_importance.csv"

    joblib.dump(checkpoint, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                k: v
                for k, v in checkpoint.items()
                if k != "model"
            },
            f,
            indent=2,
            default=str,
        )
    importance.rename("importance").to_csv(importance_path, header=True)

    # Verify no overwrite of V1/V2 names
    protected = {
        "saferoute_v2_precrash_unweighted_2026.pkl",
        "saferoute_module2_final.pkl",
        "saferoute_module2_corrected_2026.pkl",
        "saferoute_full_lightgbm_final.pkl",
    }
    assert model_path.name not in protected
    loaded = joblib.load(model_path)
    assert loaded["model_name"] == "SafeRoute-AI V3"
    assert loaded["model"].n_features_in_ == len(FEATURE_NAMES)

    elapsed = (time.time() - t0) / 60.0
    print("\n" + "=" * 70)
    print("V3 SAVE COMPLETE")
    print("=" * 70)
    print(f"  Model   : {model_path}")
    print(f"  Metrics : {metrics_path}")
    print(f"  Importance: {importance_path}")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  Candidate: {model_meta['candidate']}")
    print(f"  Replace V2?: {recommendation['replace_v2']}")
    print(f"  Rationale: {recommendation['rationale']}")
    print(f"  Runtime: {elapsed:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
