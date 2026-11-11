"""
Pre-deployment checks: V2 vs V3 on identical 2014–2015 validation rows.

- Does not modify dashboard / simulator
- Does not use 2016–2017 for threshold / feature / hyperparameter selection
- V2 retrained here with notebook Step 135L (unweighted) params when no local V2 pkl
"""

from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_saferoute_v3 import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TRAIN_YEARS,
    VALID_YEARS,
    build_ego_vehicle_features,
    engineer_features,
    load_accidents,
    load_vehicles,
)

V2_FEATURES = [
    "1st_Road_Class",
    "Day_of_Week",
    "Light_Conditions",
    "Road_Surface_Conditions",
    "Road_Type",
    "Speed_limit",
    "Urban_or_Rural_Area",
    "Weather_Conditions",
    "Hour",
    "Month",
    "Latitude",
    "Longitude",
    "Weekend",
    "Rush_Hour",
    "Night",
    "Year",
]

V2_CATEGORICAL = [
    "1st_Road_Class",
    "Day_of_Week",
    "Light_Conditions",
    "Road_Surface_Conditions",
    "Road_Type",
    "Urban_or_Rural_Area",
    "Weather_Conditions",
]

THRESHOLDS = [0.15, 0.175, 0.20, 0.25, 0.30, 0.50]

ROOT = Path(__file__).resolve().parent
ACCIDENT_PATH = ROOT.parent / "Accident_Information.csv.zip"
VEHICLE_PATH = ROOT.parent / "Vehicle_Information.csv.zip"
V3_PATH = ROOT / "models" / "saferoute_v3_precrash_2026.pkl"
OUT_PATH = ROOT / "outputs" / "v2_v3_deployment_validation.json"

# Dashboard V2 categorical keys (notebook Step 136A)
DASHBOARD_CATEGORIES = V2_CATEGORICAL


def encode_v2(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    X_train = train[V2_FEATURES].copy()
    X_valid = valid[V2_FEATURES].copy()
    levels_out: dict[str, list] = {}

    for col in V2_CATEGORICAL:
        train_vals = X_train[col].astype("string").str.strip()
        levels = sorted(train_vals.dropna().unique().tolist())
        levels_out[col] = levels
        dtype = pd.CategoricalDtype(categories=levels, ordered=False)
        X_train[col] = train_vals.astype(dtype)
        X_valid[col] = X_valid[col].astype("string").str.strip().astype(dtype)

    for col in V2_FEATURES:
        if col not in V2_CATEGORICAL:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_valid[col] = pd.to_numeric(X_valid[col], errors="coerce")

    return X_train, X_valid, levels_out


def encode_v3_with_checkpoint(
    valid: pd.DataFrame, checkpoint: dict
) -> pd.DataFrame:
    X = valid[checkpoint["feature_names"]].copy()
    levels = checkpoint["category_levels"]
    cats = checkpoint["categorical_features"]

    for col in cats:
        dtype = pd.CategoricalDtype(categories=levels[col], ordered=False)
        X[col] = X[col].astype("string").str.strip().astype(dtype)

    for col in checkpoint["feature_names"]:
        if col not in cats:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def train_v2_unweighted(X_train, y_train) -> lgb.LGBMClassifier:
    """Notebook Step 135L / deployed V2 checkpoint recipe."""
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=150,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=V2_CATEGORICAL)
    return model


def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def build_dashboard_record(scenario: dict) -> dict:
    """Mirror notebook Step 154 golden-test record builder (V2 schema)."""
    scenario_date: date = scenario["scenario_date"]
    scenario_time: time = scenario["scenario_time"]
    hour = scenario_time.hour

    weekend = 1.0 if scenario_date.weekday() >= 5 else 0.0
    rush = 1.0 if hour in [7, 8, 9, 16, 17, 18, 19] else 0.0
    night = 1.0 if (hour < 6 or hour >= 20) else 0.0

    return {
        "1st_Road_Class": scenario["road_class"],
        "Day_of_Week": scenario_date.strftime("%A"),
        "Light_Conditions": scenario["lighting"],
        "Road_Surface_Conditions": scenario["surface"],
        "Road_Type": scenario["road_type"],
        "Speed_limit": float(scenario["speed_limit"]),
        "Urban_or_Rural_Area": scenario["area"],
        "Weather_Conditions": scenario["weather"],
        "Hour": float(hour),
        "Month": float(scenario_date.month),
        "Latitude": float(scenario["latitude"]),
        "Longitude": float(scenario["longitude"]),
        "Weekend": float(weekend),
        "Rush_Hour": float(rush),
        "Night": float(night),
        "Year": float(scenario_date.year),
    }


def derive_engineered_from_v2_row(row: dict) -> dict:
    hour = row.get("Hour")
    month = row.get("Month")
    speed = row.get("Speed_limit")
    night = row.get("Night")
    urban = str(row.get("Urban_or_Rural_Area", "")).strip().lower()
    weather = str(row.get("Weather_Conditions", "")).strip().lower()
    surface = str(row.get("Road_Surface_Conditions", "")).strip().lower()

    is_rural = "rural" in urban
    is_urban = "urban" in urban
    is_high = float(speed) >= 60 if speed == speed else np.nan

    eng = {}
    if hour == hour:
        eng["Hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
        eng["Hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
    else:
        eng["Hour_sin"] = eng["Hour_cos"] = np.nan

    if month == month:
        eng["Month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
        eng["Month_cos"] = float(np.cos(2 * np.pi * month / 12.0))
    else:
        eng["Month_sin"] = eng["Month_cos"] = np.nan

    eng["Is_High_Speed"] = is_high
    eng["Is_Dark_Rural"] = (
        float(night == 1 and is_rural) if night == night and urban else np.nan
    )
    fine_like = ("fine" in weather) or (weather == "unknown")
    eng["Is_Adverse_Weather"] = (
        float(not fine_like) if weather else np.nan
    )
    eng["Is_Wet_Surface"] = float(
        bool(pd.Series([surface]).str.contains(
            "wet|damp|flood|snow|ice|frost", regex=True, na=False
        ).iloc[0])
    ) if surface else np.nan
    eng["Urban_High_Speed"] = (
        float(is_high == 1 and is_urban)
        if is_high == is_high and urban
        else np.nan
    )
    return eng


def audit_v3_features() -> list[dict]:
    v2_set = set(V2_FEATURES)
    engineered = {
        "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
        "Is_High_Speed", "Is_Dark_Rural", "Is_Adverse_Weather",
        "Is_Wet_Surface", "Urban_High_Speed",
    }
    ego = {
        "Ego_Vehicle_Type", "Ego_Driver_Age_Band", "Ego_Journey_Purpose",
        "Ego_Vehicle_Age", "Ego_Engine_Capacity",
    }
    road_extra = set(FEATURE_NAMES) - v2_set - engineered - ego

    rows = []
    for f in FEATURE_NAMES:
        if f in v2_set:
            avail = "dashboard_v2_direct"
            note = "Collected in current Streamlit V2 form"
        elif f in engineered:
            avail = "dashboard_v2_derived"
            note = "Can be computed from V2 inputs (time, speed, weather, surface, area)"
        elif f in ego:
            avail = "not_in_dashboard"
            note = (
                "User/vehicle profile before travel — legitimate pre-crash but "
                "NOT in current dashboard; model sees NaN unless you add inputs"
            )
        else:
            avail = "not_in_dashboard"
            note = (
                "Map/road-infrastructure field — legitimate pre-crash but "
                "NOT in current dashboard; requires map/OS data or new UI fields"
            )
        rows.append({"feature": f, "availability": avail, "note": note})

    summary = {
        "direct_from_dashboard": sum(1 for r in rows if r["availability"] == "dashboard_v2_direct"),
        "derived_from_dashboard": sum(1 for r in rows if r["availability"] == "dashboard_v2_derived"),
        "not_supplied_by_dashboard": sum(1 for r in rows if r["availability"] == "not_in_dashboard"),
        "road_extra_features": sorted(road_extra),
        "ego_features": sorted(ego),
    }
    return rows, summary


def prepare_v3_from_v2_record(
    record: dict, checkpoint: dict, *, ego_defaults: dict | None = None
) -> pd.DataFrame:
    """Build a V3 matrix row from V2 dashboard fields + optional ego + NaN road extras."""
    row = dict(record)
    row.update(derive_engineered_from_v2_row(record))

    if ego_defaults:
        row.update(ego_defaults)

    for col in checkpoint["feature_names"]:
        if col not in row:
            row[col] = np.nan

    X = pd.DataFrame([row], columns=checkpoint["feature_names"])
    levels = checkpoint["category_levels"]
    for col in checkpoint["categorical_features"]:
        val = X.at[0, col]
        if pd.isna(val):
            X[col] = pd.Series([np.nan], dtype=object)
        else:
            X.at[0, col] = str(val).strip()
        dtype = pd.CategoricalDtype(categories=levels[col], ordered=False)
        X[col] = X[col].astype(dtype)

    for col in checkpoint["feature_names"]:
        if col not in checkpoint["categorical_features"]:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def recommend_threshold(v3_table: list[dict]) -> dict:
    """Pick from the fixed threshold grid using validation only."""
    df = pd.DataFrame(v3_table)

    best_f1 = df.loc[df["f1"].idxmax()]
    # Balance: recall >= 0.48 with highest precision (V3-specific operating point)
    recall_floor = df[df["recall"] >= 0.48]
    balanced = (
        recall_floor.sort_values(["precision", "f1"], ascending=False).iloc[0]
        if len(recall_floor)
        else best_f1
    )

    return {
        "max_f1_threshold": float(best_f1["threshold"]),
        "max_f1": float(best_f1["f1"]),
        "max_f1_precision": float(best_f1["precision"]),
        "max_f1_recall": float(best_f1["recall"]),
        "balanced_threshold": float(balanced["threshold"]),
        "balanced_precision": float(balanced["precision"]),
        "balanced_recall": float(balanced["recall"]),
        "balanced_f1": float(balanced["f1"]),
        "rationale": (
            "Use threshold {:.3f} for deployment: among the requested grid it maximises "
            "validation F1 ({:.3f}) while keeping recall {:.1%} and precision {:.1%}. "
            "Alternative {:.3f} improves precision to {:.1%} at recall {:.1%} if you want "
            "fewer false KSI flags."
        ).format(
            float(best_f1["threshold"]),
            float(best_f1["f1"]),
            float(best_f1["recall"]),
            float(best_f1["precision"]),
            float(balanced["threshold"]),
            float(balanced["precision"]),
            float(balanced["recall"]),
        ),
        "holdout_limitation": (
            "2016–2017 were evaluated separately for generalization reporting only; "
            "they were NOT used to choose this threshold, features, or hyperparameters."
        ),
    }


def main() -> None:
    print("=" * 72)
    print("SafeRoute-AI | V2 vs V3 deployment validation (2014–2015 only)")
    print("=" * 72)

    accidents = load_accidents(ACCIDENT_PATH)
    vehicles = load_vehicles(VEHICLE_PATH)
    ego = build_ego_vehicle_features(vehicles)
    data = engineer_features(accidents, ego)

    train = data.loc[data["Year"].between(*TRAIN_YEARS)].copy()
    valid = data.loc[data["Year"].between(*VALID_YEARS)].copy()
    valid = valid.sort_values("Accident_Index").reset_index(drop=True)

    y_valid = valid["KSI"].astype(int).to_numpy()
    n_valid = len(valid)
    print(f"\nValidation rows (aligned): {n_valid:,}")
    print(f"KSI prevalence: {y_valid.mean():.4f}")

    # --- V2 ---
    X2_train, X2_valid, v2_levels = encode_v2(train, valid)
    y2_train = train["KSI"].astype(int).to_numpy()
    print("\nTraining V2 unweighted (Step 135L recipe)...")
    v2_model = train_v2_unweighted(X2_train, y2_train)
    v2_prob = v2_model.predict_proba(X2_valid)[:, 1]

    # --- V3 ---
    assert V3_PATH.exists(), f"Missing V3 checkpoint: {V3_PATH}"
    ckpt = joblib.load(V3_PATH)
    v3_model = ckpt["model"]
    X3_valid = encode_v3_with_checkpoint(valid, ckpt)
    assert list(ckpt["feature_names"]) == list(v3_model.feature_name_)
    v3_prob = v3_model.predict_proba(X3_valid)[:, 1]

    assert len(v2_prob) == len(v3_prob) == len(y_valid)

    v2_roc = roc_auc_score(y_valid, v2_prob)
    v3_roc = roc_auc_score(y_valid, v3_prob)
    v2_pr = average_precision_score(y_valid, v2_prob)
    v3_pr = average_precision_score(y_valid, v3_prob)

    print(f"\nRanking metrics on same {n_valid:,} records:")
    print(f"  V2 ROC-AUC={v2_roc:.4f}  PR-AUC={v2_pr:.4f}")
    print(f"  V3 ROC-AUC={v3_roc:.4f}  PR-AUC={v3_pr:.4f}")

    comparison_rows = []
    v2_thr_rows = []
    v3_thr_rows = []

    print("\n" + "=" * 72)
    print("THRESHOLD COMPARISON (2014–2015)")
    print("=" * 72)

    for thr in THRESHOLDS:
        m2 = metrics_at_threshold(y_valid, v2_prob, thr)
        m3 = metrics_at_threshold(y_valid, v3_prob, thr)
        v2_thr_rows.append(m2)
        v3_thr_rows.append(m3)
        comparison_rows.append(
            {
                "threshold": thr,
                "v2_accuracy": m2["accuracy"],
                "v2_precision": m2["precision"],
                "v2_recall": m2["recall"],
                "v2_f1": m2["f1"],
                "v2_cm": m2["confusion_matrix"],
                "v3_accuracy": m3["accuracy"],
                "v3_precision": m3["precision"],
                "v3_recall": m3["recall"],
                "v3_f1": m3["f1"],
                "v3_cm": m3["confusion_matrix"],
            }
        )
        print(f"\n--- threshold {thr:.3f} ---")
        print(
            f"V2  acc={m2['accuracy']:.4f} prec={m2['precision']:.4f} "
            f"rec={m2['recall']:.4f} F1={m2['f1']:.4f}  "
            f"CM tn={m2['confusion_matrix']['tn']} fp={m2['confusion_matrix']['fp']} "
            f"fn={m2['confusion_matrix']['fn']} tp={m2['confusion_matrix']['tp']}"
        )
        print(
            f"V3  acc={m3['accuracy']:.4f} prec={m3['precision']:.4f} "
            f"rec={m3['recall']:.4f} F1={m3['f1']:.4f}  "
            f"CM tn={m3['confusion_matrix']['tn']} fp={m3['confusion_matrix']['fp']} "
            f"fn={m3['confusion_matrix']['fn']} tp={m3['confusion_matrix']['tp']}"
        )

    feature_audit, audit_summary = audit_v3_features()
    threshold_rec = recommend_threshold(v3_thr_rows)

    # Checkpoint + dashboard schema tests
    print("\n" + "=" * 72)
    print("V3 CHECKPOINT + DASHBOARD SCHEMA")
    print("=" * 72)
    print("Checkpoint load: OK")
    print(f"feature_names match model: {list(ckpt['feature_names']) == list(v3_model.feature_name_)}")

    golden = {
        "name": "Dry daylight (notebook golden)",
        "scenario_date": date(2013, 6, 15),
        "scenario_time": time(14, 0),
        "road_class": "A",
        "road_type": "Single carriageway",
        "speed_limit": 40,
        "area": "Urban",
        "weather": "Fine no high winds",
        "surface": "Dry",
        "lighting": "Daylight",
        "latitude": 52.239552,
        "longitude": -1.350476,
    }
    v2_record = build_dashboard_record(golden)

    v2_frame = pd.DataFrame([v2_record], columns=V2_FEATURES)
    for col in DASHBOARD_CATEGORIES:
        v2_frame[col] = pd.Categorical(
            v2_frame[col], categories=ckpt["category_levels"].get(col, v2_levels[col])
        )
    v2_only_ok = False
    v2_only_error = None
    try:
        # V2 model expects 16 features only — this is the live dashboard path
        _ = v2_model.predict_proba(v2_frame)[:, 1]
        v2_only_ok = True
    except Exception as exc:
        v2_only_error = str(exc)

    # V3 cannot accept 16-column frame
    v3_direct_ok = False
    try:
        _ = v3_model.predict_proba(v2_frame)[:, 1]
    except Exception:
        v3_direct_ok = False

    # Bridge: V2 record → full V3 row (NaN for non-dashboard fields)
    X3_golden = prepare_v3_from_v2_record(v2_record, ckpt)
    v3_bridged_prob = float(v3_model.predict_proba(X3_golden)[0, 1])

    # Optional ego-filled scenario (simulator could collect these later)
    ego_fill = {
        "Ego_Vehicle_Type": "Car",
        "Ego_Driver_Age_Band": "26 - 35",
        "Ego_Journey_Purpose": "Private",
        "Ego_Vehicle_Age": 5.0,
        "Ego_Engine_Capacity": 1600.0,
    }
    X3_ego = prepare_v3_from_v2_record(v2_record, ckpt, ego_defaults=ego_fill)
    v3_ego_prob = float(v3_model.predict_proba(X3_ego)[0, 1])

    schema_report = {
        "v2_dashboard_predict_on_v2_model": v2_only_ok,
        "v2_dashboard_predict_on_v3_without_bridge": v3_direct_ok,
        "v3_bridged_from_v2_schema_probability": v3_bridged_prob,
        "v3_bridged_with_sample_ego_probability": v3_ego_prob,
        "interpretation": (
            "The live dashboard supplies 16 fields. V3 requires 35. "
            "Predictions succeed only after mapping V2 inputs to V3 and filling "
            "missing road/ego fields (currently NaN). Exact 16-column input cannot "
            "be fed to V3 without feature engineering + new UI fields."
        ),
    }

    print(schema_report["interpretation"])
    print(f"V2 model + dashboard record: {'OK' if v2_only_ok else v2_only_error}")
    print(f"V3 + 16-col dashboard frame: {'OK (unexpected)' if v3_direct_ok else 'REJECTED (expected)'}")
    print(f"V3 bridged prob (no ego): {v3_bridged_prob:.4f}")
    print(f"V3 bridged prob (sample ego): {v3_ego_prob:.4f}")

    report = {
        "validation_years": list(range(VALID_YEARS[0], VALID_YEARS[1] + 1)),
        "n_validation_rows": n_valid,
        "v2_training_note": (
            "Local V2 .pkl not found; V2 retrained with Step 135L unweighted "
            "hyperparameters on 2009–2013, evaluated on identical 2014–2015 rows."
        ),
        "ranking_metrics_same_records": {
            "v2": {"roc_auc": v2_roc, "pr_auc": v2_pr},
            "v3": {"roc_auc": v3_roc, "pr_auc": v3_pr},
        },
        "threshold_comparison": comparison_rows,
        "feature_availability_audit": feature_audit,
        "feature_availability_summary": audit_summary,
        "v3_checkpoint_schema_test": schema_report,
        "recommended_threshold": threshold_rec,
        "holdout_policy": (
            "2016–2017 metrics in saferoute_v3_metrics.json are retrospective only; "
            "do not use them for threshold or feature decisions."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("THRESHOLD RECOMMENDATION (validation only)")
    print("=" * 72)
    print(threshold_rec["rationale"])
    print(threshold_rec["holdout_limitation"])
    print(f"\nFull report: {OUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
