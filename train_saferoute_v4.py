"""
SafeRoute-AI V4 — pre-crash KSI severity model.

Improves on V3 with legitimate features that are known before travel:
  * interactions of speed, weather, lighting, and junction
  * cyclical day-of-week encoding and an ordinal driver age band
  * historical KSI rates from earlier years only (a row never sees its own
    year, and 2014–2015 only see 2009–2013)

Does not use accident outcomes, casualty counts, vehicle counts, other
parties, or post-crash observations. Does not read 2016–2017. Does not
overwrite the V3 checkpoint, app.py, or simulator.html.

Splits:
  Fit        2009–2012   (model fitting)
  Early stop 2013        (iteration choice only)
  Validation 2014–2015   (candidate comparison, threshold, reported metrics)
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent
ACCIDENT_PATH = ROOT.parent / "Accident_Information.csv.zip"
VEHICLE_PATH = ROOT.parent / "Vehicle_Information.csv.zip"
MODEL_PATH = ROOT / "models" / "saferoute_v4_precrash_2026.pkl"
METRICS_PATH = ROOT / "outputs" / "saferoute_v4_metrics.json"
COMPARISON_PATH = ROOT / "outputs" / "saferoute_v4_comparison.md"
IMPORTANCE_PATH = ROOT / "outputs" / "saferoute_v4_feature_importance.csv"
V3_METRICS_PATH = ROOT / "outputs" / "saferoute_v3_metrics.json"

ACCIDENT_COLS = [
    "Accident_Index",
    "Year",
    "Date",
    "Time",
    "Accident_Severity",
    "1st_Road_Class",
    "2nd_Road_Class",
    "Road_Type",
    "Speed_limit",
    "Junction_Detail",
    "Junction_Control",
    "Pedestrian_Crossing-Physical_Facilities",
    "Pedestrian_Crossing-Human_Control",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Urban_or_Rural_Area",
    "Day_of_Week",
    "Latitude",
    "Longitude",
]
VEHICLE_COLS = [
    "Accident_Index",
    "Vehicle_Reference",
    "Vehicle_Type",
    "Age_Band_of_Driver",
    "Journey_Purpose_of_Driver",
    "Age_of_Vehicle",
    "Engine_Capacity_.CC.",
]

CATEGORICAL = [
    "Road_Class_1",
    "Road_Class_2",
    "Day_of_Week",
    "Junction_Control",
    "Junction_Detail",
    "Light_Conditions",
    "Road_Surface",
    "Road_Type",
    "Area",
    "Weather",
    "Crossing_Physical",
    "Crossing_Human",
    "Ego_Vehicle_Type",
    "Ego_Driver_Age_Band",
    "Ego_Journey_Purpose",
]

# V3 numerics, renamed only where the source column contained a hyphen,
# plus V4 engineering. Historical columns are attached later.
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
    "Hour_sin",
    "Hour_cos",
    "Month_sin",
    "Month_cos",
    "Dow_sin",
    "Dow_cos",
    "Is_High_Speed",
    "Is_Dark_Rural",
    "Is_Adverse_Weather",
    "Is_Wet_Surface",
    "Is_Fog",
    "Is_Junction",
    "Urban_High_Speed",
    "Age_Band_Ord",
    "Log_Engine",
    "Speed_Gap",
    "Speed_x_Wet",
    "Speed_x_Night",
    "Speed_x_Adverse",
    "Speed_x_Junction",
    "Dark_Wet",
    "High_Speed_Wet",
]

HIST_RATES = [
    ("Hist_Geo_Fine", "_geo_fine", 120.0, True),
    ("Hist_Geo_Coarse", "_geo_coarse", 40.0, True),
    ("Hist_Road_Speed", "_road_speed", 50.0, False),
    ("Hist_Junction", "_junction", 40.0, False),
    ("Hist_Weather_Surface", "_weather_surface", 40.0, False),
    ("Hist_Light_Area", "_light_area", 40.0, False),
    ("Hist_Vehicle", "_vehicle", 40.0, False),
    ("Hist_Age", "_age", 30.0, False),
    ("Hist_Hour_Area", "_hour_area", 40.0, False),
]

AGE_ORD = {
    "0 - 5": 1,
    "6 - 10": 2,
    "11 - 15": 3,
    "16 - 20": 4,
    "17 - 20": 4,
    "21 - 25": 5,
    "26 - 35": 6,
    "36 - 45": 7,
    "46 - 55": 8,
    "56 - 65": 9,
    "66 - 75": 10,
    "over 75": 11,
}
DOW_ORD = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
TYPICAL_SPEED = {
    "motorway": 70.0,
    "a(m)": 70.0,
    "a": 60.0,
    "b": 40.0,
    "c": 30.0,
    "unclassified": 30.0,
}

# Published V2 validation figures. The V2 checkpoint is not in this project,
# so precision, recall, and the confusion matrix are not invented here.
V2_PUBLISHED = {
    "accuracy": 0.6003,
    "roc_auc": 0.6361,
    "pr_auc": 0.2426,
    "f1": 0.3135,
    "threshold": 0.15,
    "note": "Published validation figures. Precision, recall, and the confusion matrix were not re-measured because the V2 checkpoint is not stored in this project.",
}


def log(message: str) -> None:
    print(message, flush=True)


def text_key(series: pd.Series) -> pd.Series:
    key = series.astype("string").str.strip()
    key = key.fillna("missing")
    key = key.mask(key.eq("") | key.str.lower().eq("nan") | key.str.lower().eq("<na>"), "missing")
    return key.astype("object")


def grid_key(lat: pd.Series, lon: pd.Series, step: float) -> pd.Series:
    ok = lat.notna() & lon.notna()
    lat_bin = np.floor(lat.to_numpy(dtype=float) / step) * step
    lon_bin = np.floor(lon.to_numpy(dtype=float) / step) * step
    key = np.full(len(lat), "missing", dtype=object)
    if ok.any():
        lat_s = pd.Series(lat_bin).round(4).astype(str)
        lon_s = pd.Series(lon_bin).round(4).astype(str)
        key[ok.to_numpy()] = (lat_s[ok.to_numpy()] + "|" + lon_s[ok.to_numpy()]).to_numpy()
    return pd.Series(key, index=lat.index)


def load_frame() -> pd.DataFrame:
    log(f"Loading accidents from {ACCIDENT_PATH}")
    accidents = pd.read_csv(
        ACCIDENT_PATH,
        compression="zip",
        usecols=ACCIDENT_COLS,
        low_memory=False,
    )
    accidents["Year"] = pd.to_numeric(accidents["Year"], errors="coerce")
    # 2016–2017 is excluded before any feature is built.
    accidents = accidents.loc[accidents["Year"].between(2009, 2015)].copy()
    accidents["Accident_Index"] = accidents["Accident_Index"].astype("string").str.strip()
    log(f"  Accidents 2009–2015: {len(accidents):,}")

    log(f"Loading vehicles from {VEHICLE_PATH}")
    vehicles = pd.read_csv(
        VEHICLE_PATH,
        compression="zip",
        usecols=VEHICLE_COLS,
        encoding="cp1252",
        low_memory=False,
    )
    vehicles["Accident_Index"] = vehicles["Accident_Index"].astype("string").str.strip()
    vehicles = vehicles.loc[vehicles["Accident_Index"].isin(set(accidents["Accident_Index"]))].copy()
    vehicles["Vehicle_Reference"] = pd.to_numeric(vehicles["Vehicle_Reference"], errors="coerce")
    vehicles["Age_of_Vehicle"] = pd.to_numeric(vehicles["Age_of_Vehicle"], errors="coerce")
    vehicles["Engine_Capacity_.CC."] = pd.to_numeric(vehicles["Engine_Capacity_.CC."], errors="coerce")
    vehicles.loc[vehicles["Age_of_Vehicle"] < 0, "Age_of_Vehicle"] = np.nan
    vehicles.loc[vehicles["Engine_Capacity_.CC."] < 0, "Engine_Capacity_.CC."] = np.nan
    ego = (
        vehicles.sort_values(["Accident_Index", "Vehicle_Reference"], kind="mergesort", na_position="last")
        .drop_duplicates("Accident_Index", keep="first")
        .loc[:, [
            "Accident_Index",
            "Vehicle_Type",
            "Age_Band_of_Driver",
            "Journey_Purpose_of_Driver",
            "Age_of_Vehicle",
            "Engine_Capacity_.CC.",
        ]]
        .rename(columns={
            "Vehicle_Type": "Ego_Vehicle_Type",
            "Age_Band_of_Driver": "Ego_Driver_Age_Band",
            "Journey_Purpose_of_Driver": "Ego_Journey_Purpose",
            "Age_of_Vehicle": "Ego_Vehicle_Age",
            "Engine_Capacity_.CC.": "Ego_Engine_Capacity",
        })
    )
    del vehicles

    data = accidents.merge(ego, on="Accident_Index", how="left")
    del accidents, ego

    severity = data["Accident_Severity"].astype("string").str.strip().str.lower()
    data = data.loc[severity.isin(["fatal", "serious", "slight"])].copy()
    data["KSI"] = severity.loc[data.index].isin(["fatal", "serious"]).astype(np.int8)

    data["Hour"] = pd.to_numeric(
        data["Time"].astype("string").str.extract(r"^(\d{1,2}):\d{2}", expand=False),
        errors="coerce",
    )
    data.loc[~data["Hour"].between(0, 23), "Hour"] = np.nan
    parsed = pd.to_datetime(data["Date"], format="mixed", dayfirst=True, errors="coerce")
    data["Month"] = parsed.dt.month.astype("float64")

    day = data["Day_of_Week"].astype("string").str.strip().str.lower()
    data["Weekend"] = day.isin(["saturday", "sunday"]).astype("float64")
    dow = day.map(DOW_ORD).astype("float64")
    data["Dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    data["Dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    data["Rush_Hour"] = data["Hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype("float64")
    data["Night"] = ((data["Hour"] < 6) | (data["Hour"] >= 20)).astype("float64")
    data.loc[data["Hour"].isna(), ["Rush_Hour", "Night"]] = np.nan
    data["Hour_sin"] = np.sin(2 * np.pi * data["Hour"] / 24.0)
    data["Hour_cos"] = np.cos(2 * np.pi * data["Hour"] / 24.0)
    data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12.0)
    data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12.0)

    speed = pd.to_numeric(data["Speed_limit"], errors="coerce")
    speed = speed.where(speed.between(10, 70))
    data["Speed_limit"] = speed
    data["Is_High_Speed"] = (speed >= 60).astype("float64")
    data.loc[speed.isna(), "Is_High_Speed"] = np.nan

    area = data["Urban_or_Rural_Area"].astype("string").str.strip().str.lower()
    rural = area.str.contains("rural", na=False)
    urban = area.str.contains("urban", na=False)
    data["Is_Dark_Rural"] = ((data["Night"] == 1) & rural).astype("float64")
    data.loc[data["Night"].isna() | area.isna(), "Is_Dark_Rural"] = np.nan
    data["Urban_High_Speed"] = ((data["Is_High_Speed"] == 1) & urban).astype("float64")
    data.loc[data["Is_High_Speed"].isna() | area.isna(), "Urban_High_Speed"] = np.nan

    weather = data["Weather_Conditions"].astype("string").str.strip().str.lower()
    fine = weather.str.contains("fine", na=False) | weather.eq("unknown")
    data["Is_Adverse_Weather"] = (~fine & weather.notna()).astype("float64")
    data.loc[weather.isna(), "Is_Adverse_Weather"] = np.nan
    data["Is_Fog"] = weather.str.contains("fog", na=False).astype("float64")
    data.loc[weather.isna(), "Is_Fog"] = np.nan

    surface = data["Road_Surface_Conditions"].astype("string").str.strip().str.lower()
    wet = surface.str.contains("wet|damp|flood|snow|ice|frost", regex=True, na=False)
    data["Is_Wet_Surface"] = wet.astype("float64")
    data.loc[surface.isna(), "Is_Wet_Surface"] = np.nan

    junction = data["Junction_Detail"].astype("string").str.strip().str.lower()
    data["Is_Junction"] = (~junction.str.contains("not at junction", na=False) & junction.notna()).astype("float64")
    data.loc[junction.isna(), "Is_Junction"] = np.nan

    age_key = data["Ego_Driver_Age_Band"].astype("string").str.strip().str.lower()
    data["Age_Band_Ord"] = age_key.map(AGE_ORD).astype("float64")
    data["Log_Engine"] = np.log1p(pd.to_numeric(data["Ego_Engine_Capacity"], errors="coerce"))
    road_key = data["1st_Road_Class"].astype("string").str.strip().str.lower()
    typical = road_key.map(TYPICAL_SPEED).astype("float64")
    data["Speed_Gap"] = speed - typical

    data["Speed_x_Wet"] = speed * data["Is_Wet_Surface"]
    data["Speed_x_Night"] = speed * data["Night"]
    data["Speed_x_Adverse"] = speed * data["Is_Adverse_Weather"]
    data["Speed_x_Junction"] = speed * data["Is_Junction"]
    data["Dark_Wet"] = data["Night"] * data["Is_Wet_Surface"]
    data["High_Speed_Wet"] = data["Is_High_Speed"] * data["Is_Wet_Surface"]

    data = data.rename(columns={
        "1st_Road_Class": "Road_Class_1",
        "2nd_Road_Class": "Road_Class_2",
        "Road_Surface_Conditions": "Road_Surface",
        "Urban_or_Rural_Area": "Area",
        "Weather_Conditions": "Weather",
        "Pedestrian_Crossing-Physical_Facilities": "Crossing_Physical",
        "Pedestrian_Crossing-Human_Control": "Crossing_Human",
    })

    hour_bucket = pd.cut(
        data["Hour"],
        bins=[-0.1, 5, 9, 15, 19, 23.1],
        labels=["night", "am_peak", "midday", "pm_peak", "evening"],
    ).astype("string")
    data["_geo_fine"] = grid_key(data["Latitude"], data["Longitude"], 0.05)
    data["_geo_coarse"] = grid_key(data["Latitude"], data["Longitude"], 0.20)
    data["_road_speed"] = text_key(data["Road_Class_1"]) + "|" + speed.round(0).astype("string").fillna("missing")
    data["_junction"] = text_key(data["Junction_Detail"])
    data["_weather_surface"] = text_key(data["Weather"]) + "|" + text_key(data["Road_Surface"])
    data["_light_area"] = text_key(data["Light_Conditions"]) + "|" + text_key(data["Area"])
    data["_vehicle"] = text_key(data["Ego_Vehicle_Type"])
    data["_age"] = text_key(data["Ego_Driver_Age_Band"])
    data["_hour_area"] = hour_bucket.fillna("missing") + "|" + text_key(data["Area"])

    if data["Year"].max() > 2015:
        raise RuntimeError("2016–2017 rows entered the training frame.")
    data = data.reset_index(drop=True)
    log(f"  Model rows 2009–2015: {len(data):,}  KSI rate={data['KSI'].mean():.4f}")
    log(f"  Ego match rate: {data['Ego_Vehicle_Type'].notna().mean():.2%}")
    return data


def time_aware_history(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """KSI rate from strictly earlier years. A row's own year is not included."""
    train_prior = float(data.loc[data["Year"].between(2009, 2013), "KSI"].mean())
    out = pd.DataFrame(index=data.index)
    tables = {}
    year_values = data["Year"].to_numpy()
    for name, key_col, smoothing, store_count in HIST_RATES:
        grouped = (
            data.groupby([key_col, "Year"], sort=False)["KSI"]
            .agg(sum="sum", count="count")
            .reset_index()
            .sort_values([key_col, "Year"], kind="mergesort")
        )
        grouped["past_sum"] = grouped.groupby(key_col, sort=False)["sum"].cumsum() - grouped["sum"]
        grouped["past_n"] = grouped.groupby(key_col, sort=False)["count"].cumsum() - grouped["count"]
        smoothing_m = float(smoothing)
        grouped["rate"] = (grouped["past_sum"] + smoothing_m * train_prior) / (grouped["past_n"] + smoothing_m)
        grouped["logn"] = np.log1p(grouped["past_n"].clip(lower=0))
        multi = pd.MultiIndex.from_arrays([data[key_col].to_numpy(), year_values])
        rate_lookup = grouped.set_index([key_col, "Year"])["rate"]
        out[name] = pd.Series(rate_lookup.reindex(multi).to_numpy(dtype=float), index=data.index).fillna(train_prior)
        if store_count:
            log_lookup = grouped.set_index([key_col, "Year"])["logn"]
            out[name + "_LogN"] = pd.Series(log_lookup.reindex(multi).to_numpy(dtype=float), index=data.index).fillna(0.0)

        through = grouped.loc[grouped["Year"] <= 2013]
        last = through.groupby(key_col, sort=False).tail(1)
        tables[name] = {
            "m": smoothing_m,
            "sum": pd.Series((last["past_sum"] + last["sum"]).to_numpy(dtype=float), index=last[key_col].to_numpy()),
            "count": pd.Series((last["past_n"] + last["count"]).to_numpy(dtype=float), index=last[key_col].to_numpy()),
        }
    return out, {"prior": train_prior, "tables": tables}


def feature_names(history_columns: list[str]) -> list[str]:
    return CATEGORICAL + BASE_NUMERIC + history_columns


def cast_matrix(frame: pd.DataFrame, names: list[str], levels: dict[str, list[str]]) -> pd.DataFrame:
    matrix = pd.DataFrame(index=frame.index)
    for col in CATEGORICAL:
        dtype = pd.CategoricalDtype(categories=levels[col], ordered=False)
        matrix[col] = frame[col].astype("string").str.strip().astype(dtype)
    for col in names:
        if col in CATEGORICAL:
            continue
        matrix[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return matrix[names]


def category_levels(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, list[str]]:
    levels = {}
    for col in CATEGORICAL:
        values = frame.loc[mask, col].astype("string").str.strip()
        levels[col] = sorted(v for v in values.dropna().unique().tolist() if v != "")
    return levels


def metrics_at(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_table(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.81, 0.025), 3):
        row = metrics_at(y_true, probability, float(threshold))
        rows.append({
            "threshold": row["threshold"],
            "accuracy": row["accuracy"],
            "balanced_accuracy": row["balanced_accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        })
    return pd.DataFrame(rows)


def operating_points(table: pd.DataFrame) -> dict:
    best_f1 = table.loc[table["f1"].idxmax()]
    best_acc = table.loc[table["accuracy"].idxmax()]

    def constrained(min_recall: float) -> dict | None:
        eligible = table.loc[table["recall"] >= min_recall]
        if eligible.empty:
            return None
        row = eligible.loc[eligible["accuracy"].idxmax()]
        return {key: float(row[key]) for key in row.index}

    return {
        "best_f1": {key: float(best_f1[key]) for key in best_f1.index},
        "best_accuracy": {key: float(best_acc[key]) for key in best_acc.index},
        "best_accuracy_recall_at_least_0.30": constrained(0.30),
        "best_accuracy_recall_at_least_0.40": constrained(0.40),
    }


def majority_baseline(y_true: np.ndarray) -> dict:
    prevalence = float(np.mean(y_true))
    # Predicting every recorded accident as Slight.
    return {
        "rule": "Predict Slight for every recorded accident",
        "accuracy": float(1.0 - prevalence),
        "balanced_accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "ksi_prevalence": prevalence,
        "note": "Accuracy above this line requires fewer total errors than calling every case Slight. An accuracy near this line with very low KSI recall does not meet the detection objective.",
    }


def calibration_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (probability >= low) & (probability <= high)
        else:
            mask = (probability >= low) & (probability < high)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append({
            "bin_low": float(low),
            "bin_high": float(high),
            "count": count,
            "mean_predicted": float(probability[mask].mean()),
            "observed_ksi_rate": float(y_true[mask].mean()),
        })
    return rows


class AverageEnsemble:
    """Equal-weight probability average. All members share one feature frame."""

    def __init__(self, members: list[tuple[str, object]]):
        self.members = members

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        columns = []
        for _name, model in self.members:
            columns.append(np.asarray(model.predict_proba(X)[:, 1], dtype=float))
        probability = np.mean(columns, axis=0)
        return np.column_stack([1.0 - probability, probability])


def score_model(name: str, model, matrix: pd.DataFrame, y: np.ndarray, rows: np.ndarray) -> dict:
    probability = np.asarray(model.predict_proba(matrix.loc[rows])[:, 1], dtype=float)
    ranking = {
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }
    log(f"    {name}: PR-AUC={ranking['pr_auc']:.4f}  ROC-AUC={ranking['roc_auc']:.4f}  Brier={ranking['brier']:.4f}")
    return {"name": name, "model": model, "probability": probability, **ranking}


def fit_lightgbm(params: dict, X_fit, y_fit, X_stop, y_stop) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        subsample_freq=1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_X=X_stop,
        eval_y=y_stop,
        eval_metric="average_precision",
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def refit_lightgbm(params: dict, n_estimators: int, X_train, y_train) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=max(50, int(n_estimators)),
        subsample_freq=1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL)
    return model


def train_candidates(search: pd.DataFrame, y: np.ndarray, fit_rows, stop_rows, valid_rows) -> list[dict]:
    X_fit = search.loc[fit_rows]
    y_fit = y[fit_rows]
    X_stop = search.loc[stop_rows]
    y_stop = y[stop_rows]
    results = []

    fixed = {
        "lgbm_v3_params": {
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 150,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.5,
            "scale_pos_weight": 1.0,
        },
        "lgbm_mild_weight": {
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 100,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "scale_pos_weight": 2.0,
        },
    }
    for name, params in fixed.items():
        log(f"\nTraining {name}")
        model = fit_lightgbm(params, X_fit, y_fit, X_stop, y_stop)
        scored = score_model(name, model, search, y[valid_rows], valid_rows)
        scored["params"] = params
        scored["best_iteration"] = int(model.best_iteration_ or params.get("n_estimators", 1200))
        scored["family"] = "lightgbm"
        if name == "lgbm_v3_params" and scored["roc_auc"] < 0.60:
            raise RuntimeError(
                f"V3-parameter model ROC-AUC is {scored['roc_auc']:.4f}. "
                "Stopping before the rest of the search because the feature matrix is not learning."
            )
        results.append(scored)

    log("\nOptuna search: 8 LightGBM trials, objective = 2014–2015 PR-AUC")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127]),
            "min_child_samples": trial.suggest_int("min_child_samples", 80, 320, step=40),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 6.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        }
        model = fit_lightgbm(params, X_fit, y_fit, X_stop, y_stop)
        probability = model.predict_proba(search.loc[valid_rows])[:, 1]
        pr_auc = float(average_precision_score(y[valid_rows], probability))
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or 1200))
        log(f"    trial {trial.number}: PR-AUC={pr_auc:.4f}  trees={trial.user_attrs['best_iteration']}")
        return pr_auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=8, catch=(Exception,))
    best_trial = study.best_trial
    best_params = best_trial.user_attrs.get("params_used")
    if best_params is None:
        best_params = {
            "learning_rate": best_trial.params["learning_rate"],
            "num_leaves": best_trial.params["num_leaves"],
            "min_child_samples": best_trial.params["min_child_samples"],
            "subsample": best_trial.params["subsample"],
            "colsample_bytree": best_trial.params["colsample_bytree"],
            "reg_alpha": best_trial.params["reg_alpha"],
            "reg_lambda": best_trial.params["reg_lambda"],
            "scale_pos_weight": best_trial.params["scale_pos_weight"],
        }
    refit_search = fit_lightgbm(best_params, X_fit, y_fit, X_stop, y_stop)
    best_lgbm = score_model("lgbm_optuna", refit_search, search, y[valid_rows], valid_rows)
    best_lgbm["params"] = best_params
    best_lgbm["best_iteration"] = int(refit_search.best_iteration_ or best_trial.user_attrs.get("best_iteration", 1200))
    best_lgbm["family"] = "lightgbm"
    results.append(best_lgbm)

    log("\nTraining XGBoost")
    try:
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            enable_categorical=True,
            eval_metric="aucpr",
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=80,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.2,
            early_stopping_rounds=40,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)
        scored = score_model("xgboost", xgb_model, search, y[valid_rows], valid_rows)
        scored["params"] = xgb_model.get_params()
        scored["best_iteration"] = int(getattr(xgb_model, "best_iteration", 999) + 1)
        scored["family"] = "xgboost"
        results.append(scored)
    except Exception:
        log("  XGBoost failed:\n" + traceback.format_exc())

    log("\nTraining CatBoost")
    try:
        def as_catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
            converted = frame.copy()
            for col in CATEGORICAL:
                converted[col] = converted[col].astype("string").fillna("missing")
            return converted

        X_fit_cb = as_catboost_frame(X_fit)
        X_stop_cb = as_catboost_frame(X_stop)
        X_valid_cb = as_catboost_frame(search.loc[valid_rows])
        cat_columns = [X_fit_cb.columns.get_loc(col) for col in CATEGORICAL]
        cat_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC",
            iterations=800,
            learning_rate=0.08,
            depth=6,
            l2_leaf_reg=6.0,
            random_seed=42,
            early_stopping_rounds=40,
            allow_writing_files=False,
            verbose=100,
            thread_count=-1,
        )
        cat_model.fit(
            Pool(X_fit_cb, y_fit, cat_features=cat_columns),
            eval_set=Pool(X_stop_cb, y_stop, cat_features=cat_columns),
            use_best_model=True,
        )
        cat_probability = np.asarray(cat_model.predict_proba(X_valid_cb)[:, 1], dtype=float)
        scored = {
            "name": "catboost",
            "model": cat_model,
            "probability": cat_probability,
            "pr_auc": float(average_precision_score(y[valid_rows], cat_probability)),
            "roc_auc": float(roc_auc_score(y[valid_rows], cat_probability)),
            "brier": float(brier_score_loss(y[valid_rows], cat_probability)),
        }
        log(
            f"    catboost: PR-AUC={scored['pr_auc']:.4f}  "
            f"ROC-AUC={scored['roc_auc']:.4f}  Brier={scored['brier']:.4f}"
        )
        scored["params"] = {
            "iterations": 800,
            "learning_rate": 0.08,
            "depth": 6,
            "l2_leaf_reg": 6.0,
        }
        scored["best_iteration"] = int(cat_model.get_best_iteration() or 800)
        scored["family"] = "catboost"
        results.append(scored)
    except Exception:
        log("  CatBoost failed:\n" + traceback.format_exc())

    log("\nTraining HistGradientBoosting")
    try:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        encoded_fit = encoder.fit_transform(X_fit[CATEGORICAL].astype("string").fillna("missing"))
        numeric_fit = X_fit.drop(columns=CATEGORICAL).to_numpy(dtype=float)
        hgb_fit = np.hstack([encoded_fit, numeric_fit])
        hgb_model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=63,
            min_samples_leaf=80,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        hgb_model.fit(hgb_fit, y_fit)
        encoded_valid = encoder.transform(search.loc[valid_rows, CATEGORICAL].astype("string").fillna("missing"))
        numeric_valid = search.loc[valid_rows].drop(columns=CATEGORICAL).to_numpy(dtype=float)
        probability = hgb_model.predict_proba(np.hstack([encoded_valid, numeric_valid]))[:, 1]
        ranking = {
            "pr_auc": float(average_precision_score(y[valid_rows], probability)),
            "roc_auc": float(roc_auc_score(y[valid_rows], probability)),
            "brier": float(brier_score_loss(y[valid_rows], probability)),
        }
        log(
            f"    hist_gradient_boosting: PR-AUC={ranking['pr_auc']:.4f}  "
            f"ROC-AUC={ranking['roc_auc']:.4f}  Brier={ranking['brier']:.4f}"
        )
        results.append({
            "name": "hist_gradient_boosting",
            "model": hgb_model,
            "encoder": encoder,
            "probability": probability,
            "family": "hist_gradient_boosting",
            "params": {"learning_rate": 0.08, "max_iter": 250, "max_leaf_nodes": 63},
            "best_iteration": int(hgb_model.n_iter_),
            **ranking,
        })
    except Exception:
        log("  HistGradientBoosting failed:\n" + traceback.format_exc())

    return results


def serializable_params(params: dict) -> dict:
    clean = {}
    for key, value in params.items():
        if isinstance(value, (np.floating, float)):
            clean[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            clean[key] = int(value)
        elif isinstance(value, (str, bool)) or value is None:
            clean[key] = value
    return clean


def history_payload(history: dict) -> dict:
    payload = {"prior": history["prior"], "tables": {}}
    for name, table in history["tables"].items():
        payload["tables"][name] = {
            "m": table["m"],
            "key": table["sum"].index.astype(str).tolist(),
            "sum": table["sum"].astype(float).tolist(),
            "count": table["count"].astype(float).tolist(),
        }
    return payload


def feature_importance(model, names: list[str]) -> list[dict]:
    if isinstance(model, AverageEnsemble):
        rows = [feature_importance(member, names) for _name, member in model.members]
        rows = [row for row in rows if row]
        if not rows:
            return []
        combined = pd.DataFrame(rows[0]).set_index("feature")["importance"].astype(float)
        for extra in rows[1:]:
            combined = combined.add(
                pd.DataFrame(extra).set_index("feature")["importance"].astype(float),
                fill_value=0.0,
            )
        combined = combined / len(rows)
        return [
            {"feature": feature, "importance": float(value)}
            for feature, value in combined.sort_values(ascending=False).items()
        ]
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_feature_importance"):
        values = np.asarray(model.get_feature_importance(), dtype=float)
    if values is None or len(values) != len(names):
        return []
    order = np.argsort(values)[::-1]
    return [{"feature": names[i], "importance": float(values[i])} for i in order]


def write_comparison(report: dict) -> None:
    chosen = report["validation_metrics_operating_threshold"]
    points = report["operating_points"]
    lines = [
        "# SafeRoute-AI V2 vs V3 vs V4",
        "",
        "Validation years are 2014–2015. Training years are 2009–2013.",
        "V4 did not use 2016–2017. The target is KSI (serious or fatal) versus slight, conditional on a recorded accident.",
        "",
        "## Ranking metrics",
        "",
        "| Model | Accuracy | ROC-AUC | PR-AUC | KSI precision | KSI recall | KSI F1 | Brier | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| V2 published | {V2_PUBLISHED['accuracy']:.4f} | {V2_PUBLISHED['roc_auc']:.4f} | {V2_PUBLISHED['pr_auc']:.4f} | n/a | n/a | {V2_PUBLISHED['f1']:.4f} | n/a | {V2_PUBLISHED['threshold']:.3f} |",
    ]
    v3 = report["v3_reference"]
    lines.append(
        f"| V3 | {v3['accuracy']:.4f} | {v3['roc_auc']:.4f} | {v3['pr_auc']:.4f} | {v3['precision']:.4f} | {v3['recall']:.4f} | {v3['f1']:.4f} | n/a | {v3['threshold']:.3f} |"
    )
    lines.append(
        f"| V4 operating point | {chosen['accuracy']:.4f} | {chosen['roc_auc']:.4f} | {chosen['pr_auc']:.4f} | {chosen['precision']:.4f} | {chosen['recall']:.4f} | {chosen['f1']:.4f} | {chosen['brier']:.4f} | {chosen['threshold']:.3f} |"
    )
    common = report["validation_metrics_threshold_0.175"]
    lines.append(
        f"| V4 at 0.175 | {common['accuracy']:.4f} | {common['roc_auc']:.4f} | {common['pr_auc']:.4f} | {common['precision']:.4f} | {common['recall']:.4f} | {common['f1']:.4f} | {common['brier']:.4f} | 0.175 |"
    )
    half = report["validation_metrics_threshold_0.50"]
    lines.append(
        f"| V4 at 0.50 | {half['accuracy']:.4f} | {half['roc_auc']:.4f} | {half['pr_auc']:.4f} | {half['precision']:.4f} | {half['recall']:.4f} | {half['f1']:.4f} | {half['brier']:.4f} | 0.500 |"
    )
    base = report["majority_baseline"]
    lines += [
        "",
        f"Majority-class baseline accuracy (always Slight): **{base['accuracy']:.4f}**. KSI prevalence: **{base['ksi_prevalence']:.4f}**.",
        "",
        "## V4 confusion matrix at the operating threshold",
        "",
        f"TN {chosen['confusion_matrix']['tn']:,} · FP {chosen['confusion_matrix']['fp']:,} · FN {chosen['confusion_matrix']['fn']:,} · TP {chosen['confusion_matrix']['tp']:,}",
        "",
        "## Accuracy frontier",
        "",
        f"Highest accuracy on the scanned thresholds: {points['best_accuracy']['accuracy']:.4f} at threshold {points['best_accuracy']['threshold']:.3f}, with KSI recall {points['best_accuracy']['recall']:.4f}.",
    ]
    for label, key in (
        ("recall at least 0.30", "best_accuracy_recall_at_least_0.30"),
        ("recall at least 0.40", "best_accuracy_recall_at_least_0.40"),
    ):
        row = points[key]
        if row is None:
            lines.append(f"No threshold reached {label}.")
        else:
            lines.append(
                f"Best accuracy with {label}: {row['accuracy']:.4f} at threshold {row['threshold']:.3f} "
                f"(precision {row['precision']:.4f}, recall {row['recall']:.4f}, F1 {row['f1']:.4f})."
            )
    reached = report["accuracy_target"]
    lines += [
        "",
        "## 85% accuracy target",
        "",
        reached["conclusion"],
        "",
        "## Candidate comparison on the same 2014–2015 rows",
        "",
        "These scores are from models fit on 2009–2012 and early-stopped on 2013. The saved checkpoint is the refit described below.",
        "",
        "| Candidate | PR-AUC | ROC-AUC | Brier |",
        "|---|---:|---:|---:|",
    ]
    for row in report["candidates"]:
        lines.append(f"| {row['name']} | {row['pr_auc']:.4f} | {row['roc_auc']:.4f} | {row['brier']:.4f} |")
    lines += [
        "",
        "## Selected model",
        "",
        report["recommendation"],
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "V2 precision and recall are blank because that checkpoint is not in the project. V3 Brier was not in the saved V3 metrics file.",
        "",
    ]
    COMPARISON_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    log("=" * 72)
    log("SafeRoute-AI | V4 PRE-CRASH KSI MODEL")
    log("=" * 72)
    for path in (ACCIDENT_PATH, VEHICLE_PATH, V3_METRICS_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = load_frame()
    year = data["Year"].to_numpy()
    fit_rows = (year >= 2009) & (year <= 2012)
    stop_rows = year == 2013
    train_rows = (year >= 2009) & (year <= 2013)
    valid_rows = (year >= 2014) & (year <= 2015)
    y = data["KSI"].to_numpy(dtype=int)
    log("\nSPLIT SIZES")
    log(f"  Fit 2009–2012: {int(fit_rows.sum()):,}  KSI={y[fit_rows].mean():.4f}")
    log(f"  Stop 2013:     {int(stop_rows.sum()):,}  KSI={y[stop_rows].mean():.4f}")
    log(f"  Train 2009–13: {int(train_rows.sum()):,}  KSI={y[train_rows].mean():.4f}")
    log(f"  Valid 2014–15: {int(valid_rows.sum()):,}  KSI={y[valid_rows].mean():.4f}")

    log("\nHistorical rates use only earlier years than each row")
    hist_frame, final_history = time_aware_history(data)
    hist_columns = list(hist_frame.columns)
    names = feature_names(hist_columns)
    final_levels = category_levels(data, train_rows)
    full = pd.concat([data[CATEGORICAL + BASE_NUMERIC], hist_frame], axis=1)
    del hist_frame
    # One matrix. Category levels come from 2009–2013, so 2014–2015 cannot
    # introduce new category codes into the model.
    search = cast_matrix(full, names, final_levels)
    final = search
    del full

    candidates = train_candidates(search, y, fit_rows, stop_rows, valid_rows)
    candidates = sorted(candidates, key=lambda row: (row["pr_auc"], row["roc_auc"]), reverse=True)
    log("\nCANDIDATE RANKING (2014–2015, models fit on 2009–2012)")
    for row in candidates:
        log(f"  {row['name']:24s}  PR-AUC={row['pr_auc']:.4f}  ROC-AUC={row['roc_auc']:.4f}")

    winner = candidates[0]
    log(f"\nRefitting {winner['name']} on 2009–2013")
    y_train = y[train_rows]
    y_valid = y[valid_rows]
    if winner["family"] == "lightgbm":
        saved_estimator = refit_lightgbm(
            winner["params"],
            winner["best_iteration"],
            final.loc[train_rows],
            y_train,
        )
        saved_model = saved_estimator
        probability = saved_estimator.predict_proba(final.loc[valid_rows])[:, 1]
    elif winner["family"] == "xgboost":
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "enable_categorical": True,
            "eval_metric": "aucpr",
            "n_estimators": winner["best_iteration"],
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 80,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "reg_alpha": 0.2,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        saved_estimator = xgb.XGBClassifier(**params)
        saved_estimator.fit(final.loc[train_rows], y_train, verbose=False)
        saved_model = saved_estimator
        probability = saved_estimator.predict_proba(final.loc[valid_rows])[:, 1]
    elif winner["family"] == "catboost":
        cat_columns = [final.columns.get_loc(col) for col in CATEGORICAL]
        saved_estimator = CatBoostClassifier(
            loss_function="Logloss",
            iterations=winner["best_iteration"],
            learning_rate=0.08,
            depth=6,
            l2_leaf_reg=6.0,
            random_seed=42,
            allow_writing_files=False,
            verbose=100,
            thread_count=-1,
        )
        saved_estimator.fit(Pool(final.loc[train_rows], y_train, cat_features=cat_columns))
        saved_model = saved_estimator
        probability = saved_estimator.predict_proba(final.loc[valid_rows])[:, 1]
    else:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        encoded = encoder.fit_transform(final.loc[train_rows, CATEGORICAL].astype("string").fillna("missing"))
        numeric = final.loc[train_rows].drop(columns=CATEGORICAL).to_numpy(dtype=float)
        saved_estimator = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=winner["best_iteration"],
            max_leaf_nodes=63,
            min_samples_leaf=80,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
        saved_estimator.fit(np.hstack([encoded, numeric]), y_train)
        encoded_valid = encoder.transform(final.loc[valid_rows, CATEGORICAL].astype("string").fillna("missing"))
        numeric_valid = final.loc[valid_rows].drop(columns=CATEGORICAL).to_numpy(dtype=float)
        probability = saved_estimator.predict_proba(np.hstack([encoded_valid, numeric_valid]))[:, 1]
        saved_model = {"estimator": saved_estimator, "encoder": encoder}

    probability = np.asarray(probability, dtype=float)
    refit_pr = float(average_precision_score(y_valid, probability))
    refit_roc = float(roc_auc_score(y_valid, probability))
    log(f"  Refit validation PR-AUC={refit_pr:.4f}  ROC-AUC={refit_roc:.4f}")

    # Second member for a simple average, only when it is close and uses the same frame.
    ensemble_note = "No ensemble saved."
    runner = next((row for row in candidates[1:] if row["family"] in {"lightgbm", "xgboost", "catboost"} and row["family"] != "hist_gradient_boosting"), None)
    if (
        runner is not None
        and winner["family"] in {"lightgbm", "xgboost", "catboost"}
        and runner["pr_auc"] >= winner["pr_auc"] - 0.008
    ):
        log(f"\nRefitting runner-up {runner['name']} for an equal-weight average")
        try:
            if runner["family"] == "lightgbm":
                runner_model = refit_lightgbm(runner["params"], runner["best_iteration"], final.loc[train_rows], y_train)
            elif runner["family"] == "xgboost":
                runner_model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    tree_method="hist",
                    enable_categorical=True,
                    n_estimators=runner["best_iteration"],
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=80,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    reg_alpha=0.2,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
                runner_model.fit(final.loc[train_rows], y_train, verbose=False)
            else:
                cat_columns = [final.columns.get_loc(col) for col in CATEGORICAL]
                runner_model = CatBoostClassifier(
                    loss_function="Logloss",
                    iterations=runner["best_iteration"],
                    learning_rate=0.08,
                    depth=6,
                    l2_leaf_reg=6.0,
                    random_seed=42,
                    allow_writing_files=False,
                    verbose=0,
                    thread_count=-1,
                )
                runner_model.fit(Pool(final.loc[train_rows], y_train, cat_features=cat_columns))
            if winner["family"] != "hist_gradient_boosting":
                ensemble = AverageEnsemble([(winner["name"], saved_model), (runner["name"], runner_model)])
                ensemble_probability = ensemble.predict_proba(final.loc[valid_rows])[:, 1]
                ensemble_pr = float(average_precision_score(y_valid, ensemble_probability))
                ensemble_roc = float(roc_auc_score(y_valid, ensemble_probability))
                log(f"  Ensemble PR-AUC={ensemble_pr:.4f}  ROC-AUC={ensemble_roc:.4f}")
                if ensemble_pr > refit_pr + 0.001:
                    saved_model = ensemble
                    probability = ensemble_probability
                    refit_pr = ensemble_pr
                    refit_roc = ensemble_roc
                    ensemble_note = f"Saved an equal-weight average of {winner['name']} and {runner['name']} because it raised validation PR-AUC."
                else:
                    ensemble_note = f"Tried an average with {runner['name']}. It did not improve PR-AUC enough to replace {winner['name']}."
        except Exception:
            ensemble_note = "Ensemble refit failed and was discarded.\n" + traceback.format_exc()
            log(ensemble_note)

    # Calibration on 2013 scores from the 2009–2012 model. Applied only as a
    # comparison; it is not fit on 2014–2015.
    calibration = {"used_in_checkpoint": False}
    try:
        stop_probability = np.asarray(winner["model"].predict_proba(search.loc[stop_rows])[:, 1], dtype=float)
        valid_from_search = np.asarray(winner["probability"], dtype=float)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(stop_probability, y[stop_rows])
        calibrated = iso.predict(valid_from_search)
        calibration = {
            "method": "Isotonic regression fit on 2013 predictions of the 2009–2012 model",
            "used_in_checkpoint": False,
            "search_model_brier": float(brier_score_loss(y_valid, valid_from_search)),
            "calibrated_brier": float(brier_score_loss(y_valid, calibrated)),
            "search_model_pr_auc": float(average_precision_score(y_valid, valid_from_search)),
            "calibrated_pr_auc": float(average_precision_score(y_valid, calibrated)),
            "search_model_roc_auc": float(roc_auc_score(y_valid, valid_from_search)),
            "calibrated_roc_auc": float(roc_auc_score(y_valid, calibrated)),
        }
        log(
            f"\nCalibration on 2013: Brier {calibration['search_model_brier']:.4f} -> "
            f"{calibration['calibrated_brier']:.4f}; PR-AUC {calibration['calibrated_pr_auc']:.4f}"
        )
    except Exception:
        calibration["error"] = traceback.format_exc()
        log("Calibration comparison failed and was skipped.")

    table = threshold_table(y_valid, probability)
    points = operating_points(table)
    operating_threshold = float(points["best_f1"]["threshold"])
    at_operating = metrics_at(y_valid, probability, operating_threshold)
    at_0175 = metrics_at(y_valid, probability, 0.175)
    at_050 = metrics_at(y_valid, probability, 0.50)
    baseline = majority_baseline(y_valid)
    accuracy_target_hit = bool(points["best_accuracy"]["accuracy"] >= 0.85 and points["best_accuracy"]["recall"] >= 0.30)
    if points["best_accuracy"]["accuracy"] >= 0.85 and points["best_accuracy"]["recall"] < 0.30:
        conclusion = (
            f"A scanned threshold reaches {points['best_accuracy']['accuracy']:.2%} accuracy, "
            f"but KSI recall is only {points['best_accuracy']['recall']:.2%}. "
            "That point is unsuitable: it is at or just above the always-Slight baseline and does not meaningfully detect serious or fatal injuries. "
            f"The selected operating point keeps KSI F1 at {at_operating['f1']:.4f} with recall {at_operating['recall']:.2%} and accuracy {at_operating['accuracy']:.2%}."
        )
    elif not accuracy_target_hit:
        conclusion = (
            f"85% accuracy with meaningful KSI detection was not reached. "
            f"The highest scanned accuracy is {points['best_accuracy']['accuracy']:.2%} "
            f"with KSI recall {points['best_accuracy']['recall']:.2%}. "
            f"The always-Slight baseline is already {baseline['accuracy']:.2%}, so accuracy near 85% mostly means predicting Slight. "
            f"The selected operating point uses threshold {operating_threshold:.3f}: "
            f"accuracy {at_operating['accuracy']:.2%}, KSI precision {at_operating['precision']:.2%}, "
            f"recall {at_operating['recall']:.2%}, F1 {at_operating['f1']:.4f}."
        )
    else:
        conclusion = (
            f"85% accuracy was reached with KSI recall {points['best_accuracy']['recall']:.2%}. "
            "The operating point below is still the one selected by KSI F1, not by accuracy."
        )

    v3_metrics = json.loads(V3_METRICS_PATH.read_text(encoding="utf-8"))["validation_metrics"]
    better_pr = at_operating["pr_auc"] > v3_metrics["pr_auc"]
    better_roc = at_operating["roc_auc"] > v3_metrics["roc_auc"]
    better_f1 = at_operating["f1"] > v3_metrics["f1"]
    if better_pr and better_roc and better_f1:
        recommendation = (
            f"V4 improves validation PR-AUC ({at_operating['pr_auc']:.4f} vs {v3_metrics['pr_auc']:.4f}), "
            f"ROC-AUC ({at_operating['roc_auc']:.4f} vs {v3_metrics['roc_auc']:.4f}), and "
            f"KSI F1 ({at_operating['f1']:.4f} vs {v3_metrics['f1']:.4f}). "
            "It is the stronger severity model. Do not deploy it until you approve a dashboard update, because its feature set is not the V3 feature set."
        )
    elif better_pr or better_roc:
        recommendation = (
            f"V4 changes the ranking metrics (PR-AUC {at_operating['pr_auc']:.4f} vs V3 {v3_metrics['pr_auc']:.4f}, "
            f"ROC-AUC {at_operating['roc_auc']:.4f} vs V3 {v3_metrics['roc_auc']:.4f}) and "
            f"KSI F1 is {at_operating['f1']:.4f} vs V3 {v3_metrics['f1']:.4f}. "
            "Keep V3 deployed. Replace it only if this specific trade-off is the one you want."
        )
    else:
        recommendation = (
            f"V4 does not beat V3 on the selection metrics. V4 PR-AUC {at_operating['pr_auc']:.4f}, "
            f"ROC-AUC {at_operating['roc_auc']:.4f}, KSI F1 {at_operating['f1']:.4f} versus V3 "
            f"{v3_metrics['pr_auc']:.4f}, {v3_metrics['roc_auc']:.4f}, {v3_metrics['f1']:.4f}. "
            "Keep V3 as the deployed model."
        )

    importances = feature_importance(saved_model if not isinstance(saved_model, dict) else saved_model["estimator"], names)
    if importances:
        pd.DataFrame(importances).to_csv(IMPORTANCE_PATH, index=False)

    elapsed = time.time() - started
    checkpoint = {
        "model_name": "SafeRoute-AI V4",
        "model_type": winner["family"] if not isinstance(saved_model, AverageEnsemble) else "average_ensemble",
        "saved_model": saved_model,
        "feature_names": names,
        "categorical_features": CATEGORICAL,
        "category_levels": final_levels,
        "historical_rates": history_payload(final_history),
        "historical_spec": [
            {"feature": name, "key": key, "smoothing": smoothing, "stores_log_count": store}
            for name, key, smoothing, store in HIST_RATES
        ],
        "selected_threshold": operating_threshold,
        "threshold_rule": "Maximum KSI F1 on 2014–2015. Accuracy was not the selection target.",
        "validation_pr_auc": float(at_operating["pr_auc"]),
        "validation_roc_auc": float(at_operating["roc_auc"]),
        "winner_search_name": winner["name"],
        "winner_params": serializable_params(winner.get("params", {})),
        "best_iteration": winner.get("best_iteration"),
        "ensemble_note": ensemble_note,
        "years": {"fit": "2009-2012", "early_stopping": 2013, "refit": "2009-2013", "validation": "2014-2015"},
        "holdout_2016_2017": "Not loaded and not used.",
        "target": "KSI (fatal or serious) vs slight, conditional on a recorded accident",
    }
    joblib.dump(checkpoint, MODEL_PATH)
    log(f"\nSaved {MODEL_PATH}")

    candidate_rows = [
        {
            "name": row["name"],
            "family": row["family"],
            "pr_auc": row["pr_auc"],
            "roc_auc": row["roc_auc"],
            "brier": row["brier"],
            "best_iteration": row.get("best_iteration"),
            "params": serializable_params(row.get("params", {})),
        }
        for row in candidates
    ]
    report = {
        "model_name": "SafeRoute-AI V4",
        "checkpoint": str(MODEL_PATH),
        "training_seconds": elapsed,
        "training_minutes": elapsed / 60.0,
        "holdout_2016_2017": "Not loaded and not used for features, tuning, threshold selection, or reporting.",
        "target": "KSI vs Slight, conditional on a recorded accident. Not the probability that a crash occurs.",
        "split_rows": {
            "fit_2009_2012": int(fit_rows.sum()),
            "early_stop_2013": int(stop_rows.sum()),
            "train_2009_2013": int(train_rows.sum()),
            "validation_2014_2015": int(valid_rows.sum()),
        },
        "feature_count": len(names),
        "feature_names": names,
        "leakage_audit": {
            "used": [
                "Road class, road type, junction, crossings, speed limit, weather, surface, lighting, urban/rural, day, time, and coordinates.",
                "Ego vehicle only: type, driver age band, journey purpose, vehicle age, engine capacity. Other vehicles are not used.",
                "Interactions and cyclical time encodings derived from those inputs.",
                "Historical KSI rates and location counts from strictly earlier years. Validation rows use 2009–2013 only.",
            ],
            "excluded": [
                "Accident severity is the target only.",
                "Number of casualties, number of vehicles, police attendance.",
                "Other vehicles, manoeuvres, skidding, overturning, and hit-object fields.",
                "Carriageway hazards and special conditions at site: scene observations the scenario form cannot supply.",
                "LSOA, local authority, and road number: not scenario inputs. Location risk uses latitude/longitude grids instead.",
                "2016–2017 rows were not loaded.",
            ],
        },
        "majority_baseline": baseline,
        "candidates": candidate_rows,
        "selected_candidate": winner["name"],
        "selected_model_type": checkpoint["model_type"],
        "ensemble_note": ensemble_note,
        "calibration": calibration,
        "selected_threshold": operating_threshold,
        "validation_metrics_operating_threshold": at_operating,
        "validation_metrics_threshold_0.175": at_0175,
        "validation_metrics_threshold_0.50": at_050,
        "operating_points": points,
        "calibration_bins": calibration_table(y_valid, probability),
        "accuracy_target": {
            "target_accuracy": 0.85,
            "reached_with_meaningful_ksi_recall": accuracy_target_hit,
            "conclusion": conclusion,
        },
        "v2_published": V2_PUBLISHED,
        "v3_reference": {
            "threshold": v3_metrics["threshold"],
            "accuracy": v3_metrics["accuracy"],
            "precision": v3_metrics["precision"],
            "recall": v3_metrics["recall"],
            "f1": v3_metrics["f1"],
            "roc_auc": v3_metrics["roc_auc"],
            "pr_auc": v3_metrics["pr_auc"],
            "confusion_matrix": v3_metrics["confusion_matrix"],
        },
        "recommendation": recommendation,
        "top_features": importances[:25],
        "deployed": False,
        "dashboard_modified": False,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_comparison(report)
    log(f"Saved {METRICS_PATH}")
    log(f"Saved {COMPARISON_PATH}")
    log("\n" + recommendation)
    log(f"Training time: {elapsed / 60.0:.1f} minutes")
    log(conclusion)


if __name__ == "__main__":
    main()
