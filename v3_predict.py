"""Build the exact V3 feature row used at prediction time.

Derived flags match train_saferoute_v3.py. Categorical values must be
training levels from the checkpoint (including crossing codes stored
as strings such as "0.0").
"""

from __future__ import annotations

import math
import re

import pandas as pd

WET_SURFACE = re.compile(r"wet|damp|flood|snow|ice|frost", re.IGNORECASE)

# STATS20 codes, stored by V3 as the strings seen in category_levels.
CROSSING_PHYSICAL = {
    "None within 50 metres": "0.0",
    "Zebra": "1.0",
    "Pelican, puffin, toucan or similar": "4.0",
    "Pedestrian phase at traffic signal": "5.0",
    "Footbridge or subway": "7.0",
    "Central refuge": "8.0",
}
CROSSING_HUMAN = {
    "None within 50 metres": "0.0",
    "School crossing patrol": "1.0",
    "Other authorised person": "2.0",
}


def derive_engineered(raw: dict) -> dict:
    """Add the nine engineered numeric features from user inputs."""
    hour = float(raw["Hour"])
    month = float(raw["Month"])
    speed = float(raw["Speed_limit"])
    night = float(raw["Night"])
    urban = str(raw["Urban_or_Rural_Area"]).strip().lower()
    weather = str(raw["Weather_Conditions"]).strip().lower()
    surface = str(raw["Road_Surface_Conditions"]).strip().lower()

    is_rural = "rural" in urban
    is_urban = "urban" in urban
    is_high = 1.0 if speed >= 60 else 0.0
    fine_like = ("fine" in weather) or (weather == "unknown")

    out = dict(raw)
    out["Hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    out["Hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    out["Month_sin"] = math.sin(2 * math.pi * month / 12.0)
    out["Month_cos"] = math.cos(2 * math.pi * month / 12.0)
    out["Is_High_Speed"] = is_high
    out["Is_Dark_Rural"] = 1.0 if (night == 1.0 and is_rural) else 0.0
    out["Is_Adverse_Weather"] = 0.0 if fine_like else 1.0
    out["Is_Wet_Surface"] = 1.0 if WET_SURFACE.search(surface) else 0.0
    out["Urban_High_Speed"] = 1.0 if (is_high == 1.0 and is_urban) else 0.0
    return out


def build_model_frame(raw: dict, checkpoint: dict) -> pd.DataFrame:
    """One-row frame in checkpoint feature order, categories fitted on train."""
    features = list(checkpoint["feature_names"])
    categorical = list(checkpoint["categorical_features"])
    levels = checkpoint["category_levels"]

    row = derive_engineered(raw)
    missing = [name for name in features if name not in row]
    if missing:
        raise ValueError(f"Dashboard did not supply: {missing}")

    frame = pd.DataFrame([row], columns=features)

    for col in categorical:
        value = str(frame.at[0, col]).strip()
        allowed = list(levels[col])
        if value not in allowed:
            raise ValueError(
                f"Unknown category for {col}: {value!r}. "
                f"Allowed example: {allowed[:6]}"
            )
        frame[col] = pd.Categorical([value], categories=allowed)

    for col in features:
        if col not in categorical:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")

    if list(frame.columns) != features:
        raise ValueError("Feature order does not match the V3 checkpoint.")

    return frame


def predict_ksi(raw: dict, checkpoint: dict) -> float:
    frame = build_model_frame(raw, checkpoint)
    model = checkpoint["model"]
    if list(frame.columns) != list(model.feature_name_):
        raise ValueError("Frame columns do not match the saved LightGBM feature order.")
    return float(model.predict_proba(frame)[0, 1])
