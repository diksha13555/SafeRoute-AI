"""Local SHAP explanations for the deployed V3 LightGBM model.

Contributions are TreeSHAP values in the booster's raw score. For this
binary model that raw score is a log-odds. The logistic function of
baseline plus the sum of contributions is the predicted probability.
"""

from __future__ import annotations

import math
import warnings

import lightgbm as lgb
import numpy as np
import shap

SHAP_VERSION = shap.__version__
LIGHTGBM_VERSION = lgb.__version__

FEATURE_LABELS = {
    "1st_Road_Class": "Road class",
    "2nd_Road_Class": "Second road class",
    "Day_of_Week": "Day of week",
    "Junction_Control": "Junction control",
    "Junction_Detail": "Junction detail",
    "Light_Conditions": "Lighting",
    "Road_Surface_Conditions": "Road surface",
    "Road_Type": "Road type",
    "Urban_or_Rural_Area": "Urban or rural",
    "Weather_Conditions": "Weather",
    "Pedestrian_Crossing-Physical_Facilities": "Pedestrian crossing",
    "Pedestrian_Crossing-Human_Control": "Crossing control",
    "Ego_Vehicle_Type": "Vehicle type",
    "Ego_Driver_Age_Band": "Driver age band",
    "Ego_Journey_Purpose": "Journey purpose",
    "Speed_limit": "Speed limit",
    "Hour": "Hour",
    "Month": "Month",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Weekend": "Weekend",
    "Rush_Hour": "Rush hour",
    "Night": "Night",
    "Year": "Year",
    "Ego_Vehicle_Age": "Vehicle age",
    "Ego_Engine_Capacity": "Engine capacity",
    "Hour_sin": "Hour cycle (sine)",
    "Hour_cos": "Hour cycle (cosine)",
    "Month_sin": "Month cycle (sine)",
    "Month_cos": "Month cycle (cosine)",
    "Is_High_Speed": "High-speed flag",
    "Is_Dark_Rural": "Dark rural flag",
    "Is_Adverse_Weather": "Adverse weather flag",
    "Is_Wet_Surface": "Wet surface flag",
    "Urban_High_Speed": "Urban high-speed flag",
}

FEATURE_NOTES = {
    "Hour_sin": "This is the sine part of the hour encoding.",
    "Hour_cos": "This is the cosine part of the hour encoding.",
    "Month_sin": "This is the sine part of the month encoding.",
    "Month_cos": "This is the cosine part of the month encoding.",
    "Is_High_Speed": "This flag is on when the speed limit is 60 mph or higher.",
    "Is_Dark_Rural": "This flag is on when the hour is at night and the area is rural.",
    "Is_Adverse_Weather": "This flag is on when weather is outside the fine and unknown categories.",
    "Is_Wet_Surface": "This flag is on when the surface is wet, damp, flooded, snowy, icy, or frosty.",
    "Urban_High_Speed": "This flag is on when the area is urban and the speed limit is 60 mph or higher.",
    "Latitude": "Latitude is a coordinate the trees can split on.",
    "Longitude": "Longitude is a coordinate the trees can split on.",
}

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

PHYSICAL_BY_CODE = {
    "0.0": "None within 50 metres",
    "1.0": "Zebra",
    "4.0": "Pelican, puffin, toucan or similar",
    "5.0": "Pedestrian phase at traffic signal",
    "7.0": "Footbridge or subway",
    "8.0": "Central refuge",
}
HUMAN_BY_CODE = {
    "0.0": "None within 50 metres",
    "1.0": "School crossing patrol",
    "2.0": "Other authorised person",
}


def logistic(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(score)))


def make_explainer(model):
    """TreeSHAP explainer for the deployed booster.

    SHAP 0.54 warns that a LightGBM binary classifier can return a list of
    arrays. The call below uses the Explanation object and then checks it
    against LightGBM's own pred_contrib output.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
        )
        return shap.TreeExplainer(model)


def _vector_from_explanation(explanation, raw_score: float):
    values = np.asarray(explanation.values, dtype=float)
    base = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    names = [str(name) for name in explanation.feature_names]

    if values.ndim == 2 and values.shape[0] == 1:
        candidates = [(values[0], float(base[0]))]
    elif values.ndim == 1:
        candidates = [(values, float(base[0]))]
    elif values.ndim == 3 and values.shape[0] == 1:
        candidates = []
        for class_index in range(values.shape[-1]):
            base_value = float(base[min(class_index, base.size - 1)])
            candidates.append((values[0, :, class_index], base_value))
    else:
        raise ValueError(f"Unexpected SHAP value shape {values.shape}.")

    best = None
    for vector, base_value in candidates:
        error = abs(float(vector.sum()) + base_value - raw_score)
        if best is None or error < best[0]:
            best = (error, vector, base_value)
    if best[0] > 1e-5:
        raise ValueError(
            "SHAP values do not add back to the deployed model's raw score "
            f"(gap {best[0]:.3e})."
        )
    return best[1], best[2], names


def explain_frame(explainer, model, frame) -> dict:
    """Explain one row. Feature order must be the deployed booster order."""
    feature_names = [str(name) for name in model.feature_name_]
    if list(frame.columns) != feature_names:
        raise ValueError("Scenario columns are not in the deployed V3 feature order.")

    raw_score = float(model.predict(frame, raw_score=True)[0])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
        )
        explanation = explainer(frame, check_additivity=True)
    vector, base_value, names = _vector_from_explanation(explanation, raw_score)
    if names != feature_names:
        raise ValueError("SHAP feature names do not match the deployed V3 feature order.")

    reference = np.asarray(model.predict(frame, pred_contrib=True)[0, :-1], dtype=float)
    gap = float(np.max(np.abs(vector - reference)))
    if gap > 1e-6:
        raise ValueError(
            "SHAP contributions do not match LightGBM pred_contrib on the deployed model "
            f"(max gap {gap:.3e})."
        )

    contributions = [float(value) for value in vector]
    order = sorted(range(len(contributions)), key=lambda i: abs(contributions[i]), reverse=True)
    top = [
        {"feature": feature_names[i], "shap": contributions[i]}
        for i in order[:5]
    ]
    return {
        "base_log_odds": float(base_value),
        "baseline_via_logistic": logistic(base_value),
        "raw_score": raw_score,
        "probability": logistic(raw_score),
        "feature_names": feature_names,
        "contributions": contributions,
        "top": top,
        "shap_version": SHAP_VERSION,
        "lightgbm_version": LIGHTGBM_VERSION,
    }


def plain_value(name: str, frame) -> str:
    """Readable current value for one model column."""
    value = frame.iloc[0][name]
    if name == "Pedestrian_Crossing-Physical_Facilities":
        return PHYSICAL_BY_CODE.get(str(value), str(value))
    if name == "Pedestrian_Crossing-Human_Control":
        return HUMAN_BY_CODE.get(str(value), str(value))
    if name == "Speed_limit":
        return f"{int(round(float(value)))} mph"
    if name == "Hour":
        return f"{int(round(float(value))):02d}:00"
    if name == "Month":
        month = int(round(float(value)))
        return MONTHS[month - 1] if 1 <= month <= 12 else str(month)
    if name in ("Hour_sin", "Hour_cos"):
        hour = int(round(float(frame.iloc[0]["Hour"])))
        kind = "sine" if name == "Hour_sin" else "cosine"
        return f"the {kind} encoding of {hour:02d}:00 ({float(value):+.3f})"
    if name in ("Month_sin", "Month_cos"):
        month = int(round(float(frame.iloc[0]["Month"])))
        kind = "sine" if name == "Month_sin" else "cosine"
        month_name = MONTHS[month - 1] if 1 <= month <= 12 else str(month)
        return f"the {kind} encoding of {month_name} ({float(value):+.3f})"
    if name in ("Latitude", "Longitude"):
        return f"{float(value):.6f}"
    if name == "Year":
        return str(int(round(float(value))))
    if name == "Ego_Vehicle_Age":
        return f"{float(value):.0f} years"
    if name == "Ego_Engine_Capacity":
        return f"{float(value):.0f} cc"
    if name in ("Weekend", "Rush_Hour", "Night", "Is_High_Speed", "Is_Dark_Rural", "Is_Adverse_Weather", "Is_Wet_Surface", "Urban_High_Speed"):
        return "yes" if float(value) >= 0.5 else "no"
    return str(value)


def contributor_lines(result: dict, frame) -> list[dict]:
    lines = []
    for item in result["top"]:
        name = item["feature"]
        contribution = float(item["shap"])
        lines.append(
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "value": plain_value(name, frame),
                "shap": contribution,
                "increases_ksi": contribution > 0,
                "note": FEATURE_NOTES.get(name, ""),
            }
        )
    return lines
