
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SafeRoute-AI",
    page_icon="🚦",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    road = joblib.load(
        os.path.join(
            BASE_DIR,
            "saferoute_full_lightgbm_final.pkl"
        )
    )
    vehicle = joblib.load(
        os.path.join(
            BASE_DIR,
            "saferoute_module2_final.pkl"
        )
    )
    return road, vehicle

road_package, vehicle_package = load_models()

def prepare_input(values, package):
    """Build exactly the features and categories used in training."""
    features = package["features"]
    categorical = package.get(
        "categorical_cols",
        package.get("categorical_features", [])
    )
    category_levels = package["category_levels"]

    row = {}

    for feature in features:
        if feature not in values:
            raise ValueError(f"Missing feature: {feature}")
        row[feature] = values[feature]

    X = pd.DataFrame([row], columns=features)

    for col in features:
        if col in categorical:
            levels = category_levels[col]
            if X.at[0, col] not in levels:
                raise ValueError(
                    f"Unknown category for {col}: {X.at[0, col]}"
                )
            X[col] = pd.Categorical(
                X[col],
                categories=levels
            )
        else:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            ).astype("float32")

    # Module 2 allows missing vehicle age/engine information.
    # LightGBM learned how to handle NaN during training.
    allowed_missing = set()

    if package.get("module") == "Module 2" or (
        "Vehicle_Record_Count" in features
    ):
        allowed_missing = {
            "Mean_Vehicle_Age",
            "Max_Vehicle_Age",
            "Mean_Engine_Capacity",
            "Max_Engine_Capacity"
        }

    missing = X.columns[X.isna().any()].tolist()

    invalid_missing = [
        col for col in missing
        if col not in allowed_missing
    ]

    if invalid_missing:
        raise ValueError(
            f"Invalid inputs: {invalid_missing}"
        )

    for col, cap in package.get(
        "numeric_caps", {}
    ).items():
        if col in X.columns:
            X[col] = X[col].clip(upper=cap)

    return X

def predict_severity(values, package):
    X = prepare_input(values, package)

    model = package["model"]
    probabilities = model.predict_proba(X)[0]

    weights = np.array([
        package["prediction_weights"].get(
            cls, 1
        )
        for cls in model.classes_
    ])

    adjusted = probabilities * weights
    predicted = model.classes_[np.argmax(adjusted)]

    # Normalized weighted scores for display.
    # These are NOT calibrated probabilities.
    scores = adjusted / adjusted.sum()

    return predicted, pd.DataFrame({
        "Severity": model.classes_,
        "Model Score": scores
    })


def aggregate_vehicle_features(vehicles):
    """Create the 14 Module 2 vehicle features."""

    if not vehicles:
        raise ValueError("Enter at least one vehicle.")

    data = pd.DataFrame(vehicles)

    for col in ["Vehicle_Age", "Engine_Capacity"]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    types = (
        data["Vehicle_Type"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    first = data.iloc[0]

    return {
        "Vehicle_Record_Count": len(data),
        "Mean_Vehicle_Age": data["Vehicle_Age"].mean(),
        "Mean_Engine_Capacity": data["Engine_Capacity"].mean(),
        "First_Driver_Age_Band": first["Driver_Age_Band"],
        "First_Journey_Purpose": first["Journey_Purpose"],
        "First_Vehicle_Manoeuvre": first["Vehicle_Manoeuvre"],
        "First_Vehicle_Type": first["Vehicle_Type"],
        "Has_Motorcycle": int(
            types.str.contains(
                "motorcycle|motorbike|moped", regex=True
            ).any()
        ),
        "Has_Bicycle": int(
            types.str.contains(
                "pedal cycle|bicycle", regex=True
            ).any()
        ),
        "Has_Bus": int(
            types.str.contains("bus|coach", regex=True).any()
        ),
        "Has_Goods_Vehicle": int(
            types.str.contains("goods|van", regex=True).any()
        ),
        "Distinct_Vehicle_Types":
            data["Vehicle_Type"].nunique(),
        "Max_Vehicle_Age": data["Vehicle_Age"].max(),
        "Max_Engine_Capacity":
            data["Engine_Capacity"].max()
    }


def vehicle_entry_form(package, prefix):
    """Collect individual vehicle details in Streamlit."""

    st.subheader("Vehicles involved")

    count = st.number_input(
        "Number of vehicles",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key=f"{prefix}_vehicle_count"
    )

    category_levels = package["category_levels"]

    fields = {
        "Vehicle_Type": "First_Vehicle_Type",
        "Driver_Age_Band": "First_Driver_Age_Band",
        "Journey_Purpose": "First_Journey_Purpose",
        "Vehicle_Manoeuvre": "First_Vehicle_Manoeuvre"
    }

    vehicles = []

    for i in range(int(count)):
        with st.expander(
            f"Vehicle {i + 1}",
            expanded=(i == 0)
        ):
            vehicle = {}

            for field, model_field in fields.items():
                options = list(
                    category_levels[model_field]
                )

                default = {
                    "Vehicle_Type": "Car",
                    "Driver_Age_Band": "26 - 35",
                    "Journey_Purpose": "Other",
                    "Vehicle_Manoeuvre": "Going ahead other"
                }.get(field)

                index = (
                    options.index(default)
                    if default in options else 0
                )

                vehicle[field] = st.selectbox(
                    field.replace("_", " "),
                    options,
                    index=index,
                    key=f"{prefix}_{field}_{i}"
                )

            unknown_age = st.checkbox(
                "Vehicle age unknown",
                value=False,
                key=f"{prefix}_unknown_age_{i}"
            )

            vehicle["Vehicle_Age"] = (
                None if unknown_age else st.number_input(
                    "Vehicle age (years)",
                    min_value=0,
                    max_value=150,
                    value=5,
                    key=f"{prefix}_age_{i}"
                )
            )

            unknown_engine = st.checkbox(
                "Engine capacity unknown / not applicable",
                value=False,
                key=f"{prefix}_unknown_engine_{i}"
            )

            vehicle["Engine_Capacity"] = (
                None if unknown_engine else st.number_input(
                    "Engine capacity (cc)",
                    min_value=0,
                    max_value=100000,
                    value=1200,
                    key=f"{prefix}_engine_{i}"
                )
            )

            vehicles.append(vehicle)

    return aggregate_vehicle_features(vehicles)


def input_form(package, key_prefix, vehicle=False):
    """Render model-specific inputs."""

    features = package["features"]
    categorical = package.get(
        "categorical_cols",
        package.get("categorical_features", [])
    )
    category_levels = package["category_levels"]

    values = {}

    # A compact interface with expandable groups
    road_cats = [
        "1st_Road_Class",
        "Day_of_Week",
        "Junction_Control",
        "Junction_Detail",
        "Light_Conditions",
        "Road_Surface_Conditions",
        "Road_Type",
        "Urban_or_Rural_Area",
        "Weather_Conditions"
    ]

    vehicle_cats = [
        "First_Driver_Age_Band",
        "First_Journey_Purpose",
        "First_Vehicle_Manoeuvre",
        "First_Vehicle_Type"
    ]

    st.subheader("Road and Weather Conditions")

    with st.container(border=True):
        c1, c2 = st.columns(2)

        for i, col in enumerate(road_cats):
            if col in features:
                with (c1 if i % 2 == 0 else c2):
                    values[col] = st.selectbox(
                        col.replace("_", " "),
                        options=category_levels[col],
                        key=f"{key_prefix}_{col}"
                    )

        with c1:
            values["Speed_limit"] = st.number_input(
                "Speed limit (mph)",
                min_value=0,
                max_value=100,
                value=30,
                step=10,
                key=f"{key_prefix}_speed"
            )

        with c2:
            values["Latitude"] = st.number_input(
                "Latitude (UK)",
                min_value=49.0,
                max_value=61.0,
                value=52.0,
                format="%.5f",
                key=f"{key_prefix}_lat"
            )

            values["Longitude"] = st.number_input(
                "Longitude (UK)",
                min_value=-9.0,
                max_value=3.0,
                value=-1.0,
                format="%.5f",
                key=f"{key_prefix}_lon"
            )

    st.subheader("Date and Time")

    c1, c2, c3 = st.columns(3)

    with c1:
        values["Hour"] = st.number_input(
            "Hour (0–23)",
            min_value=0,
            max_value=23,
            value=12,
            key=f"{key_prefix}_hour"
        )

    with c2:
        values["Month"] = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=6,
            key=f"{key_prefix}_month"
        )

    with c3:
        values["Year"] = st.number_input(
            "Accident year",
            min_value=2005 if not vehicle else 2009,
            max_value=2017 if not vehicle else 2016,
            value=2016,
            key=f"{key_prefix}_year"
        )

    day = values["Day_of_Week"]
    hour = values["Hour"]

    # Module 2 training data used Weekend=0 for every day.
    # Preserve this for the saved Module 2 model.

    # Match the exact rules used during model training.
    if vehicle:
        # Module 2 training data has Weekend=0.
        values["Weekend"] = 0
        values["Rush_Hour"] = int(
            hour in [7, 8, 9, 16, 17, 18, 19]
        )
    else:
        # Module 1 uses actual weekend days.
        values["Weekend"] = int(
            day in ["Saturday", "Sunday"]
        )
        values["Rush_Hour"] = int(
            hour in [7, 8, 9, 16, 17, 18]
        )

    values["Night"] = int(
        hour < 6 or hour >= 20
    )

    if vehicle:
        vehicle_features = vehicle_entry_form(
            package, key_prefix
        )
        values.update(vehicle_features)

    if st.button(
        "Predict Accident Severity",
        type="primary",
        key=f"{key_prefix}_predict"
    ):
        try:
            prediction, scores = predict_severity(
                values, package
            )

            st.subheader("Prediction Result")

            st.metric(
                "Predicted Severity",
                prediction
            )

            st.write("Relative model scores:")
            st.bar_chart(
                scores.set_index("Severity")
            )

            st.caption(
                "Scores are adjusted model outputs, "
                "not calibrated probabilities. "
                "This is a research demonstration."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.title("🚦 SafeRoute-AI")
st.caption(
    "Explainable Machine Learning for Road Accident Severity"
)

st.warning(
    "Research prototype only. These models classify severity "
    "conditional on an accident occurring. They do not "
    "predict whether an accident will happen."
)

tab1, tab2, tab3 = st.tabs([
    "🛣️ Road Severity",
    "🚗 Vehicle Severity",
    "📊 Model Insights"
])

with tab1:
    st.header("Module 1: Road-Based Severity")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", "74.83%")
    c2.metric("Macro F1", "0.3935")
    c3.metric("Balanced Accuracy", "41.16%")

    input_form(
        road_package,
        "road",
        vehicle=False
    )

with tab2:
    st.header("Module 2: Vehicle-Enhanced Severity")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", "72.75%")
    c2.metric("Macro F1", "0.4421")
    c3.metric("Balanced Accuracy", "49.95%")

    input_form(
        vehicle_package,
        "vehicle",
        vehicle=True
    )

with tab3:
    st.header("Explainable AI Insights")

    st.write(
        "Global SHAP feature importance for Module 2, "
        "estimated from 300 validation records."
    )

    chart_path = os.path.join(
        BASE_DIR,
        "module2_shap_importance.png"
    )

    if os.path.exists(chart_path):
        st.image(
            chart_path,
            use_container_width=True
        )

    csv_path = os.path.join(
        BASE_DIR,
        "module2_shap_importance.csv"
    )

    if os.path.exists(csv_path):
        importance = pd.read_csv(csv_path)
        st.dataframe(
            importance.head(15),
            hide_index=True,
            use_container_width=True
        )

    st.info(
        "SHAP values measure model influence, not causation."
    )

st.divider()

st.caption(
    "SafeRoute-AI | Research Prototype | LightGBM + SHAP"
)
