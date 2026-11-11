"""SafeRoute-AI Streamlit dashboard — V3 pre-crash KSI model.

Every V3 feature is either collected in the form or derived from those
inputs with the same rules used in training. The 2016–2017 holdout is
not used here.
"""

from datetime import date, time
from pathlib import Path
import json

import joblib
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from v3_explain import contributor_lines, explain_frame, make_explainer
from v3_predict import (
    CROSSING_HUMAN,
    CROSSING_PHYSICAL,
    build_model_frame,
    predict_ksi,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "saferoute_v3_precrash_2026.pkl"

DISCLAIMER = (
    "The collision and the mechanism below are a fictional teaching illustration "
    "built from the scenario you selected. The model did not identify an accident "
    "cause, and it does not say that weather, speed, or the junction caused a crash. "
    "It only estimates how severe the injury would be if an accident were recorded."
)


def incident_story(weather, surface, lighting, junction, speed_limit, area):
    """Scenario assumptions for the illustration. Not a model explanation."""
    factors = []
    weather_l = str(weather).lower()
    surface_l = str(surface).lower()
    lighting_l = str(lighting).lower()
    junction_l = str(junction).lower()
    speed = int(speed_limit)
    slippery = any(word in surface_l for word in ("wet", "damp", "flood", "snow", "ice", "frost")) or any(
        word in weather_l for word in ("rain", "snow")
    )
    if slippery:
        factors.append(
            "The selected road or weather is wet, icy, or snowy, so this illustration assumes reduced traction and a longer stopping distance."
        )
    if "fog" in weather_l:
        factors.append(
            "Fog is selected, so this illustration assumes the drivers see the other vehicle later."
        )
    if "darkness" in lighting_l:
        factors.append(
            "Darkness is selected, so this illustration assumes reduced visibility."
        )
    if "not at junction" not in junction_l:
        factors.append(
            f"A junction is selected ({junction}), so this illustration assumes the vehicles reach the same conflict point."
        )
    if speed >= 60:
        factors.append(
            f"The speed limit is {speed} mph, so this illustration assumes greater impact energy and less time to avoid contact."
        )
    if not factors:
        factors.append(
            "No adverse weather, darkness, or junction is selected. This illustration assumes a brief loss of attention on an ordinary road."
        )
    return factors


def what_happened(weather, surface, lighting, speed_limit, area, ksi):
    """Illustrative sentence from the selected controls. Not a model cause."""
    weather_l = str(weather).lower()
    surface_l = str(surface).lower()
    lighting_l = str(lighting).lower()
    wet = any(word in surface_l for word in ("wet", "damp", "flood", "snow", "ice", "frost")) or any(
        word in weather_l for word in ("rain", "snow")
    )
    if "fog" in weather_l:
        setting = "foggy road"
    elif wet and "darkness" in lighting_l:
        setting = "dark, wet road"
    elif wet:
        setting = "wet road"
    elif "darkness" in lighting_l:
        setting = "dark road"
    else:
        setting = f"{str(area).lower()} road with a {int(speed_limit)} mph speed limit"
    article = "an" if setting[:1].lower() in "aeiou" else "a"
    setting = f"{article} {setting}"
    kind = "stronger" if ksi else "mild"
    return (
        f"On {setting}, the fictional approaching vehicle brakes late and makes contact with the simulated car. "
        f"The contact is a {kind}, non-graphic illustration. "
        "This sequence illustrates a possible road scenario; the AI did not predict that this crash would happen."
    )


def render_explanation(model_frame, probability, threshold, decision, label):
    """SHAP section for the current scenario. Units are the model log-odds."""
    result = explain_frame(load_shap_explainer(checkpoint), checkpoint["model"], model_frame)
    if abs(result["probability"] - float(probability)) > 1e-8:
        raise RuntimeError("The SHAP reconstruction does not match the displayed KSI probability.")

    lines = contributor_lines(result, model_frame)
    st.write(
        "These are the five largest SHAP contributions for this scenario. "
        "A positive value raises the model's raw KSI score. A negative value lowers it. "
        "They describe the score, not a cause of a crash."
    )

    chart_rows = list(reversed(lines))
    colors = ["#f59e0b" if row["increases_ksi"] else "#2dd4bf" for row in chart_rows]
    fig = go.Figure(
        go.Bar(
            x=[row["shap"] for row in chart_rows],
            y=[row["label"] for row in chart_rows],
            orientation="h",
            marker_color=colors,
            text=[f"{row['shap']:+.3f}" for row in chart_rows],
            textposition="outside",
            cliponaxis=False,
            hovertext=[
                f"{row['label']}: {row['value']} · {row['shap']:+.3f} log-odds"
                for row in chart_rows
            ],
            hoverinfo="text",
        )
    )
    span = max(abs(row["shap"]) for row in lines)
    span = max(span * 1.8, 0.05)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d8efff"},
        xaxis_title="SHAP contribution (log-odds)",
        height=380,
        margin=dict(l=10, r=70, t=10, b=40),
        yaxis={"automargin": True},
    )
    fig.update_xaxes(
        range=[-span, span],
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="#8fb4c9",
        gridcolor="#1e3d55",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Amber bars increase the raw KSI score. Teal bars decrease it. "
        "Units are log-odds, not percentage points."
    )
    with st.expander("Technical explanation"):
        base_col, link_col, raw_col, prob_col = st.columns(4)
        base_col.metric("Baseline log-odds", f"{result['base_log_odds']:.4f}")
        link_col.metric("Baseline through the logistic link", f"{result['baseline_via_logistic'] * 100:.2f}%")
        raw_col.metric("This scenario's raw score", f"{result['raw_score']:.4f}")
        prob_col.metric("This scenario's KSI probability", f"{result['probability'] * 100:.2f}%")
        st.caption(
            "The baseline is the expected raw score from TreeSHAP on this booster. "
            f"{result['baseline_via_logistic'] * 100:.2f}% is the logistic function of that baseline, "
            "not the average predicted probability across the training rows. "
            f"All 35 contributions plus the baseline equal the raw score {result['raw_score']:.4f}. "
            f"The logistic function of that score is the {result['probability'] * 100:.2f}% probability. "
            f"Checked with SHAP {result['shap_version']} against LightGBM {result['lightgbm_version']} "
            "pred_contrib, in the saved feature order."
        )

    for rank, row in enumerate(lines, start=1):
        movement = "increases" if row["increases_ksi"] else "decreases"
        note = f" {row['note']}" if row["note"] else ""
        st.markdown(
            f"**{rank}. {row['label']}** — {row['value']}. "
            f"SHAP contribution `{row['shap']:+.3f}` log-odds. "
            f"This {movement} the model's raw KSI score relative to the baseline.{note}"
        )

    st.info(
        f"The selected {decision['name'].lower()} threshold of {threshold:.3f} "
        f"classifies this scenario as {label}. "
        "The SHAP contributions and the predicted probability belong to the V3 model and this scenario. "
        "They stay the same when the decision threshold changes. "
        "Switching between Standard (0.175) and Safety-focused (0.145) changes only the KSI or Slight classification."
    )


st.set_page_config(
    page_title="SafeRoute-AI | Road Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_checkpoint():
    checkpoint = joblib.load(MODEL_PATH)
    model = checkpoint["model"]
    names = list(checkpoint["feature_names"])
    if list(model.feature_name_) != names:
        raise RuntimeError("Saved V3 feature order does not match the booster.")
    if model.n_features_in_ != len(names):
        raise RuntimeError("Saved V3 feature count does not match the booster.")
    return checkpoint


@st.cache_resource
def load_shap_explainer(_checkpoint):
    return make_explainer(_checkpoint["model"])


checkpoint = load_checkpoint()
levels = checkpoint["category_levels"]
validation = checkpoint["validation_metrics"]
# 0.175 remains the original validation-F1 threshold. 0.145 is an optional
# operating point measured on the same 2014–2015 rows; it does not change the model.
DECISION_MODES = {
    "Standard mode: 0.175": {
        "threshold": 0.175,
        "name": "Standard",
        "accuracy": "72.03%",
        "recall": "46.65%",
        "precision": "26.60%",
    },
    "Safety-focused mode: 0.145": {
        "threshold": 0.145,
        "name": "Safety-focused",
        "accuracy": "62.58%",
        "recall": "61.55%",
        "precision": "23.08%",
    },
}


def choices(name: str) -> list[str]:
    return list(levels[name])


CSS = """
<style>
.stApp{
 background:
 radial-gradient(circle at 80% 0%,#123657 0%,transparent 32%),
 linear-gradient(160deg,#071421,#0a1c2e 60%,#071421);
 color:#f4fbff;
}
.block-container{max-width:1180px;padding-top:1rem}
h1,h2,h3{color:#f4fbff}
div[data-testid="stForm"]{
 background:#10283f;
 border:1px solid #3d7e9a;
 border-radius:18px;
 padding:18px;
}
div[data-testid="stMetric"]{
 background:#123049;
 border:1px solid #3d7e9a;
 border-radius:14px;
 padding:12px;
}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] *{color:#f7fbff !important}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] *{color:#d5eaf6 !important}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stRadio"] [data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] [data-testid="stMarkdownContainer"],
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] p{
 color:#f4fbff !important;
}
div[data-testid="stCaptionContainer"] p{color:#c5dcea !important}
div[data-baseweb="select"] > div,
div[data-baseweb="input"]{
 background:#123049 !important;
 border-color:#3d8eab !important;
}
div[data-baseweb="select"] *,
div[data-baseweb="input"] *,
input, textarea{
 color:#f7fbff !important;
 -webkit-text-fill-color:#f7fbff !important;
}
div[data-baseweb="popover"], div[data-baseweb="menu"]{background:#123049 !important}
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li{
 color:#f7fbff !important;
 -webkit-text-fill-color:#f7fbff !important;
 background:#123049 !important;
}
div[data-testid="stAlert"]{border-radius:12px}
.stButton button, div[data-testid="stFormSubmitButton"] button{
 background:#18486a; color:#f7fbff; border:1px solid #3d8eab; font-weight:700;
}
.stButton button[kind="primary"], div[data-testid="stFormSubmitButton"] button{
 background:linear-gradient(90deg,#08b7d1,#7737d7);
 border:0; color:#ffffff; font-weight:800;
}
.result-grid{display:grid;grid-template-columns:1.3fr .9fr .9fr;gap:12px;margin:8px 0 16px}
.result-card{background:#123049;border:1px solid #3d7e9a;border-radius:16px;padding:16px 18px}
.result-card .k{color:#c5dcea;font-size:13px;font-weight:700;letter-spacing:.04em}
.result-card .v{color:#67e8f9;font-size:42px;font-weight:850;line-height:1.1;margin-top:4px}
.result-card .v.plain{color:#f7fbff;font-size:28px}
.result-card .s{color:#d7eef8;font-size:14px;line-height:1.4;margin-top:6px}
.compare{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
@media(max-width:800px){
 .result-grid,.compare{grid-template-columns:1fr}
 .result-card .v{font-size:34px}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

hero_html = (ROOT / "hero_stage.html").read_text(encoding="utf-8")
components.html(hero_html, height=620, scrolling=False)

st.markdown('<div id="scenario-anchor"></div>', unsafe_allow_html=True)
st.header("Quick Scenario")

road_classes = choices("1st_Road_Class")
road_types = choices("Road_Type")
areas = choices("Urban_or_Rural_Area")
weathers = choices("Weather_Conditions")
surfaces = choices("Road_Surface_Conditions")
lights = choices("Light_Conditions")
junction_controls = choices("Junction_Control")
junction_details = choices("Junction_Detail")
second_classes = choices("2nd_Road_Class")
vehicle_types = choices("Ego_Vehicle_Type")
age_bands = choices("Ego_Driver_Age_Band")
purposes = choices("Ego_Journey_Purpose")

physical_labels = [
    label for label, code in CROSSING_PHYSICAL.items() if code in set(levels[
        "Pedestrian_Crossing-Physical_Facilities"
    ])
]
human_labels = [
    label for label, code in CROSSING_HUMAN.items() if code in set(levels[
        "Pedestrian_Crossing-Human_Control"
    ])
]

SCENARIO_DEFAULTS = {
    "road_class": "A" if "A" in road_classes else road_classes[0],
    "second_class": "Unclassified" if "Unclassified" in second_classes else second_classes[0],
    "road_type": "Single carriageway" if "Single carriageway" in road_types else road_types[0],
    "speed_limit": 30,
    "area": "Urban" if "Urban" in areas else areas[0],
    "weather": "Fine no high winds" if "Fine no high winds" in weathers else weathers[0],
    "surface": "Dry" if "Dry" in surfaces else surfaces[0],
    "lighting": "Daylight" if "Daylight" in lights else lights[0],
    "junction_control": "Not at junction or within 20 metres",
    "junction_detail": "Not at junction or within 20 metres",
    "scenario_date": date(2013, 6, 15),
    "scenario_time": time(14, 0),
    "latitude": 52.239552,
    "longitude": -1.350476,
    "physical_label": physical_labels[0],
    "human_label": human_labels[0],
    "ego_type": "Car" if "Car" in vehicle_types else vehicle_types[0],
    "ego_age": 5.0,
    "ego_band": "26 - 35" if "26 - 35" in age_bands else age_bands[0],
    "ego_engine": 1600.0,
    "ego_purpose": "Commuting to/from work" if "Commuting to/from work" in purposes else purposes[0],
}
for _key, _value in SCENARIO_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)

PRESETS = {
    "Urban daytime commute": dict(SCENARIO_DEFAULTS),
    "Rural high-speed road": {
        **SCENARIO_DEFAULTS,
        "area": "Rural" if "Rural" in areas else SCENARIO_DEFAULTS["area"],
        "speed_limit": 70,
        "road_type": "Single carriageway" if "Single carriageway" in road_types else SCENARIO_DEFAULTS["road_type"],
    },
    "Rainy evening journey": {
        **SCENARIO_DEFAULTS,
        "weather": "Raining no high winds" if "Raining no high winds" in weathers else SCENARIO_DEFAULTS["weather"],
        "surface": "Wet or damp" if "Wet or damp" in surfaces else SCENARIO_DEFAULTS["surface"],
        "lighting": "Darkness - lights lit" if "Darkness - lights lit" in lights else SCENARIO_DEFAULTS["lighting"],
        "scenario_time": time(19, 0),
        "speed_limit": 30,
        "area": "Urban" if "Urban" in areas else SCENARIO_DEFAULTS["area"],
    },
}

st.caption("Example presets fill the real model inputs. The probability is calculated, not stored.")
preset_cols = st.columns(3)
for preset_name, preset_col in zip(PRESETS, preset_cols):
    if preset_col.button(preset_name, width="stretch"):
        for preset_key, preset_value in PRESETS[preset_name].items():
            st.session_state[preset_key] = preset_value

with st.form("scenario_form"):
    q1, q2, q3 = st.columns(3)
    with q1:
        road_type = st.selectbox("Road type", road_types, key="road_type")
        speed_limit = st.selectbox("Speed limit (mph)", [10, 20, 30, 40, 50, 60, 70], key="speed_limit")
    with q2:
        weather = st.selectbox("Weather", weathers, key="weather")
        lighting = st.selectbox("Lighting", lights, key="lighting")
    with q3:
        area = st.selectbox("Urban / Rural", areas, key="area")
        ego_type = st.selectbox("Vehicle type", vehicle_types, key="ego_type")

    with st.expander("Advanced Settings"):
        a1, a2, a3 = st.columns(3)
        with a1:
            road_class = st.selectbox("Road class", road_classes, key="road_class")
            second_class = st.selectbox("Second road class", second_classes, key="second_class")
            surface = st.selectbox("Road surface", surfaces, key="surface")
            junction_control = st.selectbox("Junction control", junction_controls, key="junction_control")
            junction_detail = st.selectbox("Junction detail", junction_details, key="junction_detail")
        with a2:
            physical_label = st.selectbox("Pedestrian crossing", physical_labels, key="physical_label")
            human_label = st.selectbox("Crossing human control", human_labels, key="human_label")
            scenario_date = st.date_input(
                "Scenario date",
                min_value=date(2009, 1, 1),
                max_value=date(2015, 12, 31),
                key="scenario_date",
            )
            scenario_time = st.time_input("Time", key="scenario_time")
            latitude = st.number_input("Latitude", 49.9, 60.7, format="%.6f", key="latitude")
            longitude = st.number_input("Longitude", -7.6, 1.8, format="%.6f", key="longitude")
        with a3:
            ego_band = st.selectbox("Driver age band", age_bands, key="ego_band")
            ego_age = st.number_input("Vehicle age (years)", 0.0, 50.0, step=1.0, key="ego_age")
            ego_engine = st.number_input("Engine capacity (cc)", 50.0, 8000.0, step=50.0, key="ego_engine")
            ego_purpose = st.selectbox("Journey purpose", purposes, key="ego_purpose")

    submitted = st.form_submit_button("RUN AI SIMULATION", type="primary", width="stretch")

hour = scenario_time.hour
weekend = int(scenario_date.weekday() >= 5)
rush = int(hour in [7, 8, 9, 16, 17, 18, 19])
night = int(hour < 6 or hour >= 20)

raw = {
    "1st_Road_Class": road_class,
    "2nd_Road_Class": second_class,
    "Day_of_Week": scenario_date.strftime("%A"),
    "Junction_Control": junction_control,
    "Junction_Detail": junction_detail,
    "Light_Conditions": lighting,
    "Road_Surface_Conditions": surface,
    "Road_Type": road_type,
    "Urban_or_Rural_Area": area,
    "Weather_Conditions": weather,
    "Pedestrian_Crossing-Physical_Facilities": CROSSING_PHYSICAL[physical_label],
    "Pedestrian_Crossing-Human_Control": CROSSING_HUMAN[human_label],
    "Ego_Vehicle_Type": ego_type,
    "Ego_Driver_Age_Band": ego_band,
    "Ego_Journey_Purpose": ego_purpose,
    "Speed_limit": float(speed_limit),
    "Hour": float(hour),
    "Month": float(scenario_date.month),
    "Latitude": float(latitude),
    "Longitude": float(longitude),
    "Weekend": float(weekend),
    "Rush_Hour": float(rush),
    "Night": float(night),
    "Year": float(scenario_date.year),
    "Ego_Vehicle_Age": float(ego_age),
    "Ego_Engine_Capacity": float(ego_engine),
}

try:
    probability = predict_ksi(raw, checkpoint)
    model_frame = build_model_frame(raw, checkpoint)
except Exception as exc:
    st.error("Model prediction error: " + str(exc))
    st.stop()

st.header("Decision threshold")
decision_choice = st.radio(
    "Decision threshold",
    list(DECISION_MODES),
    horizontal=True,
    key="decision_threshold_mode",
)
decision = DECISION_MODES[decision_choice]
threshold = float(decision["threshold"])
label = "KSI · Serious/Fatal combined" if probability >= threshold else "Slight"
st.session_state["last_probability"] = probability
st.session_state["last_label"] = label
st.session_state["last_threshold"] = threshold

mode_left, mode_right = st.columns(2)
with mode_left:
    st.markdown("**Standard mode: 0.175**")
    st.write("Accuracy 72.03% · KSI recall 46.65% · precision 26.60%")
with mode_right:
    st.markdown("**Safety-focused mode: 0.145**")
    st.write("Accuracy 62.58% · KSI recall 61.55% · precision 23.08%")
st.info(
    "Safety-focused mode detects more KSI cases but produces more false alarms. "
    "The estimated KSI probability stays the same; only the cutoff used to classify "
    "a recorded accident as KSI or Slight changes. "
    f"This scenario is {probability * 100:.2f}%, so the selected "
    f"{decision['name'].lower()} threshold of {threshold:.3f} classifies it as {label}. "
    "This is an educational injury-severity classifier conditional on an accident, "
    "not a crash probability or an emergency warning."
)

st.header("02 / Animated Road Experience")

story = incident_story(
    weather, surface, lighting, junction_detail, speed_limit, area
)
config = {
    "snow": ("Snow" in surface or "Snowing" in weather),
    "rain": ("Raining" in weather),
    "wet": ("Wet" in surface or "Flood" in surface),
    "ice": ("Frost" in surface),
    "fog": ("Fog" in weather),
    "night": ("Darkness" in lighting),
    "prob": probability,
    "threshold": float(threshold),
    "decision_mode": decision["name"],
    "label": label,
    "ksi": bool(probability >= threshold),
    "severity": label,
    "mechanism": " ".join(story),
    "what_happened": what_happened(
        weather, surface, lighting, speed_limit, area, bool(probability >= threshold)
    ),
    "conditions": [
        f"Weather: {weather}",
        f"Road surface: {surface}",
        f"Lighting: {lighting}",
        f"Junction: {junction_detail}",
        f"Speed limit: {int(speed_limit)} mph",
        f"Area: {area}",
    ],
    "disclaimer": DISCLAIMER,
}
html = (ROOT / "simulator.html").read_text(encoding="utf-8")
html = html.replace("__CONFIG__", json.dumps(config))
components.html(html, height=980, scrolling=False)
st.caption(
    "START DRIVE and FICTIONAL INCIDENT play a fictional collision for the predicted class. "
    "KSI is the stronger illustration and Slight is the lighter one. "
    "NORMAL JOURNEY has no collision. The percentage is conditional injury severity "
    "if an accident were recorded, not the chance that a crash will occur."
)

st.header("Result")
st.markdown(
    "<div class='result-grid'>"
    "<div class='result-card'><div class='k'>CONDITIONAL KSI PROBABILITY</div>"
    f"<div class='v'>{probability * 100:.2f}%</div>"
    "<div class='s'>Share of similar recorded accidents with a serious or fatal injury. "
    "This is not the chance that a crash will occur.</div></div>"
    "<div class='result-card'><div class='k'>CLASSIFICATION</div>"
    f"<div class='v plain'>{label}</div>"
    f"<div class='s'>Selected threshold {threshold:.3f} · {decision['name']} mode.</div></div>"
    "<div class='result-card'><div class='k'>WHAT THIS MEANS</div>"
    "<div class='s'>The probability stays the same if you only change the decision threshold. "
    f"Standard mode uses 0.175. Safety-focused mode uses 0.145 and flags more KSI cases, "
    "with more false alarms. Validation for this mode: "
    f"accuracy {decision['accuracy']}, KSI recall {decision['recall']}, "
    f"precision {decision['precision']}. ROC-AUC {validation['roc_auc']:.4f}, "
    f"PR-AUC {validation['pr_auc']:.4f}.</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

st.header("Why this score?")
try:
    render_explanation(model_frame, probability, threshold, decision, label)
except Exception as exc:
    st.error("SHAP explanation could not be calculated for this scenario: " + str(exc))

@st.fragment
def show_comparison(raw, probability, weather, surface, weathers, surfaces, checkpoint, model_frame):
    """Isolated so these controls do not remount the road simulator."""
    st.header("Compare conditions")
    x1, x2 = st.columns(2)
    with x1:
        alt_weather = st.selectbox(
            "Alternative weather",
            weathers,
            index=weathers.index(weather),
            key="alternative_weather",
        )
    with x2:
        alt_surface = st.selectbox(
            "Alternative road surface",
            surfaces,
            index=surfaces.index(surface),
            key="alternative_surface",
        )

    alternative = dict(raw)
    alternative["Weather_Conditions"] = alt_weather
    alternative["Road_Surface_Conditions"] = alt_surface
    alt_prob = predict_ksi(alternative, checkpoint)
    delta_points = (alt_prob - probability) * 100
    st.markdown(
        "<div class='compare'>"
        "<div class='result-card'><div class='k'>CURRENT SCENARIO</div>"
        f"<div class='v'>{probability * 100:.2f}%</div>"
        f"<div class='s'>{weather}<br>{surface}</div></div>"
        "<div class='result-card'><div class='k'>ALTERNATIVE SCENARIO</div>"
        f"<div class='v'>{alt_prob * 100:.2f}%</div>"
        f"<div class='s'>{alt_weather}<br>{alt_surface}</div></div>"
        "<div class='result-card'><div class='k'>DIFFERENCE</div>"
        f"<div class='v plain'>{delta_points:+.2f} pp</div>"
        "<div class='s'>Percentage points between the two model scores. "
        "This is not a measured causal effect.</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View exact model inputs"):
        shown = model_frame.iloc[0].astype(str).rename("value").to_frame()
        st.dataframe(shown, width="stretch")


show_comparison(
    raw, probability, weather, surface, weathers, surfaces, checkpoint, model_frame
)

st.caption(
    "SafeRoute-AI V3 · 35 pre-crash features · Training 2009–2013 · "
    "Validation 2014–2015. The model threshold 0.175 was selected on validation F1. "
    "Safety-focused mode uses 0.145, measured on those same validation rows. "
    "2016–2017 were not used to choose features, hyperparameters, or either threshold. "
    "Weather and road conditions are scenario inputs, not live measurements."
)

if submitted:
    st.toast("Scenario updated")
