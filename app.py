"""SafeRoute-AI Streamlit dashboard — V3 pre-crash KSI model.

Every V3 feature is either collected in the form or derived from those
inputs with the same rules used in training. The 2016–2017 holdout is
not used here.
"""

from datetime import date, time
from pathlib import Path
import base64
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


def render_explanation(model_frame, probability, threshold, decision, label):
    """SHAP section for the current scenario. Units are the model log-odds."""
    result = explain_frame(load_shap_explainer(checkpoint), checkpoint["model"], model_frame)
    if abs(result["probability"] - float(probability)) > 1e-8:
        raise RuntimeError("The SHAP reconstruction does not match the displayed KSI probability.")

    lines = contributor_lines(result, model_frame)
    st.write(
        "These are SHAP contributions for this exact scenario on the deployed V3 model. "
        "Each value is in raw-score units. V3 is a binary LightGBM model, so that raw score "
        "is a log-odds. The probability shown above is the logistic function of the raw score. "
        "The contributions are not percentage-point changes, and they are not proven accident causes."
    )

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
        "The five features are those with the largest absolute contribution in this scenario."
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


def pick(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


CSS = """
<style>
.stApp{
 background:
 radial-gradient(circle at 80% 0%,#123657 0%,transparent 35%),
 linear-gradient(145deg,#06101e,#081b2d 55%,#061321);
 color:#e8f5ff;
}
.block-container{max-width:1450px;padding-top:1.4rem}
h1,h2,h3,p,label{color:#e8f5ff!important}
div[data-testid="stForm"]{
 background:#0d2137;
 border:1px solid #27506a;
 border-radius:18px;
 padding:20px;
}
div[data-testid="stMetric"]{
 background:#10263d;
 border:1px solid #2a5570;
 border-radius:15px;
 padding:15px;
}
div[data-testid="stAlert"]{border-radius:12px}
.stButton button[kind="primary"],div[data-testid="stFormSubmitButton"] button{
 background:linear-gradient(90deg,#08b7d1,#7737d7);
 border:0;color:white;
 font-weight:700;
}
section[data-testid="stSidebar"]{background:#09192a}
.hero{
 position:relative;
 border-radius:22px;
 border:1px solid #28708a;
 overflow:hidden;
 background:#10243a;
 min-height:310px;
}
.hero img{width:100%;height:310px;object-fit:cover}
.hero-overlay{
 position:absolute;inset:0;
 background:linear-gradient(90deg,#031121f2 0%,#031121d9 38%,transparent 75%);
 display:flex;flex-direction:column;justify-content:center;
 padding:38px;
}
.hero-title{font-size:clamp(34px,5vw,65px);font-weight:900;line-height:1.1}
.hero-sub{color:#c0e7f4;max-width:560px;margin-top:16px}
.kicker{letter-spacing:3px;color:#67e8f9;font-size:12px;font-weight:800}
@media(max-width:700px){
 .hero-overlay{padding:18px}
 .hero{min-height:260px}
 .hero img{height:260px}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

hero64 = base64.b64encode((ROOT / "hero.svg").read_bytes()).decode()
st.markdown(
    '<div class="hero">'
    '<img src="data:image/svg+xml;base64,' + hero64 + '">'
    '<div class="hero-overlay">'
    '<div class="kicker">V3 · PRE-CRASH SEVERITY</div>'
    '<div class="hero-title">SafeRoute<span style="color:#67e8f9">-AI</span></div>'
    '<div class="hero-sub">Set the road, the weather, and your own vehicle, '
    "then see the conditional chance of a serious or fatal injury if a "
    "crash were recorded.</div>"
    "</div></div>",
    unsafe_allow_html=True,
)
st.caption(
    "Research prototype · UK historical accident records · "
    "Not a live road-safety or accident-likelihood prediction system."
)

st.header("01 / Create Your Road Scenario")

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

with st.form("scenario_form"):
    a, b, c = st.columns(3)

    with a:
        road_class = st.selectbox("Road class", road_classes, index=pick(road_classes, "A"))
        second_class = st.selectbox(
            "Second road class",
            second_classes,
            index=pick(second_classes, "Unclassified"),
        )
        road_type = st.selectbox(
            "Road type", road_types, index=pick(road_types, "Single carriageway")
        )
        speed_limit = st.selectbox("Speed limit (mph)", [10, 20, 30, 40, 50, 60, 70], index=2)
        area = st.selectbox("Urban / Rural", areas, index=pick(areas, "Urban"))

    with b:
        weather = st.selectbox(
            "Weather", weathers, index=pick(weathers, "Fine no high winds")
        )
        surface = st.selectbox(
            "Road surface", surfaces, index=pick(surfaces, "Dry")
        )
        lighting = st.selectbox("Lighting", lights, index=pick(lights, "Daylight"))
        junction_control = st.selectbox(
            "Junction control",
            junction_controls,
            index=pick(junction_controls, "Not at junction or within 20 metres"),
        )
        junction_detail = st.selectbox(
            "Junction detail",
            junction_details,
            index=pick(junction_details, "Not at junction or within 20 metres"),
        )

    with c:
        scenario_date = st.date_input(
            "Scenario date",
            value=date(2013, 6, 15),
            min_value=date(2009, 1, 1),
            max_value=date(2015, 12, 31),
        )
        scenario_time = st.time_input("Time", value=time(14, 0))
        latitude = st.number_input("Latitude", 49.9, 60.7, 52.239552, format="%.6f")
        longitude = st.number_input("Longitude", -7.6, 1.8, -1.350476, format="%.6f")
        physical_label = st.selectbox("Pedestrian crossing", physical_labels, index=0)
        human_label = st.selectbox("Crossing human control", human_labels, index=0)

    st.markdown("**Your vehicle (known before travel)**")
    d, e, f = st.columns(3)
    with d:
        ego_type = st.selectbox(
            "Vehicle type", vehicle_types, index=pick(vehicle_types, "Car")
        )
        ego_age = st.number_input("Vehicle age (years)", 0.0, 50.0, 5.0, step=1.0)
    with e:
        ego_band = st.selectbox(
            "Driver age band", age_bands, index=pick(age_bands, "26 - 35")
        )
        ego_engine = st.number_input("Engine capacity (cc)", 50.0, 8000.0, 1600.0, step=50.0)
    with f:
        ego_purpose = st.selectbox(
            "Journey purpose",
            purposes,
            index=pick(purposes, "Commuting to/from work"),
        )

    submitted = st.form_submit_button(
        "UPDATE SIMULATION & RUN AI",
        type="primary",
        width="stretch",
    )

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
components.html(html, height=760, scrolling=False)
st.caption(
    "START DRIVE and FICTIONAL INCIDENT play a fictional collision for the predicted class. "
    "KSI is the stronger illustration and Slight is the lighter one. "
    "NORMAL JOURNEY has no collision. The percentage is conditional injury severity "
    "if an accident were recorded, not the chance that a crash will occur."
)

st.header("03 / AI Severity Intelligence")
left, right = st.columns([1.15, 1])

with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"color": "#67e8f9", "size": 48}},
        title={"text": "Conditional KSI estimate", "font": {"color": "#cde7f4"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#bcdbe9"},
            "bar": {"color": "#22d3ee"},
            "bgcolor": "#10243a",
            "borderwidth": 0,
            "steps": [{"range": [0, 100], "color": "#1a344d"}],
            "threshold": {"line": {"color": "#bb86fc", "width": 5}, "value": threshold * 100},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e7f7ff"},
        height=340,
        margin=dict(l=25, r=25, t=45, b=10),
    )
    st.plotly_chart(fig, width="stretch")

with right:
    st.metric("Estimated conditional KSI probability", f"{probability * 100:.2f}%")
    st.metric("Classification at the selected threshold", label)
    st.write(f"Selected decision threshold: **{threshold:.3f}** ({decision['name']})")
    st.write("The original model threshold of **0.175** is unchanged.")
    st.info(
        "The model estimates injury severity among recorded accidents. "
        "It does not estimate whether a crash will occur. "
        f"Selected mode on 2014–2015 validation: accuracy {decision['accuracy']}, "
        f"KSI recall {decision['recall']}, KSI precision {decision['precision']}. "
        f"Ranking quality is unchanged: ROC-AUC {validation['roc_auc']:.4f}, "
        f"PR-AUC {validation['pr_auc']:.4f}."
    )

st.header("Understanding the AI Prediction")
try:
    render_explanation(model_frame, probability, threshold, decision, label)
except Exception as exc:
    st.error("SHAP explanation could not be calculated for this scenario: " + str(exc))

@st.fragment
def show_comparison(raw, probability, weather, surface, weathers, surfaces, checkpoint, model_frame):
    """Isolated so these controls do not remount the road simulator."""
    st.header("04 / Compare Road Conditions")
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

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=[probability * 100, alt_prob * 100],
        y=["Current", "Alternative"],
        orientation="h",
        marker_color=["#22d3ee", "#a78bfa"],
        text=[f"{probability * 100:.2f}%", f"{alt_prob * 100:.2f}%"],
        textposition="outside",
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d8efff"},
        xaxis_title="Conditional KSI estimate (%)",
        xaxis_range=[0, max(25, probability * 120, alt_prob * 120)],
        height=300,
        margin=dict(l=10, r=60, t=15, b=30),
    )
    st.plotly_chart(fig2, width="stretch")

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
