from pathlib import Path
from datetime import date, time
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

ROOT = Path(__file__).parent
st.set_page_config(
    page_title="SafeRoute-AI | Road Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_checkpoint():
    return joblib.load(ROOT / "model.pkl")

checkpoint = load_checkpoint()

if isinstance(checkpoint, dict):
    model = checkpoint.get("model")
    if model is None:
        model = checkpoint.get("classifier")
    if model is None:
        model = checkpoint.get("pipeline")
    if model is None:
        st.error("Model key not found in checkpoint.")
        st.stop()
    levels = checkpoint.get("category_levels", {})
    features = checkpoint.get("features", checkpoint.get("feature_columns", None))
    threshold = float(checkpoint.get("threshold", 0.15))
else:
    model = checkpoint
    levels = {}
    features = None
    threshold = 0.15

FEATURES = [
    "1st_Road_Class", "Day_of_Week", "Light_Conditions",
    "Road_Surface_Conditions", "Road_Type", "Speed_limit",
    "Urban_or_Rural_Area", "Weather_Conditions", "Hour",
    "Month", "Latitude", "Longitude", "Weekend",
    "Rush_Hour", "Night", "Year"
]

CATEGORIES = {
    "1st_Road_Class":
        ["A","A(M)","B","C","Motorway","Unclassified"],
    "Day_of_Week":
        ["Monday","Tuesday","Wednesday","Thursday","Friday",
         "Saturday","Sunday"],
    "Light_Conditions":
        ["Daylight","Darkness - lights lit",
         "Darkness - lights unlit","Darkness - no lighting",
         "Darkness - lighting unknown"],
    "Road_Surface_Conditions":
        ["Dry","Wet or damp","Frost or ice","Snow",
         "Flood over 3cm. deep","Data missing or out of range"],
    "Road_Type":
        ["Single carriageway","Dual carriageway","Roundabout",
         "One way street","Slip road","Unknown"],
    "Urban_or_Rural_Area":["Urban","Rural"],
    "Weather_Conditions":
        ["Fine no high winds","Fine + high winds",
         "Raining no high winds","Raining + high winds",
         "Snowing no high winds","Snowing + high winds",
         "Fog or mist","Other","Unknown",
         "Data missing or out of range"]
}

CSS = """
<style>
.stApp{
 background:
 radial-gradient(circle at 80% 0%,#123657 0%,transparent 35%),
 linear-gradient(145deg,#06101e,#081b2b 55%,#061321);
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
.hero-sub{color:#c0e7f4;max-width:520px;margin-top:16px}
.kicker{letter-spacing:3px;color:#67e8f9;font-size:12px;font-weight:800}
@media(max-width:700px){
 .hero-overlay{padding:18px}
 .hero{min-height:260px}
 .hero img{height:260px}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
import base64
hero64 = base64.b64encode((ROOT/"hero.svg").read_bytes()).decode()

st.markdown(
    '<div class="hero">'
    '<img src="data:image/svg+xml;base64,'+hero64+'">'
    '<div class="hero-overlay">'
    '<div class="kicker">✦ INTERACTIVE ROAD INTELLIGENCE</div>'
    '<div class="hero-title">SafeRoute<span style="color:#67e8f9">-AI</span></div>'
    '<div class="hero-sub">Explore road conditions, experience an animated '
    'journey and analyze historical accident injury severity with AI.</div>'
    '<div style="margin-top:18px;color:#6ce9f9">'
    '☀ DAYLIGHT &nbsp; 🌙 NIGHT &nbsp; ❄ SNOW &nbsp; 🌧 RAIN'
    '</div></div></div>',
    unsafe_allow_html=True
)

st.caption(
    "Research prototype · UK historical accident records · "
    "Not a live road-safety or accident-likelihood prediction system."
)

# ------------------------------------------------------------
# SCENARIO INPUT FORM
# ------------------------------------------------------------
st.header("01 / Create Your Road Scenario")

with st.form("scenario_form"):
    a,b,c = st.columns(3)

    with a:
        road_class=st.selectbox(
            "Road class",CATEGORIES["1st_Road_Class"],index=0
        )
        road_type=st.selectbox(
            "Road type",CATEGORIES["Road_Type"]
        )
        speed_limit=st.selectbox(
            "Speed limit (mph)",[10,20,30,40,50,60,70],index=2
        )
        area=st.selectbox(
            "Urban / Rural",CATEGORIES["Urban_or_Rural_Area"]
        )

    with b:
        weather=st.selectbox(
            "Weather",CATEGORIES["Weather_Conditions"]
        )
        surface=st.selectbox(
            "Road surface",CATEGORIES["Road_Surface_Conditions"]
        )
        lighting=st.selectbox(
            "Lighting",CATEGORIES["Light_Conditions"]
        )

    with c:
        scenario_date=st.date_input(
            "Historical scenario date",
            value=date(2013,6,15),
            min_value=date(2009,1,1),
            max_value=date(2015,12,31)
        )
        scenario_time=st.time_input(
            "Time",value=time(14,0)
        )
        latitude=st.number_input(
            "Latitude",49.9,60.7,52.239552,format="%.6f"
        )
        longitude=st.number_input(
            "Longitude",-7.6,1.8,-1.350476,format="%.6f"
        )

    submitted=st.form_submit_button(
        "🚀 UPDATE SIMULATION & RUN AI",
        type="primary",
        use_container_width=True
    )

# ------------------------------------------------------------
# MODEL INPUT AND PREDICTION
# ------------------------------------------------------------
hour=scenario_time.hour
weekend=int(scenario_date.weekday()>=5)
rush=int(hour in [7,8,9,16,17,18,19])
night=int(hour<6 or hour>=20)

record = {
    "1st_Road_Class":road_class,
    "Day_of_Week":scenario_date.strftime("%A"),
    "Light_Conditions":lighting,
    "Road_Surface_Conditions":surface,
    "Road_Type":road_type,
    "Speed_limit":float(speed_limit),
    "Urban_or_Rural_Area":area,
    "Weather_Conditions":weather,
    "Hour":float(hour),
    "Month":float(scenario_date.month),
    "Latitude":float(latitude),
    "Longitude":float(longitude),
    "Weekend":float(weekend),
    "Rush_Hour":float(rush),
    "Night":float(night),
    "Year":float(scenario_date.year)
}

X=pd.DataFrame([record],columns=FEATURES)

for col in CATEGORIES:
    allowed=levels.get(col,CATEGORIES[col])
    X[col]=pd.Categorical(X[col],categories=list(allowed))

try:
    probability=float(model.predict_proba(X)[0,1])
except Exception as e:
    st.error("Model prediction error: "+str(e))
    st.info("The checkpoint's categorical encoding may differ.")
    st.stop()

label="KSI · Serious/Fatal combined" if probability>=threshold else "Slight"

# Keep a single submitted scenario across the visual sections.
# Streamlit reruns automatically after form submission.
st.session_state["last_probability"]=probability
st.session_state["last_label"]=label

st.header("02 / Animated Road Experience")

is_snow=("Snow" in surface or "Snowing" in weather)
is_rain=("Raining" in weather)
is_wet=("Wet" in surface or "Flood" in surface)
is_ice=("Frost" in surface)
is_fog=("Fog" in weather)
is_night=("Darkness" in lighting)

config={
    "snow":is_snow,
    "rain":is_rain,
    "wet":is_wet,
    "ice":is_ice,
    "fog":is_fog,
    "night":is_night,
    "prob":probability,
    "label":label
}

html=(ROOT/"simulator.html").read_text(encoding="utf-8")
html=html.replace("__CONFIG__",json.dumps(config))

# Height includes canvas and side controls.
components.html(html,height=540,scrolling=False)

st.caption(
    "Press START DRIVE in the panel beside the road. "
    "The fictional incident mode is an illustration only."
)

# ------------------------------------------------------------
# AI RESULT
# ------------------------------------------------------------
st.header("03 / AI Severity Intelligence")

left,right=st.columns([1.15,1])

with left:
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        number={"suffix":"%","font":{"color":"#67e8f9","size":48}},
        title={"text":"Conditional KSI estimate",
               "font":{"color":"#cde7f4"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#bcdbe9"},
            "bar":{"color":"#22d3ee"},
            "bgcolor":"#10243a",
            "borderwidth":0,
            "steps":[{"range":[0,100],"color":"#1a344d"}],
            "threshold":{
                "line":{"color":"#bb86fc","width":5},
                "value":threshold*100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e7f7ff"},
        height=340,
        margin=dict(l=25,r=25,t=45,b=10)
    )
    st.plotly_chart(fig,use_container_width=True)

with right:
    st.metric("Estimated conditional KSI probability",
              f"{probability*100:.2f}%")
    st.metric("Model classification",label)
    st.write(f"Provisional threshold: **{threshold*100:.0f}%**")
    st.info(
        "The model estimates injury severity among recorded "
        "accidents. It does not estimate whether a crash will occur."
    )

st.header("04 / Compare Road Conditions")

x1,x2=st.columns(2)
with x1:
    alt_weather=st.selectbox(
        "Alternative weather",
        CATEGORIES["Weather_Conditions"],
        index=CATEGORIES["Weather_Conditions"].index(weather),
        key="alternative_weather"
    )
with x2:
    alt_surface=st.selectbox(
        "Alternative road surface",
        CATEGORIES["Road_Surface_Conditions"],
        index=CATEGORIES["Road_Surface_Conditions"].index(surface),
        key="alternative_surface"
    )

alternative=X.copy()
alternative["Weather_Conditions"]=pd.Categorical(
    [alt_weather],
    categories=X["Weather_Conditions"].cat.categories
)
alternative["Road_Surface_Conditions"]=pd.Categorical(
    [alt_surface],
    categories=X["Road_Surface_Conditions"].cat.categories
)

alt_prob=float(model.predict_proba(alternative)[0,1])

fig2=go.Figure()
fig2.add_trace(go.Bar(
    x=[probability*100,alt_prob*100],
    y=["Current","Alternative"],
    orientation="h",
    marker_color=["#22d3ee","#a78bfa"],
    text=[f"{probability*100:.2f}%",
          f"{alt_prob*100:.2f}%"],
    textposition="outside"
))
fig2.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color":"#d8efff"},
    xaxis_title="Conditional KSI estimate (%)",
    xaxis_range=[0,max(25,probability*120,alt_prob*120)],
    height=300,
    margin=dict(l=10,r=60,t=15,b=30)
)
st.plotly_chart(fig2,use_container_width=True)

with st.expander("View exact model inputs"):
    st.dataframe(X.T,use_container_width=True)

st.caption(
    "SafeRoute-AI V2 · 16 features · Historical training 2009–2013 · "
    "Validation 2014–2015. The 15% classification threshold was "
    "selected using development validation data, not independently "
    "established as a safety threshold. Weather and road conditions "
    "are scenario inputs, not live measurements."
)
