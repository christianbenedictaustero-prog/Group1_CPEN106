"""
IoT-Based Soil Nutrient (NPK) Monitoring System Dashboard
Front-end GUI prototype (Streamlit)

This is a UI-first build: all sensor readings, predictions, and history
below are generated with mock/random data so the layout, panels, and
navigation can be reviewed before the real Firebase feed and trained
MLP models are wired in. Look for the "MOCK DATA" section to swap in
live data later.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NPK Monitoring Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0f1611; }
    .stApp { background-color: #0f1611; }

    .metric-card {
        background: linear-gradient(145deg, #17231a, #1b2a1f);
        border: 1px solid #2c3e2f;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-label {
        color: #9fb8a3;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #eafbee;
        font-size: 30px;
        font-weight: 700;
    }
    .metric-unit {
        color: #6f8a75;
        font-size: 13px;
        margin-left: 4px;
    }
    .status-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .status-online {
        background-color: #1f3d27;
        color: #6fe38a;
        border: 1px solid #2f6b3d;
    }
    .status-offline {
        background-color: #3d1f1f;
        color: #e3746f;
        border: 1px solid #6b2f2f;
    }
    .section-title {
        color: #eafbee;
        font-size: 20px;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 2px;
    }
    .section-caption {
        color: #7d947f;
        font-size: 13px;
        margin-bottom: 14px;
    }
    .npk-compare {
        background: #141c16;
        border: 1px solid #263329;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #eafbee;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LIVE DATA (Firebase RTDB + MLP inference)
# Replace the whole "MOCK DATA" block in streamlit_dashboardUI.py with this,
# and add `import data_service as ds` with the other imports at the top.
# ----------------------------------------------------------------------------
import data_service as ds


@st.cache_resource
def _backend():
    """Connect to Firebase once and load the models once."""
    try:
        ds.init_firebase()
    except Exception as e:
        st.error(f"Could not connect to Firebase: {e}")
        st.stop()
    return ds.load_mlp_models()        # None until the model files exist


_models = _backend()


def get_current_reading():
    reading = ds.get_latest_reading()
    if reading is None:
        st.warning("No readings found in the database yet.")
        st.stop()
    return reading


def get_predictions(reading):
    return ds.predict_nutrients(reading, _models)


@st.cache_data(ttl=30)
def get_history(hours=24):
    return ds.get_historical_readings(hours)


def get_metrics():
    return ds.get_model_metrics()


# Aliases so the rest of the UI keeps working unchanged.
# Rename the call sites to the get_* names later and delete these.
mock_current_reading = get_current_reading
mock_predictions = get_predictions
mock_history = get_history
mock_model_metrics = get_metrics
# ----------------------------------------------------------------------------
# SESSION STATE INIT 
"""from data_service import init_firebase, get_latest_reading, get_historical_readings, load_mlp_models, predict_nutrients, get_model_metrics
import firebase_admin

if not firebase_admin._apps:
    init_firebase()

if "reading" not in st.session_state:
    st.session_state.reading = get_latest_reading()
    st.session_state.models = load_mlp_models()
    st.session_state.predictions = predict_nutrients(st.session_state.reading, st.session_state.models)
    st.session_state.history = get_historical_readings()
    st.session_state.metrics = get_model_metrics()
    st.session_state.device_online = st.session_state.reading is not None"""
# ----------------------------------------------------------------------------
if "reading" not in st.session_state:
    st.session_state.reading = mock_current_reading()
    st.session_state.predictions = mock_predictions(st.session_state.reading)
    st.session_state.history = mock_history()
    st.session_state.metrics = mock_model_metrics()
    st.session_state.device_online = True

reading = st.session_state.reading
predictions = st.session_state.predictions
history = st.session_state.history
metrics = st.session_state.metrics

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌱 NPK Monitor")
    st.caption("IoT Soil Nutrient Monitoring + MLP Prediction")
    st.divider()

    status_html = (
        '<span class="status-pill status-online">● DEVICE ONLINE</span>'
        if st.session_state.device_online
        else '<span class="status-pill status-offline">● DEVICE OFFLINE</span>'
    )
    st.markdown(status_html, unsafe_allow_html=True)
    st.caption(f"Last reading: {reading['timestamp'].strftime('%b %d, %Y — %I:%M %p')}")

    st.divider()
    st.markdown("**View**")
    page = st.radio(
        "Navigate",
        ["Live Overview", "Prediction Accuracy", "Historical Trends"],
        label_visibility="collapsed",
    )

    st.divider()
    if st.button("🔄 Refresh reading (mock)", use_container_width=True):
        st.session_state.reading = mock_current_reading()
        st.session_state.predictions = mock_predictions(st.session_state.reading)
        st.rerun()

    st.caption("Data source: mock generator — swap for Firebase Realtime Database feed.")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    "<div class='section-title'>Soil Nutrient (NPK) Monitoring Dashboard</div>"
    "<div class='section-caption'>ESP32 + RS485 NPK sensor · MLP regression prediction · "
    "Firebase-synced (mock data shown)</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# PAGE: LIVE OVERVIEW
# ----------------------------------------------------------------------------
if page == "Live Overview":

    st.markdown("#### Live Sensor Readings")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, unit in [
        (c1, "EC", reading["ec"], "dS/m"),
        (c2, "pH", reading["ph"], ""),
        (c3, "Soil Humidity", reading["soil_humidity"], "%"),
        (c4, "Soil Temp", reading["soil_temp"], "°C"),
    ]:
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}<span class='metric-unit'>{unit}</span></div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### Predicted vs. Actual Macronutrients")
    st.caption("Actual values shown are lab/reference values used for prototype validation; "
               "in deployment only sensor inputs + MLP predictions will display.")

    n1, n2, n3 = st.columns(3)
    for col, label, actual, pred, unit in [
        (n1, "Nitrogen (N)", reading["n_actual"], predictions["n_pred"], "mg/kg"),
        (n2, "Phosphorus (P)", reading["p_actual"], predictions["p_pred"], "mg/kg"),
        (n3, "Potassium (K)", reading["k_actual"], predictions["k_pred"], "mg/kg"),
    ]:
        delta = round(pred - actual, 1)
        with col:
            st.markdown("<div class='npk-compare'>", unsafe_allow_html=True)
            st.markdown(f"**{label}**")
            sub1, sub2 = st.columns(2)
            sub1.metric("Actual", f"{actual} {unit}")
            sub2.metric("Predicted", f"{pred} {unit}", delta=f"{delta}")
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PAGE: PREDICTION ACCURACY
# ----------------------------------------------------------------------------
elif page == "Prediction Accuracy":
    st.markdown("#### Model Accuracy Panel")
    st.caption("One MLP regression model per macronutrient, evaluated on held-out test data (80/20 split).")

    for nutrient in ["N", "P", "K"]:
        m = metrics[nutrient]
        st.markdown(f"**{nutrient} Model**")
        a, b, c = st.columns(3)
        a.markdown(f"<div class='metric-card'><div class='metric-label'>MAE</div>"
                    f"<div class='metric-value'>{m['MAE']}</div></div>", unsafe_allow_html=True)
        b.markdown(f"<div class='metric-card'><div class='metric-label'>RMSE</div>"
                    f"<div class='metric-value'>{m['RMSE']}</div></div>", unsafe_allow_html=True)
        c.markdown(f"<div class='metric-card'><div class='metric-label'>R²</div>"
                    f"<div class='metric-value'>{m['R2']}</div></div>", unsafe_allow_html=True)
        st.write("")

    st.info("Metrics shown are placeholders. Replace `mock_model_metrics()` with the trained "
            "MLP evaluation results (MAE, RMSE, R²) once training is finalized.")

# ----------------------------------------------------------------------------
# PAGE: HISTORICAL TRENDS
# ----------------------------------------------------------------------------
elif page == "Historical Trends":
    st.markdown("#### Historical NPK Trends")
    range_hrs = st.select_slider("Time range", options=[6, 12, 24, 48], value=24)
    trimmed = history[history["timestamp"] >= datetime.now() - timedelta(hours=range_hrs)]

    st.line_chart(trimmed.set_index("timestamp"))

    st.write("")
    st.markdown("#### Raw Log")
    st.dataframe(
        trimmed.sort_values("timestamp", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=300,
    )

st.write("")
st.divider()
st.caption("Prototype front-end — data layer (Firebase RTDB) and MLP inference to be connected next.")