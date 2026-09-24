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
# MOCK DATA (replace with Firebase RTDB reads + MLP model inference later)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)

def mock_current_reading():
    return {
        "timestamp": datetime.now(),
        "ec": round(rng.uniform(0.8, 2.4), 2),
        "ph": round(rng.uniform(5.5, 7.5), 2),
        "soil_humidity": round(rng.uniform(25, 65), 1),
        "soil_temp": round(rng.uniform(22, 33), 1),
        "n_actual": round(rng.uniform(20, 120), 1),
        "p_actual": round(rng.uniform(10, 60), 1),
        "k_actual": round(rng.uniform(15, 90), 1),
    }

def mock_predictions(reading):
    noise = lambda v: round(v + rng.normal(0, v * 0.06), 1)
    return {
        "n_pred": noise(reading["n_actual"]),
        "p_pred": noise(reading["p_actual"]),
        "k_pred": noise(reading["k_actual"]),
    }

def mock_history(hours=24):
    now = datetime.now()
    times = [now - timedelta(minutes=15 * i) for i in range(hours * 4)][::-1]
    df = pd.DataFrame({
        "timestamp": times,
        "N (mg/kg)": np.clip(60 + np.cumsum(rng.normal(0, 3, len(times))), 10, 140),
        "P (mg/kg)": np.clip(35 + np.cumsum(rng.normal(0, 2, len(times))), 5, 80),
        "K (mg/kg)": np.clip(50 + np.cumsum(rng.normal(0, 2.5, len(times))), 10, 110),
    })
    return df

def mock_model_metrics():
    return {
        "N": {"MAE": 4.82, "RMSE": 6.13, "R2": 0.91},
        "P": {"MAE": 3.05, "RMSE": 4.02, "R2": 0.88},
        "K": {"MAE": 5.44, "RMSE": 6.77, "R2": 0.89},
    }

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