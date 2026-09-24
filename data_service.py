"""
data_service.py
Real data layer for the NPK Monitoring Dashboard.

This file replaces the mock_* functions in streamlit_dashboardUI.py with
real reads from Firebase Realtime Database and real MLP model inference.
The UI file itself does not need to change -- only the calls that
populate st.session_state.

--------------------------------------------------------------------------
FIREBASE SCHEMA ASSUMED (adjust the paths/keys below to match yours):

/sensor_readings/
    -Nxxxxxx1: {timestamp, ec, ph, soil_humidity, soil_temp, n, p, k}
    -Nxxxxxx2: {timestamp, ec, ph, soil_humidity, soil_temp, n, p, k}
    ...

Each push under /sensor_readings is one ESP32 HTTP POST. If your ESP32
instead writes to a single fixed path (e.g. /latest_reading), use
get_latest_reading_single() instead of get_latest_reading().

/model_metrics/  (optional, written once after training/evaluation)
    N: {MAE, RMSE, R2}
    P: {MAE, RMSE, R2}
    K: {MAE, RMSE, R2}

If you don't write metrics to Firebase, just hard-code them from your
training notebook results using get_model_metrics_static() instead.
--------------------------------------------------------------------------

SETUP:
    pip install firebase-admin joblib scikit-learn pandas numpy

    1. In Firebase console -> Project Settings -> Service Accounts ->
       "Generate new private key". Save the JSON file, e.g. as
       "serviceAccountKey.json" (do NOT commit this to git / share it).
    2. Set FIREBASE_DB_URL below to your Realtime Database URL, e.g.
       "https://your-project-id-default-rtdb.asia-southeast1.firebasedatabase.app/"
    3. Put your three trained model files (and any scaler) in a
       "models/" folder next to this script.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, db

# ============================================================================
# CONFIG -- edit these to match your project
# ============================================================================
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
FIREBASE_DB_URL = "https://YOUR-PROJECT-ID-default-rtdb.asia-southeast1.firebasedatabase.app/"
READINGS_PATH = "sensor_readings"   # RTDB node the ESP32 pushes new readings to
METRICS_PATH = "model_metrics"      # optional RTDB node for pre-computed metrics

MODEL_DIR = "models"
MODEL_FILES = {
    "N": "n_model.pkl",
    "P": "p_model.pkl",
    "K": "k_model.pkl",
}
SCALER_FILE = "scaler.pkl"   # set to None if your models don't need one


# ============================================================================
# FIREBASE CONNECTION FUNCTIONS
# ============================================================================

def init_firebase():
    """
    Initialize the Firebase Admin SDK connection to your Realtime Database.
    Call this once, e.g. at the top of app.py, guarded so it only runs once
    per Streamlit session:

        if not firebase_admin._apps:
            init_firebase()
    """
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})


def get_latest_reading():
    """
    Fetch the most recent sensor reading from /sensor_readings, where each
    child was created with push() (auto-generated keys, which sort
    chronologically). Returns a dict shaped like the old mock_current_reading().
    """
    ref = db.reference(READINGS_PATH)
    latest = ref.order_by_key().limit_to_last(1).get()

    if not latest:
        return None

    _, entry = next(iter(latest.items()))
    return _normalize_reading(entry)


def get_latest_reading_single(path="latest_reading"):
    """
    Alternative to get_latest_reading(): use this if your ESP32 overwrites
    one fixed node (e.g. /latest_reading) instead of push()-ing new children.
    """
    ref = db.reference(path)
    entry = ref.get()
    if not entry:
        return None
    return _normalize_reading(entry)


def get_historical_readings(hours=24):
    """
    Fetch readings from the last `hours` hours as a DataFrame with columns:
    timestamp, N (mg/kg), P (mg/kg), K (mg/kg)  -- matches what
    streamlit_dashboardUI.py's Historical Trends page expects from
    mock_history().

    Assumes each child has a numeric/epoch "timestamp" field. If your ESP32
    stores timestamps as ISO strings instead, adjust the parsing below.
    """
    ref = db.reference(READINGS_PATH)
    cutoff = datetime.now() - timedelta(hours=hours)
    cutoff_epoch = cutoff.timestamp()

    all_readings = ref.order_by_child("timestamp").start_at(cutoff_epoch).get()
    if not all_readings:
        return pd.DataFrame(columns=["timestamp", "N (mg/kg)", "P (mg/kg)", "K (mg/kg)"])

    rows = []
    for _, entry in all_readings.items():
        norm = _normalize_reading(entry)
        rows.append({
            "timestamp": norm["timestamp"],
            "N (mg/kg)": norm["n_actual"],
            "P (mg/kg)": norm["p_actual"],
            "K (mg/kg)": norm["k_actual"],
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def get_model_metrics():
    """
    Fetch pre-computed MAE/RMSE/R2 per nutrient from Firebase, if you chose
    to write them there after training. Returns the same shape as the old
    mock_model_metrics(): {"N": {...}, "P": {...}, "K": {...}}.
    """
    ref = db.reference(METRICS_PATH)
    data = ref.get()
    if not data:
        return get_model_metrics_static()
    return data


def _normalize_reading(entry):
    """
    Internal helper: converts a raw Firebase entry into the dict shape the
    dashboard's session_state expects. Adjust the key names on the left if
    your ESP32 payload uses different field names (e.g. "N" vs "n").
    """
    ts_raw = entry.get("timestamp")
    if isinstance(ts_raw, (int, float)):
        timestamp = datetime.fromtimestamp(ts_raw)
    elif isinstance(ts_raw, str):
        timestamp = pd.to_datetime(ts_raw).to_pydatetime()
    else:
        timestamp = datetime.now()

    return {
        "timestamp": timestamp,
        "ec": float(entry.get("ec", 0)),
        "ph": float(entry.get("ph", 0)),
        "soil_humidity": float(entry.get("soil_humidity", 0)),
        "soil_temp": float(entry.get("soil_temp", 0)),
        "n_actual": float(entry.get("n", 0)),
        "p_actual": float(entry.get("p", 0)),
        "k_actual": float(entry.get("k", 0)),
    }


# ============================================================================
# MODEL INFERENCE FUNCTIONS (MLP predictions, run locally after Firebase read)
# ============================================================================

def load_mlp_models():
    """
    Load the three trained MLP regression models (one per nutrient) plus an
    optional shared feature scaler. Call once and cache the result, e.g. in
    app.py with:

        @st.cache_resource
        def get_models():
            return load_mlp_models()
    """
    models = {}
    for nutrient, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        models[nutrient] = joblib.load(path)

    scaler = None
    if SCALER_FILE:
        scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)

    return {"models": models, "scaler": scaler}


def predict_nutrients(reading, model_bundle):
    """
    Run the three MLP models on one sensor reading. Each model predicts one
    macronutrient using the other six parameters as inputs (EC, pH, soil
    humidity, soil temp, and the two other macronutrients), per the
    capstone's design.

    Returns a dict shaped like the old mock_predictions():
    {"n_pred": ..., "p_pred": ..., "k_pred": ...}
    """
    models = model_bundle["models"]
    scaler = model_bundle.get("scaler")

    base = [reading["ec"], reading["ph"], reading["soil_humidity"], reading["soil_temp"]]

    feature_sets = {
        "N": base + [reading["p_actual"], reading["k_actual"]],
        "P": base + [reading["n_actual"], reading["k_actual"]],
        "K": base + [reading["n_actual"], reading["p_actual"]],
    }

    predictions = {}
    for nutrient, features in feature_sets.items():
        X = np.array(features).reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)
        pred = models[nutrient].predict(X)[0]
        predictions[nutrient.lower() + "_pred"] = round(float(pred), 1)

    return predictions


def get_model_metrics_static():
    """
    Fallback if you don't write metrics to Firebase: hard-code the MAE,
    RMSE, and R2 you got when evaluating each model on the 80/20 test
    split, straight from your training notebook.
    """
    return {
        "N": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
        "P": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
        "K": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
    }