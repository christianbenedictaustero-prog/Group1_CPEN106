"""
data_service.py
Real data layer for the NPK Monitoring Dashboard.

Reads live sensor data from Firebase Realtime Database and runs the three
MLP models. It replaces the mock_* functions in streamlit_dashboardUI.py.

--------------------------------------------------------------------------
FIREBASE SCHEMA (matches the Firebase setup guide):

/latest/{device_id}            newest reading (overwritten by the ESP32)
/readings/{device_id}/{key}    full history, one child per reading

Each reading has:
    timestamp (Unix seconds), date, time,
    electrical_conductivity_dS_m, soil_pH, soil_humidity_pct,
    soil_temperature_C, nitrogen_mg_kg, phosphorus_mg_kg,
    potassium_mg_kg, valid (True when soil moisture >= ~25 %)

/model_metrics/{N|P|K}         optional: {MAE, RMSE, R2} written after training
--------------------------------------------------------------------------

SETUP:
    pip install firebase-admin joblib scikit-learn pandas numpy

    - Keep serviceAccountKey.json OUT of git (it is in .gitignore).
    - Locally the key is read from SERVICE_ACCOUNT_PATH, or from the
      FIREBASE_KEY_PATH environment variable if you keep it elsewhere.
    - On a deployed Streamlit app, put the key in the app's secrets under a
      [firebase] table instead of a file.
    - Put the trained models in a "models/" folder next to this file.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, db

# ============================================================================
# CONFIG
# ============================================================================
DEVICE_ID = "device01"
FIREBASE_DB_URL = "https://soil-monitoring-system-db-default-rtdb.firebaseio.com"
SERVICE_ACCOUNT_PATH = os.environ.get("FIREBASE_KEY_PATH", "serviceAccountKey.json")

LATEST_PATH = f"latest/{DEVICE_ID}"
READINGS_PATH = f"readings/{DEVICE_ID}"
METRICS_PATH = "model_metrics"

MIN_VALID_MOISTURE = 25.0   # % soil moisture below which NPK readings are unreliable

MODEL_DIR = "models"
MODEL_FILES = {"N": "n_model.pkl", "P": "p_model.pkl", "K": "k_model.pkl"}
# Each model has different input columns, so each needs its OWN scaler.
# If a scaler file does not exist, that model gets unscaled input.
SCALER_FILES = {"N": "n_scaler.pkl", "P": "p_scaler.pkl", "K": "k_scaler.pkl"}

NAN_PREDICTIONS = {"n_pred": float("nan"), "p_pred": float("nan"), "k_pred": float("nan")}


# ============================================================================
# FIREBASE CONNECTION
# ============================================================================

def _load_credentials():
    """Use Streamlit secrets ([firebase] table) if present, else the key file."""
    try:
        import streamlit as st
        if "firebase" in st.secrets:
            return credentials.Certificate(dict(st.secrets["firebase"]))
    except Exception:
        pass
    return credentials.Certificate(SERVICE_ACCOUNT_PATH)


def init_firebase():
    """Connect to Firebase once. Safe to call more than once."""
    if firebase_admin._apps:
        return
    firebase_admin.initialize_app(_load_credentials(), {"databaseURL": FIREBASE_DB_URL})


# ============================================================================
# READING DATA
# ============================================================================

def _num(value):
    """Convert to float, or NaN if the value is missing or not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normalize_reading(entry):
    """Convert a raw Firebase record into the dict the dashboard expects."""
    ts_raw = entry.get("timestamp")
    if isinstance(ts_raw, (int, float)):
        timestamp = datetime.fromtimestamp(ts_raw)
    elif isinstance(ts_raw, str):
        timestamp = pd.to_datetime(ts_raw).to_pydatetime()
    else:
        timestamp = datetime.now()

    humidity = _num(entry.get("soil_humidity_pct"))
    valid = entry.get("valid")
    if valid is None:                       # older records without the flag
        valid = humidity >= MIN_VALID_MOISTURE

    return {
        "timestamp": timestamp,
        "ec": _num(entry.get("electrical_conductivity_dS_m")),
        "ph": _num(entry.get("soil_pH")),
        "soil_humidity": humidity,
        "soil_temp": _num(entry.get("soil_temperature_C")),
        "n_actual": _num(entry.get("nitrogen_mg_kg")),
        "p_actual": _num(entry.get("phosphorus_mg_kg")),
        "k_actual": _num(entry.get("potassium_mg_kg")),
        "valid": bool(valid),
    }


def get_latest_reading():
    """Newest reading as a dict, or None if the database has no readings."""
    entry = db.reference(LATEST_PATH).get()
    if not entry:                           # fall back to the newest history record
        newest = (db.reference(READINGS_PATH)
                  .order_by_child("timestamp").limit_to_last(1).get())
        if not newest:
            return None
        entry = next(iter(newest.values()))
    return _normalize_reading(entry)


def get_historical_readings(hours=24, include_invalid=False):
    """
    Readings from the last `hours` hours as a DataFrame with columns
    timestamp, N (mg/kg), P (mg/kg), K (mg/kg). Readings flagged
    valid = False (low moisture) are left out unless include_invalid=True.
    """
    columns = ["timestamp", "N (mg/kg)", "P (mg/kg)", "K (mg/kg)"]
    cutoff_epoch = (datetime.now() - timedelta(hours=hours)).timestamp()

    data = (db.reference(READINGS_PATH)
            .order_by_child("timestamp").start_at(cutoff_epoch).get())
    if not data:
        return pd.DataFrame(columns=columns)

    rows = []
    for entry in data.values():
        r = _normalize_reading(entry)
        if not include_invalid and not r["valid"]:
            continue
        rows.append({
            "timestamp": r["timestamp"],
            "N (mg/kg)": r["n_actual"],
            "P (mg/kg)": r["p_actual"],
            "K (mg/kg)": r["k_actual"],
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def get_model_metrics():
    """MAE / RMSE / R2 per nutrient from Firebase, or the static fallback."""
    data = db.reference(METRICS_PATH).get()
    return data if data else get_model_metrics_static()


def get_model_metrics_static():
    """
    TODO: replace the NaN values with the MAE, RMSE and R2 from your
    training notebook (80/20 test split). NaN means "not filled in yet".
    """
    nan = float("nan")
    return {
        "N": {"MAE": nan, "RMSE": nan, "R2": nan},
        "P": {"MAE": nan, "RMSE": nan, "R2": nan},
        "K": {"MAE": nan, "RMSE": nan, "R2": nan},
    }


# ============================================================================
# MLP INFERENCE
# ============================================================================

def load_mlp_models():
    """
    Load the three trained models (and optional scalers) from MODEL_DIR.
    Returns None if the model files do not exist yet, so the dashboard can
    still run with live sensor data before the models are trained.
    """
    models, scalers = {}, {}
    for nutrient, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            return None
        models[nutrient] = joblib.load(path)

        scaler_path = os.path.join(MODEL_DIR, SCALER_FILES[nutrient])
        scalers[nutrient] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return {"models": models, "scalers": scalers}


def predict_nutrients(reading, model_bundle):
    """
    Predict N, P and K for one reading. Each model uses the other six
    parameters as inputs. Returns NaN predictions when the models are not
    loaded, the reading is flagged invalid (soil moisture below the
    threshold), or any input is missing.

    IMPORTANT: the order of the input columns below must match the column
    order used when each model was trained. A different order will give
    wrong predictions without any error.
    """
    if model_bundle is None or not reading.get("valid", True):
        return dict(NAN_PREDICTIONS)

    base = [reading["ec"], reading["ph"], reading["soil_humidity"], reading["soil_temp"]]
    feature_sets = {
        "N": base + [reading["p_actual"], reading["k_actual"]],
        "P": base + [reading["n_actual"], reading["k_actual"]],
        "K": base + [reading["n_actual"], reading["p_actual"]],
    }

    predictions = {}
    for nutrient, features in feature_sets.items():
        if np.isnan(features).any():
            return dict(NAN_PREDICTIONS)
        X = np.array(features, dtype=float).reshape(1, -1)
        scaler = model_bundle["scalers"].get(nutrient)
        if scaler is not None:
            X = scaler.transform(X)
        pred = model_bundle["models"][nutrient].predict(X)[0]
        predictions[nutrient.lower() + "_pred"] = round(float(pred), 1)
    return predictions