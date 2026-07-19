"""Loads the trained models, scaler, and column layout once per server
process. Cache is invalidated by core/retrain.py after a successful
automatic retrain, so a running app picks up the new model without restart.
"""

import os

import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILES = {
    "Logistic Regression": "LogisticRegression_heart.pkl",
    "KNN": "KNN_heart.pkl",
}

# Client-facing names. The technical names above stay as the internal keys
# (and in the database) so history/retraining are unaffected by display copy.
MODEL_DISPLAY = {
    "Logistic Regression": "Standard risk model",
    "KNN": "Similar-patient comparison",
}


def display_name(model_name: str) -> str:
    return MODEL_DISPLAY.get(model_name, model_name)
SCALER_FILE = "scaler.pkl"
COLUMNS_FILE = "col.pkl"


@st.cache_resource
def load_artifacts() -> dict:
    models = {
        name: joblib.load(os.path.join(BASE_DIR, filename))
        for name, filename in MODEL_FILES.items()
    }
    scaler = joblib.load(os.path.join(BASE_DIR, SCALER_FILE))
    columns = joblib.load(os.path.join(BASE_DIR, COLUMNS_FILE))
    return {"models": models, "scaler": scaler, "columns": columns}
