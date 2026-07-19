"""Automatic scheduled retraining on confirmed real-world outcomes.

Trains only on rows where staff have entered a confirmed diagnosis (see
core/database.fetch_confirmed_for_training) combined with the original
heart.csv - never on the model's own unconfirmed predictions, which would
create a feedback loop reinforcing its existing mistakes.

There is no human-review checkpoint before a retrained model goes live (per
the client's explicit choice), so a retrained model is only promoted if it
does not score worse than the currently-live model on a held-out split
evaluated fairly (the old model is re-scored on the SAME new split, not its
original historical test accuracy). Every attempt is logged to retrain_log
for audit purposes, whether or not it was applied.
"""

import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from core import database, model_registry
from core.preprocessing import clean_missing_sentinels, encode_input
from core.schema import (
    NUMERICAL_COLS,
    RAW_COLUMNS,
    RETRAIN_ACCURACY_TOLERANCE,
    RETRAIN_MIN_DAYS_ELAPSED,
    RETRAIN_MIN_NEW_RECORDS,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEART_CSV = os.path.join(BASE_DIR, "heart.csv")
BACKUP_DIR = os.path.join(BASE_DIR, "models_backup")


def should_retrain() -> bool:
    last_run = database.latest_retrain_run()
    since = last_run["ran_at"] if last_run else None
    new_count = database.count_new_confirmed_since(since)

    if new_count == 0:
        return False
    if new_count >= RETRAIN_MIN_NEW_RECORDS:
        return True
    if last_run is None:
        return False  # no baseline timestamp yet for the days-elapsed check

    days_elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_run["ran_at"])).days
    return days_elapsed >= RETRAIN_MIN_DAYS_ELAPSED


def _load_combined_training_data() -> pd.DataFrame:
    base_df = pd.read_csv(HEART_CSV)[[*RAW_COLUMNS, "HeartDisease"]]
    confirmed_df = database.fetch_confirmed_for_training()
    combined = pd.concat([base_df, confirmed_df], ignore_index=True)
    return clean_missing_sentinels(combined)


def fit_new_models(X_train_raw: pd.DataFrame, y_train: pd.Series):
    """Correct single-pass preprocessing: one-hot encode with drop_first,
    fit ONE StandardScaler on the raw training numerical columns. (The
    original notebook accidentally scaled numerical columns twice - once
    globally before the split, then again on the already-scaled train split
    - which made the shipped scaler.pkl meaningless against raw input. Fixed
    here and in train_models.py, which regenerated the shipped artifacts.)
    """
    X_train_encoded = pd.get_dummies(X_train_raw, drop_first=True)
    new_columns = X_train_encoded.columns.tolist()

    new_scaler = StandardScaler()
    X_train_encoded[NUMERICAL_COLS] = new_scaler.fit_transform(X_train_encoded[NUMERICAL_COLS])

    new_lr = LogisticRegression().fit(X_train_encoded, y_train)
    new_knn = KNeighborsClassifier().fit(X_train_encoded, y_train)
    return new_lr, new_knn, new_scaler, new_columns


def _backup_current_artifacts() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = os.path.join(BACKUP_DIR, stamp)
    os.makedirs(backup_path, exist_ok=True)
    filenames = [*model_registry.MODEL_FILES.values(), model_registry.SCALER_FILE, model_registry.COLUMNS_FILE]
    for filename in filenames:
        src = os.path.join(BASE_DIR, filename)
        if os.path.exists(src):
            with open(src, "rb") as f_src, open(os.path.join(backup_path, filename), "wb") as f_dst:
                f_dst.write(f_src.read())


def run_retraining() -> dict:
    artifacts = model_registry.load_artifacts()
    old_model = artifacts["models"]["Logistic Regression"]
    old_scaler = artifacts["scaler"]
    old_columns = artifacts["columns"]

    combined = _load_combined_training_data()
    records_used = len(combined)

    X_raw = combined[RAW_COLUMNS]
    y = combined["HeartDisease"].astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    # Fair comparison: score the CURRENTLY LIVE model on this same new
    # held-out split, not its original (different) historical test accuracy.
    X_test_old = encode_input(X_test_raw, old_columns, old_scaler)
    old_accuracy = accuracy_score(y_test, old_model.predict(X_test_old))

    new_lr, new_knn, new_scaler, new_columns = fit_new_models(X_train_raw, y_train)

    X_test_new = encode_input(X_test_raw, new_columns, new_scaler)
    new_accuracy = accuracy_score(y_test, new_lr.predict(X_test_new))

    applied = new_accuracy >= (old_accuracy - RETRAIN_ACCURACY_TOLERANCE)
    reason = "applied" if applied else (
        f"regressed ({new_accuracy:.4f} < {old_accuracy:.4f} - tolerance {RETRAIN_ACCURACY_TOLERANCE}), "
        "kept previous model"
    )

    if applied:
        _backup_current_artifacts()
        joblib.dump(new_lr, os.path.join(BASE_DIR, model_registry.MODEL_FILES["Logistic Regression"]))
        joblib.dump(new_knn, os.path.join(BASE_DIR, model_registry.MODEL_FILES["KNN"]))
        joblib.dump(new_scaler, os.path.join(BASE_DIR, model_registry.SCALER_FILE))
        joblib.dump(new_columns, os.path.join(BASE_DIR, model_registry.COLUMNS_FILE))
        model_registry.load_artifacts.clear()

    database.log_retrain_run(records_used, old_accuracy, new_accuracy, applied, reason)

    return {
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "records_used": records_used,
        "applied": applied,
        "reason": reason,
    }
