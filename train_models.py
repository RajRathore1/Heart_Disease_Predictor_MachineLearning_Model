"""Regenerates LogisticRegression_heart.pkl, KNN_heart.pkl, scaler.pkl, and
col.pkl from heart.csv.

The original training notebook accidentally scaled the numerical columns
twice: once with a StandardScaler fit on the *entire* dataset, then again
with a second StandardScaler fit on the already-scaled train split - and
only that second scaler was ever saved. Feeding it raw (unscaled) values, as
the deployed app does, made every prediction close to meaningless in
practice (verified: ~43% held-out accuracy instead of the notebook's
reported ~86%). This script fits a correct single-pass pipeline instead and
overwrites the shipped model artifacts.

Run this by hand only if you need to regenerate the baseline artifacts from
scratch (e.g. after updating heart.csv). Day-to-day learning from confirmed
real patient outcomes happens automatically - see core/retrain.py.
"""

import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from core.preprocessing import clean_missing_sentinels, encode_input
from core.retrain import fit_new_models
from core.schema import RAW_COLUMNS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    df = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))
    df = clean_missing_sentinels(df)

    X_raw = df[RAW_COLUMNS]
    y = df["HeartDisease"].astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    lr, knn, scaler, columns = fit_new_models(X_train_raw, y_train)

    X_test_encoded = encode_input(X_test_raw, columns, scaler)
    lr_accuracy = accuracy_score(y_test, lr.predict(X_test_encoded))
    knn_accuracy = accuracy_score(y_test, knn.predict(X_test_encoded))
    print(f"Logistic Regression held-out accuracy: {lr_accuracy:.4f}")
    print(f"KNN held-out accuracy: {knn_accuracy:.4f}")

    joblib.dump(lr, os.path.join(BASE_DIR, "LogisticRegression_heart.pkl"))
    joblib.dump(knn, os.path.join(BASE_DIR, "KNN_heart.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    joblib.dump(columns, os.path.join(BASE_DIR, "col.pkl"))
    print("Saved LogisticRegression_heart.pkl, KNN_heart.pkl, scaler.pkl, col.pkl")


if __name__ == "__main__":
    main()
