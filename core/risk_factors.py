"""Explains a single prediction: Logistic Regression coefficients give a
genuine linear contribution per feature; KNN has no such thing, but its
predict_proba (with the model's default uniform weighting) already *is* the
neighbor vote fraction, so it gets an analogous "N of K similar patients"
explanation instead.
"""

import math

import pandas as pd

FRIENDLY_LABELS = {
    "Age": "Age",
    "RestingBP": "Resting blood pressure",
    "Cholesterol": "Cholesterol",
    "FastingBS": "Fasting blood sugar > 120 mg/dl",
    "MaxHR": "Max heart rate achieved",
    "Oldpeak": "ST depression (Oldpeak)",
    "Sex_M": "Male sex",
    "ChestPainType_ATA": "Atypical angina chest pain",
    "ChestPainType_NAP": "Non-anginal chest pain",
    "ChestPainType_TA": "Typical angina chest pain",
    "RestingECG_Normal": "Normal resting ECG",
    "RestingECG_ST": "ST-T wave abnormality on ECG",
    "ExerciseAngina_Y": "Exercise-induced angina",
    "ST_Slope_Flat": "Flat ST slope",
    "ST_Slope_Up": "Upsloping ST slope",
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def lr_contributions(encoded_row: pd.Series, lr_model, columns: list[str], top_n: int = 5) -> pd.DataFrame:
    """Per-feature contribution = coefficient * scaled value, expressed
    relative to heart-disease risk (class 1) regardless of the predicted
    class - "increases risk" / "decreases risk" is the same statement whether
    the model ultimately predicted disease or not.
    """
    coefs = lr_model.coef_[0]
    values = encoded_row[columns].to_numpy(dtype=float)
    contributions = coefs * values

    df = pd.DataFrame({
        "feature": columns,
        "friendly_label": [FRIENDLY_LABELS.get(c, c) for c in columns],
        "contribution": contributions,
    })
    df["direction"] = df["contribution"].apply(lambda c: "increases risk" if c > 0 else "decreases risk")
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
    return df.head(top_n).reset_index(drop=True)


def knn_neighbor_summary(encoded_row: pd.DataFrame, knn_model, predicted_class: int) -> dict:
    """KNN's predict_proba with weights='uniform' (the trained model's
    default) is exactly the fraction of the k nearest neighbors in each
    class - derive the vote count directly rather than reloading data.
    """
    k = knn_model.n_neighbors
    proba = knn_model.predict_proba(encoded_row)[0]
    n_matching = round(proba[predicted_class] * k)
    label = "Heart Disease" if predicted_class == 1 else "No Heart Disease"
    return {"n_neighbors": k, "n_matching": n_matching, "label": label}
