"""Single source of truth for the app's data schema: raw columns, valid
categories, numeric bounds, and risk/retraining thresholds. Everything else
(forms, batch validation, database columns) should import from here rather
than hardcoding these values again.
"""

RAW_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

NUMERICAL_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

CATEGORICAL_OPTIONS = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ATA", "NAP", "TA", "ASY"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["Y", "N"],
    "ST_Slope": ["Up", "Flat", "Down"],
}

FASTING_BS_OPTIONS = [0, 1]

# Hard-reject bounds: outside these, a value is not physiologically plausible
# or is clearly a data entry error.
HARD_BOUNDS = {
    "Age": (18, 100),
    "RestingBP": (0, 250),
    "Cholesterol": (0, 700),
    "MaxHR": (60, 220),
    "Oldpeak": (-3.0, 7.0),
}

# The range actually observed in the training data. Values inside HARD_BOUNDS
# but outside this range are still predicted on, just flagged as extrapolation.
TRAINING_RANGE = {
    "Age": (28, 77),
    "RestingBP": (0, 200),
    "Cholesterol": (0, 603),
    "MaxHR": (60, 202),
    "Oldpeak": (-2.6, 6.2),
}

# heart.csv uses 0 as a "no reading" sentinel for these two columns. Training
# replaced 0 with the non-zero column mean before fitting the scaler; verified
# directly against heart.csv (172 zero-Cholesterol rows, 1 zero-RestingBP row).
ZERO_SENTINEL_MEANS = {"Cholesterol": 244.64, "RestingBP": 132.54}

# Probability -> risk tier, symmetric around the 0.5 decision boundary so
# "Moderate" reads as "the model is least confident here".
RISK_BUCKETS = [(0.35, "Low"), (0.65, "Moderate"), (1.01, "High")]

# Automatic retraining cadence (core/retrain.py). Either threshold being met
# triggers a retraining attempt.
RETRAIN_MIN_NEW_RECORDS = 20
RETRAIN_MIN_DAYS_ELAPSED = 30

# Retrained model must not score worse than this many accuracy points below
# the currently-live model to be promoted (core/retrain.py guardrail).
RETRAIN_ACCURACY_TOLERANCE = 0.01


def risk_bucket_for(probability: float) -> str:
    for threshold, label in RISK_BUCKETS:
        if probability < threshold:
            return label
    return RISK_BUCKETS[-1][1]
