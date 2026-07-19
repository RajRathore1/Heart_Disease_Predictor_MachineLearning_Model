"""Shared input handling for both the single-prediction form and batch CSV
uploads: replicate the notebook's zero-sentinel fix, validate raw rows before
they ever reach one-hot encoding, and encode/scale consistently either way.
"""

import pandas as pd

from core.schema import (
    CATEGORICAL_OPTIONS,
    FASTING_BS_OPTIONS,
    HARD_BOUNDS,
    NUMERICAL_COLS,
    RAW_COLUMNS,
    TRAINING_RANGE,
    ZERO_SENTINEL_MEANS,
)


def clean_missing_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 0 in Cholesterol/RestingBP with the trained non-zero mean,
    matching the notebook's training-time preprocessing. app.py previously
    skipped this step entirely; batch CSV uploads make it reachable.
    """
    df = df.copy()
    for col, mean_value in ZERO_SENTINEL_MEANS.items():
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric.mask(numeric == 0, mean_value)
    return df


def _row_errors(row: pd.Series) -> list[str]:
    errors = []

    for col in RAW_COLUMNS:
        if col not in row or pd.isna(row[col]) or str(row[col]).strip() == "":
            errors.append(f"{col} is missing")

    for col, options in CATEGORICAL_OPTIONS.items():
        if col in row and not pd.isna(row[col]) and row[col] not in options:
            errors.append(f"{col}='{row[col]}' is not one of {options}")

    if "FastingBS" in row and not pd.isna(row["FastingBS"]):
        try:
            fbs = int(float(row["FastingBS"]))
            if fbs not in FASTING_BS_OPTIONS:
                errors.append(f"FastingBS={row['FastingBS']} must be 0 or 1")
        except (TypeError, ValueError):
            errors.append(f"FastingBS='{row['FastingBS']}' is not numeric")

    for col, (low, high) in HARD_BOUNDS.items():
        if col not in row or pd.isna(row[col]):
            continue
        try:
            value = float(row[col])
        except (TypeError, ValueError):
            errors.append(f"{col}='{row[col]}' is not numeric")
            continue
        if value < low or value > high:
            errors.append(f"{col}={value} is outside the plausible range ({low}-{high})")

    return errors


def _row_warnings(row: pd.Series) -> list[str]:
    warnings = []
    for col, (low, high) in TRAINING_RANGE.items():
        if col not in row or pd.isna(row[col]):
            continue
        try:
            value = float(row[col])
        except (TypeError, ValueError):
            continue
        if value < low or value > high:
            warnings.append(f"{col}={value} is outside the training data range ({low}-{high})")
    return warnings


def validate_raw(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw input rows into (valid_df, error_df).

    valid_df keeps the original columns plus a 'warnings' column (non-blocking
    notes, e.g. the model is extrapolating beyond its training range).
    error_df has columns ['row_number', 'errors'] for rows that fail hard
    validation (unknown category, non-numeric, or physiologically implausible
    value) - these are excluded rather than silently corrupting a prediction.

    Raises ValueError if a required column is entirely absent from df_raw -
    that's a structural problem with the whole upload, not a per-row one.
    """
    missing_columns = [col for col in RAW_COLUMNS if col not in df_raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    valid_rows = []
    error_rows = []
    for row_number, row in df_raw.iterrows():
        errors = _row_errors(row)
        if errors:
            error_rows.append({"row_number": row_number, "errors": "; ".join(errors)})
            continue
        warnings = _row_warnings(row)
        valid_row = row.to_dict()
        valid_row["warnings"] = "; ".join(warnings)
        valid_rows.append(valid_row)

    valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame(columns=[*df_raw.columns, "warnings"])
    error_df = pd.DataFrame(error_rows) if error_rows else pd.DataFrame(columns=["row_number", "errors"])
    return valid_df, error_df


def encode_input(df_valid: pd.DataFrame, columns: list[str], scaler) -> pd.DataFrame:
    """Encode already-validated rows (post clean_missing_sentinels) into the
    trained model's feature layout: one-hot encode, reindex to `columns`
    (filling any dummy the input didn't produce with 0), then scale the
    numerical columns with the trained `scaler`. Works for 1 row or N rows.
    """
    raw = df_valid[RAW_COLUMNS] if set(RAW_COLUMNS).issubset(df_valid.columns) else df_valid
    encoded = pd.get_dummies(raw)

    for col in columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[columns]

    encoded[NUMERICAL_COLS] = scaler.transform(encoded[NUMERICAL_COLS])
    return encoded
