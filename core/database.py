"""Prediction logging and history, backed by SQLAlchemy so the same code
runs against two backends:

- Default: a local SQLite file (data/predictions.db, gitignored). Zero setup,
  but on Streamlit Community Cloud the disk is wiped on redeploy/restart.
- Hosted Postgres (e.g. a free Supabase/Neon project): set [database] url in
  .streamlit/secrets.toml and history/outcomes/retrain data become permanent.

Timestamps are stored as ISO8601 UTC strings in both backends, so date-range
filters are plain lexicographic comparisons everywhere.
"""

import os
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from sqlalchemy import (
    Column, Float, Index, Integer, MetaData, Table, Text, create_engine, func, select,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "predictions.db")

metadata = MetaData()

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Text, nullable=False),
    Column("source", Text, nullable=False),          # 'single' | 'batch'
    Column("batch_id", Text),                         # shared uuid4 per CSV upload
    Column("model_name", Text, nullable=False),
    Column("age", Integer), Column("sex", Text), Column("chest_pain_type", Text),
    Column("resting_bp", Float), Column("cholesterol", Float), Column("fasting_bs", Integer),
    Column("resting_ecg", Text), Column("max_hr", Integer), Column("exercise_angina", Text),
    Column("oldpeak", Float), Column("st_slope", Text),
    Column("prediction", Integer, nullable=False),
    Column("probability", Float, nullable=False),
    Column("risk_bucket", Text, nullable=False),
    Column("actual_outcome", Integer),                # NULL until staff confirm
    Index("idx_predictions_timestamp", "timestamp"),
    Index("idx_predictions_batch_id", "batch_id"),
)

retrain_log = Table(
    "retrain_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ran_at", Text, nullable=False),
    Column("records_used", Integer, nullable=False),
    Column("old_accuracy", Float), Column("new_accuracy", Float),
    Column("applied", Integer, nullable=False),
    Column("reason", Text),
)

_PATIENT_FIELDS = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol", "fasting_bs",
    "resting_ecg", "max_hr", "exercise_angina", "oldpeak", "st_slope",
]

_RAW_TO_DB = {
    "Age": "age", "Sex": "sex", "ChestPainType": "chest_pain_type",
    "RestingBP": "resting_bp", "Cholesterol": "cholesterol", "FastingBS": "fasting_bs",
    "RestingECG": "resting_ecg", "MaxHR": "max_hr", "ExerciseAngina": "exercise_angina",
    "Oldpeak": "oldpeak", "ST_Slope": "st_slope",
}

# Coerce to plain Python types: batch rows arrive as numpy scalars from
# pandas, which Postgres drivers reject.
_FIELD_CASTS = {
    "age": lambda v: int(float(v)), "resting_bp": float, "cholesterol": float,
    "fasting_bs": lambda v: int(float(v)), "max_hr": lambda v: int(float(v)),
    "oldpeak": float,
}

_engine = None


def _database_url() -> str | None:
    try:
        cfg = st.secrets.get("database")
        url = str((cfg or {}).get("url", "")).strip()
        # Template placeholders count as "not configured"
        if not url or "USER:PASSWORD" in url or "your-" in url.lower():
            return None
        return url
    except Exception:
        return None


def get_engine():
    global _engine
    if _engine is None:
        url = _database_url()
        if url:
            _engine = create_engine(url, pool_pre_ping=True)
        else:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            _engine = create_engine(f"sqlite:///{DB_PATH}")
    return _engine


def init_db() -> None:
    metadata.create_all(get_engine())


def _patient_to_db_row(patient: dict) -> dict:
    """Accepts either raw-schema keys (Age, Sex, ...) or db-shaped keys
    (age, sex, ...) and normalizes names and value types."""
    row = {}
    for raw_key, db_key in _RAW_TO_DB.items():
        value = patient.get(raw_key, patient.get(db_key))
        if value is None:
            continue
        cast = _FIELD_CASTS.get(db_key, str)
        row[db_key] = cast(value)
    return row


def log_prediction(patient: dict, model_name: str, prediction: int, probability: float,
                    risk_bucket: str, source: str = "single", batch_id: str | None = None) -> int:
    values = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source, "batch_id": batch_id, "model_name": model_name,
        **_patient_to_db_row(patient),
        "prediction": int(prediction), "probability": float(probability),
        "risk_bucket": risk_bucket,
    }
    with get_engine().begin() as conn:
        result = conn.execute(predictions.insert().values(**values))
        return int(result.inserted_primary_key[0])


def log_batch(rows: list[dict]) -> str:
    """rows: list of dicts each with patient fields + model_name, prediction,
    probability, risk_bucket. Shares one batch_id across the whole upload."""
    batch_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    values = [
        {
            "timestamp": timestamp, "source": "batch", "batch_id": batch_id,
            "model_name": r["model_name"], **_patient_to_db_row(r),
            "prediction": int(r["prediction"]), "probability": float(r["probability"]),
            "risk_bucket": r["risk_bucket"],
        }
        for r in rows
    ]
    with get_engine().begin() as conn:
        conn.execute(predictions.insert(), values)
    return batch_id


def fetch_history(limit: int = 500, model_filter: str | None = None,
                   date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    stmt = select(predictions)
    if model_filter:
        stmt = stmt.where(predictions.c.model_name == model_filter)
    if date_from:
        stmt = stmt.where(predictions.c.timestamp >= date_from)
    if date_to:
        stmt = stmt.where(predictions.c.timestamp <= date_to)
    stmt = stmt.order_by(predictions.c.timestamp.desc()).limit(limit)
    with get_engine().connect() as conn:
        return pd.read_sql(stmt, conn)


def fetch_unconfirmed(limit: int = 200) -> pd.DataFrame:
    stmt = (
        select(predictions)
        .where(predictions.c.actual_outcome.is_(None))
        .order_by(predictions.c.timestamp.desc())
        .limit(limit)
    )
    with get_engine().connect() as conn:
        return pd.read_sql(stmt, conn)


def update_actual_outcome(record_id: int, outcome: int | None) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            predictions.update().where(predictions.c.id == record_id).values(actual_outcome=outcome)
        )


def clear_history() -> None:
    with get_engine().begin() as conn:
        conn.execute(predictions.delete())


def fetch_confirmed_for_training() -> pd.DataFrame:
    """Rows with a confirmed ground-truth outcome, in the original raw
    schema, ready to be concatenated with heart.csv for retraining."""
    stmt = select(predictions).where(predictions.c.actual_outcome.isnot(None))
    with get_engine().connect() as conn:
        df = pd.read_sql(stmt, conn)
    if df.empty:
        return pd.DataFrame(columns=[*_RAW_TO_DB.keys(), "HeartDisease"])
    renamed = df.rename(columns={db_key: raw_key for raw_key, db_key in _RAW_TO_DB.items()})
    renamed["HeartDisease"] = renamed["actual_outcome"]
    return renamed[[*_RAW_TO_DB.keys(), "HeartDisease"]]


def count_new_confirmed_since(since_iso: str | None) -> int:
    stmt = select(func.count()).select_from(predictions).where(
        predictions.c.actual_outcome.isnot(None)
    )
    if since_iso is not None:
        stmt = stmt.where(predictions.c.timestamp > since_iso)
    with get_engine().connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def log_retrain_run(records_used: int, old_accuracy: float | None, new_accuracy: float | None,
                     applied: bool, reason: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(retrain_log.insert().values(
            ran_at=datetime.now(timezone.utc).isoformat(),
            records_used=int(records_used),
            old_accuracy=None if old_accuracy is None else float(old_accuracy),
            new_accuracy=None if new_accuracy is None else float(new_accuracy),
            applied=int(applied), reason=reason,
        ))


def fetch_retrain_log(limit: int = 50) -> pd.DataFrame:
    stmt = select(retrain_log).order_by(retrain_log.c.ran_at.desc()).limit(limit)
    with get_engine().connect() as conn:
        return pd.read_sql(stmt, conn)


def latest_retrain_run() -> dict | None:
    df = fetch_retrain_log(limit=1)
    if df.empty:
        return None
    return df.iloc[0].to_dict()
