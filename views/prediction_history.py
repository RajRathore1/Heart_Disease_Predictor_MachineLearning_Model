import pandas as pd
import streamlit as st

from core import database, model_registry, ui

ui.page_header(
    "Prediction History",
    "Review past assessments, record confirmed diagnoses, and audit model retraining.",
)
ui.render_disclaimer()

artifacts = model_registry.load_artifacts()
model_names = list(artifacts["models"].keys())

with st.sidebar:
    st.markdown("**Filters**")
    model_filter = st.selectbox(
        "Prediction method", ["All", *model_names],
        format_func=lambda n: "All" if n == "All" else model_registry.display_name(n),
    )
    date_from = st.date_input("From", value=None)
    date_to = st.date_input("To", value=None)

history_df = database.fetch_history(
    limit=1000,
    model_filter=None if model_filter == "All" else model_filter,
    date_from=date_from.isoformat() if date_from else None,
    date_to=date_to.isoformat() + "T23:59:59" if date_to else None,
)

if history_df.empty:
    st.info("No predictions logged yet. Run a single or batch prediction first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total assessments", len(history_df))
col2.metric("🔴 High risk", int((history_df["risk_bucket"] == "High").sum()))
col3.metric("% High risk", f"{(history_df['risk_bucket'] == 'High').mean():.1%}")
col4.metric(
    "Outcomes confirmed",
    int(history_df["actual_outcome"].notna().sum()),
    help="Assessments where staff have recorded the real diagnosis. These feed automatic retraining.",
)

trend = history_df.copy()
trend["date"] = pd.to_datetime(trend["timestamp"]).dt.date
st.line_chart(trend.groupby("date").size(), y_label="Assessments per day")

st.subheader("Records")
st.download_button(
    "Download filtered history (CSV)", history_df.to_csv(index=False), "prediction_history.csv", "text/csv",
)

OUTCOME_LABELS = {None: "Unconfirmed", 0: "No Disease", 1: "Disease"}
LABEL_TO_OUTCOME = {v: k for k, v in OUTCOME_LABELS.items()}


def _to_label(value):
    if pd.isna(value):
        return "Unconfirmed"
    return OUTCOME_LABELS[int(value)]


st.caption(
    "Once a patient's real diagnosis is confirmed (e.g. after follow-up testing), record it below. "
    "Confirmed outcomes are what the automatic model retraining pipeline learns from — unconfirmed "
    "predictions are never used for retraining."
)

editor_df = history_df[["id", "timestamp", "model_name", "risk_bucket", "probability"]].copy()
editor_df["model_name"] = editor_df["model_name"].map(model_registry.display_name)
editor_df["Confirmed Outcome"] = history_df["actual_outcome"].apply(_to_label)

edited_df = st.data_editor(
    editor_df,
    column_config={
        "Confirmed Outcome": st.column_config.SelectboxColumn(
            options=["Unconfirmed", "No Disease", "Disease"], required=True,
        ),
    },
    disabled=["id", "timestamp", "model_name", "risk_bucket", "probability"],
    hide_index=True,
    width="stretch",
    key="history_editor",
)

if st.button("Save confirmed outcomes", type="primary"):
    changed = edited_df[edited_df["Confirmed Outcome"] != editor_df["Confirmed Outcome"]]
    for _, row in changed.iterrows():
        database.update_actual_outcome(int(row["id"]), LABEL_TO_OUTCOME[row["Confirmed Outcome"]])
    st.success(f"Updated {len(changed)} record(s).") if len(changed) else st.info("No changes to save.")
    st.rerun()

st.subheader("Automatic retraining log")
retrain_log = database.fetch_retrain_log()
if retrain_log.empty:
    st.caption("No retraining runs yet.")
else:
    st.dataframe(retrain_log, hide_index=True, width="stretch")

with st.expander("⚠️ Clear all history"):
    st.warning("This permanently deletes all logged predictions and cannot be undone.")
    confirm = st.checkbox("I understand this will permanently delete all prediction history.")
    if st.button("Delete all history", disabled=not confirm):
        database.clear_history()
        st.success("History cleared.")
        st.rerun()
