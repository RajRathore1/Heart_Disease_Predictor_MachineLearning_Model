import pandas as pd
import streamlit as st

from core import database, model_registry, ui

ACCENT = "#0e7490"  # single-hue charts: identity lives on the axis labels, not color

ui.page_header(
    "Clinic Insights",
    "Aggregate analytics across every assessment your clinic has run.",
)
ui.render_disclaimer()

history_df = database.fetch_history(limit=100000)

if history_df.empty:
    st.info("No data yet — insights appear here once assessments have been run.")
    st.stop()

confirmed = history_df.dropna(subset=["actual_outcome"]).copy()

# ---- KPI row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total assessments", len(history_df))
col2.metric("High-risk rate", f"{(history_df['risk_bucket'] == 'High').mean():.1%}")
col3.metric("Outcomes confirmed", len(confirmed))
if len(confirmed):
    accuracy = (confirmed["prediction"] == confirmed["actual_outcome"]).mean()
    col4.metric(
        "Verified accuracy", f"{accuracy:.1%}",
        help="How often the model's prediction matched the real confirmed diagnosis "
             "for this clinic's own patients.",
    )
else:
    col4.metric("Verified accuracy", "—", help="Confirm outcomes on the History page to measure this.")

# ---- Risk distribution tiles ----
st.markdown("#### Risk distribution")
bucket_counts = history_df["risk_bucket"].value_counts()
col1, col2, col3 = st.columns(3)
col1.metric("🟢 Low risk", int(bucket_counts.get("Low", 0)))
col2.metric("🟡 Moderate risk", int(bucket_counts.get("Moderate", 0)))
col3.metric("🔴 High risk", int(bucket_counts.get("High", 0)))

# ---- Charts ----
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Assessments over time")
    trend = history_df.copy()
    trend["date"] = pd.to_datetime(trend["timestamp"]).dt.date
    st.line_chart(trend.groupby("date").size(), color=ACCENT, y_label="Assessments per day")

with chart_col2:
    st.markdown("#### High-risk rate by age band")
    banded = history_df.copy()
    banded["age_band"] = pd.cut(
        banded["age"].astype(float),
        bins=[0, 39, 49, 59, 69, 200],
        labels=["18–39", "40–49", "50–59", "60–69", "70+"],
    )
    by_band = (
        banded.groupby("age_band", observed=True)["risk_bucket"]
        .apply(lambda s: (s == "High").mean())
    )
    st.bar_chart(by_band, color=ACCENT, y_label="Share of assessments flagged High")

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.markdown("#### Prediction method usage")
    method_counts = history_df["model_name"].map(model_registry.display_name).value_counts()
    st.bar_chart(method_counts, color=ACCENT, y_label="Assessments")

with chart_col4:
    st.markdown("#### Assessment source")
    source_counts = history_df["source"].map({"single": "Individual", "batch": "Batch upload"}).value_counts()
    st.bar_chart(source_counts, color=ACCENT, y_label="Assessments")

# ---- Real-world performance on confirmed outcomes ----
st.markdown("#### Real-world model performance")
if confirmed.empty:
    st.caption(
        "No confirmed outcomes yet. Record real diagnoses on the Prediction History page — "
        "this section then shows how the model actually performs on your clinic's patients, "
        "not just on its original training data."
    )
else:
    pred = confirmed["prediction"].astype(int)
    actual = confirmed["actual_outcome"].astype(int)
    tp = int(((pred == 1) & (actual == 1)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Correct predictions", tp + tn)
    col2.metric("Missed cases (predicted low, was disease)", fn,
                help="The clinically costliest error type.")
    col3.metric("False alarms (predicted disease, was healthy)", fp)

    detail_col1, detail_col2 = st.columns(2)
    if tp + fn:
        detail_col1.metric("Sensitivity", f"{tp / (tp + fn):.1%}",
                            help="Of patients who really had heart disease, how many the model caught.")
    if tn + fp:
        detail_col2.metric("Specificity", f"{tn / (tn + fp):.1%}",
                            help="Of healthy patients, how many the model correctly cleared.")

    st.caption(
        f"Based on {len(confirmed)} assessment(s) with a staff-confirmed diagnosis. "
        "These same records feed the automatic retraining pipeline, so the model keeps "
        "adapting to your clinic's real patient population."
    )
