import pandas as pd
import streamlit as st

from core import database, model_registry, ui
from core.preprocessing import clean_missing_sentinels, encode_input, validate_raw
from core.schema import RAW_COLUMNS, risk_bucket_for

ui.page_header(
    "Batch Prediction",
    "Upload a CSV of multiple patients and assess them all at once.",
)
ui.render_disclaimer()

with st.expander("📋 Required CSV format"):
    st.markdown(f"The file must contain these columns: `{'`, `'.join(RAW_COLUMNS)}`")
    st.dataframe(pd.DataFrame([{
        "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 239,
        "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 130, "ExerciseAngina": "Y",
        "Oldpeak": 1.2, "ST_Slope": "Flat",
    }]), hide_index=True, width="stretch")

artifacts = model_registry.load_artifacts()
model_names = list(artifacts["models"].keys())

with st.sidebar:
    st.markdown("**Settings**")
    selected_model = st.selectbox(
        "Prediction method", model_names, key="selected_model",
        format_func=lambda n: model_registry.display_name(n)
        + (" (recommended)" if n == "Logistic Regression" else ""),
    )
    compare_both = st.checkbox("Compare both methods", value=False)

uploaded_file = st.file_uploader("Upload patient data (CSV)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    try:
        cleaned_df = clean_missing_sentinels(raw_df)
        valid_df, error_df = validate_raw(cleaned_df)
    except ValueError as exc:
        st.error(f"This file can't be processed: {exc}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows uploaded", len(raw_df))
    col2.metric("Valid rows", len(valid_df))
    col3.metric("Excluded rows", len(error_df))

    if not error_df.empty:
        with st.expander(f"⚠️ {len(error_df)} row(s) were excluded due to invalid data", expanded=True):
            st.dataframe(error_df, hide_index=True, width="stretch")
            st.download_button(
                "Download invalid rows (CSV)",
                error_df.to_csv(index=False),
                "invalid_rows.csv",
                "text/csv",
            )

    if valid_df.empty:
        st.warning("No valid rows to predict on.")
        st.stop()

    if len(valid_df) > 5000:
        st.warning(
            f"{len(valid_df)} rows uploaded — large batches can take a while to process, "
            "especially with the similar-patient comparison method."
        )

    extrapolating = valid_df["warnings"].astype(bool).sum()
    if extrapolating:
        st.info(
            f"{extrapolating} row(s) have values outside the training data's range — "
            "the model is extrapolating for those."
        )

    models_to_run = model_names if compare_both else [selected_model]
    encoded = encode_input(valid_df, artifacts["columns"], artifacts["scaler"])

    results_df = valid_df[RAW_COLUMNS].copy()
    results_df["extrapolation_warning"] = valid_df["warnings"]

    log_rows = []
    for model_name in models_to_run:
        model = artifacts["models"][model_name]
        predictions = model.predict(encoded)
        probabilities = model.predict_proba(encoded)[:, 1]
        risk_buckets = [risk_bucket_for(p) for p in probabilities]

        prefix = model_registry.display_name(model_name).replace(" ", "_").replace("-", "_")
        results_df[f"{prefix}_Prediction"] = predictions
        results_df[f"{prefix}_Probability"] = probabilities
        results_df[f"{prefix}_RiskBucket"] = risk_buckets

        for i in range(len(valid_df)):
            log_rows.append({
                **valid_df.iloc[i][RAW_COLUMNS].to_dict(),
                "model_name": model_name,
                "prediction": int(predictions[i]),
                "probability": float(probabilities[i]),
                "risk_bucket": risk_buckets[i],
            })

    st.subheader("Results")
    primary_prefix = model_registry.display_name(models_to_run[0]).replace(" ", "_").replace("-", "_")
    bucket_counts = results_df[f"{primary_prefix}_RiskBucket"].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Low risk", int(bucket_counts.get("Low", 0)))
    col2.metric("🟡 Moderate risk", int(bucket_counts.get("Moderate", 0)))
    col3.metric("🔴 High risk", int(bucket_counts.get("High", 0)))

    st.dataframe(results_df, hide_index=True, width="stretch")
    st.download_button("Download results (CSV)", results_df.to_csv(index=False), "results.csv", "text/csv")

    batch_id = database.log_batch(log_rows)
    st.success(f"Logged {len(log_rows)} prediction(s) to history (batch ID: {batch_id[:8]}).")
