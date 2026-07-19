import pandas as pd
import streamlit as st

from core import database, model_registry, notifications, risk_factors, ui
from core.preprocessing import clean_missing_sentinels, encode_input, validate_raw
from core.schema import CATEGORICAL_OPTIONS, risk_bucket_for

ui.page_header(
    "Patient Risk Assessment",
    "Enter the patient's clinical parameters to estimate their heart disease risk.",
)
ui.render_disclaimer()

latest_retrain = database.latest_retrain_run()
if latest_retrain:
    status = "applied" if latest_retrain["applied"] else "not applied (would have regressed accuracy)"
    st.caption(
        f"Model last auto-retrained on {latest_retrain['ran_at'][:10]} using "
        f"{latest_retrain['records_used']} patient records "
        f"({latest_retrain['old_accuracy']:.1%} → {latest_retrain['new_accuracy']:.1%}, {status})."
    )

artifacts = model_registry.load_artifacts()
model_names = list(artifacts["models"].keys())

with st.sidebar:
    st.markdown("**Settings**")
    selected_model = st.selectbox(
        "Prediction method", model_names, key="selected_model",
        format_func=lambda n: model_registry.display_name(n)
        + (" (recommended)" if n == "Logistic Regression" else ""),
        help="Standard risk model: weighs each clinical factor statistically. "
             "Similar-patient comparison: looks at the most similar past patients.",
    )

    notify_email = ""
    if notifications.is_configured():
        notify_email = st.text_input(
            "Notify email on high risk (optional)",
            key="notify_email",
            help="If the result comes back High risk, an alert email is sent here.",
        )
    else:
        st.markdown(
            """
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px;
                        padding: 0.7rem 0.85rem; font-size: 0.8rem; line-height: 1.55;
                        color: #cbd5e1; margin-top: 0.4rem;">
                <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem;">
                    Email alerts — setup pending
                </div>
                Add your Gmail address and an
                <a href="https://myaccount.google.com/apppasswords" target="_blank"
                   style="color: #67e8f9; text-decoration: underline;">App&nbsp;Password</a>
                in
                <code style="background: #0f172a; padding: 0.05rem 0.35rem; border-radius: 4px;
                             color: #a5f3fc;">.streamlit/secrets.toml</code>,
                then reload the app.<br>
                <span style="color: #94a3b8;">On Streamlit Cloud: Settings → Secrets.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def predict_for(patient: dict, model_name: str) -> tuple[int, float, pd.DataFrame]:
    """Validate + encode + predict one patient dict. Returns
    (prediction, probability, encoded_row)."""
    df = clean_missing_sentinels(pd.DataFrame([patient]))
    valid_df, _ = validate_raw(df)
    encoded = encode_input(valid_df, artifacts["columns"], artifacts["scaler"])
    model = artifacts["models"][model_name]
    return int(model.predict(encoded)[0]), float(model.predict_proba(encoded)[0][1]), encoded


with st.form("patient_form"):
    st.markdown("##### 🧍 Patient profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 100, 50)
    with col2:
        sex = st.selectbox("Sex", CATEGORICAL_OPTIONS["Sex"], format_func=lambda s: {"M": "Male", "F": "Female"}[s])
    with col3:
        fasting_bs = st.selectbox(
            "Fasting blood sugar > 120 mg/dl", [0, 1],
            format_func=lambda v: "Yes" if v else "No",
        )

    st.markdown("##### 🩺 Vitals & labs")
    col4, col5, col6 = st.columns(3)
    with col4:
        resting_bp = st.slider("Resting blood pressure (mm Hg)", 80, 200, 120)
    with col5:
        cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    with col6:
        max_hr = st.slider("Max heart rate achieved", 60, 220, 150)

    st.markdown("##### 📈 Cardiac findings")
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        chest_pain = st.selectbox(
            "Chest pain type", CATEGORICAL_OPTIONS["ChestPainType"],
            format_func=lambda c: {
                "ATA": "Atypical angina", "NAP": "Non-anginal pain",
                "TA": "Typical angina", "ASY": "Asymptomatic",
            }[c],
        )
    with col8:
        rest_ecg = st.selectbox(
            "Resting ECG", CATEGORICAL_OPTIONS["RestingECG"],
            format_func=lambda e: {"Normal": "Normal", "LVH": "LV hypertrophy", "ST": "ST-T abnormality"}[e],
        )
    with col9:
        exercise_angina = st.selectbox(
            "Exercise-induced angina", CATEGORICAL_OPTIONS["ExerciseAngina"],
            format_func=lambda v: "Yes" if v == "Y" else "No",
        )
    with col10:
        st_slope = st.selectbox(
            "ST slope", CATEGORICAL_OPTIONS["ST_Slope"],
            format_func=lambda s: {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}[s],
        )
    oldpeak = st.slider("Oldpeak — ST depression induced by exercise", 0.0, 6.0, 1.0, 0.1)

    submitted = st.form_submit_button("Assess risk", type="primary")

if submitted:
    patient = {
        "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
        "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": rest_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }
    cleaned_df = clean_missing_sentinels(pd.DataFrame([patient]))
    valid_df, error_df = validate_raw(cleaned_df)

    if not error_df.empty:
        st.error("Some inputs are invalid: " + error_df.iloc[0]["errors"])
        st.session_state.pop("assessment", None)
    else:
        prediction, probability, _ = predict_for(patient, selected_model)
        risk_bucket = risk_bucket_for(probability)

        database.log_prediction(
            patient=patient, model_name=selected_model, prediction=prediction,
            probability=probability, risk_bucket=risk_bucket, source="single",
        )

        if risk_bucket == "High" and notify_email:
            sent, message = notifications.send_high_risk_alert(
                recipient_email=notify_email, patient_summary=patient,
                probability=probability, model_name=selected_model,
            )
            (st.success if sent else st.warning)(message)

        # Persist across reruns so the what-if simulator's sliders can move
        # without losing the assessed result. sim_run keys the simulator
        # widgets so each new assessment resets them to the new patient.
        st.session_state["assessment"] = {
            "patient": patient, "model": selected_model,
            "warnings": valid_df.iloc[0]["warnings"],
        }
        st.session_state["sim_run"] = st.session_state.get("sim_run", 0) + 1

assessment = st.session_state.get("assessment")
if assessment:
    patient = assessment["patient"]
    model_name = assessment["model"]
    prediction, probability, encoded = predict_for(patient, model_name)
    risk_bucket = risk_bucket_for(probability)

    if assessment["warnings"]:
        st.info(f"Note: {assessment['warnings']}")

    ui.render_risk_badge(risk_bucket, probability)
    st.caption(f"Assessed with the {model_registry.display_name(model_name).lower()}.")

    with st.expander("Why this result? (top contributing factors)"):
        model = artifacts["models"][model_name]
        if model_name == "Logistic Regression":
            contributions = risk_factors.lr_contributions(encoded.iloc[0], model, artifacts["columns"])
            st.dataframe(
                contributions[["friendly_label", "direction", "contribution"]].rename(columns={
                    "friendly_label": "Factor", "direction": "Effect", "contribution": "Weight",
                }),
                hide_index=True, width="stretch",
            )
        else:
            summary = risk_factors.knn_neighbor_summary(encoded, model, prediction)
            st.write(
                f"{summary['n_matching']} of {summary['n_neighbors']} most similar patients "
                f"in the training data had outcome: **{summary['label']}**."
            )

    st.markdown("#### 🔬 What-if simulator")
    st.caption(
        "Adjust the modifiable factors below to see how this patient's estimated risk would "
        "change. This is a hypothetical model estimate to support lifestyle/treatment "
        "conversations — not a guaranteed outcome or medical advice."
    )

    run = st.session_state.get("sim_run", 0)
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_chol = st.slider(
            "Cholesterol (mg/dl)", 100, 600, int(patient["Cholesterol"]), key=f"sim_chol_{run}",
        )
    with sim_col2:
        sim_bp = st.slider(
            "Resting blood pressure (mm Hg)", 80, 200, int(patient["RestingBP"]), key=f"sim_bp_{run}",
        )
    with sim_col3:
        sim_hr = st.slider(
            "Max heart rate achieved", 60, 220, int(patient["MaxHR"]), key=f"sim_hr_{run}",
            help="Typically improves with cardiovascular fitness.",
        )

    sim_patient = {**patient, "Cholesterol": sim_chol, "RestingBP": sim_bp, "MaxHR": sim_hr}
    _, sim_probability, _ = predict_for(sim_patient, model_name)

    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Current estimated risk", f"{probability:.1%}")
    res_col2.metric(
        "Simulated risk",
        f"{sim_probability:.1%}",
        delta=f"{(sim_probability - probability) * 100:+.1f} pp",
        delta_color="inverse",
    )
