import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and columns
model = joblib.load('LogisticRegression_heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('col.pkl')

st.title("❤️ Heart Disease Prediction")

st.markdown("Fill in the following details to check your heart disease risk.")

def user_input():
    age = st.slider("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "LVH", "ST"])
    max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": rest_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    return pd.DataFrame([data])

# Get user input as DataFrame
input_df = user_input()

# One-hot encode user input (like training)
input_encoded = pd.get_dummies(input_df)

# Add missing columns (with zeros)
for col in columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns to match training
input_encoded = input_encoded[columns]

# Scale numeric columns
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction:")
    if prediction == 1:
        st.error(f"⚠️ High risk of Heart Disease! (Confidence: {proba:.2%})")
    else:
        st.success(f"✅ Low risk of Heart Disease. (Confidence: {proba:.2%})")
