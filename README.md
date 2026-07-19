# 🩺 Heart Disease Risk Predictor Machine Learning Model

A machine learning-powered web application that predicts the likelihood of heart disease based on user-provided health data. Built with *Python, **Streamlit, and **scikit-learn*, and deployed online for real-time usage.

🔗 **Live Demo:** [Heart Disease Risk Predictor](https://heartdiseasepredictormachinelearningmodel-dgz2vt73lytfpjp8r4mj.streamlit.app/)

---

## 📌 Overview

This project is a health risk classification system for clinical/business use. It allows staff to input a single patient's medical parameters, or upload a CSV of many patients at once, and get a risk prediction from either of two trained models (Logistic Regression or KNN).

> ⚠️ This app provides a statistical risk estimate, not a medical diagnosis, and should not replace evaluation by a qualified healthcare professional.

---

## 🚀 Features

- 🧠 **Model comparison** — choose Logistic Regression or KNN per prediction, with a "why this result" breakdown (top contributing risk factors for LR, nearest-neighbor vote summary for KNN)
- 📊 **Batch prediction** — upload a CSV of many patients; invalid rows are reported separately rather than silently dropped or crashing the whole upload
- 📁 **Prediction history** — every prediction is logged to a local database, filterable/searchable, with staff able to record the real confirmed diagnosis once known
- 🔁 **Automatic model retraining** — periodically retrains on newly confirmed patient outcomes (never on the model's own unconfirmed predictions), with an accuracy guardrail so a retrained model only goes live if it doesn't regress
- 📧 **High-risk email alerts** — optional email notification when a single prediction comes back high risk
- 🧪 Trained on a real-world heart failure dataset (`heart.csv`)

---

## 🧰 Tech Stack

| Layer        | Tools Used                            |
|--------------|----------------------------------------|
| Frontend     | Streamlit (multipage app)              |
| Backend      | Python, scikit-learn, pandas, NumPy    |
| ML Models    | Logistic Regression, K-Nearest Neighbors |
| Preprocessing| StandardScaler, one-hot encoding       |
| Storage      | SQLite by default; hosted Postgres (Supabase/Neon) via `[database]` secret for permanent history |
| Alerts       | SMTP email (stdlib `smtplib`)          |
| Deployment   | Streamlit Cloud                        |

---

## 📁 Project Structure

```
📦 heart-disease-predictor/
├── app.py                          → Entry point: routing, styles, startup checks
├── views/
│   ├── risk_assessment.py           → Single-patient prediction (home page)
│   ├── batch_prediction.py          → CSV upload for multiple patients
│   └── prediction_history.py        → History, outcome confirmation, retrain log
├── core/
│   ├── schema.py                    → Valid categories, bounds, risk/retrain thresholds
│   ├── preprocessing.py             → Shared validation + encode/scale pipeline
│   ├── model_registry.py            → Loads models/scaler/columns once
│   ├── risk_factors.py              → Prediction explanations (LR / KNN)
│   ├── database.py                  → SQLite logging, history, outcomes, retrain log
│   ├── notifications.py             → High-risk email alerts
│   ├── retrain.py                   → Automatic scheduled retraining + guardrail
│   └── ui.py                        → Shared disclaimer + risk badge
├── .streamlit/
│   ├── config.toml                  → App theme
│   └── secrets.toml.example         → SMTP config template (copy to secrets.toml)
├── data/                            → predictions.db lives here at runtime (gitignored)
├── Heart2_Project.ipynb             → Notebook for EDA + original model training
├── heart.csv                        → Base training dataset
├── LogisticRegression_heart.pkl, KNN_heart.pkl, scaler.pkl, col.pkl → Model artifacts
├── requirements.txt                 → App dependencies
├── requirements-notebook.txt        → Extra dependencies for the notebook only
└── .gitignore
```

## ⚙️ Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

To enable high-risk email alerts, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in real SMTP credentials (or add the same keys via Streamlit Cloud's Settings → Secrets). Without this, the app runs fine and the notify-email field is simply hidden.

## ⚠️ Known limitations

- **Default SQLite storage is not durable on Streamlit Cloud** — a redeploy/restart can wipe `data/predictions.db`. Fix: set `[database] url` in secrets to a free hosted Postgres (Supabase/Neon) and all history/outcomes/retrain data become permanent. See `.streamlit/secrets.toml.example`.
- **Automatic retraining has no human-review checkpoint** before a new model goes live — an accuracy guardrail prevents outright regressions, but it's not a substitute for periodically reviewing the retrain log on the History page.
- Storing patient vitals (even without name/MRN fields) is still sensitive health data — review your own privacy/compliance obligations before using this with real patients.
