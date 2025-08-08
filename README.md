# 🩺 Heart Disease Risk Predictor Machine Learning Model

A machine learning-powered web application that predicts the likelihood of heart disease based on user-provided health data. Built with *Python, **Streamlit, and **scikit-learn*, and deployed online for real-time usage.

🔗 *Live Demo:* [Click to Open App](https://heartdiseasepredictor-ogsshukdbbhncvui4trcm7.streamlit.app/)

---

## 📌 Overview

This project demonstrates a health risk classification system using machine learning. It allows users to input medical parameters and receive a prediction on whether they are at risk of heart disease using trained models like *Logistic Regression* and *K-Nearest Neighbors (KNN)*.

> ⚠️ This app is for *educational purposes only* and should *not be used for real-world diagnosis* or medical treatment.

---

## 🚀 Features

- 🧠 Predicts heart disease using two ML models: Logistic Regression & KNN
- 📊 Trained on a real-world dataset (heart.csv)
- 🧪 Uses StandardScaler for input normalization
- 💾 Pickle files for model deployment: *.pkl
- 🌐 Deployed via *Streamlit Cloud* for web access

---

## 🧰 Tech Stack

| Layer        | Tools Used                            |
|--------------|----------------------------------------|
| Frontend     | Streamlit                              |
| Backend      | Python, scikit-learn, pandas, NumPy    |
| ML Models    | Logistic Regression, K-Nearest Neighbors |
| Preprocessing| StandardScaler, LabelEncoder           |
| Deployment   | Streamlit Cloud                        |

---

## 📁 Project Structure

📦 heart-disease-predictor/
├── 📜 app.py → Streamlit web application
├── 📓 Heart2_Project.ipynb → Notebook for EDA + model training
├── 📊 heart.csv → UCI dataset used for model training
│
├── 🔍 LogisticRegression_heart.pkl → Trained Logistic Regression model
├── 🔍 KNN_heart.pkl → Trained KNN model
├── ⚖️ scaler.pkl → Pre-fitted StandardScaler
├── 📋 col.pkl → Preserved column order for inputs
│
├── 📄 requirements.txt → Python dependencies
└── 🗂️ .ipynb_checkpoints/ → Auto-saved notebook checkpoints
