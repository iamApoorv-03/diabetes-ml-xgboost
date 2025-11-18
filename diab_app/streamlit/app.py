# ============================================================
# Streamlit Frontend for Diabetes Prediction
# ============================================================

import streamlit as st
import requests
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# ------------------------------------------------------------
# API URL (local FastAPI backend)
# ------------------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"

# ------------------------------------------------------------
# Load SHAP images from assets
# ------------------------------------------------------------
ASSETS = "../assets"

shap_images = {
    "Global Importance (Bar Plot)": "shap_bar.png",
    "Global Importance (Beeswarm)": "shap_beeswarm.png",
    "Waterfall Example Patient": "shap_waterfall_sample10.png",
    "Dependence - Glucose": "shap_dependence_Glucose.png",
    "Dependence - Insulin": "shap_dependence_Insulin.png",
    "Dependence - BMI": "shap_dependence_BloodPressure.png",
    "Dependence - Pregnancies": "shap_dependence_Pregnancies.png",
}

# ============================================================
# Title
# ============================================================
st.title("🩺 Diabetes Prediction Web App")
st.write("This app uses **XGBoost + SMOTE** with a **Dual Threshold System** (balanced & high-sensitivity).")

# ============================================================
# Sidebar Navigation
# ============================================================
menu = st.sidebar.radio("Navigation", ["🔍 Prediction", "📊 SHAP Interpretability", "ℹ About Model"])

# ============================================================
# PAGE 1 — Prediction
# ============================================================
if menu == "🔍 Prediction":
    st.header("Enter Patient Data")

    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 2)
        Glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
        BloodPressure = st.number_input("Blood Pressure", 0.0, 200.0, 70.0)
        SkinThickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)

    with col2:
        Insulin = st.number_input("Insulin", 0.0, 1000.0, 80.0)
        BMI = st.number_input("BMI", 0.0, 70.0, 30.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        Age = st.number_input("Age", 1, 120, 33)

    mode = st.radio("Prediction Mode", ["balanced", "high"], horizontal=True)

    if st.button("Predict"):
        with st.spinner("Contacting API…"):
            payload = {
                "Pregnancies": Pregnancies,
                "Glucose": Glucose,
                "BloodPressure": BloodPressure,
                "SkinThickness": SkinThickness,
                "Insulin": Insulin,
                "BMI": BMI,
                "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
                "Age": Age
            }

            try:
                response = requests.post(f"{API_URL}?mode={mode}", json=payload)
                result = response.json()

                st.subheader("📌 Prediction Results")

                prob = result["probability"]
                pred = result["prediction"]

                st.metric("Probability of Diabetes", f"{prob:.2f}")
                st.write("**Prediction:**", "🟥 Diabetic" if pred == 1 else "🟩 Not Diabetic")

            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

# ============================================================
# PAGE 2 — SHAP Interpretability
# ============================================================
elif menu == "📊 SHAP Interpretability":
    st.header("Model Interpretability (SHAP)")

    st.write("These visualizations show **why the model makes predictions**.")

    for title, img in shap_images.items():
        path = os.path.join(ASSETS, img)
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_column_width=True)
        else:
            st.warning(f"Missing image: {img}")

# ============================================================
# PAGE 3 — About Model
# ============================================================
else:
    st.header("ℹ About This Model")

    st.write("""
    ###  Model Overview
    This system uses **XGBoost + SMOTE** trained on the PIMA Diabetes dataset.

    ###  Why This Model?
    - Highest F1-score (0.689)
    - Best recall among stable models
    - Most balanced confusion matrix
    - Strong SHAP interpretability

    ### ⚡ Dual Threshold System
    | Mode | Threshold | Use Case |
    |------|-----------|----------|
    | **Balanced Mode** | ~0.51 | Normal screening |
    | **High-Sensitivity Mode** | ~0.19 | High-risk, hospital triage |

    This allows deployment in both **general clinics** and **critical care** environments.
    """)

