import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")
API_URL = "http://127.0.0.1:8000/predict"

# --- PATH SETUP ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_SCRIPT_DIR, "..", "assets")

shap_images = {
    "Global Importance (Bar Plot)": "shap_bar.png",
    "Global Importance (Beeswarm)": "shap_beeswarm.png",
    "Waterfall Example Patient": "shap_waterfall_sample10.png",
    "Dependence - Glucose": "shap_dependence_Glucose.png",
    "Dependence - Insulin": "shap_dependence_Insulin.png",
    "Dependence - BMI": "shap_dependence_BloodPressure.png",
    "Dependence - Pregnancies": "shap_dependence_Pregnancies.png",
}

st.title("🩺 Diabetes Prediction Web App")
st.write("This app uses **XGBoost + SMOTE** with a **Dual Threshold System**.")

st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to:", ["🏥 Predict Diabetes", "📊 SHAP Interpretability", "ℹ About Model"])

if menu == "🏥 Predict Diabetes":
    st.header("Patient Data Entry")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 100)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 72)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age (years)", 0, 120, 30)
    st.subheader("⚙ Configuration")
    mode = st.radio("Select Sensitivity Mode:", ["balanced", "high"])
    if st.button("Predict Risk"):
        payload = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age
        }
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(API_URL, json=payload, params={"mode": mode})
                if response.status_code == 200:
                    result = response.json()
                    prob = result["probability"]
                    pred = result["prediction"]
                    st.divider()
                    if pred == 1:
                        st.error(f"🚨 **High Risk Detected** (Probability: {prob:.2%})")
                    else:
                        st.success(f"✅ **Low Risk** (Probability: {prob:.2%})")
                else:
                    st.error("Error from API.")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

elif menu == "📊 SHAP Interpretability":
    st.header("Model Interpretability (SHAP)")
    st.write("These visualizations show **why the model makes predictions**.")
    for title, img in shap_images.items():
        path = os.path.join(ASSETS_DIR, img)
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)
        else:
            st.warning(f"⚠️ Missing image: {img}")

else:
    st.header("ℹ About This Model")
    st.write("This system uses **XGBoost + SMOTE** trained on the PIMA Diabetes dataset.")
