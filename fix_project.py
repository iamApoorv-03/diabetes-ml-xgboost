import os

# ------------------------------------------------------------------
# 1. CONTENT FOR MAIN.PY (FastAPI)
# ------------------------------------------------------------------
main_content = r'''from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import json
import math
import os

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Try singular 'model' first, fallback to 'models'
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "model")
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(CURRENT_DIR, "..", "models")

def load_file(filename):
    return os.path.join(MODEL_DIR, filename)

# --- LOAD MODELS ---
with open(load_file("xgb_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(load_file("scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(load_file("feature_names.json"), "r") as f:
    feature_names = json.load(f)
with open(load_file("thresholds.json"), "r") as f:
    thresholds = json.load(f)

balanced_threshold = thresholds["balanced_threshold"]
high_threshold = thresholds["high_sensitivity_threshold"]

app = FastAPI(title="Diabetes Prediction API")

class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

def create_features(raw_data: dict):
    Preg = raw_data["Pregnancies"]
    Gluc = raw_data["Glucose"]
    BP = raw_data["BloodPressure"]
    Skin = raw_data["SkinThickness"]
    Ins = raw_data["Insulin"]
    BMI = raw_data["BMI"]
    DPF = raw_data["DiabetesPedigreeFunction"]
    Age = raw_data["Age"]
    
    row = {
        "Pregnancies": Preg, "Glucose": Gluc, "BloodPressure": BP, "SkinThickness": Skin,
        "Insulin": Ins, "BMI": BMI, "DiabetesPedigreeFunction": DPF, "Age": Age
    }
    # Flags
    row["Insulin_missing_flag"] = 1 if Ins == 0 else 0
    row["SkinThickness_missing_flag"] = 1 if Skin == 0 else 0
    row["Glucose_NA"] = 1 if Gluc == 0 else 0
    row["BloodPressure_NA"] = 1 if BP == 0 else 0
    row["SkinThickness_NA"] = 1 if Skin == 0 else 0
    row["Insulin_NA"] = 1 if Ins == 0 else 0
    row["BMI_NA"] = 1 if BMI == 0 else 0
    # Interactions
    row["BMI_Age_Interaction"] = BMI * Age
    row["Glucose_Insulin_Product"] = Gluc * Ins
    row["BMI_per_Age"] = BMI / Age if Age != 0 else 0
    # Transforms
    row["Log_BloodPressure"] = np.log1p(BP)
    row["Log_DiabetesPedigreeFunction"] = np.log1p(DPF)
    row["High_Glucose"] = 1 if Gluc > 140 else 0
    row["BMI_Glucose"] = BMI * Gluc
    row["Age_Glucose"] = Age * Gluc
    row["HOMA_IR"] = (Gluc * Ins) / 405
    row["Sqrt_Insulin"] = math.sqrt(Ins) if Ins > 0 else 0
    
    final = [row[f] for f in feature_names]
    return np.array(final).reshape(1, -1)

def preprocess_and_predict(raw_data, mode="balanced"):
    x = create_features(raw_data)
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0][1]
    threshold = balanced_threshold if mode == "balanced" else high_threshold
    pred = int(prob >= threshold)
    return {"probability": float(prob), "prediction": pred, "threshold_used": threshold, "mode": mode}

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running!"}

@app.post("/predict")
def predict(data: PatientData, mode: str = "balanced"):
    if mode not in ["balanced", "high"]:
        return {"error": "Mode must be 'balanced' or 'high'."}
    return preprocess_and_predict(data.dict(), mode)
'''

# ------------------------------------------------------------------
# 2. CONTENT FOR APP.PY (Streamlit)
# ------------------------------------------------------------------
app_content = r'''import streamlit as st
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
'''

# ------------------------------------------------------------------
# 3. WRITE FILES
# ------------------------------------------------------------------
with open("diab_app/api/main.py", "w", encoding="utf-8") as f:
    f.write(main_content)
    print("✅ Fixed diab_app/api/main.py")

with open("diab_app/streamlit/app.py", "w", encoding="utf-8") as f:
    f.write(app_content)
    print("✅ Fixed diab_app/streamlit/app.py")
