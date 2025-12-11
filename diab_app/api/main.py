# ============================================================
# FastAPI Backend for Diabetes Prediction (XGBoost + SMOTE)
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import json
import math
import os

# ------------------------------------------------------------
# 1. SETUP PATHS (Robust Fix)
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try 'model' (singular) first, then 'models' (plural) as fallback
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "model")
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = os.path.join(CURRENT_DIR, "..", "models")

def load_file(filename):
    return os.path.join(MODEL_DIR, filename)

# ------------------------------------------------------------
# 2. LOAD MODELS
# ------------------------------------------------------------
print(f"Loading models from: {MODEL_DIR}") 

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

# ------------------------------------------------------------
# 3. FASTAPI APP
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 4. PREPROCESSING
# ------------------------------------------------------------
def create_features(raw_data: dict):
    # Extract variables
    Preg = raw_data["Pregnancies"]
    Gluc = raw_data["Glucose"]
    BP = raw_data["BloodPressure"]
    Skin = raw_data["SkinThickness"]
    Ins = raw_data["Insulin"]
    BMI = raw_data["BMI"]
    DPF = raw_data["DiabetesPedigreeFunction"]
    Age = raw_data["Age"]

    row = {
        "Pregnancies": Preg,
        "Glucose": Gluc,
        "BloodPressure": BP,
        "SkinThickness": Skin,
        "Insulin": Ins,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DPF,
        "Age": Age
    }

    row["Insulin_missing_flag"] = 1 if Ins == 0 else 0
    row["SkinThickness_missing_flag"] = 1 if Skin == 0 else 0
    row["Glucose_NA"] = 1 if Gluc == 0 else 0
    row["BloodPressure_NA"] = 1 if BP == 0 else 0
    row["SkinThickness_NA"] = 1 if Skin == 0 else 0
    row["Insulin_NA"] = 1 if Ins == 0 else 0
    row["BMI_NA"] = 1 if BMI == 0 else 0

    row["BMI_Age_Interaction"] = BMI * Age
    row["Glucose_Insulin_Product"] = Gluc * Ins
    row["BMI_per_Age"] = BMI / Age if Age != 0 else 0

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
    return {
        "probability": float(prob),
        "prediction": pred,
        "threshold_used": threshold,
        "mode": mode
    }

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running!"}

@app.post("/predict")
def predict(data: PatientData, mode: str = "balanced"):
    raw = data.dict()
    if mode not in ["balanced", "high"]:
        return {"error": "Mode must be 'balanced' or 'high'."}
    return preprocess_and_predict(raw, mode)
