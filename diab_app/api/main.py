# ============================================================
# FastAPI Backend for Diabetes Prediction (XGBoost + SMOTE)
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import json
import math

# ------------------------------------------------------------
# Load model, scaler, feature list, thresholds
# ------------------------------------------------------------

with open("../models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("../models/feature_names.json", "r") as f:
    feature_names = json.load(f)

with open("../models/thresholds.json", "r") as f:
    thresholds = json.load(f)

balanced_threshold = thresholds["balanced_threshold"]
high_threshold = thresholds["high_sensitivity_threshold"]

# ------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------
app = FastAPI(title="Diabetes Prediction API")


# ------------------------------------------------------------
# Input Schema (RAW FEATURES ONLY)
# ------------------------------------------------------------
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
# Feature Engineering (MUST MATCH NOTEBOOK)
# ------------------------------------------------------------
def create_features(data):

    Preg = data["Pregnancies"]
    Gluc = data["Glucose"]
    BP   = data["BloodPressure"]
    Skin = data["SkinThickness"]
    Ins  = data["Insulin"]
    BMI  = data["BMI"]
    DPF  = data["DiabetesPedigreeFunction"]
    Age  = data["Age"]

    # Create dictionary
    row = {}

    # ---- raw features ----
    row["Pregnancies"] = Preg
    row["Glucose"] = Gluc
    row["BloodPressure"] = BP
    row["SkinThickness"] = Skin
    row["Insulin"] = Ins
    row["BMI"] = BMI
    row["DiabetesPedigreeFunction"] = DPF
    row["Age"] = Age

    # ---- Missing flags ----
    row["Insulin_missing_flag"] = 1 if Ins == 0 else 0
    row["SkinThickness_missing_flag"] = 1 if Skin == 0 else 0

    # ---- NA flags (as used in model training) ----
    row["Glucose_NA"] = 1 if Gluc == 0 else 0
    row["BloodPressure_NA"] = 1 if BP == 0 else 0
    row["SkinThickness_NA"] = 1 if Skin == 0 else 0
    row["Insulin_NA"] = 1 if Ins == 0 else 0
    row["BMI_NA"] = 1 if BMI == 0 else 0

    # ---- Interaction Features ----
    row["BMI_Age_Interaction"] = BMI * Age
    row["Glucose_Insulin_Product"] = Gluc * Ins
    row["BMI_per_Age"] = BMI / Age if Age != 0 else 0

    # ---- Log features ----
    row["Log_BloodPressure"] = math.log(BP) if BP > 0 else 0
    row["Log_DiabetesPedigreeFunction"] = math.log(DPF) if DPF > 0 else 0

    # ---- Binary feature ----
    row["High_Glucose"] = 1 if Gluc > 140 else 0

    # ---- Extra interactions ----
    row["BMI_Glucose"] = BMI * Gluc
    row["Age_Glucose"] = Age * Gluc

    # ---- Medical insulin resistance index ----
    row["HOMA_IR"] = (Gluc * Ins) / 405

    # ---- Sqrt ----
    row["Sqrt_Insulin"] = math.sqrt(Ins) if Ins > 0 else 0

    # -----------------------------------------------------------
    # Build final row EXACTLY in the saved feature order
    # -----------------------------------------------------------
    final = [row[f] for f in feature_names]

    return np.array(final).reshape(1, -1)


# ------------------------------------------------------------
# Main Prediction Function
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# API Routes
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running!"}


@app.post("/predict")
def predict(data: PatientData, mode: str = "balanced"):

    raw = data.dict()

    if mode not in ["balanced", "high"]:
        return {"error": "Mode must be 'balanced' or 'high'."}

    return preprocess_and_predict(raw, mode)
