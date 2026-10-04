from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="EcoTrace Transport Mode API",
    description="Random Forest API for transport mode detection",
    version="1.0.0"
)


# --------------------------------------------------
# Enable CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model_data = joblib.load("transport_mode_model.pkl")

rf_model = model_data["model"]
label_encoder = model_data["label_encoder"]
FEATURE_COLS = model_data["feature_cols"]


# --------------------------------------------------
# Feature Engineering
# Same as training notebook
# --------------------------------------------------

def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:

    d = data.copy()

    # Total 3-axis acceleration magnitude
    d["total_accel"] = np.sqrt(
        d["ax_mean"] ** 2 +
        d["ay_mean"] ** 2 +
        d["az_mean"] ** 2
    )

    # Combined acceleration noise
    d["total_accel_std"] = (
        d["ax_std"] +
        d["ay_std"] +
        d["az_std"]
    )

    # Speed-to-gyro ratio
    d["speed_gyro_ratio"] = (
        d["speed"] /
        (d["gyro_magnitude_mean"] + 1e-3)
    )

    # Motion intensity
    d["motion_intensity"] = (
        d["magnitude_std"] *
        d["speed"]
    )

    return d


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "EcoTrace Transport Mode API is running",
        "model": "Random Forest",
        "endpoint": "/predict"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_transport_mode(data: dict):

    # Create DataFrame from received input
    row = pd.DataFrame([data])

    # Apply same feature engineering
    row = add_engineered_features(row)

    # Select exact features used during training
    row_features = row[FEATURE_COLS]

    # Prediction
    pred_label = rf_model.predict(row_features)[0]

    # Prediction probabilities
    probabilities = rf_model.predict_proba(row_features)[0]

    # Convert numeric label back to transport name
    mode = label_encoder.inverse_transform([pred_label])[0]

    # Create confidence dictionary
    confidence = {
        cls: round(float(probability), 4)
        for cls, probability in zip(
            label_encoder.classes_,
            probabilities
        )
    }

    return {
        "mode": mode,
        "confidence": confidence
    }