from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="EcoTrace Transport Mode API",
    description="Random Forest API for transport mode detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model
model_data = joblib.load("transport_mode_model.pkl")

rf_model = model_data["model"]
label_encoder = model_data["label_encoder"]
FEATURE_COLS = model_data["feature_cols"]


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:

    d = data.copy()

    required_cols = [
        "ax_mean",
        "ay_mean",
        "az_mean",
        "ax_std",
        "ay_std",
        "az_std",
        "magnitude_mean",
        "magnitude_std",
        "rms",
        "speed",
        "speed_std",
        "gyro_magnitude_mean",
        "gyro_magnitude_std"
    ]

    missing = [col for col in required_cols if col not in d.columns]

    if missing:
        raise ValueError(
            f"Missing input features: {missing}"
        )

    # Make sure values are numeric
    for col in required_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Check for invalid values
    if d[required_cols].isna().any().any():
        raise ValueError(
            "Input contains missing or non-numeric values."
        )

    # Engineered features
    d["total_accel"] = np.sqrt(
        d["ax_mean"] ** 2 +
        d["ay_mean"] ** 2 +
        d["az_mean"] ** 2
    )

    d["total_accel_std"] = (
        d["ax_std"] +
        d["ay_std"] +
        d["az_std"]
    )

    d["speed_gyro_ratio"] = (
        d["speed"] /
        (d["gyro_magnitude_mean"] + 1e-3)
    )

    d["motion_intensity"] = (
        d["magnitude_std"] *
        d["speed"]
    )

    return d


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "EcoTrace Transport Mode API is running",
        "model": "Random Forest",
        "endpoint": "/predict"
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
def predict_transport_mode(data: dict):
    try:
        print("INPUT:", data)

        row = pd.DataFrame([data])

        row = add_engineered_features(row)

        row_features = row[FEATURE_COLS]

        print("MODEL FEATURES:")
        print(row_features)

        # Check input values
        print("FINITE INPUT:",
              np.isfinite(row_features.to_numpy(dtype=float)).all())

        # Test prediction only
        pred_label = rf_model.predict(row_features)[0]

        print("PREDICT LABEL:", pred_label)

        mode = label_encoder.inverse_transform([pred_label])[0]

        print("PREDICTED MODE:", mode)

        # IMPORTANT:
        # Do NOT call predict_proba() yet.
        return {
            "mode": str(mode)
        }

    except Exception as e:
        print("PREDICTION ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )