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
        # Convert request JSON to DataFrame
        row = pd.DataFrame([data])

        # Add engineered features
        row = add_engineered_features(row)

        # Select model features
        row_features = row[FEATURE_COLS]

        # Check input features
        if not np.isfinite(row_features.to_numpy(dtype=float)).all():
            raise HTTPException(
                status_code=400,
                detail="Input contains NaN or infinite values."
            )

        # Predict class
        pred_label = rf_model.predict(row_features)[0]

        # Predict probabilities
        probabilities = rf_model.predict_proba(row_features)[0]

        # Convert predicted label
        mode = label_encoder.inverse_transform([pred_label])[0]

        # Debug: check probabilities
        print("Predicted label:", pred_label)
        print("Predicted mode:", mode)
        print("Probabilities:", probabilities)
        print("Model classes:", rf_model.classes_)
        print("Label encoder classes:", label_encoder.classes_)

        # Check whether probabilities contain NaN/Inf
        if not np.isfinite(probabilities).all():

            return {
                "mode": str(mode),
                "confidence": None,
                "warning": "Model returned NaN or infinite probability values."
            }

        confidence = {
            str(cls): round(float(probability), 4)
            for cls, probability in zip(
                label_encoder.classes_,
                probabilities
            )
        }

        return {
            "mode": str(mode),
            "confidence": confidence
        }

    except HTTPException:
        raise

    except Exception as e:
        print("Prediction error:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )