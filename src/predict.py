
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import math

try:
    from .crop_yield_model import predict_yield, FEATURES
except ImportError:
    from crop_yield_model import predict_yield, FEATURES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "crop_yield_model.joblib"


@st.cache_resource
def load_model():
    """Load and cache the trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python src/train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_crop_yield(
    crop,
    crop_year,
    season,
    state,
    annual_rainfall,
    fertilizer,
    pesticide,
):
    """Predict crop yield using the trained model."""

    numeric_values = {
        "Crop Year": crop_year,
        "Annual Rainfall": annual_rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide,
    }


    for name, value in numeric_values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a valid number.")

        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite.")

        if number < 0:
            raise ValueError(f"{name} cannot be negative.")

    categorical_values = [crop, season, state]

    if any(
        value is None or not str(value).strip()
        for value in categorical_values
    ):
        raise ValueError("Crop, season and state are required.")

    data = pd.DataFrame([{
        "Crop": crop,
        "Crop_Year": crop_year,
        "Season": season,
        "State": state,
        "Annual_Rainfall": annual_rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide,
    }])

    model = load_model()

    prediction = predict_yield(model, data[FEATURES])

    return float(prediction[0])
