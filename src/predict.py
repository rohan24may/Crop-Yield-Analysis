
import joblib
import pandas as pd
from pathlib import Path

try:
    from .crop_yield_model import predict_yield, FEATURES
except ImportError:
    from crop_yield_model import predict_yield, FEATURES


# Locate the saved model in the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "crop_yield_model.joblib"

model = joblib.load(MODEL_PATH)


def predict_crop_yield(
    crop,
    crop_year,
    season,
    state,
    annual_rainfall,
    fertilizer,
    pesticide
):
    data = pd.DataFrame([{
        "Crop": crop,
        "Crop_Year": crop_year,
        "Season": season,
        "State": state,
        "Annual_Rainfall": annual_rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }])

    prediction = predict_yield(model, data[FEATURES])

    return prediction[0]


if __name__ == "__main__":
    result = predict_crop_yield(
        crop="Maize",
        crop_year=2019,
        season="Kharif",
        state="Punjab",
        annual_rainfall=800,
        fertilizer=100000,
        pesticide=500
    )

    print(f"Predicted crop yield: {result:.4f}")
