from pathlib import Path

import joblib
import pandas as pd

try:
    from .crop_yield_model import train_model
except ImportError:
    from crop_yield_model import train_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "crop_yield.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "crop_yield_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)

    model = train_model(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()