
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = ["Crop", "Season", "State"]

NUMERICAL_FEATURES = [
    "Crop_Year",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "Yield"


def build_model():
    """Build the preprocessing and log-target prediction pipeline."""

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
        (
            "numerical",
            StandardScaler(),
            NUMERICAL_FEATURES,
        ),
    ])

    model = Pipeline([
        ("preprocessing", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    return model


def train_model(df):
    """Train the model using log1p-transformed yield."""

    X = df[FEATURES]
    y = np.log1p(df[TARGET])

    model = build_model()
    model.fit(X, y)

    return model


def predict_yield(model, X):
    """Return predictions in the original yield scale."""

    log_predictions = model.predict(X)
    log_predictions = np.maximum(log_predictions, 0)

    return np.expm1(log_predictions)
