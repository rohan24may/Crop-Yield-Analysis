
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


def clean_input_data(df):
    """Standardize whitespace in categorical features."""
    df = df.copy()

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype(str).str.strip()

    return df


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

    df = clean_input_data(df)

    X = df[FEATURES]
    y = np.log1p(df[TARGET])

    model = build_model()
    model.fit(X, y)

    return model

def predict_yield(model, X):
    """Return predictions in the original yield scale."""

    X = clean_input_data(X)

    log_predictions = model.predict(X)
    log_predictions = np.maximum(log_predictions, 0)

    return np.expm1(log_predictions)

