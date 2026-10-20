
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .crop_yield_model import train_model, predict_yield, FEATURES
except ImportError:
    from crop_yield_model import train_model, predict_yield, FEATURES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "crop_yield.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    for column in ["Crop", "Season", "State"]:
     df[column] = df[column].astype(str).str.strip()

    # Train on earlier years; test on later years.
    train_df = df[df["Crop_Year"] <= 2017].copy()
    test_df = df[df["Crop_Year"].isin([2018, 2019])].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("The dataset does not contain the required years.")

    model = train_model(train_df)

    predictions = predict_yield(model, test_df[FEATURES])

    import numpy as np

    print("\nPREDICTION SANITY CHECK")
    print(f"All predictions finite: {np.isfinite(predictions).all()}")
    print(f"All predictions non-negative: {(predictions >= 0).all()}")
    print(f"Missing predictions: {np.isnan(predictions).sum()}")
    actual = test_df["Yield"].to_numpy()

    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5
    r2 = r2_score(actual, predictions)

    print("CROP YIELD MODEL EVALUATION")
    print("-" * 35)
    print(f"Training records (through 2017): {len(train_df):,}")
    print(f"Testing records (2018–2019):    {len(test_df):,}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    
    # Inspect the five largest absolute prediction errors.
    results = test_df[["Crop", "Crop_Year", "State", "Yield"]].copy()
    results["Predicted_Yield"] = predictions
    results["Absolute_Error"] = (
        results["Yield"] - results["Predicted_Yield"]
    ).abs()

    print("\nFIVE LARGEST PREDICTION ERRORS")
    print(
        results.nlargest(5, "Absolute_Error").to_string(index=False)
    )
    print("\nERROR SUMMARY BY CROP")

    crop_errors = (
        results.groupby("Crop")
        .agg(
            Records=("Absolute_Error", "size"),
            MAE=("Absolute_Error", "mean"),
        )
        .sort_values("MAE", ascending=False)
    )

    print(crop_errors.head(10).to_string())


    print("\nERROR BY CROP GROUP")

    for group_name, group in [
        ("Coconut", results[results["Crop"] == "Coconut"]),
        ("All other crops", results[results["Crop"] != "Coconut"]),
    ]:
        if group.empty:
            continue

        group_mae = mean_absolute_error(
            group["Yield"], group["Predicted_Yield"]
        )
        group_rmse = mean_squared_error(
            group["Yield"], group["Predicted_Yield"]
        ) ** 0.5

        print(
            f"{group_name}: records={len(group)}, "
            f"MAE={group_mae:.4f}, RMSE={group_rmse:.4f}"
        )

        
    print("\nPERCENTAGE ERROR BY CROP GROUP")

    for group_name, group in [
        ("Coconut", results[results["Crop"] == "Coconut"]),
        ("All other crops", results[results["Crop"] != "Coconut"]),
    ]:
        valid = group[group["Yield"] > 0].copy()

        if valid.empty:
            print(f"{group_name}: no positive actual yields")
            continue

        percentage_errors = (
            valid["Absolute_Error"] / valid["Yield"]
        ) * 100

        print(
            f"{group_name}: records={len(valid)}, "
            f"MAPE={percentage_errors.mean():.2f}%"
        )



if __name__ == "__main__":
    main()
