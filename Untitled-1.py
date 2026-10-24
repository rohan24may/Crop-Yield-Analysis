# %%
import pandas as pd
import numpy as np
from pathlib import Path

# Load the original dataset
DATA_PATH = Path("../data/crop_yield.csv")
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# %%
# Remove leading and trailing whitespace from text columns
text_columns = df.select_dtypes(include=["str", "object"]).columns

for col in text_columns:
    df[col] = df[col].str.strip()

print("Text labels cleaned.")
print("Columns:", df.columns.tolist())

# %%
for col in text_columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

# %%
# Check missing values and duplicate records
print("Missing values:")
print(df.isnull().sum())

print("\nExact duplicate rows:", df.duplicated().sum())

# Check numerical columns for negative values
numeric_cols = df.select_dtypes(include="number").columns

print("\nNegative values per numerical column:")
print((df[numeric_cols] < 0).sum())

# Check zero values in important columns
for col in ["Area", "Production", "Annual_Rainfall",
            "Fertilizer", "Pesticide", "Yield"]:
    print(f"{col}: {(df[col] == 0).sum()} zero values")

# %%
# Check the relationship between zero production and zero yield
zero_check = df[
    (df["Production"] == 0) | (df["Yield"] == 0)
]

print("Records with zero production or yield:", len(zero_check))
print("\nProduction vs Yield zero counts:")
print(pd.crosstab(
    df["Production"] == 0,
    df["Yield"] == 0,
    rownames=["Production is zero"],
    colnames=["Yield is zero"]
))

# %% [markdown]
# ## Data Validation Summary
# 
# - Total records: 19,689
# - Total columns: 10
# - Missing values: None found
# - Exact duplicate rows: None found
# - Negative numerical values: None found
# - Zero production records: 112
# - Zero yield records: 112
# - Records with both production and yield equal to zero: 108
# - Records with only one of production or yield equal to zero: 8
# 
# ### Decisions
# - Whitespace was removed from text labels.
# - No rows were deleted during initial validation.
# - Zero values and yield discrepancies remain flagged for further interpretation.
# - The recorded yield definition and units should be verified before final model training.

# %%
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# %%
# Select input features
features = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

# Create features (X) and target (y)
X = df[features]
y = df["Yield"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nInput features:")
print(X.head())

print("\nTarget values:")
print(y.head())

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Separate categorical and numerical features
categorical_features = ["Crop", "Season", "State"]

numerical_features = [
    "Crop_Year",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

# Preprocess categorical and numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to test data
X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create and train the model
model = LinearRegression()
model.fit(X_train_processed, y_train)

# Predict yield on unseen test data
y_pred = model.predict(X_test_processed)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results")
print("-------------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Simple baseline: always predict the training-set mean
dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit(X_train_processed, y_train)

dummy_pred = dummy_model.predict(X_test_processed)

print("Mean Baseline")
print("MAE:", round(mean_absolute_error(y_test, dummy_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, dummy_pred)), 4))
print("R²:", round(r2_score(y_test, dummy_pred), 4))

print("\nLinear Regression")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("R²:", round(r2_score(y_test, y_pred), 4))

# %%
# Create a table of actual and predicted yields
results = X_test.copy()

results["Actual_Yield"] = y_test
results["Predicted_Yield"] = y_pred

# Calculate absolute prediction error
results["Absolute_Error"] = abs(
    results["Actual_Yield"] - results["Predicted_Yield"]
)

# Display the 10 largest errors
largest_errors = results.sort_values(
    by="Absolute_Error",
    ascending=False
).head(10)

print(largest_errors[
    ["Crop", "Crop_Year", "State",
     "Actual_Yield", "Predicted_Yield", "Absolute_Error"]
])

# %%
# Calculate absolute errors for Coconut and other crops
results["Crop_Group"] = np.where(
    results["Crop"] == "Coconut",
    "Coconut",
    "Other crops"
)

error_summary = results.groupby("Crop_Group").agg(
    Records=("Absolute_Error", "size"),
    MAE=("Absolute_Error", "mean"),
    Median_Error=("Absolute_Error", "median"),
    Maximum_Error=("Absolute_Error", "max")
)

print(error_summary.round(3))

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

for group in ["Coconut", "Other crops"]:
    group_data = results[results["Crop_Group"] == group]

    actual = group_data["Actual_Yield"]
    predicted = group_data["Predicted_Yield"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(f"\n{group}")
    print(f"Records: {len(group_data)}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

# %%
coconut_zero = df[
    (df["Crop"] == "Coconut") &
    (df["Yield"] == 0)
]

print("Zero-yield Coconut records:", len(coconut_zero))

print(
    coconut_zero[
        ["Crop_Year", "Season", "State",
         "Area", "Production", "Yield"]
    ].to_string(index=False)
)

# %%
chhattisgarh_coconut = df[
    (df["Crop"] == "Coconut") &
    (df["State"] == "Chhattisgarh")
].sort_values("Crop_Year")

print("Total records:", len(chhattisgarh_coconut))

print(
    chhattisgarh_coconut[
        ["Crop_Year", "Season", "Area", "Production", "Yield"]
    ].to_string(index=False)
)

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train_processed, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test_processed)

# Evaluate performance
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest Results")
print("---------------------")
print(f"MAE:  {rf_mae:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R²:   {rf_r2:.4f}")

# %%
rf_results = X_test.copy()

rf_results["Actual_Yield"] = y_test
rf_results["Predicted_Yield"] = rf_pred

rf_results["Absolute_Error"] = abs(
    rf_results["Actual_Yield"] -
    rf_results["Predicted_Yield"]
)

rf_results["Crop_Group"] = np.where(
    rf_results["Crop"] == "Coconut",
    "Coconut",
    "Other crops"
)

print(
    rf_results.groupby("Crop_Group").agg(
        Records=("Absolute_Error", "size"),
        MAE=("Absolute_Error", "mean"),
        Median_Error=("Absolute_Error", "median"),
        Maximum_Error=("Absolute_Error", "max")
    ).round(3)
)

# %%
# Check how many records are available for each year
print(
    df.groupby("Crop_Year").size().to_string()
)

# %%
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Split chronologically
train_mask = df["Crop_Year"] <= 2017
test_mask = df["Crop_Year"].between(2018, 2019)

X_train_time = df.loc[train_mask, features]
y_train_time = df.loc[train_mask, "Yield"]

X_test_time = df.loc[test_mask, features]
y_test_time = df.loc[test_mask, "Yield"]

print("Training years:", X_train_time["Crop_Year"].min(), "-", X_train_time["Crop_Year"].max())
print("Testing years:", X_test_time["Crop_Year"].min(), "-", X_test_time["Crop_Year"].max())
print("Training records:", len(X_train_time))
print("Testing records:", len(X_test_time))

# Fit a fresh preprocessor using training years only
time_preprocessor = clone(preprocessor)

X_train_time_processed = time_preprocessor.fit_transform(X_train_time)
X_test_time_processed = time_preprocessor.transform(X_test_time)

# Train a fresh Random Forest
time_rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

time_rf_model.fit(X_train_time_processed, y_train_time)
time_rf_pred = time_rf_model.predict(X_test_time_processed)

print("\nChronological Random Forest Results")
print("MAE:", round(mean_absolute_error(y_test_time, time_rf_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test_time, time_rf_pred)), 4))
print("R²:", round(r2_score(y_test_time, time_rf_pred), 4))

# %%
# Analyze errors on the chronological test set
time_results = X_test_time.copy()

time_results["Actual_Yield"] = y_test_time
time_results["Predicted_Yield"] = time_rf_pred

time_results["Absolute_Error"] = abs(
    time_results["Actual_Yield"]
    - time_results["Predicted_Yield"]
)

crop_errors = time_results.groupby("Crop").agg(
    Records=("Absolute_Error", "size"),
    MAE=("Absolute_Error", "mean"),
    Median_Error=("Absolute_Error", "median"),
    Maximum_Error=("Absolute_Error", "max")
)

# Display crops with the highest MAE
print(
    crop_errors.sort_values(
        "MAE", ascending=False
    ).head(10).round(3)
)

# %%
# Summarize the agricultural input features
print(
    df[["Fertilizer", "Pesticide", "Area", "Production"]]
    .describe()
    .round(2)
)

# %%
largest_fertilizer = df.nlargest(10, "Fertilizer")

print(
    largest_fertilizer[
        ["Crop", "Crop_Year", "Season", "State",
         "Area", "Fertilizer", "Pesticide", "Yield"]
    ].to_string(index=False)
)

# %%
# Calculate fertilizer per unit of recorded area
df["Fertilizer_per_Area"] = (
    df["Fertilizer"] / df["Area"]
)

print(
    df.nlargest(10, "Fertilizer_per_Area")[
        ["Crop", "Crop_Year", "State",
         "Area", "Fertilizer", "Fertilizer_per_Area", "Yield"]
    ].to_string(index=False)
)

# %%
# Inspect fertilizer-to-area ratios for Uttarakhand in 2020
subset = df[
    (df["State"] == "Uttarakhand") &
    (df["Crop_Year"] == 2020)
].copy()

subset["Fertilizer_per_Area"] = (
    subset["Fertilizer"] / subset["Area"]
)

print("Records:", len(subset))
print(
    "Unique ratios:",
    subset["Fertilizer_per_Area"].round(2).nunique()
)
print(
    "Ratios:",
    sorted(subset["Fertilizer_per_Area"].round(2).unique())
)

# %%
# Check fertilizer and pesticide ratios for Uttarakhand in 2020
subset = df[
    (df["State"] == "Uttarakhand") &
    (df["Crop_Year"] == 2020)
].copy()

subset["Fertilizer_per_Area"] = (
    subset["Fertilizer"] / subset["Area"]
)

subset["Pesticide_per_Area"] = (
    subset["Pesticide"] / subset["Area"]
)

print("Unique fertilizer ratios:",
      subset["Fertilizer_per_Area"].round(2).nunique())

print("Unique pesticide ratios:",
      subset["Pesticide_per_Area"].round(2).nunique())

print("\nPesticide ratios:")
print(sorted(subset["Pesticide_per_Area"].round(4).unique()))

# %%

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Model A: all seven current features
features_a = [
    "Crop", "Crop_Year", "Season", "State",
    "Annual_Rainfall", "Fertilizer", "Pesticide"
]

# Model B: remove fertilizer and pesticide
features_b = [
    "Crop", "Crop_Year", "Season", "State",
    "Annual_Rainfall"
]

train_mask = df["Crop_Year"] <= 2017
test_mask = df["Crop_Year"].between(2018, 2019)

y_train_eval = df.loc[train_mask, "Yield"]
y_test_eval = df.loc[test_mask, "Yield"]

def evaluate_features(feature_list):
    X_train_eval = df.loc[train_mask, feature_list]
    X_test_eval = df.loc[test_mask, feature_list]

    prep = clone(preprocessor)
    prep.set_params(
        categorical__categories="auto"
    )

    # Reuse only the relevant preprocessing columns
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline

    categorical = [
        col for col in ["Crop", "Season", "State"]
        if col in feature_list
    ]
    numerical = [
        col for col in feature_list
        if col not in categorical
    ]

    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numerical", StandardScaler(), numerical)
    ])

    X_train_ready = prep.fit_transform(X_train_eval)
    X_test_ready = prep.transform(X_test_eval)

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_ready, y_train_eval)
    predictions = rf.predict(X_test_ready)

    return {
        "MAE": mean_absolute_error(y_test_eval, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test_eval, predictions)),
        "R2": r2_score(y_test_eval, predictions)
    }

result_a = evaluate_features(features_a)
result_b = evaluate_features(features_b)

print("Model A — All seven features")
print({k: round(v, 4) for k, v in result_a.items()})

print("\nModel B — Without fertilizer and pesticide")
print({k: round(v, 4) for k, v in result_b.items()})


# %%

features = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

target = "Yield"

print("Selected features:", features)
print("Target:", target)

print("\nFeature columns available:")
print(df[features].columns.tolist())

print("\nTarget available:", target in df.columns)
print("Missing values in selected features:")
print(df[features].isnull().sum())


# %%

from sklearn.model_selection import train_test_split

# Train on earlier years; test on later years
train_mask = df["Crop_Year"] <= 2017
test_mask = df["Crop_Year"].between(2018, 2019)

X_train = df.loc[train_mask, features]
X_test = df.loc[test_mask, features]

y_train = df.loc[train_mask, target]
y_test = df.loc[test_mask, target]

print("Training period: 1997–2017")
print("Testing period: 2018–2019")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# %%

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

categorical_features = ["Crop", "Season", "State"]

numerical_features = [
    "Crop_Year",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    (
        "numerical",
        StandardScaler(),
        numerical_features
    )
])

final_model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("Final pipeline created successfully!")


# %%

# Train the complete pipeline
final_model.fit(X_train, y_train)

print("Model training completed!")


# %%

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict yield for the test data
y_pred = final_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Final Model Evaluation (2018–2019)")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# %%

import pandas as pd
import numpy as np

# Compare actual and predicted yields
results = X_test.copy()
results["Actual_Yield"] = y_test
results["Predicted_Yield"] = y_pred

results["Absolute_Error"] = np.abs(
    results["Actual_Yield"] - results["Predicted_Yield"]
)

print("Sample predictions:")
display(results[
    ["Crop", "Crop_Year", "Actual_Yield",
     "Predicted_Yield", "Absolute_Error"]
].head(10))


# %%

largest_errors = results.sort_values(
    by="Absolute_Error",
    ascending=False
)

display(
    largest_errors[
        [
            "Crop",
            "Crop_Year",
            "Actual_Yield",
            "Predicted_Yield",
            "Absolute_Error"
        ]
    ].head(10)
)


# %%

crop_errors = (
    results.groupby("Crop")
    .agg(
        Records=("Absolute_Error", "size"),
        MAE=("Absolute_Error", "mean"),
        Median_Error=("Absolute_Error", "median"),
        Max_Error=("Absolute_Error", "max")
    )
    .sort_values("MAE", ascending=False)
)

display(crop_errors.head(10).round(4))


# %%

results["Absolute_Percentage_Error"] = np.where(
    results["Actual_Yield"] != 0,
    results["Absolute_Error"] / results["Actual_Yield"] * 100,
    np.nan
)

relative_errors = (
    results.groupby("Crop")
    .agg(
        Records=("Actual_Yield", "size"),
        Nonzero_Yield_Records=("Actual_Yield", lambda s: (s != 0).sum()),
        MAPE=("Absolute_Percentage_Error", "mean")
    )
    .sort_values("MAPE", ascending=False)
)

display(relative_errors.head(10).round(2))



# %%

crops_to_inspect = [
    "other oilseeds",
    "Cashewnut",
    "Castor seed",
    "Sesamum",
    "Niger seed"
]

inspection = results[
    results["Crop"].isin(crops_to_inspect)
][
    [
        "Crop",
        "Crop_Year",
        "Actual_Yield",
        "Predicted_Yield",
        "Absolute_Error",
        "Absolute_Percentage_Error"
    ]
].sort_values(
    ["Crop", "Absolute_Percentage_Error"],
    ascending=[True, False]
)

display(inspection.round(4))



# %%

wape_by_crop = (
    results.groupby("Crop")
    .apply(
        lambda group: (
            group["Absolute_Error"].sum()
            / group["Actual_Yield"].abs().sum()
            * 100
            if group["Actual_Yield"].abs().sum() != 0
            else np.nan
        ),
        include_groups=False
    )
    .sort_values(ascending=False)
    .rename("WAPE")
    .to_frame()
)

display(wape_by_crop.head(10).round(2))


# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    results["Actual_Yield"],
    results["Predicted_Yield"],
    alpha=0.6
)

# Ideal predictions lie on this line
max_value = max(
    results["Actual_Yield"].max(),
    results["Predicted_Yield"].max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield (2018–2019)")
plt.tight_layout()
plt.show()


# %%

coconut_results = results[results["Crop"] == "Coconut"]

plt.figure(figsize=(8, 6))

plt.scatter(
    coconut_results["Actual_Yield"],
    coconut_results["Predicted_Yield"],
    alpha=0.7
)

min_value = min(
    coconut_results["Actual_Yield"].min(),
    coconut_results["Predicted_Yield"].min()
)

max_value = max(
    coconut_results["Actual_Yield"].max(),
    coconut_results["Predicted_Yield"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Coconut Yield")
plt.ylabel("Predicted Coconut Yield")
plt.title("Actual vs Predicted Coconut Yield (2018–2019)")
plt.tight_layout()
plt.show()


# %%

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Transform the target for training
y_train_log = np.log1p(y_train)

# Reuse the preprocessing setup, but train a new model
log_model = Pipeline([
    ("preprocessing", clone(preprocessor)),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

log_model.fit(X_train, y_train_log)

# Predict in log space, then convert back to original yield units
y_pred_log = log_model.predict(X_test)
y_pred_log = np.maximum(y_pred_log, 0)
y_pred_log_original = np.expm1(y_pred_log)

print("Log-transformed target model trained!")


# %%

# Evaluate predictions in the original yield units
log_mae = mean_absolute_error(y_test, y_pred_log_original)
log_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred_log_original)
)
log_r2 = r2_score(y_test, y_pred_log_original)

print("Log-Transformed Model — Test Results")
print("-" * 40)
print(f"MAE  : {log_mae:.4f}")
print(f"RMSE : {log_rmse:.4f}")
print(f"R²   : {log_r2:.4f}")

print("\nBaseline Model — Test Results")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# %%

# Baseline predictions vs. log-transformed predictions for Coconut
coconut_mask = X_test["Crop"] == "Coconut"

coconut_comparison = pd.DataFrame({
    "Actual_Yield": y_test.loc[coconut_mask],
    "Baseline_Prediction": final_model.predict(
        X_test.loc[coconut_mask]
    ),
    "Log_Model_Prediction": y_pred_log_original[
        coconut_mask.to_numpy()
    ]
})

coconut_comparison["Baseline_Error"] = abs(
    coconut_comparison["Actual_Yield"]
    - coconut_comparison["Baseline_Prediction"]
)

coconut_comparison["Log_Model_Error"] = abs(
    coconut_comparison["Actual_Yield"]
    - coconut_comparison["Log_Model_Prediction"]
)

display(coconut_comparison.round(4))

print("\nCoconut MAE comparison")
print("Baseline:", round(coconut_comparison["Baseline_Error"].mean(), 4))
print("Log model:", round(coconut_comparison["Log_Model_Error"].mean(), 4))


# %%

print("Coconut MAE comparison")
print("-" * 35)

print(
    "Baseline MAE:",
    round(coconut_comparison["Baseline_Error"].mean(), 4)
)

print(
    "Log model MAE:",
    round(coconut_comparison["Log_Model_Error"].mean(), 4)
)

improvement = (
    coconut_comparison["Baseline_Error"].mean()
    - coconut_comparison["Log_Model_Error"].mean()
)

print("MAE improvement:", round(improvement, 4))



# %%

# Build predictions for both models on the same test set
baseline_results = results.copy()

baseline_results["Baseline_Prediction"] = final_model.predict(X_test)
baseline_results["Log_Prediction"] = y_pred_log_original

baseline_results["Baseline_Error"] = abs(
    y_test - baseline_results["Baseline_Prediction"]
)

baseline_results["Log_Error"] = abs(
    y_test - baseline_results["Log_Prediction"]
)

crop_comparison = baseline_results.groupby("Crop").agg(
    Records=("Actual_Yield", "size"),
    Baseline_MAE=("Baseline_Error", "mean"),
    Log_MAE=("Log_Error", "mean")
)

crop_comparison["MAE_Improvement"] = (
    crop_comparison["Baseline_MAE"]
    - crop_comparison["Log_MAE"]
)

crop_comparison = crop_comparison.sort_values(
    "MAE_Improvement",
    ascending=False
)

display(crop_comparison.head(10).round(4))


# %%

improved_crops = (
    crop_comparison["MAE_Improvement"] > 0
).sum()

worsened_crops = (
    crop_comparison["MAE_Improvement"] < 0
).sum()

unchanged_crops = (
    crop_comparison["MAE_Improvement"] == 0
).sum()

print("Total crops evaluated:", len(crop_comparison))
print("Crops improved:", improved_crops)
print("Crops worsened:", worsened_crops)
print("Crops unchanged:", unchanged_crops)

print("\nLargest MAE regressions:")
display(
    crop_comparison.sort_values("MAE_Improvement").head(5).round(4)
)


# %%

model_comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Baseline": [mae, rmse, r2],
    "Log_Transformed": [log_mae, log_rmse, log_r2]
})

model_comparison["Change"] = (
    model_comparison["Log_Transformed"]
    - model_comparison["Baseline"]
)

display(model_comparison.round(4))


# %%

import sys
from pathlib import Path

# Make the src folder importable
sys.path.insert(0, str(Path("../src").resolve()))

from crop_yield_model import train_model, predict_yield, FEATURES

# Use the same chronological split
train_mask = df["Crop_Year"] <= 2017
test_mask = df["Crop_Year"].between(2018, 2019)

train_df = df.loc[train_mask]
test_df = df.loc[test_mask]

X_test_final = test_df[FEATURES]
y_test_final = test_df["Yield"]

# Train the reusable model on earlier years only
reusable_model = train_model(train_df)

# Predict on later years
y_pred_final = predict_yield(reusable_model, X_test_final)

print("Reusable model trained and predictions generated!")
print("Training records:", len(train_df))
print("Testing records:", len(test_df))


# %%

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_final = mean_absolute_error(y_test_final, y_pred_final)
rmse_final = np.sqrt(
    mean_squared_error(y_test_final, y_pred_final)
)
r2_final = r2_score(y_test_final, y_pred_final)

print("Reusable Model Evaluation (2018–2019)")
print("-" * 40)
print(f"MAE  : {mae_final:.4f}")
print(f"RMSE : {rmse_final:.4f}")
print(f"R²   : {r2_final:.4f}")

print("\nPrevious log-model results")
print(f"MAE  : {log_mae:.4f}")
print(f"RMSE : {log_rmse:.4f}")
print(f"R²   : {log_r2:.4f}")


# %%

import joblib
from pathlib import Path

# Save the trained model
model_path = Path("../models/crop_yield_model.joblib")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(reusable_model, model_path)

print("Model saved successfully!")
print("Saved to:", model_path.resolve())


# %%

import joblib
import numpy as np

# Load the saved model
loaded_model = joblib.load(model_path)

# Predict using the loaded model
loaded_predictions = predict_yield(
    loaded_model,
    X_test_final
)

# Compare with predictions before saving
max_difference = np.max(
    np.abs(y_pred_final - loaded_predictions)
)

print("Saved model loaded successfully!")
print("Maximum prediction difference:", max_difference)
print("Predictions match:", np.allclose(
    y_pred_final,
    loaded_predictions
))


# %%



