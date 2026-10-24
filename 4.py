# %%
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = (
    Path.cwd().parent
    if Path.cwd().name == "notebooks"
    else Path.cwd()
)

DATA_PATH = PROJECT_ROOT / "data" / "crop_yield.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Year range:", df["Crop_Year"].min(), "-", df["Crop_Year"].max())
print("Number of crops:", df["Crop"].nunique())
print("Missing values:", df.isnull().sum().sum())

df.head()


# %%
# Features used by our current V1 model
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

print("Input features:")
for feature in features:
    print("-", feature)

print("\nTarget:", target)

# %%
df[["Area", "Production", "Yield"]].describe().round(2)

# %%
# Compare recorded yield with Production / Area
valid = df["Area"] > 0

calculated_yield = (
    df.loc[valid, "Production"] / df.loc[valid, "Area"]
)

comparison = pd.DataFrame({
    "Recorded_Yield": df.loc[valid, "Yield"],
    "Calculated_Yield": calculated_yield
})

comparison["Absolute_Difference"] = abs(
    comparison["Recorded_Yield"] - comparison["Calculated_Yield"]
)

comparison.describe().round(4)

# %%
nonzero = comparison[comparison["Recorded_Yield"] > 0].copy()

nonzero["Percentage_Difference"] = (
    nonzero["Absolute_Difference"]
    / nonzero["Recorded_Yield"]
    * 100
)

for threshold in [10, 25, 50, 100]:
    percentage = (
        nonzero["Percentage_Difference"].gt(threshold).mean() * 100
    )
    print(f"Difference greater than {threshold}%: {percentage:.2f}%")

# %%
comparison.sort_values(
    "Absolute_Difference",
    ascending=False
).head(10)

# %%
feature_review = pd.DataFrame({
    "Feature": [
        "Crop",
        "Crop_Year",
        "Season",
        "State",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide"
    ],
    "Available_before_season": [
        "Usually yes",
        "Yes",
        "Yes",
        "Yes",
        "Not necessarily",
        "Depends on scenario",
        "Depends on scenario"
    ],
    "Concern": [
        "User must select the crop",
        "Future years are outside our training period",
        "User must select the season",
        "User must select the state",
        "Actual annual rainfall may not be known yet",
        "Future input quantities may not be decided",
        "Future input quantities may not be decided"
    ]
})

feature_review

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Define features and target
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

# Chronological split
train_df = df[df["Crop_Year"] <= 2017].copy()
test_df = df[df["Crop_Year"].isin([2018, 2019])].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# %%
# Define feature groups
categorical_features = ["Crop", "Season", "State"]

numerical_features = [
    "Crop_Year",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

# Preprocessing
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

# Random Forest pipeline
baseline_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
])

# Train using log-transformed yield
baseline_model.fit(X_train, np.log1p(y_train))

# Predict and convert back to original scale
y_pred = np.expm1(baseline_model.predict(X_test))
y_pred = np.maximum(y_pred, 0)

# Evaluate
print("Baseline MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("Baseline RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("Baseline R²:", round(r2_score(y_test, y_pred), 4))

# %%
from sklearn.ensemble import GradientBoostingRegressor

# Create the Gradient Boosting pipeline
gb_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    )
])

# Train on log-transformed yield
gb_model.fit(X_train, np.log1p(y_train))

# Predict on the same test set
gb_pred = np.expm1(gb_model.predict(X_test))
gb_pred = np.maximum(gb_pred, 0)

# Evaluate
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print("Gradient Boosting MAE:", round(gb_mae, 4))
print("Gradient Boosting RMSE:", round(gb_rmse, 4))
print("Gradient Boosting R²:", round(gb_r2, 4))

# %%
comparison = pd.DataFrame({
    "Model": ["Random Forest (Baseline)", "Gradient Boosting"],
    "MAE": [10.6382, gb_mae],
    "RMSE": [150.3480, gb_rmse],
    "R2 Score": [0.9692, gb_r2]
})

display(comparison.round(4))

# %% [markdown]
# ### Model Comparison: Random Forest vs Gradient Boosting
# 
# Two regression models were evaluated using the same features and chronological train-test split.
# 
# - **Random Forest:** MAE = 10.6382, RMSE = 150.3480, R² = 0.9692.
# - **Gradient Boosting:** MAE = 38.1818, RMSE = 475.4258, R² = 0.6916.
# 
# Random Forest performed substantially better across all three evaluation metrics. Therefore, the log-transformed Random Forest remains the leading candidate for the scenario-based crop yield estimation project.
# 
# Gradient Boosting was not selected because its prediction errors were considerably higher on the evaluation dataset.

# %%
from sklearn.ensemble import RandomForestRegressor

tuned_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

tuned_model.fit(X_train, np.log1p(y_train))

tuned_pred = np.expm1(tuned_model.predict(X_test))
tuned_pred = np.maximum(tuned_pred, 0)

print("Tuned Random Forest MAE:", round(mean_absolute_error(y_test, tuned_pred), 4))

print("Tuned Random Forest RMSE:", round(np.sqrt(mean_squared_error(y_test, tuned_pred)), 4))

print("Tuned Random Forest R²:", round(r2_score(y_test, tuned_pred), 4))

# %%
comparison = pd.DataFrame({
    "Model": [
        "Baseline Random Forest",
        "Gradient Boosting",
        "Tuned Random Forest"
    ],
    "MAE": [10.6382, gb_mae, mean_absolute_error(y_test, tuned_pred)],
    "RMSE": [150.3480, gb_rmse, np.sqrt(mean_squared_error(y_test, tuned_pred))],
    "R2 Score": [0.9692, gb_r2, r2_score(y_test, tuned_pred)]
})

display(comparison.round(4))

# %%
import pandas as pd

feature_names = baseline_model.named_steps["preprocessor"].get_feature_names_out()
importances = baseline_model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

display(importance_df.head(15).round(4))

# %%
def get_original_feature(feature):
    feature = feature.replace("categorical__", "")
    feature = feature.replace("numerical__", "")

    for name in [
        "Annual_Rainfall", "Crop_Year", "Fertilizer", "Pesticide",
        "Crop", "Season", "State"
    ]:
        if feature == name or feature.startswith(name + "_"):
            return name

    return feature

importance_df["Original_Feature"] = importance_df["Feature"].apply(
    get_original_feature
)

grouped_importance = (
    importance_df.groupby("Original_Feature")["Importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

display(grouped_importance.round(4))

# %%
error_df = pd.DataFrame({
    "Crop": X_test["Crop"].values,
    "Actual_Yield": y_test.values,
    "Predicted_Yield": baseline_model.predict(X_test)
})

# Convert predictions from log scale back to original scale
error_df["Predicted_Yield"] = np.expm1(error_df["Predicted_Yield"])
error_df["Predicted_Yield"] = error_df["Predicted_Yield"].clip(lower=0)

error_df["Absolute_Error"] = abs(
    error_df["Actual_Yield"] - error_df["Predicted_Yield"]
)

crop_performance = (
    error_df.groupby("Crop")
    .agg(
        Records=("Actual_Yield", "count"),
        MAE=("Absolute_Error", "mean"),
        Mean_Actual_Yield=("Actual_Yield", "mean")
    )
    .sort_values("MAE", ascending=False)
)

display(crop_performance.head(10).round(4))

# %%
crop_performance = (
    error_df.groupby("Crop")
    .agg(
        Records=("Actual_Yield", "count"),
        Total_Absolute_Error=("Absolute_Error", "sum"),
        Total_Actual_Yield=("Actual_Yield", "sum")
    )
)

crop_performance["WAPE_Percent"] = (
    crop_performance["Total_Absolute_Error"]
    / crop_performance["Total_Actual_Yield"].replace(0, np.nan)
) * 100

crop_performance = crop_performance.sort_values(
    "WAPE_Percent", ascending=False
)

display(crop_performance.head(10).round(2))

# %%
from sklearn.ensemble import RandomForestRegressor

simple_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

simple_model.fit(X_train, np.log1p(y_train))

simple_pred = np.expm1(simple_model.predict(X_test))
simple_pred = np.maximum(simple_pred, 0)

print("Simpler Random Forest MAE:", round(mean_absolute_error(y_test, simple_pred), 4))
print("Simpler Random Forest RMSE:", round(np.sqrt(mean_squared_error(y_test, simple_pred)), 4))
print("Simpler Random Forest R²:", round(r2_score(y_test, simple_pred), 4))

# %% [markdown]
# ### Model Improvement Experiments
# 
# Three alternatives were evaluated against the baseline log-transformed Random Forest using the same features and chronological evaluation split.
# 
# 1. **Gradient Boosting:** Performed substantially worse than the baseline.
# 2. **Tuned Random Forest:** Increasing the number of trees to 200 and limiting tree depth to 20 did not improve performance.
# 3. **Simpler Random Forest:** Limiting tree depth to 12 also did not improve performance.
# 
# The baseline Random Forest achieved MAE = 10.6382, RMSE = 150.3480, and R² = 0.9692, outperforming all three alternatives on the evaluation set.
# 
# Therefore, the baseline remains the leading model. Feature importance and crop-level error analysis also revealed that crop identity strongly influences predictions and that performance varies considerably across crops.
# 
# **Limitation:** The 2018–2019 evaluation set has been reused for model comparisons and should not be treated as a completely untouched final test set.

# %%
import matplotlib.pyplot as plt

# Generate baseline predictions again to avoid variable-name issues
baseline_pred = np.expm1(baseline_model.predict(X_test))
baseline_pred = np.maximum(baseline_pred, 0)

plt.figure(figsize=(8, 6))

plt.scatter(y_test, baseline_pred, alpha=0.4)

max_value = max(y_test.max(), baseline_pred.max())
plt.plot([0, max_value], [0, max_value], linestyle="--")

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.tight_layout()
plt.show()

# %%
ordinary = error_df[error_df["Actual_Yield"] < 100]

plt.figure(figsize=(8, 6))

plt.scatter(
    ordinary["Actual_Yield"],
    ordinary["Predicted_Yield"],
    alpha=0.4
)

plt.plot([0, 100], [0, 100], linestyle="--")

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield (Actual Yield < 100)")
plt.tight_layout()
plt.show()

# %%
error_df.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)

# %%
error_df[error_df["Crop"] == "Coconut"][
    ["Actual_Yield", "Predicted_Yield", "Absolute_Error"]
].describe().round(2)

# %%
print("Available crops in error_df:")
print(error_df["Crop"].unique())

print("\nCoconut rows:")
print((error_df["Crop"].astype(str).str.strip().str.lower() == "coconut").sum())

# %%
coconut_df = error_df[
    error_df["Crop"].str.strip() == "Coconut"
]

coconut_df[
    ["Actual_Yield", "Predicted_Yield", "Absolute_Error"]
].describe().round(2)

# %%
coconut_df = coconut_df.copy()

coconut_df["Prediction_Error"] = (
    coconut_df["Predicted_Yield"] - coconut_df["Actual_Yield"]
)

coconut_df[
    ["Actual_Yield", "Predicted_Yield", "Prediction_Error"]
].sort_values("Actual_Yield")

# %%
coconut_data = df[
    df["Crop"].str.strip() == "Coconut"
]

print("Total Coconut records:", len(coconut_data))
print(
    "Coconut training records:",
    (coconut_data["Crop_Year"] <= 2017).sum()
)
print(
    "Coconut test records:",
    coconut_data["Crop_Year"].isin([2018, 2019]).sum()
)

# %%
coconut_train = df[
    (df["Crop"].str.strip() == "Coconut") &
    (df["Crop_Year"] <= 2017)
]

print("Coconut training yield range:")
print(coconut_train["Yield"].describe().round(2))

print("\nCoconut test yield range:")
print(
    df[
        (df["Crop"].str.strip() == "Coconut") &
        (df["Crop_Year"].isin([2018, 2019]))
    ]["Yield"].describe().round(2)
)

# %%
coconut_df[
    ["Actual_Yield", "Predicted_Yield", "Prediction_Error"]
].sort_values(
    "Prediction_Error",
    key=abs,
    ascending=False
).head(5)

# %%
coconut_test = test_df[
    test_df["Crop"].str.strip() == "Coconut"
].copy()

coconut_test["Actual_Yield"] = y_test.loc[
    coconut_test.index
]

coconut_test["Predicted_Yield"] = np.expm1(
    baseline_model.predict(coconut_test[features])
).clip(0)

coconut_test["Absolute_Error"] = abs(
    coconut_test["Actual_Yield"]
    - coconut_test["Predicted_Yield"]
)

coconut_test.groupby("State").agg(
    Records=("Absolute_Error", "count"),
    MAE=("Absolute_Error", "mean")
).sort_values("MAE", ascending=False).round(2)

# %%
raw_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

raw_model.fit(X_train, y_train)

raw_pred = np.maximum(raw_model.predict(X_test), 0)

print("Raw-target Random Forest")
print("MAE:", round(mean_absolute_error(y_test, raw_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, raw_pred)), 4))
print("R²:", round(r2_score(y_test, raw_pred), 4))

# %%
raw_coconut_pred = raw_model.predict(
    coconut_test[features]
).clip(0)

log_coconut_pred = np.expm1(
    baseline_model.predict(coconut_test[features])
).clip(0)

print(
    "Raw-target Coconut MAE:",
    round(mean_absolute_error(
        coconut_test["Actual_Yield"],
        raw_coconut_pred
    ), 2)
)

print(
    "Log-target Coconut MAE:",
    round(mean_absolute_error(
        coconut_test["Actual_Yield"],
        log_coconut_pred
    ), 2)
)

# %%
df[["Area", "Production", "Yield"]].corr().round(3)

# %%
features_with_area = features + ["Area"]

X_train_area = train_df[features_with_area]
X_test_area = test_df[features_with_area]

area_preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features),
    ("numerical", StandardScaler(),
     numerical_features + ["Area"])
])

area_model = Pipeline([
    ("preprocessor", area_preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

area_model.fit(X_train_area, np.log1p(y_train))

area_pred = np.expm1(area_model.predict(X_test_area))
area_pred = np.maximum(area_pred, 0)

print("With Area")
print("MAE:", round(mean_absolute_error(y_test, area_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, area_pred)), 4))
print("R²:", round(r2_score(y_test, area_pred), 4))

# %%
crop_errors = error_df.copy()

crop_errors["Crop"] = crop_errors["Crop"].str.strip()

crop_error_summary = (
    crop_errors.groupby("Crop")
    .agg(
        Samples=("Absolute_Error", "size"),
        MAE=("Absolute_Error", "mean")
    )
    .sort_values("MAE", ascending=False)
)

crop_error_summary.head(10).round(2)

# %%
coconut_errors = error_df[
    error_df["Crop"].str.strip() == "Coconut"
].copy()

coconut_errors["Error"] = (
    coconut_errors["Predicted_Yield"]
    - coconut_errors["Actual_Yield"]
)

coconut_errors[
    ["Actual_Yield", "Predicted_Yield", "Error", "Absolute_Error"]
].sort_values("Absolute_Error", ascending=False).round(2)

# %%
coconut_train = train_df[
    train_df["Crop"].str.strip() == "Coconut"
]

coconut_test_data = test_df[
    test_df["Crop"].str.strip() == "Coconut"
]

print("Coconut training records:", len(coconut_train))
print("Coconut test records:", len(coconut_test_data))

print("\nTraining yield statistics:")
print(coconut_train["Yield"].describe().round(2))

print("\nTest yield statistics:")
print(coconut_test_data["Yield"].describe().round(2))

# %%
feature_importance = pd.Series(
    baseline_model.named_steps["regressor"].feature_importances_,
    index=baseline_model.named_steps["preprocessor"].get_feature_names_out()
).sort_values(ascending=False)

feature_importance.head(15).round(4)

# %%
coconut_features = features

coconut_train = train_df[
    train_df["Crop"].str.strip() == "Coconut"
]

coconut_test_data = test_df[
    test_df["Crop"].str.strip() == "Coconut"
]

coconut_preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     ["Season", "State"]),
    ("numerical", StandardScaler(),
     ["Crop_Year", "Annual_Rainfall", "Fertilizer", "Pesticide"])
])

coconut_model = Pipeline([
    ("preprocessor", coconut_preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

coconut_model.fit(
    coconut_train[["Season", "State", "Crop_Year",
                   "Annual_Rainfall", "Fertilizer", "Pesticide"]],
    np.log1p(coconut_train["Yield"])
)

coconut_pred = np.expm1(
    coconut_model.predict(
        coconut_test_data[["Season", "State", "Crop_Year",
                           "Annual_Rainfall", "Fertilizer", "Pesticide"]]
    )
).clip(0)

print("Coconut-specific model MAE:",
      round(mean_absolute_error(coconut_test_data["Yield"], coconut_pred), 2))

print("Existing model Coconut MAE:",
      round(mean_absolute_error(
          coconut_test_data["Yield"],
          np.expm1(baseline_model.predict(coconut_test_data[features])).clip(0)
      ), 2))

# %%
# Prepare the untouched 2020 evaluation data
final_test_df = df[df["Crop_Year"] == 2020].copy()

X_final = final_test_df[features]
y_final = final_test_df[target]

print("Final evaluation records:", len(X_final))
print("Evaluation year:", final_test_df["Crop_Year"].unique())

# %%
# Predict on the 2020 evaluation data
final_log_pred = baseline_model.predict(X_final)

final_pred = np.expm1(final_log_pred)
final_pred = np.maximum(final_pred, 0)

# Evaluate performance
print("Final 2020 Evaluation")
print("MAE:", round(mean_absolute_error(y_final, final_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_final, final_pred)), 4))
print("R²:", round(r2_score(y_final, final_pred), 4))


# %%
print("Actual yield statistics:")
print(y_final.describe().round(4))

print("\nCrops in the 2020 evaluation set:")
print(final_test_df["Crop"].value_counts())

print("\nActual vs predicted yields:")
final_results = pd.DataFrame({
    "Crop": final_test_df["Crop"].values,
    "Actual_Yield": y_final.values,
    "Predicted_Yield": final_pred
})

final_results["Absolute_Error"] = abs(
    final_results["Actual_Yield"] - final_results["Predicted_Yield"]
)

display(final_results.round(4))

# %%
print("Last training year:",
      train_df["Crop_Year"].max())

print("2020 included in training:",
      (train_df["Crop_Year"] == 2020).any())

print("Model training records:",
      len(X_train))


