# Version 02 — Baseline Machine Learning Model

## 1. Objective

Build an initial machine-learning model to predict agricultural yield using crop, seasonal, geographical, and agricultural/climatic features.

## 2. Work Completed

- Selected `Yield` as the prediction target.
- Selected seven initial input features:
  - `Crop`
  - `Crop_Year`
  - `Season`
  - `State`
  - `Annual_Rainfall`
  - `Fertilizer`
  - `Pesticide`
- Split the dataset into 80% training data and 20% testing data.
- Applied One-Hot Encoding to categorical features.
- Applied Standard Scaling to numerical features.
- Built a scikit-learn preprocessing pipeline.
- Trained a Linear Regression model.
- Compared its performance with a mean-prediction baseline.

## 3. Dataset Split

- Training records: 15,751
- Testing records: 3,938
- Original input columns: 7
- Processed input features: 95

## 4. Model Evaluation

| Metric | Mean Baseline | Linear Regression |
|---|---:|---:|
| MAE | 149.8645 | 63.8830 |
| RMSE | 895.1205 | 389.6882 |
| R² | Approximately 0.0000 | 0.8105 |

## 5. Findings

- Linear Regression outperformed the mean baseline on all three reported metrics.
- Its MAE was approximately 57.4% lower than the baseline.
- Its RMSE was approximately 56.5% lower than the baseline.
- The difference between MAE and RMSE suggests that some large prediction errors may remain.
- The results are preliminary and apply to the current random test split.

## 6. Limitations

- Yield values have a highly skewed distribution and extreme outliers.
- Yield definitions and measurement units require further verification.
- The random split does not establish how well the model predicts future years.
- Model performance must be checked for large errors and possible data leakage.
- Correlation and predictive performance do not establish causal relationships.

## 7. Next Steps

1. Inspect prediction errors and extreme-yield records.
2. Evaluate whether a year-based test split is more appropriate.
3. Compare additional regression models.
4. Reassess the feature set and preprocessing decisions.
5. Save and document the final model and evaluation results.

## 8. Version Status

**Status:** Initial baseline model trained and evaluated.

**Date:** 15 October 2026
