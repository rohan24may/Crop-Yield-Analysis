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
## 9. Prediction Error Analysis

The baseline model's largest prediction errors were concentrated among Coconut records.

| Metric | Coconut | Other crops |
|---|---:|---:|
| Test records | 34 | 3,904 |
| MAE | 3,612.795 | 32.975 |
| RMSE | 4,162.021 | 48.148 |

### Interpretation

- Coconut predictions are substantially less accurate in absolute terms.
- Coconut's high yield values and variability contribute to the model's large prediction errors.
- Some Coconut records have zero recorded yield, which warrants further data-quality investigation.
- Absolute errors across crops must be interpreted carefully because yield scales differ.

### Next investigation

Inspect the Coconut records with zero yield and verify the dataset's yield definition before changing the target or removing observations.
## 10. Investigation of Zero-Yield Coconut Records

Seven Coconut records from Chhattisgarh have been identified with:

- Positive cultivated area
- Zero recorded production
- Zero recorded yield

These records occur in 2008, 2011, 2012, 2013, 2014, 2015, and 2019. All seven Chhattisgarh Coconut records in the dataset have zero production and zero yield.

### Decision

The records will not be deleted or modified without supporting evidence. Their validity remains uncertain and is a limitation of the dataset.

### Next Step

Continue model development while documenting this uncertainty. Evaluate the model's performance carefully, especially for crops with extreme yield values.

## 11. Random Forest Model Evaluation

A Random Forest Regressor was trained using the same training/test split and preprocessed features as Linear Regression.

### Results

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 63.8830 | 9.5503 |
| RMSE | 389.6882 | 127.3603 |
| R² | 0.8105 | 0.9798 |

### Findings

- Random Forest outperformed Linear Regression on all three evaluation metrics.
- Its MAE was approximately 85.0% lower.
- Its RMSE was approximately 67.3% lower.
- Its R² score was higher on the current random test split.

### Caution

These results are preliminary. Further evaluation is needed to check generalization to future years, performance across individual crops, possible data leakage, and sensitivity to extreme yield values.

**Current best model:** Random Forest Regressor, pending further validation.
## 12. Chronological Model Evaluation

To test generalization to later years, a chronological split was evaluated.

- Training period: 1997–2017
- Testing period: 2018–2019
- Training records: 17,510
- Testing records: 2,142

### Random Forest Results

| Metric | Random split | Chronological split |
|---|---:|---:|
| MAE | 9.5503 | 11.2250 |
| RMSE | 127.3603 | 154.4333 |
| R² | 0.9798 | 0.9675 |

### Crop-Level Error Analysis

Coconut remained the most difficult crop in the chronological test.

| Crop | Test records | MAE |
|---|---:|---:|
| Coconut | 17 | 1,262.048 |
| Banana | 23 | 11.620 |
| Onion | 54 | 9.114 |
| Sugarcane | 52 | 6.255 |
| Maize | 101 | 4.002 |

### Interpretation and Limitations

- Random Forest retained a high R² score on later-year records.
- Errors increased slightly under chronological evaluation.
- Coconut continues to contribute substantial prediction errors.
- Absolute errors vary in meaning across crops with different yield scales.
- Feature availability, yield definitions, and possible data-quality issues still require review.

**Status:** Chronological evaluation completed; feature validity and final model selection remain pending.
## 13. Feature Selection Review

### Prediction Target

`Yield`

### Initial Input Features

- `Crop`
- `Crop_Year`
- `Season`
- `State`
- `Annual_Rainfall`
- `Fertilizer`
- `Pesticide`

### Excluded Features

- `Production`: excluded initially because it may directly reveal information used to calculate yield.
- `Area`: excluded initially while its relationship with the recorded target remains under review.

### Agricultural Input Concern

All 37 records for Uttarakhand in 2020 had identical fertilizer-to-area and pesticide-to-area ratios.

- Fertilizer / Area: 193.61
- Pesticide / Area: 0.37

This pattern may indicate that agricultural input values were allocated using common state-year rates. The dataset methodology has not yet been verified, so this remains a hypothesis.

### Current Decision

Retain the initial seven-feature set provisionally for comparison. Review the validity and availability of each feature before selecting the final model.

The current model results are exploratory and do not establish causal relationships or guaranteed future prediction accuracy.
## 14. Dataset Source Documentation

According to the dataset description, the dataset covers historical crop-yield data from 27 Indian states and 3 Union Territories between 1997 and 2020.

- Total records: 19,689
- Total features: 10
- Crop types: 55
- Original study: Regression using a stacking ensemble model
- Original train/test split: 80% training and 20% testing

### Source Documentation Limitations

The description reviewed does not specify the measurement units of `Yield`, `Fertilizer`, or `Pesticide`. It also does not establish when the agricultural input values become available relative to crop-yield prediction.

These details remain unverified and should be considered when interpreting model performance and selecting features.

### Reference

[Mendeley Data — Crop Yield Dataset](https://data.mendeley.com/datasets/ncw2vbcgnk/2)
### 15. Feature Ablation Experiment

**Objective:** Evaluate whether Fertilizer and Pesticide improve prediction performance on future-year data.

**Evaluation setup:**
- Training period: 1997–2017
- Testing period: 2018–2019
- Model: Random Forest Regressor with 100 trees
- Model A: All seven provisional features
- Model B: Excluded Fertilizer and Pesticide

**Results:**

| Metric | Model A | Model B |
|---|---:|---:|
| MAE | 11.2250 | 11.1736 |
| RMSE | 154.4333 | 163.1045 |
| R² | 0.9675 | 0.9637 |

**Findings:** Removing Fertilizer and Pesticide slightly improved MAE but worsened RMSE and R². Their contribution is therefore mixed rather than uniformly beneficial.

**Decision:** Retain both features provisionally. Before finalizing the model, review their units, reliability, and availability at the intended prediction time. This experiment measures predictive performance, not causality.
### 16. Final Model Pipeline and Evaluation

**Objective:** Build a reusable machine-learning pipeline for historical data-based crop yield prediction.

**Model configuration:**
- Algorithm: Random Forest Regressor
- Number of trees: 100
- Random state: 42
- Features: Crop, Crop_Year, Season, State, Annual_Rainfall, Fertilizer, Pesticide
- Target: Yield

**Evaluation strategy:**
- Training data: 1997–2017 (17,510 records)
- Testing data: 2018–2019 (2,142 records)
- The year 2020 was excluded because only 37 records were available for that year.
- Preprocessing was fitted through the pipeline using training data only.

**Final evaluation results:**

| Metric | Result |
|---|---:|
| MAE | 11.2250 |
| RMSE | 154.4333 |
| R² | 0.9675 |

**Interpretation:** The model achieved an R² of 0.9675 on the chronological test set. However, RMSE is substantially higher than MAE, indicating that some large errors influence the overall evaluation. Earlier crop-level analysis also identified Coconut as a particularly difficult crop to predict.

**Limitations:** The dataset's feature units and availability at prediction time are not fully documented. The evaluation therefore demonstrates performance on historical records, not guaranteed real-world forecasting accuracy. Further error analysis and validation are needed before deployment.
### 17. Actual vs. Predicted Yield Analysis

**Objective:** Investigate prediction errors on the chronological test set covering 2018–2019.

**Methods used:**
- Inspected individual actual and predicted yields.
- Ranked records by absolute prediction error.
- Calculated crop-level MAE, median absolute error, and maximum absolute error.
- Examined MAPE and WAPE to compare errors relative to crop yield scales.
- Plotted actual versus predicted yields for all crops and separately for Coconut.

**Key findings:**
1. Coconut had the largest crop-level MAE at 1262.0479 across 17 test records, with a maximum absolute error of 3873.1538.
2. All ten records with the largest absolute errors were Coconut records.
3. Cashewnut had the highest crop-level WAPE at 153.89%, followed by Castor seed at 145.20%.
4. MAPE was highly sensitive to small actual yields. One Cashewnut record had an absolute percentage error of 4147.87%.
5. The actual-versus-predicted plots showed both overprediction and underprediction, particularly for Coconut. Some of the highest Coconut yields were underpredicted.

**Conclusion:** Overall metrics alone do not describe performance across all crops. Crop-specific errors and differences in yield scale must be considered when interpreting the model.

**Next steps:** Investigate possible reasons for crop-specific errors and evaluate improvements without changing the held-out test set repeatedly. No records have been removed and no model changes have been justified by this analysis alone.