# Version 01 — Dataset Validation

## 1. Project Overview

**Project:** Analysis and Prediction of Crop Yield in India Using Agricultural and Climatic Factors

**Objective:** Explore agricultural data and develop a machine-learning model to predict crop yield using relevant agricultural and climatic features.

## 2. Dataset Information

- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/ncw2vbcgnk/2)
- **File:** `crop_yield.csv`
- **Total records:** 19,689
- **Total columns:** 10
- **Year range:** 1997–2020
- **Number of crops:** 55

## 3. Work Completed

- Set up the Python project structure and virtual environment.
- Installed the required data science libraries.
- Loaded and inspected the dataset using Pandas.
- Explored crop, season, and yearly distributions.
- Created visualizations, including a correlation heatmap.
- Checked missing values, duplicate rows, negative values, and zero values.
- Created the initial data validation and preprocessing notebooks.

## 4. Important Findings

- No missing values or exact duplicate rows were found.
- No negative numerical values were found.
- There are 112 records with zero production and 112 with zero yield.
- Of these, 108 records have both production and yield equal to zero.
- Eight records have only one of these two values equal to zero.
- Yield values are highly right-skewed and differ substantially between crops.
- Some recorded yields differ considerably from production divided by area.
- Strong correlations exist between area, fertilizer, and pesticide quantities.

## 5. Decisions and Limitations

- No records were deleted during initial validation.
- Zero values and yield discrepancies require careful interpretation.
- The meaning and measurement units of yield should be verified.
- Correlation does not establish causation.
- The initial prediction target is `Yield`; the final feature set is not yet confirmed.

## 6. Tools Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## 7. Next Steps

1. Finalize preprocessing decisions.
2. Prepare the features and target for machine learning.
3. Split the data into training and testing sets.
4. Build a baseline regression model.
5. Evaluate its performance and document the results.

## 8. Version Status

**Status:** Initial dataset validation completed; preprocessing review underway.

**Last updated:** 15 October 2026