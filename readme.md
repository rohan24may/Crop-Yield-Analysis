# Crop Yield Analysis and Prediction in India

A machine learning project that analyzes historical agricultural and climatic data to predict crop yield across Indian states. The project includes data quality analysis, preprocessing, model comparison, prediction error investigation, and an interactive Streamlit dashboard.

## Project Objective

Explore historical crop yield patterns and build a regression model using crop type, season, state, year, annual rainfall, fertilizer, and pesticide features.

## Dataset

- **Source:** [Crop Yield Dataset — Mendeley Data](https://data.mendeley.com/datasets/ncw2vbcgnk/2)
- **Coverage:** Historical Indian agricultural records from 1997–2020
- **Size:** 19,689 records
- **Target variable:** `Yield`

The dataset is examined for missing values, duplicates, inconsistent categorical whitespace, unusual values, and potential data-quality issues. Flagged records are investigated rather than automatically removed.

## Tech Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Dashboard:** Streamlit
- **Development:** Jupyter Notebook, VS Code
- **Version control:** Git and GitHub

## Machine Learning Approach

The main model is a **Random Forest Regressor** trained using a logarithmic transformation of the target variable (`log1p`). Predictions are transformed back to the original target scale using `expm1`.

### Input Features

- Crop
- Crop year
- Season
- State
- Annual rainfall
- Fertilizer
- Pesticide

`Area` and `Production` are excluded from the main model's input features.

### Model Evaluation Results

The data was split chronologically, with records through 2017 used for training and 2018–2019 used for testing.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Random Forest — 100 trees, log target** | **10.6382** | **150.3480** | **0.9692** |
| Random Forest — 300 trees, log target | 10.7988 | 151.1103 | 0.9688 |
| Crop-only Random Forest | 34.4526 | 454.2240 | 0.7185 |
| Gradient Boosting, log target | 38.1818 | 475.4258 | 0.6916 |
| Tuned Random Forest, log target | 11.2942 | 153.5448 | 0.9678 |
| Simpler Random Forest, log target | 11.5831 | 152.4408 | 0.9683 |
| Random Forest, raw target | 11.2250 | 154.4333 | 0.9675 |

The 100-tree Random Forest was the strongest candidate among the tested configurations according to MAE, RMSE, and R².

### Additional Analysis

- **Feature importance:** Crop accounts for approximately 83.72% of the model's impurity-based feature importance. This describes how the fitted model uses the features; it does not establish causation.
- **Coconut error analysis:** Coconut's unusually large yield values substantially affect overall error metrics.
- **Baseline comparison:** On 17 Coconut test records, the Random Forest achieved an MAE of 1,208.158, compared with 1,551.179 for a historical-mean baseline.
- **Data-quality investigation:** Potentially inconsistent zero values and crop-wise IQR outliers were flagged and documented rather than automatically deleted.

### Evaluation Limitations

The 2018–2019 test set was reused during model comparisons. Therefore, these results are exploratory and should not be presented as an untouched final evaluation.

The 2020 records are limited in coverage and should not be treated as a representative test set for all crops and states. An independent evaluation on additional future-year data remains a useful next step.

## Project Structure

```text
Crop-Yield-Analysis/
├── data/
├── docs/
│   ├── model_evaluation/
│   └── preprocessing/
├── models/
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_test_crop_yield_model.ipynb
│   └── 04_model_improvement.ipynb
├── src/
│   ├── crop_yield_model.py
│   ├── predict.py
│   └── train.py
├── app.py
├── .gitignore
├── README.md
└── requirements.txt
```

The trained model file is stored locally and excluded from Git because of its size. The dataset may also need to be obtained separately, depending on repository configuration.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/rohan24may/Crop-Yield-Analysis.git
cd Crop-Yield-Analysis
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Obtain the dataset from [Mendeley Data](https://data.mendeley.com/datasets/ncw2vbcgnk/2) and place `crop_yield.csv` in the `data/` directory.

### 5. Train and save the model

```bash
python src/train.py
```

The trained model is saved to:

```text
models/crop_yield_model.joblib
```

### 6. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

### 7. Run the prediction script

After training the model:

```bash
python src/predict.py
```

The prediction script uses its configured example inputs. Predictions are estimates based on the dataset and model, not validated agricultural advice.

## Limitations

- The dataset's target measurement units and some feature definitions require further verification.
- Prediction errors vary considerably by crop, especially for crops with unusually large yield scales.
- Feature importance does not establish causal relationships.
- The evaluation covers a limited chronological test period, and the test set was reused during model comparisons.
- Some flagged data-quality issues have not been independently verified against the original source.
- Model inputs may not all be available in advance when making genuine future-yield forecasts.

## Future Improvements

- Conduct independent validation on additional future-year data.
- Investigate crop-specific errors and unusual target values.
- Verify the target's units and definition against the dataset documentation.
- Evaluate feature availability at the time a prediction would be made.
- Explore additional regression models and robust evaluation strategies.
- Improve model monitoring and dashboard documentation.

## Author

**Rohan** — MCA student working on data science and machine learning projects.