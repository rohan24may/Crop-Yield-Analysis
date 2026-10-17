# Analysis and Prediction of Crop Yield in India

A machine learning project that analyzes historical agricultural and climatic data to predict crop yield across Indian states.

## Project Objective

Explore historical crop yield patterns and build a regression model using crop type, season, state, year, rainfall, fertilizer, and pesticide features.

## Dataset

- **Source:** [Crop Yield Dataset — Mendeley Data](https://data.mendeley.com/datasets/ncw2vbcgnk/2)
- **Coverage:** Historical Indian agricultural records from 1997–2020
- **Size:** 19,689 records
- **Target variable:** `Yield`

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## Machine Learning

The project uses a Random Forest Regressor with a logarithmic transformation of the target variable.

### Evaluation Results

The model was trained on records from 1997–2017 and evaluated chronologically on records from 2018–2019.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest (raw target) | 11.2250 | 154.4333 | 0.9675 |
| Random Forest (log-transformed target) | 10.6382 | 150.3480 | 0.9692 |
| Gradient Boosting (log target) | 38.1818 | 475.4258 | 0.6916 |
| Tuned Random Forest (log target) | 11.2942 | 153.5448 | 0.9678 |
| Simpler Random Forest (log target) | 11.5831 | 152.4408 | 0.9683 |

The log-transformed Random Forest remains the leading candidate among the tested models. Crop-level error analysis showed that performance varies considerably by crop, particularly for crops with unusual yield scales.

**Evaluation limitation:** The 2018–2019 evaluation set was reused to compare alternative models. These metrics should therefore not be presented as results from an untouched final test set.

## Project Structure

```text
Crop-Yield-Analysis/
├── data/
├── docs/
├── models/
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_test_crop_yield_model.ipynb
│   └── 04_model_improvement.ipynb
├── src/
│   ├── crop_yield_model.py
│   └── predict.py
├── .gitignore
├── readme.md
└── requirements.txt
```

The trained model file is stored locally and excluded from GitHub because of its size.

## Running the Project

1. Clone the repository and activate your Python environment.
2. Install the required dependencies.
3. Obtain the dataset and place `crop_yield.csv` inside `data/`.
4. Run the notebooks in the `notebooks/` folder to explore and evaluate the project.
5. To make a prediction, ensure the locally saved model exists at `models/crop_yield_model.joblib`, then run:

```bash
python src/predict.py
```

### Recreating the Saved Model

The trained model file, `models/crop_yield_model.joblib`, is excluded from Git because of its size. A fresh clone must train and save a model before running the prediction script.

The reusable training function is available in `src/crop_yield_model.py`. It can train a model from a supplied DataFrame:

```python
import joblib
import pandas as pd
from pathlib import Path
from src.crop_yield_model import train_model

df = pd.read_csv("data/crop_yield.csv")
train_df = df[df["Crop_Year"] <= 2017].copy()

model = train_model(train_df)

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/crop_yield_model.joblib")
```

This reproduces the training pipeline on records through 2017. To reproduce the documented evaluation results, use the same feature configuration, chronological split, and prediction transformation described in the notebooks.

After saving the model, run the prediction script:

```bash
python src/predict.py
```

The example inputs in the script are illustrative, and the predictions should not be treated as validated agricultural advice.

## Limitations

- The dataset's measurement units and feature availability at prediction time are not fully documented.
- Model accuracy varies across crops.
- The evaluation covers one chronological test period, 2018–2019.
- Predictions should not be treated as validated agricultural advice.

## Future Improvements

- Investigate crop-specific prediction errors.
- Validate the model on additional future-year data.
- Improve documentation of data units and feature definitions.
- Explore alternative regression models and evaluation strategies.