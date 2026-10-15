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

The model was trained on records from 1997–2017 and evaluated on records from 2018–2019.

| Metric | Baseline Random Forest | Log-Transformed Random Forest |
|---|---:|---:|
| MAE | 11.2250 | 10.6382 |
| RMSE | 154.4333 | 150.3480 |
| R² | 0.9675 | 0.9692 |

The log-transformed model improved all three overall evaluation metrics. However, performance varied by crop, and some crops became less accurate.

## Project Structure

```text
Crop-Yield-Analysis/
├── data/
├── docs/
├── models/
├── notebooks/
├── src/
│   ├── crop_yield_model.py
│   └── predict.py
├── .gitignore
└── README.md
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