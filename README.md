# Adult Income Classification Project

## Problem Statement

This project predicts whether an individual's annual income is **`>50K`** or **`<=50K`** using demographic and employment-related features.  
It provides a full machine learning workflow for academic submission, including data acquisition, preprocessing, training multiple models, evaluation, and deployment-ready Streamlit integration.

## Dataset Details

- **Dataset:** Adult Income (Census Income)
- **Source:** OpenML (UCI-origin dataset): https://www.openml.org/d/1590
- **Task Type:** Binary classification
- **Samples:** 48,842
- **Features:** 14 input features (numeric + categorical)
- **Target:** `income` (`>50K` / `<=50K`)

The training script downloads the dataset automatically and stores:

- raw cleaned copy in `data/adult_income_raw.csv`
- cleaned copy in `data/adult_income_clean.csv`
- held-out test split in `data/test_reference.csv`

## Project Structure

```text
project-folder/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- data/
|-- model/
```

## Preprocessing Pipeline

The same preprocessing strategy is used for all models:

1. Replace missing placeholders (`?`) with null values.
2. Handle missing values:
   - numeric columns: median imputation
   - categorical columns: most-frequent imputation
3. Encode categorical variables with one-hot encoding (`handle_unknown="ignore"`).
4. Scale numeric variables using `StandardScaler`.
5. Split data with stratification:
   - train: 80%
   - test: 20%

## Models Implemented

1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest
6. XGBoost

All models use the same preprocessing pipeline and are saved in `model/` using `joblib`.

## Evaluation Metrics

Each model is evaluated on the held-out test set with:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

The comparison table is exported to:

- `model/model_comparison.csv`

### Model Comparison (Test Split)

| Model               | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| XGBoost             |   0.8742 | 0.9298 |    0.7887 | 0.6480 |   0.7114 | 0.6370 |
| Random Forest       |   0.8589 | 0.9058 |    0.7398 | 0.6334 |   0.6825 | 0.5955 |
| Logistic Regression |   0.8524 | 0.9042 |    0.7414 | 0.5885 |   0.6562 | 0.5699 |
| KNN                 |   0.8443 | 0.8876 |    0.7050 | 0.6009 |   0.6488 | 0.5525 |
| Decision Tree       |   0.8141 | 0.7475 |    0.6098 | 0.6198 |   0.6148 | 0.4923 |
| Naive Bayes         |   0.6204 | 0.8287 |    0.3794 | 0.9213 |   0.5374 | 0.3866 |

## Observations

- XGBoost is the best-performing model across F1, MCC, and AUC.
- Random Forest is the second-best option with balanced performance.
- Naive Bayes has very high recall but low precision, so it over-predicts the positive class.
- Decision threshold tuning materially changes precision-recall trade-offs and is exposed in the app sidebar.

## How to Run Locally

1. Create and activate a Python virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train and save models:

```bash
python3 train_models.py
```

4. Launch Streamlit app:

```bash
streamlit run app.py
```

### Notes on Python / dependency versions

- This repository includes pre-trained model artifacts in `model/*.joblib`.
- Those artifacts were produced with **scikit-learn 1.6.1**, so `requirements.txt` pins **`scikit-learn==1.6.1`** to ensure the Streamlit app can unpickle them reliably.
- For Streamlit Community Cloud, this repo also includes a `runtime.txt` requesting **Python 3.12** (needed because scikit-learn 1.6.1 is not compatible with Python 3.13).

## Streamlit App Features

- CSV file uploader for batch predictions
- Model selector dropdown
- Metrics display (when uploaded data contains `income`)
- Confusion matrix heatmap
- Classification report table
- Manual single-sample prediction interface
- Sidebar controls (model + threshold)

## Deployment Steps (Streamlit Community Cloud)

1. Push this folder to a GitHub repository.
2. Ensure these files exist at repository root:
   - `app.py`
   - `requirements.txt`
   - `runtime.txt` (pins Python version for deployment)
   - `model/` artifacts (or generate in a build step)
   - `data/` references used by app
3. In Streamlit Community Cloud:
   - connect the GitHub repo
   - select branch and `app.py`
   - in **Advanced settings**, select **Python 3.12** if prompted (some deployments may ignore `runtime.txt`)
   - deploy
4. Verify startup logs show successful model loading.

## Deployment Readiness Notes

- Uses only relative paths.
- No secrets or API keys required.
- Compatible with standard Python packages available on Streamlit Cloud.
- No machine-specific absolute paths.

### Important: no training during deployment

The Streamlit app is designed to **load existing artifacts** from `model/` and does **not** retrain models at startup.
If you want to refresh models, run `python3 train_models.py` locally, commit the updated `model/` artifacts, and redeploy.
