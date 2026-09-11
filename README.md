# Smartphone Usage and Addiction Prediction

## Problem Statement

This classification project predicts the binary addiction-risk label (`addicted_label`). The result can support informed, proactive decisions rather than replacing human judgment.

## Dataset

The project uses `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`, containing **7,500 rows and 16 columns** in its original form. Important inputs include age, usage hours, sleep, notifications, app opens, gender, stress, and academic/work impact. The target is the binary addiction-risk label (`addicted_label`). Identifier columns are removed because their arbitrary values should not drive predictions. The notebook documents data types, missing values, duplicates, class counts, and every cleaning decision from the actual file.

## Machine Learning Workflow

```text
Load Data → Explore → Clean → Preprocess → Train → Evaluate → Select Model → Save Model
```

The train/test split is stratified. Imputation, scaling, and one-hot encoding are fitted only on training data inside Scikit-learn pipelines, preventing leakage.

## Models Tested

Logistic Regression and Random Forest are compared on the same held-out 20% test set (`random_state=42`).

## Final Model

The final model is the pipeline with the highest held-out macro F1 score; `addiction_level` is excluded to prevent target leakage. The notebook calculates and displays held-out **accuracy, macro precision, macro recall, macro F1**, classification reports, and confusion matrices; for binary targets it also reports **ROC-AUC** and plots ROC curves. The exact recorded metrics remain next to the training run in `model_training.ipynb`, ensuring the documented result corresponds to the serialized pipeline rather than a manually copied number.

## Saved Model

```text
models/smartphone_addiction_model.joblib
```

This artifact contains preprocessing and the selected fitted classifier in one pipeline.

## Loading the Model

Run from this project directory:

```python
import joblib
import pandas as pd

model = joblib.load("models/smartphone_addiction_model.joblib")
# prediction = model.predict(pd.DataFrame([new_customer_or_observation]))
```

The model object will later be loaded inside a FastAPI application. The notebook reloads it immediately after saving and asserts that a new observation produces exactly one prediction.

## Reproduce Training

Install the dependencies, open `model_training.ipynb`, choose **Restart Kernel and Run All**, and run it with this folder as the working directory. Re-running deliberately refreshes both the displayed metrics and `models/smartphone_addiction_model.joblib`.

## Serving the Model via FastAPI

`main.py` loads `models/smartphone_addiction_model.joblib` and exposes it over HTTP.

Run from this project directory:

```bash
uv pip install -p .venv -e .   # or: pip install fastapi uvicorn joblib pandas scikit-learn
.venv/bin/uvicorn main:app --reload
```

Then send a request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 24, "gender": "Female", "daily_screen_time_hours": 8.2,
        "social_media_hours": 4.1, "gaming_hours": 1.3, "work_study_hours": 5.0,
        "sleep_hours": 5.8, "notifications_per_day": 210, "app_opens_per_day": 130,
        "weekend_screen_time": 9.0, "stress_level": "High", "academic_work_impact": "Yes"
      }'
```

which returns:

```json
{
  "addicted": true,
  "addicted_label": 1,
  "probability_not_addicted": 0.0033,
  "probability_addicted": 0.9967
}
```

Interactive docs are auto-generated at `http://127.0.0.1:8000/docs`.
