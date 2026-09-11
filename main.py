from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Smartphone Addiction Risk API",
    description="Serves the trained pipeline from model_training.ipynb.",
)

# Demo API called from a separately hosted frontend (Vercel) — open CORS is
# fine here since this endpoint only accepts non-sensitive survey inputs and
# returns a prediction, no auth/user data involved.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "models" / "smartphone_addiction_model.joblib"
model = joblib.load(MODEL_PATH)

INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text()

FEATURES = [
    "age",
    "gender",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "stress_level",
    "academic_work_impact",
]


class UsageInput(BaseModel):
    age: int = Field(ge=0)
    gender: str
    daily_screen_time_hours: float = Field(ge=0)
    social_media_hours: float = Field(ge=0)
    gaming_hours: float = Field(ge=0)
    work_study_hours: float = Field(ge=0)
    sleep_hours: float = Field(ge=0)
    notifications_per_day: int = Field(ge=0)
    app_opens_per_day: int = Field(ge=0)
    weekend_screen_time: float = Field(ge=0)
    stress_level: str
    academic_work_impact: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 24,
                "gender": "Female",
                "daily_screen_time_hours": 8.2,
                "social_media_hours": 4.1,
                "gaming_hours": 1.3,
                "work_study_hours": 5.0,
                "sleep_hours": 5.8,
                "notifications_per_day": 210,
                "app_opens_per_day": 130,
                "weekend_screen_time": 9.0,
                "stress_level": "High",
                "academic_work_impact": "Yes",
            }
        }
    }


class PredictionOutput(BaseModel):
    addicted: bool
    addicted_label: int
    probability_not_addicted: float
    probability_addicted: float


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: UsageInput):
    row = pd.DataFrame([input_data.model_dump()])[FEATURES]
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    return PredictionOutput(
        addicted=bool(prediction),
        addicted_label=prediction,
        probability_not_addicted=round(float(probabilities[0]), 4),
        probability_addicted=round(float(probabilities[1]), 4),
    )
