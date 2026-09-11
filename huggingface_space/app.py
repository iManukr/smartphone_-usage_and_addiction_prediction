from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "smartphone_addiction_model.joblib"
model = joblib.load(MODEL_PATH)

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


def predict(
    age,
    gender,
    daily_screen_time_hours,
    social_media_hours,
    gaming_hours,
    work_study_hours,
    sleep_hours,
    notifications_per_day,
    app_opens_per_day,
    weekend_screen_time,
    stress_level,
    academic_work_impact,
):
    row = pd.DataFrame(
        [[
            age, gender, daily_screen_time_hours, social_media_hours, gaming_hours,
            work_study_hours, sleep_hours, notifications_per_day, app_opens_per_day,
            weekend_screen_time, stress_level, academic_work_impact,
        ]],
        columns=FEATURES,
    )
    probabilities = model.predict_proba(row)[0]
    return {"Low risk": float(probabilities[0]), "At risk of addiction": float(probabilities[1])}


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(18, 35, step=1, value=24, label="Age"),
        gr.Dropdown(["Female", "Male", "Other"], value="Female", label="Gender"),
        gr.Slider(3.0, 12.0, step=0.1, value=8.2, label="Daily screen time (hours)"),
        gr.Slider(0.5, 6.0, step=0.1, value=4.1, label="Social media (hours)"),
        gr.Slider(0.0, 4.0, step=0.1, value=1.3, label="Gaming (hours)"),
        gr.Slider(0.5, 6.0, step=0.1, value=5.0, label="Work/study (hours)"),
        gr.Slider(4.5, 9.0, step=0.1, value=5.8, label="Sleep (hours)"),
        gr.Slider(20, 250, step=1, value=210, label="Notifications per day"),
        gr.Slider(15, 180, step=1, value=130, label="App opens per day"),
        gr.Slider(3.6, 14.9, step=0.1, value=9.0, label="Weekend screen time (hours)"),
        gr.Dropdown(["Low", "Medium", "High"], value="High", label="Stress level"),
        gr.Dropdown(["No", "Yes"], value="Yes", label="Academic/work impact"),
    ],
    outputs=gr.Label(label="Prediction"),
    title="📱 Smartphone Addiction Risk Predictor",
    description=(
        "Trained on 7,500 survey responses (Random Forest, held-out macro F1 = 0.93). "
        "Educational demo — not a clinical diagnosis. Same pipeline is also served via "
        "a FastAPI + Vercel demo."
    ),
)

if __name__ == "__main__":
    demo.launch()
