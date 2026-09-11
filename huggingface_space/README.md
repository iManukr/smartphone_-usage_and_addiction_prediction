---
title: Smartphone Addiction Risk Predictor
emoji: 📱
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Smartphone Addiction Risk Predictor

Gradio demo for a scikit-learn Random Forest pipeline trained on 7,500 survey
responses (leakage-safe preprocessing, held-out macro F1 = 0.93). Enter daily
usage/wellbeing info and get a predicted addiction-risk label with confidence.

Educational demo — not a clinical diagnosis. The same trained pipeline is also
served via a separate FastAPI + Vercel demo.
