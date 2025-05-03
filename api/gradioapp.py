import joblib
import numpy as np
import gradio as gr
from fastapi import FastAPI
from prometheus_client import Counter, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi.middleware.wsgi import WSGIMiddleware
import threading
from pathlib import Path

# Define a Prometheus counter
prediction_counter = Counter('predictions_total', 'Total number of predictions made')

# Construct the absolute path to the model file
model_path = Path(__file__).resolve().parents[1] / 'model' / 'xgboost-model.pkl'

# Load the model using the absolute path
xgb_clf = joblib.load(str(model_path))

# Function for prediction
def predict_death_event(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time):
    """Predicts the survival of a patient with heart failure."""
    prediction_counter.inc()  # Increment the Prometheus counter
    inputdata = np.array([age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time]).reshape(1, -1)
    # Make prediction using the trained model
    prediction = xgb_clf.predict(inputdata)
    if prediction[0] == 0:
        return "Patient is predicted to survive."
    else:
        return "Patient is predicted to not survive."

# Gradio interface
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gr.Interface(
    fn=predict_death_event,
    inputs=[
        gr.Slider(minimum=40, maximum=95, step=1, value=60, label="Age"),
        gr.Radio(choices=[0, 1], value=0, label="Anaemia (0=No, 1=Yes)"),
        gr.Radio(choices=[0, 1], value=0, label="High Blood Pressure (0=No, 1=Yes)"),
        gr.Slider(minimum=23, maximum=7861, step=1, value=582, label="Creatinine Phosphokinase (CPK)"),
        gr.Radio(choices=[0, 1], value=0, label="Diabetes (0=No, 1=Yes)"),
        gr.Slider(minimum=14, maximum=80, step=1, value=38, label="Ejection Fraction"),
        gr.Slider(minimum=25100, maximum=850000, step=1, value=265000, label="Platelets"),
        gr.Radio(choices=[0, 1], value=0, label="Sex (0=Female, 1=Male)"),
        gr.Slider(minimum=0.5, maximum=9.4, step=0.1, value=1.1, label="Serum Creatinine"),
        gr.Slider(minimum=113, maximum=180, step=1, value=136, label="Serum Sodium"),
        gr.Radio(choices=[0, 1], value=0, label="Smoking (0=No, 1=Yes)"),
        gr.Slider(minimum=4, maximum=285, step=1, value=100, label="Time (Follow-up Period)"),
    ],
    outputs="text",
    title=title,
    description=description,
    allow_flagging='never'
)

# FastAPI app
app = FastAPI()

# Add the Gradio app as a WSGI middleware
app.mount("/gradio", WSGIMiddleware(iface.launch(share=False, server_name="0.0.0.0", inbrowser=False)))

# Add the /metrics endpoint
@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)