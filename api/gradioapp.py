
import joblib
import numpy as np
import gradio as gr
import sys
from pathlib import Path

# Construct the absolute path to the model file
model_path = Path(__file__).resolve().parents[1] / 'model' / 'xgboost-model.pkl'

# Load the model using the absolute path
xgb_clf = joblib.load(str(model_path))


# Function for prediction

def predict_death_event(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time):
  """Predicts the survival of a patient with heart failure."""
  inputdata = np.array([age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, serum_creatinine, serum_sodium, smoking, time]).reshape(1, -1)
  # Make prediction using the trained model
  prediction = xgb_clf.predict(inputdata)
  if prediction[0] == 0:
    return "Patient is predicted to survive."
  else:
    return "Patient is predicted to not survive."

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gr.Interface(fn = predict_death_event,
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
    outputs="text",  # Output is a text string (survival prediction)
    title = title,
    description = description,
    allow_flagging='never')

iface.launch(share = True)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface