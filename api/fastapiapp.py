from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_client import Counter, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi.responses import HTMLResponse
from pathlib import Path
import requests
from r2metrics import R2Metrics

# Define the FastAPI app
app = FastAPI()

# Construct the absolute path to the model file
model_path = Path(__file__).resolve().parents[1] / 'model' / 'xgboost-model.pkl'
# Load the model
xgb_clf = joblib.load(str(model_path))

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")
# Serve static files (if needed for CSS/JS)
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the input schema using Pydantic
class PatientData(BaseModel):
    age: int
    anaemia: int
    high_blood_pressure: int
    creatinine_phosphokinase: int
    diabetes: int
    ejection_fraction: int
    platelets: float
    sex: int
    serum_creatinine: float
    serum_sodium: int
    smoking: int
    time: int


# Render the UI form
@app.get("/", response_class=HTMLResponse)
def render_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Handle form submission and make predictions
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    anaemia: int = Form(...),
    high_blood_pressure: int = Form(...),
    creatinine_phosphokinase: int = Form(...),
    diabetes: int = Form(...),
    ejection_fraction: int = Form(...),
    platelets: float = Form(...),
    sex: int = Form(...),
    serum_creatinine: float = Form(...),
    serum_sodium: int = Form(...),
    smoking: int = Form(...),
    time: int = Form(...)
):
    # Prepare input data for the model
    inputdata = np.array([
        age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes,
        ejection_fraction, platelets, sex, serum_creatinine, serum_sodium,
        smoking, time
    ]).reshape(1, -1)

    # Make prediction
    prediction = xgb_clf.predict(inputdata)
    result = "Patient is predicted to survive." if prediction[0] == 0 else "Patient is predicted to not survive."

    # Render the result
    return templates.TemplateResponse("result.html", {"request": request, "result": result})


    
# Add the /metrics endpoint
@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics and calculate R² metric."""
    r2_metrics = R2Metrics()  # Initialize the R2Metrics class
    try:
        # Calculate R² metric using a random sample
        r2 = r2_metrics.calculate_r2_from_sample()

        # Include R² metric in Prometheus output
        r2_metric = f"# HELP r2_metric Coefficient of determination (R²)\n# TYPE r2_metric gauge\nr2_metric {r2}\n"
        prometheus_metrics = generate_latest().decode("utf-8")
        return Response(prometheus_metrics + r2_metric, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        # Handle errors and return Prometheus metrics without R²
        print(f"Error calculating R² metric: {e}")
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 