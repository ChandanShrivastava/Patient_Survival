# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

RUN mkdir -p /app/model
RUN mkdir -p /app/dataset


COPY requirements.txt /app/requirements.txt

# Copy only the 'api' folder
COPY api /app/api

# Copy only the 'model' folder
COPY model/xgboost-model.pkl /app/model/xgboost-model.pkl
#COPY dataset/heart_failure_clinical_records_dataset.csv /app/dataset/heart_failure_clinical_records_dataset.csv

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port Gradio will run on
EXPOSE 8001

# Command to run the Gradio app
#CMD ["python", "/app/api/gradioapp.py"]
CMD ["python", "/app/api/r2metrics.py"]
CMD ["uvicorn", "api.fastapiapp:app", "--host", "0.0.0.0", "--port", "8001"]