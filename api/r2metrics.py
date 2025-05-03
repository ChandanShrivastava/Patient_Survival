import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import joblib
from pathlib import Path
import requests

class R2Metrics:
    def __init__(self):
        # Define paths for dataset and model
        self.parent_dir = Path(__file__).resolve().parents[1]
        self.dataset_path = self.parent_dir / 'dataset' / 'heart_failure_clinical_records_dataset.csv'
        self.model_path = self.parent_dir / 'model' / 'xgboost-model.pkl'
        # Ensure the dataset and model files exist
        self.download_if_not_exists(
            "https://cdn.iisc.talentsprint.com/CDS/Datasets/heart_failure_clinical_records_dataset.csv", self.dataset_path
        )
        self.download_if_not_exists(
            "https://example.com/xgboost-model.pkl", self.model_path
        )
        # Load the model
        self.xgb_clf = joblib.load(str(self.model_path))


    def download_if_not_exists(self, url, save_path):
        """
        Download a file from a URL if it does not already exist.
        :param url: The URL of the file to download.
        :param save_path: The local path where the file should be saved.
        """
        if not save_path.exists():
            print(f"File not found at {save_path}. Downloading from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded successfully: {save_path}")
        else:
            print(f"File already exists: {save_path}")
            
    def calculate_r2_metric(self, X, y):
        """
        Calculate the R² metric for the predict_death_event function.
        :param X: Feature matrix (2D array).
        :param y: Actual labels (1D array).
        :return: R² score.
        """
        # Generate predictions using the model
        predictions = self.xgb_clf.predict(X)
        
        # Calculate R² score
        r2 = r2_score(y, predictions)
        return r2

    def load_sample_data(self, sample_size=50, random_state=42):
        """
        Load a random sample of data from the dataset.
        :param sample_size: Number of rows to sample.
        :param random_state: Random state for reproducibility.
        :return: Feature matrix (X) and target vector (y).
        """
        # Load the dataset
        data = pd.read_csv(self.dataset_path)
        data_sample = data.sample(n=sample_size, random_state=random_state)

        # Split the dataset into features (X) and target (y)
        X = data_sample.drop(columns=['DEATH_EVENT'])  # Features
        y = data_sample['DEATH_EVENT']  # Target

        # Ensure the feature order matches the predict_death_event function
        X = X[['age', 'anaemia', 'high_blood_pressure', 'creatinine_phosphokinase', 'diabetes',
               'ejection_fraction', 'platelets', 'sex', 'serum_creatinine', 'serum_sodium',
               'smoking', 'time']]

        # Convert to NumPy arrays
        X = X.to_numpy()
        y = y.to_numpy()

        return X, y

    def calculate_r2_from_sample(self, sample_size=50, random_state=42):
        """
        Calculate the R² metric using a random sample from the dataset.
        :param sample_size: Number of rows to sample.
        :param random_state: Random state for reproducibility.
        :return: R² score.
        """
        # Load sample data
        X, y = self.load_sample_data(sample_size, random_state)

        # Calculate R² metric
        return self.calculate_r2_metric(X, y)


# Entry point for testing the class
if __name__ == "__main__":
    r2_metrics = R2Metrics()
    try:
        r2 = r2_metrics.calculate_r2_from_sample()
        print(f"R² Metric: {r2}")
    except Exception as e:
        print(f"Error calculating R² metric: {e}")