from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# ==================================================
# LOAD MODEL AND SCALER
# ==================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ==================================================
# CREATE FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Advertisement Sale Prediction API",
    description="ML API for predicting whether a customer will purchase an advertisement product.",
    version="1.0.0"
)


# ==================================================
# REQUEST DATA MODEL
# ==================================================

class CustomerData(BaseModel):

    age: float
    estimated_salary: float


# ==================================================
# ROOT ENDPOINT
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Advertisement Sale Prediction API",
        "status": "running"
    }


# ==================================================
# PREDICTION ENDPOINT
# ==================================================

@app.post("/predict")
def predict_purchase(customer: CustomerData):

    # Create DataFrame
    input_data = pd.DataFrame(
        [[customer.age, customer.estimated_salary]],
        columns=["Age", "EstimatedSalary"]
    )

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Get probabilities
    probability = model.predict_proba(input_scaled)

    # Convert prediction to readable text
    if prediction[0] == 1:
        result = "Purchased"
    else:
        result = "Not Purchased"

    return {
        "age": customer.age,
        "estimated_salary": customer.estimated_salary,
        "prediction": result,
        "not_purchased_probability": round(
            float(probability[0][0]), 4
        ),
        "purchased_probability": round(
            float(probability[0][1]), 4
        )
    }