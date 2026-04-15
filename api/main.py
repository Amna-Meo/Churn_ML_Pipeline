from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib
import pandas as pd
import numpy as np
import logging

from src.preprocessing import clean_data, engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using ML pipeline",
    version="1.0.0",
)

model = None


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    count: int


@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load("models/churn_pipeline.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API", "status": "running"}


@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([customer.model_dump()])
        df = clean_data(df)
        df = engineer_features(df)
        X = df.drop("Churn", axis=1)

        prob = model.predict_proba(X)[0, 1]
        pred = model.predict(X)[0]

        risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"

        return PredictionResponse(
            churn_prediction="Yes" if pred == 1 else "No",
            churn_probability=round(float(prob), 4),
            risk_level=risk,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(data: List[CustomerData]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([c.model_dump() for c in data])
        df = clean_data(df)
        df = engineer_features(df)
        X = df.drop("Churn", axis=1)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist(),
            count=len(predictions),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
