from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib
import numpy as np
import logging
import os
import json
import traceback
from functools import lru_cache

from opencensus.ext.azure.log_exporter import AzureLogHandler
from app.models import CustomerFeatures, PredictionResponse, HealthResponse
from app.drift_detect import detect_drift
from app.utils import hash_features

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bank-churn-api")

APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if APPINSIGHTS_CONN:
    logger.addHandler(AzureLogHandler(connection_string=APPINSIGHTS_CONN))

app = FastAPI(title="Bank Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.pkl")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Modèle chargé : {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Erreur chargement : {e}")

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy", "is_model_active": model is not None}

@lru_cache(maxsize=1000)
def get_cached_prediction(features_json: str):
    """Fonction de cache pour éviter de recalculer les mêmes prédictions"""
    features_dict = json.loads(features_json)
    X = np.array([list(features_dict.values())])
    proba = float(model.predict_proba(X)[0][1])
    prediction = int(proba > 0.5)
    risk = "Low" if proba < 0.3 else "Medium" if proba < 0.7 else "High"
    return {"churn_probability": round(proba, 4), "prediction": prediction, "risk_level": risk}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible")
    
    try:
        # On utilise le cache
        feat_dict = features.model_dump()
        feat_json = json.dumps(feat_dict, sort_keys=True)
        result = get_cached_prediction(feat_json)
        
        logger.info("prediction_made", extra={"custom_dimensions": result})
        return result
    except Exception as e:
        logger.error(f"Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift/check", tags=["Monitoring"])
def check_drift(threshold: float = 0.05):
    try:
        results = detect_drift(
            reference_file="data/bank_churn.csv",
            production_file="data/production_data.csv",
            threshold=threshold
        )
        drifted = [f for f, r in results.items() if r["drift_detected"]]
        return {"status": "success", "drift_percentage": (len(drifted)/len(results))*100, "drifted_features": drifted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))