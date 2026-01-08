from pydantic import BaseModel
from typing import List

class CustomerFeatures(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int
    risk_level: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool