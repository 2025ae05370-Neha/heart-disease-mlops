"""FastAPI service that serves heart-disease predictions.

Endpoints:
    GET  /         - basic info
    GET  /health   - health check (used by Docker/Kubernetes)
    POST /predict  - predict disease risk from patient features
    GET  /docs     - auto-generated interactive API documentation
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Make the project's src/ importable (predict.py, pipeline.py)
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import SAMPLE_PATIENT, load_model  # noqa: E402
from predict import predict as run_prediction  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("heart-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so the first request is fast."""
    load_model()
    logger.info("Model loaded and ready.")
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the risk of heart disease from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)


class PatientFeatures(BaseModel):
    """Input schema: the 13 clinical features the model expects."""

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: float = Field(..., ge=0, le=3, description="Number of major vessels (0-3)")
    thal: float = Field(..., description="Thalassemia (3=normal, 6=fixed, 7=reversible)")

    model_config = {"json_schema_extra": {"example": SAMPLE_PATIENT}}


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    label: str = Field(..., description="'disease' or 'no_disease'")
    probability: float = Field(..., description="Probability of disease (0-1)")


@app.get("/")
def root() -> dict:
    return {
        "service": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    payload = features.model_dump()
    result = run_prediction(payload)
    logger.info("prediction=%s probability=%s", result["prediction"], result["probability"])
    return PredictionResponse(**result)
