"""Load the trained model and make predictions on new patient data.

Used both as a command-line tool and imported by the serving API so that
training and serving share the exact same prediction logic.

Example (CLI):
    python src/predict.py                 # predicts on a built-in sample
    python src/predict.py '{"age": 54, "sex": 1, ...}'   # predict from JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from pipeline import FEATURE_COLUMNS, PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "heart_model.joblib"

# A valid example patient (all 13 features) for quick testing/docs.
SAMPLE_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}

_model = None  # cached so we load from disk only once


def load_model(path: Path = MODEL_PATH):
    """Load the trained pipeline from disk (cached after first call)."""
    global _model
    if _model is None:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Train the model first: python src/train.py"
            )
        _model = joblib.load(path)
    return _model


def predict(features: dict) -> dict:
    """Predict heart-disease risk for one patient.

    Parameters
    ----------
    features : dict
        Mapping of the 13 feature names to values. Missing keys become NaN and
        are imputed by the pipeline.

    Returns
    -------
    dict with keys: prediction (0/1), label, probability (of disease).
    """
    model = load_model()
    row = pd.DataFrame([{col: features.get(col) for col in FEATURE_COLUMNS}])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "prediction": prediction,
        "label": "disease" if prediction == 1 else "no_disease",
        "probability": round(probability, 4),
    }


def main() -> None:
    if len(sys.argv) > 1:
        features = json.loads(sys.argv[1])
    else:
        print("No input given, using built-in sample patient.")
        features = SAMPLE_PATIENT

    result = predict(features)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
