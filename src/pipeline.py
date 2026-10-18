"""Reusable data loading and preprocessing for the heart-disease model.

Both the training script (``src/train.py``) and the serving API import from
here so that the *exact same* preprocessing is used at train and predict time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart.csv"

# The 13 clinical features the model uses (we drop 'source' to avoid leakage).
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# chol/trestbps of 0 are physically impossible -> treat as missing.
ZERO_AS_MISSING = ["chol", "trestbps"]


def _zero_to_nan(X: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible 0 values in selected columns with NaN.

    Defined at module level (not a lambda) so it can be pickled with the model.
    """
    X = X.copy()
    for col in ZERO_AS_MISSING:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
    return X


def build_preprocessor() -> ColumnTransformer:
    """Return the ColumnTransformer that imputes/scales/encodes features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """Wrap zero-handling + preprocessing + a classifier into one Pipeline."""
    return Pipeline(
        steps=[
            ("zeros", FunctionTransformer(_zero_to_nan, feature_names_out="one-to-one")),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned dataset and return (X features, y target)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/download_data.py and src/preprocess.py first."
        )
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].astype(int)
    return X, y
