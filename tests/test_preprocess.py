"""Unit tests for the data cleaning logic in src/preprocess.py."""

import numpy as np
import pandas as pd

from preprocess import clean


def _raw_sample() -> pd.DataFrame:
    """A tiny raw-style frame with the original 'num' target and a '?' already
    converted to NaN (as pandas would read it)."""
    return pd.DataFrame(
        {
            "age": [63.0, 67.0, 37.0],
            "sex": [1.0, 1.0, 1.0],
            "cp": [1.0, 4.0, 3.0],
            "trestbps": [145.0, 160.0, 130.0],
            "chol": [233.0, 286.0, 250.0],
            "fbs": [1.0, 0.0, 0.0],
            "restecg": [2.0, 2.0, 0.0],
            "thalach": [150.0, 108.0, 187.0],
            "exang": [0.0, 1.0, 0.0],
            "oldpeak": [2.3, 1.5, 3.5],
            "slope": [3.0, 2.0, 3.0],
            "ca": [0.0, 3.0, np.nan],
            "thal": [6.0, 3.0, 3.0],
            "num": [0, 2, 1],  # 0 -> no disease, >0 -> disease
        }
    )


def test_clean_binarizes_target():
    cleaned = clean(_raw_sample())
    # num 0,2,1 -> target 0,1,1
    assert list(cleaned["target"]) == [0, 1, 1]


def test_clean_drops_num_column():
    cleaned = clean(_raw_sample())
    assert "num" not in cleaned.columns
    assert "target" in cleaned.columns


def test_clean_target_is_binary_int():
    cleaned = clean(_raw_sample())
    assert cleaned["target"].dtype == int
    assert set(cleaned["target"].unique()).issubset({0, 1})
