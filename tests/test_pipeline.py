"""Unit tests for the shared preprocessing pipeline in src/pipeline.py."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    _zero_to_nan,
    build_pipeline,
    load_data,
)


def test_zero_to_nan_only_affects_selected_columns():
    df = pd.DataFrame({"chol": [0, 200], "trestbps": [0, 120], "age": [0, 50]})
    out = _zero_to_nan(df)
    # chol/trestbps zeros become NaN...
    assert np.isnan(out.loc[0, "chol"])
    assert np.isnan(out.loc[0, "trestbps"])
    # ...but age is left untouched (0 stays 0)
    assert out.loc[0, "age"] == 0


def test_load_data_shapes_and_target():
    X, y = load_data()
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_feature_groups_are_disjoint_and_complete():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(FEATURE_COLUMNS)


def test_pipeline_fits_and_predicts():
    X, y = load_data()
    pipe = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=0))
    pipe.fit(X, y)
    preds = pipe.predict(X.head(5))
    proba = pipe.predict_proba(X.head(5))
    assert len(preds) == 5
    assert set(np.unique(preds)).issubset({0, 1})
    # probabilities are valid
    assert proba.shape == (5, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_pipeline_handles_missing_values():
    """A row full of NaNs should still predict (imputation fills the gaps)."""
    X, y = load_data()
    pipe = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=0))
    pipe.fit(X, y)
    all_nan = pd.DataFrame([{col: np.nan for col in FEATURE_COLUMNS}])
    pred = pipe.predict(all_nan)
    assert pred[0] in (0, 1)
