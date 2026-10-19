"""Automated tests for the prediction logic (src/predict.py).

These check that a prediction returns the expected fields, produces sensible
values (a 0/1 prediction and a probability between 0 and 1), and that the text
label always matches the numeric prediction.
"""

import pytest
from sklearn.ensemble import RandomForestClassifier

import predict
from pipeline import build_pipeline, load_data


@pytest.fixture
def trained_model():
    """Train a small model so the test does not depend on a saved .joblib file."""
    X, y = load_data()
    pipe = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0))
    pipe.fit(X, y)
    return pipe


def test_predict_returns_expected_keys(trained_model, monkeypatch):
    # Inject our trained model into predict's cache so it does not read from disk
    monkeypatch.setattr(predict, "_model", trained_model)
    result = predict.predict(predict.SAMPLE_PATIENT)
    assert set(result.keys()) == {"prediction", "label", "probability"}


def test_predict_values_are_valid(trained_model, monkeypatch):
    monkeypatch.setattr(predict, "_model", trained_model)
    result = predict.predict(predict.SAMPLE_PATIENT)
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("disease", "no_disease")
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_handles_missing_features(trained_model, monkeypatch):
    """Missing feature keys should be imputed, not crash."""
    monkeypatch.setattr(predict, "_model", trained_model)
    partial = {"age": 55, "sex": 1, "cp": 4}  # only a few features supplied
    result = predict.predict(partial)
    assert result["prediction"] in (0, 1)


def test_label_matches_prediction(trained_model, monkeypatch):
    monkeypatch.setattr(predict, "_model", trained_model)
    result = predict.predict(predict.SAMPLE_PATIENT)
    expected = "disease" if result["prediction"] == 1 else "no_disease"
    assert result["label"] == expected
