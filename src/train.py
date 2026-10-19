"""Model development and experiment tracking.

This script is the modelling heart of the assignment. It trains two different
classifiers - Logistic Regression and Random Forest - so we can compare them
fairly instead of guessing which one is better.

For each model it:
  * wraps the model in the shared preprocessing pipeline,
  * searches for good hyperparameters using GridSearchCV with 5-fold
    cross-validation,
  * measures performance on a held-out test set (accuracy, precision, recall,
    F1 and ROC-AUC),
  * saves ROC-curve and confusion-matrix plots for the report, and
  * logs every run to MLflow so the experiments are recorded and comparable.

The best model (highest test ROC-AUC) is saved to
``models/heart_model.joblib`` for the API to serve.
"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from pipeline import PROJECT_ROOT, build_pipeline, load_data

RANDOM_STATE = 42
MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "report" / "screenshots"

# MLflow logs runs into a local ./mlruns folder (open later with `mlflow ui`).
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "heart-disease-uci"

# Each entry: the base estimator + a small hyperparameter grid to search.
MODEL_CONFIGS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000),
        "param_grid": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
    },
}


def evaluate(estimator, X_test, y_test) -> dict[str, float]:
    """Compute standard classification metrics on the test set."""
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def save_plots(name: str, estimator, X_test, y_test) -> None:
    """Save ROC curve and confusion matrix for one model."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_estimator(estimator, X_test, y_test)
    plt.title(f"ROC curve \u2014 {name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"roc_{name}.png", bbox_inches="tight")
    plt.close()

    ConfusionMatrixDisplay.from_estimator(estimator, X_test, y_test,
                                          display_labels=["No disease", "Disease"])
    plt.title(f"Confusion matrix \u2014 {name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"confusion_{name}.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    results: dict[str, dict] = {}
    for name, config in MODEL_CONFIGS.items():
        print(f"\n=== Training {name} ===")
        pipe = build_pipeline(config["estimator"])
        search = GridSearchCV(
            pipe,
            param_grid=config["param_grid"],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        metrics = evaluate(search.best_estimator_, X_test, y_test)
        save_plots(name, search.best_estimator_, X_test, y_test)

        # ---- Log this run to MLflow ----
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_roc_auc", search.best_score_)
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            mlflow.log_artifact(str(FIG_DIR / f"roc_{name}.png"))
            mlflow.log_artifact(str(FIG_DIR / f"confusion_{name}.png"))
            signature = infer_signature(X_test, search.best_estimator_.predict(X_test))
            mlflow.sklearn.log_model(
                search.best_estimator_,
                artifact_path="model",
                signature=signature,
                input_example=X_test.iloc[:2],
            )

        results[name] = {
            "best_params": search.best_params_,
            "cv_roc_auc": search.best_score_,
            "estimator": search.best_estimator_,
            "metrics": metrics,
        }
        print(f"  best params: {search.best_params_}")
        print(f"  CV ROC-AUC : {search.best_score_:.4f}")
        print("  test metrics:")
        for metric, value in metrics.items():
            print(f"    {metric:10s}: {value:.4f}")

    # ---- Compare and select the best model by test ROC-AUC ----
    print("\n=== Model comparison (test set) ===")
    header = (
        f"{'model':22s} {'accuracy':>9s} {'precision':>10s} "
        f"{'recall':>8s} {'f1':>7s} {'roc_auc':>8s}"
    )
    print(header)
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:22s} {m['accuracy']:9.4f} {m['precision']:10.4f} "
              f"{m['recall']:8.4f} {m['f1']:7.4f} {m['roc_auc']:8.4f}")

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best_estimator = results[best_name]["estimator"]
    model_path = MODELS_DIR / "heart_model.joblib"
    joblib.dump(best_estimator, model_path)
    print(f"\nBest model: {best_name}")
    print(f"Saved -> {model_path}")


if __name__ == "__main__":
    main()
