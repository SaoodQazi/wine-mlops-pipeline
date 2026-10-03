"""Inference verification: load the registered champion and score the test split."""
from typing import Any, Dict

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import get_train_test_split
from src.train import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, setup_mlflow


def champion_uri() -> str:
    return f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"


def evaluate_champion(tracking_uri: str = None) -> Dict[str, Any]:
    """Load models:/WineClassifier@champion and compute final test metrics."""
    setup_mlflow(tracking_uri)
    _, X_test, _, y_test = get_train_test_split()
    model = mlflow.sklearn.load_model(champion_uri())
    preds = model.predict(X_test.astype("float64"))
    proba = model.predict_proba(X_test.astype("float64"))
    metrics = {
        "test_f1_macro": f1_score(y_test, preds, average="macro"),
        "test_accuracy": accuracy_score(y_test, preds),
        "test_log_loss": log_loss(y_test, proba),
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_champion()
