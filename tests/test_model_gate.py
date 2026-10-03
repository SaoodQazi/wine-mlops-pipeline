"""MLOps Quality Gate: metric threshold, latency and output schema checks."""
import statistics
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from src.data import get_train_test_split
from src.evaluate import champion_uri
from src.train import REGISTERED_MODEL_NAME, run_pipeline

F1_THRESHOLD = 0.88
LATENCY_THRESHOLD_MS = 30.0


@pytest.fixture(scope="module")
def champion(tmp_path_factory):
    """Run the full pipeline in an isolated tracking store, return champion + model."""
    tmp = tmp_path_factory.mktemp("mlflow_gate")
    uri = f"sqlite:///{tmp / 'mlflow.db'}"
    info = run_pipeline(tracking_uri=uri)
    model = mlflow.sklearn.load_model(champion_uri())
    return info, model


def test_minimum_runs_logged(champion):
    info, _ = champion
    assert info["n_runs"] >= 6


def test_champion_alias_registered(champion):
    info, _ = champion
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    assert str(mv.version) == str(info["version"])


def test_metric_threshold_gate(champion):
    info, _ = champion
    assert info["val_f1_macro"] >= F1_THRESHOLD, (
        f"Validation macro F1 {info['val_f1_macro']:.4f} < {F1_THRESHOLD}"
    )


def test_inference_latency_gate(champion):
    _, model = champion
    _, X_test, _, _ = get_train_test_split()
    batch = X_test.astype("float64")
    model.predict(batch)  # warm-up
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict(batch)
        timings.append((time.perf_counter() - start) * 1000)
    median_ms = statistics.median(timings)
    assert median_ms <= LATENCY_THRESHOLD_MS, (
        f"Median batch latency {median_ms:.2f} ms > {LATENCY_THRESHOLD_MS} ms"
    )


def test_output_schema_integrity(champion):
    _, model = champion
    _, X_test, _, _ = get_train_test_split()
    preds = model.predict(X_test.astype("float64"))
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1, 2})
