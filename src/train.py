"""Hyperparameter search, MLflow tracking and model registry promotion."""
import itertools
import os
from typing import Any, Dict, List

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import SEED, get_train_test_split

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
RESULTS_CSV = os.path.join("results", "hyperparameter_results.csv")

SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "log_loss": "neg_log_loss",
}

MODEL_FAMILIES = {
    "RandomForest": {
        "class": RandomForestClassifier,
        "fixed": {"random_state": SEED, "n_jobs": 1},
        "grid": {
            "n_estimators": [25, 50, 100],
            "max_depth": [3, None],
        },
    },
    "GradientBoosting": {
        "class": GradientBoostingClassifier,
        "fixed": {"random_state": SEED},
        "grid": {
            "n_estimators": [25, 50, 100],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    },
}


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a parameter grid -> list of configs."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def cross_validate_config(model, X_train, y_train) -> Dict[str, float]:
    """5-fold stratified CV; returns mean train/validation metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_validate(
        model, X_train, y_train, cv=cv, scoring=SCORING,
        return_train_score=True, n_jobs=1,
    )
    return {
        "train_f1_macro": float(np.mean(scores["train_f1_macro"])),
        "train_accuracy": float(np.mean(scores["train_accuracy"])),
        "train_log_loss": float(-np.mean(scores["train_log_loss"])),
        "val_f1_macro": float(np.mean(scores["test_f1_macro"])),
        "val_accuracy": float(np.mean(scores["test_accuracy"])),
        "val_log_loss": float(-np.mean(scores["test_log_loss"])),
    }


def setup_mlflow(tracking_uri: str = None) -> str:
    """Point MLflow to a local SQLite backend and set the experiment."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def run_experiments(X_train, y_train) -> List[Dict[str, Any]]:
    """Run every config of every family as an individual MLflow run."""
    results = []
    signature_input = X_train.astype("float64")
    input_example = signature_input.head(5)

    for family, spec in MODEL_FAMILIES.items():
        for params in expand_grid(spec["grid"]):
            model = spec["class"](**spec["fixed"], **params)
            run_name = f"{family}-" + "-".join(f"{k}={v}" for k, v in params.items())

            with mlflow.start_run(run_name=run_name) as run:
                metrics = cross_validate_config(model, X_train, y_train)

                # Final fit on the full training split for the logged artifact.
                model.fit(X_train, y_train)
                train_pred = model.predict(signature_input)
                signature = infer_signature(signature_input, train_pred)

                mlflow.log_params({"model_family": family, **spec["fixed"], **params})
                mlflow.log_metrics(metrics)
                mlflow.log_metrics({
                    "fit_train_f1_macro": f1_score(y_train, train_pred, average="macro"),
                    "fit_train_accuracy": accuracy_score(y_train, train_pred),
                    "fit_train_log_loss": log_loss(y_train, model.predict_proba(X_train)),
                })
                mlflow.set_tags({
                    "model_family": family,
                    "cv": "StratifiedKFold-5",
                    "dataset": "sklearn.load_wine",
                    "seed": str(SEED),
                    "stage": "experiment",
                })
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )
                results.append({
                    "run_id": run.info.run_id,
                    "run_name": run_name,
                    "model_family": family,
                    "params": params,
                    **metrics,
                })
                print(f"[{family}] {params} -> val_f1_macro={metrics['val_f1_macro']:.4f}")
    return results


def save_results_table(results: List[Dict[str, Any]], path: str = RESULTS_CSV) -> pd.DataFrame:
    """Persist Table 1 (hyperparameter search results) as CSV."""
    df = pd.DataFrame(results).drop(columns=["run_id"])
    df["params"] = df["params"].astype(str)
    df = df.sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def register_champion(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Register the best run (by val macro F1) and alias it as 'champion'."""
    best = max(results, key=lambda r: r["val_f1_macro"])
    client = MlflowClient()
    model_uri = f"runs:/{best['run_id']}/model"
    version = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version
    )
    client.set_tag(best["run_id"], "stage", "champion")
    print(
        f"Registered {REGISTERED_MODEL_NAME} v{version.version} "
        f"(run {best['run_id']}, val_f1_macro={best['val_f1_macro']:.4f}) "
        f"with alias '{CHAMPION_ALIAS}'."
    )
    return {**best, "version": version.version}


def run_pipeline(tracking_uri: str = None) -> Dict[str, Any]:
    """Full pipeline: data -> experiments -> registry. Returns champion info."""
    setup_mlflow(tracking_uri)
    X_train, _, y_train, _ = get_train_test_split()
    results = run_experiments(X_train, y_train)
    table = save_results_table(results)
    print("\nTable 1 - Hyperparameter search results\n")
    print(table.to_string(index=False))
    champion = register_champion(results)
    champion["n_runs"] = len(results)
    return champion


if __name__ == "__main__":
    run_pipeline()
