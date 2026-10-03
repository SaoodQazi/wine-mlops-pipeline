# Wine MLOps Pipeline

![CI](https://github.com/<YOUR_USERNAME>/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

End-to-end, reproducible MLOps pipeline on `sklearn.datasets.load_wine`
(178 samples, 13 features, 3 classes) with Makefile automation, MLflow tracking and
registry, and a GitHub Actions CI pipeline with an automated Model Quality Gate.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10 recommended
make install    # upgrade pip + install pinned requirements
make lint       # flake8, max line length 100
make test       # pytest -v (unit tests + model quality gate)
make train      # run 2 model families x configs, log to MLflow, register champion
make evaluate   # load models:/WineClassifier@champion, score the test split
make clean      # remove caches / bytecode / local MLflow store
```

View the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

## Design

| Component | Details |
|---|---|
| Data | Stratified 80/20 split, `random_state=42`, null and 13-feature validation |
| Models | RandomForest (6 configs), GradientBoosting (12 configs) |
| CV | 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on the train split |
| Metrics | Macro F1, Accuracy, Log Loss (train and validation) |
| Tracking | MLflow SQLite backend, experiment `Wine-Cultivar-Classification` |
| Registry | Best run by validation macro F1 -> `WineClassifier`, alias `champion` |
| Gate | Val macro F1 >= 0.88, batch latency <= 30 ms, labels in {0, 1, 2} |

All randomness uses seed **42**.

## Structure

```
.github/workflows/ci.yml   CI: checkout, python 3.10, make install/lint/test
src/data.py                loading, validation, split
src/train.py               grid search + MLflow + registry
src/evaluate.py            inference verification on test split
tests/test_data.py         data unit tests
tests/test_model_gate.py   MLOps quality gate
```

## Branching

`main` is protected/clean. Work happens on `feature/*` branches merged via Pull Requests.
