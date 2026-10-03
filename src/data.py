"""Data loading, validation and splitting for the Wine dataset."""
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

SEED = 42
EXPECTED_FEATURES = 13
TEST_SIZE = 0.2


def load_wine_frame() -> Tuple[pd.DataFrame, pd.Series]:
    """Return the Wine dataset as a (features DataFrame, target Series) pair."""
    bunch = load_wine(as_frame=True)
    return bunch.data, bunch.target


def validate_data(X: pd.DataFrame, y: pd.Series) -> bool:
    """Validate that there are no nulls and exactly 13 features.

    Raises ValueError on failure, returns True otherwise.
    """
    if X.isnull().values.any() or y.isnull().any():
        raise ValueError("Dataset contains null values.")
    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, found {X.shape[1]}."
        )
    if len(X) != len(y):
        raise ValueError("Feature and target lengths do not match.")
    return True


def get_train_test_split(
    test_size: float = TEST_SIZE, random_state: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, validate and return a stratified 80/20 split (random_state=42)."""
    X, y = load_wine_frame()
    validate_data(X, y)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
