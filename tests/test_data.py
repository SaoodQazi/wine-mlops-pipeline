import pandas as pd
import pytest

from src.data import (
    EXPECTED_FEATURES, get_train_test_split, load_wine_frame, validate_data,
)


def test_dataset_shape():
    X, y = load_wine_frame()
    assert X.shape == (178, EXPECTED_FEATURES)
    assert len(y) == 178
    assert set(y.unique()) == {0, 1, 2}


def test_validation_passes_on_clean_data():
    X, y = load_wine_frame()
    assert validate_data(X, y) is True


def test_validation_fails_on_nulls():
    X, y = load_wine_frame()
    X = X.copy()
    X.iloc[0, 0] = None
    with pytest.raises(ValueError, match="null"):
        validate_data(X, y)


def test_validation_fails_on_wrong_feature_count():
    X, y = load_wine_frame()
    with pytest.raises(ValueError, match="features"):
        validate_data(X.iloc[:, :12], y)


def test_split_is_80_20_and_stratified():
    X_train, X_test, y_train, y_test = get_train_test_split()
    assert len(X_train) + len(X_test) == 178
    assert abs(len(X_test) / 178 - 0.2) < 0.01
    _, y = load_wine_frame()
    full = y.value_counts(normalize=True).sort_index()
    train = y_train.value_counts(normalize=True).sort_index()
    assert (full - train).abs().max() < 0.03


def test_split_is_reproducible():
    a = get_train_test_split()
    b = get_train_test_split()
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_series_equal(a[3], b[3])
