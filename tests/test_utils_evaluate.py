from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from test_confounding.utils_evaluate import get_quantile_regressor


@pytest.fixture
def mock_quantile_reg():
    # Create a mock Quantile Regressor
    quant_reg = MagicMock()
    quant_reg.fit.return_value = None
    return quant_reg


def create_mock_search_cv(best_score):
    mock_search_cv = MagicMock()
    mock_search_cv.fit.return_value = mock_search_cv
    mock_search_cv.best_score_ = best_score
    mock_search_cv.best_estimator_ = MagicMock()
    return mock_search_cv


@patch("test_confounding.utils_evaluate.RandomizedSearchCV")
@patch("test_confounding.utils_evaluate.GridSearchCV")
def test_get_quantile_regressor(mock_grid_search, mock_random_search):
    # Set mock return values
    mock_random_search.return_value = create_mock_search_cv(-1)
    mock_grid_search.return_value = create_mock_search_cv(-2)

    # Mock Data
    X = np.array([[1, 2], [1, 3], [1, 4]])
    y = np.array([1, 2, 3])
    tau = 0.5

    # Call the function
    best_model = get_quantile_regressor(X, y, tau)

    # Assertions
    assert best_model is not None
    mock_random_search.assert_called()
    mock_grid_search.assert_called()
