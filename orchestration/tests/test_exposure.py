from __future__ import annotations

import numpy as np
import pytest
from curve_orchestration.neutralization import cashflow_matrix

pytestmark = pytest.mark.unit


def test_cashflow_matrix_combines_weight_and_price_corrections() -> None:
    delivery = np.array([[1.0, 1.0, 0.0], [0.0, 0.5, 1.0]])
    monthly_weight = np.array([2.0, 3.0, 5.0])
    monthly_factor = np.array([1.0, 1.1, 1.2])
    quote_factor = np.array([1.05, 1.10])
    prices = np.array([90.0, 100.0, 110.0])

    matrix = cashflow_matrix(
        delivery,
        monthly_weight=monthly_weight,
        monthly_price_factor=monthly_factor,
        quote_price_factor=quote_factor,
    )
    expected = (delivery * monthly_weight * monthly_factor * prices).sum(axis=1) / (
        (delivery * monthly_weight).sum(axis=1) * quote_factor
    )

    np.testing.assert_allclose(matrix @ prices, expected, atol=1e-12)


def test_cashflow_rows_sum_to_one_without_price_corrections() -> None:
    generator = np.random.default_rng(7)
    delivery = generator.uniform(0.1, 1.0, size=(8, 12))
    matrix = cashflow_matrix(
        delivery,
        monthly_weight=generator.uniform(1.0, 5.0, 12),
    )

    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-12)


def test_empty_delivery_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive delivered weight"):
        cashflow_matrix(np.zeros((1, 4)))


def test_negative_delivery_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive delivered weight"):
        cashflow_matrix([[1.0, -0.5, 1.0]])
