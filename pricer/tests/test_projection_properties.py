from __future__ import annotations

import numpy as np
import pytest
from pricer.curves.arbitrage import LinearObservations, solve_curve

pytestmark = pytest.mark.unit


def test_curve_anchored_to_its_implied_product_vector_is_identity(
    liquidity_regime_curve: np.ndarray,
    liquidity_regime_basis: np.ndarray,
    liquidity_regime_exposure: np.ndarray,
) -> None:
    implied_anchors = liquidity_regime_exposure @ liquidity_regime_curve

    result = solve_curve(
        liquidity_regime_curve,
        LinearObservations.exact(liquidity_regime_exposure, implied_anchors),
        basis=liquidity_regime_basis,
    )

    np.testing.assert_allclose(result.prices, liquidity_regime_curve, atol=1e-10)


def test_exact_projection_is_idempotent(
    initial_curve: np.ndarray,
    annual_exposure: np.ndarray,
) -> None:
    observations = LinearObservations.exact(annual_exposure, [295.0])

    first = solve_curve(initial_curve, observations)
    second = solve_curve(first.prices, observations)

    np.testing.assert_allclose(second.prices, first.prices, atol=1e-10)


def test_exact_projection_is_idempotent_for_each_price_scenario(
    initial_curve: np.ndarray,
    annual_q1_exposure: np.ndarray,
    residual_basis: np.ndarray,
) -> None:
    prices = np.array(
        [
            [295.0, 305.0, 285.0],
            [325.0, 315.0, 300.0],
        ]
    )
    batched = solve_curve(
        initial_curve,
        LinearObservations.exact(annual_q1_exposure, prices),
        basis=residual_basis,
    )

    repeated = np.column_stack(
        [
            solve_curve(
                batched.prices[:, scenario],
                LinearObservations.exact(
                    annual_q1_exposure,
                    prices[:, scenario],
                ),
                basis=residual_basis,
            ).prices
            for scenario in range(prices.shape[1])
        ]
    )

    np.testing.assert_allclose(repeated, batched.prices, atol=1e-10)


def test_soft_calibration_is_explicitly_not_an_idempotent_projection() -> None:
    observations = LinearObservations.soft([[1.0]], [110.0], weight=1.0)

    first = solve_curve([90.0], observations)
    second = solve_curve(first.prices, observations)

    np.testing.assert_allclose(first.prices, [100.0], atol=1e-10)
    np.testing.assert_allclose(second.prices, [105.0], atol=1e-10)
    assert not np.allclose(second.prices, first.prices)
