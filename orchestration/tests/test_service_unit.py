from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration.neutralization import (
    AnchorMatrix,
    build_anchor_matrix,
    neutralize_curve,
)
from pricer.curves.arbitrage import CurveSolution, LinearObservations
from pytest_mock import MockerFixture

pytestmark = pytest.mark.unit


def test_build_anchor_matrix_aligns_frames_and_delegates_exposure_construction(
    mocker: MockerFixture,
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    delivery = pd.DataFrame(
        [np.ones(12)],
        index=["CAL27"],
        columns=tenors,
    )
    prices = pd.Series([100.0], index=delivery.index)
    expected = np.full((1, 12), 1.0 / 12.0)
    cashflow = mocker.patch(
        "curve_orchestration.neutralization.cashflow_matrix",
        return_value=expected,
    )

    anchors = build_anchor_matrix(canonical_curve, delivery, prices)

    np.testing.assert_allclose(anchors.exposure, expected)
    cashflow.assert_called_once()
    call = cashflow.call_args
    np.testing.assert_array_equal(call.args[0], delivery.to_numpy())
    np.testing.assert_array_equal(
        call.kwargs["monthly_weight"],
        canonical_curve["energy_weight"].to_numpy(),
    )


def test_neutralize_curve_delegates_only_arrays_to_pricer(
    mocker: MockerFixture,
    block_curve: pd.DataFrame,
    exact_anchors: AnchorMatrix,
) -> None:
    expected_prices = np.r_[np.full(3, 90.0), np.full(9, 130.0)][:, None]
    solver = mocker.patch(
        "curve_orchestration.neutralization.solve_curve",
        return_value=CurveSolution(
            prices=expected_prices,
            fitted=np.array([[120.0], [90.0]]),
            residuals=np.zeros((2, 1)),
            latent=np.array([[90.0], [130.0]]),
            objective=np.array([1.0]),
            max_violation=np.array([0.0]),
        ),
    )

    result = neutralize_curve(block_curve, exact_anchors)

    solver.assert_called_once()
    initial, observations = solver.call_args.args
    assert isinstance(observations, LinearObservations)
    np.testing.assert_array_equal(initial, block_curve["price"].to_numpy())
    np.testing.assert_array_equal(observations.values, [[120.0], [90.0]])
    assert solver.call_args.kwargs["basis"].shape == (12, 2)
    np.testing.assert_array_equal(result.wide_curve()["base"], expected_prices[:, 0])
