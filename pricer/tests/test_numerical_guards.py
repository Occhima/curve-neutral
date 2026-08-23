from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pricer.curves.arbitrage import LinearObservations, solve_curve
from pytest_mock import MockerFixture

pytestmark = pytest.mark.unit


def test_optimizer_failure_is_reported_at_the_scipy_boundary(
    mocker: MockerFixture,
    annual_exposure: np.ndarray,
) -> None:
    mocker.patch(
        "pricer.curves.arbitrage.minimize",
        return_value=SimpleNamespace(success=False, message="synthetic failure"),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        solve_curve(
            np.full(12, 90.0),
            LinearObservations.soft(annual_exposure, [110.0]),
        )


def test_post_solver_violation_guard_rejects_a_bad_scipy_result(
    mocker: MockerFixture,
    annual_exposure: np.ndarray,
) -> None:
    mocker.patch(
        "pricer.curves.arbitrage.minimize",
        return_value=SimpleNamespace(success=True, x=np.full(12, 200.0)),
    )

    with pytest.raises(RuntimeError, match="violated a restriction"):
        solve_curve(
            np.full(12, 90.0),
            LinearObservations.soft(annual_exposure, [110.0]),
            cap=150.0,
        )
