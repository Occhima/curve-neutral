from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, fields, is_dataclass

import numpy as np
import pytest
from pricer.curves import arbitrage
from pricer.curves.arbitrage import CurveSolution, LinearObservations

pytestmark = pytest.mark.unit


def test_exact_observation_constructor_is_an_exact_data_contract(
    annual_exposure: np.ndarray,
) -> None:
    observations = LinearObservations.exact(annual_exposure, [100.0])

    assert is_dataclass(observations)
    assert observations.matrix is annual_exposure
    assert observations.values == [100.0]
    assert observations.lower is None
    assert observations.upper is None
    assert observations.weight is None


def test_soft_observation_constructor_sets_only_objective_weight(
    annual_exposure: np.ndarray,
) -> None:
    observations = LinearObservations.soft(
        annual_exposure,
        [100.0],
        weight=3.0,
    )

    assert observations.lower == -np.inf
    assert observations.upper == np.inf
    assert observations.weight == 3.0


def test_bounded_observation_constructor_carries_bounds_and_weight(
    annual_exposure: np.ndarray,
) -> None:
    observations = LinearObservations.bounded(
        annual_exposure,
        [100.0],
        lower=[98.0],
        upper=[102.0],
        weight=[2.0],
    )

    assert observations.lower == [98.0]
    assert observations.upper == [102.0]
    assert observations.weight == [2.0]


def test_observation_contract_is_frozen(
    annual_exposure: np.ndarray,
) -> None:
    observations = LinearObservations.exact(annual_exposure, [100.0])

    with pytest.raises(FrozenInstanceError):
        observations.values = [101.0]  # type: ignore[misc]


def test_curve_solution_is_only_a_result_carrier() -> None:
    solution = CurveSolution(
        prices=np.array([100.0, 101.0]),
        fitted=np.array([100.5]),
        residuals=np.array([0.5]),
        latent=np.array([100.0, 101.0]),
        objective=1.0,
        max_violation=0.0,
    )

    assert is_dataclass(solution)
    assert {field.name for field in fields(solution)} == {
        "prices",
        "fitted",
        "residuals",
        "latent",
        "objective",
        "max_violation",
    }
    assert not callable(solution.prices)


def test_pricer_core_has_no_dataframe_or_market_dependencies() -> None:
    tree = ast.parse(inspect.getsource(arbitrage))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert imports.isdisjoint({"pandas", "pandera", "curve_orchestration", "datetime"})
