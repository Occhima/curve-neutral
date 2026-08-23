from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import numpy as np
import pandas as pd
import pytest
from curve_orchestration.neutralization import (
    AnchorMatrix,
    AnchorOutput,
    CurveInput,
    CurveOutput,
    NeutralizationResult,
    ScenarioOutput,
)

pytestmark = pytest.mark.unit


def test_anchor_matrix_constructors_are_data_contracts(
    anchor_exposure: pd.DataFrame,
    anchor_prices: pd.Series,
) -> None:
    exact = AnchorMatrix.exact(anchor_exposure, anchor_prices)
    soft = AnchorMatrix.soft(anchor_exposure, anchor_prices, weight=3.0)
    bounded = AnchorMatrix.bounded(
        anchor_exposure,
        anchor_prices,
        lower=98.0,
        upper=102.0,
        weight=2.0,
    )

    assert is_dataclass(exact)
    assert exact.exposure is anchor_exposure
    assert exact.prices is anchor_prices
    assert (soft.lower, soft.upper, soft.weight) == (-np.inf, np.inf, 3.0)
    assert (bounded.lower, bounded.upper, bounded.weight) == (98.0, 102.0, 2.0)


def test_anchor_matrix_is_frozen(
    exact_anchors: AnchorMatrix,
) -> None:
    with pytest.raises(FrozenInstanceError):
        exact_anchors.weight = 2.0  # type: ignore[misc]


def test_pandera_contracts_validate_each_public_frame(
    canonical_curve: pd.DataFrame,
) -> None:
    curve_output = pd.DataFrame(
        {
            "tenor": canonical_curve["tenor"].iloc[:2],
            "scenario": ["base", "base"],
            "initial_price": [100.0, 100.0],
            "price": [99.0, 101.0],
        }
    )
    anchor_output = pd.DataFrame(
        {
            "anchor": ["CAL27"],
            "scenario": ["base"],
            "target": [100.0],
            "fitted": [100.0],
            "residual": [0.0],
        }
    )
    scenario_output = pd.DataFrame(
        {
            "scenario": ["base"],
            "objective": [1.0],
            "max_violation": [0.0],
        }
    )

    CurveInput.validate(canonical_curve, lazy=True)
    CurveOutput.validate(curve_output, lazy=True)
    AnchorOutput.validate(anchor_output, lazy=True)
    ScenarioOutput.validate(scenario_output, lazy=True)


def test_neutralization_result_carries_outputs_and_offers_a_bound_view() -> None:
    curve = pd.DataFrame(
        {
            "tenor": pd.to_datetime(["2027-01-01", "2027-02-01"]),
            "scenario": ["base", "base"],
            "initial_price": [100.0, 100.0],
            "price": [99.0, 101.0],
        }
    )
    anchors = pd.DataFrame(
        {
            "anchor": ["CAL27"],
            "scenario": ["base"],
            "target": [100.0],
            "fitted": [100.0],
            "residual": [0.0],
        }
    )
    scenarios = pd.DataFrame(
        {
            "scenario": ["base"],
            "objective": [1.0],
            "max_violation": [0.0],
        }
    )
    result = NeutralizationResult(curve=curve, anchors=anchors, scenarios=scenarios)

    assert is_dataclass(result)
    pd.testing.assert_series_equal(
        result.wide_curve()["base"],
        curve.set_index("tenor")["price"].rename("base"),
        check_names=True,
    )
