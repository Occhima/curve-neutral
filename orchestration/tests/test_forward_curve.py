from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pandera.errors
import pytest
from curve_orchestration import (
    ForwardCurveOutput,
    build_dual_anchor_plan,
    neutralize_anchor_plan,
    neutralize_forward_curve,
)
from pricer.curves.arbitrage import InfeasibleCurveError

pytestmark = pytest.mark.functional


def _market_problem(
    *,
    floor: float = 0.0,
    cap: float = 500.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tenors = pd.date_range("2027-01-01", "2028-12-01", freq="MS")
    market_tenors = tenors[:12]
    factor = 1.02
    curve = pd.DataFrame(
        {
            "tenor": tenors,
            "price": np.r_[np.full(12, 295.0), np.full(12, 999.0)],
            "index_factor": np.r_[np.full(12, factor), np.full(12, 9.0)],
            "floor": floor,
            "cap": cap,
        }
    )
    delivery = pd.DataFrame(
        [np.ones(12), np.r_[np.ones(3), np.zeros(9)]],
        index=["CAL27", "Q127"],
        columns=market_tenors,
    )
    raw = pd.DataFrame(
        {
            "base": [295.0, 325.0],
            "stress": [300.0, 330.0],
        },
        index=delivery.index,
    )
    dcide = pd.DataFrame(
        {
            "tenor": tenors[12:],
            "price": 280.0,
            "index_factor": 1.03,
        }
    )
    return curve, delivery, raw, dcide


def test_market_is_neutralized_before_dcide_is_appended_and_indexed() -> None:
    curve, delivery, raw, dcide = _market_problem(floor=280.0, cap=340.0)
    plan = build_dual_anchor_plan(
        curve,
        delivery,
        raw,
        raw * 1.02,
        cutoff="2027-12-31",
    )

    result = neutralize_forward_curve(plan, dcide)
    raw_curve = result.wide_raw_curve()
    indexed_curve = result.wide_indexed_curve()
    market_tenors = raw_curve.index[:12]
    hours = market_tenors.days_in_month.to_numpy(dtype=float) * 24.0
    residual = (295.0 * hours.sum() - 325.0 * hours[:3].sum()) / hours[3:].sum()

    np.testing.assert_allclose(raw_curve.loc[market_tenors[:3], "base"], 325.0)
    np.testing.assert_allclose(raw_curve.loc[market_tenors[3:], "base"], residual)
    np.testing.assert_allclose(raw_curve.iloc[12:], 280.0)
    np.testing.assert_allclose(
        indexed_curve.loc[market_tenors, "base"],
        raw_curve.loc[market_tenors, "base"] * 1.02,
    )
    np.testing.assert_allclose(indexed_curve.iloc[12:], 280.0 * 1.03)
    assert result.curve.groupby("source").size().to_dict() == {
        "dcide": 24,
        "market": 24,
    }
    ForwardCurveOutput.validate(result.curve, lazy=True)


def test_complete_forward_pipeline_is_idempotent() -> None:
    curve, delivery, raw, dcide = _market_problem()
    first_plan = build_dual_anchor_plan(
        curve,
        delivery,
        raw,
        raw * 1.02,
        cutoff="2027-12-31",
    )
    first = neutralize_forward_curve(first_plan, dcide)
    rebalanced = curve.assign(
        price=np.r_[first.wide_raw_curve()["base"].iloc[:12], np.full(12, 999.0)]
    )

    second = neutralize_forward_curve(
        build_dual_anchor_plan(
            rebalanced,
            delivery,
            raw,
            raw * 1.02,
            cutoff="2027-12-31",
        ),
        dcide,
    )

    pd.testing.assert_frame_equal(second.curve, first.curve)


@pytest.mark.parametrize(
    ("floor", "cap"),
    [
        (0.0, 320.0),
        (290.0, 500.0),
    ],
)
def test_market_bounds_limit_the_soft_solution_instead_of_making_quotes_infeasible(
    floor: float,
    cap: float,
) -> None:
    curve, delivery, raw, dcide = _market_problem(floor=floor, cap=cap)
    plan = build_dual_anchor_plan(
        curve,
        delivery,
        raw["base"],
        raw["base"] * 1.02,
        cutoff="2027-12-31",
    )

    result = neutralize_forward_curve(plan, dcide)
    market = result.curve.query("source == 'market'")

    assert market["raw_price"].between(floor, cap).all()
    assert result.neutralization.anchors["residual"].abs().max() > 0.0


def test_shared_block_can_make_monthly_bounds_themselves_infeasible() -> None:
    tenors = pd.date_range("2027-01-01", periods=2, freq="MS")
    curve = pd.DataFrame(
        {
            "tenor": tenors,
            "price": 105.0,
            "floor": [110.0, 0.0],
            "cap": [120.0, 100.0],
        }
    )
    delivery = pd.DataFrame([np.ones(2)], index=["CAL27"], columns=tenors)
    plan = build_dual_anchor_plan(
        curve,
        delivery,
        pd.Series([105.0], index=delivery.index),
        pd.Series([105.0], index=delivery.index),
    )

    with pytest.raises(InfeasibleCurveError):
        neutralize_anchor_plan(plan, prior_strength=0.0)


def test_forward_result_is_frozen_and_exposes_wide_views() -> None:
    curve, delivery, raw, dcide = _market_problem()
    result = neutralize_forward_curve(
        build_dual_anchor_plan(
            curve,
            delivery,
            raw["base"],
            raw["base"] * 1.02,
            cutoff="2027-12-31",
        ),
        dcide,
    )

    assert result.wide_raw_curve().shape == (24, 1)
    assert result.wide_indexed_curve().shape == (24, 1)
    with pytest.raises(FrozenInstanceError):
        result.curve = result.curve.iloc[:1]  # type: ignore[misc]


def test_forward_output_contract_rejects_duplicate_scenario_tenors() -> None:
    duplicated = pd.DataFrame(
        {
            "tenor": ["2027-01-01", "2027-01-01"],
            "scenario": ["base", "base"],
            "source": ["market", "market"],
            "raw_price": [100.0, 100.0],
            "index_factor": [1.0, 1.0],
            "indexed_price": [100.0, 100.0],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        ForwardCurveOutput.validate(duplicated, lazy=True)
