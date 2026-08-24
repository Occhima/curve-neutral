from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors
import pytest
from curve_orchestration import (
    ForwardCurveOutput,
    build_dual_anchor_plan,
    finalize_forward_curve,
    neutralize_anchor_plan,
    wide_curve,
)
from pricer.curves.arbitrage import InfeasibleCurveError

pytestmark = pytest.mark.functional


def _market_problem(
    *,
    floor: float = 0.0,
    cap: float = 500.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tenors = pd.date_range("2027-01-01", "2028-12-01", freq="MS")
    curve = pd.DataFrame(
        {
            "tenor": tenors,
            "price": np.r_[np.full(12, 295.0), np.full(12, 999.0)],
            "index_factor": np.r_[np.full(12, 1.02), np.full(12, 9.0)],
            "floor": floor,
            "cap": cap,
        }
    )
    delivery = pd.DataFrame(
        [np.ones(12), np.r_[np.ones(3), np.zeros(9)]],
        index=["CAL27", "Q127"],
        columns=tenors[:12],
    )
    raw = pd.DataFrame(
        {"base": [295.0, 325.0], "stress": [300.0, 330.0]},
        index=delivery.index,
    )
    dcide = pd.DataFrame({"tenor": tenors[12:], "price": 280.0, "index_factor": 1.03})
    return curve, delivery, raw, dcide


def _forward(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
    raw: pd.DataFrame | pd.Series,
    dcide: pd.DataFrame,
) -> tuple[pd.DataFrame, object]:
    plan = build_dual_anchor_plan(
        curve,
        delivery,
        raw,
        raw * 1.02,
        cutoff="2027-12-31",
    )
    neutralized = neutralize_anchor_plan(plan)
    return finalize_forward_curve(plan, neutralized, dcide), neutralized


def test_market_is_neutralized_before_dcide_is_appended_and_indexed() -> None:
    curve, delivery, raw, dcide = _market_problem(floor=280.0, cap=340.0)

    forward, _ = _forward(curve, delivery, raw, dcide)

    raw_curve = wide_curve(forward, "raw_price")
    indexed_curve = wide_curve(forward, "indexed_price")
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
    assert forward.groupby("origin").size().to_dict() == {"dcide": 24, "market": 24}
    ForwardCurveOutput.validate(forward, lazy=True)


def test_complete_forward_pipeline_is_idempotent() -> None:
    curve, delivery, raw, dcide = _market_problem()

    first, _ = _forward(curve, delivery, raw, dcide)
    rebalanced = curve.assign(
        price=np.r_[
            wide_curve(first, "raw_price")["base"].iloc[:12], np.full(12, 999.0)
        ]
    )
    second, _ = _forward(rebalanced, delivery, raw, dcide)

    pd.testing.assert_frame_equal(second, first)


@pytest.mark.parametrize(("floor", "cap"), [(0.0, 320.0), (290.0, 500.0)])
def test_market_bounds_limit_the_soft_solution_instead_of_making_quotes_infeasible(
    floor: float,
    cap: float,
) -> None:
    curve, delivery, raw, dcide = _market_problem(floor=floor, cap=cap)

    forward, neutralized = _forward(curve, delivery, raw["base"], dcide)

    market = forward.query("origin == 'market'")
    assert market["raw_price"].between(floor, cap).all()
    assert neutralized.anchors["residual"].abs().max() > 0.0


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
        neutralize_anchor_plan(plan)


def test_wide_views_expose_one_column_per_scenario() -> None:
    curve, delivery, raw, dcide = _market_problem()

    forward, _ = _forward(curve, delivery, raw, dcide)

    assert wide_curve(forward, "raw_price").shape == (24, 2)
    assert wide_curve(forward, "indexed_price").columns.tolist() == ["base", "stress"]


def test_forward_output_contract_rejects_duplicate_scenario_tenors() -> None:
    duplicated = pd.DataFrame(
        {
            "tenor": ["2027-01-01", "2027-01-01"],
            "scenario": ["base", "base"],
            "origin": ["market", "market"],
            "raw_price": [100.0, 100.0],
            "index_factor": [1.0, 1.0],
            "indexed_price": [100.0, 100.0],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        ForwardCurveOutput.validate(duplicated, lazy=True)
