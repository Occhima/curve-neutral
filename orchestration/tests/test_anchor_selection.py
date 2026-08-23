from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import numpy as np
import pandas as pd
import pandera.errors
import pytest
from curve_orchestration import (
    AnchorDecisionOutput,
    AnchorDiagnosticOutput,
    atomic_block_labels,
    build_anchor_plan,
    build_dual_anchor_plan,
    build_exact_dual_anchor_plan,
    neutralize_anchor_plan,
    select_anchor_basis,
)
from pricer.curves.arbitrage import InfeasibleCurveError

pytestmark = pytest.mark.unit


def _hierarchy(tenors: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    delivery = pd.DataFrame(
        np.vstack(
            [
                np.ones(12),
                np.r_[np.ones(6), np.zeros(6)],
                np.r_[np.ones(3), np.zeros(9)],
                np.r_[np.zeros(3), np.ones(3), np.zeros(6)],
            ]
        ),
        index=["CAL27", "H127", "Q127", "Q227"],
        columns=tenors,
    )
    exposure = delivery.div(delivery.sum(axis="columns"), axis="index")
    return delivery, exposure


def test_atomic_blocks_are_local_non_overlapping_and_year_qualified() -> None:
    tenors = pd.date_range("2026-12-01", periods=5, freq="MS")
    cross_year = pd.DataFrame(
        [np.ones(5)],
        index=["CROSS_YEAR"],
        columns=tenors,
    )

    labels = atomic_block_labels(cross_year, tenors)

    assert labels.tolist() == [
        "ATOM|2026-12:2026-12",
        *(["ATOM|2027-01:2027-04"] * 4),
    ]


def test_uncovered_months_remain_independent() -> None:
    tenors = pd.date_range("2026-12-01", periods=5, freq="MS")
    delivery = pd.DataFrame(
        [[1.0, 1.0, 1.0]],
        index=["Q127"],
        columns=tenors[1:4],
    )

    labels = atomic_block_labels(delivery, tenors)

    assert labels.tolist() == [
        "MONTH|2026-12",
        *(["ATOM|2027-01:2027-03"] * 3),
        "MONTH|2027-04",
    ]


def test_q1_q2_make_h1_redundant_when_they_have_better_quality(
    tenors: pd.DatetimeIndex,
) -> None:
    _, exposure = _hierarchy(tenors)
    prices = pd.Series(
        [295.0, 305.0, 325.0, 285.0],
        index=exposure.index,
    )
    quality = pd.Series({"CAL27": 1.0, "H127": 2.0, "Q127": 9.0, "Q227": 8.0})

    result = select_anchor_basis(
        exposure,
        prices,
        quality=quality,
        mandatory=["CAL27"],
    )

    selected = result.decisions.query("selected")["product_id"].tolist()
    assert selected == ["CAL27", "Q127", "Q227"]
    assert result.decisions.set_index("product_id").loc["H127", "reason"] == "redundant"
    diagnostic = result.diagnostics.set_index("product_id")
    assert diagnostic.loc["H127", "implied_price"] == pytest.approx(305.0)
    assert diagnostic.loc["H127", "market_gap"] == pytest.approx(0.0)


def test_quality_can_make_h1_replace_q2(
    tenors: pd.DatetimeIndex,
) -> None:
    _, exposure = _hierarchy(tenors)
    prices = pd.Series(
        [295.0, 305.0, 325.0, 285.0],
        index=exposure.index,
    )
    quality = pd.Series({"CAL27": 1.0, "H127": 8.0, "Q127": 9.0, "Q227": 2.0})

    result = select_anchor_basis(
        exposure,
        prices,
        quality=quality,
        mandatory=["CAL27"],
    )

    assert result.decisions.query("selected")["product_id"].tolist() == [
        "CAL27",
        "H127",
        "Q127",
    ]
    assert not result.decisions.set_index("product_id").loc["Q227", "selected"]


def test_unreconciled_redundant_price_becomes_an_explicit_gap() -> None:
    exposure = pd.DataFrame(
        [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]],
        index=["CAL", "Q1", "Q2"],
    )
    prices = pd.Series([100.0, 90.0, 120.0], index=exposure.index)

    result = select_anchor_basis(
        exposure,
        prices,
        quality=pd.Series([1.0, 9.0, 1.0], index=exposure.index),
        mandatory=["CAL"],
    )

    assert result.anchors.prices.loc["Q1", "base"] == pytest.approx(90.0)
    diagnostic = result.diagnostics.set_index("product_id")
    assert diagnostic.loc["Q2", "implied_price"] == pytest.approx(110.0)
    assert diagnostic.loc["Q2", "market_gap"] == pytest.approx(-10.0)


def test_weighted_reconciliation_moves_the_less_reliable_product_more() -> None:
    exposure = pd.DataFrame(
        [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]],
        index=["CAL", "Q1", "Q2"],
    )
    prices = pd.DataFrame(
        {
            "base": [100.0, 90.0, 120.0],
            "stress": [110.0, 95.0, 135.0],
        },
        index=exposure.index,
    )
    quality = pd.Series([1.0, 9.0, 1.0], index=exposure.index)

    result = select_anchor_basis(
        exposure,
        prices,
        quality=quality,
        mandatory=["CAL"],
        reconcile=True,
    )

    np.testing.assert_allclose(result.anchors.prices.loc["CAL"], [100.0, 110.0])
    np.testing.assert_allclose(result.anchors.prices.loc["Q1"], [89.0, 94.0])
    diagnostic = result.diagnostics.set_index(["product_id", "scenario"])
    assert diagnostic.loc[("Q2", "base"), "anchor_price"] == pytest.approx(111.0)
    assert diagnostic.loc[("Q2", "stress"), "anchor_price"] == pytest.approx(126.0)
    np.testing.assert_allclose(
        result.diagnostics["implied_price"],
        result.diagnostics["anchor_price"],
        atol=1e-8,
    )


@pytest.mark.parametrize("reconcile", [False, True])
def test_inconsistent_mandatory_products_are_rejected(reconcile: bool) -> None:
    exposure = pd.DataFrame(
        [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]],
        index=["CAL", "Q1", "Q2"],
    )
    prices = pd.Series([100.0, 90.0, 120.0], index=exposure.index)

    with pytest.raises(InfeasibleCurveError):
        select_anchor_basis(
            exposure,
            prices,
            mandatory=exposure.index,
            reconcile=reconcile,
        )


def test_scalar_quality_and_identity_basis_are_valid_defaults() -> None:
    exposure = pd.DataFrame(np.eye(2), index=["M1", "M2"])
    prices = pd.Series([100.0, 110.0], index=exposure.index)

    result = select_anchor_basis(exposure, prices, quality=2.0)

    assert result.decisions["quality"].tolist() == [2.0, 2.0]
    assert result.decisions["selected"].all()
    assert set(result.decisions["reason"]) == {"independent"}


def test_plan_builds_q1_and_residual_and_preserves_other_years_exactly() -> None:
    tenors = pd.date_range("2026-08-01", "2028-12-01", freq="MS")
    curve = pd.DataFrame(
        {
            "tenor": tenors,
            "price": np.where(tenors.year == 2027, 295.0, 250.0),
        }
    )
    year = tenors[tenors.year == 2027]
    delivery = pd.DataFrame(
        [np.ones(12), np.r_[np.ones(3), np.zeros(9)]],
        index=["CAL27", "Q127"],
        columns=year,
    )
    prices = pd.Series([295.0, 325.0], index=delivery.index)

    plan = build_anchor_plan(
        curve,
        delivery,
        prices,
        quality=pd.Series({"CAL27": 1.0, "Q127": 5.0}),
        mandatory=["CAL27"],
    )
    first = neutralize_anchor_plan(plan)
    balanced = first.wide_curve()["base"]
    hours = year.days_in_month.to_numpy(dtype=float) * 24.0
    expected = (295.0 * hours.sum() - 325.0 * hours[:3].sum()) / hours[3:].sum()

    assert plan.active_tenors.equals(year)
    assert plan.curve.query("tenor.dt.year == 2027")[
        "block"
    ].value_counts().tolist() == [9, 3]
    np.testing.assert_allclose(balanced.loc[year[:3]], 325.0, atol=1e-9)
    np.testing.assert_allclose(balanced.loc[year[3:]], expected, atol=1e-9)
    np.testing.assert_array_equal(balanced.loc[tenors.year != 2027], 250.0)

    rebalanced = curve.assign(price=balanced.reindex(tenors).to_numpy())
    second = neutralize_anchor_plan(
        build_anchor_plan(
            rebalanced,
            delivery,
            prices,
            mandatory=["CAL27"],
        )
    )
    np.testing.assert_allclose(second.wide_curve()["base"], balanced, atol=1e-10)


def test_annual_only_override_creates_one_year_block(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    delivery = pd.DataFrame([np.ones(12)], index=["CAL27"], columns=tenors)
    plan = build_anchor_plan(
        canonical_curve.assign(price=np.linspace(280.0, 310.0, 12)),
        delivery,
        pd.Series([295.0], index=delivery.index),
        mandatory=["CAL27"],
    )

    result = neutralize_anchor_plan(plan)

    assert plan.curve["block"].nunique() == 1
    np.testing.assert_allclose(result.wide_curve()["base"], 295.0, atol=1e-9)


def test_plan_keeps_indexation_in_the_economic_rank_problem(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    curve = canonical_curve.assign(index_factor=np.linspace(1.0, 1.1, 12))
    delivery, _ = _hierarchy(tenors)
    prices = pd.Series([295.0, 300.0, 325.0, 275.0], index=delivery.index)
    factors = pd.Series(1.02, index=delivery.index)

    plan = build_anchor_plan(
        curve,
        delivery,
        prices,
        mandatory=["CAL27"],
        quote_index_factor=factors,
    )

    assert plan.anchors.exposure.index.tolist() == ["CAL27", "H127", "Q127"]
    assert plan.decisions.set_index("product_id").loc["Q227", "reason"] == "redundant"


def test_selection_and_plan_are_frozen_typed_data_contracts(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    delivery = pd.DataFrame([np.ones(12)], index=["CAL27"], columns=tenors)
    plan = build_anchor_plan(
        canonical_curve,
        delivery,
        pd.Series([100.0], index=delivery.index),
        mandatory=["CAL27"],
    )

    assert is_dataclass(plan)
    AnchorDecisionOutput.validate(plan.decisions, lazy=True)
    AnchorDiagnosticOutput.validate(plan.diagnostics, lazy=True)
    with pytest.raises(FrozenInstanceError):
        plan.active_tenors = tenors[:1]  # type: ignore[misc]


def test_decision_contract_rejects_duplicate_products() -> None:
    duplicated = pd.DataFrame(
        {
            "product_id": ["A", "A"],
            "mandatory": [True, False],
            "quality": [1.0, 1.0],
            "selected": [True, False],
            "reason": ["mandatory", "redundant"],
            "rank_before": [0, 1],
            "rank_after": [1, 1],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        AnchorDecisionOutput.validate(duplicated, lazy=True)


@pytest.mark.parametrize(
    ("exposure", "prices", "quality", "basis", "error"),
    [
        (
            pd.DataFrame([[1.0]], index=["A"]),
            pd.Series([100.0], index=["A"]),
            pd.Series([0.0], index=["A"]),
            None,
            "quality",
        ),
        (
            pd.DataFrame([[np.nan]], index=["A"]),
            pd.Series([100.0], index=["A"]),
            1.0,
            None,
            "Exposure",
        ),
        (
            pd.DataFrame([[1.0]], index=["A"]),
            pd.Series([100.0], index=["A"]),
            1.0,
            np.ones((2, 1)),
            "basis",
        ),
    ],
)
def test_selection_rejects_invalid_numerical_contracts(
    exposure: pd.DataFrame,
    prices: pd.Series,
    quality: pd.Series | float,
    basis: np.ndarray | None,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        select_anchor_basis(exposure, prices, quality=quality, basis=basis)


def test_selection_rejects_missing_mandatory_and_string_collisions() -> None:
    exposure = pd.DataFrame(np.eye(2), index=[1, "1"])
    prices = pd.Series([100.0, 110.0], index=exposure.index)

    with pytest.raises(ValueError, match="unique"):
        select_anchor_basis(exposure, prices)
    with pytest.raises(KeyError, match="MISSING"):
        select_anchor_basis(
            pd.DataFrame([[1.0]], index=["A"]),
            pd.Series([100.0], index=["A"]),
            mandatory=["MISSING"],
        )


@pytest.mark.parametrize(
    "delivery",
    [
        pd.DataFrame([[-1.0]], index=["A"], columns=["2027-01-01"]),
        pd.DataFrame([[0.0]], index=["A"], columns=["2027-01-01"]),
        pd.DataFrame(
            [[1.0, 1.0]],
            index=["A"],
            columns=["2027-01-01", "2027-01-15"],
        ),
        pd.DataFrame([[1.0], [1.0]], index=["A", "A"], columns=["2027-01-01"]),
    ],
)
def test_atomic_blocks_reject_invalid_delivery_contracts(
    delivery: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        atomic_block_labels(delivery, pd.date_range("2027-01-01", periods=1, freq="MS"))


def test_build_plan_does_not_mutate_inputs(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    curve = canonical_curve.copy()
    delivery = pd.DataFrame([np.ones(12)], index=["CAL27"], columns=tenors)
    prices = pd.Series([100.0], index=delivery.index)
    expected_curve = curve.copy(deep=True)
    expected_delivery = delivery.copy(deep=True)
    expected_prices = prices.copy(deep=True)

    build_anchor_plan(curve, delivery, prices, mandatory=["CAL27"])

    pd.testing.assert_frame_equal(curve, expected_curve)
    pd.testing.assert_frame_equal(delivery, expected_delivery)
    pd.testing.assert_series_equal(prices, expected_prices)


def test_cutoff_limits_the_anchor_problem_to_the_market_segment() -> None:
    tenors = pd.date_range("2027-01-01", "2028-12-01", freq="MS")
    curve = pd.DataFrame({"tenor": tenors, "price": 295.0})
    market_tenors = tenors[:12]
    delivery = pd.DataFrame(
        [np.ones(12)],
        index=["CAL27"],
        columns=market_tenors,
    )

    plan = build_anchor_plan(
        curve,
        delivery,
        pd.Series([295.0], index=delivery.index),
        mandatory=["CAL27"],
        cutoff="2027-12-31",
    )

    assert plan.cutoff == pd.Timestamp("2027-12-01")
    assert plan.curve["tenor"].tolist() == market_tenors.tolist()
    assert plan.active_tenors.equals(market_tenors)


def test_soft_dual_plan_fits_one_raw_state_to_both_price_surfaces(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    factor = 1.02
    delivery = pd.DataFrame(
        [np.ones(12), np.r_[np.ones(3), np.zeros(9)]],
        index=["CAL27", "Q127"],
        columns=tenors,
    )
    raw = pd.Series([295.0, 325.0], index=delivery.index)

    plan = build_dual_anchor_plan(
        canonical_curve.assign(index_factor=factor),
        delivery,
        raw,
        raw * factor,
    )
    result = neutralize_anchor_plan(plan, prior_strength=0.0)

    assert plan.anchors.exposure.index.tolist() == [
        "raw:CAL27",
        "raw:Q127",
        "indexed:CAL27",
        "indexed:Q127",
    ]
    assert set(plan.decisions["reason"]) == {"soft"}
    np.testing.assert_allclose(result.anchors["residual"], 0.0, atol=1e-7)


def test_dual_plan_rejects_misaligned_price_surfaces(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    delivery = pd.DataFrame([np.ones(12)], index=["CAL27"], columns=tenors)

    with pytest.raises(ValueError, match="identical labels"):
        build_dual_anchor_plan(
            canonical_curve,
            delivery,
            pd.DataFrame({"base": [295.0]}, index=delivery.index),
            pd.DataFrame({"stress": [300.0]}, index=delivery.index),
        )


def test_exact_dual_plan_remains_available_as_an_explicit_audit_mode(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    factor = 1.02
    delivery = pd.DataFrame([np.ones(12)], index=["CAL27"], columns=tenors)
    raw = pd.Series([295.0], index=delivery.index)

    plan = build_exact_dual_anchor_plan(
        canonical_curve.assign(index_factor=factor),
        delivery,
        raw,
        raw * factor,
        mandatory=["CAL27"],
    )

    assert plan.anchors.lower is None
    assert plan.anchors.exposure.index.tolist() == [
        "raw:CAL27",
        "indexed:CAL27",
    ]


def test_inconsistent_mandatory_raw_and_indexed_marks_are_infeasible() -> None:
    tenors = pd.date_range("2027-01-01", periods=2, freq="MS")
    curve = pd.DataFrame(
        {
            "tenor": tenors,
            "price": 100.0,
            "index_factor": [1.0, 2.0],
        }
    )
    delivery = pd.DataFrame([np.ones(2)], index=["CAL27"], columns=tenors)

    with pytest.raises(InfeasibleCurveError):
        build_exact_dual_anchor_plan(
            curve,
            delivery,
            pd.Series([100.0], index=delivery.index),
            pd.Series([160.0], index=delivery.index),
            mandatory=["CAL27"],
        )


def test_soft_dual_plan_returns_the_weighted_least_squares_compromise() -> None:
    tenor = pd.date_range("2027-01-01", periods=1, freq="MS")
    curve = pd.DataFrame({"tenor": tenor, "price": 0.0, "index_factor": 2.0})
    delivery = pd.DataFrame([[1.0]], index=["M127"], columns=tenor)

    result = neutralize_anchor_plan(
        build_dual_anchor_plan(
            curve,
            delivery,
            pd.Series([100.0], index=delivery.index),
            pd.Series([220.0], index=delivery.index),
        ),
        prior_strength=0.0,
    )

    assert result.wide_curve().loc[tenor[0], "base"] == pytest.approx(108.0)
    residuals = result.anchors.set_index("anchor")["residual"]
    assert residuals["raw:M127"] == pytest.approx(8.0)
    assert residuals["indexed:M127"] == pytest.approx(-4.0)


def test_surface_weights_control_the_soft_dual_compromise() -> None:
    tenor = pd.date_range("2027-01-01", periods=1, freq="MS")
    curve = pd.DataFrame({"tenor": tenor, "price": 0.0, "index_factor": 2.0})
    delivery = pd.DataFrame([[1.0]], index=["M127"], columns=tenor)
    plan = build_dual_anchor_plan(
        curve,
        delivery,
        pd.Series([100.0], index=delivery.index),
        pd.Series([220.0], index=delivery.index),
        raw_weight=4.0,
    )

    result = neutralize_anchor_plan(plan, prior_strength=0.0)

    assert result.wide_curve().loc[tenor[0], "base"] == pytest.approx(105.0)
