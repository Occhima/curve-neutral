from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration.neutralization import (
    AnchorMatrix,
    build_anchor_matrix,
    build_raw_and_indexed_anchor_matrix,
    neutralize_curve,
)
from pricer.curves.arbitrage import InfeasibleCurveError

pytestmark = pytest.mark.functional


def test_delivery_to_balanced_curve_flow_without_product_discovery(
    block_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
    anchor_prices: pd.Series,
) -> None:
    delivery = pd.DataFrame(
        np.vstack(
            [
                np.ones(12),
                np.r_[np.ones(3), np.zeros(9)],
            ]
        ),
        index=anchor_prices.index,
        columns=tenors,
    )

    anchors = build_anchor_matrix(block_curve, delivery, anchor_prices)
    result = neutralize_curve(block_curve, anchors)

    hours = tenors.days_in_month.to_numpy(dtype=float) * 24.0
    expected_residual = (120.0 * hours.sum() - 90.0 * hours[:3].sum()) / hours[3:].sum()
    balanced = result.wide_curve()["base"]
    np.testing.assert_allclose(balanced.iloc[:3], 90.0, atol=1e-8)
    np.testing.assert_allclose(balanced.iloc[3:], expected_residual, atol=1e-8)
    np.testing.assert_allclose(result.anchors["residual"], 0.0, atol=1e-8)


def test_matrix_prices_run_multiple_market_scenarios_end_to_end(
    block_curve: pd.DataFrame,
    anchor_exposure: pd.DataFrame,
) -> None:
    prices = pd.DataFrame(
        {
            "base": [120.0, 90.0],
            "stress": [130.0, 110.0],
        },
        index=anchor_exposure.index,
    )

    result = neutralize_curve(
        block_curve,
        anchors=AnchorMatrix.exact(anchor_exposure, prices),
    )

    assert set(result.curve["scenario"]) == {"base", "stress"}
    fitted = result.anchors.pivot(index="anchor", columns="scenario", values="fitted")
    np.testing.assert_allclose(fitted.reindex_like(prices), prices, atol=1e-8)


def test_self_anchored_liquidity_regimes_are_an_identity_transformation(
    liquidity_regime_frame: pd.DataFrame,
    liquidity_regime_exposure: pd.DataFrame,
) -> None:
    initial = liquidity_regime_frame.set_index("tenor")["price"]
    implied_prices = liquidity_regime_exposure @ initial

    result = neutralize_curve(
        liquidity_regime_frame,
        AnchorMatrix.exact(liquidity_regime_exposure, implied_prices),
    )

    np.testing.assert_allclose(result.wide_curve()["base"], initial, atol=1e-10)


def test_exact_neutralization_is_idempotent_through_the_pandas_service(
    block_curve: pd.DataFrame,
    exact_anchors: AnchorMatrix,
) -> None:
    first = neutralize_curve(block_curve, exact_anchors)
    rebalanced_input = block_curve.assign(price=first.wide_curve()["base"].to_numpy())

    second = neutralize_curve(rebalanced_input, exact_anchors)

    np.testing.assert_allclose(
        second.wide_curve()["base"],
        first.wide_curve()["base"],
        atol=1e-10,
    )


def test_raw_and_indexed_anchors_are_exact_and_idempotent_together(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    common_factor = 300.83 / 295.0
    curve = canonical_curve.assign(
        price=295.0,
        index_factor=common_factor,
        block=["Q1"] * 3 + ["residual"] * 9,
    )
    delivery = pd.DataFrame(
        np.vstack(
            [
                np.ones(12),
                np.r_[np.ones(3), np.zeros(9)],
            ]
        ),
        index=["CAL27", "Q127"],
        columns=tenors,
    )
    raw = pd.Series([295.0, 325.0], index=delivery.index, name="base")
    indexed = raw * common_factor
    anchors = build_raw_and_indexed_anchor_matrix(curve, delivery, raw, indexed)

    first = neutralize_curve(curve, anchors)
    rebalanced = curve.assign(price=first.wide_curve()["base"].to_numpy())
    second = neutralize_curve(rebalanced, anchors)

    hours = tenors.days_in_month.to_numpy(dtype=float) * 24.0
    expected_residual = (295.0 * hours.sum() - 325.0 * hours[:3].sum()) / hours[
        3:
    ].sum()
    balanced = first.wide_curve()["base"]
    np.testing.assert_allclose(balanced.iloc[:3], 325.0, atol=1e-9)
    np.testing.assert_allclose(balanced.iloc[3:], expected_residual, atol=1e-9)
    np.testing.assert_allclose(first.anchors["residual"], 0.0, atol=1e-9)
    np.testing.assert_allclose(second.wide_curve()["base"], balanced, atol=1e-10)
    assert first.anchors["anchor"].tolist() == [
        "raw:CAL27",
        "raw:Q127",
        "indexed:CAL27",
        "indexed:Q127",
    ]


def test_raw_and_indexed_builder_accepts_price_scenario_matrices(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    factor = 1.02
    curve = canonical_curve.assign(
        index_factor=factor,
        block=["Q1"] * 3 + ["residual"] * 9,
    )
    delivery = pd.DataFrame(
        np.vstack([np.ones(12), np.r_[np.ones(3), np.zeros(9)]]),
        index=["CAL27", "Q127"],
        columns=tenors,
    )
    raw = pd.DataFrame(
        {"base": [295.0, 325.0], "stress": [300.0, 330.0]},
        index=delivery.index,
    )

    result = neutralize_curve(
        curve,
        build_raw_and_indexed_anchor_matrix(curve, delivery, raw, raw * factor),
    )

    assert result.wide_curve().columns.tolist() == ["base", "stress"]
    np.testing.assert_allclose(result.anchors["residual"], 0.0, atol=1e-9)


def test_incompatible_raw_and_indexed_anchors_fail_instead_of_compromising(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    annual_raw = 295.0
    january_factor = 300.83 / annual_raw
    curve = canonical_curve.assign(
        index_factor=np.r_[np.full(3, january_factor), np.full(9, 1.025)],
        block=["Q1"] * 3 + ["residual"] * 9,
    )
    delivery = pd.DataFrame(
        np.vstack([np.ones(12), np.r_[np.ones(3), np.zeros(9)]]),
        index=["CAL27", "Q127"],
        columns=tenors,
    )
    raw = pd.Series([annual_raw, 325.0], index=delivery.index)
    indexed = pd.Series(
        [300.83, 325.0 * january_factor],
        index=delivery.index,
    )

    with pytest.raises(InfeasibleCurveError):
        neutralize_curve(
            curve,
            build_raw_and_indexed_anchor_matrix(curve, delivery, raw, indexed),
        )


@pytest.mark.parametrize(
    ("april_factor", "expected_residual"),
    [
        (300.83 / 295.0, 285.1818181818182),
        (1.025, 283.72466759367137),
    ],
)
def test_indexed_q1_and_annual_determine_the_april_to_december_residual(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
    april_factor: float,
    expected_residual: float,
) -> None:
    annual_raw = 295.0
    january_factor = 300.83 / annual_raw
    curve = canonical_curve.assign(
        price=annual_raw,
        index_factor=np.r_[
            np.full(3, january_factor),
            np.full(9, april_factor),
        ],
        block=["Q1"] * 3 + ["residual"] * 9,
    )
    delivery = pd.DataFrame(
        np.vstack(
            [
                np.ones(12),
                np.r_[np.ones(3), np.zeros(9)],
            ]
        ),
        index=["CAL27", "Q127"],
        columns=tenors,
    )
    prices = pd.Series([annual_raw, 325.0], index=delivery.index, name="base")
    quote_factors = pd.Series(
        [january_factor, january_factor],
        index=delivery.index,
    )

    result = neutralize_curve(
        curve,
        build_anchor_matrix(
            curve,
            delivery,
            prices,
            quote_index_factor=quote_factors,
        ),
    )

    balanced = result.wide_curve()["base"]
    np.testing.assert_allclose(balanced.iloc[:3], 325.0, atol=1e-9)
    np.testing.assert_allclose(balanced.iloc[3:], expected_residual, atol=1e-9)
    np.testing.assert_allclose(result.anchors["residual"], 0.0, atol=1e-9)


def test_q1_q2_and_annual_anchor_determine_h2_end_to_end(
    canonical_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    annual_raw = 295.0
    annual_adjusted = 300.83
    factors = pd.Series(
        {
            "Q127": annual_adjusted / annual_raw,
            "Q227": 1.023,
            "H227": 1.028,
        }
    )
    curve = canonical_curve.assign(
        price=np.linspace(280.0, 310.0, 12),
        index_factor=np.repeat(factors.to_numpy(), [3, 3, 6]),
        block=np.repeat(factors.index, [3, 3, 6]),
    )
    delivery = pd.DataFrame(
        np.vstack(
            [
                np.r_[np.ones(3), np.zeros(9)],
                np.r_[np.zeros(3), np.ones(3), np.zeros(6)],
                np.ones(12),
            ]
        ),
        index=["Q127", "Q227", "CAL27"],
        columns=tenors,
    )
    anchor_prices = pd.Series(
        [325.0, 310.0, annual_raw],
        index=delivery.index,
        name="base",
    )
    quote_factors = pd.Series(
        [factors["Q127"], factors["Q227"], factors["Q127"]],
        index=delivery.index,
    )

    result = neutralize_curve(
        curve,
        build_anchor_matrix(
            curve,
            delivery,
            anchor_prices,
            quote_index_factor=quote_factors,
        ),
    )

    balanced = result.wide_curve()["base"]
    np.testing.assert_allclose(balanced.iloc[:3], 325.0, atol=1e-9)
    np.testing.assert_allclose(balanced.iloc[3:6], 310.0, atol=1e-9)
    np.testing.assert_allclose(balanced.iloc[6:], 270.2380132272781, atol=1e-9)
    np.testing.assert_allclose(result.anchors["residual"], 0.0, atol=1e-9)
