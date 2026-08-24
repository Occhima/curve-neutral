"""A real 2027: Q1 traded, so the annual anchor no longer reprices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration import (
    build_dual_anchor_plan,
    build_market_plan,
    curve_standard_error,
    neutralize_anchor_plan,
    normalize_products,
)

pytestmark = pytest.mark.functional

BASE_DATE = pd.Timestamp("2026-08-01")
YEAR_2027 = pd.date_range("2027-01-01", periods=12, freq="MS")
HOURS = YEAR_2027.days_in_month.to_numpy(float) * 24.0

Q1_PRICE = 325.40
CAL27_PRICE = 295.50

# IPCA index at each block head, as quoted by the desk.
IPCA_AT = {
    "2026-08-01": 7657.73,  # base, and all of 2026 without adjustment
    "2026-12-01": 7657.73,
    "2027-01-01": 7792.18,  # 15/01/2027 adjustment -> Q1 block
    "2027-04-01": 7918.74,  # 15/04/2027 adjustment -> residual block
    "2028-01-01": 8260.75,  # 17/01/2028 adjustment -> CAL28 block
    "2028-12-01": 8260.75,
}


@pytest.fixture
def ipca() -> pd.DataFrame:
    known = pd.Series({pd.Timestamp(k): v for k, v in IPCA_AT.items()}).sort_index()
    months = pd.date_range(known.index.min(), known.index.max(), freq="MS")
    return pd.DataFrame(
        {
            "tenor": months,
            "indice": np.exp(
                np.interp(
                    months.astype("int64"),
                    known.index.astype("int64"),
                    np.log(known.to_numpy()),
                )
            ),
        }
    )


@pytest.fixture
def quotes() -> pd.DataFrame:
    return normalize_products(
        pd.DataFrame(
            [
                {
                    "product_id": product_id,
                    "description": product_id,
                    "start": start,
                    "end": end,
                    "price": price,
                    "traded_at": pd.Timestamp("2026-08-21"),
                }
                for product_id, start, end, price in [
                    ("M1226", "2026-12-01", "2026-12-31", 304.12),
                    ("Q127", "2027-01-01", "2027-03-31", Q1_PRICE),
                    ("CAL27", "2027-01-01", "2027-12-31", CAL27_PRICE),
                    ("CAL28", "2028-01-01", "2028-12-31", 264.00),
                ]
            ]
        )
    )


@pytest.fixture
def curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenor": pd.date_range("2026-12-01", "2028-12-01", freq="MS"),
            "price": 300.0,
            "floor": 0.0,
            "cap": 2_000.0,
        }
    )


def solve(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    ipca: pd.DataFrame,
    **weights: float,
) -> tuple[pd.Series, pd.DataFrame]:
    plan = build_market_plan(
        curve, quotes, cutoff="2028-12-31", ipca=ipca, base_date=BASE_DATE, **weights
    )
    neutralized = neutralize_anchor_plan(plan)
    return (
        neutralized.curve.set_index("tenor")["price"],
        plan.curve.set_index("tenor"),
    )


def test_copying_the_annual_price_into_the_residual_leaves_arbitrage() -> None:
    """The state the desk starts from: Q1 traded, residual carrying CAL27."""

    naive = np.r_[np.full(3, Q1_PRICE), np.full(9, CAL27_PRICE)]

    assert np.average(naive, weights=HOURS) == pytest.approx(302.8726, abs=1e-4)


def test_the_residual_absorbs_the_quarterly_liquidity(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    ipca: pd.DataFrame,
) -> None:
    solved, _ = solve(curve, quotes, ipca)

    assert solved["2027-01-01"] == pytest.approx(Q1_PRICE, abs=1e-2)
    assert solved["2027-04-01"] == pytest.approx(283.3537, abs=1e-3)
    assert solved.loc["2027-04-01":"2027-12-01"].nunique() == 1


def test_only_the_unbalanced_year_moves(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    ipca: pd.DataFrame,
) -> None:
    """2026 and 2028 own their anchors, so neutralizing 2027 must not touch them."""

    solved, _ = solve(curve, quotes, ipca)

    assert solved["2026-12-01"] == pytest.approx(304.12)
    np.testing.assert_allclose(solved.loc["2028-01-01":"2028-12-01"], 264.00)


def test_ipca_is_frozen_at_each_block_head_not_at_each_month(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    ipca: pd.DataFrame,
) -> None:
    _, planned = solve(curve, quotes, ipca)

    year = planned.loc[YEAR_2027]
    assert (
        year["index_start"].dt.strftime("%Y-%m").tolist()
        == ["2027-01"] * 3 + ["2027-04"] * 9
    )
    np.testing.assert_allclose(
        year["index_factor"].iloc[:3], IPCA_AT["2027-01-01"] / IPCA_AT["2026-08-01"]
    )
    np.testing.assert_allclose(
        year["index_factor"].iloc[3:], IPCA_AT["2027-04-01"] / IPCA_AT["2026-08-01"]
    )


def test_surface_weights_bracket_the_two_single_surface_answers(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    ipca: pd.DataFrame,
) -> None:
    """Raw alone wants 285.71; indexed alone wants that times f(Jan)/f(Apr)."""

    raw_only = (CAL27_PRICE * HOURS.sum() - Q1_PRICE * HOURS[:3].sum()) / HOURS[
        3:
    ].sum()
    ratio = IPCA_AT["2027-01-01"] / IPCA_AT["2027-04-01"]

    assert raw_only == pytest.approx(285.7145, abs=1e-4)

    raw_heavy, _ = solve(curve, quotes, ipca, raw_weight=1_000.0)
    indexed_heavy, _ = solve(curve, quotes, ipca, indexed_weight=1_000.0)
    balanced, _ = solve(curve, quotes, ipca)

    assert raw_heavy["2027-04-01"] == pytest.approx(raw_only, abs=5e-2)
    assert indexed_heavy["2027-04-01"] == pytest.approx(raw_only * ratio, abs=5e-2)
    assert (
        indexed_heavy["2027-04-01"] < balanced["2027-04-01"] < raw_heavy["2027-04-01"]
    )


def test_the_illiquid_stub_carries_more_error_than_the_traded_quarter(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    ipca: pd.DataFrame,
) -> None:
    """The residual inherits the annual anchor's error, amplified by 1/(1-s)."""

    priced = quotes.assign(
        precision=quotes["product_id"].map({"Q127": 25.0}).fillna(1.0)
    )
    plan = build_market_plan(
        curve, priced, cutoff="2028-12-31", ipca=ipca, base_date=BASE_DATE
    )

    error = curve_standard_error(plan).set_index("tenor")

    q1_std = error.loc["2027-01-01", "price_std"]
    residual_std = error.loc["2027-06-01", "price_std"]
    assert residual_std > q1_std
    # 3 of 12 months already priced, so the stub absorbs roughly 1/(1-s)
    share = HOURS[:3].sum() / HOURS.sum()
    assert residual_std / error.loc["2028-06-01", "price_std"] == pytest.approx(
        1.0 / (1.0 - share), rel=0.35
    )
    assert error.loc["2027-06-01", "leverage"] > 1.0


def test_the_band_widens_as_the_anchors_get_noisier(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    ipca: pd.DataFrame,
) -> None:
    """Quartering every precision must double every standard error."""

    def band(scale: float) -> pd.Series:
        plan = build_market_plan(
            curve,
            quotes.assign(precision=scale),
            cutoff="2028-12-31",
            ipca=ipca,
            base_date=BASE_DATE,
        )
        return curve_standard_error(plan).set_index("tenor")["price_std"]

    np.testing.assert_allclose(band(0.25), band(1.0) * 2.0, rtol=1e-9)


def test_a_two_year_contract_cannot_identify_both_calendar_years() -> None:
    """Years are always separate blocks, so one anchor leaves a free direction."""

    tenors = pd.date_range("2027-01-01", "2028-12-01", freq="MS")
    flat = pd.DataFrame({"tenor": tenors, "price": 300.0})
    delivery = pd.DataFrame([np.ones(len(tenors))], index=["CAL2Y"], columns=tenors)
    prices = pd.Series([295.0], index=delivery.index, name="base")

    plan = build_dual_anchor_plan(flat, delivery, prices, prices)

    with pytest.raises(ValueError, match="do not identify every block"):
        neutralize_anchor_plan(plan)


def test_an_anchor_starting_before_the_curve_is_rejected(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    ipca: pd.DataFrame,
) -> None:
    """A part-delivered annual has no adjustment date inside the curve."""

    late = curve.query("tenor >= '2027-02-01'")

    with pytest.raises(KeyError, match="start outside the priced curve"):
        build_market_plan(
            late, quotes, cutoff="2028-12-31", ipca=ipca, base_date=BASE_DATE
        )


def test_the_indexed_anchor_is_actually_indexed(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    ipca: pd.DataFrame,
) -> None:
    """Guards the quote factor against being read off the un-indexed input curve."""

    plan = build_market_plan(
        curve, quotes, cutoff="2028-12-31", ipca=ipca, base_date=BASE_DATE
    )

    targets = plan.anchors.prices["base"]
    factor = IPCA_AT["2027-01-01"] / IPCA_AT["2026-08-01"]
    assert targets["raw:CAL27"] == pytest.approx(CAL27_PRICE)
    assert targets["indexed:CAL27"] == pytest.approx(CAL27_PRICE * factor)
    assert targets["indexed:CAL27"] > targets["raw:CAL27"]
