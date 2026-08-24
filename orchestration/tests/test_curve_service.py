from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration import (
    CurveGranularity,
    build_market_plan,
    build_unbalanced_curve,
    finalize_forward_curve,
    market_cutoff,
    neutralize_anchor_plan,
    normalize_products,
    select_tenor_prices,
    wide_curve,
)

pytestmark = pytest.mark.functional


def catalog(**products: tuple[str, str, float, str]) -> pd.DataFrame:
    """Build a priced product catalog from ``id=(start, end, price, traded_at)``."""

    return normalize_products(
        pd.DataFrame(
            [
                {
                    "product_id": product_id,
                    "description": product_id,
                    "start": start,
                    "end": end,
                    "price": price,
                    "traded_at": pd.Timestamp(traded_at),
                }
                for product_id, (start, end, price, traded_at) in products.items()
            ]
        )
    )


@pytest.fixture
def q1_annual() -> pd.DataFrame:
    return catalog(
        Q127=("2027-01-01", "2027-03-31", 325.0, "2026-08-21T12:00:00"),
        CAL27=("2027-01-01", "2027-12-31", 295.0, "2026-08-21T11:00:00"),
    )


@pytest.fixture
def two_year_curve() -> pd.DataFrame:
    tenors = pd.date_range("2027-01-01", "2029-12-01", freq="MS")
    return pd.DataFrame(
        {
            "tenor": tenors,
            "price": 295.0,
            "index_factor": np.linspace(1.020, 1.080, len(tenors)),
            "floor": 0.0,
            "cap": 500.0,
        }
    )


def test_market_cutoff_is_the_end_of_the_second_year_ahead() -> None:
    assert market_cutoff("2026-08-22") == pd.Timestamp("2028-12-31")
    assert market_cutoff("2027-01-01") == pd.Timestamp("2029-12-31")


def test_unbalanced_curve_freezes_ipca_at_each_selected_block_head(
    q1_annual: pd.DataFrame,
) -> None:
    tenors = pd.date_range("2027-01-01", periods=12, freq="MS")
    curve = pd.DataFrame(
        {
            "tenor": tenors,
            "price": 295.0,
            "index_factor": np.linspace(1.019, 1.040, 12),
            "floor": 0.0,
            "cap": 500.0,
        }
    )

    unbalanced = build_unbalanced_curve(
        curve,
        CurveGranularity().select(q1_annual, tenors),
    )

    np.testing.assert_allclose(unbalanced["price"].iloc[:3], 325.0)
    np.testing.assert_allclose(unbalanced["price"].iloc[3:], 295.0)
    np.testing.assert_allclose(unbalanced["index_factor"].iloc[:3], 1.019)
    np.testing.assert_allclose(
        unbalanced["index_factor"].iloc[3:],
        curve["index_factor"].iloc[3],
    )
    assert unbalanced["block"].tolist() == ["Q127:1"] * 3 + ["CAL27:2"] * 9


def test_granularity_override_forces_the_annual_product_without_fallback(
    q1_annual: pd.DataFrame,
) -> None:
    curve = pd.DataFrame(
        {"tenor": pd.date_range("2027-01-01", periods=12, freq="MS"), "price": 295.0}
    )

    default = select_tenor_prices(q1_annual, curve, cutoff="2027-12-31")
    forced = select_tenor_prices(
        q1_annual,
        curve,
        cutoff="2027-12-31",
        granularity=CurveGranularity().for_year(2027, ("ANU",)),
    )

    assert default["product_id"].tolist() == ["Q127"] * 3 + ["CAL27"] * 9
    assert not default["overridden"].any()
    assert forced["product_id"].eq("CAL27").all()
    assert forced["overridden"].all()


def test_the_plan_stops_at_the_cutoff_and_never_reads_dcide(
    q1_annual: pd.DataFrame,
    two_year_curve: pd.DataFrame,
) -> None:
    plan = build_market_plan(two_year_curve, q1_annual, cutoff="2027-12-31")

    assert plan.cutoff == pd.Timestamp("2027-12-01")
    assert plan.curve["tenor"].max() == pd.Timestamp("2027-12-01")
    assert set(plan.anchors.prices.index) == {
        "raw:Q127",
        "raw:CAL27",
        "indexed:Q127",
        "indexed:CAL27",
    }


def test_one_raw_residual_serves_both_the_raw_and_the_indexed_anchor(
    q1_annual: pd.DataFrame,
    two_year_curve: pd.DataFrame,
) -> None:
    plan = build_market_plan(two_year_curve, q1_annual, cutoff="2027-12-31")

    neutralized = neutralize_anchor_plan(plan)

    residuals = neutralized.anchors.set_index("anchor")["residual"]
    # CAL27 spans both blocks, so no single raw state reproduces the anchor on
    # both surfaces: the solve splits the error instead of satisfying one twice.
    assert residuals["raw:CAL27"] < 0.0 < residuals["indexed:CAL27"]
    # Q127 owns its own block, so it can still be met almost exactly.
    assert abs(residuals["raw:Q127"]) < 1e-2


def test_dcide_only_supplies_months_after_the_cutoff_and_ipca_applies_once(
    q1_annual: pd.DataFrame,
    two_year_curve: pd.DataFrame,
) -> None:
    plan = build_market_plan(two_year_curve, q1_annual, cutoff="2027-12-31")
    neutralized = neutralize_anchor_plan(plan)
    dcide = pd.DataFrame(
        {"tenor": two_year_curve["tenor"], "price": 280.0, "index_factor": 1.09}
    )

    forward = finalize_forward_curve(plan, neutralized, dcide)

    market = forward.query("origin == 'market'")
    tail = forward.query("origin == 'dcide'")
    assert market["tenor"].max() == pd.Timestamp("2027-12-01")
    assert tail["tenor"].min() == pd.Timestamp("2028-01-01")
    np.testing.assert_allclose(tail["raw_price"], 280.0)
    np.testing.assert_allclose(tail["indexed_price"], 280.0 * 1.09)
    np.testing.assert_allclose(
        forward["indexed_price"],
        forward["raw_price"] * forward["index_factor"],
    )


def test_the_forward_curve_is_idempotent_under_its_own_output(
    q1_annual: pd.DataFrame,
    two_year_curve: pd.DataFrame,
) -> None:
    dcide = pd.DataFrame(
        {"tenor": two_year_curve["tenor"], "price": 280.0, "index_factor": 1.09}
    )

    def run(curve: pd.DataFrame) -> pd.DataFrame:
        plan = build_market_plan(curve, q1_annual, cutoff="2027-12-31")
        return finalize_forward_curve(plan, neutralize_anchor_plan(plan), dcide)

    first = run(two_year_curve)
    rebuilt = two_year_curve.assign(
        price=wide_curve(first, "raw_price")["base"].to_numpy()
    )
    second = run(rebuilt)

    pd.testing.assert_frame_equal(first, second)


def test_annual_anchor_is_kept_when_every_month_has_a_monthly_price() -> None:
    tenors = pd.date_range("2027-01-01", periods=12, freq="MS")
    products = catalog(
        **{
            tenor.strftime("M%m27"): (
                tenor.strftime("%Y-%m-%d"),
                (tenor + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d"),
                295.0,
                "2026-08-21T12:00:00",
            )
            for tenor in tenors
        },
        CAL27=("2027-01-01", "2027-12-31", 295.0, "2026-08-21T12:00:00"),
    )
    curve = pd.DataFrame(
        {
            "tenor": tenors.append(pd.DatetimeIndex(["2028-01-01"])),
            "price": 295.0,
            "index_factor": 1.02,
            "floor": 0.0,
            "cap": 500.0,
        }
    )

    plan = build_market_plan(curve, products, cutoff="2027-12-31")

    anchors = set(plan.anchors.prices.index)
    assert {"raw:CAL27", "indexed:CAL27"} <= anchors
    assert {tenor.strftime("raw:M%m27") for tenor in tenors} <= anchors
