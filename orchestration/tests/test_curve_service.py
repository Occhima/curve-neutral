from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration import (
    CurveGranularity,
    build_unbalanced_curve,
    neutralize_market_curve,
    normalize_products,
)
from curve_orchestration.granularity import ProductMarkInput

pytestmark = pytest.mark.functional


@pytest.fixture
def q1_annual_products() -> pd.DataFrame:
    return normalize_products(
        pd.DataFrame(
            {
                "product_id": ["Q127", "CAL27"],
                "description": ["Q1 2027", "Calendar 2027"],
                "start": ["2027-01-01", "2027-01-01"],
                "end": ["2027-03-31", "2027-12-31"],
            }
        )
    )


@pytest.fixture
def q1_annual_marks() -> pd.DataFrame:
    return ProductMarkInput.validate(
        pd.DataFrame(
            {
                "product_id": ["Q127", "CAL27"],
                "price": [325.0, 295.0],
                "traded_at": pd.to_datetime(
                    ["2026-08-21T12:00:00", "2026-08-21T11:00:00"]
                ),
            }
        ),
        lazy=True,
    )


def test_unbalanced_curve_freezes_ipca_at_each_selected_block_head(
    q1_annual_products: pd.DataFrame,
    q1_annual_marks: pd.DataFrame,
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
    selected = CurveGranularity().select(
        q1_annual_products,
        q1_annual_marks,
        tenors,
    )

    unbalanced = build_unbalanced_curve(curve, selected)

    np.testing.assert_allclose(unbalanced["price"].iloc[:3], 325.0)
    np.testing.assert_allclose(unbalanced["price"].iloc[3:], 295.0)
    np.testing.assert_allclose(unbalanced["index_factor"].iloc[:3], 1.019)
    np.testing.assert_allclose(
        unbalanced["index_factor"].iloc[3:],
        curve["index_factor"].iloc[3],
    )
    assert unbalanced["block"].tolist() == ["Q127:1"] * 3 + ["CAL27:2"] * 9


def test_full_market_service_jointly_neutralizes_raw_and_indexed_scenarios(
    q1_annual_products: pd.DataFrame,
) -> None:
    tenors = pd.date_range("2027-01-01", "2028-12-01", freq="MS")
    january_factor = 300.83 / 295.0
    curve = pd.DataFrame(
        {
            "tenor": tenors,
            "price": np.r_[np.full(12, 295.0), np.full(12, 280.0)],
            "index_factor": np.r_[
                np.linspace(january_factor, 1.035, 12),
                np.full(12, 1.03),
            ],
            "floor": 0.0,
            "cap": 500.0,
        }
    )
    prices = pd.DataFrame(
        {
            "base": [325.0, 295.0],
            "stress": [330.0, 300.0],
        },
        index=["Q127", "CAL27"],
    )
    observed_at = pd.Series(
        pd.to_datetime(["2026-08-21T12:00:00Z", "2026-08-21T11:00:00Z"]),
        index=prices.index,
    )
    dcide = pd.DataFrame({"tenor": tenors[12:], "price": 280.0, "index_factor": 1.03})

    first = neutralize_market_curve(
        curve,
        q1_annual_products,
        prices,
        observed_at,
        dcide,
        reference_date="2026-08-22",
        cutoff="2027-12-31",
        quality=pd.Series({"Q127": 5.0, "CAL27": 1.0}),
    )
    first_raw = first.wide_raw_curve()
    rebuilt = curve.assign(price=np.r_[first_raw["base"].iloc[:12], np.full(12, 280.0)])
    second = neutralize_market_curve(
        rebuilt,
        q1_annual_products,
        prices,
        observed_at,
        dcide,
        reference_date="2026-08-22",
        cutoff="2027-12-31",
        quality=pd.Series({"Q127": 5.0, "CAL27": 1.0}),
    )

    assert first_raw.columns.tolist() == ["base", "stress"]
    expected_anchors = {
        "raw:Q127",
        "raw:CAL27",
        "indexed:Q127",
        "indexed:CAL27",
    }
    assert set(first.neutralization.anchors["anchor"]) == expected_anchors
    assert first.neutralization.anchors.groupby("anchor").size().eq(2).all()
    np.testing.assert_allclose(first_raw.iloc[12:], 280.0)
    np.testing.assert_allclose(first.wide_indexed_curve().iloc[12:], 288.4)
    pd.testing.assert_frame_equal(first.curve, second.curve)


def test_annual_anchor_is_kept_when_every_month_has_a_monthly_price() -> None:
    tenors = pd.date_range("2027-01-01", periods=12, freq="MS")
    monthly_ids = tenors.strftime("M%m27").tolist()
    products = normalize_products(
        pd.DataFrame(
            {
                "product_id": [*monthly_ids, "CAL27"],
                "description": [*monthly_ids, "Calendar 2027"],
                "start": [*tenors, tenors[0]],
                "end": [*tenors, tenors[-1]],
            }
        )
    )
    prices = pd.Series(295.0, index=[*monthly_ids, "CAL27"], name="base")
    observed_at = pd.Series(
        pd.Timestamp("2026-08-21T12:00:00Z"),
        index=prices.index,
    )
    curve_tenors = tenors.append(pd.DatetimeIndex(["2028-01-01"]))
    curve = pd.DataFrame(
        {
            "tenor": curve_tenors,
            "price": 295.0,
            "index_factor": 1.02,
            "floor": 0.0,
            "cap": 500.0,
        }
    )
    dcide = pd.DataFrame(
        {"tenor": ["2028-01-01"], "price": [280.0], "index_factor": [1.03]}
    )

    result = neutralize_market_curve(
        curve,
        products,
        prices,
        observed_at,
        dcide,
        reference_date="2026-08-22",
        cutoff="2027-12-31",
    )

    anchors = set(result.neutralization.anchors["anchor"])
    assert {"raw:CAL27", "indexed:CAL27"} <= anchors
    assert {f"raw:{product_id}" for product_id in monthly_ids} <= anchors
