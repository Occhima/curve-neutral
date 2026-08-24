from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration import (
    build_dual_anchor_plan,
    build_market_plan,
    normalize_products,
)

pytestmark = pytest.mark.unit

TENORS = pd.date_range("2027-01-01", periods=12, freq="MS")


@pytest.fixture
def curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenor": TENORS,
            "price": 295.0,
            "index_factor": 1.02,
            "floor": 0.0,
            "cap": 500.0,
        }
    )


@pytest.fixture
def delivery() -> pd.DataFrame:
    return pd.DataFrame([np.ones(12)], index=["CAL27"], columns=TENORS)


@pytest.fixture
def prices(delivery: pd.DataFrame) -> pd.Series:
    return pd.Series([295.0], index=delivery.index, name="base")


def test_dual_plan_rejects_a_different_number_of_scenarios(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
    prices: pd.Series,
) -> None:
    two_scenarios = pd.DataFrame(
        {"base": [295.0], "stress": [300.0]}, index=prices.index
    )

    with pytest.raises(ValueError, match="same scenarios"):
        build_dual_anchor_plan(curve, delivery, prices, two_scenarios)


def test_dual_plan_rejects_mismatched_product_labels(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
    prices: pd.Series,
) -> None:
    other = pd.Series([295.0], index=["CAL28"], name="base")

    with pytest.raises(ValueError, match="identical product labels"):
        build_dual_anchor_plan(curve, delivery, prices, other)


def test_duplicate_delivery_labels_are_rejected(
    curve: pd.DataFrame,
    prices: pd.Series,
) -> None:
    duplicated = pd.DataFrame(
        [np.ones(12), np.ones(12)],
        index=["CAL27", "CAL27"],
        columns=TENORS,
    )

    with pytest.raises(ValueError, match="labels must be unique"):
        build_dual_anchor_plan(curve, duplicated, prices, prices)


def test_a_candidate_without_delivered_energy_is_rejected(
    curve: pd.DataFrame,
    prices: pd.Series,
) -> None:
    empty = pd.DataFrame([np.zeros(12)], index=["CAL27"], columns=TENORS)

    with pytest.raises(ValueError, match="positive energy"):
        build_dual_anchor_plan(curve, empty, prices, prices)


def test_product_labels_must_stay_unique_once_stringified(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
) -> None:
    collided = pd.Series([295.0, 296.0], index=[1, "1"], name="base")

    with pytest.raises(ValueError, match="unique as strings"):
        build_dual_anchor_plan(curve, delivery, collided, collided)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_anchor_quality_must_be_finite_and_positive(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
    prices: pd.Series,
    bad: float,
) -> None:
    quality = pd.Series([bad], index=prices.index)

    with pytest.raises(ValueError, match="finite and positive"):
        build_dual_anchor_plan(curve, delivery, prices, prices, quality=quality)


def test_explicit_quality_series_overrides_the_estimated_weight(
    curve: pd.DataFrame,
) -> None:
    quotes = normalize_products(
        pd.DataFrame(
            {
                "product_id": ["Q127", "CAL27"],
                "description": ["Q127", "CAL27"],
                "start": ["2027-01-01", "2027-01-01"],
                "end": ["2027-03-31", "2027-12-31"],
                "price": [325.0, 295.0],
                "traded_at": pd.Timestamp("2026-08-21"),
                "effective_weight": [1.0, 1.0],
            }
        )
    )

    plan = build_market_plan(
        curve,
        quotes,
        cutoff="2027-12-31",
        quality=pd.Series({"Q127": 9.0, "CAL27": 1.0}),
    )

    weights = plan.anchors.weight
    assert weights["raw:Q127"] == 9.0
    assert weights["raw:CAL27"] == 1.0


def test_default_quality_is_the_estimated_precision(curve: pd.DataFrame) -> None:
    quotes = normalize_products(
        pd.DataFrame(
            {
                "product_id": ["Q127", "CAL27"],
                "description": ["Q127", "CAL27"],
                "start": ["2027-01-01", "2027-01-01"],
                "end": ["2027-03-31", "2027-12-31"],
                "price": [325.0, 295.0],
                "traded_at": pd.Timestamp("2026-08-21"),
                "precision": [4.0, 0.5],
            }
        )
    )

    plan = build_market_plan(curve, quotes, cutoff="2027-12-31")

    weights = plan.anchors.weight
    assert weights["raw:Q127"] == 4.0
    assert weights["indexed:CAL27"] == 0.5
