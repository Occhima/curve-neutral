from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration import AnchorPolicy, estimate_anchor_prices

pytestmark = pytest.mark.unit

AS_OF = pd.Timestamp("2026-08-22T00:00:00")


def market_deals(*trades: tuple[str, float, str, float]) -> pd.DataFrame:
    """Build a joined deal table from ``(product_id, price, traded_at, qty)``."""

    return pd.DataFrame(
        [
            {
                "deal_id": f"deal-{index}",
                "product_id": product_id,
                "price": price,
                "quantity": quantity,
                "traded_at": pd.Timestamp(traded_at),
                "description": f"desc-{product_id}",
                "start": pd.Timestamp("2027-01-01"),
                "end": pd.Timestamp("2027-12-01"),
                "delivery_months": 12,
                "granularity": "ANU",
                "submarket": "SE",
                "source": "CON",
                "price_type": "FIX",
            }
            for index, (product_id, price, traded_at, quantity) in enumerate(trades)
        ]
    )


def test_mad_filter_drops_the_outlier_and_keeps_the_survivors() -> None:
    deals = market_deals(
        ("CAL27", 300.0, "2026-08-20T10:00:00", 1.0),
        ("CAL27", 301.0, "2026-08-20T11:00:00", 1.0),
        ("CAL27", 302.0, "2026-08-20T12:00:00", 1.0),
        ("CAL27", 900.0, "2026-08-20T13:00:00", 1.0),
    )

    quotes = estimate_anchor_prices(deals, AS_OF)

    row = quotes.set_index("product_id").loc["CAL27"]
    assert row["trade_count"] == 4
    assert row["retained_count"] == 3
    assert 300.0 <= row["price"] <= 302.0


def test_zero_mad_keeps_only_observations_equal_to_the_median() -> None:
    deals = market_deals(
        ("CAL27", 300.0, "2026-08-20T10:00:00", 1.0),
        ("CAL27", 300.0, "2026-08-20T11:00:00", 1.0),
        ("CAL27", 300.0, "2026-08-20T12:00:00", 1.0),
        ("CAL27", 400.0, "2026-08-20T13:00:00", 1.0),
    )

    quotes = estimate_anchor_prices(deals, AS_OF)

    row = quotes.set_index("product_id").loc["CAL27"]
    assert row["mad"] == 0.0
    assert row["retained_count"] == 3
    np.testing.assert_allclose(row["price"], 300.0)


def test_recency_weight_pulls_the_anchor_towards_the_freshest_trade() -> None:
    deals = market_deals(
        ("CAL27", 280.0, "2026-08-16T00:00:00", 1.0),
        ("CAL27", 320.0, "2026-08-22T00:00:00", 1.0),
    )

    fast = estimate_anchor_prices(deals, AS_OF, policy=AnchorPolicy(pd.Timedelta("1D")))
    slow = estimate_anchor_prices(
        deals, AS_OF, policy=AnchorPolicy(pd.Timedelta("365D"))
    )

    assert fast["price"].iloc[0] > slow["price"].iloc[0]
    np.testing.assert_allclose(slow["price"].iloc[0], 300.0, atol=0.5)
    assert fast["traded_at"].iloc[0] == pd.Timestamp("2026-08-22")


def test_volume_weighting_is_opt_in() -> None:
    deals = market_deals(
        ("CAL27", 280.0, "2026-08-22T00:00:00", 1.0),
        ("CAL27", 320.0, "2026-08-22T00:00:00", 9.0),
    )

    flat = estimate_anchor_prices(deals, AS_OF)
    weighted = estimate_anchor_prices(
        deals, AS_OF, policy=AnchorPolicy(volume_weighted=True)
    )

    np.testing.assert_allclose(flat["price"].iloc[0], 300.0)
    np.testing.assert_allclose(weighted["price"].iloc[0], 316.0)


def test_trades_after_the_reference_date_are_ignored() -> None:
    deals = market_deals(
        ("CAL27", 300.0, "2026-08-20T10:00:00", 1.0),
        ("CAL27", 999.0, "2026-09-01T10:00:00", 1.0),
    )

    quotes = estimate_anchor_prices(deals, AS_OF)

    assert quotes["trade_count"].iloc[0] == 1
    np.testing.assert_allclose(quotes["price"].iloc[0], 300.0)


def test_minimum_trades_removes_thin_products() -> None:
    deals = market_deals(
        ("CAL27", 300.0, "2026-08-20T10:00:00", 1.0),
        ("CAL28", 290.0, "2026-08-20T10:00:00", 1.0),
        ("CAL28", 291.0, "2026-08-20T11:00:00", 1.0),
    )

    quotes = estimate_anchor_prices(deals, AS_OF, policy=AnchorPolicy(minimum_trades=2))

    assert quotes["product_id"].tolist() == ["CAL28"]


def test_a_genuine_repricing_is_followed_not_rejected_as_an_outlier() -> None:
    """A flat yearly median would call the new level an outlier and stay stale."""

    deals = market_deals(
        *[("CAL27", 300.0, f"2026-0{month}-15T10:00:00", 1.0) for month in range(1, 6)],
        *[("CAL27", 380.0, f"2026-08-{day}T10:00:00", 1.0) for day in range(18, 22)],
    )

    quotes = estimate_anchor_prices(deals, AS_OF)

    np.testing.assert_allclose(quotes["median_price"].iloc[0], 380.0)
    np.testing.assert_allclose(quotes["price"].iloc[0], 380.0, atol=1e-6)


def test_a_fresh_fat_finger_is_still_filtered_out() -> None:
    """The newest trade must not become the anchor just for being newest."""

    deals = market_deals(
        *[("CAL27", 300.0, f"2026-08-{day}T10:00:00", 1.0) for day in range(15, 21)],
        ("CAL27", 900.0, "2026-08-21T23:00:00", 1.0),
    )

    quotes = estimate_anchor_prices(deals, AS_OF)

    assert quotes["retained_count"].iloc[0] == 6
    np.testing.assert_allclose(quotes["price"].iloc[0], 300.0)


def test_two_nearby_quotes_in_a_thin_market_are_both_retained() -> None:
    """Guards the MAD estimator against collapsing onto the heaviest observation."""

    deals = market_deals(
        ("CAL27", 290.0, "2026-08-20T10:00:00", 1.0),
        ("CAL27", 291.0, "2026-08-20T11:00:00", 1.0),
    )

    quotes = estimate_anchor_prices(deals, AS_OF)

    assert quotes["retained_count"].iloc[0] == 2
    assert 290.0 <= quotes["price"].iloc[0] <= 291.0


def test_precision_is_recency_mass_over_robust_variance() -> None:
    """Fresh but disagreeing quotes must not outweigh a tight stack."""

    deals = market_deals(
        # tight: three quotes within 1.0 of each other
        ("TIGHT", 300.0, "2026-08-21T10:00:00", 1.0),
        ("TIGHT", 300.5, "2026-08-21T11:00:00", 1.0),
        ("TIGHT", 301.0, "2026-08-21T12:00:00", 1.0),
        # loose: same recency, prices 40 apart
        ("LOOSE", 280.0, "2026-08-21T10:00:00", 1.0),
        ("LOOSE", 300.0, "2026-08-21T11:00:00", 1.0),
        ("LOOSE", 320.0, "2026-08-21T12:00:00", 1.0),
    )

    quotes = estimate_anchor_prices(deals, AS_OF).set_index("product_id")

    np.testing.assert_allclose(
        quotes["precision"],
        quotes["effective_weight"] / (1.4826 * quotes["mad"]) ** 2,
    )
    assert quotes.loc["TIGHT", "precision"] > quotes.loc["LOOSE", "precision"]
    # similar recency mass: the gap comes from dispersion, not from decay
    assert quotes.loc["TIGHT", "effective_weight"] == pytest.approx(
        quotes.loc["LOOSE", "effective_weight"]
    )


def test_zero_mad_borrows_the_median_scale_instead_of_infinite_confidence() -> None:
    deals = market_deals(
        ("FLAT", 300.0, "2026-08-21T10:00:00", 1.0),
        ("FLAT", 300.0, "2026-08-21T11:00:00", 1.0),
        ("WIDE", 280.0, "2026-08-21T10:00:00", 1.0),
        ("WIDE", 300.0, "2026-08-21T11:00:00", 1.0),
        ("WIDE", 320.0, "2026-08-21T12:00:00", 1.0),
    )

    quotes = estimate_anchor_prices(deals, AS_OF).set_index("product_id")

    scale = 1.4826 * quotes.loc["WIDE", "mad"]
    np.testing.assert_allclose(
        quotes.loc["FLAT", "precision"],
        quotes.loc["FLAT", "effective_weight"] / scale**2,
    )
    assert np.isfinite(quotes["precision"]).all()


def test_when_no_product_dispersed_precision_falls_back_to_the_weight() -> None:
    deals = market_deals(
        ("FLAT", 300.0, "2026-08-21T10:00:00", 1.0),
        ("FLAT", 300.0, "2026-08-21T11:00:00", 1.0),
    )

    quotes = estimate_anchor_prices(deals, AS_OF)

    np.testing.assert_allclose(quotes["precision"], quotes["effective_weight"])


@pytest.mark.parametrize(
    "policy",
    [
        {"half_life": pd.Timedelta(0)},
        {"mad_threshold": 0.0},
        {"mad_threshold": float("nan")},
        {"minimum_trades": 0},
    ],
)
def test_anchor_policy_rejects_invalid_values(policy: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        AnchorPolicy(**policy)  # type: ignore[arg-type]
