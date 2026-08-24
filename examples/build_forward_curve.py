"""End-to-end forward curve, from EHUB deals to the indexed monthly curve.

Run it offline with the bundled fake client:

    uv run python examples/build_forward_curve.py

Swap ``FakeEhubClient`` for the real one to go live:

    with httpx.Client(base_url=BBCE_URL, timeout=30.0) as http:
        client = EhubClient.login(http, EhubCredentials(code, email, password, key))
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from curve_orchestration import (
    CurveGranularity,
    EhubMarketService,
    apply_ipca,
    build_market_plan,
    estimate_anchor_prices,
    expand_curve,
    finalize_forward_curve,
    market_cutoff,
    neutralize_anchor_plan,
    select_market_universe,
    wide_curve,
)

REFERENCE_DATE = pd.Timestamp("2026-08-22")
IPCA_BASE_DATE = pd.Timestamp("2026-08-01")
LAST_TENOR = "2030-12-01"
WALLET_ID = 7


# --------------------------------------------------------------------------- #
# Stand-in for the authenticated BBCE reader, so the example runs offline.
# --------------------------------------------------------------------------- #
TICKERS = [
    # id, description, start, end, submarket, source, price_type
    (1, "M0127", "2027-01-01", "2027-01-31", "SE", "CON", "FIX"),
    (2, "Q127", "2027-01-01", "2027-03-31", "SE", "CON", "FIX"),
    (3, "H227", "2027-07-01", "2027-12-31", "SE", "CON", "FIX"),
    (4, "CAL27", "2027-01-01", "2027-12-31", "SE", "CON", "FIX"),
    (5, "CAL28", "2028-01-01", "2028-12-31", "SE", "CON", "FIX"),
    (6, "CAL27-N", "2027-01-01", "2027-12-31", "N", "CON", "FIX"),
    (7, "CAL27-I5", "2027-01-01", "2027-12-31", "SE", "I5", "FIX"),
]

DEALS = [
    # product_id, price, traded_at
    (1, 325.0, "2026-08-20T10:00:00Z"),
    (2, 320.0, "2026-08-19T11:00:00Z"),
    (2, 900.0, "2026-08-20T11:30:00Z"),  # fat-finger, dropped by the MAD filter
    (2, 321.5, "2026-08-21T12:00:00Z"),
    (3, 288.0, "2026-08-20T12:00:00Z"),
    (4, 295.0, "2026-08-21T13:00:00Z"),
    (5, 286.0, "2026-08-21T13:00:00Z"),
    (6, 210.0, "2026-08-21T13:00:00Z"),  # north submarket, filtered out
    (7, 180.0, "2026-08-21T13:00:00Z"),  # incentivised source, filtered out
]


class FakeEhubClient:
    """Returns the payload shapes the real ``EhubClient`` returns."""

    def list_negotiable_tickers(self, wallet_id: int) -> tuple[dict, ...]:
        return tuple(
            {
                "id": product_id,
                "description": description,
                "features_": [
                    {"name": "start", "value": start},
                    {"name": "end", "value": end},
                    {"name": "submarket", "value": submarket},
                    {"name": "source", "value": source},
                    {"name": "priceType", "value": price_type},
                ],
            }
            for product_id, description, start, end, submarket, source, price_type in (
                TICKERS
            )
        )

    def list_all_deals(self, start: date, end: date, **_: object) -> tuple[dict, ...]:
        return tuple(
            {
                "id": f"deal-{index}",
                "productId": product_id,
                "unitPrice": price,
                "quantity": 1.0,
                "createdAt": traded_at,
                "status": "Ativo",
                "originOperationType": "Match",
            }
            for index, (product_id, price, traded_at) in enumerate(DEALS)
        )


def prior_curve(last_tenor: str) -> pd.DataFrame:
    """The monthly state the desk starts from: prior price and bounds."""

    return pd.DataFrame(
        {
            "tenor": pd.date_range("2027-01-01", last_tenor, freq="MS"),
            "price": 300.0,
            "floor": 0.0,
            "cap": 1_000.0,
        }
    )


def ipca_index(last_tenor: str) -> pd.DataFrame:
    """The only IPCA input: the forward index by maturity, ``tenor | indice``."""

    months = pd.date_range(IPCA_BASE_DATE, last_tenor, freq="MS")
    return pd.DataFrame(
        {"tenor": months, "indice": 7000.0 * 1.0037 ** np.arange(len(months))}
    )


def submarket_spreads(tenors: pd.DatetimeIndex) -> pd.DataFrame:
    """Additive basis over SE conventional, per month and market."""

    return pd.DataFrame(
        {
            "vencimento": [*tenors, *tenors],
            "spread": [-18.0] * len(tenors) + [55.0] * len(tenors),
            "submercado": ["NE"] * len(tenors) + ["SE"] * len(tenors),
            "tipo_energia": ["CON"] * len(tenors) + ["I5"] * len(tenors),
        }
    )


def main() -> None:
    cutoff = market_cutoff(REFERENCE_DATE)
    print(f"reference {REFERENCE_DATE.date()} -> liquid through {cutoff.date()}\n")

    # 1. EHUB: two API reads, their inner join, then the tradeable universe.
    service = EhubMarketService(FakeEhubClient())
    tickers = service.get_negotiable_tickers(WALLET_ID)
    deals = service.get_deals(REFERENCE_DATE)
    universe = select_market_universe(
        service.get_market_deals(tickers, deals),
        submarket="SE",
        source="CON",
        price_type="FIX",
    )
    print(f"1. universe: {len(deals)} deals -> {len(universe)} in SE/CON/FIX")

    # 2. Fair ("anchor") price per product: MAD outlier filter, then EWMA decay.
    quotes = estimate_anchor_prices(universe, REFERENCE_DATE)
    print("\n2. anchor price per product")
    print(
        quotes[
            ["product_id", "granularity", "price", "trade_count", "retained_count"]
        ].to_string(index=False)
    )

    # 3. Plan: one product per tenor under the liquidity policy. Selecting the
    #    products is what fixes the IPCA adjustment dates, so the index is
    #    applied here, off each block's head. Override: price all of 2028 off
    #    the annual product, with no fallback.
    curve = prior_curve(LAST_TENOR)
    plan = build_market_plan(
        curve,
        quotes,
        cutoff=cutoff,
        ipca=ipca_index(LAST_TENOR),
        base_date=IPCA_BASE_DATE,
        granularity=CurveGranularity.from_dict(
            {"rules": [{"year": 2028, "priority": ["ANU"]}]}
        ),
    )
    print("\n3. unbalanced curve, indexed off each block head")
    print(
        plan.curve[["tenor", "price", "index_start", "ipca_forecast", "index_factor"]]
        .head(13)
        .to_string(index=False)
    )

    # 4. Solve. One raw residual makes the raw price and the IPCA-corrected price
    #    both as close as possible to their anchors. DCIDE is not involved.
    neutralized = neutralize_anchor_plan(plan)
    print("\n4. anchor fit (target vs fitted)")
    print(
        neutralized.anchors[["anchor", "target", "fitted", "residual"]].to_string(
            index=False
        )
    )

    # 5. Only now: append DCIDE past the cutoff and multiply IPCA once. The tail
    #    has no blocks, so each of its months indexes off its own maturity.
    dcide = apply_ipca(
        prior_curve(LAST_TENOR).assign(price=272.0),
        ipca_index(LAST_TENOR),
        base_date=IPCA_BASE_DATE,
    )
    forward = finalize_forward_curve(plan, neutralized, dcide)
    print("\n5. forward curve around the cutoff")
    print(
        forward.query("'2028-10-01' <= tenor <= '2029-03-01'")[
            ["tenor", "origin", "raw_price", "index_factor", "indexed_price"]
        ].to_string(index=False)
    )

    raw = wide_curve(forward, "raw_price")
    indexed = wide_curve(forward, "indexed_price")
    hours = raw.index.days_in_month.to_numpy(dtype=float)[:12] * 24.0
    print(
        f"\nCAL27 anchor 295.00"
        f" | raw {np.average(raw.iloc[:12, 0], weights=hours):.2f}"
        f" | indexed {np.average(indexed.iloc[:12, 0], weights=hours):.2f}"
        "  <- one raw state, both surfaces near the anchor"
    )

    # 6. Deliver: every submarket and energy type, as reference + spread.
    full = expand_curve(forward, submarket_spreads(raw.index))
    markets = sorted(
        map(tuple, full[["submercado", "tipo_energia"]].drop_duplicates().to_numpy())
    )
    print("\n6. delivered curve")
    print(full[full["vencimento"] == pd.Timestamp("2027-01-01")].to_string(index=False))
    print(f"\n{len(full)} rows, markets {markets}")


if __name__ == "__main__":
    main()
