from __future__ import annotations

import inspect
from datetime import date

import pandas as pd
import pandera.errors
import pytest
from curve_orchestration import EhubClient, EhubMarketService, select_market_universe
from pytest_mock import MockerFixture

pytestmark = pytest.mark.unit

TICKERS = [
    (1, "M0127", "2027-01-01", "2027-01-31", "SE", "CON", "FIX"),
    (2, "Q127", "2027-01-01", "2027-03-31", "SE", "CON", "FIX"),
    (3, "H127", "2027-01-01", "2027-06-30", "SE", "CON", "FIX"),
    (4, "CAL27", "2027-01-01", "2027-12-31", "SE", "CON", "FIX"),
    (5, "Q127-NORTE", "2027-01-01", "2027-03-31", "N", "CON", "FIX"),
    (6, "Q127-I5", "2027-01-01", "2027-03-31", "SE", "I5", "FIX"),
    (7, "Q127-SPOT", "2027-01-01", "2027-03-31", "SE", "CON", "SPT"),
]

DEALS = [
    (1, 325.0, "2026-08-20T10:00:00Z", "Ativo", "Match"),
    (2, 320.0, "2026-08-20T11:00:00Z", "Ativo", "Match"),
    (3, 305.0, "2026-08-20T12:00:00Z", "Ativo", "Match"),
    (4, 295.0, "2026-08-20T13:00:00Z", "Ativo", "Match"),
    (5, 999.0, "2026-08-20T13:00:00Z", "Ativo", "Match"),
    (6, 888.0, "2026-08-20T13:00:00Z", "Ativo", "Match"),
    (7, 777.0, "2026-08-20T13:00:00Z", "Ativo", "Match"),
    (4, 111.0, "2026-08-20T14:00:00Z", "Cancelado", "Match"),
    (4, 222.0, "2026-08-20T15:00:00Z", "Ativo", "Leilao"),
]


@pytest.fixture
def raw_tickers() -> tuple[dict[str, object], ...]:
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
        for product_id, description, start, end, submarket, source, price_type in TICKERS
    )


@pytest.fixture
def raw_deals() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "id": f"deal-{index}",
            "productId": product_id,
            "unitPrice": price,
            "quantity": 1.0,
            "createdAt": traded_at,
            "status": status,
            "originOperationType": operation,
        }
        for index, (product_id, price, traded_at, status, operation) in enumerate(DEALS)
    )


@pytest.fixture
def service(
    mocker: MockerFixture,
    raw_tickers: tuple[dict[str, object], ...],
    raw_deals: tuple[dict[str, object], ...],
) -> tuple[EhubMarketService, object]:
    client = mocker.Mock(spec=EhubClient)
    client.list_negotiable_tickers.return_value = raw_tickers
    client.list_all_deals.return_value = raw_deals
    return EhubMarketService(client), client


def test_ehub_service_has_exactly_the_three_market_operations() -> None:
    public = {
        name
        for name, member in inspect.getmembers(EhubMarketService, inspect.isfunction)
        if not name.startswith("_")
    }

    assert public == {"get_negotiable_tickers", "get_deals", "get_market_deals"}


def test_negotiable_tickers_pivot_nested_features_into_columns(
    service: tuple[EhubMarketService, object],
) -> None:
    market, client = service

    tickers = market.get_negotiable_tickers(7)

    assert tickers.columns.tolist() == [
        "product_id",
        "description",
        "start",
        "end",
        "delivery_months",
        "granularity",
        "submarket",
        "source",
        "price_type",
    ]
    assert tickers.set_index("product_id")["granularity"].to_dict() == {
        "1": "MEN",
        "2": "TRI",
        "3": "SEM",
        "4": "ANU",
        "5": "TRI",
        "6": "TRI",
        "7": "TRI",
    }
    client.list_negotiable_tickers.assert_called_once_with(7)


def test_deals_cover_january_first_through_the_reference_date(
    service: tuple[EhubMarketService, object],
) -> None:
    market, client = service

    deals = market.get_deals(
        "2026-08-22T18:00:00-03:00",
        origin_operation_type="Match",
        contract_id=9,
    )

    assert deals["traded_at"].dt.tz is None
    client.list_all_deals.assert_called_once_with(
        date(2026, 1, 1),
        date(2026, 8, 22),
        origin_operation_type="Match",
        contract_id=9,
    )


def test_deals_drop_rejected_statuses_and_operation_types(
    service: tuple[EhubMarketService, object],
) -> None:
    market, _ = service

    deals = market.get_deals("2026-08-22")

    assert not deals["price"].isin([111.0, 222.0]).any()
    assert len(deals) == 7


def test_market_deals_inner_join_tickers_and_deals(
    service: tuple[EhubMarketService, object],
) -> None:
    market, _ = service
    tickers = market.get_negotiable_tickers(7)
    deals = market.get_deals("2026-08-22")

    joined = market.get_market_deals(tickers, deals)

    assert len(joined) == len(deals)
    assert set(joined["product_id"]) == set(tickers["product_id"])
    assert {"submarket", "source", "price_type", "granularity"} <= set(joined.columns)


def test_select_market_universe_keeps_only_se_conventional_fixed_price(
    service: tuple[EhubMarketService, object],
) -> None:
    market, _ = service
    joined = market.get_market_deals(
        market.get_negotiable_tickers(7),
        market.get_deals("2026-08-22"),
    )

    universe = select_market_universe(joined)

    assert set(universe["product_id"]) == {"1", "2", "3", "4"}
    assert universe["submarket"].eq("SE").all()
    assert universe["source"].eq("CON").all()
    assert universe["price_type"].eq("FIX").all()


def test_select_market_universe_restricts_the_granularities(
    service: tuple[EhubMarketService, object],
) -> None:
    market, _ = service
    joined = market.get_market_deals(
        market.get_negotiable_tickers(7),
        market.get_deals("2026-08-22"),
    )

    universe = select_market_universe(joined, granularities=frozenset({"ANU"}))

    assert set(universe["product_id"]) == {"4"}


def test_check_types_rejects_a_ticker_frame_that_is_not_the_contract(
    service: tuple[EhubMarketService, object],
) -> None:
    market, _ = service
    deals = market.get_deals("2026-08-22")

    with pytest.raises(pandera.errors.SchemaErrors):
        market.get_market_deals(pd.DataFrame({"product_id": ["1"]}), deals)
