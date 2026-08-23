from __future__ import annotations

import inspect
from datetime import date

import pandas as pd
import pandera.errors
import pytest
from curve_orchestration import CurveGranularity, EhubClient, EhubMarketService
from pytest_mock import MockerFixture

pytestmark = pytest.mark.unit


def test_ehub_service_has_exactly_the_three_market_operations() -> None:
    public = {
        name
        for name, member in inspect.getmembers(EhubMarketService, inspect.isfunction)
        if not name.startswith("_")
    }

    assert public == {"get_negotiable_tickers", "get_deals", "get_latest_prices"}


@pytest.fixture
def raw_tickers() -> tuple[dict[str, object], ...]:
    definitions = [
        (1, "M0127", "2027-01-01", "2027-01-31"),
        (2, "Q127", "2027-01-01", "2027-03-31"),
        (3, "H127", "2027-01-01", "2027-06-30"),
        (4, "CAL27", "2027-01-01", "2027-12-31"),
        (5, "Q127-new", "2027-01-01", "2027-03-31"),
    ]
    return tuple(
        {
            "id": product_id,
            "description": description,
            "features_": [
                {"name": "start", "value": start},
                {"name": "end", "value": end},
                {"name": "submarket", "value": "SE"},
            ],
        }
        for product_id, description, start, end in definitions
    )


@pytest.fixture
def raw_deals() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "id": f"deal-{product_id}",
            "productId": product_id,
            "unitPrice": price,
            "quantity": 1.0,
            "createdAt": traded_at,
            "status": "Ativo",
            "originOperationType": "Match",
        }
        for product_id, price, traded_at in [
            (1, 325.0, "2026-08-20T10:00:00Z"),
            (2, 320.0, "2026-08-20T11:00:00Z"),
            (3, 305.0, "2026-08-20T12:00:00Z"),
            (4, 295.0, "2026-08-20T13:00:00Z"),
            (5, 321.0, "2026-08-21T11:00:00Z"),
        ]
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


def test_negotiable_tickers_pivot_nested_features_into_columns(
    service: tuple[EhubMarketService, object],
) -> None:
    market, client = service

    tickers = market.get_negotiable_tickers(7)

    assert tickers.columns.tolist() == [
        "product_id",
        "description",
        "end",
        "start",
        "submarket",
        "delivery_months",
        "granularity",
    ]
    assert tickers.set_index("product_id")["granularity"].to_dict() == {
        "1": "monthly",
        "2": "quarterly",
        "3": "semiannual",
        "4": "annual",
        "5": "quarterly",
    }
    assert tickers["submarket"].eq("SE").all()
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

    assert deals["product_id"].tolist() == ["1", "2", "3", "4", "5"]
    assert deals["traded_at"].dt.tz is None
    client.list_all_deals.assert_called_once_with(
        date(2026, 1, 1),
        date(2026, 8, 22),
        origin_operation_type="Match",
        contract_id=9,
    )


def test_latest_prices_apply_default_liquidity_and_freshness_order(
    service: tuple[EhubMarketService, object],
) -> None:
    market, _ = service
    tickers = market.get_negotiable_tickers(7)
    deals = market.get_deals("2026-08-22")
    curve = pd.DataFrame(
        {"tenor": pd.date_range("2027-01-01", "2028-12-01", freq="MS")}
    )

    selected = market.get_latest_prices(
        tickers,
        deals,
        curve,
        reference_date="2026-08-22",
        cutoff="2027-12-31",
    )

    assert selected["product_id"].tolist() == [
        "1",
        "5",
        "5",
        "3",
        "3",
        "3",
        "4",
        "4",
        "4",
        "4",
        "4",
        "4",
    ]
    assert selected["granularity"].tolist() == [
        "monthly",
        "quarterly",
        "quarterly",
        "semiannual",
        "semiannual",
        "semiannual",
        "annual",
        "annual",
        "annual",
        "annual",
        "annual",
        "annual",
    ]
    assert not selected["overridden"].any()


def test_latest_prices_accept_an_annual_override_at_curve_construction(
    service: tuple[EhubMarketService, object],
) -> None:
    market, _ = service
    tickers = market.get_negotiable_tickers(7)
    deals = market.get_deals("2026-08-22")
    curve = pd.DataFrame({"tenor": pd.date_range("2027-01-01", periods=12, freq="MS")})
    policy = CurveGranularity(overrides=(("2027-01-01", "2027-12-01", "annual"),))

    selected = market.get_latest_prices(
        tickers,
        deals,
        curve,
        reference_date="2026-08-22",
        cutoff="2027-12-31",
        granularity=policy,
    )

    assert selected["product_id"].eq("4").all()
    assert selected["overridden"].all()


def test_check_types_rejects_duplicate_curve_tenors(
    service: tuple[EhubMarketService, object],
) -> None:
    market, _ = service
    tickers = market.get_negotiable_tickers(7)
    deals = market.get_deals("2026-08-22")
    duplicated = pd.DataFrame({"tenor": ["2027-01-01", "2027-01-01"]})

    with pytest.raises(pandera.errors.SchemaErrors):
        market.get_latest_prices(
            tickers,
            deals,
            duplicated,
            reference_date="2026-08-22",
            cutoff="2027-12-31",
        )
