from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import numpy as np
import pandas as pd
import pytest
from curve_orchestration.ehub import EhubSnapshot
from curve_orchestration.market_anchors import (
    AnchorPolicy,
    AnchorPriceOutput,
    DealInput,
    EhubFields,
    LatestPriceOutput,
    ProductInput,
    estimate_anchor_prices,
    latest_prices,
    normalize_ehub_snapshot,
)
from pandas.testing import assert_frame_equal, assert_series_equal
from pandera.errors import SchemaErrors

pytestmark = pytest.mark.unit


@pytest.fixture
def ehub_snapshot() -> EhubSnapshot:
    return EhubSnapshot(
        negotiable_tickers=(
            {"id": 10, "description": "old CAL27"},
            {"id": 10, "description": "CAL27"},
            {"id": 20, "description": "Q127"},
            {"id": 30, "description": "H127"},
        ),
        deals=(
            {
                "id": "d1",
                "productId": 10,
                "unitPrice": 294.0,
                "quantity": 1.0,
                "createdAt": "2026-08-20T12:00:00Z",
                "status": "Ativo",
                "originOperationType": "Match",
            },
            {
                "id": "d1",
                "productId": 10,
                "unitPrice": 295.0,
                "quantity": 2.0,
                "createdAt": "2026-08-21T12:00:00Z",
                "status": "Ativo",
                "originOperationType": "Match",
            },
            {
                "id": "d2",
                "productId": 20,
                "unitPrice": 325.0,
                "quantity": 3.0,
                "createdAt": "2026-08-21T13:00:00Z",
                "status": "Ativo",
                "originOperationType": "Registro",
            },
            {
                "id": "d3",
                "productId": 20,
                "unitPrice": 900.0,
                "quantity": 1.0,
                "createdAt": "2026-08-21T14:00:00Z",
                "status": "CANCELLED",
                "originOperationType": "Match",
            },
            {
                "id": "d4",
                "productId": 999,
                "unitPrice": 1.0,
                "quantity": 1.0,
                "createdAt": "2026-08-21T15:00:00Z",
                "status": "Ativo",
                "originOperationType": "Match",
            },
        ),
    )


def test_snapshot_normalization_filters_status_universe_and_duplicate_deals(
    ehub_snapshot: EhubSnapshot,
) -> None:
    market = normalize_ehub_snapshot(
        ehub_snapshot,
        accepted_statuses={"Ativo"},
    )

    assert market.products.to_dict("records") == [
        {"product_id": "10", "description": "CAL27"},
        {"product_id": "20", "description": "Q127"},
        {"product_id": "30", "description": "H127"},
    ]
    assert market.deals["deal_id"].tolist() == ["d1", "d2"]
    assert market.deals["price"].tolist() == [295.0, 325.0]
    assert market.deals["traded_at"].dt.tz is None
    ProductInput.validate(market.products, lazy=True)
    DealInput.validate(market.deals, lazy=True)


def test_snapshot_normalization_keeps_all_statuses_when_no_policy_is_given(
    ehub_snapshot: EhubSnapshot,
) -> None:
    market = normalize_ehub_snapshot(ehub_snapshot)

    assert market.deals["deal_id"].tolist() == ["d1", "d2", "d3"]


def test_snapshot_normalization_can_filter_operation_type(
    ehub_snapshot: EhubSnapshot,
) -> None:
    market = normalize_ehub_snapshot(
        ehub_snapshot,
        accepted_statuses={"Ativo"},
        accepted_operation_types={"Registro"},
    )

    assert market.deals["deal_id"].tolist() == ["d2"]


def test_payload_fields_are_configuration_not_provider_logic() -> None:
    snapshot = EhubSnapshot(
        negotiable_tickers=({"ticker_id": "t1", "label": "CAL27"},),
        deals=(
            {
                "deal_id": "d1",
                "product_id": "t1",
                "unit_price": 295.0,
                "amount": 1.0,
                "created_at": "2026-08-21T12:00:00Z",
                "deal_status": "DONE",
                "operation": "MATCH",
            },
        ),
    )
    fields = EhubFields(
        ticker_product_id="ticker_id",
        ticker_description="label",
        deal_id="deal_id",
        deal_product_id="product_id",
        deal_price="unit_price",
        deal_quantity="amount",
        deal_traded_at="created_at",
        deal_status="deal_status",
        deal_origin_operation_type="operation",
    )

    market = normalize_ehub_snapshot(snapshot, fields=fields)

    assert market.products.iloc[0].to_dict() == {
        "product_id": "t1",
        "description": "CAL27",
    }
    assert market.deals.iloc[0]["price"] == 295.0


def test_mad_then_ewma_is_robust_recency_weighted_and_auditable() -> None:
    products = pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],
            "description": ["product 1", "product 2", "no price"],
        }
    )
    deals = pd.DataFrame(
        {
            "deal_id": [
                "p1-1",
                "p1-2",
                "p1-3",
                "p1-outlier",
                "p2-1",
                "p2-2",
                "p2-3",
                "p2-outlier",
                "future",
                "unknown",
            ],
            "product_id": ["P1"] * 4 + ["P2"] * 4 + ["P1", "UNKNOWN"],
            "price": [
                100.0,
                101.0,
                99.0,
                1_000.0,
                200.0,
                200.0,
                200.0,
                250.0,
                98.0,
                1.0,
            ],
            "quantity": 1.0,
            "traded_at": pd.to_datetime(
                [
                    "2026-08-18",
                    "2026-08-19",
                    "2026-08-20",
                    "2026-08-21",
                    "2026-08-18",
                    "2026-08-19",
                    "2026-08-20",
                    "2026-08-21",
                    "2026-08-23",
                    "2026-08-20",
                ]
            ),
        }
    )
    before_products = products.copy(deep=True)
    before_deals = deals.copy(deep=True)
    policy = AnchorPolicy(half_life=pd.Timedelta("1D"), minimum_trades=2)

    batch = estimate_anchor_prices(products, deals, "2026-08-22", policy=policy)

    anchors = batch.anchors.set_index("product_id")
    p1_weights = np.exp2(-np.array([4.0, 3.0, 2.0]))
    expected_p1 = np.average([100.0, 101.0, 99.0], weights=p1_weights)
    assert anchors.index.tolist() == ["P1", "P2"]
    assert anchors.loc["P1", "ewma_price"] == pytest.approx(expected_p1)
    assert anchors.loc["P1", "last_inlier_price"] == 99.0
    assert anchors.loc["P1", "last_inlier_at"] == pd.Timestamp("2026-08-20")
    assert anchors.loc["P1", "trade_count"] == 4
    assert anchors.loc["P1", "retained_count"] == 3
    assert anchors.loc["P1", "outlier_count"] == 1
    assert anchors.loc["P1", "age_days"] == 2.0
    assert anchors.loc["P2", "ewma_price"] == 200.0
    assert anchors.loc["P2", "mad"] == 0.0
    assert batch.as_of == pd.Timestamp("2026-08-22")
    assert batch.deals["deal_id"].tolist() == [
        "p1-1",
        "p1-2",
        "p1-3",
        "p1-outlier",
        "p2-1",
        "p2-2",
        "p2-3",
        "p2-outlier",
    ]
    assert_series_equal(batch.price_series(), anchors["ewma_price"])
    assert_series_equal(batch.weight_series(), anchors["effective_weight"])
    latest = batch.latest_prices().set_index("product_id")
    assert latest.loc["P1", "last_price"] == 1_000.0
    assert latest.loc["P1", "last_traded_at"] == pd.Timestamp("2026-08-21")
    assert_series_equal(batch.last_price_series(), latest["last_price"])
    assert batch.missing_products()["product_id"].tolist() == ["P3"]
    AnchorPriceOutput.validate(batch.anchors, lazy=True)
    LatestPriceOutput.validate(batch.latest_prices(), lazy=True)
    assert_frame_equal(products, before_products)
    assert_frame_equal(deals, before_deals)


def test_quantity_can_enter_the_ewma_only_when_the_policy_requests_it() -> None:
    products = pd.DataFrame({"product_id": ["P"], "description": ["product"]})
    deals = pd.DataFrame(
        {
            "deal_id": ["d1", "d2"],
            "product_id": ["P", "P"],
            "price": [100.0, 110.0],
            "quantity": [1.0, 9.0],
            "traded_at": pd.to_datetime(["2026-08-21", "2026-08-21"]),
        }
    )

    batch = estimate_anchor_prices(
        products,
        deals,
        "2026-08-22",
        policy=AnchorPolicy(volume_weighted=True),
    )

    assert batch.anchors["ewma_price"].iat[0] == pytest.approx(109.0)


def test_latest_price_uses_deal_id_to_break_equal_timestamp_ties() -> None:
    products = pd.DataFrame({"product_id": ["P"], "description": ["product"]})
    deals = pd.DataFrame(
        {
            "deal_id": ["1", "2"],
            "product_id": ["P", "P"],
            "price": [100.0, 101.0],
            "quantity": [1.0, 1.0],
            "traded_at": pd.to_datetime(["2026-08-21", "2026-08-21"]),
        }
    )

    result = latest_prices(products, deals, "2026-08-22")

    assert result.iloc[0].to_dict() == {
        "product_id": "P",
        "description": "product",
        "deal_id": "2",
        "last_price": 101.0,
        "last_traded_at": pd.Timestamp("2026-08-21"),
    }


def test_products_without_observations_return_a_typed_empty_surface() -> None:
    products = pd.DataFrame({"product_id": ["P"], "description": ["product"]})
    deals = pd.DataFrame(
        {
            "deal_id": ["future"],
            "product_id": ["P"],
            "price": [100.0],
            "quantity": [1.0],
            "traded_at": pd.to_datetime(["2026-08-23"]),
        }
    )

    batch = estimate_anchor_prices(products, deals, "2026-08-22")

    assert batch.anchors.empty
    assert batch.price_series().empty
    assert batch.weight_series().empty
    assert batch.latest_prices().empty
    assert batch.last_price_series().empty
    assert batch.missing_products()["product_id"].tolist() == ["P"]
    AnchorPriceOutput.validate(batch.anchors, lazy=True)


@pytest.mark.parametrize(
    "policy",
    [
        AnchorPolicy(half_life=pd.Timedelta("1D")),
        AnchorPolicy(mad_threshold=1.0),
        AnchorPolicy(minimum_trades=2),
    ],
)
def test_anchor_policy_is_a_frozen_data_contract(policy: AnchorPolicy) -> None:
    assert is_dataclass(policy)
    with pytest.raises(FrozenInstanceError):
        policy.minimum_trades = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"half_life": pd.Timedelta(0)},
        {"mad_threshold": 0.0},
        {"mad_threshold": np.inf},
        {"minimum_trades": 0},
    ],
)
def test_anchor_policy_rejects_non_positive_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive"):
        AnchorPolicy(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "products,deals",
    [
        (
            pd.DataFrame({"product_id": ["P", "P"], "description": ["a", "b"]}),
            pd.DataFrame(
                {
                    "deal_id": ["d1"],
                    "product_id": ["P"],
                    "price": [100.0],
                    "quantity": [1.0],
                    "traded_at": pd.to_datetime(["2026-08-21"]),
                }
            ),
        ),
        (
            pd.DataFrame({"product_id": ["P"], "description": ["a"]}),
            pd.DataFrame(
                {
                    "deal_id": ["d1", "d1"],
                    "product_id": ["P", "P"],
                    "price": [100.0, 101.0],
                    "quantity": [1.0, 1.0],
                    "traded_at": pd.to_datetime(["2026-08-20", "2026-08-21"]),
                }
            ),
        ),
    ],
)
def test_canonical_contracts_reject_duplicate_keys(
    products: pd.DataFrame,
    deals: pd.DataFrame,
) -> None:
    with pytest.raises(SchemaErrors):
        estimate_anchor_prices(products, deals, "2026-08-22")
