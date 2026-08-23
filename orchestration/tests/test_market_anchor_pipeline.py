from __future__ import annotations

from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest
from curve_orchestration import (
    AnchorPolicy,
    EhubClient,
    build_anchor_matrix,
    load_ehub_anchors,
    load_ehub_latest_prices,
    neutralize_curve,
)
from pytest_mock import MockerFixture

pytestmark = pytest.mark.functional


def test_ehub_deals_become_exact_curve_anchors_end_to_end(
    mocker: MockerFixture,
    block_curve: pd.DataFrame,
    tenors: pd.DatetimeIndex,
) -> None:
    tickers = [
        {"id": 1, "description": "CAL27"},
        {"id": 2, "description": "Q127"},
    ]
    deals = [
        {
            "id": "cal-1",
            "productId": 1,
            "unitPrice": 295.0,
            "quantity": 1.0,
            "createdAt": "2026-08-21T10:00:00Z",
            "status": "Ativo",
            "originOperationType": "Match",
        },
        {
            "id": "q1-1",
            "productId": 2,
            "unitPrice": 325.0,
            "quantity": 1.0,
            "createdAt": "2026-08-21T11:00:00Z",
            "status": "Ativo",
            "originOperationType": "Match",
        },
    ]
    http = mocker.Mock(spec=httpx.Client)
    ticker_response = mocker.Mock(spec=httpx.Response)
    deal_response = mocker.Mock(spec=httpx.Response)
    ticker_response.json.return_value = {"tickers": tickers}
    deal_response.json.return_value = deals
    deal_response.headers = {}
    http.get.side_effect = [
        ticker_response,
        deal_response,
        ticker_response,
        deal_response,
    ]

    batch = load_ehub_anchors(
        EhubClient(http),
        wallet_id=7,
        start=date(2026, 8, 1),
        as_of="2026-08-22T12:00:00-03:00",
        accepted_statuses={"Ativo"},
        origin_operation_type="Match",
        policy=AnchorPolicy(half_life=pd.Timedelta("2D")),
    )
    latest = load_ehub_latest_prices(
        EhubClient(http),
        wallet_id=7,
        start=date(2026, 8, 1),
        as_of="2026-08-22T12:00:00-03:00",
    )
    delivery = pd.DataFrame(
        np.vstack(
            [
                np.ones(12),
                np.r_[np.ones(3), np.zeros(9)],
            ]
        ),
        index=["1", "2"],
        columns=tenors,
    )

    result = neutralize_curve(
        block_curve,
        build_anchor_matrix(block_curve, delivery, batch.price_series()),
    )

    hours = tenors.days_in_month.to_numpy(dtype=float) * 24.0
    expected_residual = (295.0 * hours.sum() - 325.0 * hours[:3].sum()) / hours[
        3:
    ].sum()
    balanced = result.wide_curve()["ewma_price"]
    np.testing.assert_allclose(balanced.iloc[:3], 325.0, atol=1e-8)
    np.testing.assert_allclose(balanced.iloc[3:], expected_residual, atol=1e-8)
    np.testing.assert_allclose(result.anchors["residual"], 0.0, atol=1e-8)
    assert batch.as_of == pd.Timestamp("2026-08-22T15:00:00")
    assert latest.set_index("product_id")["last_price"].to_dict() == {
        "1": 295.0,
        "2": 325.0,
    }
    assert http.get.call_args_list[1].kwargs["params"] == {
        "initialPeriod": "2026-08-01",
        "finalPeriod": "2026-08-22",
        "originOperationType": "Match",
    }
