from __future__ import annotations

from datetime import date

import httpx
import pytest
from curve_orchestration.ehub import EhubClient, EhubCredentials, EhubEndpoints
from pytest_mock import MockerFixture

pytestmark = pytest.mark.unit


def test_client_reads_the_documented_market_data_routes(
    mocker: MockerFixture,
) -> None:
    http = mocker.Mock(spec=httpx.Client)
    tickers_response = mocker.Mock(spec=httpx.Response)
    deals_response = mocker.Mock(spec=httpx.Response)
    tickers_response.json.return_value = {"tickers": [{"id": 41}]}
    deals_response.json.return_value = [{"id": 73}]
    deals_response.headers = {}
    http.get.side_effect = [tickers_response, deals_response]

    snapshot = EhubClient(http).snapshot(
        9,
        date(2026, 8, 1),
        date(2026, 8, 22),
        origin_operation_type="Match",
    )

    assert snapshot.negotiable_tickers == ({"id": 41},)
    assert snapshot.deals == ({"id": 73},)
    assert http.get.call_args_list == [
        mocker.call(
            "/bus/v1/negotiable-tickers",
            params={"walletId": 9},
            headers={},
        ),
        mocker.call(
            "/bus/v2/all-deals/report",
            params={
                "initialPeriod": "2026-08-01",
                "finalPeriod": "2026-08-22",
                "originOperationType": "Match",
            },
            headers={"page": "0"},
        ),
    ]
    tickers_response.raise_for_status.assert_called_once_with()
    deals_response.raise_for_status.assert_called_once_with()


def test_optional_operation_type_is_omitted_and_routes_are_configurable(
    mocker: MockerFixture,
) -> None:
    http = mocker.Mock(spec=httpx.Client)
    first_response = mocker.Mock(spec=httpx.Response)
    second_response = mocker.Mock(spec=httpx.Response)
    first_response.json.return_value = [{"id": 1}]
    second_response.json.return_value = [{"id": 2}]
    first_response.headers = {"pages": "2"}
    second_response.headers = {"pages": "2"}
    http.get.side_effect = [first_response, second_response]
    endpoints = EhubEndpoints(
        all_deals="/custom/deals",
        initial_period="from",
        final_period="to",
        origin_operation_type="origin",
        page_header="requested-page",
        page_count_header="pages",
    )

    records = EhubClient(http, endpoints).list_all_deals(
        date(2026, 8, 1),
        date(2026, 8, 2),
        contract_id=42,
    )

    assert records == ({"id": 1}, {"id": 2})
    assert http.get.call_args_list == [
        mocker.call(
            "/custom/deals",
            params={
                "from": "2026-08-01",
                "to": "2026-08-02",
                "contractId": 42,
            },
            headers={"requested-page": "0"},
        ),
        mocker.call(
            "/custom/deals",
            params={
                "from": "2026-08-01",
                "to": "2026-08-02",
                "contractId": 42,
            },
            headers={"requested-page": "1"},
        ),
    ]


def test_login_builds_an_authenticated_client_without_exposing_secrets(
    mocker: MockerFixture,
) -> None:
    http = mocker.Mock(spec=httpx.Client)
    response = mocker.Mock(spec=httpx.Response)
    response.json.return_value = {"idToken": "private-token"}
    http.post.return_value = response
    credentials = EhubCredentials(
        company_external_code=1000,
        email="trader@example.com",
        password="private-password",
        api_key="private-api-key",
    )

    client = EhubClient.login(http, credentials)

    http.post.assert_called_once_with(
        "/bus/v2/login",
        headers={"apiKey": "private-api-key", "Accept": "application/json"},
        json={
            "companyExternalCode": 1000,
            "email": "trader@example.com",
            "password": "private-password",
        },
    )
    response.raise_for_status.assert_called_once_with()
    assert client.authentication_headers == {
        "apiKey": "private-api-key",
        "Authorization": "Bearer private-token",
        "Accept": "application/json",
    }
    assert "private" not in repr(credentials)
    assert "private" not in repr(client)


def test_authenticated_headers_reach_market_requests(
    mocker: MockerFixture,
) -> None:
    http = mocker.Mock(spec=httpx.Client)
    response = mocker.Mock(spec=httpx.Response)
    response.json.return_value = {"tickers": []}
    http.get.return_value = response
    headers = {
        "apiKey": "api-key",
        "Authorization": "Bearer id-token",
        "Accept": "application/json",
    }

    EhubClient(http, authentication_headers=headers).list_negotiable_tickers(9)

    http.get.assert_called_once_with(
        "/bus/v1/negotiable-tickers",
        params={"walletId": 9},
        headers=headers,
    )


def test_http_failures_are_not_hidden(mocker: MockerFixture) -> None:
    http = mocker.Mock(spec=httpx.Client)
    response = mocker.Mock(spec=httpx.Response)
    error = httpx.HTTPStatusError(
        "unauthorized",
        request=httpx.Request("GET", "https://example.test"),
        response=httpx.Response(401),
    )
    response.raise_for_status.side_effect = error
    http.get.return_value = response

    with pytest.raises(httpx.HTTPStatusError, match="unauthorized"):
        EhubClient(http).list_negotiable_tickers(9)
