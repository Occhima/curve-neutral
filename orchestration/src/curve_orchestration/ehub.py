"""HTTP adapter for the BBCE Connect EHUB market-data routes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast

import httpx

Record = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EhubCredentials:
    """Credentials required by the BBCE Connect login route."""

    company_external_code: int
    email: str
    password: str = field(repr=False)
    api_key: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class EhubEndpoints:
    """Documented paths and parameter names kept at the HTTP boundary."""

    login: str = "/bus/v2/login"
    negotiable_tickers: str = "/bus/v1/negotiable-tickers"
    all_deals: str = "/bus/v2/all-deals/report"
    wallet_id: str = "walletId"
    initial_period: str = "initialPeriod"
    final_period: str = "finalPeriod"
    origin_operation_type: str = "originOperationType"
    contract_id: str = "contractId"
    negotiable_tickers_key: str = "tickers"
    page_header: str = "page"
    page_count_header: str = "x-number-of-pages"


DEFAULT_ENDPOINTS = EhubEndpoints()


@dataclass(frozen=True, slots=True)
class EhubSnapshot:
    """Raw API records captured from one query interval."""

    negotiable_tickers: tuple[Record, ...]
    deals: tuple[Record, ...]


@dataclass(frozen=True, slots=True)
class EhubClient:
    """Authenticated BBCE reader with product and deal operations."""

    http: httpx.Client = field(repr=False)
    endpoints: EhubEndpoints = field(default_factory=EhubEndpoints)
    authentication_headers: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
    )

    @classmethod
    def login(
        cls,
        http: httpx.Client,
        credentials: EhubCredentials,
        *,
        endpoints: EhubEndpoints = DEFAULT_ENDPOINTS,
    ) -> EhubClient:
        """Authenticate and retain only the headers needed by market requests."""

        response = http.post(
            endpoints.login,
            headers={"apiKey": credentials.api_key, "Accept": "application/json"},
            json={
                "companyExternalCode": credentials.company_external_code,
                "email": credentials.email,
                "password": credentials.password,
            },
        )
        response.raise_for_status()
        return cls(
            http=http,
            endpoints=endpoints,
            authentication_headers={
                "apiKey": credentials.api_key,
                "Authorization": f"Bearer {response.json()['idToken']}",
                "Accept": "application/json",
            },
        )

    def list_negotiable_tickers(self, wallet_id: int) -> tuple[Record, ...]:
        return self._get(
            self.endpoints.negotiable_tickers,
            {self.endpoints.wallet_id: wallet_id},
            records_key=self.endpoints.negotiable_tickers_key,
        )

    def list_all_deals(
        self,
        start: date,
        end: date,
        *,
        origin_operation_type: str | None = None,
        contract_id: int | None = None,
    ) -> tuple[Record, ...]:
        params = {
            key: value
            for key, value in {
                self.endpoints.initial_period: start.isoformat(),
                self.endpoints.final_period: end.isoformat(),
                self.endpoints.origin_operation_type: origin_operation_type,
                self.endpoints.contract_id: contract_id,
            }.items()
            if value is not None
        }
        first, page_count = self._deal_page(params, 0)
        remaining = (
            record
            for page in range(1, page_count)
            for record in self._deal_page(params, page)[0]
        )
        return (*first, *remaining)

    def snapshot(
        self,
        wallet_id: int,
        start: date,
        end: date,
        *,
        origin_operation_type: str | None = None,
        contract_id: int | None = None,
    ) -> EhubSnapshot:
        return EhubSnapshot(
            negotiable_tickers=self.list_negotiable_tickers(wallet_id),
            deals=self.list_all_deals(
                start,
                end,
                origin_operation_type=origin_operation_type,
                contract_id=contract_id,
            ),
        )

    def _get(
        self,
        path: str,
        params: Mapping[str, object],
        *,
        records_key: str | None = None,
    ) -> tuple[Record, ...]:
        response = self.http.get(
            path,
            params=params,
            headers=self.authentication_headers,
        )
        response.raise_for_status()
        payload = response.json()
        records = payload[records_key] if records_key else payload
        return tuple(cast(list[Record], records))

    def _deal_page(
        self,
        params: Mapping[str, object],
        page: int,
    ) -> tuple[tuple[Record, ...], int]:
        response = self.http.get(
            self.endpoints.all_deals,
            params=params,
            headers={
                **self.authentication_headers,
                self.endpoints.page_header: str(page),
            },
        )
        response.raise_for_status()
        page_count = int(response.headers.get(self.endpoints.page_count_header, 1))
        return tuple(cast(list[Record], response.json())), page_count
