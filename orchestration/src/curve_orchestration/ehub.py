"""EHUB market data: HTTP boundary, three typed operations and anchor pricing."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, cast

import httpx
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

from .curve import (
    GRANULARITY_MONTHS,
    ProductDeliveryInput,
    ProductQuoteInput,
    normalize_products,
)

Record = Mapping[str, Any]
TimestampLike = str | date | datetime | pd.Timestamp


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


DEFAULT_ACCEPTED_STATUSES = ("Ativo",)
DEFAULT_ACCEPTED_OPERATION_TYPES = ("Match", "Registro")
DEFAULT_GRANULARITIES = frozenset(GRANULARITY_MONTHS)


@dataclass(frozen=True, slots=True)
class EhubFields:
    """Provider JSON names, kept as data so they never reach the statistics.

    ``ticker_*_feature`` entries name the nested ``features_`` items that the
    service pivots into ordinary columns.
    """

    ticker_product_id: str = "id"
    ticker_description: str = "description"
    ticker_features: str = "features_"
    ticker_feature_name: str = "name"
    ticker_feature_value: str = "value"
    ticker_start_feature: str = "start"
    ticker_end_feature: str = "end"
    ticker_submarket_feature: str = "submarket"
    ticker_source_feature: str = "source"
    ticker_price_type_feature: str = "priceType"
    deal_id: str = "id"
    deal_product_id: str = "productId"
    deal_price: str = "unitPrice"
    deal_quantity: str = "quantity"
    deal_traded_at: str = "createdAt"
    deal_status: str = "status"
    deal_origin_operation_type: str = "originOperationType"

    def ticker_columns(self) -> dict[str, str]:
        return {
            self.ticker_product_id: "product_id",
            self.ticker_description: "description",
            self.ticker_start_feature: "start",
            self.ticker_end_feature: "end",
            self.ticker_submarket_feature: "submarket",
            self.ticker_source_feature: "source",
            self.ticker_price_type_feature: "price_type",
        }

    def deal_columns(self) -> dict[str, str]:
        return {
            self.deal_id: "deal_id",
            self.deal_product_id: "product_id",
            self.deal_price: "price",
            self.deal_quantity: "quantity",
            self.deal_traded_at: "traded_at",
            self.deal_status: "status",
            self.deal_origin_operation_type: "origin_operation_type",
        }


DEFAULT_EHUB_FIELDS = EhubFields()


class TickerOutput(ProductDeliveryInput):
    """Negotiable ticker with every nested feature pivoted into a column."""

    submarket: Series[str]
    source: Series[str]
    price_type: Series[str]

    class Config:
        coerce = True
        strict = True


class DealOutput(pa.DataFrameModel):
    """Canonical executed deals returned by the EHUB report."""

    deal_id: Series[str] = pa.Field(unique=True)
    product_id: Series[str]
    price: Series[float] = pa.Field(gt=0)
    quantity: Series[float] = pa.Field(gt=0)
    traded_at: Series[pa.DateTime]

    class Config:
        coerce = True
        strict = True


class MarketDealOutput(TickerOutput):
    """Deals joined to their negotiable-ticker metadata."""

    deal_id: Series[str] = pa.Field(unique=True)
    product_id: Series[str]
    price: Series[float] = pa.Field(gt=0)
    quantity: Series[float] = pa.Field(gt=0)
    traded_at: Series[pa.DateTime]

    class Config:
        coerce = True
        strict = True


@dataclass(frozen=True, slots=True)
class EhubMarketService:
    """EHUB boundary with exactly three public operations."""

    client: EhubClient = field(repr=False)
    fields: EhubFields = DEFAULT_EHUB_FIELDS

    @pa.check_types(lazy=True)
    def get_negotiable_tickers(self, wallet_id: int) -> DataFrame[TickerOutput]:
        """Fetch tickers and pivot every ``features_`` item into a column."""

        records = self.client.list_negotiable_tickers(wallet_id)
        columns = self.fields.ticker_columns()
        base = pd.json_normalize(records).drop(columns=self.fields.ticker_features)
        features = pd.json_normalize(
            records,
            record_path=self.fields.ticker_features,
            meta=[self.fields.ticker_product_id],
        ).pivot(
            index=self.fields.ticker_product_id,
            columns=self.fields.ticker_feature_name,
            values=self.fields.ticker_feature_value,
        )
        merged = (
            base.merge(
                features.rename_axis(columns=None).reset_index(),
                on=self.fields.ticker_product_id,
                how="left",
                validate="one_to_one",
            )
            .rename(columns=columns)
            .pipe(normalize_products)
        )
        return cast(
            DataFrame[TickerOutput],
            merged[list(TickerOutput.to_schema().columns)],
        )

    @pa.check_types(lazy=True)
    def get_deals(
        self,
        reference_date: TimestampLike,
        *,
        origin_operation_type: str | None = None,
        contract_id: int | None = None,
        accepted_statuses: Collection[str] = DEFAULT_ACCEPTED_STATUSES,
        accepted_operation_types: Collection[str] = DEFAULT_ACCEPTED_OPERATION_TYPES,
    ) -> DataFrame[DealOutput]:
        """Fetch every accepted deal from January 1 through the reference date."""

        reference = _timestamp(reference_date)
        columns = self.fields.deal_columns()
        records = self.client.list_all_deals(
            date(reference.year, 1, 1),
            reference.date(),
            origin_operation_type=origin_operation_type,
            contract_id=contract_id,
        )
        deals = (
            pd.DataFrame.from_records(records, columns=list(columns))
            .rename(columns=columns)
            .assign(
                deal_id=lambda frame: frame["deal_id"].astype("string"),
                product_id=lambda frame: frame["product_id"].astype("string"),
                price=lambda frame: pd.to_numeric(frame["price"], errors="raise"),
                quantity=lambda frame: pd.to_numeric(frame["quantity"], errors="raise"),
                traded_at=lambda frame: pd.to_datetime(
                    frame["traded_at"], utc=True
                ).dt.tz_localize(None),
            )
            .query(
                "traded_at <= @reference"
                " and status in @accepted_statuses"
                " and origin_operation_type in @accepted_operation_types",
                local_dict={
                    "reference": reference,
                    "accepted_statuses": list(accepted_statuses),
                    "accepted_operation_types": list(accepted_operation_types),
                },
            )
            .drop_duplicates("deal_id", keep="last")
            .sort_values(["product_id", "traded_at", "deal_id"], ignore_index=True)
        )
        return cast(
            DataFrame[DealOutput],
            deals[list(DealOutput.to_schema().columns)],
        )

    @pa.check_types(lazy=True)
    def get_market_deals(
        self,
        tickers: DataFrame[TickerOutput],
        deals: DataFrame[DealOutput],
    ) -> DataFrame[MarketDealOutput]:
        """Join deals to their negotiable-ticker metadata."""

        merged = deals.merge(
            tickers,
            on="product_id",
            how="inner",
            validate="many_to_one",
        ).sort_values(["product_id", "traded_at", "deal_id"], ignore_index=True)
        return cast(
            DataFrame[MarketDealOutput],
            merged[list(MarketDealOutput.to_schema().columns)],
        )


@pa.check_types(lazy=True)
def select_market_universe(
    deals: DataFrame[MarketDealOutput],
    *,
    submarket: str = "SE",
    granularities: frozenset[str] = DEFAULT_GRANULARITIES,
    source: str = "CON",
    price_type: str = "FIX",
) -> DataFrame[MarketDealOutput]:
    """Keep only the tradeable universe the forward curve is built from."""

    return cast(
        DataFrame[MarketDealOutput],
        deals.query(
            "submarket == @submarket"
            " and granularity in @granularities"
            " and source == @source"
            " and price_type == @price_type"
        ).reset_index(drop=True),
    )


class ProductQuoteOutput(ProductQuoteInput):
    """One fair price per product, ready for the liquidity policy."""

    median_price: Series[float] = pa.Field(gt=0)
    mad: Series[float] = pa.Field(ge=0)
    trade_count: Series[int] = pa.Field(ge=1)
    retained_count: Series[int] = pa.Field(ge=1)
    effective_weight: Series[float] = pa.Field(gt=0)
    precision: Series[float] = pa.Field(gt=0)

    class Config:
        coerce = True
        strict = True


@dataclass(frozen=True, slots=True)
class AnchorPolicy:
    """Statistical policy for robust recency-weighted prices."""

    half_life: pd.Timedelta = field(default_factory=lambda: pd.Timedelta("3D"))
    mad_threshold: float = 3.5
    minimum_trades: int = 1
    volume_weighted: bool = False

    def __post_init__(self) -> None:
        if (
            self.half_life <= pd.Timedelta(0)
            or not np.isfinite(self.mad_threshold)
            or self.mad_threshold <= 0
            or self.minimum_trades < 1
        ):
            raise ValueError("Anchor policy values must be positive")


DEFAULT_ANCHOR_POLICY = AnchorPolicy()

_PRODUCT_COLUMNS = [
    "product_id",
    "description",
    "start",
    "end",
    "delivery_months",
    "granularity",
]


@pa.check_types(lazy=True)
def estimate_anchor_prices(
    deals: DataFrame[MarketDealOutput],
    as_of: TimestampLike,
    *,
    policy: AnchorPolicy = DEFAULT_ANCHOR_POLICY,
) -> DataFrame[ProductQuoteOutput]:
    """Drop MAD outliers, then average the survivors with an exponential decay.

    The median is weighted by the same decay as the mean, so the centre tracks
    where the market is now: against a flat yearly median a genuine repricing
    would be rejected as an outlier. The MAD is deliberately *not* weighted,
    because a decay-weighted median concentrates most of the weight on one
    observation, which drives the MAD to zero and turns the filter into an
    equality test that discards perfectly ordinary quotes.

    A price is retained when ``|P - median| <= threshold * 1.4826 * MAD``; when
    ``MAD == 0`` only observations equal to the median survive. The anchor is
    the normalized weighted mean with ``w = 2 ** (-age / half_life)``, so a
    common decay cancels from the price and survives only in
    ``effective_weight``. ``precision`` turns that recency mass into the
    inverse-variance weight the solver uses; see ``_precision``.
    """

    cutoff = _timestamp(as_of)
    scored = deals.query("traded_at <= @cutoff", local_dict={"cutoff": cutoff}).assign(
        decay=lambda data: np.exp2(
            -data["traded_at"].rsub(cutoff).dt.total_seconds()
            / policy.half_life.total_seconds()
        ),
        trade_count=lambda data: data.groupby("product_id")["deal_id"].transform(
            "size"
        ),
    )
    scored = scored.assign(
        median_price=lambda data: data["product_id"].map(
            _weighted_median(data, "price")
        )
    )
    scored = scored.assign(
        deviation=lambda data: data["price"].sub(data["median_price"]).abs()
    )
    retained = (
        scored.assign(
            mad=lambda data: data.groupby("product_id")["deviation"].transform("median")
        )
        .query(
            "deviation == 0 or deviation <= @threshold * 1.4826 * mad",
            local_dict={"threshold": policy.mad_threshold},
        )
        .assign(
            retained_count=lambda data: data.groupby("product_id")["deal_id"].transform(
                "size"
            ),
        )
        .query(
            "retained_count >= @minimum",
            local_dict={"minimum": policy.minimum_trades},
        )
    )
    weighted = retained.assign(
        observation_weight=retained["decay"]
        * (retained["quantity"] if policy.volume_weighted else 1.0)
    ).assign(
        weighted_price=lambda data: data["price"] * data["observation_weight"],
    )
    quotes = (
        weighted.groupby(_PRODUCT_COLUMNS, as_index=False, sort=False, observed=True)
        .agg(
            price_sum=("weighted_price", "sum"),
            effective_weight=("observation_weight", "sum"),
            traded_at=("traded_at", "max"),
            median_price=("median_price", "first"),
            mad=("mad", "first"),
            trade_count=("trade_count", "first"),
            retained_count=("retained_count", "first"),
        )
        .assign(price=lambda data: data["price_sum"] / data["effective_weight"])
        .assign(precision=_precision)
        .sort_values("product_id", ignore_index=True)
    )
    return cast(
        DataFrame[ProductQuoteOutput],
        quotes[list(ProductQuoteOutput.to_schema().columns)],
    )


def _precision(quotes: pd.DataFrame) -> pd.Series:
    """Anchor precision: recency mass divided by robust price variance.

    ``effective_weight`` alone says how much *recent* evidence exists, not how
    much the evidence agrees: two fresh trades at 280 and 320 would outweigh a
    tight stack of quotes. Dividing by ``(1.4826 * MAD)**2`` makes the weight an
    inverse-variance estimate, which is what the solver's WLS expects. Products
    with ``MAD == 0`` observed no dispersion, so they borrow the median robust
    scale of the other products (or 1.0 when nobody dispersed) instead of
    claiming infinite confidence.
    """

    scale = 1.4826 * quotes["mad"]
    positive = scale[scale > 0.0]
    fallback = float(positive.median()) if len(positive) else 1.0
    return quotes["effective_weight"] / scale.where(scale > 0.0, fallback) ** 2


def _weighted_median(frame: pd.DataFrame, column: str) -> pd.Series:
    """Lowest value whose cumulative ``decay`` weight reaches half the product's."""

    ordered = frame.sort_values(["product_id", column])
    half = ordered.groupby("product_id")["decay"].transform("sum") * 0.5
    reached = ordered.groupby("product_id")["decay"].cumsum() >= half
    return ordered[reached].groupby("product_id")[column].first()


def _timestamp(value: TimestampLike) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp
