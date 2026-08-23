"""Canonical market anchors built from negotiable products and recent deals."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import cast

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

from .ehub import EhubClient, EhubSnapshot

TimestampLike = str | date | datetime | pd.Timestamp


class ProductInput(pa.DataFrameModel):
    """Canonical product universe accepted by the anchor estimator."""

    product_id: Series[str]
    description: Series[str]

    @pa.dataframe_check
    def unique_products(cls, frame: pd.DataFrame) -> bool:
        return bool(frame["product_id"].is_unique)

    class Config:
        coerce = True
        strict = True


class DealInput(pa.DataFrameModel):
    """Canonical executed-deal observations."""

    deal_id: Series[str]
    product_id: Series[str]
    price: Series[float] = pa.Field(gt=0)
    quantity: Series[float] = pa.Field(gt=0)
    traded_at: Series[pa.DateTime]

    @pa.dataframe_check
    def unique_deals(cls, frame: pd.DataFrame) -> bool:
        return bool(frame["deal_id"].is_unique)

    class Config:
        coerce = True
        strict = True


class LatestPriceOutput(pa.DataFrameModel):
    """Literal latest eligible deal for each negotiable product."""

    product_id: Series[str]
    description: Series[str]
    deal_id: Series[str]
    last_price: Series[float] = pa.Field(gt=0)
    last_traded_at: Series[pa.DateTime]

    @pa.dataframe_check
    def unique_prices(cls, frame: pd.DataFrame) -> bool:
        return bool(frame["product_id"].is_unique and frame["deal_id"].is_unique)

    class Config:
        coerce = True
        strict = True


class AnchorPriceOutput(pa.DataFrameModel):
    """One robust, time-decayed anchor price per product."""

    product_id: Series[str]
    description: Series[str]
    ewma_price: Series[float] = pa.Field(gt=0)
    last_inlier_price: Series[float] = pa.Field(gt=0)
    last_inlier_at: Series[pa.DateTime]
    median_price: Series[float] = pa.Field(gt=0)
    mad: Series[float] = pa.Field(ge=0)
    trade_count: Series[int] = pa.Field(ge=1)
    retained_count: Series[int] = pa.Field(ge=1)
    outlier_count: Series[int] = pa.Field(ge=0)
    age_days: Series[float] = pa.Field(ge=0)
    effective_weight: Series[float] = pa.Field(gt=0)

    @pa.dataframe_check
    def unique_anchors(cls, frame: pd.DataFrame) -> bool:
        return bool(frame["product_id"].is_unique)

    class Config:
        coerce = True
        strict = True


@dataclass(frozen=True, slots=True)
class EhubFields:
    """Mapping from the tenant payload into the canonical market contract.

    Current defaults use the camel-case representation of the public report
    model. Keeping the mapping as data avoids spreading provider field names
    through the statistical code.
    """

    ticker_product_id: str = "id"
    ticker_description: str = "description"
    ticker_features: str = "features_"
    ticker_feature_name: str = "name"
    ticker_feature_value: str = "value"
    ticker_start_feature: str = "start"
    ticker_end_feature: str = "end"
    deal_id: str = "id"
    deal_product_id: str = "productId"
    deal_price: str = "unitPrice"
    deal_quantity: str = "quantity"
    deal_traded_at: str = "createdAt"
    deal_status: str = "status"
    deal_origin_operation_type: str = "originOperationType"


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


DEFAULT_EHUB_FIELDS = EhubFields()
DEFAULT_ANCHOR_POLICY = AnchorPolicy()
DEFAULT_ACCEPTED_STATUSES = ("Ativo",)
DEFAULT_ACCEPTED_OPERATION_TYPES = ("Match", "Registro")


@dataclass(frozen=True, slots=True)
class NormalizedMarketData:
    """Provider-independent product and deal tables."""

    products: DataFrame[ProductInput]
    deals: DataFrame[DealInput]


@dataclass(frozen=True, slots=True)
class MarketAnchorBatch:
    """Inputs, cutoff and estimated prices from one market-data run."""

    as_of: pd.Timestamp
    products: DataFrame[ProductInput]
    deals: DataFrame[DealInput]
    anchors: DataFrame[AnchorPriceOutput]

    def price_series(self) -> pd.Series:
        return self.anchors.set_index("product_id")["ewma_price"].rename("ewma_price")

    def weight_series(self) -> pd.Series:
        return self.anchors.set_index("product_id")["effective_weight"].rename(
            "effective_weight"
        )

    def latest_prices(self) -> DataFrame[LatestPriceOutput]:
        return latest_prices(self.products, self.deals, self.as_of)

    def last_price_series(self) -> pd.Series:
        return (
            self.latest_prices()
            .set_index("product_id")["last_price"]
            .rename("last_price")
        )

    def missing_products(self) -> pd.DataFrame:
        priced = self.anchors["product_id"]
        return self.products.query(
            "product_id not in @priced",
            local_dict={"priced": priced},
        ).reset_index(drop=True)


def normalize_ehub_snapshot(
    snapshot: EhubSnapshot,
    *,
    fields: EhubFields = DEFAULT_EHUB_FIELDS,
    accepted_statuses: Collection[str] | None = None,
    accepted_operation_types: Collection[str] | None = None,
) -> NormalizedMarketData:
    """Translate EHUB JSON records and retain deals from negotiable tickers."""

    tickers = (
        pd.DataFrame.from_records(
            snapshot.negotiable_tickers,
            columns=[fields.ticker_product_id, fields.ticker_description],
        )
        .rename(
            columns={
                fields.ticker_product_id: "product_id",
                fields.ticker_description: "description",
            }
        )
        .astype({"product_id": "string", "description": "string"})
        .drop_duplicates("product_id", keep="last")
    )
    deals = (
        pd.DataFrame.from_records(
            snapshot.deals,
            columns=[
                fields.deal_id,
                fields.deal_product_id,
                fields.deal_price,
                fields.deal_quantity,
                fields.deal_traded_at,
                fields.deal_status,
                fields.deal_origin_operation_type,
            ],
        )
        .rename(
            columns={
                fields.deal_id: "deal_id",
                fields.deal_product_id: "product_id",
                fields.deal_price: "price",
                fields.deal_quantity: "quantity",
                fields.deal_traded_at: "traded_at",
                fields.deal_status: "status",
                fields.deal_origin_operation_type: "origin_operation_type",
            }
        )
        .assign(
            deal_id=lambda data: data["deal_id"].astype("string"),
            product_id=lambda data: data["product_id"].astype("string"),
            price=lambda data: pd.to_numeric(data["price"], errors="raise"),
            quantity=lambda data: pd.to_numeric(data["quantity"], errors="raise"),
            traded_at=lambda data: _timestamp_series(data["traded_at"]),
            status=lambda data: data["status"].astype("string"),
            origin_operation_type=lambda data: data["origin_operation_type"].astype(
                "string"
            ),
        )
        .drop_duplicates("deal_id", keep="last")
    )
    selected = deals.assign(
        _accepted_status=lambda data: (
            True
            if accepted_statuses is None
            else data["status"].isin(accepted_statuses)
        ),
        _accepted_operation=lambda data: (
            True
            if accepted_operation_types is None
            else data["origin_operation_type"].isin(accepted_operation_types)
        ),
    )
    eligible = selected.query("_accepted_status and _accepted_operation").merge(
        tickers,
        on="product_id",
        how="inner",
        validate="many_to_one",
    )
    canonical_deals = (
        eligible[["deal_id", "product_id", "price", "quantity", "traded_at"]]
        .reset_index(drop=True)
        .pipe(DealInput.validate, lazy=True)
    )
    products = tickers.reset_index(drop=True).pipe(ProductInput.validate, lazy=True)
    return NormalizedMarketData(
        products=cast(DataFrame[ProductInput], products),
        deals=cast(DataFrame[DealInput], canonical_deals),
    )


def latest_prices(
    products: DataFrame[ProductInput],
    deals: DataFrame[DealInput],
    as_of: TimestampLike,
) -> DataFrame[LatestPriceOutput]:
    """Select the latest eligible deal per product without statistical smoothing."""

    _, universe, eligible = _market_before(products, deals, as_of)
    frame = (
        eligible.merge(
            universe,
            on="product_id",
            how="inner",
            validate="many_to_one",
        )
        .drop_duplicates("product_id", keep="last")
        .rename(columns={"price": "last_price", "traded_at": "last_traded_at"})[
            [
                "product_id",
                "description",
                "deal_id",
                "last_price",
                "last_traded_at",
            ]
        ]
        .sort_values("product_id", ignore_index=True)
        .pipe(LatestPriceOutput.validate, lazy=True)
    )
    return cast(DataFrame[LatestPriceOutput], frame)


def estimate_anchor_prices(
    products: DataFrame[ProductInput],
    deals: DataFrame[DealInput],
    as_of: TimestampLike,
    *,
    policy: AnchorPolicy = DEFAULT_ANCHOR_POLICY,
) -> MarketAnchorBatch:
    """Filter outliers with median absolute deviation, then apply EWMA.

    The final price is a normalized exponentially decayed average. A common
    decay from the latest trade to ``as_of`` cancels from the price, while its
    value remains visible in ``effective_weight`` as a recency diagnostic.
    """

    cutoff, universe, eligible = _market_before(products, deals, as_of)
    scored = (
        eligible.assign(
            trade_count=lambda data: data.groupby("product_id")["deal_id"].transform(
                "size"
            ),
            median_price=lambda data: data.groupby("product_id")["price"].transform(
                "median"
            ),
        )
        .assign(deviation=lambda data: data["price"].sub(data["median_price"]).abs())
        .assign(
            mad=lambda data: data.groupby("product_id")["deviation"].transform(
                "median"
            ),
            robust_scale=lambda data: 1.4826 * data["mad"],
        )
        .assign(
            inlier=lambda data: (
                data["deviation"].eq(0.0)
                | (
                    data["robust_scale"].gt(0.0)
                    & data["deviation"].le(policy.mad_threshold * data["robust_scale"])
                )
            )
        )
        .query("inlier")
        .assign(
            retained_count=lambda data: data.groupby("product_id")["deal_id"].transform(
                "size"
            )
        )
        .query("retained_count >= @policy.minimum_trades")
    )
    half_life_seconds = policy.half_life.total_seconds()
    quantity_weight = scored["quantity"] if policy.volume_weighted else 1.0
    weighted = (
        scored.assign(
            age_seconds=lambda data: data["traded_at"].rsub(cutoff).dt.total_seconds(),
        )
        .assign(
            decay_weight=lambda data: np.exp2(-data["age_seconds"] / half_life_seconds)
        )
        .assign(
            observation_weight=lambda data: data["decay_weight"] * quantity_weight,
            weighted_price=lambda data: data["price"] * data["observation_weight"],
        )
    )
    summary = (
        weighted.groupby("product_id", as_index=False, sort=False)
        .agg(
            price_sum=("weighted_price", "sum"),
            effective_weight=("observation_weight", "sum"),
            median_price=("median_price", "first"),
            mad=("mad", "first"),
            trade_count=("trade_count", "first"),
            retained_count=("retained_count", "first"),
        )
        .assign(
            ewma_price=lambda data: data["price_sum"] / data["effective_weight"],
            outlier_count=lambda data: data["trade_count"] - data["retained_count"],
        )
        .drop(columns="price_sum")
    )
    latest = weighted.drop_duplicates("product_id", keep="last").rename(
        columns={"price": "last_inlier_price", "traded_at": "last_inlier_at"}
    )[["product_id", "last_inlier_price", "last_inlier_at"]]
    anchors = (
        universe.merge(summary, on="product_id", how="inner", validate="one_to_one")
        .merge(latest, on="product_id", how="inner", validate="one_to_one")
        .assign(
            age_days=lambda data: (
                data["last_inlier_at"].rsub(cutoff).dt.total_seconds() / 86_400.0
            )
        )[
            [
                "product_id",
                "description",
                "ewma_price",
                "last_inlier_price",
                "last_inlier_at",
                "median_price",
                "mad",
                "trade_count",
                "retained_count",
                "outlier_count",
                "age_days",
                "effective_weight",
            ]
        ]
        .sort_values("product_id", ignore_index=True)
        .pipe(AnchorPriceOutput.validate, lazy=True)
    )
    return MarketAnchorBatch(
        as_of=cutoff,
        products=universe,
        deals=eligible,
        anchors=cast(DataFrame[AnchorPriceOutput], anchors),
    )


def load_ehub_anchors(
    client: EhubClient,
    wallet_id: int,
    start: date,
    as_of: TimestampLike,
    *,
    fields: EhubFields = DEFAULT_EHUB_FIELDS,
    policy: AnchorPolicy = DEFAULT_ANCHOR_POLICY,
    accepted_statuses: Collection[str] | None = DEFAULT_ACCEPTED_STATUSES,
    accepted_operation_types: Collection[str] | None = (
        DEFAULT_ACCEPTED_OPERATION_TYPES
    ),
    origin_operation_type: str | None = None,
    contract_id: int | None = None,
) -> MarketAnchorBatch:
    """Fetch, normalize and estimate the current EHUB anchor surface."""

    cutoff, market = _load_ehub_market(
        client,
        wallet_id,
        start,
        as_of,
        fields=fields,
        accepted_statuses=accepted_statuses,
        accepted_operation_types=accepted_operation_types,
        origin_operation_type=origin_operation_type,
        contract_id=contract_id,
    )
    return estimate_anchor_prices(
        market.products,
        market.deals,
        cutoff,
        policy=policy,
    )


def load_ehub_latest_prices(
    client: EhubClient,
    wallet_id: int,
    start: date,
    as_of: TimestampLike,
    *,
    fields: EhubFields = DEFAULT_EHUB_FIELDS,
    accepted_statuses: Collection[str] | None = DEFAULT_ACCEPTED_STATUSES,
    accepted_operation_types: Collection[str] | None = (
        DEFAULT_ACCEPTED_OPERATION_TYPES
    ),
    origin_operation_type: str | None = None,
    contract_id: int | None = None,
) -> DataFrame[LatestPriceOutput]:
    """Fetch and return one literal latest eligible deal per product."""

    cutoff, market = _load_ehub_market(
        client,
        wallet_id,
        start,
        as_of,
        fields=fields,
        accepted_statuses=accepted_statuses,
        accepted_operation_types=accepted_operation_types,
        origin_operation_type=origin_operation_type,
        contract_id=contract_id,
    )
    return latest_prices(market.products, market.deals, cutoff)


def _load_ehub_market(
    client: EhubClient,
    wallet_id: int,
    start: date,
    as_of: TimestampLike,
    *,
    fields: EhubFields,
    accepted_statuses: Collection[str] | None,
    accepted_operation_types: Collection[str] | None,
    origin_operation_type: str | None,
    contract_id: int | None,
) -> tuple[pd.Timestamp, NormalizedMarketData]:
    cutoff = _timestamp(as_of)
    snapshot = client.snapshot(
        wallet_id,
        start,
        cutoff.date(),
        origin_operation_type=origin_operation_type,
        contract_id=contract_id,
    )
    market = normalize_ehub_snapshot(
        snapshot,
        fields=fields,
        accepted_statuses=accepted_statuses,
        accepted_operation_types=accepted_operation_types,
    )
    return cutoff, market


def _market_before(
    products: DataFrame[ProductInput],
    deals: DataFrame[DealInput],
    as_of: TimestampLike,
) -> tuple[pd.Timestamp, DataFrame[ProductInput], DataFrame[DealInput]]:
    cutoff = _timestamp(as_of)
    universe = cast(
        DataFrame[ProductInput],
        ProductInput.validate(products, lazy=True),
    )
    observations = cast(
        DataFrame[DealInput],
        DealInput.validate(
            deals.assign(traded_at=lambda data: _timestamp_series(data["traded_at"])),
            lazy=True,
        ),
    )
    eligible = (
        observations.merge(
            universe[["product_id"]],
            on="product_id",
            how="inner",
            validate="many_to_one",
        )
        .query("traded_at <= @cutoff")
        .sort_values(["product_id", "traded_at", "deal_id"], ignore_index=True)
        .pipe(DealInput.validate, lazy=True)
    )
    return cutoff, universe, cast(DataFrame[DealInput], eligible)


def _timestamp(value: TimestampLike) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True).tz_localize(None)


def _timestamp_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert(None)
