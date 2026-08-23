"""Small typed service around the three EHUB market-data operations."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

from .ehub import EhubClient
from .granularity import (
    DEFAULT_CURVE_GRANULARITY,
    CurveGranularity,
    ProductDeliveryInput,
    ProductMarkInput,
    SelectedTenorPriceOutput,
    normalize_products,
)
from .market_anchors import DEFAULT_EHUB_FIELDS, EhubFields

TimestampLike = str | date | datetime | pd.Timestamp


class MarketDealOutput(pa.DataFrameModel):
    """Canonical deal rows returned by the EHUB report."""

    deal_id: Series[str] = pa.Field(unique=True)
    product_id: Series[str]
    price: Series[float] = pa.Field(gt=0)
    quantity: Series[float] = pa.Field(gt=0)
    traded_at: Series[pa.DateTime]
    status: Series[str]
    origin_operation_type: Series[str]

    class Config:
        coerce = True
        strict = True


class CurveTenorInput(pa.DataFrameModel):
    """Monthly tenors requested from the market segment."""

    tenor: Series[pa.DateTime] = pa.Field(unique=True)

    class Config:
        coerce = True
        strict = False


@dataclass(frozen=True, slots=True)
class EhubMarketService:
    """EHUB boundary with exactly three public operations."""

    client: EhubClient = field(repr=False)
    fields: EhubFields = DEFAULT_EHUB_FIELDS

    @pa.check_types(lazy=True)
    def get_negotiable_tickers(
        self,
        wallet_id: int,
    ) -> DataFrame[ProductDeliveryInput]:
        """Fetch tickers and pivot every ``features_`` item into a column."""

        records = self.client.list_negotiable_tickers(wallet_id)
        base = (
            pd.json_normalize(records)
            .drop(columns=self.fields.ticker_features)
            .rename(
                columns={
                    self.fields.ticker_product_id: "product_id",
                    self.fields.ticker_description: "description",
                }
            )
        )
        features = _feature_columns(records, self.fields).rename(
            columns={
                self.fields.ticker_product_id: "product_id",
                self.fields.ticker_start_feature: "start",
                self.fields.ticker_end_feature: "end",
            }
        )
        return normalize_products(
            base.merge(features, on="product_id", how="left", validate="one_to_one")
        )

    @pa.check_types(lazy=True)
    def get_deals(
        self,
        reference_date: TimestampLike,
        *,
        origin_operation_type: str | None = None,
        contract_id: int | None = None,
    ) -> DataFrame[MarketDealOutput]:
        """Fetch all deals from January 1 through the reference date."""

        reference = pd.Timestamp(reference_date)
        records = self.client.list_all_deals(
            date(reference.year, 1, 1),
            reference.date(),
            origin_operation_type=origin_operation_type,
            contract_id=contract_id,
        )
        deals = (
            pd.DataFrame.from_records(
                records,
                columns=[
                    self.fields.deal_id,
                    self.fields.deal_product_id,
                    self.fields.deal_price,
                    self.fields.deal_quantity,
                    self.fields.deal_traded_at,
                    self.fields.deal_status,
                    self.fields.deal_origin_operation_type,
                ],
            )
            .rename(
                columns={
                    self.fields.deal_id: "deal_id",
                    self.fields.deal_product_id: "product_id",
                    self.fields.deal_price: "price",
                    self.fields.deal_quantity: "quantity",
                    self.fields.deal_traded_at: "traded_at",
                    self.fields.deal_status: "status",
                    self.fields.deal_origin_operation_type: "origin_operation_type",
                }
            )
            .assign(
                deal_id=lambda frame: frame["deal_id"].astype("string"),
                product_id=lambda frame: frame["product_id"].astype("string"),
                price=lambda frame: pd.to_numeric(frame["price"], errors="raise"),
                quantity=lambda frame: pd.to_numeric(frame["quantity"], errors="raise"),
                traded_at=lambda frame: _timestamps(frame["traded_at"]),
                status=lambda frame: frame["status"].astype("string"),
                origin_operation_type=lambda frame: frame[
                    "origin_operation_type"
                ].astype("string"),
            )
            .drop_duplicates("deal_id", keep="last")
            .sort_values(["product_id", "traded_at", "deal_id"], ignore_index=True)
        )
        return cast(DataFrame[MarketDealOutput], deals)

    @pa.check_types(lazy=True)
    def get_latest_prices(
        self,
        tickers: DataFrame[ProductDeliveryInput],
        deals: DataFrame[MarketDealOutput],
        curve: DataFrame[CurveTenorInput],
        *,
        reference_date: TimestampLike,
        cutoff: TimestampLike,
        granularity: CurveGranularity = DEFAULT_CURVE_GRANULARITY,
        accepted_statuses: Collection[str] = ("Ativo",),
        accepted_operation_types: Collection[str] = ("Match", "Registro"),
    ) -> DataFrame[SelectedTenorPriceOutput]:
        """Select one latest market product per tenor through the cutoff."""

        as_of = _timestamp(reference_date)
        start = _month(reference_date)
        end = _month(cutoff)
        marks = (
            deals.query(
                "traded_at <= @as_of and "
                "status in @accepted_statuses and "
                "origin_operation_type in @accepted_operation_types",
                local_dict={
                    "as_of": as_of,
                    "accepted_statuses": accepted_statuses,
                    "accepted_operation_types": accepted_operation_types,
                },
            )
            .sort_values(["product_id", "traded_at", "deal_id"])
            .drop_duplicates("product_id", keep="last")
            .loc[:, ["product_id", "price", "traded_at"]]
            .pipe(ProductMarkInput.validate, lazy=True)
        )
        tenors = pd.DatetimeIndex(
            curve.assign(
                tenor=lambda frame: (
                    pd.to_datetime(frame["tenor"]).dt.to_period("M").dt.to_timestamp()
                )
            ).query(
                "@start <= tenor <= @end",
                local_dict={"start": start, "end": end},
            )["tenor"]
        )
        return granularity.select(tickers, marks, tenors)


def _feature_columns(
    records: tuple[object, ...],
    fields: EhubFields,
) -> pd.DataFrame:
    return (
        pd.json_normalize(
            records,
            record_path=fields.ticker_features,
            meta=[fields.ticker_product_id],
        )
        .pivot(
            index=fields.ticker_product_id,
            columns=fields.ticker_feature_name,
            values=fields.ticker_feature_value,
        )
        .rename_axis(columns=None)
        .reset_index()
    )


def _timestamp(value: TimestampLike) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp


def _month(value: TimestampLike) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    local = stamp.tz_localize(None) if stamp.tzinfo else stamp
    return local.to_period("M").to_timestamp()


def _timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_localize(None)
