"""Functional assembly of the market block before numerical neutralization."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

from .anchor_selection import build_dual_anchor_plan
from .forward_curve import ForwardCurveResult, neutralize_forward_curve
from .granularity import (
    DEFAULT_CURVE_GRANULARITY,
    CurveGranularity,
    ProductDeliveryInput,
    ProductMarkInput,
    SelectedTenorPriceOutput,
    annual_anchor_ids,
    delivery_profiles,
)
from .neutralization import CurveInput, _prepare_curve, _price_frame


class UnbalancedCurveOutput(CurveInput):
    """Market curve after one product has been selected for each tenor."""

    product_id: Series[str]
    granularity: Series[str] = pa.Field(
        isin=["monthly", "quarterly", "semiannual", "annual"]
    )
    traded_at: Series[pa.DateTime]
    overridden: Series[bool]

    class Config:
        coerce = True
        strict = False


@pa.check_types(lazy=True)
def build_unbalanced_curve(
    curve: pd.DataFrame,
    selected: DataFrame[SelectedTenorPriceOutput],
) -> DataFrame[UnbalancedCurveOutput]:
    """Apply selected prices and freeze IPCA at each contiguous block head."""

    monthly = _prepare_curve(curve)
    market = (
        monthly.merge(
            selected.rename(columns={"price": "selected_price"})[
                [
                    "tenor",
                    "product_id",
                    "granularity",
                    "selected_price",
                    "traded_at",
                    "overridden",
                ]
            ],
            on="tenor",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("tenor", ignore_index=True)
        .assign(
            block_number=lambda frame: (
                frame["product_id"].ne(frame["product_id"].shift()).cumsum()
            ),
            price=lambda frame: frame["selected_price"],
        )
        .assign(
            block=lambda frame: (
                frame["product_id"].astype(str)
                + ":"
                + frame["block_number"].astype(str)
            ),
            index_factor=lambda frame: frame.groupby("block_number", sort=False)[
                "index_factor"
            ].transform("first"),
        )
        .drop(columns=["selected_price", "block_number"])
        .pipe(UnbalancedCurveOutput.validate, lazy=True)
    )
    return cast(DataFrame[UnbalancedCurveOutput], market)


def neutralize_market_curve(
    curve: pd.DataFrame,
    products: DataFrame[ProductDeliveryInput],
    raw_prices: pd.Series | pd.DataFrame,
    observed_at: pd.Series,
    dcide_curve: pd.DataFrame,
    *,
    reference_date: object,
    cutoff: object,
    granularity: CurveGranularity = DEFAULT_CURVE_GRANULARITY,
    quality: pd.Series | float = 1.0,
    raw_weight: float = 1.0,
    indexed_weight: float = 1.0,
    smoothness: float = 0.0,
    tolerance: float = 1e-9,
) -> ForwardCurveResult:
    """Select, index by block, jointly fit raw/indexed anchors, then append DCIDE."""

    prices = _price_frame(raw_prices)
    prices.index = pd.Index(prices.index.map(str), name="product_id")
    marks = pd.DataFrame(
        {
            "product_id": prices.index,
            "price": prices.iloc[:, 0].to_numpy(),
            "traded_at": pd.to_datetime(
                observed_at.rename(index=str).reindex(prices.index),
                utc=True,
            )
            .dt.tz_localize(None)
            .to_numpy(),
        }
    ).pipe(ProductMarkInput.validate, lazy=True)
    prepared = _prepare_curve(curve)
    start = _month(reference_date)
    end = _month(cutoff)
    tenors = pd.DatetimeIndex(
        prepared.query(
            "@start <= tenor <= @end",
            local_dict={"start": start, "end": end},
        )["tenor"]
    )
    selected = granularity.select(products, marks, tenors)
    unbalanced = build_unbalanced_curve(prepared, selected)

    selected_ids = pd.Index(selected["product_id"].drop_duplicates(), name="product_id")
    anchor_ids = selected_ids.append(
        annual_anchor_ids(products, marks, tenors)
    ).drop_duplicates()
    profiles = delivery_profiles(products, anchor_ids, tenors)
    raw = prices.reindex(anchor_ids)
    factor_by_month = prepared.set_index("tenor")["index_factor"]
    quote_factor = (
        products.set_index("product_id")
        .reindex(anchor_ids)["start"]
        .map(factor_by_month)
    )
    indexed = raw.mul(quote_factor, axis="index")
    anchor_quality = (
        quality.rename(index=str).reindex(anchor_ids)
        if isinstance(quality, pd.Series)
        else quality
    )
    plan = build_dual_anchor_plan(
        unbalanced,
        profiles,
        raw,
        indexed,
        quality=anchor_quality,
        raw_weight=raw_weight,
        indexed_weight=indexed_weight,
        cutoff=end,
    )
    return neutralize_forward_curve(
        plan,
        dcide_curve,
        prior_strength=0.0,
        smoothness=smoothness,
        tolerance=tolerance,
    )


def _month(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    local = stamp.tz_localize(None) if stamp.tzinfo else stamp
    return local.to_period("M").to_timestamp()
