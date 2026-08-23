"""Liquidity policy used while assembling the unbalanced monthly curve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

Granularity = Literal["monthly", "quarterly", "semiannual", "annual"]
GranularityOverride = tuple[object, object, Granularity]

GRANULARITY_MONTHS: dict[Granularity, int] = {
    "monthly": 1,
    "quarterly": 3,
    "semiannual": 6,
    "annual": 12,
}


class ProductDeliveryInput(pa.DataFrameModel):
    """Normalized product definition extracted from negotiable-ticker features."""

    product_id: Series[str] = pa.Field(unique=True)
    description: Series[str]
    start: Series[pa.DateTime]
    end: Series[pa.DateTime]
    delivery_months: Series[int] = pa.Field(isin=[1, 3, 6, 12])
    granularity: Series[str] = pa.Field(
        isin=["monthly", "quarterly", "semiannual", "annual"]
    )

    @pa.dataframe_check
    def valid_delivery(cls, frame: pd.DataFrame) -> bool:
        return bool((frame["start"] <= frame["end"]).all())

    class Config:
        coerce = True
        strict = False


class ProductMarkInput(pa.DataFrameModel):
    """Latest eligible mark for each product."""

    product_id: Series[str] = pa.Field(unique=True)
    price: Series[float] = pa.Field(gt=0)
    traded_at: Series[pa.DateTime]

    class Config:
        coerce = True
        strict = False


class SelectedTenorPriceOutput(pa.DataFrameModel):
    """One product selected by the liquidity policy for each covered tenor."""

    tenor: Series[pa.DateTime] = pa.Field(unique=True)
    product_id: Series[str]
    description: Series[str]
    start: Series[pa.DateTime]
    end: Series[pa.DateTime]
    delivery_months: Series[int] = pa.Field(isin=[1, 3, 6, 12])
    granularity: Series[str] = pa.Field(
        isin=["monthly", "quarterly", "semiannual", "annual"]
    )
    price: Series[float] = pa.Field(gt=0)
    traded_at: Series[pa.DateTime]
    overridden: Series[bool]

    class Config:
        coerce = True
        strict = True


DEFAULT_PRIORITY: tuple[Granularity, ...] = (
    "monthly",
    "quarterly",
    "semiannual",
    "annual",
)


@dataclass(frozen=True, slots=True)
class CurveGranularity:
    """Injectable product-selection policy.

    Each override is ``(start, end, granularity)`` and replaces the default
    priority inside its inclusive monthly interval. A forced granularity has
    no fallback: this is what makes "use annual for all of 2027" deterministic.
    """

    priority: tuple[Granularity, ...] = DEFAULT_PRIORITY
    overrides: tuple[GranularityOverride, ...] = ()

    def select(
        self,
        products: DataFrame[ProductDeliveryInput],
        marks: DataFrame[ProductMarkInput],
        tenors: pd.DatetimeIndex,
    ) -> DataFrame[SelectedTenorPriceOutput]:
        """Choose the freshest product at the best allowed granularity."""

        months = pd.DataFrame(
            {"tenor": _months(tenors).drop_duplicates().sort_values()}
        )
        candidates = (
            months.merge(products.merge(marks, on="product_id"), how="cross")
            .query("start <= tenor <= end")
            .assign(
                priority=lambda frame: frame["granularity"].map(
                    dict(zip(self.priority, range(len(self.priority)), strict=True))
                ),
            )
        )
        overrides = _override_frame(self.overrides, months["tenor"])
        ranked = (
            candidates.merge(overrides, on="tenor", how="left")
            .query("forced.isna() or granularity == forced")
            .assign(
                overridden=lambda frame: frame["forced"].notna(),
                effective_priority=lambda frame: frame["priority"].where(
                    frame["forced"].isna(),
                    0,
                ),
            )
            .dropna(subset=["effective_priority"])
            .sort_values(
                ["tenor", "effective_priority", "traded_at", "product_id"],
                ascending=[True, True, False, True],
            )
            .drop_duplicates("tenor", keep="first")
            .sort_values("tenor", ignore_index=True)
        )
        selected = ranked[
            [
                "tenor",
                "product_id",
                "description",
                "start",
                "end",
                "delivery_months",
                "granularity",
                "price",
                "traded_at",
                "overridden",
            ]
        ].pipe(SelectedTenorPriceOutput.validate, lazy=True)
        return cast(DataFrame[SelectedTenorPriceOutput], selected)


DEFAULT_CURVE_GRANULARITY = CurveGranularity()


def normalize_products(frame: pd.DataFrame) -> DataFrame[ProductDeliveryInput]:
    """Canonicalize delivery months and derive the four supported regimes."""

    start = _months(frame["start"])
    end = _months(frame["end"])
    delivery_months = (
        (end.dt.year - start.dt.year) * 12 + end.dt.month - start.dt.month + 1
    )
    normalized = (
        frame.assign(
            product_id=lambda data: data["product_id"].astype("string"),
            description=lambda data: data["description"].astype("string"),
            start=start,
            end=end,
            delivery_months=delivery_months,
            granularity=delivery_months.map(
                {months: name for name, months in GRANULARITY_MONTHS.items()}
            ),
        )
        .sort_values("product_id", ignore_index=True)
        .pipe(ProductDeliveryInput.validate, lazy=True)
    )
    return cast(DataFrame[ProductDeliveryInput], normalized)


def delivery_profiles(
    products: DataFrame[ProductDeliveryInput],
    product_ids: pd.Index,
    tenors: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create product-by-month delivery indicators without row-wise Python."""

    selected = products.set_index("product_id").reindex(product_ids)
    months = _months(tenors)
    values = (months.to_numpy()[None, :] >= selected["start"].to_numpy()[:, None]) & (
        months.to_numpy()[None, :] <= selected["end"].to_numpy()[:, None]
    )
    return pd.DataFrame(
        values.astype(float),
        index=pd.Index(product_ids, name="product_id"),
        columns=pd.DatetimeIndex(months, name="tenor"),
    )


def annual_anchor_ids(
    products: DataFrame[ProductDeliveryInput],
    marks: DataFrame[ProductMarkInput],
    tenors: pd.DatetimeIndex,
) -> pd.Index:
    """Return the freshest annual product for each quoted delivery interval."""

    first, last = _months(tenors).min(), _months(tenors).max()
    return pd.Index(
        products.merge(marks, on="product_id")
        .query(
            "granularity == 'annual' and start <= @last and end >= @first",
            local_dict={"first": first, "last": last},
        )
        .sort_values(["start", "end", "traded_at", "product_id"])
        .drop_duplicates(["start", "end"], keep="last")["product_id"],
        name="product_id",
    )


def _override_frame(
    overrides: tuple[GranularityOverride, ...],
    tenors: pd.Series,
) -> pd.DataFrame:
    rules = pd.DataFrame(overrides, columns=["start", "end", "forced"]).assign(
        rule=lambda frame: frame.index,
        start=lambda frame: _months(frame["start"]),
        end=lambda frame: _months(frame["end"]),
    )
    if rules.empty:
        return pd.DataFrame(
            {"tenor": tenors, "forced": pd.Series(pd.NA, index=tenors.index)}
        )
    return (
        tenors.to_frame("tenor")
        .merge(rules, how="cross")
        .query("start <= tenor <= end")
        .sort_values(["tenor", "rule"])
        .drop_duplicates("tenor", keep="last")[["tenor", "forced"]]
    )


def _months(values: object) -> pd.Series:
    return pd.Series(pd.to_datetime(values)).dt.to_period("M").dt.to_timestamp()
