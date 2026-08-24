"""Pandas boundary between market-data flows and the pure Pricer solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import pandera.pandas as pa
from numpy.typing import ArrayLike, NDArray
from pandera.typing.pandas import DataFrame, Series
from pricer.curves.arbitrage import (
    LinearObservations,
    block_basis,
    solve_curve,
)

FloatArray = NDArray[np.float64]


def cashflow_matrix(
    delivery: ArrayLike,
    *,
    monthly_weight: ArrayLike = 1.0,
    monthly_price_factor: ArrayLike = 1.0,
    quote_price_factor: ArrayLike = 1.0,
) -> FloatArray:
    """Convert prepared delivery economics into a numerical exposure matrix.

    This is the orchestration boundary where energy, discounting and price
    corrections become coefficients. The Pricer receives only the resulting
    matrix and never sees these domain concepts.
    """

    profile = np.atleast_2d(np.asarray(delivery, dtype=float))
    month_weight = np.broadcast_to(
        np.asarray(monthly_weight, dtype=float),
        (profile.shape[1],),
    )
    month_factor = np.broadcast_to(
        np.asarray(monthly_price_factor, dtype=float),
        (profile.shape[1],),
    )
    quote_factor = np.broadcast_to(
        np.asarray(quote_price_factor, dtype=float),
        (profile.shape[0],),
    )
    denominator = (profile * month_weight).sum(axis=1) * quote_factor
    if (
        not np.isfinite(profile).all()
        or not np.isfinite(month_weight).all()
        or not np.isfinite(month_factor).all()
        or not np.isfinite(quote_factor).all()
        or np.any(profile < 0.0)
        or np.any(month_weight <= 0.0)
        or np.any(month_factor <= 0.0)
        or np.any(quote_factor <= 0.0)
        or np.any(denominator <= 0.0)
    ):
        raise ValueError("Every observation must have positive delivered weight")
    return profile * month_weight * month_factor / denominator[:, None]


class CurveInput(pa.DataFrameModel):
    """Canonical monthly state handed to the neutralization service."""

    tenor: Series[pa.DateTime]
    price: Series[float]
    energy_weight: Series[float] = pa.Field(gt=0)
    discount_factor: Series[float] = pa.Field(gt=0)
    index_factor: Series[float] = pa.Field(gt=0)
    index_base: Series[pa.DateTime]
    index_start: Series[pa.DateTime]
    ipca_base: Series[float] = pa.Field(gt=0)
    ipca_forecast: Series[float] = pa.Field(gt=0)
    floor: Series[float]
    cap: Series[float]
    block: Series[str]
    seasonal_shape: Series[float] = pa.Field(gt=0)

    @pa.dataframe_check
    def unique_tenors(cls, frame: pd.DataFrame) -> bool:
        return bool(frame["tenor"].is_unique)

    @pa.dataframe_check
    def ordered_bounds(cls, frame: pd.DataFrame) -> bool:
        return bool(frame["floor"].le(frame["cap"]).all())

    class Config:
        coerce = True
        strict = False


class CurveOutput(pa.DataFrameModel):
    tenor: Series[pa.DateTime]
    scenario: Series[str]
    initial_price: Series[float]
    price: Series[float]

    class Config:
        coerce = True
        strict = False


class AnchorOutput(pa.DataFrameModel):
    anchor: Series[str]
    scenario: Series[str]
    target: Series[float]
    fitted: Series[float]
    residual: Series[float]

    class Config:
        coerce = True
        strict = False


class ScenarioOutput(pa.DataFrameModel):
    scenario: Series[str]
    objective: Series[float]
    max_violation: Series[float] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = False


@dataclass(frozen=True, slots=True)
class AnchorMatrix:
    """Already identified anchors, with rows aligned by their index.

    The row index is metadata only.  ``exposure`` contains the complete
    economic definition consumed by Pricer, while ``prices`` may be one Series
    or a DataFrame of scenarios.
    """

    exposure: pd.DataFrame
    prices: pd.Series | pd.DataFrame
    lower: pd.Series | pd.DataFrame | float | None = None
    upper: pd.Series | pd.DataFrame | float | None = None
    weight: pd.Series | pd.DataFrame | float | None = None

    @classmethod
    def exact(
        cls,
        exposure: pd.DataFrame,
        prices: pd.Series | pd.DataFrame,
    ) -> AnchorMatrix:
        return cls(exposure=exposure, prices=prices)

    @classmethod
    def soft(
        cls,
        exposure: pd.DataFrame,
        prices: pd.Series | pd.DataFrame,
        *,
        weight: pd.Series | pd.DataFrame | float = 1.0,
    ) -> AnchorMatrix:
        return cls(
            exposure=exposure,
            prices=prices,
            lower=-np.inf,
            upper=np.inf,
            weight=weight,
        )

    @classmethod
    def bounded(
        cls,
        exposure: pd.DataFrame,
        prices: pd.Series | pd.DataFrame,
        *,
        lower: pd.Series | pd.DataFrame | float,
        upper: pd.Series | pd.DataFrame | float,
        weight: pd.Series | pd.DataFrame | float | None = None,
    ) -> AnchorMatrix:
        return cls(
            exposure=exposure,
            prices=prices,
            lower=lower,
            upper=upper,
            weight=weight,
        )


@dataclass(frozen=True, slots=True)
class NeutralizationResult:
    curve: DataFrame[CurveOutput]
    anchors: DataFrame[AnchorOutput]
    scenarios: DataFrame[ScenarioOutput]

    def wide_curve(self) -> pd.DataFrame:
        """Return one scenario per column, indexed by tenor."""

        return self.curve.pivot(index="tenor", columns="scenario", values="price")


def build_anchor_matrix(
    curve: DataFrame[CurveInput],
    delivery: pd.DataFrame,
    prices: pd.Series | pd.DataFrame,
    *,
    quote_index_factor: pd.Series | float = 1.0,
    lower: pd.Series | pd.DataFrame | float | None = None,
    upper: pd.Series | pd.DataFrame | float | None = None,
    weight: pd.Series | pd.DataFrame | float | None = None,
) -> AnchorMatrix:
    """Convert prepared delivery profiles into raw-price exposures.

    Product lookup and anchor estimation happen before this function.  This
    adapter only aligns the supplied profiles and puts monthly and quoted
    prices on the same cash-flow basis.
    """

    monthly = _prepare_curve(curve)
    tenors = pd.Index(monthly["tenor"], name="tenor")
    aligned_delivery = delivery.reindex(columns=tenors)
    quote_factor = (
        quote_index_factor.reindex(aligned_delivery.index).to_numpy(dtype=float)
        if isinstance(quote_index_factor, pd.Series)
        else quote_index_factor
    )
    exposure = cashflow_matrix(
        aligned_delivery.to_numpy(dtype=float),
        monthly_weight=(
            monthly["energy_weight"] * monthly["discount_factor"]
        ).to_numpy(),
        monthly_price_factor=monthly["index_factor"].to_numpy(),
        quote_price_factor=quote_factor,
    )
    return AnchorMatrix(
        exposure=pd.DataFrame(
            exposure,
            index=aligned_delivery.index,
            columns=tenors,
        ),
        prices=prices,
        lower=lower,
        upper=upper,
        weight=weight,
    )


def build_raw_and_indexed_anchor_matrix(
    curve: DataFrame[CurveInput],
    delivery: pd.DataFrame,
    raw_prices: pd.Series | pd.DataFrame,
    indexed_prices: pd.Series | pd.DataFrame,
) -> AnchorMatrix:
    """Build exact raw-price and indexed-price restrictions together.

    ``raw_prices`` are matched against delivery-weighted monthly prices with
    no index correction. ``indexed_prices`` are matched against the same
    monthly unknowns after applying the curve's ``index_factor``. Prefixing
    the row labels keeps both observations independently auditable.
    """

    raw = build_anchor_matrix(
        curve.assign(index_factor=1.0),
        delivery,
        raw_prices,
    )
    indexed = build_anchor_matrix(
        curve,
        delivery,
        indexed_prices,
    )
    return AnchorMatrix.exact(
        exposure=pd.concat(
            [
                _with_basis(raw.exposure, "raw"),
                _with_basis(indexed.exposure, "indexed"),
            ]
        ),
        prices=pd.concat(
            [
                _with_basis(raw.prices, "raw"),
                _with_basis(indexed.prices, "indexed"),
            ]
        ),
    )


def neutralize_curve(
    curve: DataFrame[CurveInput],
    anchors: AnchorMatrix,
    *,
    prior_strength: float = 1.0,
    smoothness: float = 0.0,
    tolerance: float = 1e-9,
) -> NeutralizationResult:
    """Neutralize a prepared monthly curve against an anchor matrix."""

    monthly = _prepare_curve(curve)
    prices = _price_frame(anchors.prices)
    tenors = pd.Index(monthly["tenor"], name="tenor")
    exposure = anchors.exposure.reindex(index=prices.index, columns=tenors)
    scenarios = pd.Index(prices.columns.astype(str), name="scenario")
    prices = prices.set_axis(scenarios, axis="columns")

    economic_weight = (
        monthly["energy_weight"] * monthly["discount_factor"] * monthly["index_factor"]
    ).to_numpy()
    basis, _ = block_basis(
        monthly["block"].to_numpy(),
        shape=monthly["seasonal_shape"].to_numpy(),
        economic_weight=economic_weight,
    )
    solution = solve_curve(
        monthly["price"].to_numpy(),
        LinearObservations(
            matrix=exposure.to_numpy(dtype=float),
            values=prices.to_numpy(dtype=float),
            lower=_aligned(anchors.lower, prices.index, scenarios),
            upper=_aligned(anchors.upper, prices.index, scenarios),
            weight=_aligned(anchors.weight, prices.index, scenarios),
        ),
        basis=basis,
        floor=monthly["floor"].to_numpy(),
        cap=monthly["cap"].to_numpy(),
        prior_strength=prior_strength,
        prior_weight=economic_weight / economic_weight.mean(),
        smoothness=smoothness,
        tolerance=tolerance,
    )

    balanced = pd.DataFrame(
        solution.prices,
        index=tenors,
        columns=scenarios,
    )
    fitted = pd.DataFrame(
        solution.fitted,
        index=prices.index,
        columns=scenarios,
    )
    residuals = pd.DataFrame(
        solution.residuals,
        index=prices.index,
        columns=scenarios,
    )
    curve_output = (
        _long(balanced, "price", "tenor")
        .merge(
            monthly[["tenor", "price"]].rename(columns={"price": "initial_price"}),
            on="tenor",
            how="left",
            validate="many_to_one",
        )[["tenor", "scenario", "initial_price", "price"]]
        .pipe(CurveOutput.validate, lazy=True)
    )
    anchor_output = (
        pd.concat(
            [
                _long(prices, "target", "anchor").set_index(["anchor", "scenario"]),
                _long(fitted, "fitted", "anchor").set_index(["anchor", "scenario"]),
                _long(residuals, "residual", "anchor").set_index(
                    ["anchor", "scenario"]
                ),
            ],
            axis="columns",
        )
        .reset_index()
        .assign(anchor=lambda data: data["anchor"].astype(str))
        .pipe(AnchorOutput.validate, lazy=True)
    )
    scenario_output = pd.DataFrame(
        {
            "scenario": scenarios,
            "objective": np.atleast_1d(solution.objective),
            "max_violation": np.atleast_1d(solution.max_violation),
        }
    ).pipe(ScenarioOutput.validate, lazy=True)
    return NeutralizationResult(
        curve=cast(DataFrame[CurveOutput], curve_output),
        anchors=cast(DataFrame[AnchorOutput], anchor_output),
        scenarios=cast(DataFrame[ScenarioOutput], scenario_output),
    )


def _prepare_curve(frame: pd.DataFrame) -> DataFrame[CurveInput]:
    tenor = pd.to_datetime(frame["tenor"]).dt.to_period("M").dt.to_timestamp()
    hours = tenor.dt.days_in_month.astype(float) * 24.0
    identity = tenor.dt.strftime("%Y-%m")
    ones = pd.Series(1.0, index=frame.index)
    factor = frame.get("index_factor", ones).fillna(1.0)
    base = frame.get("ipca_base", ones).fillna(1.0)
    prepared = (
        frame.assign(
            tenor=tenor,
            # ``index_factor`` is the input. The levels are carried for reporting
            # and kept consistent, never used to recompute the factor.
            index_factor=factor,
            ipca_base=base,
            ipca_forecast=factor * base,
        )
        .assign(
            energy_weight=lambda data: data.get("energy_weight", hours).fillna(hours),
            discount_factor=lambda data: data.get(
                "discount_factor", pd.Series(1.0, index=data.index)
            ).fillna(1.0),
            index_base=lambda data: pd.to_datetime(
                data.get("index_base", pd.Series(tenor.min(), index=data.index))
            ).fillna(tenor.min()),
            index_start=lambda data: pd.to_datetime(
                data.get("index_start", tenor)
            ).fillna(tenor),
            floor=lambda data: data.get(
                "floor", pd.Series(-np.inf, index=data.index)
            ).fillna(-np.inf),
            cap=lambda data: data.get(
                "cap", pd.Series(np.inf, index=data.index)
            ).fillna(np.inf),
            block=lambda data: data.get("block", identity).fillna(identity).astype(str),
            seasonal_shape=lambda data: data.get(
                "seasonal_shape", pd.Series(1.0, index=data.index)
            ).fillna(1.0),
        )
        .sort_values("tenor", ignore_index=True)
        .pipe(CurveInput.validate, lazy=True)
    )
    return cast(DataFrame[CurveInput], prepared)


def _price_frame(prices: pd.Series | pd.DataFrame) -> pd.DataFrame:
    frame = (
        prices.rename(prices.name or "base").to_frame()
        if isinstance(prices, pd.Series)
        else prices.copy()
    ).rename(columns=str)
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Anchor and scenario labels must be unique")
    return frame.apply(pd.to_numeric, errors="raise")


def _with_basis(
    values: pd.Series | pd.DataFrame,
    basis: str,
) -> pd.Series | pd.DataFrame:
    labelled = values.copy()
    labelled.index = pd.Index(
        [f"{basis}:{label}" for label in labelled.index],
        name="anchor",
    )
    return labelled


def _aligned(
    values: pd.Series | pd.DataFrame | float | None,
    index: pd.Index,
    columns: pd.Index,
) -> np.ndarray | float | None:
    if values is None or np.isscalar(values):
        return values
    if isinstance(values, pd.Series):
        return values.reindex(index).to_numpy(dtype=float)
    return (
        values.rename(columns=str)
        .reindex(index=index, columns=columns)
        .to_numpy(dtype=float)
    )


def _long(frame: pd.DataFrame, value: str, index_name: str) -> pd.DataFrame:
    return (
        frame.rename_axis(index=index_name, columns="scenario")
        .stack(future_stack=True)
        .rename(value)
        .reset_index()
        .assign(scenario=lambda data: data["scenario"].astype(str))
    )
