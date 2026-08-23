"""Build a coherent, non-redundant anchor plan from product-level marks."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import pandera.pandas as pa
from numpy.typing import ArrayLike
from pandera.typing.pandas import DataFrame, Series
from pricer.curves.arbitrage import (
    InfeasibleCurveError,
    LinearObservations,
    block_basis,
    solve_curve,
)

from .neutralization import (
    AnchorMatrix,
    CurveInput,
    CurveOutput,
    NeutralizationResult,
    _prepare_curve,
    _price_frame,
    build_anchor_matrix,
    build_raw_and_indexed_anchor_matrix,
    neutralize_curve,
)


class AnchorDecisionOutput(pa.DataFrameModel):
    """One deterministic rank-selection decision per candidate product."""

    product_id: Series[str]
    mandatory: Series[bool]
    quality: Series[float] = pa.Field(gt=0)
    selected: Series[bool]
    reason: Series[str]
    rank_before: Series[int] = pa.Field(ge=0)
    rank_after: Series[int] = pa.Field(ge=0)

    @pa.dataframe_check
    def unique_products(cls, frame: pd.DataFrame) -> bool:
        return bool(frame["product_id"].is_unique)

    class Config:
        coerce = True
        strict = True


class AnchorDiagnosticOutput(pa.DataFrameModel):
    """Market, reconciled and implied prices for every candidate product."""

    product_id: Series[str]
    scenario: Series[str]
    selected: Series[bool]
    market_price: Series[float]
    anchor_price: Series[float]
    implied_price: Series[float]
    adjustment: Series[float]
    market_gap: Series[float]

    class Config:
        coerce = True
        strict = True


@dataclass(frozen=True, slots=True)
class AnchorSelection:
    """Exact independent anchors plus auditable selection diagnostics."""

    anchors: AnchorMatrix
    decisions: DataFrame[AnchorDecisionOutput]
    diagnostics: DataFrame[AnchorDiagnosticOutput]


@dataclass(frozen=True, slots=True)
class AnchorPlan:
    """Prepared local block curve and its product-level anchor selection."""

    curve: DataFrame[CurveInput]
    anchors: AnchorMatrix
    active_tenors: pd.DatetimeIndex
    decisions: DataFrame[AnchorDecisionOutput]
    diagnostics: DataFrame[AnchorDiagnosticOutput]
    cutoff: pd.Timestamp | None = None


@dataclass(frozen=True, slots=True)
class _AnchorProblem:
    curve: DataFrame[CurveInput]
    profiles: pd.DataFrame
    basis: np.ndarray
    active_tenors: pd.DatetimeIndex
    cutoff: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class _DualProblem:
    anchor: _AnchorProblem
    candidates: AnchorMatrix
    quality: pd.Series


def atomic_block_labels(
    delivery: pd.DataFrame,
    tenors: Collection[object] | pd.Index,
) -> pd.Series:
    """Return contiguous non-overlapping strips induced by product boundaries.

    Months outside every supplied delivery profile remain independent. Calendar
    years are always separated, so a generic block can never leak into another
    year of the curve.
    """

    month_index = _month_index(tenors)
    profiles = _align_delivery(delivery, month_index)
    active = profiles.gt(0.0).T
    covered = active.any(axis="columns")
    serial = pd.Series(
        month_index.year * 12 + month_index.month,
        index=month_index,
    )
    boundary = (
        active.ne(active.shift()).any(axis="columns")
        | serial.diff().ne(1)
        | pd.Series(month_index.year, index=month_index).diff().ne(0)
        | ~covered
    )
    groups = boundary.cumsum()
    ranges = pd.DataFrame(
        {"tenor": month_index, "covered": covered.to_numpy(), "group": groups}
    ).assign(
        start=lambda data: data.groupby("group")["tenor"].transform("min"),
        end=lambda data: data.groupby("group")["tenor"].transform("max"),
    )
    labels = (
        ranges["start"]
        .dt.strftime("%Y-%m")
        .str.cat(
            ranges["end"].dt.strftime("%Y-%m"),
            sep=":",
        )
    )
    return pd.Series(
        np.where(
            ranges["covered"],
            "ATOM|" + labels,
            "MONTH|" + ranges["tenor"].dt.strftime("%Y-%m"),
        ),
        index=month_index,
        name="block",
        dtype="string",
    )


def select_anchor_basis(
    exposure: pd.DataFrame,
    prices: pd.Series | pd.DataFrame,
    *,
    basis: ArrayLike | None = None,
    quality: pd.Series | float = 1.0,
    mandatory: Collection[str] = (),
    reconcile: bool = False,
    rank_tolerance: float | None = None,
) -> AnchorSelection:
    """Select a maximal exact product basis ordered by economic reliability.

    Mandatory products are retained first even when redundant. Remaining rows
    are accepted only when they increase the rank of ``exposure @ basis``.
    When ``reconcile`` is true, all market marks first enter an inverse-error
    weighted least-squares reconciliation while mandatory prices remain exact.
    The resulting independent prices are then frozen as hard anchors.
    """

    market = _canonical_prices(prices)
    products = market.index
    economic = _canonical_exposure(exposure, products)
    loading = _loading_matrix(basis, economic.shape[1])
    reduced = economic.to_numpy(dtype=float) @ loading
    precision = _quality_series(quality, products)
    mandatory_set = set(map(str, mandatory))
    missing = mandatory_set.difference(products)
    if missing:
        raise KeyError(f"Mandatory products are not candidates: {sorted(missing)}")
    mandatory_mask = products.isin(mandatory_set)
    anchor_prices = (
        _reconcile_prices(reduced, market, precision, mandatory_mask)
        if reconcile
        else market.copy()
    )
    if not reconcile:
        _validate_mandatory_prices(
            reduced[mandatory_mask],
            anchor_prices.to_numpy(dtype=float)[mandatory_mask],
            rank_tolerance,
        )

    order = (
        pd.DataFrame(
            {
                "position": np.arange(len(products)),
                "product_id": products,
                "mandatory": mandatory_mask,
                "quality": precision.to_numpy(),
            }
        )
        .sort_values(
            ["mandatory", "quality", "position"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    selected_positions: list[int] = []
    decisions: list[dict[str, object]] = []
    current = np.empty((0, reduced.shape[1]))
    for candidate in order.itertuples(index=False):
        before = _rank(current, rank_tolerance)
        proposed = np.vstack([current, reduced[candidate.position]])
        after = _rank(proposed, rank_tolerance)
        selected = bool(candidate.mandatory or after > before)
        if selected:
            selected_positions.append(candidate.position)
            current = proposed
        decisions.append(
            {
                "position": candidate.position,
                "product_id": candidate.product_id,
                "mandatory": candidate.mandatory,
                "quality": candidate.quality,
                "selected": selected,
                "reason": (
                    "mandatory"
                    if candidate.mandatory
                    else "independent"
                    if selected
                    else "redundant"
                ),
                "rank_before": before,
                "rank_after": after,
            }
        )

    selected_products = products.take(selected_positions)
    selected_mask = products.isin(selected_products)
    decision_frame = (
        pd.DataFrame(decisions)
        .sort_values("position", ignore_index=True)
        .drop(columns="position")
        .pipe(AnchorDecisionOutput.validate, lazy=True)
    )
    diagnostic_frame = _anchor_diagnostics(
        reduced,
        market,
        anchor_prices,
        selected_mask,
    ).pipe(AnchorDiagnosticOutput.validate, lazy=True)
    return AnchorSelection(
        anchors=AnchorMatrix.exact(
            economic.loc[selected_products],
            anchor_prices.loc[selected_products],
        ),
        decisions=cast(DataFrame[AnchorDecisionOutput], decision_frame),
        diagnostics=cast(DataFrame[AnchorDiagnosticOutput], diagnostic_frame),
    )


def build_anchor_plan(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
    prices: pd.Series | pd.DataFrame,
    *,
    quality: pd.Series | float = 1.0,
    mandatory: Collection[str] = (),
    quote_index_factor: pd.Series | float = 1.0,
    cutoff: object | None = None,
    reconcile: bool = False,
    rank_tolerance: float | None = None,
) -> AnchorPlan:
    """Build local atomic strips and an exact non-redundant anchor matrix."""

    market = _canonical_prices(prices)
    problem = _prepare_anchor_problem(curve, delivery, market.index, cutoff)
    candidates = build_anchor_matrix(
        problem.curve,
        problem.profiles,
        market,
        quote_index_factor=quote_index_factor,
    )
    selection = select_anchor_basis(
        candidates.exposure,
        candidates.prices,
        basis=problem.basis,
        quality=quality,
        mandatory=mandatory,
        reconcile=reconcile,
        rank_tolerance=rank_tolerance,
    )
    return _anchor_plan(problem, selection)


def build_dual_anchor_plan(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
    raw_prices: pd.Series | pd.DataFrame,
    indexed_prices: pd.Series | pd.DataFrame,
    *,
    quality: pd.Series | float = 1.0,
    raw_weight: float = 1.0,
    indexed_weight: float = 1.0,
    cutoff: object | None = None,
) -> AnchorPlan:
    """Build one raw state fitted jointly to soft raw and indexed marks."""

    dual = _prepare_dual_problem(
        curve,
        delivery,
        raw_prices,
        indexed_prices,
        quality,
        cutoff,
    )
    weights = _dual_quality(dual.quality, raw_weight, indexed_weight)
    selection = _soft_anchor_selection(
        dual.candidates,
        dual.anchor.basis,
        weights,
    )
    return _anchor_plan(dual.anchor, selection)


def build_exact_dual_anchor_plan(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
    raw_prices: pd.Series | pd.DataFrame,
    indexed_prices: pd.Series | pd.DataFrame,
    *,
    quality: pd.Series | float = 1.0,
    mandatory: Collection[str] = (),
    cutoff: object | None = None,
    reconcile: bool = False,
    rank_tolerance: float | None = None,
) -> AnchorPlan:
    """Build the optional audit mode with exact raw and indexed equalities."""

    dual = _prepare_dual_problem(
        curve,
        delivery,
        raw_prices,
        indexed_prices,
        quality,
        cutoff,
    )
    dual_quality = _dual_quality(dual.quality, 1.0, 1.0)
    dual_mandatory = [
        f"{surface}:{product}"
        for surface in ("raw", "indexed")
        for product in map(str, mandatory)
    ]
    selection = select_anchor_basis(
        dual.candidates.exposure,
        dual.candidates.prices,
        basis=dual.anchor.basis,
        quality=dual_quality,
        mandatory=dual_mandatory,
        reconcile=reconcile,
        rank_tolerance=rank_tolerance,
    )
    return _anchor_plan(dual.anchor, selection)


def _prepare_dual_problem(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
    raw_prices: pd.Series | pd.DataFrame,
    indexed_prices: pd.Series | pd.DataFrame,
    quality: pd.Series | float,
    cutoff: object | None,
) -> _DualProblem:
    raw = _canonical_prices(raw_prices)
    indexed = _canonical_prices(indexed_prices).reindex(
        index=raw.index,
        columns=raw.columns,
    )
    if indexed.isna().any(axis=None):
        raise ValueError("Raw and indexed prices must have identical labels")
    anchor = _prepare_anchor_problem(curve, delivery, raw.index, cutoff)
    return _DualProblem(
        anchor=anchor,
        candidates=build_raw_and_indexed_anchor_matrix(
            anchor.curve,
            anchor.profiles,
            raw,
            indexed,
        ),
        quality=_quality_series(quality, raw.index),
    )


def _dual_quality(
    quality: pd.Series,
    raw_weight: float,
    indexed_weight: float,
) -> pd.Series:
    weights = pd.concat(
        [
            (quality * raw_weight).rename(index=lambda label: f"raw:{label}"),
            (quality * indexed_weight).rename(index=lambda label: f"indexed:{label}"),
        ]
    )
    return _quality_series(weights, weights.index)


def _soft_anchor_selection(
    candidates: AnchorMatrix,
    basis: np.ndarray,
    quality: pd.Series,
) -> AnchorSelection:
    market = _canonical_prices(candidates.prices)
    economic = _canonical_exposure(candidates.exposure, market.index)
    reduced = economic.to_numpy(dtype=float) @ basis
    ranks = np.asarray(
        [_rank(reduced[:position], None) for position in range(len(reduced) + 1)]
    )
    decisions = pd.DataFrame(
        {
            "product_id": market.index,
            "mandatory": False,
            "quality": quality.reindex(market.index).to_numpy(),
            "selected": True,
            "reason": "soft",
            "rank_before": ranks[:-1],
            "rank_after": ranks[1:],
        }
    ).pipe(AnchorDecisionOutput.validate, lazy=True)
    diagnostics = _anchor_diagnostics(
        reduced,
        market,
        market,
        np.ones(len(market), dtype=bool),
    ).pipe(AnchorDiagnosticOutput.validate, lazy=True)
    return AnchorSelection(
        anchors=AnchorMatrix.soft(
            economic,
            market,
            weight=quality,
        ),
        decisions=cast(DataFrame[AnchorDecisionOutput], decisions),
        diagnostics=cast(DataFrame[AnchorDiagnosticOutput], diagnostics),
    )


def _prepare_anchor_problem(
    curve: pd.DataFrame,
    delivery: pd.DataFrame,
    products: pd.Index,
    cutoff: object | None,
) -> _AnchorProblem:
    cutoff_month = _cutoff_month(cutoff)
    monthly = _prepare_curve(curve)
    if cutoff_month is not None:
        monthly = cast(
            DataFrame[CurveInput],
            monthly.query(
                "tenor <= @cutoff_month",
                local_dict={"cutoff_month": cutoff_month},
            ),
        )
    tenors = pd.DatetimeIndex(monthly["tenor"], name="tenor")
    profiles = _align_delivery(delivery, tenors, products=products)
    blocks = atomic_block_labels(profiles, tenors)
    planned_curve = cast(
        DataFrame[CurveInput],
        monthly.assign(block=monthly["tenor"].map(blocks).astype("string")),
    )
    economic_weight = (
        planned_curve["energy_weight"]
        * planned_curve["discount_factor"]
        * planned_curve["index_factor"]
    ).to_numpy()
    basis, _ = block_basis(
        planned_curve["block"].to_numpy(),
        shape=planned_curve["seasonal_shape"].to_numpy(),
        economic_weight=economic_weight,
    )
    return _AnchorProblem(
        curve=planned_curve,
        profiles=profiles,
        basis=basis,
        active_tenors=tenors[profiles.gt(0.0).any(axis="index")],
        cutoff=cutoff_month,
    )


def _anchor_plan(
    problem: _AnchorProblem,
    selection: AnchorSelection,
) -> AnchorPlan:
    return AnchorPlan(
        curve=problem.curve,
        anchors=selection.anchors,
        active_tenors=problem.active_tenors,
        decisions=selection.decisions,
        diagnostics=selection.diagnostics,
        cutoff=problem.cutoff,
    )


def _cutoff_month(value: object | None) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value).to_period("M").to_timestamp()


def neutralize_anchor_plan(
    plan: AnchorPlan,
    *,
    prior_strength: float = 1.0,
    smoothness: float = 0.0,
    tolerance: float = 1e-9,
) -> NeutralizationResult:
    """Neutralize only the plan's delivery scope and preserve every other month."""

    active = plan.active_tenors
    scoped_curve = plan.curve.query(
        "tenor in @active",
        local_dict={"active": active},
    )
    scoped_anchors = AnchorMatrix(
        exposure=plan.anchors.exposure.reindex(columns=active),
        prices=plan.anchors.prices,
        lower=plan.anchors.lower,
        upper=plan.anchors.upper,
        weight=plan.anchors.weight,
    )
    scoped = neutralize_curve(
        scoped_curve,
        scoped_anchors,
        prior_strength=prior_strength,
        smoothness=smoothness,
        tolerance=tolerance,
    )
    scenarios = scoped.scenarios["scenario"].drop_duplicates()
    untouched = (
        plan.curve.query(
            "tenor not in @active",
            local_dict={"active": active},
        )[["tenor", "price"]]
        .merge(pd.DataFrame({"scenario": scenarios}), how="cross")
        .rename(columns={"price": "initial_price"})
        .assign(price=lambda data: data["initial_price"])
    )
    curve = (
        pd.concat([scoped.curve, untouched], ignore_index=True)
        .sort_values(["scenario", "tenor"], ignore_index=True)
        .pipe(CurveOutput.validate, lazy=True)
    )
    return NeutralizationResult(
        curve=cast(DataFrame[CurveOutput], curve),
        anchors=scoped.anchors,
        scenarios=scoped.scenarios,
    )


def _month_index(values: Collection[object] | pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values))).to_period("M").to_timestamp()


def _align_delivery(
    delivery: pd.DataFrame,
    tenors: pd.DatetimeIndex,
    *,
    products: pd.Index | None = None,
) -> pd.DataFrame:
    profiles = delivery.copy()
    profiles.index = pd.Index(profiles.index.map(str), name="product_id")
    profiles.columns = _month_index(profiles.columns)
    if not profiles.index.is_unique or not profiles.columns.is_unique:
        raise ValueError("Product and delivery-month labels must be unique")
    if products is not None:
        profiles = profiles.reindex(products)
    aligned = profiles.reindex(columns=tenors, fill_value=0.0).apply(
        pd.to_numeric,
        errors="raise",
    )
    values = aligned.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or (len(aligned) and np.any(values.sum(axis=1) <= 0.0))
    ):
        raise ValueError("Every candidate must deliver positive energy in the curve")
    return aligned


def _canonical_prices(prices: pd.Series | pd.DataFrame) -> pd.DataFrame:
    frame = _price_frame(prices)
    frame.index = pd.Index(frame.index.map(str), name="product_id")
    if not frame.index.is_unique:
        raise ValueError("Product labels must remain unique as strings")
    return frame


def _canonical_exposure(exposure: pd.DataFrame, products: pd.Index) -> pd.DataFrame:
    frame = exposure.copy()
    frame.index = pd.Index(frame.index.map(str), name="product_id")
    aligned = frame.reindex(products).apply(pd.to_numeric, errors="raise")
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Exposure must be finite and aligned with every product")
    return aligned


def _loading_matrix(basis: ArrayLike | None, months: int) -> np.ndarray:
    loading = np.eye(months) if basis is None else np.asarray(basis, dtype=float)
    if loading.ndim != 2 or loading.shape[0] != months:
        raise ValueError("basis must have one row per exposure month")
    return loading


def _quality_series(
    quality: pd.Series | float,
    products: pd.Index,
) -> pd.Series:
    values = (
        quality.rename(index=str).reindex(products)
        if isinstance(quality, pd.Series)
        else pd.Series(float(quality), index=products)
    ).astype(float)
    if not np.isfinite(values).all() or values.le(0.0).any():
        raise ValueError("Every candidate quality must be finite and positive")
    return values.rename("quality")


def _reconcile_prices(
    matrix: np.ndarray,
    market: pd.DataFrame,
    quality: pd.Series,
    mandatory: np.ndarray,
) -> pd.DataFrame:
    targets = market.to_numpy(dtype=float)
    bounds = np.where(mandatory[:, None], targets, np.nan)
    solution = solve_curve(
        np.zeros(matrix.shape[1]),
        LinearObservations(
            matrix=matrix,
            values=targets,
            lower=np.where(np.isnan(bounds), -np.inf, bounds),
            upper=np.where(np.isnan(bounds), np.inf, bounds),
            weight=np.broadcast_to(quality.to_numpy()[:, None], targets.shape),
        ),
        prior_strength=0.0,
    )
    return pd.DataFrame(solution.fitted, index=market.index, columns=market.columns)


def _validate_mandatory_prices(
    matrix: np.ndarray,
    prices: np.ndarray,
    tolerance: float | None,
) -> None:
    if not len(matrix):
        return
    fitted = matrix @ np.linalg.lstsq(matrix, prices, rcond=None)[0]
    scale = np.maximum(1.0, np.abs(prices).max(axis=0))
    threshold = (1e-9 if tolerance is None else tolerance) * scale
    if np.any(np.abs(fitted - prices).max(axis=0) > threshold):
        raise InfeasibleCurveError("Mandatory anchor prices are inconsistent")


def _rank(matrix: np.ndarray, tolerance: float | None) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=tolerance))


def _anchor_diagnostics(
    matrix: np.ndarray,
    market: pd.DataFrame,
    anchors: pd.DataFrame,
    selected: np.ndarray,
) -> pd.DataFrame:
    selected_matrix = matrix[selected]
    coefficients = np.linalg.lstsq(
        selected_matrix.T,
        matrix.T,
        rcond=None,
    )[0].T
    implied = pd.DataFrame(
        coefficients @ anchors.to_numpy(dtype=float)[selected],
        index=market.index,
        columns=market.columns,
    )
    selection = pd.Series(selected, index=market.index, name="selected")
    indexed = ["product_id", "scenario"]
    frames = [
        _long_prices(market, "market_price").set_index(indexed),
        _long_prices(anchors, "anchor_price").set_index(indexed),
        _long_prices(implied, "implied_price").set_index(indexed),
    ]
    return (
        pd.concat(frames, axis="columns")
        .reset_index()
        .assign(
            selected=lambda data: data["product_id"].map(selection),
            adjustment=lambda data: data["anchor_price"] - data["market_price"],
            market_gap=lambda data: data["implied_price"] - data["market_price"],
        )[
            [
                "product_id",
                "scenario",
                "selected",
                "market_price",
                "anchor_price",
                "implied_price",
                "adjustment",
                "market_gap",
            ]
        ]
    )


def _long_prices(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        frame.rename_axis(index="product_id", columns="scenario")
        .stack(future_stack=True)
        .rename(value)
        .reset_index()
        .assign(
            product_id=lambda data: data["product_id"].astype(str),
            scenario=lambda data: data["scenario"].astype(str),
        )
    )
