"""Forward-curve construction: liquidity policy, anchors, solve and DCIDE splice."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series
from pricer.curves.arbitrage import block_basis

from .neutralization import (
    AnchorMatrix,
    CurveInput,
    CurveOutput,
    NeutralizationResult,
    _prepare_curve,
    _price_frame,
    build_raw_and_indexed_anchor_matrix,
    neutralize_curve,
)

Priority = tuple[str, ...]

DEFAULT_PRIORITY: Priority = ("MEN", "TRI", "SEM", "ANU")
ALL_MONTHS = frozenset(range(1, 13))
GRANULARITY_MONTHS: dict[str, int] = {"MEN": 1, "TRI": 3, "SEM": 6, "ANU": 12}


@dataclass(frozen=True, slots=True)
class GranularityRule:
    """Restricted priority applied to a set of months inside one year."""

    year: int
    months: frozenset[int]
    priority: Priority

    def applies_to(self, month: pd.Timestamp) -> bool:
        return month.year == self.year and month.month in self.months


@dataclass(frozen=True, slots=True)
class CurveGranularity:
    """Injectable product-selection policy.

    ``default`` is the liquidity order of the market. A rule replaces it inside
    its months, and a one-element priority such as ``("ANU",)`` therefore has no
    fallback: that is what makes "use CAL27 for all of 2027" deterministic.
    """

    rules: tuple[GranularityRule, ...] = ()
    default: Priority = DEFAULT_PRIORITY

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> CurveGranularity:
        """Build the policy from plain config, as loaded from JSON or YAML.

        ``months`` is optional and defaults to the whole year. Years arrive as
        ``int`` or ``str`` so a JSON object can be used directly.

        ```python
        CurveGranularity.from_dict(
            {
                "default": ["MEN", "TRI", "SEM", "ANU"],
                "rules": [
                    {"year": 2027, "priority": ["ANU"]},
                    {"year": 2028, "months": [1, 2, 3], "priority": ["TRI", "ANU"]},
                ],
            }
        )
        ```
        """

        rules = cast(list[Mapping[str, object]], config.get("rules", []))
        unknown = {str(key) for key in config} - {"default", "rules"}
        if unknown:
            raise KeyError(f"Unknown granularity settings: {sorted(unknown)}")
        return cls(
            rules=tuple(
                GranularityRule(
                    year=int(cast(int, rule["year"])),
                    months=frozenset(
                        int(month)
                        for month in cast(
                            "list[int]", rule.get("months", sorted(ALL_MONTHS))
                        )
                    ),
                    priority=_priority(rule["priority"]),
                )
                for rule in rules
            ),
            default=_priority(config.get("default", DEFAULT_PRIORITY)),
        )

    def for_year(self, year: int, priority: Priority) -> CurveGranularity:
        return self.for_months(year, ALL_MONTHS, priority)

    def for_months(
        self,
        year: int,
        months: frozenset[int] | range | tuple[int, ...],
        priority: Priority,
    ) -> CurveGranularity:
        rule = GranularityRule(year, frozenset(months), priority)
        return replace(self, rules=(*self.rules, rule))

    def priority_for(self, month: pd.Timestamp) -> Priority:
        return next(
            (rule.priority for rule in reversed(self.rules) if rule.applies_to(month)),
            self.default,
        )

    @pa.check_types(lazy=True)
    def select(
        self,
        quotes: DataFrame[ProductQuoteInput],
        tenors: pd.DatetimeIndex,
    ) -> DataFrame[SelectedTenorPriceOutput]:
        """Choose the freshest product at the best granularity allowed per month."""

        allowed = self._allowed_frame(_months(tenors).drop_duplicates().sort_values())
        selected = (
            allowed.merge(quotes, on="granularity", how="inner")
            .query("start <= tenor <= end")
            .sort_values(
                ["tenor", "rank", "traded_at", "product_id"],
                ascending=[True, True, False, True],
            )
            .drop_duplicates("tenor", keep="first")
            .sort_values("tenor", ignore_index=True)
        )
        return cast(
            DataFrame[SelectedTenorPriceOutput],
            selected[list(SelectedTenorPriceOutput.to_schema().columns)],
        )

    def _allowed_frame(self, tenors: pd.Series) -> pd.DataFrame:
        priorities = {tenor: self.priority_for(pd.Timestamp(tenor)) for tenor in tenors}
        return pd.DataFrame(
            [
                {
                    "tenor": tenor,
                    "granularity": granularity,
                    "rank": rank,
                    "overridden": priority != self.default,
                }
                for tenor, priority in priorities.items()
                for rank, granularity in enumerate(priority)
            ]
        )


DEFAULT_CURVE_GRANULARITY = CurveGranularity()


class ProductDeliveryInput(pa.DataFrameModel):
    """Normalized product definition extracted from negotiable-ticker features."""

    product_id: Series[str] = pa.Field(unique=True)
    description: Series[str]
    start: Series[pa.DateTime]
    end: Series[pa.DateTime]
    delivery_months: Series[int] = pa.Field(isin=list(GRANULARITY_MONTHS.values()))
    granularity: Series[str] = pa.Field(isin=list(GRANULARITY_MONTHS))

    @pa.dataframe_check
    def valid_delivery(cls, frame: pd.DataFrame) -> bool:
        return bool((frame["start"] <= frame["end"]).all())

    class Config:
        coerce = True
        strict = False


class ProductQuoteInput(ProductDeliveryInput):
    """One priced product: its delivery definition plus its freshest mark."""

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
    delivery_months: Series[int] = pa.Field(isin=list(GRANULARITY_MONTHS.values()))
    granularity: Series[str] = pa.Field(isin=list(GRANULARITY_MONTHS))
    price: Series[float] = pa.Field(gt=0)
    traded_at: Series[pa.DateTime]
    overridden: Series[bool]

    class Config:
        coerce = True
        strict = True


def normalize_products(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize delivery months and derive the four supported regimes."""

    start = _months(frame["start"])
    end = _months(frame["end"])
    delivery_months = (
        (end.dt.year - start.dt.year) * 12 + end.dt.month - start.dt.month + 1
    )
    return frame.assign(
        product_id=lambda data: data["product_id"].astype("string"),
        description=lambda data: data["description"].astype("string"),
        start=start,
        end=end,
        delivery_months=delivery_months,
        granularity=delivery_months.map(
            {months: name for name, months in GRANULARITY_MONTHS.items()}
        ),
    ).sort_values("product_id", ignore_index=True)


def delivery_profiles(
    products: pd.DataFrame,
    product_ids: pd.Index,
    tenors: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create product-by-month delivery indicators without row-wise Python."""

    selected = products.drop_duplicates("product_id").set_index("product_id")
    selected = selected.reindex(product_ids)
    months = _months(tenors)
    values = (months.to_numpy()[None, :] >= selected["start"].to_numpy()[:, None]) & (
        months.to_numpy()[None, :] <= selected["end"].to_numpy()[:, None]
    )
    return pd.DataFrame(
        values.astype(float),
        index=pd.Index(product_ids, name="product_id"),
        columns=pd.DatetimeIndex(months, name="tenor"),
    )


def annual_anchor_ids(quotes: pd.DataFrame, tenors: pd.DatetimeIndex) -> pd.Index:
    """Return the freshest annual product for each quoted delivery interval."""

    first, last = _months(tenors).min(), _months(tenors).max()
    return pd.Index(
        quotes.query(
            "granularity == 'ANU' and start <= @last and end >= @first",
            local_dict={"first": first, "last": last},
        )
        .sort_values(["start", "end", "traded_at", "product_id"])
        .drop_duplicates(["start", "end"], keep="last")["product_id"],
        name="product_id",
    )


def _months(values: object) -> pd.Series:
    return pd.Series(pd.to_datetime(values)).dt.to_period("M").dt.to_timestamp()


def _priority(value: object) -> Priority:
    priority = tuple(str(code) for code in cast("list[str]", value))
    unknown = set(priority) - set(GRANULARITY_MONTHS)
    if not priority or unknown:
        raise ValueError(
            f"Priority must be a non-empty subset of {list(GRANULARITY_MONTHS)}"
        )
    return priority


class UnbalancedCurveOutput(CurveInput):
    """Market curve after one product has been selected for each tenor."""

    product_id: Series[str]
    granularity: Series[str]
    traded_at: Series[pa.DateTime]
    overridden: Series[bool]

    class Config:
        coerce = True
        strict = False


class IpcaIndexInput(pa.DataFrameModel):
    """Forecast IPCA index by maturity: the only IPCA input the flow needs."""

    tenor: Series[pa.DateTime] = pa.Field(unique=True)
    indice: Series[float] = pa.Field(gt=0)

    class Config:
        coerce = True
        strict = False


@pa.check_types(lazy=True)
def apply_ipca(
    curve: pd.DataFrame,
    ipca: DataFrame[IpcaIndexInput],
    *,
    base_date: object,
) -> pd.DataFrame:
    """Set each month's IPCA factor from its adjustment date.

    Call this after ``build_unbalanced_curve``, which is what decides the
    adjustment dates: ``index_start`` holds each block's head, so a year split
    into Q1 plus a nine-month residual reindexes Q1 off January's forecast and
    the residual off April's. Months with no block fall back to their own tenor.

    ``ipca`` is ``tenor | indice`` — the forward index by maturity. No complete
    IPCA curve is assumed: only the adjustment dates and ``base_date`` are read,
    and a missing one is an error rather than a silent 1.0.
    """

    index = (
        ipca.assign(tenor=lambda data: _months(data["tenor"]))
        .set_index("tenor")["indice"]
        .astype(float)
    )
    base_month = _months(pd.Series([base_date])).iloc[0]
    if base_month not in index.index:
        raise KeyError(f"IPCA index has no value for the base date {base_month.date()}")
    prepared = _prepare_curve(curve)
    start = pd.to_datetime(prepared["index_start"]).dt.to_period("M").dt.to_timestamp()
    forecast = start.map(index)
    if forecast.isna().any():
        missing = sorted(start[forecast.isna()].dt.strftime("%Y-%m").unique())
        raise KeyError(f"IPCA index has no value for adjustment dates: {missing}")
    return prepared.assign(
        index_base=base_month,
        ipca_base=float(index[base_month]),
        ipca_forecast=forecast.to_numpy(),
        index_factor=forecast.to_numpy() / float(index[base_month]),
    )


def market_cutoff(reference_date: object) -> pd.Timestamp:
    """Liquid horizon: December 31 of the second year after the reference date."""

    return pd.Timestamp(
        year=pd.Timestamp(reference_date).year + 2,
        month=12,
        day=31,
    )


@pa.check_types(lazy=True)
def select_tenor_prices(
    quotes: DataFrame[ProductQuoteInput],
    curve: pd.DataFrame,
    *,
    cutoff: object,
    granularity: CurveGranularity = DEFAULT_CURVE_GRANULARITY,
) -> DataFrame[SelectedTenorPriceOutput]:
    """Price every curve tenor through the cutoff under the granularity policy."""

    end = _month(cutoff)
    tenors = pd.DatetimeIndex(
        _prepare_curve(curve).query("tenor <= @end", local_dict={"end": end})["tenor"]
    )
    return granularity.select(quotes, tenors)


@pa.check_types(lazy=True)
def build_unbalanced_curve(
    curve: pd.DataFrame,
    selected: DataFrame[SelectedTenorPriceOutput],
) -> DataFrame[UnbalancedCurveOutput]:
    """Apply selected prices and freeze IPCA at each contiguous block head.

    A contiguous run of the same product is one delivery block, so a Q1/residual
    split reindexes Q1 by January's factor and the residual by April's.
    ``index_start`` records the month each block's forecast was taken from, which
    is the IPCA adjustment date the desk quotes against, and ``index_factor``
    stays derived as ``ipca_forecast / ipca_base``.
    """

    market = (
        _prepare_curve(curve)
        .merge(
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
            index_start=lambda frame: frame.groupby("block_number", sort=False)[
                "tenor"
            ].transform("first"),
        )
        .drop(columns=["selected_price", "block_number"])
    )
    return cast(DataFrame[UnbalancedCurveOutput], market)


def build_market_plan(
    curve: pd.DataFrame,
    quotes: DataFrame[ProductQuoteInput],
    *,
    cutoff: object,
    ipca: DataFrame[IpcaIndexInput] | None = None,
    base_date: object | None = None,
    granularity: CurveGranularity = DEFAULT_CURVE_GRANULARITY,
    quality: pd.Series | float | None = None,
    raw_weight: float = 1.0,
    indexed_weight: float = 1.0,
) -> AnchorPlan:
    """Price each tenor, index by block and state the raw/indexed anchors.

    Selecting a product per tenor is what fixes the IPCA adjustment dates, so
    ``ipca`` is applied here, after the blocks exist and before the anchors are
    stated. Without it the curve's own ``index_factor`` is used as given.

    The plan covers only the liquid segment through ``cutoff``. It knows nothing
    about DCIDE: the illiquid tail is spliced afterwards by
    ``finalize_forward_curve``.
    """

    prepared = _prepare_curve(curve)
    selected = select_tenor_prices(
        quotes,
        prepared,
        cutoff=cutoff,
        granularity=granularity,
    )
    unbalanced = build_unbalanced_curve(prepared, selected)
    if ipca is not None:
        unbalanced = apply_ipca(unbalanced, ipca, base_date=base_date)
    tenors = pd.DatetimeIndex(unbalanced["tenor"])

    anchor_ids = (
        pd.Index(selected["product_id"].drop_duplicates(), name="product_id")
        .append(annual_anchor_ids(quotes, tenors))
        .drop_duplicates()
    )
    profiles = delivery_profiles(quotes, anchor_ids, tenors)
    marks = quotes.drop_duplicates("product_id").set_index("product_id")
    raw = marks["price"].reindex(anchor_ids).rename("base")
    # Read the factor off the indexed curve, not the input one: with ``ipca``
    # supplied the input carries no factor, which would silently make every
    # indexed anchor equal its raw anchor and collapse the dual system.
    quote_factor = (
        marks["start"]
        .reindex(anchor_ids)
        .map(unbalanced.set_index("tenor")["index_factor"])
    )
    if quote_factor.isna().any():
        missing = sorted(anchor_ids[quote_factor.isna()])
        raise KeyError(f"Anchors start outside the priced curve: {missing}")
    indexed = raw.mul(quote_factor)
    if quality is None:
        quality = (
            marks["precision"].reindex(anchor_ids) if "precision" in marks else 1.0
        )
    elif isinstance(quality, pd.Series):
        quality = quality.rename(index=str).reindex(anchor_ids)

    return build_dual_anchor_plan(
        unbalanced,
        profiles,
        raw,
        indexed,
        quality=quality,
        raw_weight=raw_weight,
        indexed_weight=indexed_weight,
        cutoff=_month(cutoff),
    )


def _month(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    local = stamp.tz_localize(None) if stamp.tzinfo else stamp
    return local.to_period("M").to_timestamp()


@dataclass(frozen=True, slots=True)
class AnchorPlan:
    """Prepared local block curve plus the anchors constraining it."""

    curve: pd.DataFrame
    anchors: AnchorMatrix
    active_tenors: pd.DatetimeIndex
    cutoff: pd.Timestamp | None = None


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
    ranges = pd.DataFrame(
        {
            "tenor": month_index,
            "covered": covered.to_numpy(),
            "group": boundary.cumsum(),
        }
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
    """Build one raw state fitted jointly to soft raw and indexed marks.

    Both surfaces constrain the same latent raw curve, so the solve looks for
    the single raw residual whose price and whose IPCA-corrected price are both
    as close as possible to their anchors.
    """

    raw = _canonical_prices(raw_prices)
    indexed = _canonical_prices(indexed_prices)
    if indexed.shape[1] != raw.shape[1]:
        raise ValueError("Raw and indexed prices must have the same scenarios")
    indexed = indexed.set_axis(raw.columns, axis="columns").reindex(index=raw.index)
    if indexed.isna().any(axis=None):
        raise ValueError("Raw and indexed prices must have identical product labels")

    cutoff_month = (
        None if cutoff is None else pd.Timestamp(cutoff).to_period("M").to_timestamp()
    )
    monthly = _prepare_curve(curve)
    if cutoff_month is not None:
        monthly = cast(
            pd.DataFrame,
            monthly.query(
                "tenor <= @cutoff_month",
                local_dict={"cutoff_month": cutoff_month},
            ),
        )
    tenors = pd.DatetimeIndex(monthly["tenor"], name="tenor")
    profiles = _align_delivery(delivery, tenors, products=raw.index)
    blocks = atomic_block_labels(profiles, tenors)
    planned = monthly.assign(block=monthly["tenor"].map(blocks).astype("string"))
    candidates = build_raw_and_indexed_anchor_matrix(
        cast(CurveInput, planned),
        profiles,
        raw,
        indexed,
    )
    precision = _quality_series(quality, raw.index)
    return AnchorPlan(
        curve=planned,
        anchors=AnchorMatrix.soft(
            candidates.exposure,
            _price_frame(candidates.prices),
            weight=pd.concat(
                [
                    (precision * raw_weight).rename(index=lambda tag: f"raw:{tag}"),
                    (precision * indexed_weight).rename(
                        index=lambda tag: f"indexed:{tag}"
                    ),
                ]
            ),
        ),
        active_tenors=tenors[profiles.gt(0.0).any(axis="index")],
        cutoff=cutoff_month,
    )


class CurveErrorOutput(pa.DataFrameModel):
    """Per-month standard error implied by the anchor precisions."""

    tenor: Series[pa.DateTime] = pa.Field(unique=True)
    price_std: Series[float] = pa.Field(ge=0)
    leverage: Series[float] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = True


def _information(plan: AnchorPlan) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return the basis, the Fisher information over latent blocks, and its tenors.

    ``information = (A B)' W (A B)`` is what the weighted least squares actually
    inverts. Singular means the anchors do not identify every block: with no
    prior the solver would then return an arbitrary feasible point.
    """

    active = plan.active_tenors
    scoped = plan.curve.query("tenor in @active", local_dict={"active": active})
    basis, names = block_basis(
        scoped["block"].to_numpy(),
        shape=scoped["seasonal_shape"].to_numpy(),
        economic_weight=(
            scoped["energy_weight"] * scoped["discount_factor"] * scoped["index_factor"]
        ).to_numpy(),
    )
    economic = plan.anchors.exposure.reindex(columns=active).to_numpy(dtype=float)
    weight = plan.anchors.weight
    precision = (
        weight.reindex(plan.anchors.exposure.index).to_numpy(dtype=float)
        if isinstance(weight, pd.Series)
        else np.full(len(economic), 1.0 if weight is None else float(weight))
    )
    reduced = economic @ basis
    information = reduced.T @ (precision[:, None] * reduced)
    if np.linalg.matrix_rank(information) < information.shape[0]:
        raise ValueError(
            f"Anchors do not identify every block: {sorted(map(str, names))}"
        )
    return basis, information, pd.DatetimeIndex(scoped["tenor"])


@pa.check_types(lazy=True)
def curve_standard_error(plan: AnchorPlan) -> DataFrame[CurveErrorOutput]:
    """Propagate anchor precision into a per-month error on the solved curve.

    ``cov(x) = information**-1`` and ``cov(p) = B cov(x) B'``. Calibrated only
    when ``weight`` is a true inverse variance, which ``estimate_anchor_prices``
    supplies as ``precision``; otherwise the numbers are relative. It also
    assumes no monthly bound binds, since an active bound truncates the
    distribution.

    ``leverage`` is the month's error divided by the smallest anchor error, the
    amplification a thin market pays: an illiquid stub left after liquid
    products have priced a share ``s`` of the year inherits the annual anchor's
    error scaled by roughly ``1 / (1 - s)``.
    """

    basis, information, tenors = _information(plan)
    covariance = basis @ np.linalg.inv(information) @ basis.T
    price_std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    weight = plan.anchors.weight
    anchor_std = (
        1.0 / np.sqrt(weight.to_numpy(dtype=float).max())
        if isinstance(weight, pd.Series)
        else 1.0
    )
    return cast(
        DataFrame[CurveErrorOutput],
        pd.DataFrame(
            {
                "tenor": tenors,
                "price_std": price_std,
                "leverage": price_std / anchor_std,
            }
        ),
    )


def neutralize_anchor_plan(
    plan: AnchorPlan,
    *,
    prior_strength: float = 0.0,
    smoothness: float = 0.0,
    tolerance: float = 1e-9,
) -> NeutralizationResult:
    """Neutralize only the plan's delivery scope and preserve every other month.

    With ``prior_strength == 0`` the anchors alone must identify every block, so
    that is checked up front: otherwise the solve would silently return one
    arbitrary point out of a whole feasible subspace.
    """

    if prior_strength == 0.0 and smoothness == 0.0:
        _information(plan)
    active = plan.active_tenors
    scoped = neutralize_curve(
        cast(
            CurveInput,
            plan.curve.query("tenor in @active", local_dict={"active": active}),
        ),
        AnchorMatrix(
            exposure=plan.anchors.exposure.reindex(columns=active),
            prices=plan.anchors.prices,
            lower=plan.anchors.lower,
            upper=plan.anchors.upper,
            weight=plan.anchors.weight,
        ),
        prior_strength=prior_strength,
        smoothness=smoothness,
        tolerance=tolerance,
    )
    untouched = (
        plan.curve.query("tenor not in @active", local_dict={"active": active})[
            ["tenor", "price"]
        ]
        .merge(
            pd.DataFrame({"scenario": scoped.scenarios["scenario"].drop_duplicates()}),
            how="cross",
        )
        .rename(columns={"price": "initial_price"})
        .assign(price=lambda data: data["initial_price"])
    )
    curve = (
        pd.concat([scoped.curve, untouched], ignore_index=True)
        .sort_values(["scenario", "tenor"], ignore_index=True)
        .pipe(CurveOutput.validate, lazy=True)
    )
    return NeutralizationResult(
        curve=curve,
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


def _quality_series(quality: pd.Series | float, products: pd.Index) -> pd.Series:
    values = (
        quality.rename(index=str).reindex(products)
        if isinstance(quality, pd.Series)
        else pd.Series(float(quality), index=products)
    ).astype(float)
    if not np.isfinite(values).all() or values.le(0.0).any():
        raise ValueError("Every candidate quality must be finite and positive")
    return values.rename("quality")


class ForwardCurveOutput(pa.DataFrameModel):
    """Raw and indexed monthly prices after the market/DCIDE splice."""

    tenor: Series[pa.DateTime]
    scenario: Series[str]
    origin: Series[str] = pa.Field(isin=["market", "dcide"])
    raw_price: Series[float]
    index_factor: Series[float] = pa.Field(gt=0)
    index_base: Series[pa.DateTime]
    index_start: Series[pa.DateTime]
    ipca_base: Series[float] = pa.Field(gt=0)
    ipca_forecast: Series[float] = pa.Field(gt=0)
    indexed_price: Series[float]

    @pa.dataframe_check
    def unique_scenario_tenors(cls, frame: pd.DataFrame) -> bool:
        return bool(not frame.duplicated(["scenario", "tenor"]).any())

    class Config:
        coerce = True
        strict = True


def finalize_forward_curve(
    plan: AnchorPlan,
    neutralization: NeutralizationResult,
    dcide_curve: pd.DataFrame,
) -> DataFrame[ForwardCurveOutput]:
    """Splice raw prices first, then index the complete curve in one operation.

    ``neutralization`` already solved the market block on its own. DCIDE only
    supplies months after ``plan.cutoff``, and the IPCA factor multiplies the
    concatenated raw curve exactly once.
    """

    cutoff = (
        plan.cutoff
        if plan.cutoff is not None
        else pd.Timestamp(plan.curve["tenor"].max())
    )
    market = (
        neutralization.curve.query("tenor <= @cutoff", local_dict={"cutoff": cutoff})[
            ["tenor", "scenario", "price"]
        ]
        .rename(columns={"price": "raw_price"})
        .merge(
            plan.curve[
                [
                    "tenor",
                    "index_factor",
                    "index_base",
                    "index_start",
                    "ipca_base",
                    "ipca_forecast",
                ]
            ],
            on="tenor",
            how="left",
            validate="many_to_one",
        )
        .assign(origin="market")
    )
    dcide = (
        _prepare_curve(dcide_curve)
        .query("tenor > @cutoff", local_dict={"cutoff": cutoff})[
            [
                "tenor",
                "price",
                "index_factor",
                "index_base",
                "index_start",
                "ipca_base",
                "ipca_forecast",
            ]
        ]
        .rename(columns={"price": "raw_price"})
        .merge(neutralization.scenarios[["scenario"]].drop_duplicates(), how="cross")
        .assign(origin="dcide")
    )
    output = (
        pd.concat([market, dcide], ignore_index=True)
        .assign(indexed_price=lambda data: data["raw_price"] * data["index_factor"])
        .sort_values(["scenario", "tenor"], ignore_index=True)
        .pipe(ForwardCurveOutput.validate, lazy=True)
    )
    return cast(
        DataFrame[ForwardCurveOutput],
        output[list(ForwardCurveOutput.to_schema().columns)],
    )


class SpreadInput(pa.DataFrameModel):
    """Additive basis over the reference curve, per month and per market."""

    tenor: Series[pa.DateTime] = pa.Field(alias="vencimento")
    spread: Series[float]
    submarket: Series[str] = pa.Field(alias="submercado")
    source: Series[str] = pa.Field(alias="tipo_energia")

    @pa.dataframe_check
    def unique_market_months(cls, frame: pd.DataFrame) -> bool:
        return bool(
            not frame.duplicated(["vencimento", "submercado", "tipo_energia"]).any()
        )

    class Config:
        coerce = True
        strict = True


class ExpandedCurveOutput(pa.DataFrameModel):
    """The delivered curve: one raw price per month, market and energy type.

    Shaped like the desk's own curve file. The price is raw; ``data_base_ipca``
    and ``data_inicio_ipca`` say how to index it, so no factor is carried.
    """

    tenor: Series[pa.DateTime] = pa.Field(alias="vencimento")
    source: Series[str] = pa.Field(alias="tipo_energia")
    submarket: Series[str] = pa.Field(alias="submercado")
    scenario: Series[str] = pa.Field(alias="cenario")
    price: Series[float] = pa.Field(alias="preco")
    index_base: Series[pa.DateTime] = pa.Field(alias="data_base_ipca")
    index_start: Series[pa.DateTime] = pa.Field(alias="data_inicio_ipca")
    ipca_base: Series[float] = pa.Field(alias="ipca_base", gt=0)
    ipca_forecast: Series[float] = pa.Field(alias="ipca_previsto", gt=0)

    @pa.dataframe_check
    def unique_market_scenario_tenors(cls, frame: pd.DataFrame) -> bool:
        return bool(
            not frame.duplicated(
                ["vencimento", "submercado", "tipo_energia", "cenario"]
            ).any()
        )

    class Config:
        coerce = True
        strict = True


@pa.check_types(lazy=True)
def expand_curve(
    curve: DataFrame[ForwardCurveOutput],
    spread: DataFrame[SpreadInput],
    *,
    base_submarket: str = "SE",
    base_source: str = "CON",
) -> DataFrame[ExpandedCurveOutput]:
    """Replicate the reference curve over every quoted market, adding its spread.

    ``curve`` is the SE conventional curve, market block and DCIDE tail already
    spliced. Every ``(submarket, source)`` pair present in ``spread`` becomes its
    own curve at ``reference + spread``; the reference pair itself is emitted
    with a zero spread. The spread is additive on the raw price, and the IPCA
    dates are carried unchanged, since a basis does not reset the indexation.

    A spread that does not cover every month of a non-reference market is an
    error rather than a silent zero, which would quietly publish the SE curve
    under another submarket's name.
    """

    basis = spread.rename(
        columns={
            "vencimento": "tenor",
            "submercado": "submarket",
            "tipo_energia": "source",
        }
    )
    markets = pd.concat(
        [
            pd.DataFrame({"submarket": [base_submarket], "source": [base_source]}),
            basis[["submarket", "source"]].drop_duplicates(),
        ],
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)
    expanded = (
        curve.merge(markets, how="cross")
        .merge(basis, on=["tenor", "submarket", "source"], how="left")
        .assign(
            spread=lambda data: data["spread"].where(
                (data["submarket"] != base_submarket) | (data["source"] != base_source),
                data["spread"].fillna(0.0),
            )
        )
    )
    missing = expanded.loc[expanded["spread"].isna(), ["submarket", "source"]]
    if not missing.empty:
        pairs = sorted(map(tuple, missing.drop_duplicates().to_numpy()))
        raise ValueError(f"Spread does not cover every month of: {pairs}")
    delivered = expanded.assign(
        price=lambda data: data["raw_price"] + data["spread"]
    ).rename(
        columns={
            "tenor": "vencimento",
            "source": "tipo_energia",
            "submarket": "submercado",
            "scenario": "cenario",
            "price": "preco",
            "index_base": "data_base_ipca",
            "index_start": "data_inicio_ipca",
            "ipca_forecast": "ipca_previsto",
        }
    )
    return cast(
        DataFrame[ExpandedCurveOutput],
        delivered[list(ExpandedCurveOutput.to_schema().columns)].sort_values(
            ["submercado", "tipo_energia", "cenario", "vencimento"],
            ignore_index=True,
        ),
    )


def wide_curve(curve: DataFrame[ForwardCurveOutput], column: str) -> pd.DataFrame:
    """Return one scenario per column, indexed by tenor."""

    return curve.pivot(index="tenor", columns="scenario", values=column)
