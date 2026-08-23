"""Compose the liquid market segment with the external illiquid curve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

from .anchor_selection import AnchorPlan, neutralize_anchor_plan
from .neutralization import NeutralizationResult, _prepare_curve


class ForwardCurveOutput(pa.DataFrameModel):
    """Raw and indexed monthly prices after the market/DCIDE splice."""

    tenor: Series[pa.DateTime]
    scenario: Series[str]
    source: Series[str] = pa.Field(isin=["market", "dcide"])
    raw_price: Series[float]
    index_factor: Series[float] = pa.Field(gt=0)
    indexed_price: Series[float]

    @pa.dataframe_check
    def unique_scenario_tenors(cls, frame: pd.DataFrame) -> bool:
        return bool(not frame.duplicated(["scenario", "tenor"]).any())

    class Config:
        coerce = True
        strict = True


@dataclass(frozen=True, slots=True)
class ForwardCurveResult:
    """The final curve together with the liquid-segment optimization result."""

    curve: DataFrame[ForwardCurveOutput]
    neutralization: NeutralizationResult

    def wide_raw_curve(self) -> pd.DataFrame:
        return self.curve.pivot(
            index="tenor",
            columns="scenario",
            values="raw_price",
        )

    def wide_indexed_curve(self) -> pd.DataFrame:
        return self.curve.pivot(
            index="tenor",
            columns="scenario",
            values="indexed_price",
        )


def finalize_forward_curve(
    plan: AnchorPlan,
    neutralization: NeutralizationResult,
    dcide_curve: pd.DataFrame,
) -> DataFrame[ForwardCurveOutput]:
    """Splice raw prices first, then index the complete curve in one operation."""

    cutoff = (
        plan.cutoff
        if plan.cutoff is not None
        else pd.Timestamp(plan.curve["tenor"].max())
    )
    scenarios = neutralization.scenarios[["scenario"]].drop_duplicates()
    factors = plan.curve[["tenor", "index_factor"]]
    market = (
        neutralization.curve.query(
            "tenor <= @cutoff",
            local_dict={"cutoff": cutoff},
        )[["tenor", "scenario", "price"]]
        .rename(columns={"price": "raw_price"})
        .merge(factors, on="tenor", how="left", validate="many_to_one")
        .assign(source="market")
    )
    dcide = (
        _prepare_curve(dcide_curve)
        .query(
            "tenor > @cutoff",
            local_dict={"cutoff": cutoff},
        )[["tenor", "price", "index_factor"]]
        .rename(columns={"price": "raw_price"})
        .merge(scenarios, how="cross")
        .assign(source="dcide")
    )
    output = (
        pd.concat([market, dcide], ignore_index=True)
        .assign(indexed_price=lambda data: data["raw_price"] * data["index_factor"])[
            [
                "tenor",
                "scenario",
                "source",
                "raw_price",
                "index_factor",
                "indexed_price",
            ]
        ]
        .sort_values(["scenario", "tenor"], ignore_index=True)
        .pipe(ForwardCurveOutput.validate, lazy=True)
    )
    return cast(DataFrame[ForwardCurveOutput], output)


def neutralize_forward_curve(
    plan: AnchorPlan,
    dcide_curve: pd.DataFrame,
    *,
    prior_strength: float = 0.0,
    smoothness: float = 0.0,
    tolerance: float = 1e-9,
) -> ForwardCurveResult:
    """Neutralize the market segment, append DCIDE and apply indexation once."""

    neutralization = neutralize_anchor_plan(
        plan,
        prior_strength=prior_strength,
        smoothness=smoothness,
        tolerance=tolerance,
    )
    return ForwardCurveResult(
        curve=finalize_forward_curve(plan, neutralization, dcide_curve),
        neutralization=neutralization,
    )
