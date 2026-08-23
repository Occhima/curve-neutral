from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration.neutralization import (
    AnchorMatrix,
    AnchorOutput,
    CurveOutput,
    ScenarioOutput,
    build_anchor_matrix,
    neutralize_curve,
)
from pandas.testing import assert_frame_equal, assert_series_equal
from pandera.errors import SchemaErrors

pytestmark = pytest.mark.functional


def _tenors(months: int = 12) -> pd.DatetimeIndex:
    return pd.date_range("2027-01-01", periods=months, freq="MS")


def _curve(*, blocks: list[str] | None = None, months: int = 12) -> pd.DataFrame:
    tenor = _tenors(months)
    data: dict[str, object] = {
        "tenor": tenor,
        "price": np.linspace(95.0, 105.0, months),
        "floor": 0.0,
        "cap": 500.0,
    }
    if blocks is not None:
        data["block"] = blocks
    return pd.DataFrame(data)


def _annual_and_q1() -> pd.DataFrame:
    tenor = _tenors()
    return pd.DataFrame(
        np.vstack(
            [
                np.ones(12) / 12.0,
                np.r_[np.ones(3) / 3.0, np.zeros(9)],
            ]
        ),
        index=["CAL27", "Q127"],
        columns=tenor,
    )


def test_primary_api_only_needs_curve_and_anchor_matrix() -> None:
    curve = _curve(blocks=["Q1"] * 3 + ["residual"] * 9)
    exposure = _annual_and_q1()
    anchors = AnchorMatrix.exact(
        exposure,
        pd.Series([120.0, 90.0], index=exposure.index),
    )

    result = neutralize_curve(curve, anchors)

    prices = result.wide_curve()["base"]
    np.testing.assert_allclose(prices.iloc[:3], 90.0, atol=1e-8)
    np.testing.assert_allclose(prices.iloc[3:], 130.0, atol=1e-8)
    np.testing.assert_allclose(result.anchors["fitted"], [120.0, 90.0], atol=1e-8)
    CurveOutput.validate(result.curve, lazy=True)
    AnchorOutput.validate(result.anchors, lazy=True)
    ScenarioOutput.validate(result.scenarios, lazy=True)


def test_price_dataframe_is_treated_as_independent_scenarios() -> None:
    curve = _curve(blocks=["Q1"] * 3 + ["residual"] * 9)
    exposure = _annual_and_q1()
    prices = pd.DataFrame(
        {
            "base": [120.0, 90.0],
            "low": [100.0, 80.0],
            "high": [130.0, 110.0],
        },
        index=exposure.index,
    )

    result = neutralize_curve(curve, AnchorMatrix.exact(exposure, prices))

    assert result.wide_curve().columns.tolist() == ["base", "high", "low"]
    assert set(result.curve["scenario"]) == {"base", "low", "high"}
    assert len(result.curve) == 36
    np.testing.assert_allclose(
        result.anchors.pivot(
            index="anchor", columns="scenario", values="fitted"
        ).reindex(index=prices.index, columns=prices.columns),
        prices,
        atol=1e-8,
    )


def test_non_string_scenario_labels_are_canonicalized_once() -> None:
    curve = _curve(blocks=["Q1"] * 3 + ["residual"] * 9)
    exposure = _annual_and_q1()
    prices = pd.DataFrame(
        {1: [120.0, 90.0], 2: [100.0, 80.0]},
        index=exposure.index,
    )
    lower = pd.DataFrame(-np.inf, index=exposure.index, columns=prices.columns)
    upper = pd.DataFrame(np.inf, index=exposure.index, columns=prices.columns)
    weight = pd.DataFrame(10.0, index=exposure.index, columns=prices.columns)

    result = neutralize_curve(
        curve,
        AnchorMatrix(exposure, prices, lower=lower, upper=upper, weight=weight),
    )

    assert set(result.curve["scenario"]) == {"1", "2"}


def test_labels_align_shuffled_exposure_prices_and_tenors() -> None:
    curve = _curve(blocks=["Q1"] * 3 + ["residual"] * 9)
    exposure = _annual_and_q1().iloc[::-1, ::-1]
    prices = pd.Series({"CAL27": 120.0, "Q127": 90.0})

    result = neutralize_curve(curve, AnchorMatrix.exact(exposure, prices))

    fitted = result.anchors.set_index("anchor")["fitted"]
    np.testing.assert_allclose(fitted.reindex(prices.index), prices, atol=1e-8)


def test_product_labels_are_optional_metadata() -> None:
    exposure = _annual_and_q1().set_axis([41, 73])
    prices = pd.Series([120.0, 90.0], index=[41, 73])

    result = neutralize_curve(
        _curve(blocks=["Q1"] * 3 + ["residual"] * 9),
        AnchorMatrix.exact(exposure, prices),
    )

    assert result.anchors["anchor"].tolist() == ["41", "73"]


def test_build_anchor_matrix_places_indexation_in_the_coefficients() -> None:
    tenor = _tenors(3)
    curve = pd.DataFrame(
        {
            "tenor": tenor,
            "price": [90.0, 100.0, 110.0],
            "energy_weight": [2.0, 3.0, 5.0],
            "discount_factor": [1.0, 0.9, 0.8],
            "index_factor": [1.0, 1.1, 1.2],
        }
    )
    delivery = pd.DataFrame(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        index=["ALL", "FIRST_TWO"],
        columns=tenor,
    )
    quote_factor = pd.Series({"ALL": 1.05, "FIRST_TWO": 1.10})

    anchors = build_anchor_matrix(
        curve,
        delivery,
        pd.Series([100.0, 90.0], index=delivery.index),
        quote_index_factor=quote_factor,
    )

    base_weight = curve["energy_weight"] * curve["discount_factor"]
    expected = (
        delivery.to_numpy() * base_weight.to_numpy() * curve["index_factor"].to_numpy()
    ) / (
        (delivery.to_numpy() * base_weight.to_numpy()).sum(axis=1)
        * quote_factor.to_numpy()
    )[:, None]
    np.testing.assert_allclose(anchors.exposure, expected, atol=1e-12)


def test_default_calendar_weights_create_an_hour_weighted_average() -> None:
    tenor = _tenors(3)
    curve = pd.DataFrame({"tenor": tenor, "price": 100.0})
    delivery = pd.DataFrame([np.ones(3)], index=["Q1"], columns=tenor)

    anchors = build_anchor_matrix(
        curve,
        delivery,
        pd.Series([100.0], index=delivery.index),
    )

    hours = tenor.days_in_month.to_numpy(dtype=float) * 24.0
    np.testing.assert_allclose(
        anchors.exposure.iloc[0], hours / hours.sum(), atol=1e-12
    )


def test_soft_anchor_is_not_silently_promoted_to_an_exact_constraint() -> None:
    tenor = _tenors(4)
    curve = pd.DataFrame({"tenor": tenor, "price": 90.0})
    exposure = pd.DataFrame(
        [np.ones(4) / 4.0],
        index=["EWMA"],
        columns=tenor,
    )

    result = neutralize_curve(
        curve,
        AnchorMatrix.soft(
            exposure,
            pd.Series([110.0], index=exposure.index),
            weight=4.0,
        ),
    )

    fitted = result.anchors["fitted"].iat[0]
    assert 90.0 < fitted < 110.0


def test_exact_and_soft_anchor_policy_is_expressed_only_by_arrays() -> None:
    tenor = _tenors(4)
    curve = pd.DataFrame({"tenor": tenor, "price": 90.0})
    exposure = pd.DataFrame(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ],
        index=["exact", "soft"],
        columns=tenor,
    )
    prices = pd.Series([100.0, 120.0], index=exposure.index)
    lower = pd.Series([100.0, -np.inf], index=exposure.index)
    upper = pd.Series([100.0, np.inf], index=exposure.index)
    weight = pd.Series([0.0, 2.0], index=exposure.index)

    result = neutralize_curve(
        curve,
        AnchorMatrix(exposure, prices, lower=lower, upper=upper, weight=weight),
    )

    fitted = result.anchors.set_index("anchor")["fitted"]
    np.testing.assert_allclose(fitted["exact"], 100.0, atol=1e-8)
    assert 90.0 < fitted["soft"] < 120.0


def test_curve_floor_and_cap_are_forwarded_to_pricer() -> None:
    tenor = _tenors(2)
    curve = pd.DataFrame(
        {
            "tenor": tenor,
            "price": [-10.0, 110.0],
            "floor": 0.0,
            "cap": 100.0,
        }
    )
    exposure = pd.DataFrame(index=pd.Index([], dtype=str), columns=tenor, dtype=float)
    prices = pd.Series(index=exposure.index, dtype=float)

    result = neutralize_curve(curve, AnchorMatrix.exact(exposure, prices))

    np.testing.assert_allclose(result.wide_curve()["base"], [0.0, 100.0], atol=2e-7)


def test_bounded_anchor_constructor_forwards_bid_ask_region() -> None:
    tenor = _tenors(3)
    curve = pd.DataFrame({"tenor": tenor, "price": 80.0})
    exposure = pd.DataFrame(
        [np.ones(3) / 3.0],
        index=["A"],
        columns=tenor,
    )

    result = neutralize_curve(
        curve,
        AnchorMatrix.bounded(
            exposure,
            pd.Series([100.0], index=exposure.index),
            lower=95.0,
            upper=105.0,
        ),
    )

    np.testing.assert_allclose(result.anchors["fitted"], [95.0], atol=2e-7)


def test_seasonal_shape_and_block_labels_are_applied_without_product_logic() -> None:
    tenor = _tenors(4)
    curve = pd.DataFrame(
        {
            "tenor": tenor,
            "price": 100.0,
            "block": ["H1", "H1", "H2", "H2"],
            "seasonal_shape": [0.8, 1.2, 0.7, 1.3],
            "energy_weight": [1.0, 3.0, 2.0, 4.0],
        }
    )
    exposure = pd.DataFrame(
        [np.array([1.0, 3.0, 2.0, 4.0]) / 10.0],
        index=["ALL"],
        columns=tenor,
    )

    result = neutralize_curve(
        curve,
        AnchorMatrix.exact(exposure, pd.Series([110.0], index=exposure.index)),
    )

    np.testing.assert_allclose(result.anchors["fitted"], [110.0], atol=1e-8)
    prices = result.wide_curve()["base"].to_numpy()
    np.testing.assert_allclose(prices[1] / prices[0], 1.2 / 0.8, atol=1e-8)
    np.testing.assert_allclose(prices[3] / prices[2], 1.3 / 0.7, atol=1e-8)


def test_inputs_are_not_mutated() -> None:
    curve = _curve(blocks=["Q1"] * 3 + ["residual"] * 9)
    exposure = _annual_and_q1()
    prices = pd.Series([120.0, 90.0], index=exposure.index)
    curve_before = curve.copy(deep=True)
    exposure_before = exposure.copy(deep=True)
    prices_before = prices.copy(deep=True)

    neutralize_curve(curve, AnchorMatrix.exact(exposure, prices))

    assert_frame_equal(curve, curve_before)
    assert_frame_equal(exposure, exposure_before)
    assert_series_equal(prices, prices_before)


def test_duplicate_curve_tenor_is_rejected_at_the_dataframe_boundary() -> None:
    curve = _curve(months=3)
    curve.loc[1, "tenor"] = curve.loc[0, "tenor"]
    exposure = pd.DataFrame(
        [[1.0, 0.0, 0.0]],
        index=["A"],
        columns=_tenors(3),
    )

    with pytest.raises(SchemaErrors):
        neutralize_curve(
            curve,
            AnchorMatrix.exact(exposure, pd.Series([100.0], index=exposure.index)),
        )


def test_invalid_curve_bounds_are_rejected_at_the_dataframe_boundary() -> None:
    curve = _curve(months=2).assign(floor=[0.0, 20.0], cap=[10.0, 10.0])
    exposure = pd.DataFrame(
        index=pd.Index([], dtype=str), columns=_tenors(2), dtype=float
    )

    with pytest.raises(SchemaErrors):
        neutralize_curve(curve, AnchorMatrix.exact(exposure, pd.Series(dtype=float)))


def test_missing_exposure_tenor_is_not_guessed() -> None:
    curve = _curve(months=3)
    exposure = pd.DataFrame(
        [[0.5, 0.5]],
        index=["A"],
        columns=_tenors(2),
    )

    with pytest.raises(ValueError, match="Invalid values"):
        neutralize_curve(
            curve,
            AnchorMatrix.exact(exposure, pd.Series([100.0], index=exposure.index)),
        )


def test_duplicate_anchor_labels_are_rejected_before_alignment() -> None:
    exposure = _annual_and_q1().set_axis(["A", "A"])
    prices = pd.Series([100.0, 90.0], index=["A", "A"])

    with pytest.raises(ValueError, match="unique"):
        neutralize_curve(_curve(), AnchorMatrix.exact(exposure, prices))
