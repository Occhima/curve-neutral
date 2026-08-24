from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration import (
    CurveGranularity,
    apply_ipca,
    build_market_plan,
    build_unbalanced_curve,
    expand_curve,
    finalize_forward_curve,
    neutralize_anchor_plan,
    normalize_products,
    select_tenor_prices,
)

pytestmark = pytest.mark.unit

TENORS = pd.date_range("2027-01-01", periods=12, freq="MS")
IPCA_MONTHS = pd.date_range("2026-08-01", "2030-12-01", freq="MS")
IPCA = pd.DataFrame(
    {
        "tenor": IPCA_MONTHS,
        "indice": 7000.0 * (1.0045 ** np.arange(len(IPCA_MONTHS))),
    }
)


@pytest.fixture
def quotes() -> pd.DataFrame:
    return normalize_products(
        pd.DataFrame(
            [
                {
                    "product_id": "Q127",
                    "description": "Q127",
                    "start": "2027-01-01",
                    "end": "2027-03-31",
                    "price": 325.0,
                    "traded_at": pd.Timestamp("2026-08-21"),
                },
                {
                    "product_id": "CAL27",
                    "description": "CAL27",
                    "start": "2027-01-01",
                    "end": "2027-12-31",
                    "price": 295.0,
                    "traded_at": pd.Timestamp("2026-08-21"),
                },
            ]
        )
    )


@pytest.fixture
def curve() -> pd.DataFrame:
    return pd.DataFrame({"tenor": TENORS, "price": 300.0, "floor": 0.0, "cap": 1_000.0})


# --------------------------------------------------------------------------- #
# CurveGranularity.from_dict
# --------------------------------------------------------------------------- #
def test_from_dict_matches_the_chained_builders() -> None:
    config = CurveGranularity.from_dict(
        {
            "default": ["MEN", "TRI", "SEM", "ANU"],
            "rules": [
                {"year": 2027, "priority": ["ANU"]},
                {"year": 2028, "months": [1, 2, 3], "priority": ["TRI", "ANU"]},
            ],
        }
    )

    assert config == CurveGranularity().for_year(2027, ("ANU",)).for_months(
        2028, [1, 2, 3], ("TRI", "ANU")
    )
    assert config.priority_for(pd.Timestamp("2027-05-01")) == ("ANU",)
    assert config.priority_for(pd.Timestamp("2028-02-01")) == ("TRI", "ANU")
    assert config.priority_for(pd.Timestamp("2028-07-01")) == config.default


def test_from_dict_defaults_to_the_whole_year_and_the_market_priority() -> None:
    config = CurveGranularity.from_dict(
        {"rules": [{"year": 2027, "priority": ["SEM"]}]}
    )

    assert config.default == CurveGranularity().default
    assert config.rules[0].months == frozenset(range(1, 13))


def test_from_dict_rejects_unknown_settings_and_codes() -> None:
    with pytest.raises(KeyError, match="Unknown granularity settings"):
        CurveGranularity.from_dict({"regras": []})
    with pytest.raises(ValueError, match="non-empty subset"):
        CurveGranularity.from_dict({"rules": [{"year": 2027, "priority": ["XPTO"]}]})
    with pytest.raises(ValueError, match="non-empty subset"):
        CurveGranularity.from_dict({"default": []})


# --------------------------------------------------------------------------- #
# apply_ipca
# --------------------------------------------------------------------------- #
def test_ipca_factor_comes_from_each_block_head_not_from_each_month(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
) -> None:
    unbalanced = build_unbalanced_curve(
        curve, select_tenor_prices(quotes, curve, cutoff="2027-12-31")
    )

    indexed = apply_ipca(unbalanced, IPCA, base_date="2026-08-01")

    heads = indexed["index_start"].dt.strftime("%Y-%m")
    assert sorted(set(heads)) == ["2027-01", "2027-04"]
    np.testing.assert_allclose(
        indexed["index_factor"],
        indexed["ipca_forecast"] / indexed["ipca_base"],
    )
    assert indexed["index_factor"].nunique() == 2
    np.testing.assert_allclose(indexed["ipca_base"], 7000.0)


def test_a_missing_ipca_value_is_an_error_not_a_silent_one(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
) -> None:
    unbalanced = build_unbalanced_curve(
        curve, select_tenor_prices(quotes, curve, cutoff="2027-12-31")
    )

    with pytest.raises(KeyError, match="adjustment dates"):
        apply_ipca(unbalanced, IPCA.head(2), base_date="2026-08-01")
    with pytest.raises(KeyError, match="base date"):
        apply_ipca(unbalanced, IPCA, base_date="2020-01-01")


def test_the_plan_indexes_from_the_ipca_table_when_one_is_supplied(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
) -> None:
    plan = build_market_plan(
        curve,
        quotes,
        cutoff="2027-12-31",
        ipca=IPCA,
        base_date="2026-08-01",
    )

    np.testing.assert_allclose(
        plan.curve["index_factor"],
        plan.curve["ipca_forecast"] / plan.curve["ipca_base"],
    )
    assert plan.curve["index_factor"].nunique() == 2


# --------------------------------------------------------------------------- #
# expand_curve
# --------------------------------------------------------------------------- #
@pytest.fixture
def forward(curve: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    plan = build_market_plan(
        curve, quotes, cutoff="2027-12-31", ipca=IPCA, base_date="2026-08-01"
    )
    return finalize_forward_curve(
        plan, neutralize_anchor_plan(plan), curve.assign(price=280.0)
    )


@pytest.fixture
def spread() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vencimento": [*TENORS, *TENORS],
            "spread": [-15.0] * 12 + [40.0] * 12,
            "submercado": ["NE"] * 12 + ["SE"] * 12,
            "tipo_energia": ["CON"] * 12 + ["I5"] * 12,
        }
    )


def test_expanded_curve_has_the_delivered_shape(
    forward: pd.DataFrame,
    spread: pd.DataFrame,
) -> None:
    full = expand_curve(forward, spread)

    assert full.columns.tolist() == [
        "vencimento",
        "tipo_energia",
        "submercado",
        "cenario",
        "preco",
        "data_base_ipca",
        "data_inicio_ipca",
        "ipca_base",
        "ipca_previsto",
    ]
    assert len(full) == 36
    np.testing.assert_allclose(
        full["ipca_previsto"] / full["ipca_base"],
        full["vencimento"].map(forward.set_index("tenor")["index_factor"]),
    )


def test_every_market_is_the_reference_curve_plus_its_spread(
    forward: pd.DataFrame,
    spread: pd.DataFrame,
) -> None:
    full = expand_curve(forward, spread).set_index(
        ["submercado", "tipo_energia", "vencimento"]
    )["preco"]

    reference = full.loc["SE", "CON"]
    np.testing.assert_allclose(full.loc["NE", "CON"], reference - 15.0)
    np.testing.assert_allclose(full.loc["SE", "I5"], reference + 40.0)
    np.testing.assert_allclose(reference, forward["raw_price"])


def test_a_partial_spread_is_rejected_instead_of_publishing_the_reference_curve(
    forward: pd.DataFrame,
    spread: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match=r"does not cover every month of: \[\('NE'"):
        expand_curve(forward, spread.head(5))


def test_the_reference_market_needs_no_spread_row(
    forward: pd.DataFrame,
    spread: pd.DataFrame,
) -> None:
    only_north = spread.query("submercado == 'NE'")

    full = expand_curve(forward, only_north)

    markets = full[["submercado", "tipo_energia"]].drop_duplicates()
    assert set(map(tuple, markets.to_numpy())) == {("SE", "CON"), ("NE", "CON")}
    np.testing.assert_allclose(
        full.query("submercado == 'SE'")["preco"], forward["raw_price"]
    )
