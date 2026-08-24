from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from curve_orchestration.neutralization import AnchorMatrix


@pytest.fixture
def tenors() -> pd.DatetimeIndex:
    return pd.date_range("2027-01-01", periods=12, freq="MS")


@pytest.fixture
def block_curve(tenors: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenor": tenors,
            "price": np.linspace(95.0, 105.0, len(tenors)),
            "floor": 0.0,
            "cap": 500.0,
            "block": ["Q1"] * 3 + ["residual"] * 9,
        }
    )


@pytest.fixture
def canonical_curve(tenors: pd.DatetimeIndex) -> pd.DataFrame:
    hours = tenors.days_in_month.to_numpy(dtype=float) * 24.0
    return pd.DataFrame(
        {
            "tenor": tenors,
            "price": 100.0,
            "energy_weight": hours,
            "discount_factor": 1.0,
            "index_factor": 1.0,
            "index_base": tenors[0],
            "index_start": tenors,
            "ipca_base": 1.0,
            "ipca_forecast": 1.0,
            "floor": 0.0,
            "cap": 500.0,
            "block": tenors.strftime("%Y-%m"),
            "seasonal_shape": 1.0,
        }
    )


@pytest.fixture
def anchor_exposure(tenors: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        np.vstack(
            [
                np.ones(12) / 12.0,
                np.r_[np.ones(3) / 3.0, np.zeros(9)],
            ]
        ),
        index=["CAL27", "Q127"],
        columns=tenors,
    )


@pytest.fixture
def anchor_prices(anchor_exposure: pd.DataFrame) -> pd.Series:
    return pd.Series([120.0, 90.0], index=anchor_exposure.index, name="base")


@pytest.fixture
def exact_anchors(
    anchor_exposure: pd.DataFrame,
    anchor_prices: pd.Series,
) -> AnchorMatrix:
    return AnchorMatrix.exact(anchor_exposure, anchor_prices)


@pytest.fixture
def liquidity_regime_frame(tenors: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenor": tenors,
            "price": [
                310.0,
                312.0,
                314.0,
                300.0,
                300.0,
                300.0,
                *([290.0] * 6),
            ],
            "block": [
                "M1",
                "M2",
                "M3",
                "Q2",
                "Q2",
                "Q2",
                *(["H2"] * 6),
            ],
        }
    )


@pytest.fixture
def liquidity_regime_exposure(tenors: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        np.vstack(
            [
                np.eye(12)[0],
                np.eye(12)[1],
                np.eye(12)[2],
                np.r_[np.zeros(3), np.ones(3) / 3.0, np.zeros(6)],
                np.r_[np.zeros(6), np.ones(6) / 6.0],
            ]
        ),
        index=["M0127", "M0227", "M0327", "Q227", "H227"],
        columns=tenors,
    )
