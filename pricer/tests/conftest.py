from __future__ import annotations

import numpy as np
import pytest
from pricer.curves.arbitrage import block_basis


@pytest.fixture
def initial_curve() -> np.ndarray:
    return np.linspace(95.0, 105.0, 12)


@pytest.fixture
def annual_exposure() -> np.ndarray:
    return np.ones((1, 12)) / 12.0


@pytest.fixture
def annual_q1_exposure() -> np.ndarray:
    return np.vstack(
        [
            np.ones(12) / 12.0,
            np.r_[np.ones(3) / 3.0, np.zeros(9)],
        ]
    )


@pytest.fixture
def residual_basis() -> np.ndarray:
    basis, _ = block_basis(["Q1"] * 3 + ["residual"] * 9)
    return basis


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(2027)


@pytest.fixture
def liquidity_regime_curve() -> np.ndarray:
    return np.array([310.0, 312.0, 314.0, 300.0, 300.0, 300.0, *([290.0] * 6)])


@pytest.fixture
def liquidity_regime_basis() -> np.ndarray:
    basis, _ = block_basis(["M1", "M2", "M3", "Q2", "Q2", "Q2", *(["H2"] * 6)])
    return basis


@pytest.fixture
def liquidity_regime_exposure() -> np.ndarray:
    return np.vstack(
        [
            np.eye(12)[0],
            np.eye(12)[1],
            np.eye(12)[2],
            np.r_[np.zeros(3), np.ones(3) / 3.0, np.zeros(6)],
            np.r_[np.zeros(6), np.ones(6) / 6.0],
        ]
    )
