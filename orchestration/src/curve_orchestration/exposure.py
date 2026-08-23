"""Economic exposure construction owned by the market-data layer."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

__all__ = ["cashflow_matrix"]


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
