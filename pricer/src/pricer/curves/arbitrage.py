"""Project a curve onto a set of linear no-arbitrage restrictions.

This module deliberately knows nothing about products, dates, tickers, EWMA,
IPCA or pandas.  Its complete market input is a matrix ``A`` and observed
prices ``q``.  Each row states that ``A @ curve`` is a traded or preserved
linear price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import qr
from scipy.optimize import LinearConstraint, linprog, minimize

FloatArray = NDArray[np.float64]

__all__ = [
    "CurveSolution",
    "InfeasibleCurveError",
    "LinearObservations",
    "block_basis",
    "solve_curve",
]


class InfeasibleCurveError(ValueError):
    """Raised when linear observations and curve bounds cannot coexist."""


@dataclass(frozen=True, slots=True)
class LinearObservations:
    """Linear prices and their admissible regions.

    ``matrix`` has shape ``(observations, months)`` and ``values`` has shape
    ``(observations,)`` or ``(observations, scenarios)``.  With no explicit
    bounds every value is exact.  Finite ``weight`` adds a soft squared-error
    term independently of any bounds.
    """

    matrix: ArrayLike
    values: ArrayLike
    lower: ArrayLike | None = None
    upper: ArrayLike | None = None
    weight: ArrayLike | None = None

    @classmethod
    def exact(cls, matrix: ArrayLike, values: ArrayLike) -> LinearObservations:
        return cls(matrix=matrix, values=values)

    @classmethod
    def soft(
        cls,
        matrix: ArrayLike,
        values: ArrayLike,
        *,
        weight: ArrayLike = 1.0,
    ) -> LinearObservations:
        return cls(
            matrix=matrix,
            values=values,
            lower=-np.inf,
            upper=np.inf,
            weight=weight,
        )

    @classmethod
    def bounded(
        cls,
        matrix: ArrayLike,
        values: ArrayLike,
        *,
        lower: ArrayLike,
        upper: ArrayLike,
        weight: ArrayLike | None = None,
    ) -> LinearObservations:
        return cls(
            matrix=matrix,
            values=values,
            lower=lower,
            upper=upper,
            weight=weight,
        )


@dataclass(frozen=True, slots=True)
class CurveSolution:
    """Balanced monthly prices and observation diagnostics."""

    prices: FloatArray
    fitted: FloatArray
    residuals: FloatArray
    latent: FloatArray
    objective: float | FloatArray
    max_violation: float | FloatArray


def block_basis(
    labels: ArrayLike,
    *,
    shape: ArrayLike = 1.0,
    economic_weight: ArrayLike = 1.0,
) -> tuple[FloatArray, NDArray[np.object_]]:
    """Create a normalized month-to-block basis.

    Equal labels share one latent block price.  ``shape`` may vary inside a
    block; its weighted mean is normalized to one, so the block price keeps
    its economic meaning.
    """

    block = np.asarray(labels, dtype=object)
    if block.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    first = np.sort(np.unique(block, return_index=True)[1])
    names = block[first]
    indicator = (block[:, None] == names[None, :]).astype(float)
    loading = np.broadcast_to(np.asarray(shape, dtype=float), block.shape)
    weight = np.broadcast_to(np.asarray(economic_weight, dtype=float), block.shape)
    if np.any(loading <= 0.0) or np.any(weight <= 0.0):
        raise ValueError("shape and economic_weight must be positive")
    normalizer = (indicator * weight[:, None] * loading[:, None]).sum(axis=0)
    normalizer /= (indicator * weight[:, None]).sum(axis=0)
    return indicator * loading[:, None] / normalizer[None, :], names


def solve_curve(
    initial_curve: ArrayLike,
    observations: LinearObservations,
    *,
    basis: ArrayLike | None = None,
    floor: ArrayLike = -np.inf,
    cap: ArrayLike = np.inf,
    prior_strength: float = 1.0,
    prior_weight: ArrayLike = 1.0,
    smoothness: float = 0.0,
    tolerance: float = 1e-9,
) -> CurveSolution:
    """Return the nearest curve satisfying all exact linear restrictions.

    The monthly curve is ``p = Bx`` where ``B`` defaults to the identity.  The
    selected feasible curve minimizes

    ``prior_strength * ||p - p0||^2_R
       + ||A p - q||^2_W
       + smoothness * ||D2 p||^2``.

    Quote values may contain several scenario columns.  They are solved
    independently against the same exposure matrix and curve structure.
    """

    initial = np.asarray(initial_curve, dtype=float)
    matrix = np.asarray(observations.matrix, dtype=float)
    if initial.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != initial.size:
        raise ValueError(
            "Expected initial_curve (months,) and matrix (observations, months)"
        )

    targets, vector_input = _columns(observations.values, matrix.shape[0])
    shape = targets.shape
    if observations.lower is None and observations.upper is None:
        lower, upper = targets.copy(), targets.copy()
    else:
        lower = _broadcast(observations.lower, shape, -np.inf)
        upper = _broadcast(observations.upper, shape, np.inf)
    quote_weight = _broadcast(observations.weight, shape, 0.0)

    loading = np.eye(initial.size) if basis is None else np.asarray(basis, dtype=float)
    if loading.ndim != 2 or loading.shape[0] != initial.size:
        raise ValueError("basis must have shape (months, latent_prices)")

    month_floor = _vector(floor, initial.size)
    month_cap = _vector(cap, initial.size)
    curve_weight = _vector(prior_weight, initial.size)
    if (
        not np.isfinite(initial).all()
        or not np.isfinite(matrix).all()
        or not np.isfinite(loading).all()
        or not np.isfinite(targets).all()
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.isnan(month_floor).any()
        or np.isnan(month_cap).any()
        or not np.isfinite(quote_weight).all()
        or not np.isfinite(curve_weight).all()
        or np.any(lower > upper)
        or np.any(month_floor > month_cap)
        or np.any(quote_weight < 0.0)
        or np.any(curve_weight < 0.0)
        or not np.isfinite([prior_strength, smoothness, tolerance]).all()
        or min(prior_strength, smoothness) < 0.0
        or tolerance <= 0.0
    ):
        raise ValueError("Invalid values, bounds or penalties in curve problem")

    quote_loading = matrix @ loading
    second_difference = np.diff(np.eye(initial.size), n=2, axis=0) @ loading
    solved = [
        _solve_column(
            initial,
            loading,
            quote_loading,
            targets[:, column],
            lower[:, column],
            upper[:, column],
            quote_weight[:, column],
            month_floor,
            month_cap,
            curve_weight,
            second_difference,
            prior_strength,
            smoothness,
            tolerance,
        )
        for column in range(targets.shape[1])
    ]

    latent = np.column_stack([item[0] for item in solved])
    prices = loading @ latent
    fitted = matrix @ prices
    residuals = fitted - targets
    objectives = np.asarray([item[1] for item in solved])
    violations = np.asarray([item[2] for item in solved])
    if vector_input:
        return CurveSolution(
            prices=prices[:, 0],
            fitted=fitted[:, 0],
            residuals=residuals[:, 0],
            latent=latent[:, 0],
            objective=float(objectives[0]),
            max_violation=float(violations[0]),
        )
    return CurveSolution(
        prices=prices,
        fitted=fitted,
        residuals=residuals,
        latent=latent,
        objective=objectives,
        max_violation=violations,
    )


def _columns(values: ArrayLike, rows: int) -> tuple[FloatArray, bool]:
    array = np.asarray(values, dtype=float)
    vector = array.ndim == 1
    if vector:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != rows:
        raise ValueError("Observation values must have one row per matrix row")
    return array, vector


def _broadcast(
    value: ArrayLike | None, shape: tuple[int, int], default: float
) -> FloatArray:
    if value is None:
        return np.full(shape, default, dtype=float)
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    try:
        return np.broadcast_to(array, shape).astype(float, copy=True)
    except ValueError as error:
        raise ValueError(f"Cannot broadcast observation data to {shape}") from error


def _vector(value: ArrayLike, size: int) -> FloatArray:
    try:
        return np.broadcast_to(np.asarray(value, dtype=float), (size,)).astype(
            float, copy=True
        )
    except ValueError as error:
        raise ValueError(f"Cannot broadcast monthly data to ({size},)") from error


def _constraint_parts(
    matrix: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    exact = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & np.isclose(
            lower,
            upper,
            atol=tolerance,
            rtol=tolerance,
        )
    )
    bounded = ~exact & (np.isfinite(lower) | np.isfinite(upper))
    return (
        matrix[exact],
        (lower[exact] + upper[exact]) / 2.0,
        matrix[bounded],
        lower[bounded],
        upper[bounded],
    )


def _independent_rows(
    matrix: FloatArray, values: FloatArray
) -> tuple[FloatArray, FloatArray]:
    if matrix.shape[0] == 0:
        return matrix, values
    _, triangular, pivots = qr(matrix.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    threshold = max(matrix.shape) * np.finfo(float).eps * diagonal.max(initial=0.0)
    keep = np.sort(pivots[: int(np.sum(diagonal > threshold))])
    return matrix[keep], values[keep]


def _feasible_point(
    dimension: int,
    equality: FloatArray,
    equality_values: FloatArray,
    interval: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    finite_upper = np.isfinite(upper)
    finite_lower = np.isfinite(lower)
    inequality_parts = [interval[finite_upper], -interval[finite_lower]]
    value_parts = [upper[finite_upper], -lower[finite_lower]]
    inequality = (
        np.vstack([part for part in inequality_parts if len(part)])
        if (finite_upper.any() or finite_lower.any())
        else None
    )
    inequality_values = (
        np.concatenate([part for part in value_parts if len(part)])
        if (finite_upper.any() or finite_lower.any())
        else None
    )
    result = linprog(
        np.zeros(dimension),
        A_ub=inequality,
        b_ub=inequality_values,
        A_eq=equality if len(equality) else None,
        b_eq=equality_values if len(equality) else None,
        bounds=[(None, None)] * dimension,
        method="highs",
    )
    if not result.success:
        raise InfeasibleCurveError(result.message)
    return result.x


def _solve_column(
    initial: FloatArray,
    loading: FloatArray,
    quote_loading: FloatArray,
    target: FloatArray,
    quote_lower: FloatArray,
    quote_upper: FloatArray,
    quote_weight: FloatArray,
    month_floor: FloatArray,
    month_cap: FloatArray,
    curve_weight: FloatArray,
    second_difference: FloatArray,
    prior_strength: float,
    smoothness: float,
    tolerance: float,
) -> tuple[FloatArray, float, float]:
    constraint_matrix = np.vstack([quote_loading, loading])
    lower = np.concatenate([quote_lower, month_floor])
    upper = np.concatenate([quote_upper, month_cap])
    equality, equality_values, interval, interval_lower, interval_upper = (
        _constraint_parts(
            constraint_matrix,
            lower,
            upper,
            tolerance,
        )
    )
    feasible = _feasible_point(
        loading.shape[1],
        equality,
        equality_values,
        interval,
        interval_lower,
        interval_upper,
    )

    weighted_loading = loading * (prior_strength * curve_weight)[:, None]
    hessian = loading.T @ weighted_loading
    linear = weighted_loading.T @ initial
    hessian += quote_loading.T @ (quote_weight[:, None] * quote_loading)
    linear += quote_loading.T @ (quote_weight * target)
    hessian += smoothness * second_difference.T @ second_difference

    if np.linalg.matrix_rank(equality) == loading.shape[1] or not np.any(hessian):
        latent = feasible
    else:
        independent, independent_values = _independent_rows(equality, equality_values)
        constraints = [
            *(
                [LinearConstraint(independent, independent_values, independent_values)]
                if len(independent)
                else []
            ),
            *(
                [LinearConstraint(interval, interval_lower, interval_upper)]
                if len(interval)
                else []
            ),
        ]
        result = minimize(
            lambda x: 0.5 * float(x @ hessian @ x) - float(linear @ x),
            feasible,
            jac=lambda x: hessian @ x - linear,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": tolerance, "maxiter": 2_000, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"Curve optimization failed: {result.message}")
        latent = result.x

    prices = loading @ latent
    fitted = quote_loading @ latent
    violation = _max_violation(constraint_matrix @ latent, lower, upper)
    if violation > max(100.0 * tolerance, 1e-7):
        raise RuntimeError(
            f"Curve optimization violated a restriction by {violation:.3g}"
        )
    objective = 0.5 * prior_strength * np.sum(curve_weight * (prices - initial) ** 2)
    objective += 0.5 * np.sum(quote_weight * (fitted - target) ** 2)
    objective += 0.5 * smoothness * np.sum((second_difference @ latent) ** 2)
    return latent, float(objective), violation


def _max_violation(values: FloatArray, lower: FloatArray, upper: FloatArray) -> float:
    below = np.where(np.isfinite(lower), np.maximum(lower - values, 0.0), 0.0)
    above = np.where(np.isfinite(upper), np.maximum(values - upper, 0.0), 0.0)
    return float(max(below.max(initial=0.0), above.max(initial=0.0)))
