from __future__ import annotations

import ast
import inspect

import numpy as np
import pricer.curves.arbitrage as arbitrage_module
import pytest
from pricer.curves.arbitrage import (
    InfeasibleCurveError,
    LinearObservations,
    block_basis,
    solve_curve,
)

pytestmark = pytest.mark.unit


def test_pricer_kernel_has_no_dataframe_or_exposure_builder_dependency() -> None:
    tree = ast.parse(inspect.getsource(arbitrage_module))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported.isdisjoint({"pandas", "pandera"})
    assert not hasattr(arbitrage_module, "cashflow_matrix")


def _flat_average(rows: int, months: int) -> np.ndarray:
    return np.full((rows, months), 1.0 / months)


def test_exact_annual_is_the_nearest_level_shift() -> None:
    initial = np.arange(90.0, 102.0)
    annual = _flat_average(1, 12)

    result = solve_curve(initial, LinearObservations.exact(annual, [100.0]))

    np.testing.assert_allclose(result.fitted, [100.0], atol=1e-9)
    np.testing.assert_allclose(
        result.prices - initial, 100.0 - initial.mean(), atol=1e-8
    )


def test_annual_and_quarter_determine_the_flat_residual_analytically() -> None:
    basis, names = block_basis(["Q1"] * 3 + ["residual"] * 9)
    annual = np.full(12, 1.0 / 12.0)
    quarter = np.r_[np.full(3, 1.0 / 3.0), np.zeros(9)]

    result = solve_curve(
        np.full(12, 100.0),
        LinearObservations.exact(np.vstack([annual, quarter]), [120.0, 90.0]),
        basis=basis,
    )

    assert names.tolist() == ["Q1", "residual"]
    np.testing.assert_allclose(result.latent, [90.0, 130.0], atol=1e-9)
    np.testing.assert_allclose(result.prices[:3], 90.0, atol=1e-9)
    np.testing.assert_allclose(result.prices[3:], 130.0, atol=1e-9)


def test_hour_weighted_residual_matches_the_cashflow_identity() -> None:
    hours = np.array([744.0, 672.0, 744.0, 720.0, 744.0, 720.0])
    delivery = np.vstack([np.ones(6), np.r_[np.ones(3), np.zeros(3)]])
    weighted_delivery = delivery * hours
    matrix = weighted_delivery / weighted_delivery.sum(axis=1, keepdims=True)
    basis, _ = block_basis(["Q1"] * 3 + ["Q2"] * 3, economic_weight=hours)

    result = solve_curve(
        np.full(6, 100.0),
        LinearObservations.exact(matrix, [105.0, 90.0]),
        basis=basis,
    )

    expected_q2 = (105.0 * hours.sum() - 90.0 * hours[:3].sum()) / hours[3:].sum()
    np.testing.assert_allclose(result.latent, [90.0, expected_q2], atol=1e-9)


def test_redundant_annual_semester_and_quarter_constraints_are_allowed() -> None:
    basis, _ = block_basis(np.repeat(["Q1", "Q2", "Q3", "Q4"], 3))
    quarter = np.kron(np.eye(4), np.full((1, 3), 1.0 / 3.0))
    half = np.vstack([quarter[:2].sum(axis=0) / 2.0, quarter[2:].sum(axis=0) / 2.0])
    annual = quarter.sum(axis=0, keepdims=True) / 4.0
    values = np.array([100.0, 100.0, 100.0, 90.0, 110.0, 100.0, 100.0])
    matrix = np.vstack([annual, half, quarter])

    result = solve_curve(
        np.full(12, 100.0),
        LinearObservations.exact(matrix, values),
        basis=basis,
    )

    np.testing.assert_allclose(result.latent, [90.0, 110.0, 100.0, 100.0], atol=1e-8)
    np.testing.assert_allclose(result.fitted, values, atol=1e-8)


def test_incompatible_exact_anchors_are_rejected() -> None:
    matrix = np.vstack([np.ones(3) / 3.0, np.ones(3) / 3.0])

    with pytest.raises(InfeasibleCurveError):
        solve_curve(
            np.full(3, 100.0),
            LinearObservations.exact(matrix, [100.0, 110.0]),
        )


def test_interval_anchor_selects_the_nearest_boundary() -> None:
    annual = _flat_average(1, 12)
    observations = LinearObservations.bounded(
        annual,
        [100.0],
        lower=[99.0],
        upper=[101.0],
    )

    result = solve_curve(np.full(12, 90.0), observations)

    np.testing.assert_allclose(result.prices, 99.0, atol=2e-7)
    np.testing.assert_allclose(result.fitted, [99.0], atol=2e-7)


def test_one_sided_anchor_bound_is_supported() -> None:
    observations = LinearObservations(
        matrix=np.ones((1, 4)) / 4.0,
        values=[100.0],
        lower=[100.0],
        upper=[np.inf],
    )

    result = solve_curve(np.full(4, 80.0), observations)

    np.testing.assert_allclose(result.fitted, [100.0], atol=2e-7)


def test_soft_quote_and_prior_form_the_expected_compromise() -> None:
    annual = _flat_average(1, 12)

    result = solve_curve(
        np.full(12, 90.0),
        LinearObservations.soft(annual, [110.0], weight=12.0),
    )

    np.testing.assert_allclose(result.prices, 100.0, atol=1e-7)
    np.testing.assert_allclose(result.fitted, [100.0], atol=1e-7)
    np.testing.assert_allclose(result.residuals, [-10.0], atol=1e-7)


def test_hard_and_soft_information_can_coexist_in_the_same_matrix() -> None:
    matrix = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
        ]
    )
    target = np.array([100.0, 120.0])
    observations = LinearObservations(
        matrix=matrix,
        values=target,
        lower=np.array([100.0, -np.inf]),
        upper=np.array([100.0, np.inf]),
        weight=np.array([0.0, 2.0]),
    )

    result = solve_curve(np.full(4, 90.0), observations)

    np.testing.assert_allclose(result.fitted[0], 100.0, atol=1e-8)
    assert 90.0 < result.fitted[1] < 120.0


def test_monthly_floor_and_cap_are_hard_restrictions() -> None:
    empty = LinearObservations.exact(np.empty((0, 3)), np.empty(0))

    result = solve_curve(
        np.array([-5.0, 50.0, 105.0]),
        empty,
        floor=0.0,
        cap=100.0,
    )

    np.testing.assert_allclose(result.prices, [0.0, 50.0, 100.0], atol=2e-7)


def test_incompatible_curve_bounds_are_rejected() -> None:
    empty = LinearObservations.exact(np.empty((0, 2)), np.empty(0))

    with pytest.raises(ValueError, match="Invalid values"):
        solve_curve([100.0, 100.0], empty, floor=[0.0, 10.0], cap=[5.0, 9.0])


def test_exact_anchor_outside_monthly_bounds_is_infeasible() -> None:
    with pytest.raises(InfeasibleCurveError):
        solve_curve(
            np.full(3, 100.0),
            LinearObservations.exact(np.ones((1, 3)) / 3.0, [200.0]),
            cap=150.0,
        )


def test_block_basis_preserves_each_blocks_weighted_economic_level() -> None:
    labels = np.array(["A", "A", "B", "B", "B"])
    shape = np.array([0.8, 1.2, 0.7, 1.0, 1.4])
    weight = np.array([1.0, 3.0, 2.0, 4.0, 1.0])

    basis, names = block_basis(labels, shape=shape, economic_weight=weight)

    for column, name in enumerate(names):
        selected = labels == name
        weighted_mean = np.average(basis[selected, column], weights=weight[selected])
        np.testing.assert_allclose(weighted_mean, 1.0, atol=1e-12)
        np.testing.assert_allclose(basis[~selected, column], 0.0)


@pytest.mark.parametrize(
    ("labels", "shape"),
    [
        ([["A", "B"]], 1.0),
        (["A", "B"], [1.0, -1.0]),
    ],
)
def test_invalid_block_basis_is_rejected(labels, shape) -> None:
    with pytest.raises(ValueError):
        block_basis(labels, shape=shape)


def test_fixed_seasonal_shape_does_not_change_the_anchor_quote() -> None:
    hours = np.arange(1.0, 7.0)
    shape = np.array([0.8, 1.0, 1.2, 0.7, 1.1, 1.4])
    basis, _ = block_basis(
        ["H1"] * 3 + ["H2"] * 3,
        shape=shape,
        economic_weight=hours,
    )
    delivery = np.vstack([np.ones(6), np.r_[np.ones(3), np.zeros(3)]])
    weighted_delivery = delivery * hours
    matrix = weighted_delivery / weighted_delivery.sum(axis=1, keepdims=True)

    result = solve_curve(
        np.full(6, 100.0),
        LinearObservations.exact(matrix, [110.0, 90.0]),
        basis=basis,
    )

    np.testing.assert_allclose(result.fitted, [110.0, 90.0], atol=1e-8)


def test_price_matrix_solves_multiple_scenarios() -> None:
    basis, _ = block_basis(["Q1"] * 3 + ["residual"] * 9)
    matrix = np.vstack(
        [
            np.ones(12) / 12.0,
            np.r_[np.ones(3) / 3.0, np.zeros(9)],
        ]
    )
    targets = np.array([[120.0, 100.0, 130.0], [90.0, 80.0, 110.0]])

    result = solve_curve(
        np.full(12, 100.0),
        LinearObservations.exact(matrix, targets),
        basis=basis,
    )

    assert result.prices.shape == (12, 3)
    assert result.fitted.shape == targets.shape
    np.testing.assert_allclose(result.fitted, targets, atol=1e-8)


def test_batched_scenarios_equal_independent_solves() -> None:
    rng = np.random.default_rng(18)
    matrix = rng.uniform(size=(3, 6))
    matrix /= matrix.sum(axis=1, keepdims=True)
    targets = rng.uniform(80.0, 120.0, size=(3, 4))
    batched = solve_curve(
        np.full(6, 100.0),
        LinearObservations.soft(matrix, targets, weight=5.0),
    )

    independent = np.column_stack(
        [
            solve_curve(
                np.full(6, 100.0),
                LinearObservations.soft(matrix, targets[:, column], weight=5.0),
            ).prices
            for column in range(targets.shape[1])
        ]
    )

    np.testing.assert_allclose(batched.prices, independent, atol=1e-8)


def test_random_exact_system_recovers_its_quotes() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(7, 7))
    true_curve = rng.uniform(50.0, 150.0, 7)

    result = solve_curve(
        rng.uniform(50.0, 150.0, 7),
        LinearObservations.exact(matrix, matrix @ true_curve),
    )

    np.testing.assert_allclose(result.prices, true_curve, atol=1e-7)
    np.testing.assert_allclose(result.max_violation, 0.0, atol=1e-7)


def test_anchor_row_order_does_not_change_the_solution() -> None:
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
    targets = np.array([90.0, 110.0])
    direct = solve_curve([100.0] * 3, LinearObservations.exact(matrix, targets))
    reverse = solve_curve(
        [100.0] * 3,
        LinearObservations.exact(matrix[::-1], targets[::-1]),
    )

    np.testing.assert_allclose(direct.prices, reverse.prices, atol=1e-9)


def test_smoothness_reduces_monthly_second_differences() -> None:
    initial = np.array([100.0, 150.0, 50.0, 150.0, 50.0, 100.0])
    endpoints = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    observations = LinearObservations.exact(endpoints, [100.0, 100.0])

    rough = solve_curve(initial, observations, smoothness=0.0)
    smooth = solve_curve(initial, observations, smoothness=100.0)

    assert np.linalg.norm(np.diff(smooth.prices, n=2)) < np.linalg.norm(
        np.diff(rough.prices, n=2)
    )
    np.testing.assert_allclose(smooth.fitted, [100.0, 100.0], atol=1e-8)


def test_solver_does_not_mutate_input_arrays() -> None:
    initial = np.arange(4.0)
    matrix = np.ones((1, 4)) / 4.0
    values = np.array([10.0])
    snapshots = [item.copy() for item in (initial, matrix, values)]

    solve_curve(initial, LinearObservations.exact(matrix, values))

    for actual, expected in zip((initial, matrix, values), snapshots):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("initial", "matrix", "values"),
    [
        ([[1.0, 2.0]], [[1.0, 0.0]], [1.0]),
        ([1.0, 2.0], [1.0, 0.0], [1.0]),
        ([1.0, 2.0], [[1.0, 0.0]], [[1.0], [2.0]]),
    ],
)
def test_minimal_shape_errors_are_reported(initial, matrix, values) -> None:
    with pytest.raises(ValueError):
        solve_curve(initial, LinearObservations.exact(matrix, values))


def test_basis_and_broadcast_shape_errors_are_reported() -> None:
    observations = LinearObservations.exact([[0.5, 0.5]], [100.0])
    with pytest.raises(ValueError, match="basis"):
        solve_curve([100.0, 100.0], observations, basis=np.eye(3))
    with pytest.raises(ValueError, match="monthly data"):
        solve_curve([100.0, 100.0], observations, floor=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="observation data"):
        solve_curve(
            [100.0, 100.0],
            LinearObservations(
                [[0.5, 0.5]],
                [100.0],
                lower=[90.0, 90.0],
                upper=[110.0],
            ),
        )


def test_negative_soft_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid values"):
        solve_curve(
            [100.0, 100.0],
            LinearObservations.soft([[0.5, 0.5]], [100.0], weight=-1.0),
        )


def test_nan_monthly_bound_is_rejected_instead_of_becoming_unbounded() -> None:
    with pytest.raises(ValueError, match="Invalid values"):
        solve_curve(
            [100.0, 100.0],
            LinearObservations.exact([[0.5, 0.5]], [100.0]),
            floor=[0.0, np.nan],
        )
