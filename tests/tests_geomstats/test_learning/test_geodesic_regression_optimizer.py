import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.learning.geodesic_regression import (
    GeodesicRegression,
    RiemannianGradientDescent,
)
from geomstats.test.test_case import autodiff_only


@pytest.mark.parametrize("space", [Euclidean(2), Matrices(2, 2)])
@pytest.mark.parametrize("max_iter", [1, 2, 5])
@pytest.mark.parametrize("step_size", [0.1, 1.0, 10.0])
def test_gradient_descent_returns_best_evaluated_point(
    space, max_iter, step_size, monkeypatch
):
    initial = gs.reshape(gs.arange(1.0, 2 * space.dim + 1), (2, space.dim))
    weights = gs.reshape(1.0 + gs.arange(2 * space.dim) % 2, initial.shape)
    objective = lambda point: 0.5 * gs.sum(weights * point**2)
    evaluated = []

    def value_and_grad(point):
        value = objective(point)
        evaluated.append(float(gs.to_numpy(value)))
        return value, weights * point

    optimizer = RiemannianGradientDescent(
        max_iter=max_iter, init_step_size=step_size, tol=1e-12
    )
    monkeypatch.setattr(
        optimizer, "_handle_jac", lambda fun, point_ndim: value_and_grad
    )

    result = optimizer.minimize(space, objective, initial)

    assert result.x.shape == initial.shape
    np.testing.assert_allclose(
        gs.to_numpy(result.fun), gs.to_numpy(objective(result.x))
    )
    np.testing.assert_allclose(gs.to_numpy(result.fun), min(evaluated))
    assert result.fun <= objective(initial)


@pytest.mark.parametrize("space", [Euclidean(2), Matrices(2, 2)])
@pytest.mark.parametrize("step_size", [1.0, 10.0])
def test_gradient_descent_recovers_after_rejected_steps(space, step_size, monkeypatch):
    initial = gs.ones((2, space.dim))
    objective = lambda point: gs.sum(point**2)
    optimizer = RiemannianGradientDescent(
        max_iter=100, init_step_size=step_size, tol=1e-12
    )
    monkeypatch.setattr(
        optimizer,
        "_handle_jac",
        lambda fun, point_ndim: lambda point: (fun(point), 2 * point),
    )

    result = optimizer.minimize(space, objective, initial)

    np.testing.assert_allclose(gs.to_numpy(result.x), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        gs.to_numpy(result.fun), gs.to_numpy(objective(result.x))
    )


@autodiff_only
@pytest.mark.parametrize("space", [Euclidean(2), Matrices(2, 2)])
def test_riemannian_regression_with_backtracking(space):
    times = gs.array([-1.0, 0.0, 1.0])
    intercept = gs.reshape(gs.arange(1.0, space.dim + 1), space.shape)
    coefficient = 0.2 * gs.ones(space.shape)
    data = intercept + gs.reshape(times, (-1,) + (1,) * space.point_ndim) * coefficient
    estimator = GeodesicRegression(
        space,
        method="riemannian",
        center_X=False,
        initialization=(gs.zeros(space.shape), gs.zeros(space.shape)),
    )
    estimator.optimizer = RiemannianGradientDescent(
        max_iter=200, init_step_size=1.0, tol=1e-12
    )

    estimator.fit(times, data)

    np.testing.assert_allclose(
        gs.to_numpy(estimator.predict(times)), gs.to_numpy(data), atol=1e-5
    )
