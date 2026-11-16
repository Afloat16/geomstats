import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.test.test_case import autodiff_only


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("scaled", [False, True])
def test_squared_dist_known_angles(dtype_name, scaled):
    dtype = getattr(gs, dtype_name)
    angles = gs.array(
        [[0.0, 1e-8, 1e-5, 0.003, 0.008], [0.012, 0.1, 1.0, gs.pi - 0.01, gs.pi]],
        dtype=dtype,
    )
    points = gs.stack([gs.cos(angles), gs.sin(angles), gs.zeros_like(angles)], axis=-1)
    base = gs.array([1.0, 0.0, 0.0], dtype=dtype)
    if scaled:
        points = 2.3 * points
        base = 0.7 * base

    result = Hypersphere(2).metric.squared_dist(points, base)

    assert result.shape == angles.shape
    tolerance = 5e-5 if dtype_name == "float32" else 1e-10
    np.testing.assert_allclose(
        gs.to_numpy(result), gs.to_numpy(angles**2), rtol=tolerance, atol=1e-25
    )


@autodiff_only
@pytest.mark.parametrize("angle", [0.0, 1e-10, 0.001, 0.008, 0.012, 0.4])
def test_squared_dist_gradient_and_hessian(angle):
    space = Hypersphere(2)
    point = gs.array([1.0, 0.0, 0.0])

    def objective(parameter):
        other = gs.concatenate(
            [gs.cos(parameter), gs.sin(parameter), gs.zeros_like(parameter)]
        )
        return space.metric.squared_dist(point, other)

    parameter = gs.array([angle])
    value, gradient = gs.autodiff.value_and_grad(objective)(parameter)
    hessian = gs.autodiff.hessian(objective)(parameter)

    np.testing.assert_allclose(gs.to_numpy(value), angle**2, rtol=1e-9, atol=1e-25)
    np.testing.assert_allclose(
        gs.to_numpy(gradient), [2 * angle], rtol=1e-8, atol=1e-12
    )
    np.testing.assert_allclose(gs.to_numpy(hessian), [[2.0]], rtol=1e-8, atol=1e-10)


@autodiff_only
def test_squared_dist_batched_gradient_with_matching_observation():
    space = Hypersphere(2)
    angles = gs.array([[0.0, 0.2, 0.4], [-0.3, 0.0, 0.1]])
    points = gs.stack([gs.cos(angles), gs.sin(angles), gs.zeros_like(angles)], axis=-1)
    base = gs.array([1.0, 0.0, 0.0])

    def objective(parameter):
        return gs.sum(space.metric.squared_dist(parameter, points))

    _, gradient = gs.autodiff.value_and_grad(objective)(base)

    expected = gs.array([0.0, -2 * gs.sum(angles), 0.0])
    np.testing.assert_allclose(gs.to_numpy(gradient), gs.to_numpy(expected), atol=1e-10)
