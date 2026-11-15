"""Regression tests for per-batch Frechet weight normalization."""

import numpy as np
import pytest
from scipy.special import softmax

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.expectation_maximization import RiemannianEM
from geomstats.learning.frechet_mean import FrechetMean


def _points_and_reference(kind, weights):
    times = np.array([-0.6, -0.2, 0.3, 0.8])[:, None]
    times = times + np.array([-0.1, 0.0, 0.1])[None, :]
    if kind == "poincare":
        space = PoincareBall(2)
        points = np.stack((np.tanh(times / 2), np.zeros_like(times)), axis=-1)
        averages = np.array(
            [np.average(times[:, k], weights=weights[:, k]) for k in range(3)]
        )
        expected = np.stack((np.tanh(averages / 2), np.zeros_like(averages)), axis=-1)
    else:
        if kind == "euclidean":
            space = Euclidean(2)
            points = np.stack((times, times**2), axis=-1)
        else:
            space = Matrices(2, 2)
            points = np.stack(
                (times, 2 * times + 1, times**2, -times), axis=-1
            ).reshape(4, 3, 2, 2)
        expected = np.stack(
            [np.average(points[:, k], axis=0, weights=weights[:, k]) for k in range(3)]
        )
    return space, gs.array(points), expected


@pytest.mark.parametrize("kind", ["euclidean", "matrices", "poincare"])
@pytest.mark.parametrize(
    "scales", [[1.0, 1.0, 1.0], [1e-6, 1.0, 1e6], [1.0, 100.0, 0.01]]
)
def test_batch_means_are_invariant_to_independent_weight_scales(kind, scales):
    weights = np.array([[1, 2, 0], [2, 1, 1], [1, 0, 2], [0, 1, 1]], dtype=float)
    weights *= scales
    space, points, expected = _points_and_reference(kind, weights)
    input_weights = gs.array(weights)
    original_points = gs.copy(points)
    estimator = FrechetMean(space, method="batch", optimizer_kwargs={"epsilon": 1e-12})

    assert estimator.fit(points, weights=input_weights) is estimator

    np.testing.assert_allclose(
        gs.to_numpy(estimator.estimate_), expected, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_array_equal(gs.to_numpy(input_weights), weights)
    np.testing.assert_array_equal(gs.to_numpy(points), gs.to_numpy(original_points))


@pytest.mark.parametrize("kind", ["euclidean", "matrices", "poincare"])
def test_zero_weight_batch_retains_its_initial_estimate(kind):
    weights = np.array([[1, 2, 0], [2, 1, 1], [1, 0, 2], [0, 1, 1]], dtype=float)
    space, points, expected = _points_and_reference(kind, weights)
    weights[:, 1] = 0
    expected[1] = gs.to_numpy(points[0, 1])

    estimator = FrechetMean(
        space, method="batch", optimizer_kwargs={"epsilon": 1e-12}
    ).fit(points, weights=gs.array(weights))

    np.testing.assert_allclose(
        gs.to_numpy(estimator.estimate_), expected, rtol=1e-7, atol=1e-7
    )


def test_em_updates_low_mass_component_mean_from_all_its_weights():
    times = np.array([-0.6, -0.2, 0.4])
    centers = np.array([-0.3, 2.0])

    def diameter(values):
        return gs.array(np.stack((np.tanh(values / 2), np.zeros_like(values)), axis=-1))

    class InitializedEM(RiemannianEM):
        def _initialization(self, X):
            return diameter(centers), gs.array([0.5, 0.5])

    logits = -((times[:, None] - centers[None, :]) ** 2) / (2 * 0.5**2)
    weights = softmax(logits, axis=1)
    expected_times = np.array(
        [np.average(times, weights=weights[:, k]) for k in range(2)]
    )
    estimator = InitializedEM(PoincareBall(2), n_gaussians=2, max_iter=1)

    estimator.fit(diameter(times))

    actual_times = 2 * np.arctanh(gs.to_numpy(estimator.means_)[:, 0])
    np.testing.assert_allclose(actual_times, expected_times, atol=1e-6)
