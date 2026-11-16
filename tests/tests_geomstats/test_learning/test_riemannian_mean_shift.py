import random

import joblib
import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.riemannian_mean_shift import RiemannianMeanShift
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    ClusterMixinsTestCase,
)

from .data.riemannian_mean_shift import RiemannianMeanShiftTestData


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=2), 0.6, random.randint(2, 4)),
    ],
)
def estimators(request):
    space, bandwidth, n_clusters = request.param
    request.cls.estimator = RiemannianMeanShift(
        space,
        bandwidth,
        n_clusters=n_clusters,
    )


@pytest.mark.usefixtures("estimators")
class TestRiemannianMeanShift(
    ClusterMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = RiemannianMeanShiftTestData()


@pytest.mark.parametrize("space", [Euclidean(3), Hypersphere(2), SPDMatrices(2)])
@pytest.mark.parametrize("n_a, n_b", [(1, 1), (1, 4), (2, 3), (0, 3), (2, 0)])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_mean_shift_distance_rows(space, n_a, n_b, n_jobs):
    points = space.random_point(n_samples=6)
    points_a, points_b = points[:n_a], points[-n_b:] if n_b else points[:0]
    estimator = RiemannianMeanShift(space, bandwidth=1.0, n_jobs=n_jobs)
    expected = np.array(
        [
            [gs.to_numpy(space.metric.dist(first, second)) for second in points_b]
            for first in points_a
        ]
    ).reshape(n_a, n_b)

    with joblib.parallel_backend("threading"):
        result = estimator._dist_intersets(points_a, points_b)

    assert result.shape == (n_a, n_b)
    np.testing.assert_allclose(gs.to_numpy(result), expected, atol=1e-12)


def test_mean_shift_vectorized_distances_preserve_clusters(monkeypatch):
    data = gs.array([[0.0, 0.0], [0.2, 0.0], [3.0, 3.0], [3.2, 3.0]])
    estimator = RiemannianMeanShift(Euclidean(2), bandwidth=0.5, n_clusters=2)
    monkeypatch.setattr(estimator, "_initialization", lambda X: data[[0, 2]])

    estimator.fit(data)

    np.testing.assert_allclose(
        gs.to_numpy(estimator.cluster_centers_), [[0.1, 0.0], [3.1, 3.0]]
    )
    np.testing.assert_array_equal(gs.to_numpy(estimator.predict(data)), [0, 0, 1, 1])
