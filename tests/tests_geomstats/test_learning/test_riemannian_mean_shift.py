import random

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.stiefel import Stiefel
from geomstats.learning.riemannian_mean_shift import RiemannianMeanShift
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    ClusterMixinsTestCase,
)

from .data.riemannian_mean_shift import RiemannianMeanShiftTestData


@pytest.mark.parametrize("space", [Hypersphere(2), Stiefel(3, 2)])
@pytest.mark.parametrize("n_clusters", [1, 3])
def test_uniform_mean_shift_center_shape(space, n_clusters):
    estimator = RiemannianMeanShift(
        space, bandwidth=1.0, n_clusters=n_clusters, init_centers="random_uniform"
    )

    centers = estimator._initialization(space.random_point(n_samples=5))

    assert centers.shape == (n_clusters,) + space.shape
    assert gs.all(space.belongs(centers))


@pytest.mark.parametrize("dim", [2, 4])
def test_uniform_mean_shift_one_cluster(dim, monkeypatch):
    space = Hypersphere(dim)
    center = gs.array([1.0] + [0.0] * dim)
    tangent = gs.array([0.0, 0.2] + [0.0] * (dim - 1))
    data = space.metric.exp(gs.stack([tangent, -tangent]), center)
    monkeypatch.setattr(space, "random_uniform", lambda n_samples: center)

    estimator = RiemannianMeanShift(
        space, bandwidth=1.0, init_centers="random_uniform"
    ).fit(data)

    assert estimator.cluster_centers_.shape == (1, dim + 1)
    np.testing.assert_allclose(
        gs.to_numpy(estimator.cluster_centers_[0]), gs.to_numpy(center), atol=1e-8
    )
    np.testing.assert_array_equal(gs.to_numpy(estimator.predict(data)), [0, 0])


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
