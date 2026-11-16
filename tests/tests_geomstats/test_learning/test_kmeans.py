import random

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    ClusterMixinsTestCase,
)
from geomstats.test_cases.learning.kmeans import (
    AgainstFrechetMeanTestCase,
    ClusterInitializationTestCase,
)

from .data.kmeans import (
    AgainstFrechetMeanTestData,
    ClusterInitializationTestData,
    RiemannianKMeansTestData,
)


def _make_centroids(space, X, n_clusters):
    return space.random_von_mises_fisher(kappa=10, n_samples=n_clusters)


def _get_params():
    params = []
    spaces = (
        Hypersphere(dim=random.randint(3, 4)),
        SPDMatrices(n=random.randint(2, 4)),
    )

    init = ["kmeans++", "random"]
    for space in spaces:
        n_clusters = random.randint(2, 4)
        for init_ in init:
            params.append((space, n_clusters, init_))

    params.append(
        (
            spaces[0],
            params[0][1],
            lambda X, n_clusters: _make_centroids(spaces[0], X, n_clusters),
        )
    )

    return params


@pytest.fixture(
    scope="class",
    params=_get_params(),
)
def init_estimators(request):
    space, n_clusters, init = request.param
    request.cls.estimator = RiemannianKMeans(space, n_clusters=n_clusters, init=init)


@pytest.mark.usefixtures("init_estimators")
class TestClusterInitialization(
    ClusterInitializationTestCase, metaclass=DataBasedParametrizer
):
    testing_data = ClusterInitializationTestData()


@pytest.fixture(
    scope="class",
    params=[
        (Euclidean(dim=random.randint(3, 4)), random.randint(2, 4)),
        (Hypersphere(dim=random.randint(3, 4)), random.randint(2, 4)),
        (SPDMatrices(n=random.randint(2, 4)), random.randint(2, 4)),
    ],
)
def estimators(request):
    space, n_clusters = request.param
    request.cls.estimator = RiemannianKMeans(space, n_clusters=n_clusters)


@pytest.mark.usefixtures("estimators")
class TestRiemannianKMeans(
    ClusterMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = RiemannianKMeansTestData()


@pytest.fixture(
    scope="class",
    params=(
        Hypersphere(dim=random.randint(3, 4)),
        SPDMatrices(n=random.randint(2, 4)),
    ),
)
def estimators_one_cluster(request):
    space = request.param
    request.cls.estimator = RiemannianKMeans(space, n_clusters=1, tol=1e-4)
    request.cls.other_estimator = FrechetMean(space, method="adaptive")


@pytest.mark.usefixtures("estimators_one_cluster")
class TestAgainstFrechetMean(
    AgainstFrechetMeanTestCase, metaclass=DataBasedParametrizer
):
    testing_data = AgainstFrechetMeanTestData()


@pytest.mark.parametrize("space", [Euclidean(2), Matrices(2, 2)])
@pytest.mark.parametrize("dtype", [gs.int32, gs.int64, gs.float32, gs.float64])
@pytest.mark.parametrize("init", ["array", "callable", "random", "kmeans++"])
def test_kmeans_fractional_centers(space, dtype, init, monkeypatch):
    base = gs.reshape(gs.arange(space.dim), space.shape)
    data = gs.cast(gs.stack([base, base + 1, base + 10, base + 11]), dtype)
    original = gs.copy(data)
    centers = gs.stack([data[0], data[2]])
    if init == "array":
        init = centers
    elif init == "callable":
        init = lambda X, n_clusters: centers
    else:
        indices = iter([0, 2])
        monkeypatch.setattr(
            "geomstats.learning.kmeans.randint", lambda low, high: next(indices)
        )

    estimator = RiemannianKMeans(space, n_clusters=2, init=init).fit(data)

    expected = gs.stack([base + 0.5, base + 10.5])
    np.testing.assert_allclose(gs.to_numpy(estimator.cluster_centers_), expected)
    np.testing.assert_allclose(gs.to_numpy(estimator.inertia_), space.dim)
    np.testing.assert_array_equal(gs.to_numpy(estimator.labels_), [0, 0, 1, 1])
    np.testing.assert_array_equal(gs.to_numpy(data), gs.to_numpy(original))
    expected_dtype = dtype if gs.is_floating(data) else gs.get_default_dtype()
    assert estimator.cluster_centers_.dtype == expected_dtype
