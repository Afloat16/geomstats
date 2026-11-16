import random

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.online_kmeans import OnlineKMeans
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    ClusterMixinsTestCase,
)

from .data.online_kmeans import OnlineKMeansTestData


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=random.randint(3, 4)), random.randint(2, 4)),
        (SPDMatrices(n=random.randint(2, 4)), random.randint(2, 4)),
    ],
)
def estimators(request):
    space, n_clusters = request.param
    request.cls.estimator = OnlineKMeans(space, n_clusters=n_clusters)


@pytest.mark.usefixtures("estimators")
class TestRiemannianKMeans(
    ClusterMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = OnlineKMeansTestData()


@pytest.mark.parametrize("space", [Euclidean(2), Matrices(2, 2)])
@pytest.mark.parametrize("dtype", [gs.int32, gs.int64, gs.float64])
def test_online_kmeans_fractional_center(space, dtype, monkeypatch):
    base = gs.reshape(gs.arange(space.dim), space.shape)
    data = gs.cast(gs.stack([base, base + 1]), dtype)
    original = gs.copy(data)
    indices = iter([gs.array([0]), gs.array([1])])
    monkeypatch.setattr(gs.random, "randint", lambda **kwargs: next(indices))

    estimator = OnlineKMeans(space, n_clusters=1, max_iter=1).fit(data)

    np.testing.assert_allclose(
        gs.to_numpy(estimator.cluster_centers_), gs.to_numpy((base + 0.5)[None])
    )
    np.testing.assert_array_equal(gs.to_numpy(data), gs.to_numpy(original))
    expected_dtype = dtype if gs.is_floating(data) else gs.get_default_dtype()
    assert estimator.cluster_centers_.dtype == expected_dtype
