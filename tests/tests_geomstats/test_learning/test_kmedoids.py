import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.kmedoids import RiemannianKMedoids
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    ClusterMixinsTestCase,
)

from .data.kmedoids import RiemannianKMedoidsTestData


@pytest.mark.parametrize("n_clusters", [1, 32])
def test_random_initialization_keeps_all_clusters(n_clusters):
    """Distinct samples can each become a medoid without empty clusters."""
    samples = gs.linspace(0.0, 31.0, 32)[:, None]
    estimator = RiemannianKMedoids(Euclidean(1), n_clusters=n_clusters)

    estimator.fit(samples)

    assert len(gs.unique(estimator.medoid_indices_)) == n_clusters
    assert len(gs.unique(estimator.labels_)) == n_clusters
    assert gs.all(estimator.cluster_centers_ == samples[estimator.medoid_indices_])
    assert gs.all(estimator.predict(samples) == estimator.labels_)


def test_random_initialization_rejects_too_many_clusters():
    """A medoid must be selected from a different training sample."""
    estimator = RiemannianKMedoids(Euclidean(1), n_clusters=5)

    with pytest.raises(ValueError):
        estimator.fit(gs.array([[0.0], [1.0], [2.0], [3.0]]))


@pytest.fixture(
    scope="class",
    params=[
        (Hypersphere(dim=random.randint(3, 4)), random.randint(2, 4)),
        (SPDMatrices(n=random.randint(2, 4)), random.randint(2, 4)),
    ],
)
def estimators(request):
    space, n_clusters = request.param
    request.cls.estimator = RiemannianKMedoids(space, n_clusters=n_clusters)


@pytest.mark.usefixtures("estimators")
class TestRiemannianKMedoids(
    ClusterMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = RiemannianKMedoidsTestData()
