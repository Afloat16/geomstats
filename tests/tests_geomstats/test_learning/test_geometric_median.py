import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.learning.geometric_median import GeometricMedian
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import TestCase
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    MeanEstimatorMixinsTestCase,
)

from .data.geometric_median import GeometricMedianFitTestData, GeometricMedianTestData


@pytest.mark.parametrize("matrix_points", [False, True])
@pytest.mark.parametrize("start_at_median", [False, True])
def test_weighted_median_at_a_sample(matrix_points, start_at_median):
    """A point with more than half the total weight is the unique median."""
    if matrix_points:
        space = SPDMatrices(2)
        samples = gs.array([[[1.0, 0.0], [0.0, 1.0]], [[4.0, 0.0], [0.0, 9.0]]])
    else:
        space = Euclidean(2)
        samples = gs.array([[0.0, 0.0], [2.0, 3.0]])
    estimator = GeometricMedian(
        space,
        init_point=samples[0] if start_at_median else None,
        max_iter=1 if start_at_median else 100,
    )

    estimate = estimator.fit(samples, weights=gs.array([0.9, 0.1])).estimate_

    assert space.metric.dist(estimate, samples[0]) < 1e-5


@pytest.mark.parametrize("weights", [[0.2, 0.4, 0.4], [0.0, 0.5, 0.5]])
def test_coincident_samples_keep_their_total_weight(weights):
    """Coincident observations must contribute to the optimality condition."""
    samples = gs.array([[0.0, 0.0], [2.0, 1.0], [2.0, 1.0]])
    estimator = GeometricMedian(Euclidean(2), max_iter=1)

    estimate = estimator.fit(samples, weights=gs.array(weights)).estimate_

    assert gs.allclose(estimate, samples[-1])


def test_noncollinear_optimal_sample_stays_fixed():
    samples = gs.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    estimator = GeometricMedian(Euclidean(2), init_point=samples[0], max_iter=1)

    estimate = estimator.fit(samples, weights=gs.array([0.8, 0.1, 0.1])).estimate_

    assert gs.allclose(estimate, samples[0])


def test_coincident_nonoptimal_sample_reduces_the_weiszfeld_step():
    samples = gs.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    estimator = GeometricMedian(Euclidean(2), init_point=samples[0], max_iter=1)

    estimate = estimator.fit(samples, weights=gs.array([0.2, 0.4, 0.4])).estimate_

    residual_norm = (0.4**2 + 0.4**2) ** 0.5
    expected_step = (1.0 - 0.2 / residual_norm) * 0.4 / (0.4 / 2.0 + 0.4 / 3.0)
    assert gs.allclose(estimate, gs.array([expected_step, expected_step]))


@pytest.fixture(
    scope="class",
    params=[
        Euclidean(random.randint(3, 5)),
        Hyperboloid(random.randint(3, 5)),
        Hypersphere(random.randint(3, 5)),
        SPDMatrices(random.randint(3, 5), equip=False).equip_with_metric(
            SPDAffineMetric
        ),
        SPDMatrices(random.randint(3, 5), equip=False).equip_with_metric(
            SPDLogEuclideanMetric
        ),
    ],
)
def estimators(request):
    space = request.param
    request.cls.estimator = GeometricMedian(space)


@pytest.mark.usefixtures("estimators")
class TestGeometricMedian(
    MeanEstimatorMixinsTestCase, BaseEstimatorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeometricMedianTestData()


@pytest.mark.smoke
class TestGeometricMedianFit(TestCase, metaclass=DataBasedParametrizer):
    testing_data = GeometricMedianFitTestData()

    def test_fit(self, estimator, X, expected, atol):
        estimate = estimator.fit(X).estimate_
        self.assertAllClose(estimate, expected, atol=atol)
