import random

import pytest
from sklearn.metrics import r2_score

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.geodesic_regression import GeodesicRegression
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import autograd_only
from geomstats.test_cases.learning.geodesic_regression import GeodesicRegressionTestCase

from .data.geodesic_regression import GeodesicRegressionTestData

# TODO: test loss decreases for `RiemannianGradientDescent`? bring callback
# TODO: add tests for initialization (keep it simple)


@pytest.fixture(
    scope="class",
    params=[
        (Euclidean(random.randint(3, 5)), "extrinsic"),
        (Euclidean(random.randint(3, 5)), "riemannian"),
        (Hypersphere(random.randint(3, 5)), "extrinsic"),
        (Hypersphere(random.randint(3, 5)), "riemannian"),
        (
            DiscreteCurvesStartingAtOrigin(
                ambient_dim=2,
                k_sampling_points=random.randint(5, 10),
            ),
            "extrinsic",
        ),
    ],
)
def estimators(request):
    space, method = request.param
    request.cls.estimator = GeodesicRegression(space, method=method)


@pytest.mark.slow
@pytest.mark.usefixtures("estimators")
class TestGeodesicRegression(
    GeodesicRegressionTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeodesicRegressionTestData()


@pytest.fixture(
    scope="class",
    params=[
        (SpecialEuclidean(n=2), "extrinsic"),
        (SpecialEuclidean(n=2), "riemannian"),
    ],
)
def estimators2(request):
    space, method = request.param
    request.cls.estimator = GeodesicRegression(space, method=method)


@autograd_only
@pytest.mark.slow
@pytest.mark.usefixtures("estimators2")
class TestGeodesicRegressionOnlyAutograd(
    GeodesicRegressionTestCase, metaclass=DataBasedParametrizer
):
    testing_data = GeodesicRegressionTestData()


@pytest.mark.parametrize("center_X", [False, True])
@pytest.mark.parametrize("weights", [None, [1.0, 1.0, 1.0, 1.0], [8.0, 1.0, 3.0, 2.0]])
def test_weighted_score_matches_euclidean_regression(center_X, weights):
    inputs = gs.array([1.0, 2.0, 3.0, 5.0])
    targets = gs.array([[1.0, -1.0], [2.0, 3.0], [4.0, 2.0], [5.0, 8.0]])
    weights = None if weights is None else gs.array(weights)
    estimator = GeodesicRegression(
        Euclidean(2),
        center_X=center_X,
        initialization=(gs.zeros(2), gs.zeros(2)),
        compute_training_score=True,
    ).fit(inputs, targets, weights=weights)

    expected = r2_score(
        gs.to_numpy(targets),
        gs.to_numpy(estimator.predict(inputs)),
        sample_weight=None if weights is None else gs.to_numpy(weights),
        multioutput="variance_weighted",
    )
    assert gs.isclose(estimator.score(inputs, targets, weights=weights), expected)
    assert gs.isclose(estimator.training_score_, expected)


@pytest.mark.parametrize("weights", [None, [8.0, 1.0, 3.0, 2.0]])
def test_weighted_score_on_circle(weights):
    angles = gs.array([0.1, 0.2, 0.7, 1.0])
    targets = gs.stack([gs.cos(angles), gs.sin(angles)], axis=-1)
    inputs = gs.array([0.0, 1.0, 2.0, 3.0])
    estimator = GeodesicRegression(Hypersphere(1), center_X=False)
    estimator.intercept_ = gs.array([gs.cos(0.3), gs.sin(0.3)])
    estimator.coef_ = 0.15 * gs.array([-gs.sin(0.3), gs.cos(0.3)])

    weights = None if weights is None else gs.array(weights)
    expected = r2_score(
        gs.to_numpy(angles),
        gs.to_numpy(0.3 + 0.15 * inputs),
        sample_weight=None if weights is None else gs.to_numpy(weights),
    )
    assert gs.isclose(estimator.score(inputs, targets, weights=weights), expected)
