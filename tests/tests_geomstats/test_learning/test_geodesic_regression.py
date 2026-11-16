import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import DiscreteCurvesStartingAtOrigin
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
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


@pytest.mark.parametrize("space", [Euclidean(2), Matrices(2, 2)])
@pytest.mark.parametrize("dtype", [gs.int32, gs.int64])
@pytest.mark.parametrize("center_X", [False, True])
def test_integer_predictors_match_floating_predictors(space, dtype, center_X):
    inputs = gs.array([1, 2, 3, 5], dtype=dtype)
    original_inputs = gs.copy(inputs)
    floating_inputs = gs.cast(inputs, gs.get_default_dtype())
    intercept = gs.reshape(gs.linspace(0.5, 1.5, space.dim), space.shape)
    coefficient = gs.reshape(gs.linspace(-0.3, 0.2, space.dim), space.shape)
    targets = intercept + gs.einsum("n,...->n...", floating_inputs, coefficient)
    kwargs = dict(
        center_X=center_X,
        initialization=(gs.zeros(space.shape), gs.zeros(space.shape)),
    )
    expected = GeodesicRegression(space, **kwargs).fit(floating_inputs, targets)
    actual = GeodesicRegression(space, **kwargs).fit(inputs, targets)

    assert gs.allclose(actual.predict(inputs), targets, atol=1e-4)
    assert gs.allclose(actual.predict(inputs), expected.predict(floating_inputs))
    assert gs.allclose(actual.coef_, expected.coef_)
    assert gs.allclose(actual.intercept_, expected.intercept_)
    assert gs.all(inputs == original_inputs)
    assert inputs.dtype == dtype


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
