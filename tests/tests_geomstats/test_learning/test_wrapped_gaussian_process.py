import math

import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.learning.wrapped_gaussian_process import WrappedGaussianProcess
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.wrapped_gaussian_process import (
    WrappedGaussianProcessTestCase,
)

from .data.wrapped_gaussian_process import WrappedGaussianProcessTestData


def _get_params():
    space = Hypersphere(dim=2)

    intercept_sphere_true = gs.array([0.0, -1.0, 0.0])
    coef_sphere_true = gs.array([1.0, 0.0, 0.5])

    prior = lambda x: space.metric.exp(
        x * coef_sphere_true,
        base_point=intercept_sphere_true,
    )

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))

    return space, prior, kernel


@pytest.fixture(
    scope="class",
    params=[
        _get_params(),
    ],
)
def estimators(request):
    space, prior, kernel = request.param
    request.cls.estimator = WrappedGaussianProcess(space, prior).set(kernel=kernel)


@pytest.mark.usefixtures("estimators")
class TestWrappedGaussianProcess(
    WrappedGaussianProcessTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = WrappedGaussianProcessTestData()


@pytest.mark.parametrize("space", [Euclidean(2), Matrices(2, 2)])
@pytest.mark.parametrize("uncertainty", [None, "std", "cov"])
def test_unfitted_predict_uses_every_tangent_coordinate(space, uncertainty):
    """Prior prediction has the point shape and one uncertainty per coordinate."""
    X = gs.array([[0.0], [0.5], [1.0]])

    def prior(X):
        return gs.reshape(X, (-1,) + (1,) * space.point_ndim) * gs.ones(space.shape)

    kernel = ConstantKernel(1.5) * RBF(0.7)
    estimator = WrappedGaussianProcess(space, prior).set(kernel=kernel)
    result = estimator.predict(
        X,
        return_tangent_std=uncertainty == "std",
        return_tangent_cov=uncertainty == "cov",
    )

    mean = result if uncertainty is None else result[0]
    assert mean.shape == (len(X), *space.shape)
    assert gs.allclose(mean, prior(X))
    if uncertainty is not None:
        expected = (
            gs.sqrt(gs.from_numpy(kernel.diag(gs.to_numpy(X))))
            if uncertainty == "std"
            else gs.from_numpy(kernel(gs.to_numpy(X)))
        )
        expected = gs.repeat(expected[..., None], math.prod(space.shape), axis=-1)
        assert result[1].shape == expected.shape
        assert gs.allclose(result[1], expected)


@pytest.mark.parametrize("space", [Euclidean(2), Matrices(2, 2)])
def test_unfitted_samples_match_independent_euclidean_process(space):
    """Prior samples retain independent tangent coordinates and sample axes."""
    X = gs.array([[0.0], [0.5], [1.0]])

    def prior(X):
        return gs.reshape(X, (-1,) + (1,) * space.point_ndim) * gs.ones(space.shape)

    kernel = ConstantKernel(1.5) * RBF(0.7)
    estimator = WrappedGaussianProcess(space, prior).set(kernel=kernel)
    actual = estimator.sample_y(X, n_samples=4, random_state=17)

    reference = GaussianProcessRegressor(
        kernel=kernel, n_targets=math.prod(space.shape)
    )
    tangent_samples = gs.from_numpy(reference.sample_y(gs.to_numpy(X), 4, 17))
    expected = prior(X)[..., None] + gs.reshape(
        tangent_samples, (len(X), *space.shape, 4)
    )
    assert actual.shape == expected.shape
    assert gs.allclose(actual, expected)
