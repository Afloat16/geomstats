import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.wrapped_gaussian_process import WrappedGaussianProcess


@pytest.mark.parametrize(
    "space",
    [Euclidean(1), Matrices(1, 1), SPDMatrices(1), Euclidean(2), Matrices(2, 2)],
)
@pytest.mark.parametrize("n_samples", [1, 4])
@pytest.mark.parametrize("n_queries", [1, 3])
def test_posterior_samples_preserve_point_axes(space, n_samples, n_queries):
    train = gs.array([[-1.0], [0.0], [1.0]])
    queries = gs.reshape(gs.linspace(-0.7, 0.6, n_queries), (-1, 1))

    def prior(values):
        return 2.0 + 0.2 * gs.reshape(
            values, (-1,) + (1,) * space.point_ndim
        ) * gs.ones(space.shape)

    tangent = gs.reshape(0.2 + 0.1 * train, (-1,) + (1,) * space.point_ndim) * gs.ones(
        space.shape
    )
    targets = space.metric.exp(tangent, prior(train))
    estimator = WrappedGaussianProcess(space, prior).set(
        kernel=RBF(0.7), alpha=1e-8, optimizer=None
    )
    estimator.fit(train, targets)
    reference = GaussianProcessRegressor(kernel=RBF(0.7), alpha=1e-8, optimizer=None)
    reference.fit(gs.to_numpy(train), gs.to_numpy(tangent).reshape(len(train), -1))
    tangent_samples = reference.sample_y(
        gs.to_numpy(queries), n_samples, random_state=7
    )
    tangent_samples = tangent_samples.reshape(n_queries, *space.shape, n_samples)
    base = gs.to_numpy(prior(queries))[..., None]
    expected = (
        base * np.exp(tangent_samples / base)
        if isinstance(space, SPDMatrices)
        else base + tangent_samples
    )

    result = estimator.sample_y(queries, n_samples=n_samples, random_state=7)

    assert result.shape == (n_queries, *space.shape, n_samples)
    np.testing.assert_allclose(gs.to_numpy(result), expected, rtol=1e-8, atol=1e-10)
