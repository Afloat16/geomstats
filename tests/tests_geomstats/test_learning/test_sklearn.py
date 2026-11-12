import math

import pytest

import geomstats.backend as gs
from geomstats.learning._sklearn import PCA


@pytest.mark.parametrize("point_shape", [(3,), (4,), (2, 3)])
@pytest.mark.parametrize("fit_transform", [False, True])
def test_pca_refit_restores_the_current_point_shape(point_shape, fit_transform):
    """Reusing an adapter must discard the previous training-data shape."""
    estimator = PCA(n_components=2, svd_solver="full")
    previous = gs.reshape(gs.linspace(0.0, 1.0, 24), (6, 2, 2))
    estimator.fit(previous)
    values = gs.arange(6 * math.prod(point_shape), dtype=gs.float64)
    samples = gs.reshape(gs.sin(0.7 * values) + gs.cos(0.3 * values), (6, *point_shape))
    reference = PCA(n_components=2, svd_solver="full").fit(samples)

    if fit_transform:
        scores = estimator.fit_transform(samples)
    else:
        estimator.fit(samples)
        scores = estimator.transform(samples)
    reconstructed = estimator.inverse_transform(scores)

    assert reconstructed.shape == samples.shape
    assert gs.allclose(
        reconstructed, reference.inverse_transform(reference.transform(samples))
    )
    assert estimator.reshaped_components_.shape == (2, *point_shape)
    assert gs.allclose(estimator.reshaped_components_, reference.reshaped_components_)
