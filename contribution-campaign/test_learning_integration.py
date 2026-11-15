"""Integration checks requiring both matrix coordinates and whitening."""

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.learning.pca import TangentPCA


@pytest.mark.parametrize("symmetric", [False, True])
def test_matrix_coordinate_round_trip_with_whitening(symmetric):
    """Preserve both matrix coordinates and whitened component variance."""
    rng = np.random.RandomState(29)
    if symmetric:
        samples = rng.normal(size=(12, 2, 2))
        samples = (samples + samples.transpose(0, 2, 1)) / 2
        shape = (2, 2)
        components = 3
    else:
        samples = rng.normal(size=(12, 2, 3))
        shape = (2, 3)
        components = 6
    space = Matrices(*shape)
    model = TangentPCA(space, n_components=components, whiten=True)
    values = gs.array(samples)

    scores = model.fit_transform(values, base_point=gs.zeros(shape))

    np.testing.assert_allclose(
        np.cov(gs.to_numpy(scores), rowvar=False), np.eye(components), atol=1e-10
    )
    np.testing.assert_allclose(
        gs.to_numpy(model.transform(values)), gs.to_numpy(scores), atol=1e-10
    )
    np.testing.assert_allclose(
        gs.to_numpy(model.inverse_transform(scores)), samples, atol=1e-10
    )
