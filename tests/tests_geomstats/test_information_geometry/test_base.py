import math

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.information_geometry.dirichlet import DirichletDistributions
from geomstats.information_geometry.multinomial import MultinomialDistributions
from geomstats.information_geometry.normal import (
    CenteredNormalDistributions,
    DiagonalNormalDistributions,
    GeneralNormalDistributions,
)


@pytest.fixture(
    params=[
        (DirichletDistributions(3, equip=False), [2.0, 3.0, 4.0]),
        (MultinomialDistributions(2, 5, equip=False), [0.2, 0.3, 0.5]),
        (CenteredNormalDistributions(1, equip=False), [[1.5]]),
        (CenteredNormalDistributions(2, equip=False), [[1.5, 0.2], [0.2, 2.0]]),
        (DiagonalNormalDistributions(1, equip=False), [0.3, 1.5]),
        (DiagonalNormalDistributions(2, equip=False), [0.3, -0.4, 1.5, 2.0]),
        (GeneralNormalDistributions(1, equip=False), [0.3, 1.5]),
        (GeneralNormalDistributions(2, equip=False), [0.3, -0.4, 1.5, 0.2, 0.2, 2.0]),
    ]
)
def multivariate_distribution(request):
    space, point = request.param
    return space, gs.array(point)


def _batched_parameters(space, point, batch_shape):
    n_points = math.prod(batch_shape)
    points = gs.stack([point * (1.0 + 0.1 * index) for index in range(n_points)])
    if isinstance(space, MultinomialDistributions):
        points = points + gs.linspace(0.0, 0.25, n_points)[:, None] * gs.array(
            [1.0, -1.0, 0.0]
        )
        points = points / gs.sum(points, axis=-1, keepdims=True)
    return gs.reshape(points, batch_shape + space.shape)


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("n_samples", [1, 4])
def test_multivariate_sample_preserves_all_axes(
    multivariate_distribution, batch_shape, n_samples
):
    space, point = multivariate_distribution
    points = _batched_parameters(space, point, batch_shape)
    flat_points = gs.reshape(points, (-1,) + space.shape)
    expected_shape = batch_shape + (n_samples,) + space.support_shape

    np.random.seed(71)
    expected = np.stack(
        [
            np.asarray(space._scp_rv.scp_rvs(point_, size=n_samples)).reshape(
                (n_samples,) + space.support_shape
            )
            for point_ in flat_points
        ]
    ).reshape(expected_shape)
    np.random.seed(71)
    actual = space.sample(points, n_samples=n_samples)

    assert actual.shape == expected_shape
    np.testing.assert_allclose(gs.to_numpy(actual), expected)


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("n_samples", [1, 4])
def test_multivariate_pdf_preserves_all_batch_axes(
    multivariate_distribution, batch_shape, n_samples
):
    space, point = multivariate_distribution
    points = _batched_parameters(space, point, batch_shape)
    flat_points = gs.reshape(points, (-1,) + space.shape)
    np.random.seed(72)
    samples = gs.array(
        np.asarray(space._scp_rv.scp_rvs(point, size=n_samples)).reshape(
            (n_samples,) + space.support_shape
        )
    )
    expected_shape = batch_shape + (n_samples,)
    expected = np.stack(
        [
            np.asarray(space._scp_rv.scp_pdf(samples, point_)).reshape((n_samples,))
            for point_ in flat_points
        ]
    ).reshape(expected_shape)

    actual = space._scp_rv.pdf(samples, points)

    assert actual.shape == expected_shape
    np.testing.assert_allclose(gs.to_numpy(actual), expected)
