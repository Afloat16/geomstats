import math

import pytest

import geomstats.backend as gs
from geomstats.information_geometry.gamma import (
    GammaDistributions,
    GammaDistributionsRandomVariable,
    NaturalToStandardDiffeo,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import GammaRandomDataGenerator, RandomDataGenerator
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.information_geometry.gamma import (
    GammaDistributionsTestCase,
    GammaMetricTestCase,
    StandardGammaDistributions,
)

from .data.gamma import (
    GammaDistributionsSmokeTestData,
    GammaDistributionsTestData,
    GammaMetricTestData,
    NaturalToStandardDiffeoTestData,
)


class TestNaturalToStandardDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    space = GammaDistributions(equip=False)
    image_space = StandardGammaDistributions()
    diffeo = NaturalToStandardDiffeo()
    testing_data = NaturalToStandardDiffeoTestData()


class TestGammaDistributions(
    GammaDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = GammaDistributions(equip=False)
    data_generator = GammaRandomDataGenerator(space)
    random_variable = GammaDistributionsRandomVariable(space)

    testing_data = GammaDistributionsTestData()


@pytest.mark.smoke
class TestGammaDistributionsSmoke(
    GammaDistributionsTestCase, metaclass=DataBasedParametrizer
):
    space = GammaDistributions(equip=False)
    testing_data = GammaDistributionsSmokeTestData()


class TestGammaMetric(GammaMetricTestCase, metaclass=DataBasedParametrizer):
    space = GammaDistributions()
    data_generator = RandomDataGenerator(space, amplitude=20.0)
    testing_data = GammaMetricTestData()


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3), (2, 2), (2, 1, 3)])
def test_jacobian_christoffels_multiple_batch_axes(batch_shape):
    metric = GammaDistributions().metric
    n_points = math.prod(batch_shape)
    points = gs.reshape(gs.linspace(0.8, 3.1, 2 * n_points), batch_shape + (2,))
    actual = metric.jacobian_christoffels(points)
    assert actual.shape == batch_shape + (2, 2, 2, 2)

    step = 1e-5
    expected = gs.stack(
        [
            (
                metric.christoffels(points + step * direction)
                - metric.christoffels(points - step * direction)
            )
            / (2 * step)
            for direction in gs.eye(2)
        ],
        axis=-1,
    )
    assert gs.allclose(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("batch_shape", [(2, 3), (2, 2), (2, 1, 3)])
def test_gamma_curvature_multiple_batch_axes(batch_shape):
    metric = GammaDistributions().metric
    flat_points = gs.reshape(gs.linspace(0.8, 3.1, 2 * math.prod(batch_shape)), (-1, 2))
    points = gs.reshape(flat_points, batch_shape + (2,))
    actual = metric.riemann_tensor(points)
    expected = gs.reshape(
        gs.stack([metric.riemann_tensor(point) for point in flat_points]),
        batch_shape + (2, 2, 2, 2),
    )
    assert actual.shape == expected.shape
    assert gs.allclose(actual, expected, atol=1e-7, rtol=1e-6)
