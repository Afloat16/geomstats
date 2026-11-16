import math

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.scalar_product_metric import (
    CustomizationTestCase,
    InstantiationTestCase,
    WrapperTestCase,
)

from .data.scalar_product_metric import (
    CustomizationTestData,
    InstantiationTestData,
    WrapperTestData,
)


class TestWrapper(WrapperTestCase, metaclass=DataBasedParametrizer):
    testing_data = WrapperTestData()


class TestInstantiation(InstantiationTestCase, metaclass=DataBasedParametrizer):
    space = Euclidean(dim=3)
    testing_data = InstantiationTestData()


class TestCustomization(CustomizationTestCase, metaclass=DataBasedParametrizer):
    Metric = EuclideanMetric
    space = Euclidean(dim=3)
    testing_data = CustomizationTestData()


@pytest.mark.parametrize("space_type", ["euclidean", "weighted_euclidean", "matrices"])
@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("nested", [False, True])
def test_scaled_flat_sharp_preserve_metric_duality(space_type, scale, nested):
    if space_type == "matrices":
        space = Matrices(2, 2)
    else:
        space = Euclidean(3)
        if space_type == "weighted_euclidean":
            space.equip_with_metric(
                EuclideanMetric,
                metric_matrix=gs.array(
                    [[2.0, 0.1, 0.0], [0.1, 3.0, 0.2], [0.0, 0.2, 4.0]]
                ),
            )

    space.equip_with_metric(ScalarProductMetric, scale=scale)
    if nested:
        space.equip_with_metric(ScalarProductMetric, scale=3.0)

    n_coordinates = math.prod(space.shape)
    vector = gs.reshape(gs.linspace(0.2, 1.1, n_coordinates), space.shape)
    covector = gs.reshape(gs.linspace(-0.5, 0.7, n_coordinates), space.shape)
    base_point = gs.zeros((2, 3) + space.shape)
    axes = tuple(range(-space.point_ndim, 0))

    flat = space.metric.flat(vector, base_point)
    sharp = space.metric.sharp(covector, base_point)

    assert flat.shape == base_point.shape
    assert sharp.shape == base_point.shape
    assert gs.allclose(
        gs.sum(flat * vector, axis=axes),
        space.metric.inner_product(vector, vector, base_point),
    )
    assert gs.allclose(
        space.metric.inner_product(sharp, vector, base_point),
        gs.sum(covector * vector, axis=axes),
    )
    assert gs.allclose(space.metric.sharp(space.metric.flat(vector)), vector)
    assert gs.allclose(space.metric.flat(space.metric.sharp(covector)), covector)
