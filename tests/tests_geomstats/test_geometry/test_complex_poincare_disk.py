import pytest

import geomstats.backend as gs
from geomstats.geometry.complex_poincare_disk import ComplexPoincareDisk
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import np_only
from geomstats.test_cases.geometry.base import ComplexVectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)

from .data.complex_poincare_disk import (
    ComplexPoincareDiskMetricTestData,
    ComplexPoincareDiskTestData,
)


class TestComplexPoincareDisk(
    ComplexVectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = ComplexPoincareDisk(equip=False)
    testing_data = ComplexPoincareDiskTestData()


@np_only
class TestComplexPoincareDiskMetric(
    ComplexRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = ComplexPoincareDisk()
    testing_data = ComplexPoincareDiskMetricTestData()


@pytest.mark.parametrize("bound", [0.0, 0.2, 0.8, 0.95])
@pytest.mark.parametrize("n_samples", [1, 10000])
def test_random_point_respects_modulus_bound(bound, n_samples):
    gs.random.seed(32)
    space = ComplexPoincareDisk()
    points = space.random_point(n_samples=n_samples, bound=bound)

    expected_shape = (1,) if n_samples == 1 else (n_samples, 1)
    assert points.shape == expected_shape
    assert gs.is_complex(points)
    assert gs.all(gs.abs(points) <= bound)
    assert gs.all(space.belongs(points))
