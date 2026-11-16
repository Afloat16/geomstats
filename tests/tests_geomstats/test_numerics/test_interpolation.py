import random

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.numerics.interpolation import (
    LinearInterpolator1D,
    UniformUnitIntervalLinearInterpolator,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.numerics.interpolation import InterpolatorTestCase

from .data.interpolation import InterpolatorTestData


@pytest.fixture(
    scope="class",
    params=[((1,), True), ((2,), True), ((1,), False), ((2,), False)],
)
def interpolators(request):
    point_shape, uniform = request.param

    num_points = random.randint(5, 12)

    data = request.cls.data = gs.random.uniform(size=(num_points,) + point_shape)

    if uniform:
        request.cls.times = gs.linspace(0.0, 1.0, num=num_points)
        request.cls.interpolator = UniformUnitIntervalLinearInterpolator(
            data, point_ndim=len(point_shape)
        )
    else:
        times = request.cls.times = gs.sort(
            gs.random.uniform(0.0, 1.0, size=(num_points,))
        )
        request.cls.interpolator = LinearInterpolator1D(
            times, data, point_ndim=len(point_shape)
        )


@pytest.mark.usefixtures("interpolators")
class TestLinearInterpolator(InterpolatorTestCase, metaclass=DataBasedParametrizer):
    testing_data = InterpolatorTestData()


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("point_shape", [(2,), (2, 2)])
@pytest.mark.parametrize("input_kind", ["array", "list", "tuple"])
@pytest.mark.parametrize("query_kind", ["array", "list", "tuple", "scalar"])
def test_interpolate_array_like_inputs(uniform, point_shape, input_kind, query_kind):
    times = np.linspace(0.0, 1.0, 4) if uniform else np.array([0.0, 0.2, 0.65, 1.0])
    n_coordinates = int(np.prod(point_shape))
    data = (np.arange(2 * 4 * n_coordinates, dtype=float) / 5) ** 2
    data = data.reshape((2, 4) + point_shape)
    query = (
        np.array([0.4])
        if query_kind == "scalar"
        else np.array([0.0, 0.1, 0.4, 0.8, 1.0])
    )

    def convert(array, kind):
        if kind == "list":
            return array.tolist()
        if kind == "tuple":
            return tuple(array.tolist())
        if kind == "scalar":
            return float(array[0])
        return gs.array(array)

    if uniform:
        interpolator = UniformUnitIntervalLinearInterpolator(
            convert(data, input_kind), point_ndim=len(point_shape)
        )
    else:
        interpolator = LinearInterpolator1D(
            convert(times, input_kind),
            convert(data, input_kind),
            point_ndim=len(point_shape),
        )

    flattened = data.reshape((2, 4, n_coordinates))
    expected = np.stack(
        [
            np.stack(
                [
                    np.interp(query, times, values[:, coordinate])
                    for coordinate in range(n_coordinates)
                ],
                axis=-1,
            )
            for values in flattened
        ]
    ).reshape((2, len(query)) + point_shape)
    actual = interpolator(convert(query, query_kind))

    assert actual.shape == expected.shape
    np.testing.assert_allclose(gs.to_numpy(actual), expected)
