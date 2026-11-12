import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.positive_reals import PositiveReals
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.positive_reals import PositiveRealsMetricTestData, PositiveRealsTestData


@pytest.mark.parametrize(
    "dtype_name", ["float32", "float64", "complex64", "complex128"]
)
@pytest.mark.parametrize("method", ["log", "dist", "squared_dist"])
@pytest.mark.parametrize("wide_range", [False, True])
def test_logarithmic_operations_without_unrepresentable_ratios(
    dtype_name, method, wide_range
):
    """Representable logarithms need not have representable intermediate ratios."""
    low_precision = dtype_name in ("float32", "complex64")
    exponent = 20 if low_precision else 200
    base_values = [10.0**exponent, 10.0**-exponent] if wide_range else [1.0, 1.0]
    point_values = base_values[::-1] if wide_range else [0.5, 2.0]
    base = gs.array(base_values, dtype=getattr(gs, dtype_name))[:, None]
    point = gs.array(point_values, dtype=getattr(gs, dtype_name))[:, None]
    magnitude = 2 * exponent * math.log(10) if wide_range else math.log(2)
    expected_log_ratio = np.array([-magnitude, magnitude])
    expected = {
        "log": np.array(base_values)[:, None] * expected_log_ratio[:, None],
        "dist": np.abs(expected_log_ratio),
        "squared_dist": expected_log_ratio**2,
    }[method]

    result = getattr(PositiveReals().metric, method)(point, base)

    np.testing.assert_allclose(
        gs.to_numpy(result), expected, rtol=5e-6 if low_precision else 1e-12, atol=0
    )


@pytest.mark.parametrize("dtype_name", ["float64", "complex128"])
@pytest.mark.parametrize("method", ["log", "dist", "squared_dist"])
def test_logarithmic_operations_preserve_nearby_point_accuracy(dtype_name, method):
    base_values = [1e-200, 1e200]
    point_values = [base * (1.0 + 1e-12) for base in base_values]
    with localcontext() as context:
        context.prec = 80
        ratios = np.array(
            [
                float((Decimal.from_float(point) / Decimal.from_float(base)).ln())
                for point, base in zip(point_values, base_values)
            ]
        )
    expected = {
        "log": np.array(base_values)[:, None] * ratios[:, None],
        "dist": np.abs(ratios),
        "squared_dist": ratios**2,
    }[method]
    point = gs.array(point_values, dtype=getattr(gs, dtype_name))[:, None]
    base = gs.array(base_values, dtype=getattr(gs, dtype_name))[:, None]

    result = getattr(PositiveReals().metric, method)(point, base)

    np.testing.assert_allclose(gs.to_numpy(result), expected, rtol=1e-12, atol=0)


@autodiff_only
@pytest.mark.parametrize("method", ["dist", "squared_dist"])
@pytest.mark.parametrize(
    "point_value,base_value", [(1e-200, 1e200), (1e200, 1e-200), (1.0 + 1e-12, 1.0)]
)
def test_logarithmic_distance_gradient(point_value, base_value, method):
    metric = PositiveReals().metric
    point = gs.array([point_value], dtype=gs.float64)
    base = gs.array([base_value], dtype=gs.float64)
    function = lambda value: getattr(metric, method)(value, base)

    _, gradient = gs.autodiff.value_and_grad(function)(point)

    log_ratio = math.log(point_value) - math.log(base_value)
    expected = (
        math.copysign(1.0, log_ratio) / point_value
        if method == "dist"
        else 2 * log_ratio / point_value
    )
    np.testing.assert_allclose(gs.to_numpy(gradient), [expected], rtol=1e-12, atol=0)


class TestPositiveReals(VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer):
    space = PositiveReals(equip=False)
    testing_data = PositiveRealsTestData()


class TestPositiveRealsMetric(
    RiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveReals()
    testing_data = PositiveRealsMetricTestData()
