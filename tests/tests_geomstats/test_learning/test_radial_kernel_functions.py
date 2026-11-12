from decimal import Decimal, localcontext

import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.learning.radial_kernel_functions import (
    logistic_radial_kernel,
    sigmoid_radial_kernel,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.learning.radial_kernel_functions import (
    RadialKernelFunctionsTestCase,
)

from .data.radial_kernel_functions import RadialKernelFunctionsTestData


@pytest.mark.parametrize("kernel", [logistic_radial_kernel, sigmoid_radial_kernel])
@pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0])
def test_exponential_kernel_preserves_representable_tail(kernel, bandwidth):
    scaled_distances = [0.0, 1.0, 20.0, 720.0]
    with localcontext() as context:
        context.prec = 80
        expected = []
        for value in scaled_distances:
            distance = Decimal.from_float(value)
            denominator = distance.exp() + (-distance).exp()
            if kernel is logistic_radial_kernel:
                denominator += 2
            expected.append(float(1 / denominator))
    distances = gs.array(scaled_distances, dtype=gs.float64) * bandwidth

    result = kernel(distances, bandwidth=bandwidth)

    np.testing.assert_allclose(gs.to_numpy(result), expected, rtol=1e-12, atol=0)


@autodiff_only
@pytest.mark.parametrize("kernel", [logistic_radial_kernel, sigmoid_radial_kernel])
@pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0])
def test_exponential_kernel_has_finite_large_distance_gradient(kernel, bandwidth):
    distances = gs.array([1.0, 1000.0], dtype=gs.float64) * bandwidth

    _, gradient = gs.autodiff.value_and_grad(
        lambda value: gs.sum(kernel(value, bandwidth=bandwidth))
    )(distances)

    with localcontext() as context:
        context.prec = 80
        exp_positive = Decimal(1).exp()
        exp_negative = Decimal(-1).exp()
        denominator = exp_positive + exp_negative
        if kernel is logistic_radial_kernel:
            denominator += 2
        derivative = -(exp_positive - exp_negative) / denominator**2
    expected = [float(derivative) / bandwidth, 0.0]
    np.testing.assert_allclose(gs.to_numpy(gradient), expected, rtol=1e-12, atol=0)


@pytest.mark.smoke
class TestRadialKernelFunctions(
    RadialKernelFunctionsTestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = RadialKernelFunctionsTestData()
