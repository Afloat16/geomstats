import numpy as np
import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import autodiff_only
from geomstats.test_cases.backend import AgainstEinsumTestCase

from .data.against_einsum import AgainstEinsumTestData


class TestAgainstEinsum(AgainstEinsumTestCase, metaclass=DataBasedParametrizer):
    testing_data = AgainstEinsumTestData()


@pytest.mark.parametrize(
    "matrix_shape, vector_shape",
    [
        ((2, 3), (3,)),
        ((2, 3), (4, 3)),
        ((2, 3), (2, 4, 3)),
        ((3, 3), (2, 3, 3)),
        ((3, 3), (2, 1, 4, 3)),
        ((2, 1, 2, 3), (1, 4, 3)),
    ],
)
@pytest.mark.parametrize("dtype", [gs.float64, gs.complex128])
def test_matvec_multiple_batch_axes(matrix_shape, vector_shape, dtype):
    matrix = np.arange(np.prod(matrix_shape)).reshape(matrix_shape) + 0.5
    vector = np.arange(np.prod(vector_shape)).reshape(vector_shape) - 1.5
    if dtype == gs.complex128:
        matrix = matrix + 0.5j
        vector = vector - 0.25j
    expected = np.sum(matrix * vector[..., None, :], axis=-1)

    result = gs.matvec(gs.array(matrix, dtype=dtype), gs.array(vector, dtype=dtype))

    assert result.shape == expected.shape
    np.testing.assert_allclose(gs.to_numpy(result), expected)


@pytest.mark.parametrize(
    "first_shape, second_shape",
    [
        ((3,), (3,)),
        ((3,), (4, 3)),
        ((3,), (2, 4, 3)),
        ((3,), (3, 3, 3)),
        ((3,), (2, 1, 4, 3)),
        ((2, 1, 3), (1, 4, 3)),
    ],
)
@pytest.mark.parametrize("dtype", [gs.float64, gs.complex128])
def test_dot_multiple_batch_axes(first_shape, second_shape, dtype):
    first = np.arange(np.prod(first_shape)).reshape(first_shape) + 0.5
    second = np.arange(np.prod(second_shape)).reshape(second_shape) - 1.5
    if dtype == gs.complex128:
        first = first + 0.5j
        second = second - 0.25j
    expected = np.sum(first * second, axis=-1)

    result = gs.dot(gs.array(first, dtype=dtype), gs.array(second, dtype=dtype))

    assert result.shape == expected.shape
    np.testing.assert_allclose(gs.to_numpy(result), expected)


def test_vector_batch_metric_duality():
    matrix = gs.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.25], [0.0, 0.25, 4.0]])
    space = Euclidean(3, equip=False).equip_with_metric(
        EuclideanMetric, metric_matrix=matrix
    )
    tangent_vec = gs.reshape(gs.arange(18.0), (2, 3, 3))
    base_point = gs.zeros_like(tangent_vec)

    cotangent_vec = space.metric.flat(tangent_vec, base_point)
    recovered = space.metric.sharp(cotangent_vec, base_point)

    np.testing.assert_allclose(
        gs.to_numpy(recovered), gs.to_numpy(tangent_vec), atol=1e-12
    )


@autodiff_only
@pytest.mark.parametrize("operation", ["matvec", "dot"])
def test_vector_batch_product_gradient(operation):
    vector = gs.reshape(gs.arange(24.0), (2, 4, 3)) / 10
    if operation == "matvec":
        first = gs.array([[1.0, 2.0, -1.0], [0.5, 1.0, 3.0]])
        value = np.sum(gs.to_numpy(first) * gs.to_numpy(vector)[..., None, :], -1)
        expected = 2 * np.sum(value[..., :, None] * gs.to_numpy(first), -2)
    else:
        first = gs.array([1.0, 2.0, -1.0])
        value = np.sum(gs.to_numpy(first) * gs.to_numpy(vector), -1)
        expected = 2 * value[..., None] * gs.to_numpy(first)

    product = getattr(gs, operation)
    _, gradient = gs.autodiff.value_and_grad(
        lambda point: gs.sum(product(first, point) ** 2), point_ndims=3
    )(vector)

    np.testing.assert_allclose(gs.to_numpy(gradient), expected)
