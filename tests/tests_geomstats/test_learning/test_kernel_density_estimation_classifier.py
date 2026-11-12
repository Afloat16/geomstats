import math

import numpy as np
import pytest

from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.kernel_density_estimation_classifier import (
    KernelDensityEstimationClassifier,
)
from geomstats.learning.radial_kernel_functions import (
    gaussian_radial_kernel,
    triangular_radial_kernel,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.kernel_density_estimation_classifier import (
    KernelDensityEstimationClassifierTestCase,
)

from .data.kernel_density_estimation_classifier import (
    KernelDensityEstimationClassifierTestData,
)


@pytest.mark.parametrize("radius", [2.0, math.inf])
@pytest.mark.parametrize("multi_output", [False, True])
@pytest.mark.parametrize("kernel", [gaussian_radial_kernel, triangular_radial_kernel])
def test_callable_kernel_with_different_neighborhood_sizes(
    radius, multi_output, kernel
):
    """Each query keeps its own kernel weights, including empty neighborhoods."""
    samples = np.array([[0.0], [1.0], [10.0]])
    labels = np.array([[0, 1], [1, 0], [1, 0]])
    queries = np.array([[0.0], [9.0], [50.0]]) if radius == 2.0 else samples[:2]
    bandwidth = 2.0 if kernel is gaussian_radial_kernel else 20.0
    model = KernelDensityEstimationClassifier(
        Euclidean(1),
        radius=radius,
        kernel=kernel,
        bandwidth=bandwidth,
        outlier_label=0,
    ).fit(samples, labels if multi_output else labels[:, 0])

    probabilities = model.predict_proba(queries)

    if not multi_output:
        probabilities = [probabilities]
    expected_predictions = []
    for output, actual in enumerate(probabilities):
        expected = []
        for query in queries:
            distances = np.abs(samples[:, 0] - query[0])
            neighbors = distances <= radius
            if not neighbors.any():
                expected.append([1.0, 0.0])
                continue
            scaled = distances[neighbors] / bandwidth
            weights = (
                np.exp(-0.5 * scaled**2)
                if kernel is gaussian_radial_kernel
                else np.maximum(1.0 - scaled, 0.0)
            )
            votes = np.bincount(labels[neighbors, output], weights=weights, minlength=2)
            expected.append(votes / votes.sum())
        np.testing.assert_allclose(actual, expected)
        expected_predictions.append(np.argmax(expected, axis=-1))

    expected_predictions = np.array(expected_predictions).T
    if not multi_output:
        expected_predictions = expected_predictions[:, 0]
    np.testing.assert_array_equal(model.predict(queries), expected_predictions)


@pytest.mark.smoke
class TestKernelDensityEstimationClassifier(
    KernelDensityEstimationClassifierTestCase, metaclass=DataBasedParametrizer
):
    testing_data = KernelDensityEstimationClassifierTestData()
