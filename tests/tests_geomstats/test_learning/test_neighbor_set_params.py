import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.positive_reals import PositiveReals
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.kernel_density_estimation_classifier import (
    KernelDensityEstimationClassifier,
)
from geomstats.learning.knn import KNearestNeighborsClassifier
from geomstats.learning.radial_kernel_functions import (
    gaussian_radial_kernel,
    triangular_radial_kernel,
)


@pytest.mark.parametrize(
    "initial_kernel", ["uniform", gaussian_radial_kernel, triangular_radial_kernel]
)
@pytest.mark.parametrize(
    "target_kernel", ["uniform", gaussian_radial_kernel, triangular_radial_kernel]
)
def test_kernel_update_matches_fresh_classifier(initial_kernel, target_kernel):
    space = Euclidean(dim=1)
    points = gs.array([[0.0], [1.0], [4.0]])
    labels = np.array([0, 0, 1])
    queries = gs.array([[0.2], [2.8]])
    classifier = KernelDensityEstimationClassifier(
        space, kernel=initial_kernel, bandwidth=2.0
    )
    assert classifier.set_params(kernel=target_kernel) is classifier
    classifier.fit(points, labels)
    reference = KernelDensityEstimationClassifier(
        space, kernel=target_kernel, bandwidth=2.0
    ).fit(points, labels)

    np.testing.assert_allclose(
        classifier.predict_proba(queries), reference.predict_proba(queries)
    )


def test_grid_search_evaluates_each_kernel():
    points = gs.array([[0.0], [1.0], [4.0], [2.8]])
    labels = np.array([0, 0, 1, 1])
    classifier = KernelDensityEstimationClassifier(Euclidean(dim=1), kernel="uniform")
    search = GridSearchCV(
        classifier,
        {"kernel": ["uniform", gaussian_radial_kernel]},
        cv=[(np.array([0, 1, 2]), np.array([3]))],
        error_score="raise",
    ).fit(points, labels)

    np.testing.assert_array_equal(search.cv_results_["mean_test_score"], [0.0, 1.0])
    assert search.best_params_["kernel"] is gaussian_radial_kernel


@pytest.mark.parametrize(
    "classifier_class", [KernelDensityEstimationClassifier, KNearestNeighborsClassifier]
)
@pytest.mark.parametrize("positive_reals_target", [False, True])
def test_space_update_matches_fresh_classifier(classifier_class, positive_reals_target):
    initial_space = Euclidean(dim=1) if positive_reals_target else PositiveReals()
    target_space = PositiveReals() if positive_reals_target else Euclidean(dim=1)
    parameters = (
        {"n_neighbors": 1} if classifier_class is KNearestNeighborsClassifier else {}
    )
    points = gs.array([[1.0], [100.0]])
    labels = np.array([0, 1])
    queries = gs.array([[20.0], [21.0]])
    classifier = classifier_class(initial_space, **parameters)
    assert classifier.set_params(space=target_space) is classifier
    classifier.fit(points, labels)
    reference = classifier_class(target_space, **parameters).fit(points, labels)

    np.testing.assert_array_equal(
        classifier.predict(queries), reference.predict(queries)
    )
    np.testing.assert_array_equal(
        classifier.predict(queries), [int(positive_reals_target)] * 2
    )


def test_knn_space_update_refreshes_matrix_interoperability():
    target_space = SPDMatrices(n=2)
    points = gs.array([[[1.0, 0.0], [0.0, 1.0]], [[4.0, 0.0], [0.0, 4.0]]])
    labels = np.array([0, 1])
    queries = gs.array([[[3.5, 0.0], [0.0, 3.5]], [[3.8, 0.0], [0.0, 3.8]]])
    classifier = KNearestNeighborsClassifier(Euclidean(dim=2), n_neighbors=1)
    classifier.set_params(space=target_space).fit(points, labels)
    reference = KNearestNeighborsClassifier(target_space, n_neighbors=1).fit(
        points, labels
    )

    np.testing.assert_array_equal(
        classifier.predict(queries), reference.predict(queries)
    )


@pytest.mark.parametrize(
    "classifier_class", [KernelDensityEstimationClassifier, KNearestNeighborsClassifier]
)
def test_unrelated_or_empty_parameter_updates_keep_predictions(classifier_class):
    parameters = (
        {"n_neighbors": 1} if classifier_class is KNearestNeighborsClassifier else {}
    )
    classifier = classifier_class(Euclidean(dim=1), **parameters)
    assert classifier.set_params() is classifier
    classifier.set_params(n_jobs=1).fit(gs.array([[0.0], [1.0]]), np.array([0, 1]))

    np.testing.assert_array_equal(classifier.predict(gs.array([[0.0], [1.0]])), [0, 1])
