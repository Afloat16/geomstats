"""Regression tests for fitted tangent PCA matrix coordinates."""

import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.learning.pca import TangentPCA
from geomstats.test.test_case import assert_allclose, np_and_autograd_only

pytestmark = np_and_autograd_only


def _matrix_data(shape):
    """Return an affinely independent set of matrices."""
    size = shape[0] * shape[1]
    vectors = gs.concatenate([gs.zeros((1, size)), gs.eye(size)], axis=0)
    return gs.reshape(vectors, (size + 1,) + shape)


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (2, 2)])
def test_matrix_pca_reconstructs_original_point_shape(shape):
    """Rectangular and unrestricted square matrices round-trip at full rank."""
    data = _matrix_data(shape)
    estimator = TangentPCA(Matrices(*shape))

    scores = estimator.fit_transform(data, base_point=gs.zeros(shape))
    reconstructed = estimator.inverse_transform(scores)

    assert reconstructed.shape == data.shape
    assert_allclose(reconstructed, data, atol=1e-10)


def test_transform_preserves_full_coordinates_for_symmetric_held_out_points():
    """A symmetric new observation does not change the fitted feature count."""
    data = _matrix_data((2, 2))
    estimator = TangentPCA(Matrices(2, 2)).fit(data, base_point=gs.zeros((2, 2)))
    held_out = gs.array([[[2.0, 1.0], [1.0, 3.0]], [[-1.0, 0.0], [0.0, 2.0]]])
    flattened = gs.reshape(held_out, (2, 4))
    expected = (flattened - estimator.mean_) @ gs.transpose(estimator.components_)

    scores = estimator.transform(held_out)

    assert_allclose(scores, expected, atol=1e-10)
    assert_allclose(estimator.inverse_transform(scores), held_out, atol=1e-10)


def test_symmetric_tangents_do_not_require_a_symmetric_base_point():
    """Inverse coordinates are determined by training tangents, not the base."""
    base_point = gs.array([[0.0, -2.0], [2.0, 0.0]])
    tangents = gs.concatenate([gs.zeros((1, 2, 2)), SymmetricMatrices(2).basis], axis=0)
    data = base_point + tangents
    estimator = TangentPCA(Matrices(2, 2))

    scores = estimator.fit_transform(data, base_point=base_point)

    assert estimator.components_.shape == (3, 3)
    assert_allclose(estimator.inverse_transform(scores), data, atol=1e-10)


def test_spd_pca_keeps_symmetric_coordinates():
    """The existing packed representation of SPD tangent vectors is retained."""
    tangents = gs.concatenate(
        [gs.zeros((1, 2, 2)), 0.2 * SymmetricMatrices(2).basis], axis=0
    )
    data = gs.linalg.expm(tangents)
    estimator = TangentPCA(SPDMatrices(2))

    scores = estimator.fit_transform(data, base_point=gs.eye(2))

    assert estimator.components_.shape == (3, 3)
    assert_allclose(scores, estimator.transform(data), atol=1e-10)
    assert_allclose(estimator.inverse_transform(scores), data, atol=1e-10)


def test_refit_refreshes_the_coordinate_representation():
    """Refitting does not reuse the previous fit's matrix packing choice."""
    estimator = TangentPCA(Matrices(2, 2))
    symmetric = gs.concatenate(
        [gs.zeros((1, 2, 2)), SymmetricMatrices(2).basis], axis=0
    )
    for data in [_matrix_data((2, 2)), symmetric, _matrix_data((2, 2))]:
        scores = estimator.fit_transform(data, base_point=gs.zeros((2, 2)))
        assert_allclose(estimator.inverse_transform(scores), data, atol=1e-10)


def test_packed_model_rejects_incompatible_tangent_symmetry():
    """A packed fit must not silently discard the lower matrix triangle."""
    data = gs.concatenate([gs.zeros((1, 2, 2)), SymmetricMatrices(2).basis], axis=0)
    estimator = TangentPCA(Matrices(2, 2)).fit(data, base_point=gs.zeros((2, 2)))

    with pytest.raises(ValueError, match="symmetric tangent"):
        estimator.transform(gs.array([[[0.0, 1.0], [0.0, 0.0]]]))


def test_vector_pca_reconstruction_is_unchanged():
    """Matrix coordinate handling leaves ordinary vector observations intact."""
    data = gs.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 3.0]])
    estimator = TangentPCA(Euclidean(2))
    scores = estimator.fit_transform(data, base_point=gs.zeros(2))

    assert_allclose(estimator.inverse_transform(scores), data, atol=1e-10)
