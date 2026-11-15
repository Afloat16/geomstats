import pytest

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.pca import TangentPCA
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import np_and_autograd_only
from geomstats.test_cases.learning.pca import TangentPCATestCase

from .data.pca import TangentPCATestData


@pytest.fixture(
    scope="class",
    params=[
        SpecialOrthogonal(n=3, point_type="vector", equip=False),
        SPDMatrices(3),
        SpecialEuclidean(n=3, equip=False),
    ],
)
def estimators(request):
    space = request.param
    request.cls.estimator = TangentPCA(space)


@np_and_autograd_only
@pytest.mark.usefixtures("estimators")
class TestTangentPCA(
    TangentPCATestCase,
    metaclass=DataBasedParametrizer,
):
    testing_data = TangentPCATestData()


@np_and_autograd_only
@pytest.mark.parametrize("n", [2, 3])
def test_rotation_matrix_round_trip(n):
    """Reconstruct full matrix coordinates even when the base point is symmetric."""
    space = SpecialOrthogonal(n=n, point_type="matrix", equip=False)
    coordinates = 0.1 * gs.concatenate([gs.eye(space.dim), -gs.eye(space.dim)], axis=0)
    points = space.exp(SkewSymmetricMatrices(n).matrix_representation(coordinates))
    estimator = TangentPCA(space)
    scores = estimator.fit_transform(points, base_point=space.identity)

    reconstructed = estimator.inverse_transform(scores)

    assert reconstructed.shape == points.shape
    assert gs.allclose(reconstructed, points, atol=1e-10)


@np_and_autograd_only
@pytest.mark.parametrize("shape", [(2, 3), (3, 2)])
def test_rectangular_matrix_round_trip(shape):
    """Reconstruct rectangular tangent vectors with their original dimensions."""
    space = Matrices(*shape)
    base_point = gs.reshape(gs.arange(space.dim, dtype=float), shape)
    basis = gs.reshape(gs.eye(space.dim), (space.dim, *shape))
    points = base_point + gs.concatenate([basis, -basis], axis=0)
    estimator = TangentPCA(space)
    scores = estimator.fit_transform(points, base_point=base_point)

    reconstructed = estimator.inverse_transform(scores)

    assert reconstructed.shape == points.shape
    assert gs.allclose(reconstructed, points, atol=1e-10)


@np_and_autograd_only
def test_transform_retains_fitted_matrix_coordinates():
    """Symmetric queries use the full coordinates learned from nonsymmetric data."""
    space = Matrices(2, 2)
    basis = gs.reshape(gs.eye(space.dim), (space.dim, *space.shape))
    points = gs.concatenate([basis, -basis], axis=0)
    estimator = TangentPCA(space).fit(points, base_point=gs.zeros(space.shape))
    query = gs.array([[[2.0, 1.0], [1.0, 3.0]]])

    scores = estimator.transform(query)

    assert scores.shape == (1, space.dim)
    assert gs.allclose(estimator.inverse_transform(scores), query, atol=1e-10)


@np_and_autograd_only
def test_compressed_matrix_coordinates_reject_nonsymmetric_queries():
    """Reject incompatible queries instead of discarding their lower triangle."""
    space = Matrices(2, 2)
    basis = gs.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    )
    points = gs.concatenate([basis, -basis], axis=0)
    estimator = TangentPCA(space).fit(points, base_point=gs.zeros(space.shape))
    query = gs.array([[[0.0, 1.0], [-1.0, 0.0]]])

    with pytest.raises(ValueError, match="symmetric"):
        estimator.transform(query)
