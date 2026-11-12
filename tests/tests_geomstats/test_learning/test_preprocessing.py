import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.preprocessing import ToTangentSpace
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.learning.preprocessing import ToTangentSpaceTestCase

from .data.preprocessing import ToTangentSpaceNdim2TestData, ToTangentSpaceTestData

# TODO: fix main code and make this unique


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_rotation_matrix_round_trip_for_each_matrix_size(n):
    """Invert skew coordinates using the matrix size, not the algebra dimension."""
    space = SpecialOrthogonal(n, point_type="matrix", equip=False)
    algebra = SkewSymmetricMatrices(n)
    coordinates = gs.reshape(gs.linspace(-0.1, 0.1, 3 * algebra.dim), (3, -1))
    points = space.exp(algebra.matrix_representation(coordinates))
    transformer = ToTangentSpace(space).fit(points, base_point=space.identity)

    transformed = transformer.transform(points)
    reconstructed = transformer.inverse_transform(transformed)

    assert transformed.shape == coordinates.shape
    assert reconstructed.shape == points.shape
    assert gs.allclose(transformed, coordinates, atol=1e-8)
    assert gs.allclose(reconstructed, points, atol=1e-8)


@pytest.fixture(
    scope="class",
    params=[
        Hypersphere(dim=4),
        Hyperboloid(dim=3),
        Euclidean(dim=2),
        Minkowski(dim=2),
        SpecialOrthogonal(n=3, point_type="vector", equip=False),
    ],
)
def estimators(request):
    space = request.param
    request.cls.estimator = ToTangentSpace(space)


@pytest.mark.usefixtures("estimators")
class TestToTangent(ToTangentSpaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = ToTangentSpaceTestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialOrthogonal(n=3, point_type="matrix", equip=False),
        SPDMatrices(3),
    ],
)
def estimators_ndim2(request):
    space = request.param
    request.cls.estimator = ToTangentSpace(space)


@pytest.mark.usefixtures("estimators_ndim2")
class TestToTangentNdim2(ToTangentSpaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = ToTangentSpaceNdim2TestData()
