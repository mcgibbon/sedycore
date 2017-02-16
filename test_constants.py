from main import get_taylor_1996_constants, get_grid
import unittest
import pytest
import numpy as np


class GetTaylor1996ConstantsTests(unittest.TestCase):

    n_subfaces_x = 2
    n_subfaces_y = 2

    def setUp(self):
        self.x, self.y, self.lmbda, self.theta = get_grid(
            self.n_subfaces_x, self.n_subfaces_y)
        self.p, self.q, self.D, self.D_inverse = get_taylor_1996_constants(
            self.lmbda, self.theta)

    def test_D_inverse_is_inverse_of_D_on_sides(self):
        product = np.einsum('ijklmno,ijklmop->ijklmnp', self.D[:4, :], self.D_inverse[:4, :])
        assert np.allclose(1, product[:, :, :, :, :, 0, 0])
        assert np.allclose(0, product[:, :, :, :, :, 0, 1])
        assert np.allclose(1, product[:, :, :, :, :, 1, 1])
        assert np.allclose(0, product[:, :, :, :, :, 1, 0])

    def test_D_inverse_is_inverse_of_D_on_polar_faces(self):
        product = np.einsum('ijklmno,ijklmop->ijklmnp', self.D[4:, :], self.D_inverse[4:, :])
        assert np.allclose(1, product[:, :, :, :, :, 0, 0])
        assert np.allclose(0, product[:, :, :, :, :, 0, 1])
        assert np.allclose(1, product[:, :, :, :, :, 1, 1])
        assert np.allclose(0, product[:, :, :, :, :, 1, 0])


if __name__ == '__main__':
    pytest.main([__file__])
