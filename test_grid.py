from main import get_grid
import unittest
import pytest
import numpy as np

class GetGridTests(unittest.TestCase):
    """
    Test that the relations in Appendix of Taylor et al. 1996 are satisfied.
    """

    n_subfaces_x = 5
    n_subfaces_y = 5

    def setUp(self):
        self.x, self.y, self.lmbda, self.theta = get_grid(
            self.n_subfaces_x, self.n_subfaces_y)

    def test_first_face_relation_holds(self):
        x = np.tan(self.lmbda[0, :])
        y = np.tan(self.theta[0, :])/np.cos(self.lmbda[0, :])
        assert np.allclose(x, self.x[0, :])
        assert np.allclose(y, self.y[0, :])

    def test_second_face_relation_holds(self):
        x = np.tan(self.lmbda[1, :] - np.pi/2)
        y = np.tan(self.theta[1, :])/np.cos(self.lmbda[1, :] - np.pi/2)
        assert np.allclose(x, self.x[1, :])
        assert np.allclose(y, self.y[1, :])

    def test_third_face_relation_holds(self):
        x = np.tan(self.lmbda[2, :] - np.pi)
        y = np.tan(self.theta[2, :])/np.cos(self.lmbda[2, :] - np.pi)
        assert np.allclose(x, self.x[2, :])
        assert np.allclose(y, self.y[2, :])

    def test_fourth_face_relation_holds(self):
        x = np.tan(self.lmbda[3, :] - 3./2*np.pi)
        y = np.tan(self.theta[3, :])/np.cos(self.lmbda[3, :] - 3./2*np.pi)
        assert np.allclose(x, self.x[3, :])
        assert np.allclose(y, self.y[3, :])

    def test_fifth_top_face_relation_holds(self):
        x = np.tan(self.theta[4, :] - np.pi/2) * np.sin(self.lmbda[4, :])
        y = -1 * np.tan(self.theta[4, :] - np.pi/2) * np.cos(self.lmbda[4, :])
        assert np.allclose(x, self.x[4, :])
        assert np.allclose(y, self.y[4, :])

    def test_sixth_bottom_face_relation_holds(self):
        x = np.tan(-1*self.theta[5, :] - np.pi/2) * np.sin(self.lmbda[5, :])
        y = -1 * np.tan(-1*self.theta[5, :] - np.pi/2) * np.cos(self.lmbda[5, :])
        assert np.allclose(x, self.x[5, :])
        assert np.allclose(y, self.y[5, :])


if __name__ == '__main__':
    pytest.main([__file__])
