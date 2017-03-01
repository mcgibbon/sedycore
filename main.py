import numpy as np
import matplotlib.pyplot as plt
import sympl

n_subfaces_x = 5
n_subfaces_y = 5
len_spec_basis_x = 5
len_spec_basis_y = 5


def get_grid(n_subfaces_x, n_subfaces_y):
    n_subface_x = np.arange(n_subfaces_x)[None, :, None, None, None]

    legendre_lobatto_points = np.array([-1., -(3./7)**0.5, 0, (3./7)**0.5, 1.])

    x = np.empty([6, n_subfaces_x, n_subfaces_y, len_spec_basis_x, len_spec_basis_y])
    x[:] = -1 + n_subface_x/n_subfaces_x * 2 + (legendre_lobatto_points[None, None, None, :, None] + 1) / n_subfaces_x

    n_subface_y = np.arange(n_subfaces_y)[None, None, :, None, None]

    y = np.empty_like(x)
    y[:] = -1 + n_subface_y/n_subfaces_y * 2 + (legendre_lobatto_points[None, None, None, None, :] + 1) / n_subfaces_y

    lmbda = np.empty_like(x)
    theta = np.empty_like(x)

    for i in range(4):
        lmbda[i, :] = np.arctan(x[i, :]) + i * np.pi / 2
        theta[i, :] = np.arctan(y[i, :] * np.cos(lmbda[i, :] - i * np.pi / 2))

    lmbda[4:, :] = np.arctan(-1*x[4:, :]/y[4:, :])
    lmbda[4:, :][y[4:, :] == 0] = 0.
    theta[4:, :] = np.arctan(x[4:, :]/np.sin(lmbda[4:, :])) + np.pi/2
    invalid_theta = np.sin(lmbda[4:, :]) == 0
    theta[4:, :][invalid_theta] = np.arctan(-y[4:, :][invalid_theta]/np.cos(lmbda[4:, :][invalid_theta])) + np.pi/2
    # at this point theta goes past pi/2 to the other side of the globe, need to
    # change this to have theta < pi/2 and lon on other side of globe
    lmbda[4:, :][theta[4:, :] > np.pi/2] += np.pi
    theta[4:, :][theta[4:, :] > np.pi/2] = np.pi - theta[4:, :][theta[4:, :] > np.pi/2]
    # 6th face (index 5) is south pole, opposite lat of 5th face
    theta[5, :] *= -1
    return x, y, lmbda, theta


def P0(x):
    return 0*x + 1.


def P1(x):
    return x


def P2(x):
    return 0.5*(3*x**2 - 1)


def P3(x):
    return 0.5*(5*x**3 - 3*x)


def P4(x):
    return 3./8 - 15./4*x**2 + 35./8*x**4


def get_taylor_1996_constants(lmbda, theta):
    # these are the p and q vectors in the appendix of Taylor et al. 1997.
    p = np.empty(list(lmbda.shape) + [2], dtype=lmbda.dtype)
    q = np.empty(list(lmbda.shape) + [2], dtype=lmbda.dtype)

    lmbda_minus_k_pi_over_2 = np.empty([4] + list(lmbda.shape)[1:])
    for k in range(4):
        lmbda_minus_k_pi_over_2[k, :] = lmbda[k, :] - k * np.pi / 2.

    const = -3. * np.cos(theta[:4, :]) * np.cos(lmbda_minus_k_pi_over_2[:4, :])
    p[:4, :, :, :, :, 0] = const * np.cos(theta[:4, :]) * np.sin(lmbda_minus_k_pi_over_2[:4, :])
    p[:4, :, :, :, :, 1] = const * np.sin(theta[:4, :])
    q[:4, :, :, :, :, 0] = np.sin(theta[:4, :])
    q[:4, :, :, :, :, 1] = -1 * np.cos(theta[:4, :]) * np.sin(lmbda_minus_k_pi_over_2[:4, :])

    const = 3./2*np.sin(2*theta[4:, :])
    p[4:, :, :, :, :, 0] = -1 * const * np.sin(lmbda[4:, :])
    p[4:, :, :, :, :, 1] = const * np.cos(lmbda[4:, :])
    const = -1 * np.cos(theta[4:, :])
    q[4:, :, :, :, :, 0] = const * np.cos(lmbda[4:, :])
    q[4:, :, :, :, :, 0] = const * np.sin(lmbda[4:, :])

    D = np.empty(list(lmbda.shape) + [2, 2])
    D_inverse = np.empty(list(lmbda.shape) + [2, 2])

    const = np.cos(theta[:4, :]) * np.cos(lmbda_minus_k_pi_over_2[:4, :])
    D[:4, :, :, :, :, 0, 0] = const * np.cos(lmbda_minus_k_pi_over_2)
    D[:4, :, :, :, :, 0, 1] = 0.
    D[:4, :, :, :, :, 1, 0] = -1 * const * np.sin(theta[:4, :]) * np.sin(lmbda_minus_k_pi_over_2)
    D[:4, :, :, :, :, 1, 1] = const * np.cos(theta[:4, :])

    const = np.sin(theta[4, :])
    D[4, :, :, :, :, 0, 0] = const * np.cos(lmbda[4, :])
    D[4, :, :, :, :, 0, 1] = const * np.sin(lmbda[4, :])
    D[4, :, :, :, :, 1, 0] = -1 * const * np.sin(theta[4, :]) * np.sin(lmbda[4, :])
    D[4, :, :, :, :, 1, 1] = const * np.sin(theta[4, :]) * np.cos(lmbda[4, :])

    const = np.sin(-1*theta[5, :])
    D[5, :, :, :, :, 0, 0] = const * np.cos(lmbda[5, :])
    D[5, :, :, :, :, 0, 1] = const * np.sin(lmbda[5, :])
    D[5, :, :, :, :, 1, 0] = -1 * const * np.sin(-1*theta[5, :]) * np.sin(lmbda[5, :])
    D[5, :, :, :, :, 1, 1] = const * np.sin(-1*theta[5, :]) * np.cos(lmbda[5, :])

    const = 1. / (np.cos(theta[:4, :]) * np.cos(lmbda_minus_k_pi_over_2))
    D_inverse[:4, :, :, :, :, 0, 0] = const / np.cos(lmbda_minus_k_pi_over_2)
    D_inverse[:4, :, :, :, :, 0, 1] = 0.
    D_inverse[:4, :, :, :, :, 1, 0] = const * np.tan(theta[:4, :]) * np.tan(lmbda_minus_k_pi_over_2)
    D_inverse[:4, :, :, :, :, 1, 1] = const / np.cos(theta[:4, :])

    const = 1. / np.sin(theta[4, :])
    D_inverse[4, :, :, :, :, 0, 0] = const * np.cos(lmbda[4, :])
    D_inverse[4, :, :, :, :, 0, 1] = -1 * const * np.sin(lmbda[4, :]) / np.sin(theta[4, :])
    D_inverse[4, :, :, :, :, 1, 0] = const * np.sin(lmbda[4, :])
    D_inverse[4, :, :, :, :, 1, 1] = const * np.cos(lmbda[4, :]) / np.sin(theta[4, :])

    const = 1. / np.sin(-1 * theta[5, :])
    D_inverse[5, :, :, :, :, 0, 0] = const * np.cos(lmbda[5, :])
    D_inverse[5, :, :, :, :, 0, 1] = -1 * const * np.sin(lmbda[5, :]) / np.sin(-1 * theta[5, :])
    D_inverse[5, :, :, :, :, 1, 0] = const * np.sin(lmbda[5, :])
    D_inverse[5, :, :, :, :, 1, 1] = const * np.cos(lmbda[5, :]) / np.sin(-1 * theta[5, :])

    return p, q, D, D_inverse


class SpectralElementAdvection(sympl.Prognostic):

    def __init__(self, longitude_radians, latitude_radians):
        self.lmbda, self.theta = longitude_radians, latitude_radians

        # compute and cache constants once to reduce computational cost
        self.p, self.q, self.D, self.D_inverse = get_taylor_1996_constants(
            self.lmbda, self.theta)
        legendre_derivative = get_legendre_derivative_matrix()
        real_to_computational, computational_to_real = get_transformation_matrices()
        self.derivative_matrix = np.dot(
            np.dot(computational_to_real, legendre_derivative),
            real_to_computational)
        rotate_90_degrees = np.array(
            [[0, -1],
             [1, 0]])
        self.DT_D = np.dot(self.D.transpose(), self.D)
        self.Dinv_Rot_D = np.dot(np.dot(self.D_inverse, rotate_90_degrees), self.D)
        self.Dinv_DinvT = np.dot(self.D_inverse, self.D_inverse.transpose())
        self.Dinv_Rot_D_derivative = np.dot(self.Dinv_Rot_D, self.derivative_matrix)
        self.coriolis_frequency = 2*Omega*np.sin(latitude_radians)

    def __call__(self, state):
        v1 = state['x_wind'].values
        v2 = state['y_wind'].values
        h = state['fluid_depth'].values
        n_subfaces = v1.shape[0]*v1.shape[1]*v1.shape[2]
        target_shape = [n_subfaces] + list(v1.shape[3:])
        v1, v2, h = v1.reshape(target_shape), v2.reshape(target_shape), h.reshape(target_shape)
        k_cross_v = np.empty(target_shape + [2])
        k_cross_v[:, :, :, 0] = np.einsum(
                'ipqjkl,ipqjkl->ipqjk',
                self.Dinv_Rot_D[:, :, :, :, 0], v1[:, :, :, None])
        k_cross_v[:, :, :, 1] = np.einsum(
                'ipqjkl,ipqjkl->ipqjk',
                self.Dinv_Rot_D[:, :, :, :, 1], v2[:, :, :, None])
        vorticity = -1 * (
            np.einsum(
                'jm,ipqjk->ipqmk', self.derivative_matrix, k_cross_v[:, :, :, 0]) +
            np.einsum(
                'km,ipqjk->ipqjm', self.derivative_matrix, k_cross_v[:, :, :, 1])
        ) + self.q[:, :, :, :, :, 0]*v1 + self.q[:, :, :, :, :, 1]*v2
        phi = 0.5 * (
            self.DT_D[:, :, :, :, :, 0, 0]*v1**2 +
            (self.DT_D[:, :, :, :, :, 1, 0] + self.DT_D[:, :, :, :, :, 0, 1])*v1*v2 +
            self.DT_D[:, :, :, :, :, 1, 1]*v2**2) + g*h
        phi_x = np.einsum(
            'jm,ipqjk->ipqmk', self.derivative_matrix, phi)
        phi_y = np.einsum(
            'km,ipqjk->ipqjm', self.derivative_matrix, phi)
        dh_dt = -1 * (
            np.einsum(
                'jm,ipqjk->ipqmk', self.derivative_matrix, h*v1) +
            np.einsum(
                'km,ipqjk->ipqjm', self.derivative_matrix, h*v2)
            ) - p[:, :, :, :, :, 0]*h*v1 - p[:, :, :, :, :, 1]*h*v2
        dv1_dt = -(vorticity + self.coriolis_frequency) * (
            self.Dinv_Rot_D[:, :, :, :, :, 0, 0] * v1 +
            self.Dinv_Rot_D[:, :, :, :, :, 0, 1] * v2) - (
            self.Dinv_DinvT[:, :, :, :, :, 0, 0] * phi_x +
            self.Dinv_DinvT[:, :, :, :, :, 0, 1] * phi_y
        )
        dv2_dt = -(vorticity + self.coriolis_frequency) * (
            self.Dinv_Rot_D[:, :, :, :, :, 1, 0] * v1 +
            self.Dinv_Rot_D[:, :, :, :, :, 1, 1] * v2) - (
            self.Dinv_DinvT[:, :, :, :, :, 1, 0] * phi_x +
            self.Dinv_DinvT[:, :, :, :, :, 1, 1] * phi_y
        )



def get_legendre_derivative_matrix():
    N = 5
    legendre_derivative = np.empty([N, N])
    for i in range(N):
        for j in range(N):
            if i == j == 0:
                legendre_derivative[i, j] = 0.25 * N * (N + 1)
            elif i == j == N:
                legendre_derivative[i, j] = -0.25 * N * (N + 1)
            elif i == j:
                legendre_derivative[i, j] = 0.
            else:
                legendre_derivative[i, j] = P4(x_leg[i]) / (P4(x_leg[j]) * (x_leg[i] - x_leg[j]))
    return legendre_derivative


def get_transformation_matrices():
    N = 5
    legendre_basis_functions = [P0, P1, P2, P3, P4]
    real_to_computational = np.empty([N, N])
    computational_to_real = np.empty([N, N])
    for j in range(N):
        wj = 2./(N*(N+1)*P4(x_leg[j])**2)
        for i in range(N):
            real_to_computational[i, j] = legendre_basis_functions[i](x_leg[j])*wj*(2*i + 1)/2.
            computational_to_real[i, j] = legendre_basis_functions[j](x_leg[i])
    return real_to_computational, computational_to_real


if __name__ == '__main__':
    x, y, lmbda, theta = get_grid(
        n_subfaces_x, n_subfaces_y)

    legendre_lobatto_points = np.array(
        [-1., -(3. / 7) ** 0.5, 0, (3. / 7) ** 0.5, 1.])
    x_leg = legendre_lobatto_points

    p, q, D, D_inverse = get_taylor_1996_constants(lmbda, theta)

    plt.figure()
    for i in range(6):
        plt.scatter(lmbda[i, :].flatten(), theta[i, :].flatten(), c=x[i, :].flatten(), cmap='gray')
    #plt.ylim(-0.5*np.pi, 0.5*np.pi)
    #plt.xlim(-0.25*np.pi, 1.75*np.pi)
    plt.show()

    # for i in range(6):
    #     plt.figure()
    #     plt.plot(y[i, :].flatten(), x[i, :].flatten(), 'o')
    #     plt.show()

    # lmbda = np.linspace(0, 2*np.pi)
    # theta = np.linspace(np.pi*0.25, np.pi*0.5)
    # lmbda, theta = np.broadcast_arrays(lmbda[:, None], theta[None, :])
    # x, y = np.tan(theta)*np.sin(lmbda), -1*np.tan(theta)*np.cos(lmbda)
    # plt.figure()
    # plt.plot(x.flatten(), y.flatten(), 'o')
    # plt.show()