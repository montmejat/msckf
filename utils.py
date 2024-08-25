import numpy as np


def cov_matrix(x: np.ndarray):
    return np.diag(x) ** 2


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.reshape(3)
    return np.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ]
    )


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def omega(vector: np.ndarray) -> np.ndarray:
    vector = vector.reshape(3, 1)
    vect_x = skew(vector)
    return np.block([[-vect_x, vector], [-vector.T, 0]])


def rk4(func: callable, h: float, x_0: object) -> object:
    k_1 = func(x_0)
    k_2 = func(x_0 + 0.5 * h * k_1)
    k_3 = func(x_0 + 0.5 * h * k_2)
    k_4 = func(x_0 + h * k_3)
    return x_0 + h * ((k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4) / 6.0)
