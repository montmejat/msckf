import numpy as np


def cov_matrix(x: np.ndarray):
    return np.diag(x) ** 2
