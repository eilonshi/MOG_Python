import numpy as np

from mog.consts import EPSILON


def calc_mahalanobis_distances(deltas: np.ndarray, variances: np.ndarray):
    return np.sqrt(deltas ** 2 / (variances + EPSILON))
