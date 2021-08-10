import numpy as np

from mog.consts import ALPHA, NUM_OF_GAUSSIANS, DISTANCE_THRESHOLD, VAR_DEFAULT
from mog.utils import calc_mahalanobis_distances, plot_gaussians, get_random_from_2_gaussians


class MOG:

    def __init__(self, num_of_gaussians: int = NUM_OF_GAUSSIANS, var_default: float = VAR_DEFAULT,
                 distance_threshold: float = DISTANCE_THRESHOLD, alpha: float = ALPHA):
        self.weights = np.zeros(num_of_gaussians) + 1
        self.means = np.zeros(num_of_gaussians) + 20
        self.var_default = var_default
        self.variances = np.asarray([var_default])
        self.distance_threshold = distance_threshold
        self.alpha = alpha
        self.epsilon = 1e-8

    def update_background_model(self, new_value, plot=False):
        deltas = new_value - self.means

        distances_ = calc_mahalanobis_distances(deltas, self.variances)
        min_distance_index = np.argmin(distances_)

        if distances_[min_distance_index] > self.distance_threshold:
            self.weights = np.append(self.weights, self.alpha)
            self.means = np.append(self.means, new_value)
            self.variances = np.append(self.variances, self.var_default)
        else:
            ownerships = np.zeros_like(distances_)
            ownerships[min_distance_index] = 1
            self.weights += self.alpha * (ownerships - self.weights)
            self.means += ownerships * self.alpha / (self.weights + self.epsilon) * deltas
            self.variances += ownerships * self.alpha / (self.weights + self.epsilon) * (deltas ** 2 - self.variances)

        if plot:
            plot_gaussians(self.weights, self.means, self.variances)

        print('means:', self.means)
        print('variances:', self.variances)
        print()

    def is_background(self, new_value):
        deltas = new_value - self.means
        distances = calc_mahalanobis_distances(deltas, self.variances)

        if np.min(distances) > self.distance_threshold:
            return False

        return True

    def apply(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


if __name__ == '__main__':

    s = []
    for i in range(1000):
        s.append(get_random_from_2_gaussians())

    mog = MOG()

    for value in s:
        mog.update_background_model(value, plot=True)

    check_list = [-5, 0, 2, 4, 5, 7]

    for value in check_list:
        print(mog.is_background(value))
