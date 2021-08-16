import numpy as np
from typing import Tuple

from experiment.utils import plot_gaussians, get_random_from_2_gaussians
from mog.consts import HISTORY_DEFAULT, MAX_NUM_OF_GAUSSIANS, VAR_DEFAULT, VAR_THRESHOLD_DEFAULT, ALPHA, EPSILON
from mog.utils import calc_mahalanobis_distances


class MOG:

    def __init__(self, history: int = HISTORY_DEFAULT, var_threshold: float = VAR_THRESHOLD_DEFAULT,
                 var_default: float = VAR_DEFAULT, max_num_of_gaussians: int = MAX_NUM_OF_GAUSSIANS,
                 alpha: float = ALPHA, epsilon: float = EPSILON):
        self.history = history
        self.var_threshold = var_threshold
        self.var_default = var_default
        self.max_num_of_gaussians = max_num_of_gaussians
        self.alpha = alpha
        self.epsilon = epsilon

        self.num_iterations = 0
        self.means = None
        self.variances = None
        self.weights = None

    def update_background_model(self, new_frame: np.ndarray, plot: bool = False):
        print(self.num_iterations)

        if self.num_iterations == 0:
            self._initialize_members(new_frame)
            self.num_iterations += 1
            return

        assert len(new_frame.shape) == 3

        for row in range(new_frame.shape[0]):
            for column in range(new_frame.shape[1]):
                for channel in range(new_frame.shape[2]):
                    distances_, deltas = self._calc_distances_and_deltas(new_frame, position=(row, column, channel))
                    min_distance_index = np.min(distances_, axis=-1)

                    indices_far_from_modes = np.asarray(min_distance_index > self.var_threshold)
                    indices_close_to_modes = np.asarray(min_distance_index <= self.var_threshold)

                    self._add_new_modes(new_frame, indices_far_from_modes)
                    self._update_existing_modes(distances_, deltas, indices_close_to_modes)

        if plot:
            plot_gaussians(self.weights, self.means, self.variances)

        self.num_iterations += 1

        print('means:', self.means)
        print('variances:', self.variances)
        print()

    def apply(self, new_frame: np.ndarray) -> np.ndarray:
        self.update_background_model(new_frame)

        result_image = np.zeros_like(new_frame)

        for row in range(new_frame.shape[0]):
            for column in range(new_frame.shape[1]):
                for channel in range(new_frame.shape[2]):
                    result_image[row, column, channel] = self._is_background(new_frame, position=(row, column, channel))

        return result_image

    def _is_background(self, new_frame: np.ndarray, position: Tuple[int, int, int]) -> bool:
        distances, _ = self._calc_distances_and_deltas(new_frame, position)

        if np.min(distances) > self.var_threshold:
            return False

        return True

    def _initialize_members(self, first_frame):
        self.means = [[[np.asarray([pixel_value]) for pixel_value in channel] for channel in row] for row in
                      first_frame]
        self.variances = [[[np.asarray([self.var_default]) for _ in channel] for channel in row] for row in first_frame]
        self.weights = [[[np.asarray([1]) for _ in channel] for channel in row] for row in first_frame]

    def _calc_distances_and_deltas(self, new_frame: np.ndarray, position: Tuple[int, int, int]) -> \
            Tuple[np.ndarray, np.ndarray]:
        deltas = new_frame[position] - self.means[position[0]][position[1]][position[2]]
        distances = calc_mahalanobis_distances(deltas, self.variances[position[0]][position[1]][position[2]])

        return distances, deltas

    def _add_new_modes(self, new_frame: np.ndarray, indices: np.ndarray):
        for index in indices:
            self.weights[index[0]][index[1]][index[2]].append(self.alpha)
            self.means[index[0]][index[1]][index[2]].append(new_frame[index[0]][index[1]][index[2]])
            self.variances[index[0]][index[1]][index[2]].append(self.var_default)

    def _update_existing_modes(self, distances_: np.ndarray, deltas: np.ndarray, indices: np.ndarray):
        ownerships = np.zeros_like(distances_)
        ownerships[indices] = 1
        self.weights += self.alpha * (ownerships - self.weights)
        self.means += ownerships * self.alpha / (self.weights + self.epsilon) * deltas
        self.variances += ownerships * self.alpha / (self.weights + self.epsilon) * (deltas ** 2 - self.variances)

    @staticmethod
    def _update_array(array: np.ndarray, new_values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        pass


if __name__ == '__main__':

    s = []
    for i in range(1000):
        s.append(get_random_from_2_gaussians())

    mog = MOG()

    for value in s:
        mog.update_background_model(value, plot=False)

    check_list = [-5, 0, 2, 4, 5, 7]

    for value in check_list:
        print(mog._is_background(value))
