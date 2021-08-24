import numpy as np
from typing import Tuple

from experiment.utils import plot_gaussians
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
        self.num_modes = None

    def apply(self, new_frame: np.ndarray) -> np.ndarray:
        self.update_background_model(new_frame)
        mask = (self.is_foreground(new_frame) * 255)
        print(np.sum(mask) / 255)

        return mask.astype(np.uint8)

    def is_foreground(self, new_frame: np.ndarray) -> np.ndarray:
        distances, _ = self._calc_distances_and_deltas(new_frame)

        irrelevant_indices = (np.arange(self.num_modes.max()) >= self.num_modes[..., None]).astype(int)
        distances[irrelevant_indices] = float("inf")

        mask = (np.min(distances, axis=3) > self.var_threshold).astype(np.uint8)

        return mask

    def update_background_model(self, new_frame: np.ndarray, plot: bool = False):
        print(self.num_iterations)

        if self.num_iterations == 0:
            self._initialize_members(new_frame)
            self.num_iterations += 1
            return

        distances_, deltas = self._calc_distances_and_deltas(new_frame)
        min_distance_index = np.min(distances_, axis=-1)

        indices_far_from_modes = min_distance_index > self.var_threshold
        indices_close_to_modes = min_distance_index <= self.var_threshold

        self._update_existing_modes(distances_, deltas, indices_close_to_modes)
        self._add_new_modes(new_frame, indices_far_from_modes)

        if plot:
            plot_gaussians(self.weights, self.means, self.variances)

        self.num_iterations += 1

    def _initialize_members(self, first_frame):
        self.means = first_frame.copy().astype('float64')
        self.means = np.expand_dims(self.means, axis=3)

        self.variances = (np.zeros_like(first_frame) + self.var_default).astype('float64')
        self.variances = np.expand_dims(self.variances, axis=3)

        self.weights = (np.zeros_like(first_frame) + 1).astype('float64')
        self.weights = np.expand_dims(self.weights, axis=3)

        self.num_modes = np.zeros_like(first_frame) + 1

    def _calc_distances_and_deltas(self, new_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        deltas = np.expand_dims(new_frame, axis=3) - self.means
        distances = calc_mahalanobis_distances(deltas, self.variances)

        return distances, deltas

    def _add_new_modes(self, new_frame: np.ndarray, indices: np.ndarray):
        ownerships = np.zeros_like(new_frame)
        ownerships[indices] = 1

        alpha_mask = ownerships * self.alpha
        self.weights = np.concatenate([self.weights, np.expand_dims(alpha_mask, axis=3)], axis=3)

        means_mask = ownerships * new_frame
        self.means = np.concatenate([self.means, np.expand_dims(means_mask, axis=3)], axis=3)

        var_mask = ownerships * self.var_default
        self.variances = np.concatenate([self.variances, np.expand_dims(var_mask, axis=3)], axis=3)

        self.num_modes[indices] += 1

        if np.max(self.num_modes) > self.max_num_of_gaussians:
            self._remove_worst_mode()

    def _update_existing_modes(self, distances_: np.ndarray, deltas: np.ndarray, indices: np.ndarray):
        ownerships = np.zeros_like(distances_)
        ownerships[indices] = 1

        self.weights += self.alpha * (ownerships - self.weights)
        self.means += ownerships * self.alpha / (self.weights + self.epsilon) * deltas
        self.variances += ownerships * self.alpha / (self.weights + self.epsilon) * (deltas ** 2 - self.variances)

    def _remove_worst_mode(self):
        sorted_indices = np.argsort(self.weights, axis=3)

        self.weights = self.weights[sorted_indices]
        self.weights = self.weights[:, :, :, :-1]

        self.means = self.means[sorted_indices]
        self.means = self.means[:, :, :, :-1]

        self.variances = self.variances[sorted_indices]
        self.variances = self.variances[:, :, :, :-1]

        self.num_modes[self.num_modes == np.max(self.num_modes)] -= 1
