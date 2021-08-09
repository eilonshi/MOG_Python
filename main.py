import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

num_of_gaussians = 1
relevant_frames = 50
alpha = 1 / relevant_frames
epsilon = 1e-5
distance_threshold = 5.
var_default = 1.


def plot_gaussians(weights_, means_, variances_):
    sigmas = np.sqrt(variances_)

    for weight_, mean_, sigma_ in zip(weights_, means_, sigmas):
        x = np.linspace(mean_ - 3 * sigma_, mean_ + 3 * sigma_, 100)
        plt.plot(x, weight_ * stats.norm.pdf(x, mean_, sigma_))

    plt.show()


def get_random_from_2_gaussians():
    choice = np.random.choice([0, 1])

    if choice == 0:
        mean, sigma = 0, 0.1  # mean and standard deviation
    else:
        mean, sigma = 5, 0.5  # mean and standard deviation

    return np.random.normal(mean, sigma, 1)


def calc_mahalanobis_distances(deltas, variances):
    return np.sqrt(deltas ** 2 / (variances + epsilon))


class MOG:

    def __init__(self):
        self.weights = np.zeros(num_of_gaussians) + 1
        self.means = np.zeros(num_of_gaussians) + 20
        self.variances = np.asarray([var_default])

    def update_background_model(self, new_value, plot=False):
        deltas = new_value - self.means

        distances_ = calc_mahalanobis_distances(deltas, self.variances)
        min_distance_index = np.argmin(distances_)

        if distances_[min_distance_index] > distance_threshold:
            self.weights = np.append(self.weights, alpha)
            self.means = np.append(self.means, new_value)
            self.variances = np.append(self.variances, var_default)
        else:
            ownerships = np.zeros_like(distances_)
            ownerships[min_distance_index] = 1
            self.weights += alpha * (ownerships - self.weights)
            self.means += ownerships * alpha / (self.weights + epsilon) * deltas
            self.variances += ownerships * alpha / (self.weights + epsilon) * (deltas ** 2 - self.variances)

        if plot:
            plot_gaussians(self.weights, self.means, self.variances)

        print('means:', self.means)
        print('variances:', self.variances)
        print()

    def is_background(self, new_value):
        deltas = new_value - self.means
        distances = calc_mahalanobis_distances(deltas, self.variances)

        if np.min(distances) > distance_threshold:
            return False

        return True


if __name__ == '__main__':

    s = []
    for i in range(1000):
        s.append(get_random_from_2_gaussians())

    mog = MOG()

    for value in s:
        mog.update_background_model(value, plot=False)

    check_list = [-5, 0, 2, 4, 5, 7]

    for value in check_list:
        print(mog.is_background(value))
