import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


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
