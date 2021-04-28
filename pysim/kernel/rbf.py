from typing import Optional
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.gaussian_process.kernels import RBF
from pysim.kernel.utils import subsample_data


def init_rbf_kernel(n_sub_samples: int = 1_000, seed: int = 123):
    def k(X, Y):

        # subsample data

        sigma = sigma_median_heuristic(
            subsample_data(X, n_sub_samples, seed),
            subsample_data(Y, n_sub_samples, seed),
        )

        gamma = sigma_to_gamma(sigma)

        # calculate kernel

        return rbf_kernel(X, Y, gamma=gamma)

    return k


def sigma_median_heuristic(
    X: np.ndarray, Y: Optional[np.ndarray] = None, sqrt: bool = True
) -> np.ndarray:

    if Y is None:
        dists = squareform(pdist(np.concatenate([X, Y], axis=0), metric="euclidean"))
    else:
        dists = squareform(pdist(X, metric="euclidean"))

    median_dist = np.median(dists[dists > 0])

    if sqrt:
        sigma = median_dist / np.sqrt(2.0)
    else:
        sigma = median_dist / 2.0

    return sigma


def sigma_to_gamma(sigma):
    return 1 / (2 * sigma ** 2)


def gamma_to_sigma(gamma):
    return 1 / np.sqrt(2 * gamma)
