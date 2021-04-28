from typing import Optional
import numpy as np
from scipy.spatial.distance import squareform, pdist


def scotts_factor(X: np.ndarray) -> float:
    """Scotts Method to estimate the length scale of the 
    rbf kernel.

        factor = n**(-1./(d+4))

    Parameters
    ----------
    X : np.ndarry
        Input array

    Returns
    -------
    factor : float
        the length scale estimated

    """
    n_samples, n_features = X.shape

    return np.power(n_samples, -1.0 / (n_features + 4.0))


def silvermans_factor(X: np.ndarray) -> float:
    """Silvermans method used to estimate the length scale
    of the rbf kernel.

    factor = (n * (d + 2) / 4.)**(-1. / (d + 4)).

    Parameters
    ----------
    X : np.ndarray,
        Input array

    Returns
    -------
    factor : float
        the length scale estimated
    """
    n_samples, n_features = X.shape

    base = (n_samples * (n_features + 2.0)) / 4.0

    return np.power(base, -1 / (n_features + 4.0))


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
