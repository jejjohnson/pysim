from typing import Optional, Union, Dict, List, Iterable
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.special import gamma, psi
from sklearn.utils import check_array, check_random_state


def knn_entropy(
    X: np.ndarray, k: int = 5, algorithm="brute", n_jobs=1, norm: int = 2, **kwargs
):
    """Calculates the Entropy using the knn method.

    Parameters
    ----------
    X : np.ndarray, (n_samples x d_dimensions)
        The data to find the nearest neighbors for.

    k : int, default=10
        The number of nearest neighbors to find.

    algorithm : str, default='brute',
        The knn algorithm to use.
        ('brute', 'ball_tree', 'kd_tree', 'auto')

    n_jobs : int, default=-1
        The number of cores to use to find the nearest neighbors

    kwargs: dict, Optional
        any extra kwargs to use for the KNN estimator

    Returns
    -------
    H : float
        Entropy calculated from kNN algorithm
    """
    X = check_array(X, ensure_2d=True)

    # initialize KNN estimator
    n_samples, d_dimensions = X.shape

    # volume of unit ball in d^n
    # vol = (np.pi ** (0.5 * d_dimensions)) / gamma(0.5 * d_dimensions + 1)
    vol = volume_unit_ball(d_dimensions, norm=norm)

    # calculate the K-nearest neighbors
    distances = knn_distance(**kwargs)

    # estimation
    return (
        d_dimensions * np.mean(np.log(distances))
        + np.log(vol)
        + psi(n_samples)
        - psi(k)
    )


def knn_total_corr(X: np.ndarray, n_neighbours: int = 1, **kwargs) -> float:

    marginal_h = 0.0
    for ix in X.T:
        marginal_h += knn_entropy(ix, n_neighbors=n_neighbours, **kwargs)

    # H = sum h_i - H(X, ...)
    h = marginal_h - knn_entropy(X, n_neighbors=n_neighbours, **kwargs)

    return float(h)


def knn_mutual_info(
    X: np.ndarray, Y: np.ndarray, n_neighbours: int = 1, **kwargs
) -> float:

    # calculate the marginal entropy
    H_x = knn_entropy(X, n_neighbors=n_neighbours, **kwargs)
    H_y = knn_entropy(Y, n_neighbors=n_neighbours, **kwargs)
    H_marg = H_x + H_y

    # H = sum h_i - H(X, ...)
    H_xy = knn_entropy(np.hstack([X, Y]), n_neighbors=n_neighbours, **kwargs)
    knn_mi = H_marg - H_xy

    return {
        "knn_mi": knn_mi,
        "knn_H_joint": H_xy,
        "knn_H_marg": H_marg,
        "knn_H_x": H_x,
        "knn_H_y": H_y,
    }


# KNN Distances
def knn_distance(X: np.ndarray, n_neighbors: int = 20, **kwargs,) -> np.ndarray:
    """Light wrapper around sklearn library.

    Parameters
    ----------
    X : np.ndarray, (n_samples x d_dimensions)
        The data to find the nearest neighbors for.

    n_neighbors : int, default=20
        The number of nearest neighbors to find.

    algorithm : str, default='brute',
        The knn algorithm to use.
        ('brute', 'ball_tree', 'kd_tree', 'auto')

    n_jobs : int, default=-1
        The number of cores to use to find the nearest neighbors

    kwargs : dict, Optional
        Any extra keyword arguments.

    Returns
    -------
    distances : np.ndarray, (n_samples x d_dimensions)
    """
    X = check_array(X, ensure_2d=True)

    clf_knn = NearestNeighbors(n_neighbors=n_neighbors, **kwargs)

    clf_knn.fit(X)

    dists, _ = clf_knn.kneighbors(X)

    # numerical errors
    dists += np.finfo(X.dtype).eps

    return dists


# volume of unit ball
def volume_unit_ball(d_dimensions: int, norm=2) -> float:
    """Volume of the d-dimensional unit ball
    
    Parameters
    ----------
    d_dimensions : int
        Number of dimensions to estimate the volume
    
    norm : int, default=2
        The type of ball to get the volume.
        * 2 : euclidean distance
        * 1 : manhattan distance
        * 0 : chebyshev distance
    
    Returns
    -------
    vol : float
        The volume of the d-dimensional unit ball
    """

    # get ball
    if norm == 0:
        b = float("inf")
    elif norm == 1:
        b = 1.0
    elif norm == 2:
        b = 2.0
    else:
        raise ValueError(f"Unrecognized norm: {norm}")

    return (np.pi ** (0.5 * d_dimensions)) ** d_dimensions / gamma(b / d_dimensions + 1)

