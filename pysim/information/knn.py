from typing import Optional, Union, Dict, List, Iterable
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.special import gamma, digamma
from sklearn.utils import check_array, check_random_state

NOISE = 1e-10


def knn_entropy_npeet(
    X: np.ndarray, n_neighbors: int = 3, seed: int = 123, base: int = 2, **kwargs
):
    """Entropy Estimation from the NPEET Toolbox
    Claims of a more numerically stable version

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        The data to find the nearest neighbors for.

    n_neighbors : int, default=10
        The number of nearest neighbors to find.

    seed : int, default=123
        the random seed for the slight permutation of the data.
        claims to break degeneracies
    
    base : int, default=2
        the base for the entropy metric


    Returns
    -------
    H : float
        the entropy

    References
    ----------
    [1]: https://github.com/gregversteeg/NPEET
    """
    # add noise to break degeneracy
    X += np.random.RandomState(seed).random_sample(X.shape)

    # choose distance method
    if X.shape[1] >= 20:
        algorithm = "ball_tree"
    else:
        algorithm = "kd_tree"

    # calculate entropy
    H = knn_entropy(
        X=X,
        n_neighbors=n_neighbors,
        norm=0,
        base=base,
        algorithm=algorithm,
        metric="chebyshev",
        **kwargs,
    )
    # change of base
    H /= np.log(base)
    return H


def knn_entropy(
    X: np.ndarray,
    n_neighbors: int = 5,
    norm: int = 2,
    base: int = 2,
    algorithm: str = "kd_tree",
    n_jobs: int = -1,
    **kwargs,
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

    Resources
    =========
    [1] : 
    [2] : https://github.com/mutualinfo/mutual_info
    """
    X = check_array(X, ensure_2d=True, copy=True)

    n_samples, n_features = X.shape

    # volume of unit ball in d^n
    vol = volume_unit_ball(n_features, norm=norm)

    # calculate the K-nearest neighbors
    distances = knn_distance(
        X=X, n_neighbors=n_neighbors, algorithm=algorithm, n_jobs=n_jobs, **kwargs
    )

    # calvulate volume
    constant = -digamma(n_neighbors) + digamma(n_samples) + np.log(vol)

    # estimation
    entropy = constant + n_features * np.mean(np.log(distances))

    # # change of base
    # entropy /= np.log(base)

    return entropy


def knn_total_corr(X: np.ndarray, n_neighbors: int = 1, **kwargs) -> float:

    marginal_h = 0.0
    for ix in X.T:
        marginal_h += knn_entropy(ix, n_neighbors=n_neighbors, **kwargs)

    # H = sum h_i - H(X, ...)
    h = marginal_h - knn_entropy(X, n_neighbors=n_neighbors, **kwargs)

    return float(h)


def knn_mutual_info(
    X: np.ndarray, Y: np.ndarray, n_neighbors: int = 1, **kwargs
) -> float:

    # calculate the marginal entropy
    H_x = knn_entropy(X, n_neighbors=n_neighbors, **kwargs)
    H_y = knn_entropy(Y, n_neighbors=n_neighbors, **kwargs)
    H_marg = H_x + H_y

    # H = sum h_i - H(X, ...)
    H_xy = knn_entropy(np.hstack([X, Y]), n_neighbors=n_neighbors, **kwargs)
    knn_mi = H_marg - H_xy

    return {
        "knn_mi": knn_mi,
        "knn_H_joint": H_xy,
        "knn_H_marg": H_marg,
        "knn_H_x": H_x,
        "knn_H_y": H_y,
    }


# KNN Distances
def knn_distance(X: np.ndarray, n_neighbors: int = 20, **kwargs) -> np.ndarray:
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

    # initialize knn function
    clf_knn = NearestNeighbors(n_neighbors=n_neighbors + 1, **kwargs)

    # build tree
    clf_knn.fit(X)

    # query tree
    dists, _ = clf_knn.kneighbors(X)

    # extract dist
    dists = dists[:, n_neighbors]

    # numerical errors
    dists += np.finfo(X.dtype).eps

    return dists


# volume of unit ball
def volume_unit_ball(d_dimensions: int, norm=2) -> float:
    """Volume of the unit l_p-ball in d-dimensional

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

    References
    ----------
    [1]:    Demystifying Fixed k-Nearest Neighbor Information 
            Estimators - Gao et al (2016)
    """

    # get ball
    if norm == 0:
        return 1.0
    elif norm == 1:
        raise NotImplementedError()
    elif norm == 2:
        b = 2.0
    else:
        raise ValueError(f"Unrecognized norm: {norm}")

    numerator = gamma(1.0 + 1.0 / b) ** d_dimensions
    denomenator = gamma(1.0 + d_dimensions / b)
    vol = 2 ** d_dimensions * numerator / denomenator

    return vol
