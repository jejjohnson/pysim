import numpy as np
from scipy import stats
from typing import Union, Optional, Dict
from .histogram import hist_entropy
from .knn import knn_entropy
from .kde import kde_entropy_uni
from .gaussian import gauss_entropy_uni, gauss_entropy_multi
from sklearn.utils import check_array


def univariate_entropy(X: np.ndarray, method: str = "histogram", **kwargs) -> float:
    """Calculates the entropy given the method initiali"""
    # check input array
    X = check_array(X, ensure_2d=True)

    n_samples, n_features = X.shape

    msg = "n_features is greater than 1. Please use Multivariate instead."
    assert 1 == n_features, msg

    if method == "histogram":
        return hist_entropy(X, **kwargs)

    elif method == "knn":
        return knn_entropy(X, **kwargs)

    elif method == "kde":
        return kde_entropy_uni(X, **kwargs)

    elif method in ["gauss", "gaussian"]:
        return gauss_entropy_uni(X)
    else:
        raise ValueError(f"Unrecognized method: {method}")


def marginal_entropy(X: np.ndarray, method: str = "histogram", **kwargs) -> np.ndarray:
    # check input array
    X = check_array(X, ensure_2d=True)

    n_samples, n_features = X.shape

    H_entropy = np.empty(n_features)
    for i, ifeature in enumerate(X.T):
        H_entropy[i] = univariate_entropy(X.T[i][:, None], method, **kwargs)

    return H_entropy


def multivariate_entropy(X: np.ndarray, method: str = "knn", **kwargs) -> float:
    if method == "knn":
        return knn_entropy(X, **kwargs)

    elif method in ["gauss", "gaussian"]:
        return gauss_entropy_multi(X)
    else:
        raise ValueError(f"Unrecognized method: {method}")
