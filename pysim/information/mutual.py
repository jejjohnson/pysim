import numpy as np
from typing import Optional, Callable
from .entropy import univariate_entropy, multivariate_entropy
from sklearn.utils import check_array


def mutual_information(X: np.ndarray, f: Callable, **kwargs) -> float:

    assert X.shape[1] == 2

    marginal_h = 0.0
    for ix in X.T:
        marginal_h += f(ix, **kwargs)

    # H = sum h_i - H(X, ...)
    h = marginal_h - f(X, **kwargs)

    return float(h)


def multivariate_mutual_information(
    X: np.ndarray, Y: np.ndarray, f: Callable, **kwargs
) -> float:

    # Marginal Entropy
    H_x = f(X, **kwargs)
    H_y = f(Y, **kwargs)
    H_marg = H_x + H_y

    # Joint Entropy
    XY = np.concatenate([X, Y], axis=1)

    H_xy = f(XY, **kwargs)

    # H = sum h_i - H(X, ...)
    mi = H_marg - H_xy

    return {
        "mi": mi,
        "H_xy": H_xy,
        "H_marg": H_marg,
        "H_x": H_x,
        "H_y": H_y,
    }


def total_correlation(X: np.ndarray, f: Callable, **kwargs) -> float:

    marginal_h = 0.0
    for ix in X.T:
        marginal_h += f(ix, **kwargs)

    # H = sum h_i - H(X, ...)
    h = marginal_h - f(X, **kwargs)

    return float(h)
