import numpy as np
from typing import Optional
from .entropy import univariate_entropy, multivariate_entropy
from sklearn.utils import check_array


def univariate_mutual_info(
    X: np.ndarray, Y: np.ndarray, method: str = "knn", **kwargs
) -> float:

    # check input array
    X = check_array(X, dtype=np.float, ensure_2d=True)
    Y = check_array(Y, dtype=np.float, ensure_2d=True)

    # H(X), entropy
    H_x = univariate_entropy(X, method=method, **kwargs)

    # H(Y), entropy
    H_y = univariate_entropy(Y, method=method, **kwargs)

    # H(X,Y), joint entropy
    H_xy = multivariate_entropy(np.hstack([X, Y]), method=method, **kwargs)

    return H_x + H_y - H_xy
