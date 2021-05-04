import numpy as np
from scipy import stats
from typing import Union, Optional, Dict, Callable
from pysim.information.histogram import hist_entropy
from pysim.information.knn import knn_entropy
from pysim.information.kde import kde_entropy_uni
from pysim.information.gaussian import gauss_entropy_uni, gauss_entropy_multi
from sklearn.utils import check_array


def marginal_entropy(X: np.ndarray, estimator: Callable, **kwargs) -> float:
    # check input array
    X = check_array(X, ensure_2d=True)

    n_samples, n_features = X.shape

    H_entropy = np.empty(n_features)

    # loop through features
    for i, ifeature in enumerate(X.T):
        H_entropy[i] = estimator(ifeature, **kwargs)

    return H_entropy

