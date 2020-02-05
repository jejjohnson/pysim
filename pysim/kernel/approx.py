from typing import Optional

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_array, check_random_state


class RandomFourierFeatures(BaseEstimator, TransformerMixin):
    """Random Fourier Features Kernel Matrix Approximation
    Author: J. Emmanuel Johnson
    Email : jemanjohnson34@gmail.com
            emanjohnson91@gmail.com
    Date  : 3rd - August, 2018
    """

    def __init__(self, n_components=50, gamma=None, random_state=None):
        self.gamma = gamma
        # Dimensionality D (number of MonteCarlo samples)
        self.n_components = n_components
        self.rng = check_random_state(random_state)
        self.fitted = False

    def fit(self, X, y=None):
        """ Generates MonteCarlo random samples """
        X = check_array(X, ensure_2d=True, accept_sparse="csr")

        n_features = X.shape[1]

        # Generate D iid samples from p(w)
        self.weights = (2 * self.gamma) * self.rng.normal(
            size=(n_features, self.n_components)
        )

        # Generate D iid samples from Uniform(0,2*pi)
        self.bias = 2 * np.pi * self.rng.rand(self.n_components)

        # set fitted flag
        self.fitted = True
        return self

    def transform(self, X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        if not self.fitted:
            raise NotFittedError(
                "RBF_MonteCarlo must be fitted beform computing the feature map Z"
            )
        # Compute feature map Z(x):
        Z = np.dot(X, self.weights) + self.bias[np.newaxis, :]

        np.cos(Z, out=Z)

        Z *= np.sqrt(2 / self.n_components)

        return Z

    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        if not self.fitted:
            raise NotFittedError(
                "RBF_MonteCarlo must be fitted beform computing the kernel matrix"
            )
        Z = self.transform(X)

        return np.dot(Z, Z.T)
