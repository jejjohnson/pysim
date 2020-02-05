from typing import Callable, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import linear_kernel, pairwise_kernels
from sklearn.preprocessing import KernelCenterer
from sklearn.utils import check_array, check_random_state


class LinearRV(BaseEstimator):
    """Linear RV Coefficient. This method transforms the data to the sample 
    space and computes the similarity. Options to center (recommended) and
    normalize will change the configuration.
    
    Parameters
    ----------    
    center : bool, default=True
        The option to center the kernel matrices after construction

    subsample : int, default=None
        The option to subsample the data.

    bias : bool, default=True
        Bias term for the RV coefficient
        
    random_state : int, default=123
        The random state for the subsample.

    Attributes
    ----------
    rv_value : float
        The rv value is scored after fitting.

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 5-Feb-2020
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        center: Optional[int] = True,
        subsample: Optional[int] = None,
        bias: bool = True,
    ):

        self.random_state = random_state
        self.center = center
        self.rng = check_random_state(random_state)
        self.subsample = subsample
        self.bias = bias
        self.normalized = False

    def fit(self, X, Y):

        # Check sizes of X, Y
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)

        # Check samples are the same
        assert (
            X.shape[0] == Y.shape[0]
        ), f"Samples of X ({X.shape[0]}) and Samples of Y ({Y.shape[0]}) are not the same"

        self.n_samples = X.shape[0]
        self.dx_dimensions = X.shape[1]
        self.dy_dimensions = Y.shape[1]

        # subsample data if necessary
        if self.subsample is not None:
            X = self.rng.permutation(X)[: self.subsample, :]
            Y = self.rng.permutation(Y)[: self.subsample, :]

        self.X_train_ = X
        self.Y_train_ = Y

        # Calculate Kernel Matrices
        K_x = linear_kernel(X,)
        K_y = linear_kernel(Y,)

        # Center Kernel
        # H = np.eye(n_samples) - (1 / n_samples) * np.ones(n_samples)
        # K_xc = K_x @ H
        if self.center == True:
            K_x = KernelCenterer().fit_transform(K_x)
            K_y = KernelCenterer().fit_transform(K_y)

        self.K_x = K_x
        self.K_y = K_y

        # Compute covariance value
        self.hsic_value = np.sum(K_x * K_y)

        # Compute normalization
        self.K_x_norm = np.linalg.norm(self.K_x)
        self.K_y_norm = np.linalg.norm(self.K_y)
        return self

    def score(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, normalized: bool = False
    ):
        """This is not needed. It's only needed to comply with sklearn API.
        
        We will use the target kernel alignment algorithm as a score
        function. This can be used to find the best parameters."""

        if normalized == True:

            # Compute and cache normalized coefficients
            return self.hsic_value / self.K_x_norm / self.K_y_norm

        elif normalized == False:

            if self.bias:
                self.hsic_bias = 1 / (self.n_samples ** 2)
            else:
                self.hsic_bias = 1 / (self.n_samples - 1) ** 2

            return self.hsic_bias * self.hsic_value
        else:
            raise ValueError(f"Unrecognized normalize argument: {normalized}")


def demo():

    # fix random seed
    np.random.seed(123)
    n_samples = 1000
    n_features = 2
    A = np.random.rand(n_features, n_features)

    X = np.random.randn(n_samples, n_features)
    Y = X @ A

    print("Centered Method:")
    # initialize method
    clf_linear_rv = LinearRV()

    # fit method
    clf_linear_rv.fit(X, Y)

    # calculate scores
    rv_score = clf_linear_rv.score(X, Y, normalized=False)
    nrv_score = clf_linear_rv.score(X, Y, normalized=True)

    print(f"RV Score: {rv_score:.6f}\nNormalized RV Score: {nrv_score:.6f}.")

    print("\nUncentered Method:")
    # initialize method
    clf_linear_rv = LinearRV(center=False)

    # fit method
    clf_linear_rv.fit(X, Y)

    # calculate scores
    rv_score = clf_linear_rv.score(X, Y, normalized=False)
    nrv_score = clf_linear_rv.score(X, Y, normalized=True)

    print(f"RV Score: {rv_score:.6f}\nNormalized RV Score: {nrv_score:.6f}.")


if __name__ == "__main__":
    demo()
