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
        center: Optional[int] = False,
        features: bool = True,
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
            X.shape[1] == Y.shape[1]
        ), f"Samples of X ({X.shape[1]}) and Samples of Y ({Y.shape[1]}) are not the same"

        self.n_samples = X.shape[0]
        self.dx_dimensions = X.shape[1]
        self.dy_dimensions = Y.shape[1]

        # subsample data if necessary
        if self.subsample is not None:
            X = self.rng.permutation(X)[: self.subsample, :]
            Y = self.rng.permutation(Y)[: self.subsample, :]

        self.X_train_ = X
        self.Y_train_ = Y

        # compute covariance
        C_xy = covariance(X, Y)
        C_xx = covariance(X)
        C_yy = covariance(Y)

        # Center Kernel
        # H = np.eye(n_samples) - (1 / n_samples) * np.ones(n_samples)
        # K_xc = K_x @ H
        if self.center == True:
            C_xy = KernelCenterer().fit_transform(C_xy)
            C_xy = KernelCenterer().fit_transform(C_xy)
            C_xy = KernelCenterer().fit_transform(C_xy)

        self.C_xy = C_xy
        self.C_xx = C_xx
        self.C_yy = C_yy

        # Compute covariance value
        self.C_xy_norm = np.sum(C_xy ** 2)

        # Compute normalization
        self.C_xx_norm = np.linalg.norm(C_xx)
        self.C_yy_norm = np.linalg.norm(C_yy)

        return self

    def score(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, normalized: bool = False
    ):
        """This is not needed. It's only needed to comply with sklearn API.
        
        We will use the target kernel alignment algorithm as a score
        function. This can be used to find the best parameters."""

        return self.C_xy_norm / self.C_xx_norm / self.C_yy_norm


# TODO: Testing - check size constraint inputs to fail
# TODO: Testing - check size constraint outputs
# TODO: Testing - self covariance v.s. cross covariance
# TODO: Testing - check 1D case
# TODO: Testing - check 2D case
# TODO: Testing - check 3D case fails
# TODO: Testing - check bias
# TODO: Documentation


def covariance(
    X: np.ndarray, Y: Optional[np.ndarray] = None, bias: bool = False
) -> np.ndarray:

    if Y is None:
        Y = X

    # check space arguments
    msg = f"# X features '{X.shape[1]}' is not equal to # X features '{Y.shape[1]}'."
    assert X.shape[1] == Y.shape[1], msg

    # Remove the Mean
    X -= np.mean(X, axis=0)[None, :]
    Y -= np.mean(Y, axis=0)[None, :]

    # calculate covariance
    covar = X.T @ Y

    # normalize data
    if bias == True:
        covar *= 1 / (covar.shape[0] - 1)

    else:
        covar *= 1 / covar.shape[0]

    return covar


def demo():

    # fix random seed
    np.random.seed(123)
    n_samples = 1000
    n_features = 50
    A = np.random.rand(n_features, n_features)

    X = np.random.randn(n_samples, n_features)
    Y = X @ A

    print("RV Method (centered):")
    # initialize method
    clf_linear_rv = LinearRV()

    # fit method
    clf_linear_rv.fit(X, Y)

    # calculate scores
    rv_score = clf_linear_rv.score(X, Y)

    print(f"RV Score: {rv_score:.6f}")
    print(f"||C_xy||: {clf_linear_rv.C_xy_norm:.6f}")
    print(f"||C_xx||: {clf_linear_rv.C_xx_norm:.6f}")
    print(f"||C_xx||: {clf_linear_rv.C_yy_norm:.6f}")


if __name__ == "__main__":
    demo()
