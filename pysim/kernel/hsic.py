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
    alpha : {float, array-like}, shape = [n_targets]
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.  Alpha corresponds to
        ``(2*C)^-1`` in other linear models such as LogisticRegression or
        LinearSVC. If an array is passed, penalties are assumed to be specific
        to the targets. Hence they must correspond in number.
    
    center : bool, default=True
        The option to center the kernel matrices after construction

    subsample : int, default=None
        The option to subsample the data.

    random_state : int, default=123
        The random state for the subsample.

    Attributes
    ----------
    rv_value : float
        The HSIC value is scored after fitting.
        
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 14-Feb-2019
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
        K_x = self.linear_kernel(X,)
        K_y = self.linear_kernel(Y,)

        # Center Kernel
        # H = np.eye(n_samples) - (1 / n_samples) * np.ones(n_samples)
        # K_xc = K_x @ H
        if self.centered == True:
            K_x = KernelCenterer().fit_transform(K_x)
            K_y = KernelCenterer().fit_transform(K_y)

        # Compute covariance value
        self.hsic_value = np.sum(K_x * K_y)

        return self

    def score(self, X, y=None, normalize=False):
        """This is not needed. It's only needed to comply with sklearn API.
        
        We will use the target kernel alignment algorithm as a score
        function. This can be used to find the best parameters."""

        if normalize == True:

            # Compute and cache normalized coefficients
            if self.normalized = False:

                # Re-Normalize Data
                self.K_x_norm = np.linalg.norm(K_x)
                self.K_y_norm = np.linalg.norm(K_y)

                self.normalized = True
            
            return self.hsic_value / self.K_x_norm / self.K_y_norm

        elif normalize == False:

            if self.bias:
                self.hsic_bias = 1 / (self.n_samples ** 2)
            else:
                self.hsic_bias = 1 / (self.n_samples - 1) ** 2

            return self.hsic_bias * self.hsic_value
        else:
            raise ValueError(f"Unrecognized normalize argument: {normalize}")


def kernel_alignment(K_x: np.array, K_y: np.array, center: bool = False) -> float:
    """Gives a target kernel alignment score: how aligned the kernels are. Very
    useful for measures which depend on comparing two different kernels, e.g.
    Hilbert-Schmidt Independence Criterion (a.k.a. Maximum Mean Discrepency)
    
    Note: the centered target kernel alignment score is the same function
          with the center flag = True.
    
    Parameters
    ----------
    K_x : np.array, (n_samples, n_samples)
        The first kernel matrix, K(X,X')
    
    K_y : np.array, (n_samples, n_samples)
        The second kernel matrix, K(Y,Y')
        
    center : Bool, (default: False)
        The option to center the kernels (independently) before hand.
    
    Returns
    -------
    kta_score : float,
        (centered) target kernel alignment score.
    """

    # center kernels
    if center:
        K_x = KernelCenterer().fit_transform(K_x)
        K_y = KernelCenterer().fit_transform(K_y)

    # target kernel alignment
    return np.sum(K_x * K_y) / np.linalg.norm(K_x) / np.linalg.norm(K_y)


class HSIC(BaseEstimator):
    """Hilbert-Schmidt Independence Criterion (HSIC). This is
    a method for measuring independence between two variables.
    
    Parameters
    ----------
    alpha : {float, array-like}, shape = [n_targets]
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.  Alpha corresponds to
        ``(2*C)^-1`` in other linear models such as LogisticRegression or
        LinearSVC. If an array is passed, penalties are assumed to be specific
        to the targets. Hence they must correspond in number.
    
    kernel : string or callable, default="linear"
        Kernel mapping used internally. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number. Set to "precomputed" in
        order to pass a precomputed kernel matrix to the estimator
        methods instead of samples.
    
    gamma_X : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Used only by the X parameter. 
        Interpretation of the default value is left to the kernel; 
        see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.
    gamma_Y : float, default=None
        The same gamma parameter as the X. If None, the same gamma_X will be
        used for the Y.
    
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    
    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.
    
    scorer : str, default='ctka'
        The method to score how well the sigma fits the two datasets.
        
        The following options are:
        * 'ctka': centered target kernel alignment
        * 'tka' : target kernel alignment
        * 'hsic': the hsic value
    
    random_state : str
    Attributes
    ----------
    hsic_value : float
        The HSIC value is scored after fitting.
        
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 14-Feb-2019
    """

    def __init__(
        self,
        gamma_X: float = 1.0,
        gamma_Y: Optional[None] = None,
        kernel: Union[Callable, str] = "linear",
        degree: float = 3,
        coef0: float = 1,
        kernel_params: Optional[dict] = None,
        random_state: Optional[int] = None,
        centered: Optional[int] = True,
        normalized: Optional[bool] = True,
        subsample: Optional[int] = None,
        bias: bool = True,
    ):
        self.gamma_X = gamma_X
        self.gamma_Y = gamma_Y
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.scorer = scorer
        self.subsample = subsample
        self.bias = bias

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
        K_x = self.compute_kernel(X, gamma=self.gamma_X)
        K_y = self.compute_kernel(Y, gamma=self.gamma_Y)

        # Center Kernel
        # H = np.eye(n_samples) - (1 / n_samples) * np.ones(n_samples)
        # K_xc = K_x @ H
        if self.centered == True:
            K_x = KernelCenterer().fit_transform(K_x)
            K_y = KernelCenterer().fit_transform(K_y)

        # Compute HSIC value
        self.hsic_value = self._calculate_hsic(K_x, K_y)

        return self

    def compute_kernel(self, X, Y=None, gamma=1.0):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _calculate_hsic(self, K_x, K_y):
        if self.scorer == "tka":
            return kernel_alignment(K_x, K_y, center=False)

        elif self.scorer == "ctka":
            return kernel_alignment(K_x, K_y, center=False)

        elif self.scorer == "hsic":

            if self.bias:
                self.hsic_bias = 1 / (self.n_samples ** 2)
            else:
                self.hsic_bias = 1 / (self.n_samples - 1) ** 2

            return self.hsic_bias * np.sum(K_x * K_y)
        else:
            raise ValueError(f"Unrecognized scorer: {self.scorer}")

    def score(self, X, y=None):
        """This is not needed. It's only needed to comply with sklearn API.
        
        We will use the target kernel alignment algorithm as a score
        function. This can be used to find the best parameters."""

        return self.hsic_value
