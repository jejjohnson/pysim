from typing import List, Optional, Union, NamedTuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import check_array, check_random_state
import collections


class GammaParam(NamedTuple):
    """Helpful data holder which stores gamma parameters

    This allows me to iterate really quickly
    
    Parameters
    ----------
    method : str, default='median'
        the method to estimate the gamma
    
    percent : float, (optional, default=None)
    
    scale : float, (optional, default=None)
        the mutual information value

    Example
    -------
    >> from pysim.kernel.utils import GammaParam
    >> from sklearn import datasets
    >> X, _ = datasets.make_blobs(n_samples=1_000, n_features=2, random_state=123)
    >> gamma_estimator = GammaParam(method='median', percent=None, scale=None)
    >> gamma_X = gamma_estimator.estimate_gamma(X)
    >> print(gamma_X)
    0.2747661935717474
    """

    method: str = "median"
    percent: Optional[Union[float, int]] = None
    scale: Optional[Union[float, int]] = None

    def estimate_gamma(self, X: np.ndarray, **kwargs) -> float:
        """Estimate the gamma parameter from params
        
        Parameters
        ----------
        X : np.ndarray
            the data to estimate the gamma

        kwargs : dict, optional
            any extra keyword arguments to input into the 
            sigma estimator
        
        Returns
        -------
        gamma_est : float
            the estimated gamma value
        """
        return estimate_gamma(X, **kwargs)


class SigmaParam(NamedTuple):
    """Helpful data holder which stores:
    
    method : str, default='median'
        the method to estimate the sigma
    
    percent : float, (optional, default=None)
    
    scale : float, (optional, default=None)
        the mutual information value"""

    method: str = "median"
    percent: Optional[Union[float, int]] = None
    scale: Optional[Union[float, int]] = None

    def estimate_sigma(self, X: np.ndarray, **kwargs) -> float:
        """Estimate the sigma parameter from params
        
        Parameters
        ----------
        X : np.ndarray
            the data to estimate the sigma

        kwargs : dict, optional
            any extra keyword arguments to input into the 
            sigma estimator
        
        Returns
        -------
        sigma_est : float
            the estimated sigma value
        """
        return estimate_sigma(X, **kwargs)


def estimate_gamma(
    X: np.ndarray,
    subsample: Optional[int] = None,
    method: str = "median",
    percent: Optional[float] = 0.15,
    scale: float = 1.0,
    random_state: Optional[int] = None,
) -> float:

    init_sigma = estimate_sigma(
        X=X,
        subsample=subsample,
        method=method,
        percent=percent,
        scale=scale,
        random_state=random_state,
    )
    return sigma_to_gamma(init_sigma)


def estimate_sigma(
    X: np.ndarray,
    subsample: Optional[int] = None,
    method: str = "median",
    percent: Optional[float] = 0.15,
    scale: float = 1.0,
    random_state: Optional[int] = None,
) -> float:
    """A function to provide a reasonable estimate of the sigma values
    for the RBF kernel using different methods. 

    Parameters
    ----------
    X : array, (n_samples, d_dimensions)
        The data matrix to be estimated.
    
    method : str, default: 'median'
        different methods used to estimate the sigma for the rbf kernel
        matrix.
        * Mean
        * Median
        * Silverman
        * Scott - very common for density estimation
    percent : float, default=0.15
        The kth percentage of distance chosen
    
    scale : float, default=None
        Option to scale the sigma chosen. Typically used with the
        median or mean method as they are data dependent.
    
    random_state : int, (default: None)
        controls the seed for the subsamples drawn to represent
        the data distribution
    
    Returns
    -------
    sigma : float
        The estimated sigma value
        
    Resources
    ---------
    - Original MATLAB function: https://goo.gl/xYoJce
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
           : juan.johnson@uv.es
    Date   : 6 - July - 2018
    """
    X = check_array(X, ensure_2d=True)

    rng = check_random_state(random_state)

    # subsampling
    [n_samples, d_dimensions] = X.shape

    if subsample is not None:
        X = rng.permutation(X)[:subsample, :]

    if method == "mean":
        if percent is None:
            sigma = np.mean(pdist(X))
        else:
            kth_sample = int(percent * n_samples)
            sigma = np.mean(np.sort(squareform(pdist(X)))[:, kth_sample])

    elif method == "median":
        if percent is None:
            sigma = np.median(pdist(X))
        else:
            kth_sample = int(percent * n_samples)
            sigma = np.median(np.sort(squareform(pdist(X)))[:, kth_sample])

    elif method == "silverman":
        sigma = np.power(
            n_samples * (d_dimensions + 2.0) / 4.0, -1.0 / (d_dimensions + 4)
        )

    elif method == "scott":
        sigma = np.power(n_samples, -1.0 / (d_dimensions + 4))

    else:
        raise ValueError('Unrecognized mode "{}".'.format(method))

    # scale the sigma by a factor
    if scale is not None:
        sigma *= scale

    # return sigma
    return sigma


def get_sigma_grid(
    init_sigma: float = 1.0, factor: int = 2, n_grid_points: int = 20
) -> List[float]:
    """Get a standard parameter grid for the cross validation strategy.
    
    Parameters
    ----------
    init_sigma : float, default=1.0
        The initial sigma to use to populate the grid points.
    
    factor : int, default=2
        The log scale factor to use for both the beginning and end of the grid.
    
    n_grid_points : int, default=20
        The number of grid points to use.
    
    Returns
    -------
    param_grid : List[float]
        The parameter grid as per the specifications

    Example
    -------
    >> param_grid = get_param_grid()

    >> param_grid = get_param_grid(10.0, 3, 1_000)
    """

    # create bounds for search space (logscale)
    init_space = 10 ** (-factor)
    end_space = 10 ** (factor)

    # create param grid
    param_grid = np.logspace(
        np.log10(init_sigma * init_space),
        np.log10(init_sigma * end_space),
        n_grid_points,
    )

    return param_grid


def get_gamma_grid(
    init_gamma: float = 1.0, factor: int = 2, n_grid_points: int = 20
) -> List[float]:

    # convert to sigma
    init_sigma = gamma_to_sigma(init_gamma)

    # get sigma grid
    sigma_grid = get_sigma_grid(init_sigma, factor=factor, n_grid_points=n_grid_points)

    # convert to gamma
    gamma_grid = gamma_to_sigma(sigma_grid)

    return gamma_grid


def get_init_gammas(X, Y=None, method="median", percent=None, scale=1.0):

    # Estimate Sigma
    sigma_x = estimate_sigma(X, method=method, percent=percent, scale=scale)

    # convert to gamma
    gamma_x = sigma_to_gamma(sigma_x)

    if Y is None:
        return gamma_x
    else:
        # estimate sigma for X
        sigma_y = estimate_sigma(Y, method=method, percent=percent, scale=scale)
        # convert to gamma
        gamma_y = sigma_to_gamma(sigma_y)
        return gamma_x, gamma_y


def gamma_to_sigma(gamma: float) -> float:
    """Transforms the gamma parameter into sigma using the 
    following relationship:
       
                         1
        sigma =  -----------------
                 sqrt( 2 * gamma )
    """
    return 1 / np.sqrt(2 * gamma)


def sigma_to_gamma(sigma: float) -> float:
    """Transforms the sigma parameter into gamma using the 
    following relationship:
       
                      1
         gamma = -----------
                 2 * sigma^2
    """
    return 1 / (2 * sigma ** 2)


from typing import Tuple, Optional, Callable
from sklearn.utils import check_random_state
import numpy as np
from sklearn.utils import gen_batches, check_array
from joblib import Parallel, delayed


def subset_indices(
    X: np.ndarray, subsample: Optional[int] = None, random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:

    if subsample is not None and subsample < X.shape[0]:
        rng = check_random_state(random_state)
        indices = np.arange(X.shape[0])
        subset_indices = rng.permutation(indices)[:subsample]
        X = X[subset_indices, :]

    return X


def subsample_data(X, n_samples: int = 1_000, seed: int = 123):

    rng = np.random.RandomState(seed)

    if n_samples < X.shape[0]:

        idx = rng.permutation(n_samples)[:n_samples]

        X = X[idx]

    return X
