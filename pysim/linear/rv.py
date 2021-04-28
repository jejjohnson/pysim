from typing import Callable, Optional, Union, Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import linear_kernel, pairwise_kernels
from sklearn.preprocessing import KernelCenterer
from sklearn.utils import check_array, check_random_state
from pysim.kernel.hsic import hsic_v_statistic
from pysim.kernel.utils import subsample_data


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> Dict:
    """Linear RV Coefficient. This method transforms the data to the sample 
    space and computes the similarity. Options to center (recommended) and
    normalize will change the configuration.
    
    Parameters
    ----------    
    X : np.ndarray
        The first input array
    Y : np.ndarray
        The second input array

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
    # calculate the kernel matrices
    Kx = linear_kernel(X)
    Ky = linear_kernel(Y)

    # frobenius norm of the cross terms (numerator)
    xy_norm = hsic_v_statistic(Kx, Ky)

    # normalizing coefficients (denomenator)
    x_norm = np.sqrt(hsic_v_statistic(Kx, Kx))
    y_norm = np.sqrt(hsic_v_statistic(Ky, Ky))

    # rv coefficient
    rv_coeff = xy_norm / x_norm / y_norm

    return {
        "rv_coeff": rv_coeff,
        "rv_xy_norm": xy_norm,
        "rv_x_norm": x_norm,
        "rv_y_norm": y_norm,
    }
