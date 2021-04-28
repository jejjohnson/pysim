from typing import Callable

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from typing import Dict, Optional
import time
from scipy.spatial.distance import squareform, pdist, cdist
from pysim.kernel.rbf import init_rbf_kernel


class HSIC:
    def __init__(
        self, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.kernel = kernel

    def score_u_stat(self, X, Y, normalized: bool = True):

        # calculate kernel matrices
        Kx = self.kernel(X, X)
        Ky = self.kernel(Y, Y)

        # calculate hsic score
        hsic_score = hsic_u_statistic(Kx, Ky)

        # normalized
        if normalized:
            norm = hsic_u_statistic(Kx, Kx) * hsic_u_statistic(Ky, Ky)

            hsic_score /= np.sqrt(norm)

        # numerical error
        hsic_score = np.clip(hsic_score, a_min=0.0, a_max=hsic_score)

        return hsic_score

    def score_v_stat(self, X, Y, normalized: bool = True):

        # calculate kernel matrices
        Kx = self.kernel(X, X)
        Ky = self.kernel(Y, Y)

        # calculate hsic score
        hsic_score = hsic_v_statistic(Kx, Ky)

        # normalized
        if normalized:
            norm = hsic_v_statistic(Kx, Kx) * hsic_v_statistic(Ky, Ky)

            hsic_score /= np.sqrt(norm)

        # numerical error
        hsic_score = np.clip(hsic_score, a_min=0.0, a_max=hsic_score)

        return hsic_score


def hsic_u_statistic(K_x: np.ndarray, K_y: np.ndarray) -> float:
    """Calculate the unbiased statistic

    Parameters
    ----------
    K_x : np.ndarray
        the kernel matrix for samples, X
        (n_samples, n_samples)

    K_y : np.ndarray
        the kernel matrix for samples, Y

    Returns
    -------
    score : float
        the hsic score using the unbiased statistic
    """
    n_samples = K_x.shape[0]

    np.fill_diagonal(K_x, 0.0)
    np.fill_diagonal(K_y, 0.0)

    K_xy = K_x @ K_y

    # Term 1
    a = 1 / n_samples / (n_samples - 3)
    A = np.trace(K_xy)

    # Term 2
    b = a / (n_samples - 1) / (n_samples - 2)
    B = np.sum(K_x) * np.sum(K_y)

    # Term 3
    c = (a * 2) / (n_samples - 2)
    C = np.sum(K_xy)

    # calculate hsic statistic
    return a * A + b * B - c * C


def hsic_v_statistic(K_x: np.ndarray, K_y: np.ndarray) -> float:
    """Calculate the biased statistic

    Parameters
    ----------
    K_x : np.ndarray
        the kernel matrix for samples, X
        (n_samples, n_samples)

    K_y : np.ndarray
        the kernel matrix for samples, Y

    Returns
    -------
    score : float
        the hsic score using the biased statistic
    """
    n_samples = K_x.shape[0]

    # center the kernel matrices
    K_x = KernelCenterer().fit_transform(K_x)
    K_y = KernelCenterer().fit_transform(K_y)

    # calculate hsic statistic
    return float(np.einsum("ij,ij->", K_x, K_y) / n_samples ** 2)


def cka_coefficient_rbf(
    X: np.ndarray, Y: np.ndarray, subsample: Optional[int] = None, seed: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""
    # estimate kernel
    kern = init_rbf_kernel(n_sub_samples=subsample, seed=seed)

    # calculate the kernel matrices
    Kx = kern(X, X)
    Ky = kern(Y, Y)

    # frobenius norm of the cross terms (numerator)
    xy_norm = hsic_v_statistic(Kx, Ky)

    # normalizing coefficients (denomenator)
    x_norm = np.sqrt(hsic_v_statistic(Kx, Kx))
    y_norm = np.sqrt(hsic_v_statistic(Ky, Ky))

    # rv coefficient
    cka_coeff = xy_norm / x_norm / y_norm

    return {
        "cka_coeff": cka_coeff,
        "cka_xy_norm": xy_norm,
        "cka_x_norm": x_norm,
        "cka_y_norm": y_norm,
    }
