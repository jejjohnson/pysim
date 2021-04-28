from typing import Callable, Optional, Dict
import numpy as np
from pysim.kernel.rbf import init_rbf_kernel


class MMD:
    def __init__(
        self, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.kernel = kernel

    def score_u_stat(self, X, Y):

        # calculate kernel matrices
        Kx = self.kernel(X, X)
        Ky = self.kernel(Y, Y)
        Kxy = self.kernel(X, Y)

        # calculate hsic score
        mmd_score = self.u_statistic(Kx, Ky, Kxy)

        # numerical error
        mmd_score = np.clip(mmd_score, a_min=0.0, a_max=mmd_score)

        return mmd_score

    def score_v_stat(self, X, Y):

        # calculate kernel matrices
        Kx = self.kernel(X, X)
        Ky = self.kernel(Y, Y)
        Kxy = self.kernel(X, Y)

        # calculate hsic score
        mmd_score = self.v_statistic(Kx, Ky, Kxy)

        # numerical error
        mmd_score = np.clip(mmd_score, a_min=0.0, a_max=mmd_score)

        return mmd_score


def u_statistic(K_xx: np.ndarray, K_yy: np.ndarray, K_xy: np.ndarray) -> float:
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
    n_samples, m_samples = K_xx.shape[0], K_yy.shape[1]

    # remove diagonal elements
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)

    # Term 1
    a = 1 / (np.power(n_samples, 2) - n_samples)
    A = np.einsum("ij->", K_xx)

    # Term II
    b = 1 / (np.power(m_samples, 2) - m_samples)
    B = np.einsum("ij->", K_yy)

    # Term III
    c = 1 / (n_samples * m_samples)
    C = np.einsum("ij->", K_xy)

    # estimate MMD
    mmd_est = a * A + b * B - 2 * c * C

    return mmd_est


def v_statistic(K_xx: np.ndarray, K_yy: np.ndarray, K_xy: np.ndarray) -> float:
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
    # Term 1
    A = np.mean(K_xx[:])

    # Term II
    B = np.mean(K_yy[:])

    # Term III
    C = np.mean(K_xy[:])

    # estimate MMD
    mmd_est = A + B - 2 * C

    return mmd_est


def mmd_coefficient_rbf(
    X: np.ndarray, Y: np.ndarray, subsample: Optional[int] = None, seed: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""
    # estimate kernel
    kern = init_rbf_kernel(n_sub_samples=subsample, seed=seed)

    # calculate the kernel matrices
    K_xx = kern(X, X)
    K_yy = kern(Y, Y)
    K_xy = kern(X, Y)
    n_samples, m_samples = K_xx.shape[0], K_yy.shape[1]

    # frobenius norm of the cross terms (numerator)
    # remove diagonal elements
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)

    # Term 1
    a = 1 / (np.power(n_samples, 2) - n_samples)
    A = np.einsum("ij->", K_xx)
    x_norm = a * A

    # Term II
    b = 1 / (np.power(m_samples, 2) - m_samples)
    B = np.einsum("ij->", K_yy)
    y_norm = b * B

    # Term III
    c = 1 / (n_samples * m_samples)
    C = np.einsum("ij->", K_xy)
    xy_norm = c * C

    # rv coefficient
    mmd_est = x_norm + y_norm - 2 * xy_norm
    mmd_coeff = xy_norm / np.sqrt(x_norm) / np.sqrt(y_norm)
    return {
        "mmd_coeff": mmd_coeff,
        "mmd_est": _fix_numerical_error(mmd_est),
        "mmd_xy_norm": xy_norm,
        "mmd_x_norm": x_norm,
        "mmd_y_norm": y_norm,
    }


def _fix_numerical_error(score: float):
    return np.clip(score, a_min=0.0, a_max=score)
