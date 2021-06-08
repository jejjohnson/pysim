from scipy.special import gamma as Γ, psi as ψ, beta as β
import numpy as np
from numpy import log, pi as π
from plum import dispatch
from typing import Union


def studentt_entropy_symmetric(
    df: Union[float, int, np.ndarray], cov: np.ndarray
) -> float:
    """Entropy for Multivariate T-Student
    In this case, we are only considering the degrees
    of freedom (df) to be greater than 2.
    
    Parameters
    ----------
    df : float
        the degrees of freedom for the t-student
        (must be greater than 2)
    
    cov : np.ndarray
        the covariance matrix for the t-student distribution.
        (n_features, n_features)
        
    Returns
    -------
    H : float
        the differential entropy (nats)
    
    Resources
    ---------
    [1] :   A measure of total variability for the 
            multivariate t distribution with applications to 
            finance - Guerrero-Cusumano (1996)
            (equation 8)
    """

    assert df > 2
    ν = df
    Σ = cov

    assert Σ.shape[0] == Σ.shape[1]
    D = cov.shape[0]

    # Term I - change of variables
    term_a = 0.5 * np.linalg.slogdet(Σ)[1]

    # Term II
    term_b1 = ((ν - 2) * π) ** (0.5 * D)
    term_b2 = Γ(0.5 * D)
    term_b3 = β(0.5 * D, 0.5 * ν)
    term_b = log((term_b1 / term_b2) * term_b3)

    # Term III
    term_c = ψ(0.5 * (ν + D)) - ψ(0.5 * ν)
    term_c *= (ν + D) / 2.0

    # final term
    final_term = term_a + term_b + term_c

    return final_term


def cauchy_entropy_symmetric(cov: np.ndarray) -> float:
    """Entropy for Multivariate Cauchy
    This is the special case of the T-Student Distribution
    when the degrees of freedom are equal to 1.
    
    Parameters
    ----------
    
    cov : np.ndarray
        the covariance matrix for the t-student distribution.
        (n_features, n_features)
        
    Returns
    -------
    H : float
        the differential entropy (nats)
    
    Resources
    ---------
    [1] :   A measure of total variability for the 
            multivariate t distribution with applications to 
            finance - Guerrero-Cusumano (1996)
            (equation 9)
    """

    Σ = cov

    assert Σ.shape[0] == Σ.shape[1]
    D = cov.shape[0]

    # Term I - change of variables
    term_a = 0.5 * np.linalg.slogdet(Σ)[1]

    # Term II
    term_b1 = π ** (0.5 * D)
    term_b2 = Γ(0.5 * D)
    term_b3 = β(0.5 * D, 0.5)
    term_b = log((term_b1 / term_b2) * term_b3)

    # Term III
    term_c = ψ(0.5 * (1 + D)) - ψ(0.5)
    term_c *= (1 + D) / 2.0

    # final term
    final_term = term_a + term_b + term_c

    return final_term


def studentt_total_corr(df, cov) -> None:
    H = studentt_entropy_symmetric(df, cov)
    h_marg = 0
    for n_dim in range(cov.shape[0]):

        h_marg += studentt_entropy_symmetric(df, cov[n_dim, n_dim].reshape(-1, 1))
    return h_marg - H


def cauchy_total_corr(cov) -> None:

    H = cauchy_entropy_symmetric(cov)
    h_marg = 0
    for n_dim in range(cov.shape[0]):

        h_marg += cauchy_entropy_symmetric(cov[n_dim, n_dim].reshape(-1, 1))
    return h_marg - H
