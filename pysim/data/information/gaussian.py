import numpy as np

from typing import Callable, Dict, Optional, NamedTuple, List
from pysim.information.gaussian import gaussian_entropy_symmetric
from sklearn.datasets import make_spd_matrix


class GaussianEntropyData(NamedTuple):
    X: np.ndarray
    A: np.ndarray
    H: float
    C: np.ndarray
    H_marg: List[int]
    dataset: str
    estimator: str
    seed: int


def generate_gaussian_data(
    n_samples: int,
    n_features: int,
    seed: int = 123,
    n_base_samples: int = 500_000,
    jitter: float = 1e-8,
    **kwargs,
) -> NamedTuple:
    """Generate multivariate Gaussian data
    Uses the standard formula for entropy:

        H(X) = D/2 + D/2 log 2π + 1/2 log |Σ_xx|

    where:
        * D - number of features
        * |.| - absolute determinant
        * log - the natural log
    
    Parameters
    ----------
    n_samples : int
        the number of samples
    n_features : int
        the number of features
    seed : int, optional
        the random seed, by default 123
    n_base_samples : int, optional
        number of base samples for true distribution.
        The `n_samples` parameter is a subset, by default 500_000
    jitter : float
        the jitter to make the covariance matrix non-singular
        by default 500_000

    Returns
    -------
    NamedTuple
        X - data, (n_samples, n_features)
        H - float
        C - covariance, (n_features, n_features)
        dataset - "gaussian"
    """

    assert n_base_samples > n_samples

    # create seed (trial number)
    rng = np.random.RandomState(seed=int(seed))

    # generate random matrix
    C = make_spd_matrix(
        n_dim=int(n_features), random_state=seed
    )  # rng.rand(int(n_features), int(n_features))
    C += jitter * np.eye(int(n_features))

    # joint covariance matrix
    # C = A @ A.T
    mu = np.zeros((n_features))

    # generate samples
    data_original = rng.multivariate_normal(mu, C, int(n_base_samples))

    # subsample
    data = data_original[:n_samples]

    # compute marginal entropy
    H = gaussian_entropy_symmetric(C)

    return GaussianEntropyData(
        X=data,
        A=None,
        H=H,
        H_marg=None,
        C=C,
        seed=seed,
        dataset="gaussian",
        estimator=None,
    )


def generate_gaussian_rotation_data(
    n_samples,
    n_features,
    seed: int = 123,
    n_base_samples: int = 500_000,
    jitter: float = 1e-8,
    **kwargs,
) -> NamedTuple:
    """Generate rotated multivariate Gaussian
    Uses the formula:

        Y = AX
    
    where A is generated from a uniform distribution,
    and X is generated from a normal distribution N(0,1)

    Uses the standard formula for entropy:
    
        H(Y) = H(X) + log |A|

    where:
        * |.| - absolute determinant
        * log - the natural log
    
    Parameters
    ----------
    n_samples : int
        the number of samples
    n_features : int
        the number of features
    seed : int, optional
        the random seed, by default 123
    n_base_samples : int, optional
        number of base samples for true distribution.
        The `n_samples` parameter is a subset, by default 500_000
    jitter : float
        the jitter to make the covariance matrix non-singular
        by default 500_000

    Returns
    -------
    NamedTuple
        X - data, (n_samples, n_features)
        A - rotation matrix, (n_features, n_features)
        seed - the random seed
        H - float
        C - covariance, (n_features, n_features)
        dataset - "gaussian"
        estimater - None
    """
    assert n_base_samples > n_samples

    # create seed (trial number)
    rng = np.random.RandomState(seed=int(seed))

    # generate random matrix
    C = make_spd_matrix(
        n_dim=int(n_features), random_state=seed
    )  # rng.rand(int(n_features), int(n_features))
    C += jitter * np.eye(int(n_features))

    # joint covariance matrix
    # C = A @ A.T
    mu = np.zeros((n_features))

    # generate samples
    data_original = rng.multivariate_normal(mu, C, int(n_base_samples))

    # compute marginal entropy
    H = gaussian_entropy_symmetric(C)

    # generate random rotation matrix
    rng = np.random.RandomState(seed=int(seed + 100))
    A = rng.rand(int(n_features), int(n_features))

    # rotate matrix
    data_original = data_original @ A

    # estimate total entropy
    H_ori = H + np.linalg.slogdet(A)[1]

    # take a subsample
    data = data_original[:n_samples]

    return GaussianEntropyData(
        X=data,
        A=A,
        C=C,
        H=H_ori,
        H_marg=None,
        seed=seed,
        dataset="gaussian",
        estimator=None,
    )


def generate_gaussian_mi_data(
    n_samples: int,
    n_features: int,
    n_base_samples: int = 5e5,
    seed: int = 123,
    jitter: float = 1e-8,
):

    # joint covariance matrix
    C = make_spd_matrix(n_dim=int(2 * n_features), random_state=seed)
    C += jitter * np.eye(int(2 * n_features))
    mu = np.zeros((2 * n_features))

    # sub covariance matrices
    C_X = C[:n_features, :n_features]
    C_Y = C[n_features:, n_features:]

    # marginal Entropy
    H_X = gaussian_entropy_symmetric(C_X)
    H_Y = gaussian_entropy_symmetric(C_Y)
    H_XY = gaussian_entropy_symmetric(C)

    # mutual information
    mutual_info = H_X + H_Y - H_XY

    # generate random gaussian sample
    # create seed (trial number)
    rng = np.random.RandomState(seed=int(seed + 100))
    data_original = rng.multivariate_normal(mu, C, int(n_base_samples))
    data = data_original[:n_samples]
    X = data[:, :n_features]
    Y = data[:, n_features:]

    return GaussianMIData(
        n_samples=n_samples,
        n_features=n_features,
        seed=seed,
        X=X,
        Y=Y,
        H_X=H_X,
        H_Y=H_Y,
        H_XY=H_XY,
        C=C,
        C_X=C_X,
        C_Y=C_Y,
        MI=mutual_info,
        dataset="gaussian",
    )


class GaussianMIData(NamedTuple):
    n_samples: int
    n_features: int
    seed: int
    X: np.ndarray
    Y: np.ndarray
    C: np.ndarray
    C_X: np.ndarray
    C_Y: np.ndarray
    H_X: float
    H_Y: float
    H_XY: float
    MI: float
    dataset: str
