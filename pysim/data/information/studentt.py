import numpy as np

from typing import Callable, Dict, Optional, NamedTuple, List
from pysim.information.studentt import studentt_entropy_symmetric
from sklearn.datasets import make_spd_matrix
from scipy.stats import multivariate_t


class EntropyData(NamedTuple):
    X: np.ndarray
    C: np.ndarray
    H: float
    dataset: str
    seed: int


def generate_studentt_data(
    n_samples,
    n_features,
    df: float = 3.0,
    seed: int = 123,
    n_base_samples: int = 500_000,
    **kwargs,
) -> NamedTuple:

    assert n_base_samples > n_samples
    assert df == 1.0 or df >= 2.0

    # generate random matrix
    C = make_spd_matrix(n_dim=int(n_features), random_state=seed)
    C += 1e-8 * np.eye(int(n_features))

    # joint covariance matrix
    mu = np.zeros((n_features))

    # generate samples
    data_original = multivariate_t(loc=mu, shape=C, df=df, seed=seed + 100).rvs(
        size=(n_samples)
    )

    # subsample
    data = data_original[:n_samples]

    if data.ndim < 2:
        data = data[:, None]

    # compute marginal entropy
    if df < 2.0:
        H = studentt_entropy_symmetric(C)
        dataset = "cauchy"
    else:
        H = studentt_entropy_symmetric(df, C)
        dataset = "studentt"

    return EntropyData(X=data, H=H, C=C, seed=seed, dataset=dataset)


def generate_studentt_mi_data(
    n_samples, n_features, df: float = 3.0, n_base_samples: int = 5e5, seed: int = 123,
):  # joint covariance matrix
    C = make_spd_matrix(n_dim=2 * int(n_features), random_state=seed)
    mu = np.zeros((2 * n_features))

    # sub covariance matrices
    C_X = C[:n_features, :n_features]
    C_Y = C[n_features:, n_features:]

    # marginal Entropy
    # compute marginal entropy
    if df < 2.0:
        H_X = studentt_entropy_symmetric(C_X)
        H_Y = studentt_entropy_symmetric(C_Y)
        H_XY = studentt_entropy_symmetric(C)
        dataset = "cauchy"
    else:
        H_X = studentt_entropy_symmetric(df, C_X)
        H_Y = studentt_entropy_symmetric(df, C_Y)
        H_XY = studentt_entropy_symmetric(df, C)
        dataset = "studentt"

    # mutual information
    mutual_info = H_X + H_Y - H_XY

    # generate random gaussian sample
    # create seed (trial number)
    # generate samples
    data_original = multivariate_t(loc=mu, shape=C, df=df, seed=seed + 100).rvs(
        size=(n_samples)
    )

    data = data_original[:n_samples]
    X = data[:, :n_features]
    Y = data[:, n_features:]

    return MIData(
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
        dataset=dataset,
    )


class MIData(NamedTuple):
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

