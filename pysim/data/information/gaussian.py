import numpy as np

from typing import Callable, Dict, Optional, NamedTuple, List
from pysim.information.gaussian import gaussian_entropy_symmetric
from sklearn.datasets import make_spd_matrix


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


class GaussianEntropyData(NamedTuple):
    X: np.ndarray
    C: np.ndarray
    H: float
    dataset: str
    seed: int


def generate_gaussian_data(
    n_samples, n_features, seed: int = 123, n_base_samples: int = 500_000, **kwargs,
) -> NamedTuple:

    assert n_base_samples > n_samples

    # create seed (trial number)
    rng = np.random.RandomState(seed=int(seed))

    # generate random matrix
    C = make_spd_matrix(
        n_dim=int(n_features), random_state=seed
    )  # rng.rand(int(n_features), int(n_features))

    # joint covariance matrix
    # C = A @ A.T
    mu = np.zeros((n_features))

    # generate samples
    data_original = rng.multivariate_normal(mu, C, int(n_base_samples))

    # subsample
    data = data_original[:n_samples]

    # compute marginal entropy
    H = gaussian_entropy_symmetric(C)

    return GaussianEntropyData(X=data, H=H, C=C, seed=seed, dataset="gaussian")


def generate_gaussian_rotation_data(
    n_samples,
    n_features,
    marg_h_estimator: Callable,
    estimator_name: Optional[str] = None,
    seed: int = 123,
    n_base_samples: int = 500_000,
    **kwargs,
) -> NamedTuple:

    assert n_base_samples > n_samples

    # create seed (trial number)
    rng = np.random.RandomState(seed=int(seed))

    # generate random gaussian sample
    data_original = rng.randn(int(n_base_samples), int(n_features))

    # generate random rotation matrix
    rng = np.random.RandomState(seed=int(seed + 100))
    A = rng.rand(int(n_features), int(n_features))

    # rotate matrix
    data = data_original @ A

    # compute marginal entropy
    H_marg = marg_h_estimator(data, **kwargs)

    # estimate total entropy
    H_ori = np.sum(H_marg) + np.linalg.slogdet(A)[1]

    # convert to nats
    H_ori_nats = H_ori * np.log(2)

    data = data[:n_samples]
    return GaussianRotationH(
        X=data,
        A=A,
        H=H_ori_nats,
        H_marg=H_marg,
        seed=seed,
        dataset="gaussian",
        estimator=estimator_name,
    )
    # return {
    #     "data": data,
    #     "H_nats": H_ori_nats,
    #     "H": H_ori,
    #     "A": A,
    #     "dataset": "gaussian",
    #     "entropy_est": estimator_name,
    # }


class GaussianRotationH(NamedTuple):
    X: np.ndarray
    A: np.ndarray
    H: float
    H_marg: List[int]
    dataset: str
    estimator: str
    seed: int


def generate_gaussian_mi_data(
    n_samples, n_features, n_base_samples: int = 5e5, seed: int = 123,
):

    # joint covariance matrix
    C = make_spd_matrix(n_dim=int(2 * n_features), random_state=seed)
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
        dataset="gaussian",
    )
