import numpy as np
from typing import Callable, Dict, Optional, NamedTuple, List


def generate_linear_entropy_data(
    n_samples: int,
    n_features: int,
    marg_h_estimator: Callable,
    estimator_name: Optional[str] = None,
    seed: int = 123,
    n_base_samples: int = 500_000,
    **kwargs,
) -> Dict:
    """[summary]

    Parameters
    ----------
    n_samples : int
        the number of samples to generate
    n_features : int
        number of features to for the generated data
    marg_h_estimator : Callable
        marginal entropy estimator
    estimator_name : Optional[str], optional
        [description], by default None
    seed : int, optional
        [description], by default 123
    n_base_samples : int, optional
        [description], by default 500_000

    Returns
    -------
    Dict
        [description]
    """
    # create seed (trial number)
    rng = np.random.RandomState(seed=int(seed))

    # generate random Gaussian data
    data_original = rng.randn(int(n_base_samples), int(n_features))

    # generate marginal uniform data
    # create seed (trial number)
    rng = np.random.RandomState(seed=int(seed + 100))
    random_data = rng.rand(int(n_features))

    for idim in range(int(n_features)):
        exponent = 2 * random_data[idim] + 0.5
        data_original[:, idim] = (
            np.sign(data_original[:, idim]) * np.abs(data_original[:, idim]) ** exponent
        )

    # generate random rotation matrix
    # create seed (trial number)
    rng = np.random.RandomState(seed=int(seed + 200))
    A = rng.rand(int(n_features), int(n_features))

    # rotate matrix
    data = data_original @ A

    # compute marginal entropy
    H_X = marg_h_estimator(data, **kwargs)
    H_Y = marg_h_estimator(data_original, **kwargs)

    # estimate total entropy
    H_ori = np.sum(H_Y) + np.linalg.slogdet(A)[1]

    TC = np.sum(H_X) - np.sum(H_Y) + np.linalg.slogdet(A)[1]

    # convert to nats
    H_ori_nats = H_ori

    data = data[:n_samples]
    return RotationEntropyData(
        X=data,
        A=A,
        data=data_original,
        H=H_ori_nats,
        H_X=H_X,
        H_Y=H_Y,
        TC=TC,
        seed=seed,
        dataset="linear_rotated",
        estimator=estimator_name,
    )


class RotationEntropyData(NamedTuple):
    X: np.ndarray
    A: np.ndarray
    data: np.ndarray
    H: float
    H_X: float
    H_Y: float
    TC: float
    dataset: str
    estimator: str
    seed: int
