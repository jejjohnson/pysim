from typing import Optional

import numpy as np

# TODO: Testing - check size constraint inputs to fail
# TODO: Testing - check size constraint outputs
# TODO: Testing - self covariance v.s. cross covariance
# TODO: Testing - check 1D case
# TODO: Testing - check 2D case
# TODO: Testing - check 3D case fails
# TODO: Testing - check bias
# TODO: Documentation


def covariance(
    X: np.ndarray, Y: Optional[np.ndarray] = None, bias: bool = False
) -> np.ndarray:

    if Y is None:
        Y = X

    # check space arguments
    msg = f"# X features '{X.shape[1]}' is not equal to # X features '{Y.shape[1]}'."
    assert X.shape[1] == Y.shape[1], msg

    # Remove the Mean
    X -= np.mean(X, axis=1)
    Y -= np.mean(Y, axis=1)

    # calculate covariance
    covar = X.T @ Y

    # normalize data
    if bias == True:
        covar *= 1 / (covar.shape[0] - 1)

    else:
        covar *= 1 / covar.shape[0]

    return covar
