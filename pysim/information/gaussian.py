import numpy as np
from scipy import stats


def gauss_entropy_uni(X: np.ndarray) -> None:

    loc = X.mean(axis=0)
    scale = np.cov(X.T)

    # assume it's a Gaussian
    norm_dist = stats.norm(loc=loc, scale=scale)

    return norm_dist.entropy()[0]


def gauss_entropy_multi(X: np.ndarray) -> None:

    n_samples, n_features = X.shape

    # remove mean
    mean = X.mean(axis=0)

    # calculate covariance
    if n_features > 1:
        cov = np.cov(X, bias=1, rowvar=False)
    else:
        cov = np.array([[np.var(X.T, ddof=1)]])

    # assume it's a Gaussian
    cov += 1e-8 * np.eye(n_features)
    norm_dist = stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)

    return norm_dist.entropy()


def gauss_total_corr(X: np.ndarray) -> None:
    n_samples, n_features = X.shape
    # calculate covariance
    if n_features > 1:
        cov = np.cov(X, bias=1, rowvar=False)
    else:
        cov = np.array([[np.var(X.T, ddof=1)]])
    return gaussian_total_corr_symmetric(cov)


def covar_to_corr(C):
    assert np.allclose(C, C.T), "Covariance matrix not symmetric"
    d = 1 / np.sqrt(np.diag(C))
    # same as np.diag(d) @ C @ np.diag(d), but using broadcasting
    return d * (d * C).T


def gaussian_entropy_symmetric(C):

    assert C.shape[0] == C.shape[1]

    n_features = C.shape[0]

    # closed form solution
    H = (
        n_features / 2.0
        + (n_features / 2.0) * np.log(2 * np.pi)
        + 0.5 * np.linalg.slogdet(C)[1]
    )

    return H


def gaussian_total_corr_symmetric(C):

    assert C.shape[0] == C.shape[1]

    # closed form solution
    vv = np.diag(C)
    H_mg = np.sum(np.log(np.sqrt(vv)))
    H_joint = 0.5 * np.linalg.slogdet(C)[1]
    TC = H_mg - H_joint

    return TC


def gaussian_entropy(C):

    n_features = C.shape[0]

    # closed form solution
    H = (
        n_features / 2.0
        + (n_features / 2.0) * np.log(2 * np.pi)
        + 0.5 * np.linalg.slogdet(C.T @ C)[1]
    )

    return H
