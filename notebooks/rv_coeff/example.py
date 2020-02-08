import numpy as np


def covariance_self():

    np.random.seed(123)

    n_samples = 1000
    d_dimensions = 100

    A = np.random.rand(d_dimensions, d_dimensions)

    X = np.random.randn(n_samples, d_dimensions)
    Y = X @ A

    Sigma_x = np.cov(X, rowvar=False)
    print(Sigma_x.shape)

    print(f"RV Cov: {np.einsum('ij,ji->', Sigma_x, Sigma_x):.3f}")
    print(f"RV Cov: {np.trace(Sigma_x @ Sigma_x.T):.3f}")
    print(f"RV Cov: {np.sum(Sigma_x * Sigma_x):.3f}")
    print(f"RV Cov: {np.linalg.norm(Sigma_x) ** 2:.3f}")
    print(f"RV Cov: {(np.linalg.eigvalsh(Sigma_x) ** 2).sum():.3f}")

    Sigma_x = np.cov(X, rowvar=True)
    print(Sigma_x.shape)

    print(f"RV Cov: {np.einsum('ij,ji->', Sigma_x, Sigma_x):.3f}")
    print(f"RV Cov: {np.trace(Sigma_x @ Sigma_x.T):.3f}")
    print(f"RV Cov: {np.sum(Sigma_x * Sigma_x):.3f}")
    print(f"RV Cov: {np.linalg.norm(Sigma_x) ** 2:.3f}")
    print(f"RV Cov: {(np.linalg.eigvalsh(Sigma_x) ** 2).sum():.3f}")


def main():

    np.random.seed(123)

    n_samples = 1000
    d_dimensions = 100

    A = np.random.rand(d_dimensions, d_dimensions)

    X = np.random.randn(n_samples, d_dimensions)
    Y = X @ A

    Sigma_xy = np.cov(X, Y, rowvar=False)
    print(Sigma_xy.shape)

    print(f"RV Cov: {np.trace(Sigma_xy @ Sigma_xy.T):.3f}")
    print(f"RV Cov: {np.sum(Sigma_xy * Sigma_xy):.3f}")
    print(f"RV Cov: {np.linalg.norm(Sigma_xy) ** 2:.3f}")
    print(f"RV Cov: {(np.linalg.eigvalsh(Sigma_xy) ** 2).sum():.3f}")


def covariance_example():

    np.random.seed(123)

    n_samples = 1_000
    d_dimensions = 100

    A = np.random.rand(d_dimensions, d_dimensions)

    X = np.random.randn(n_samples, d_dimensions)
    Y = X @ A

    # Sample Covariance
    sigma_xy = np.cov(X, Y, rowvar=True)
    print(sigma_xy.shape)
    print(f"Cov (HS): {np.trace(sigma_xy @ sigma_xy.T):.3f}")

    # Feature Covariance
    sigma_xy = np.cov(X, Y, rowvar=False)
    print(sigma_xy.shape)
    print(f"Cov (HS): {np.trace(sigma_xy @ sigma_xy.T):.3f}")

    return None


def rv_coeff_example():

    np.random.seed(123)

    n_samples = 1_000
    d_dimensions = 100

    A = np.random.rand(d_dimensions, d_dimensions)

    X = np.random.randn(n_samples, d_dimensions)
    Y = X @ A

    # ===============
    # Primal Space
    # ===============

    # HS Norm - cross term
    sigma_xy = np.cov(X, Y, rowvar=True)

    sigma_xy_norm = np.trace(sigma_xy @ sigma_xy.T)

    # HS Norm - individual terms
    sigma_x = np.cov(X, X, rowvar=True)
    sigma_x_norm = np.linalg.norm(sigma_x)

    sigma_y = np.cov(Y, Y, rowvar=True)
    sigma_y_norm = np.linalg.norm(sigma_y)
    print(sigma_xy.shape, sigma_x.shape, sigma_y.shape)

    rv_coeff = sigma_xy_norm / (sigma_x_norm * sigma_y_norm)

    print(f"RV coeff (Samples): {rv_coeff:.3f}")

    # Dual Space
    sigma_xy = np.cov(X, Y, rowvar=False)
    print(sigma_xy.shape)
    print(f"Cov (HS): {np.trace(sigma_xy @ sigma_xy.T):.3f}")

    return None


if __name__ == "__main__":
    covariance_self()
