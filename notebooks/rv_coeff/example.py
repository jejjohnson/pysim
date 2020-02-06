import numpy as np


def main():

    np.random.seed(123)

    n_samples = 1
    d_dimensions = 100

    A = np.random.rand(d_dimensions, d_dimensions)

    X = np.random.randn(n_samples, d_dimensions)
    Y = X @ A

    Sigma_xy = np.cov(X, Y)

    print(f"RV Cov: {np.trace(Sigma_xy @ Sigma_xy.T):.3f}")
    print(f"RV Cov: {np.sum(Sigma_xy * Sigma_xy):.3f}",)
    print(f"RV Cov: {np.linalg.norm(Sigma_xy) ** 2:.3f}",)


if __name__ == "__main__":
    main()
