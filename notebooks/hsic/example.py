import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler


def main():

    np.random.seed(123)

    n_samples = 1000
    d_dimensions = 100

    A = np.random.rand(d_dimensions, d_dimensions)

    X = np.random.randn(n_samples, d_dimensions)
    Y = X @ A

    # Calculate Kernel Matrices
    K_x = linear_kernel(X)
    K_y = linear_kernel(Y)

    # normalize data
    X_n = StandardScaler(with_std=False).fit_transform(X)
    Y_n = StandardScaler().fit_transform(Y)

    K_x_n = linear_kernel(X_n)
    K_y_n = linear_kernel(Y_n)

    np.testing.assert_array_almost_equal(K_x, K_x_n)

    pass


if __name__ == "__main__":
    main()
