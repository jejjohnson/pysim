from typing import Callable, Optional, Union

import numpy as np

from hsic import HSIC, RFFHSIC, RandomizedHSIC


def HSIC_demo():

    # fix random seed
    np.random.seed(123)
    n_samples = 1_000
    n_features = 50
    n_components = 100
    A = np.random.rand(n_features, n_features)

    X = np.random.randn(n_samples, n_features)
    Y = X @ A

    # =======================================
    print("\nLinearHSIC Method (centered):")
    # =======================================

    # initialize method
    hsic_clf = HSIC(center=True, kernel="linear")

    # fit method
    hsic_clf.fit(X, Y)

    # calculate scores
    hsic_score = hsic_clf.score(X, Y)
    nhsic_score = hsic_clf.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {hsic_clf.K_x_norm:.6f}")
    print(f"||K_yy||: {hsic_clf.K_y_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRBF HSIC Method (centered):")
    # =======================================

    # initialize method
    hsic_clf = HSIC(center=True, kernel="rbf")

    # fit method
    hsic_clf.fit(X, Y)

    # calculate scores
    hsic_score = hsic_clf.score(X, Y)
    nhsic_score = hsic_clf.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {hsic_clf.K_x_norm:.6f}")
    print(f"||K_yy||: {hsic_clf.K_y_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRFF HSIC Method (centered):")
    # =======================================

    # initialize method
    rffhsic_clf = RFFHSIC(center=True, n_components=n_components)

    # fit method
    rffhsic_clf.fit(X, Y)

    # calculate scores
    hsic_score = rffhsic_clf.score(X, Y)
    nhsic_score = rffhsic_clf.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {rffhsic_clf.Zx_norm:.6f}")
    print(f"||K_yy||: {rffhsic_clf.Zy_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRBF Nystrom HSIC Method (centered):")
    # =======================================

    # initialize method
    rhsic_clf = RandomizedHSIC(center=True, n_components=n_components, kernel="rbf")

    # fit method
    rhsic_clf.fit(X, Y)

    # calculate scores
    hsic_score = rhsic_clf.score(X, Y)
    nhsic_score = rhsic_clf.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {rhsic_clf.Zx_norm:.6f}")
    print(f"||K_yy||: {rhsic_clf.Zy_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nLinear HSIC Method (not centered):")
    # =======================================

    # initialize method
    hsic_clf = HSIC(center=False, kernel="linear")

    # fit method
    hsic_clf.fit(X, Y)

    # calculate scores
    hsic_score = hsic_clf.score(X, Y)
    nhsic_score = hsic_clf.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {hsic_clf.K_x_norm:.6f}")
    print(f"||K_yy||: {hsic_clf.K_y_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRBF HSIC Method (not centered):")
    # =======================================

    # initialize method
    hsic_clf = HSIC(center=False, kernel="rbf")

    # fit method
    hsic_clf.fit(X, Y)

    # calculate scores
    hsic_score = hsic_clf.score(X, Y)
    nhsic_score = hsic_clf.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {hsic_clf.K_x_norm:.6f}")
    print(f"||K_yy||: {hsic_clf.K_y_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRFF HSIC Method (not centered):")
    # =======================================

    # initialize method
    rffhsic_clf = RFFHSIC(center=False, n_components=n_components)

    # fit method
    rffhsic_clf.fit(X, Y)

    # calculate scores
    hsic_score = rffhsic_clf.score(X, Y)
    nhsic_score = rffhsic_clf.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {rffhsic_clf.Zx_norm:.6f}")
    print(f"||K_yy||: {rffhsic_clf.Zy_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRBF Nystrom HSIC Method (uncentered):")
    # =======================================

    # initialize method
    rhsic_clf = RandomizedHSIC(center=False, n_components=n_components, kernel="rbf")

    # fit method
    rhsic_clf.fit(X, Y)

    # calculate scores
    hsic_score = rhsic_clf.score(X, Y)
    nhsic_score = rhsic_clf.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {rhsic_clf.Zx_norm:.6f}")
    print(f"||K_yy||: {rhsic_clf.Zy_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")


if __name__ == "__main__":
    HSIC_demo()
