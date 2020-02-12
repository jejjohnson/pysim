from typing import Callable, Optional, Union

import numpy as np
from hsic import HSIC, RFFHSIC, RHSIC


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
    clf_RFFHSIC = HSIC(center=True, kernel="linear")

    # fit method
    clf_RFFHSIC.fit(X, Y)

    # calculate scores
    hsic_score = clf_RFFHSIC.score(X, Y)
    nhsic_score = clf_RFFHSIC.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {clf_RFFHSIC.K_x_norm:.6f}")
    print(f"||K_yy||: {clf_RFFHSIC.K_y_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRBF HSIC Method (centered):")
    # =======================================

    # initialize method
    clf_RFFHSIC = HSIC(center=True, kernel="rbf")

    # fit method
    clf_RFFHSIC.fit(X, Y)

    # calculate scores
    hsic_score = clf_RFFHSIC.score(X, Y)
    nhsic_score = clf_RFFHSIC.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {clf_RFFHSIC.K_x_norm:.6f}")
    print(f"||K_yy||: {clf_RFFHSIC.K_y_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRFF HSIC Method (centered):")
    # =======================================

    # initialize method
    clf_HSIC = RFFHSIC(center=True, n_components=n_components)

    # fit method
    clf_HSIC.fit(X, Y)

    # calculate scores
    hsic_score = clf_HSIC.score(X, Y)
    nhsic_score = clf_HSIC.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {clf_HSIC.Zx_norm:.6f}")
    print(f"||K_yy||: {clf_HSIC.Zy_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRBF Nystrom HSIC Method (centered):")
    # =======================================

    # initialize method
    clf_RHSIC = RHSIC(center=True, n_components=n_components, kernel="rbf")

    # fit method
    clf_RHSIC.fit(X, Y)

    # calculate scores
    hsic_score = clf_RHSIC.score(X, Y)
    nhsic_score = clf_RHSIC.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {clf_RHSIC.Zx_norm:.6f}")
    print(f"||K_yy||: {clf_RHSIC.Zy_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nLinear HSIC Method (not centered):")
    # =======================================

    # initialize method
    clf_HSIC = HSIC(center=False, kernel="linear")

    # fit method
    clf_HSIC.fit(X, Y)

    # calculate scores
    hsic_score = clf_HSIC.score(X, Y)
    nhsic_score = clf_HSIC.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {clf_HSIC.K_x_norm:.6f}")
    print(f"||K_yy||: {clf_HSIC.K_y_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRBF HSIC Method (not centered):")
    # =======================================

    # initialize method
    clf_HSIC = HSIC(center=False, kernel="rbf")

    # fit method
    clf_HSIC.fit(X, Y)

    # calculate scores
    hsic_score = clf_HSIC.score(X, Y)
    nhsic_score = clf_HSIC.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {clf_HSIC.K_x_norm:.6f}")
    print(f"||K_yy||: {clf_HSIC.K_y_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRFF HSIC Method (not centered):")
    # =======================================

    # initialize method
    clf_HSIC = RFFHSIC(center=False, n_components=n_components)

    # fit method
    clf_HSIC.fit(X, Y)

    # calculate scores
    hsic_score = clf_HSIC.score(X, Y)
    nhsic_score = clf_HSIC.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {clf_HSIC.Zx_norm:.6f}")
    print(f"||K_yy||: {clf_HSIC.Zy_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")

    # =======================================
    print("\nRBF Nystrom HSIC Method (uncentered):")
    # =======================================

    # initialize method
    clf_HSIC = RHSIC(center=False, n_components=n_components, kernel="rbf")

    # fit method
    clf_HSIC.fit(X, Y)

    # calculate scores
    hsic_score = clf_HSIC.score(X, Y)
    nhsic_score = clf_HSIC.score(X, Y, normalize=True)

    print(f"<K_x,K_y>: {hsic_score:.6f}")
    print(f"||K_xx||: {clf_HSIC.Zx_norm:.6f}")
    print(f"||K_yy||: {clf_HSIC.Zy_norm:.6f}")
    print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {nhsic_score:.6f}")


if __name__ == "__main__":
    HSIC_demo()
