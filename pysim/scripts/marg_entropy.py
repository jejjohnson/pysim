import sys

sys.path.insert(0, "/Users/eman/Documents/code_projects/pysim")

import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use(["seaborn-paper"])

from pysim.information.entropy import marginal_entropy


seed = 123
np.random.seed(seed)

n_samples = 1_000
a = 5
b = 10

# initialize data distribution
data_dist1 = stats.gamma(a=a)
data_dist2 = stats.beta(a=a, b=b)

# get some samples
X1_samples = data_dist1.rvs(size=n_samples)[:, None]
X2_samples = data_dist2.rvs(size=n_samples)[:, None]
X_samples = np.hstack([X1_samples, X2_samples])

assert X_samples.shape[1] == 2

sns.jointplot(X_samples[:, 0], X_samples[:, 1])
plt.show()

# ===========================
# True Entropy
# ===========================
H1_true = data_dist1.entropy()
H2_true = data_dist2.entropy()

print(f"Entropy (True): {H1_true:.4f}, {H2_true:.4f}")


# ===========================
# Histogram Entropy
# ===========================
method = "histogram"
correction = True

# entropy_clf = Univariate(method)

X_hist_entropy = marginal_entropy(X_samples, method=method, correction=correction)

print(f"Entropy (Histogram): {X_hist_entropy}")

# ===========================
# KNN Entropy
# ===========================
method = "knn"
n_neighbors = 5
n_jobs = 1
algorithm = "brute"

# entropy_clf = Univariate(method)

X_knn_entropy = marginal_entropy(
    X_samples,
    method=method,
    n_neighbors=n_neighbors,
    n_jobs=n_jobs,
    algorithm=algorithm,
)

print(f"Entropy (KNN): {X_knn_entropy}")


# ===========================
# KDE Entropy
# ===========================
method = "kde"
kernel = "gau"
bw = "normal_reference"

# entropy_clf = Univariate(method)

X_kde_entropy = marginal_entropy(X_samples, method=method, kernel=kernel, bw=bw)

print(f"Entropy (KDE): {X_kde_entropy}")

# ===========================
# Gaussian Assumption
# ===========================
method = "gaussian"

# entropy_clf = Univariate(method)

X_gaus_entropy = marginal_entropy(X_samples, method=method)

print(f"Entropy (Gauss): {X_gaus_entropy}")
