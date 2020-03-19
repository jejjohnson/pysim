import sys

sys.path.insert(0, "/Users/eman/Documents/code_projects/pysim")

import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use(["seaborn-paper"])

from pysim.information.entropy import univariate_entropy


seed = 123
np.random.seed(seed)

n_samples = 1_000
a = 5

# initialize data distribution
data_dist = stats.gamma(a=a)

# get some samples
X_samples = data_dist.rvs(size=n_samples)[:, None]

# sns.distplot(X_samples)
# plt.show()

# ===========================
# True Entropy
# ===========================
H_true = data_dist.entropy()

print(f"Entropy (True): {H_true:.4f}")

# ===========================
# Histogram Entropy
# ===========================
method = "histogram"
correction = True

X_hist_entropy = univariate_entropy(X_samples, method=method, correction=correction)

print(f"Entropy (Histogram): {X_hist_entropy:.4f}")

# ===========================
# KNN Entropy
# ===========================
method = "knn"
n_neighbors = 5
n_jobs = 1
algorithm = "brute"

X_knn_entropy = univariate_entropy(
    X_samples,
    method=method,
    n_neighbors=n_neighbors,
    n_jobs=n_jobs,
    algorithm=algorithm,
)

print(f"Entropy (KNN): {X_knn_entropy:.4f}")

# ===========================
# KDE Entropy
# ===========================
method = "kde"
kernel = "gau"
bw = "normal_reference"

X_kde_entropy = univariate_entropy(X_samples, method=method, kernel=kernel, bw=bw)

print(f"Entropy (KDE): {X_kde_entropy:.4f}")

# ===========================
# Gaussian Assumption
# ===========================
method = "gaussian"

X_gaus_entropy = univariate_entropy(X_samples, method=method)

print(f"Entropy (Gauss): {X_gaus_entropy:.4f}")
